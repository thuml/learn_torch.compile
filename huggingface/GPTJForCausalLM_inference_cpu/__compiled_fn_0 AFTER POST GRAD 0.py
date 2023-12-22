from __future__ import annotations



def forward(self, arg0_1: "f32[50400, 4096]", arg1_1: "f32[4096]", arg2_1: "f32[4096]", arg3_1: "f32[4096, 4096]", arg4_1: "f32[4096, 4096]", arg5_1: "f32[4096, 4096]", arg6_1: "f32[4096, 4096]", arg7_1: "f32[16384, 4096]", arg8_1: "f32[16384]", arg9_1: "f32[4096, 16384]", arg10_1: "f32[4096]", arg11_1: "f32[4096]", arg12_1: "f32[4096]", arg13_1: "f32[4096, 4096]", arg14_1: "f32[4096, 4096]", arg15_1: "f32[4096, 4096]", arg16_1: "f32[4096, 4096]", arg17_1: "f32[16384, 4096]", arg18_1: "f32[16384]", arg19_1: "f32[4096, 16384]", arg20_1: "f32[4096]", arg21_1: "f32[4096]", arg22_1: "f32[4096]", arg23_1: "f32[4096, 4096]", arg24_1: "f32[4096, 4096]", arg25_1: "f32[4096, 4096]", arg26_1: "f32[4096, 4096]", arg27_1: "f32[16384, 4096]", arg28_1: "f32[16384]", arg29_1: "f32[4096, 16384]", arg30_1: "f32[4096]", arg31_1: "f32[4096]", arg32_1: "f32[4096]", arg33_1: "f32[4096, 4096]", arg34_1: "f32[4096, 4096]", arg35_1: "f32[4096, 4096]", arg36_1: "f32[4096, 4096]", arg37_1: "f32[16384, 4096]", arg38_1: "f32[16384]", arg39_1: "f32[4096, 16384]", arg40_1: "f32[4096]", arg41_1: "f32[4096]", arg42_1: "f32[4096]", arg43_1: "f32[4096, 4096]", arg44_1: "f32[4096, 4096]", arg45_1: "f32[4096, 4096]", arg46_1: "f32[4096, 4096]", arg47_1: "f32[16384, 4096]", arg48_1: "f32[16384]", arg49_1: "f32[4096, 16384]", arg50_1: "f32[4096]", arg51_1: "f32[4096]", arg52_1: "f32[4096]", arg53_1: "f32[4096, 4096]", arg54_1: "f32[4096, 4096]", arg55_1: "f32[4096, 4096]", arg56_1: "f32[4096, 4096]", arg57_1: "f32[16384, 4096]", arg58_1: "f32[16384]", arg59_1: "f32[4096, 16384]", arg60_1: "f32[4096]", arg61_1: "f32[4096]", arg62_1: "f32[4096]", arg63_1: "f32[4096, 4096]", arg64_1: "f32[4096, 4096]", arg65_1: "f32[4096, 4096]", arg66_1: "f32[4096, 4096]", arg67_1: "f32[16384, 4096]", arg68_1: "f32[16384]", arg69_1: "f32[4096, 16384]", arg70_1: "f32[4096]", arg71_1: "f32[4096]", arg72_1: "f32[4096]", arg73_1: "f32[4096, 4096]", arg74_1: "f32[4096, 4096]", arg75_1: "f32[4096, 4096]", arg76_1: "f32[4096, 4096]", arg77_1: "f32[16384, 4096]", arg78_1: "f32[16384]", arg79_1: "f32[4096, 16384]", arg80_1: "f32[4096]", arg81_1: "f32[4096]", arg82_1: "f32[4096]", arg83_1: "f32[4096, 4096]", arg84_1: "f32[4096, 4096]", arg85_1: "f32[4096, 4096]", arg86_1: "f32[4096, 4096]", arg87_1: "f32[16384, 4096]", arg88_1: "f32[16384]", arg89_1: "f32[4096, 16384]", arg90_1: "f32[4096]", arg91_1: "f32[4096]", arg92_1: "f32[4096]", arg93_1: "f32[4096, 4096]", arg94_1: "f32[4096, 4096]", arg95_1: "f32[4096, 4096]", arg96_1: "f32[4096, 4096]", arg97_1: "f32[16384, 4096]", arg98_1: "f32[16384]", arg99_1: "f32[4096, 16384]", arg100_1: "f32[4096]", arg101_1: "f32[4096]", arg102_1: "f32[4096]", arg103_1: "f32[4096, 4096]", arg104_1: "f32[4096, 4096]", arg105_1: "f32[4096, 4096]", arg106_1: "f32[4096, 4096]", arg107_1: "f32[16384, 4096]", arg108_1: "f32[16384]", arg109_1: "f32[4096, 16384]", arg110_1: "f32[4096]", arg111_1: "f32[4096]", arg112_1: "f32[4096]", arg113_1: "f32[4096, 4096]", arg114_1: "f32[4096, 4096]", arg115_1: "f32[4096, 4096]", arg116_1: "f32[4096, 4096]", arg117_1: "f32[16384, 4096]", arg118_1: "f32[16384]", arg119_1: "f32[4096, 16384]", arg120_1: "f32[4096]", arg121_1: "f32[4096]", arg122_1: "f32[4096]", arg123_1: "f32[4096, 4096]", arg124_1: "f32[4096, 4096]", arg125_1: "f32[4096, 4096]", arg126_1: "f32[4096, 4096]", arg127_1: "f32[16384, 4096]", arg128_1: "f32[16384]", arg129_1: "f32[4096, 16384]", arg130_1: "f32[4096]", arg131_1: "f32[4096]", arg132_1: "f32[4096]", arg133_1: "f32[4096, 4096]", arg134_1: "f32[4096, 4096]", arg135_1: "f32[4096, 4096]", arg136_1: "f32[4096, 4096]", arg137_1: "f32[16384, 4096]", arg138_1: "f32[16384]", arg139_1: "f32[4096, 16384]", arg140_1: "f32[4096]", arg141_1: "f32[4096]", arg142_1: "f32[4096]", arg143_1: "f32[4096, 4096]", arg144_1: "f32[4096, 4096]", arg145_1: "f32[4096, 4096]", arg146_1: "f32[4096, 4096]", arg147_1: "f32[16384, 4096]", arg148_1: "f32[16384]", arg149_1: "f32[4096, 16384]", arg150_1: "f32[4096]", arg151_1: "f32[4096]", arg152_1: "f32[4096]", arg153_1: "f32[4096, 4096]", arg154_1: "f32[4096, 4096]", arg155_1: "f32[4096, 4096]", arg156_1: "f32[4096, 4096]", arg157_1: "f32[16384, 4096]", arg158_1: "f32[16384]", arg159_1: "f32[4096, 16384]", arg160_1: "f32[4096]", arg161_1: "f32[4096]", arg162_1: "f32[4096]", arg163_1: "f32[4096, 4096]", arg164_1: "f32[4096, 4096]", arg165_1: "f32[4096, 4096]", arg166_1: "f32[4096, 4096]", arg167_1: "f32[16384, 4096]", arg168_1: "f32[16384]", arg169_1: "f32[4096, 16384]", arg170_1: "f32[4096]", arg171_1: "f32[4096]", arg172_1: "f32[4096]", arg173_1: "f32[4096, 4096]", arg174_1: "f32[4096, 4096]", arg175_1: "f32[4096, 4096]", arg176_1: "f32[4096, 4096]", arg177_1: "f32[16384, 4096]", arg178_1: "f32[16384]", arg179_1: "f32[4096, 16384]", arg180_1: "f32[4096]", arg181_1: "f32[4096]", arg182_1: "f32[4096]", arg183_1: "f32[4096, 4096]", arg184_1: "f32[4096, 4096]", arg185_1: "f32[4096, 4096]", arg186_1: "f32[4096, 4096]", arg187_1: "f32[16384, 4096]", arg188_1: "f32[16384]", arg189_1: "f32[4096, 16384]", arg190_1: "f32[4096]", arg191_1: "f32[4096]", arg192_1: "f32[4096]", arg193_1: "f32[4096, 4096]", arg194_1: "f32[4096, 4096]", arg195_1: "f32[4096, 4096]", arg196_1: "f32[4096, 4096]", arg197_1: "f32[16384, 4096]", arg198_1: "f32[16384]", arg199_1: "f32[4096, 16384]", arg200_1: "f32[4096]", arg201_1: "f32[4096]", arg202_1: "f32[4096]", arg203_1: "f32[4096, 4096]", arg204_1: "f32[4096, 4096]", arg205_1: "f32[4096, 4096]", arg206_1: "f32[4096, 4096]", arg207_1: "f32[16384, 4096]", arg208_1: "f32[16384]", arg209_1: "f32[4096, 16384]", arg210_1: "f32[4096]", arg211_1: "f32[4096]", arg212_1: "f32[4096]", arg213_1: "f32[4096, 4096]", arg214_1: "f32[4096, 4096]", arg215_1: "f32[4096, 4096]", arg216_1: "f32[4096, 4096]", arg217_1: "f32[16384, 4096]", arg218_1: "f32[16384]", arg219_1: "f32[4096, 16384]", arg220_1: "f32[4096]", arg221_1: "f32[4096]", arg222_1: "f32[4096]", arg223_1: "f32[4096, 4096]", arg224_1: "f32[4096, 4096]", arg225_1: "f32[4096, 4096]", arg226_1: "f32[4096, 4096]", arg227_1: "f32[16384, 4096]", arg228_1: "f32[16384]", arg229_1: "f32[4096, 16384]", arg230_1: "f32[4096]", arg231_1: "f32[4096]", arg232_1: "f32[4096]", arg233_1: "f32[4096, 4096]", arg234_1: "f32[4096, 4096]", arg235_1: "f32[4096, 4096]", arg236_1: "f32[4096, 4096]", arg237_1: "f32[16384, 4096]", arg238_1: "f32[16384]", arg239_1: "f32[4096, 16384]", arg240_1: "f32[4096]", arg241_1: "f32[4096]", arg242_1: "f32[4096]", arg243_1: "f32[4096, 4096]", arg244_1: "f32[4096, 4096]", arg245_1: "f32[4096, 4096]", arg246_1: "f32[4096, 4096]", arg247_1: "f32[16384, 4096]", arg248_1: "f32[16384]", arg249_1: "f32[4096, 16384]", arg250_1: "f32[4096]", arg251_1: "f32[4096]", arg252_1: "f32[4096]", arg253_1: "f32[4096, 4096]", arg254_1: "f32[4096, 4096]", arg255_1: "f32[4096, 4096]", arg256_1: "f32[4096, 4096]", arg257_1: "f32[16384, 4096]", arg258_1: "f32[16384]", arg259_1: "f32[4096, 16384]", arg260_1: "f32[4096]", arg261_1: "f32[4096]", arg262_1: "f32[4096]", arg263_1: "f32[4096, 4096]", arg264_1: "f32[4096, 4096]", arg265_1: "f32[4096, 4096]", arg266_1: "f32[4096, 4096]", arg267_1: "f32[16384, 4096]", arg268_1: "f32[16384]", arg269_1: "f32[4096, 16384]", arg270_1: "f32[4096]", arg271_1: "f32[4096]", arg272_1: "f32[4096]", arg273_1: "f32[4096, 4096]", arg274_1: "f32[4096, 4096]", arg275_1: "f32[4096, 4096]", arg276_1: "f32[4096, 4096]", arg277_1: "f32[16384, 4096]", arg278_1: "f32[16384]", arg279_1: "f32[4096, 16384]", arg280_1: "f32[4096]", arg281_1: "f32[4096]", arg282_1: "f32[4096]", arg283_1: "f32[50400, 4096]", arg284_1: "f32[50400]", arg285_1: "f32[2048, 64]", arg286_1: "b8[1, 1, 2048, 2048]", arg287_1: "f32[]", arg288_1: "f32[2048, 64]", arg289_1: "b8[1, 1, 2048, 2048]", arg290_1: "f32[]", arg291_1: "f32[2048, 64]", arg292_1: "b8[1, 1, 2048, 2048]", arg293_1: "f32[]", arg294_1: "f32[2048, 64]", arg295_1: "b8[1, 1, 2048, 2048]", arg296_1: "f32[]", arg297_1: "f32[2048, 64]", arg298_1: "b8[1, 1, 2048, 2048]", arg299_1: "f32[]", arg300_1: "f32[2048, 64]", arg301_1: "b8[1, 1, 2048, 2048]", arg302_1: "f32[]", arg303_1: "f32[2048, 64]", arg304_1: "b8[1, 1, 2048, 2048]", arg305_1: "f32[]", arg306_1: "f32[2048, 64]", arg307_1: "b8[1, 1, 2048, 2048]", arg308_1: "f32[]", arg309_1: "f32[2048, 64]", arg310_1: "b8[1, 1, 2048, 2048]", arg311_1: "f32[]", arg312_1: "f32[2048, 64]", arg313_1: "b8[1, 1, 2048, 2048]", arg314_1: "f32[]", arg315_1: "f32[2048, 64]", arg316_1: "b8[1, 1, 2048, 2048]", arg317_1: "f32[]", arg318_1: "f32[2048, 64]", arg319_1: "b8[1, 1, 2048, 2048]", arg320_1: "f32[]", arg321_1: "f32[2048, 64]", arg322_1: "b8[1, 1, 2048, 2048]", arg323_1: "f32[]", arg324_1: "f32[2048, 64]", arg325_1: "b8[1, 1, 2048, 2048]", arg326_1: "f32[]", arg327_1: "f32[2048, 64]", arg328_1: "b8[1, 1, 2048, 2048]", arg329_1: "f32[]", arg330_1: "f32[2048, 64]", arg331_1: "b8[1, 1, 2048, 2048]", arg332_1: "f32[]", arg333_1: "f32[2048, 64]", arg334_1: "b8[1, 1, 2048, 2048]", arg335_1: "f32[]", arg336_1: "f32[2048, 64]", arg337_1: "b8[1, 1, 2048, 2048]", arg338_1: "f32[]", arg339_1: "f32[2048, 64]", arg340_1: "b8[1, 1, 2048, 2048]", arg341_1: "f32[]", arg342_1: "f32[2048, 64]", arg343_1: "b8[1, 1, 2048, 2048]", arg344_1: "f32[]", arg345_1: "f32[2048, 64]", arg346_1: "b8[1, 1, 2048, 2048]", arg347_1: "f32[]", arg348_1: "f32[2048, 64]", arg349_1: "b8[1, 1, 2048, 2048]", arg350_1: "f32[]", arg351_1: "f32[2048, 64]", arg352_1: "b8[1, 1, 2048, 2048]", arg353_1: "f32[]", arg354_1: "f32[2048, 64]", arg355_1: "b8[1, 1, 2048, 2048]", arg356_1: "f32[]", arg357_1: "f32[2048, 64]", arg358_1: "b8[1, 1, 2048, 2048]", arg359_1: "f32[]", arg360_1: "f32[2048, 64]", arg361_1: "b8[1, 1, 2048, 2048]", arg362_1: "f32[]", arg363_1: "f32[2048, 64]", arg364_1: "b8[1, 1, 2048, 2048]", arg365_1: "f32[]", arg366_1: "f32[2048, 64]", arg367_1: "b8[1, 1, 2048, 2048]", arg368_1: "f32[]", arg369_1: "i64[1, 128]", arg370_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:582, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.reshape.default(arg369_1, [-1, 128]);  arg369_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:635, code: inputs_embeds = self.wte(input_ids)
    embedding: "f32[1, 128, 4096]" = torch.ops.aten.embedding.default(arg0_1, view);  arg0_1 = view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(embedding, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg285_1, [1, 1, 1]);  arg285_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:605, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:606, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    view_1: "i64[1, 128]" = torch.ops.aten.reshape.default(unsqueeze, [-1, 128]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_1: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_1: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_1, [1, 1, 64]);  unsqueeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat, 1, repeat_1);  repeat = repeat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes = torch.ops.aten.split_with_sizes.default(gather, [32, 32], 2);  gather = None
    getitem_2: "f32[1, 128, 32]" = split_with_sizes[0]
    getitem_3: "f32[1, 128, 32]" = split_with_sizes[1];  split_with_sizes = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant0: "f32[]" = self._tensor_constant0
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_47: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg286_1, 2, 0, 128);  arg286_1 = None
    slice_48: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_47, 3, 0, 128);  slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(embedding, getitem_1);  getitem_1 = None
    add: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    mul: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
    add_1: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_2: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_1, [128, 4096])
    permute: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
    mm: "f32[128, 4096]" = torch.ops.aten.mm.default(view_2, permute);  view_2 = permute = None
    view_3: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm, [1, 128, 4096]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_8: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_3, [1, 128, 16, 256]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_12: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_8, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_10: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_3, 2)
    unsqueeze_11: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 4);  unsqueeze_10 = None
    expand_3: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_11, [1, 128, 1, 32, 2]);  unsqueeze_11 = None
    clone_4: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_15: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_4, [1, 128, 1, 64]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_4: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_12, view_15);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_44: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_12, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_1: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_44);  slice_44 = None
    unsqueeze_12: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_1, 4);  neg_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_40: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_12, 3, 0, 9223372036854775807, 2);  slice_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_13: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_40, 4);  slice_40 = None
    cat_1: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_12, unsqueeze_13], 4);  unsqueeze_12 = unsqueeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_16: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_1, [1, 128, 16, 64]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_8: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_2, 2)
    unsqueeze_9: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, 4);  unsqueeze_8 = None
    expand_2: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_9, [1, 128, 1, 32, 2]);  unsqueeze_9 = None
    clone_3: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_14: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_3, [1, 128, 1, 64]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_5: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_16, view_14);  view_16 = view_14 = None
    add_3: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_16: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_8, 3, 64, 9223372036854775807);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_3: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_3, slice_16], 3);  add_3 = slice_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_5: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_3, [0, 2, 1, 3]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_4: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_5, [1, 16, 128, 256]);  permute_5 = None
    view_17: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_4, [16, 128, 256]);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_4: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_1, [128, 4096])
    permute_1: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    mm_1: "f32[128, 4096]" = torch.ops.aten.mm.default(view_4, permute_1);  view_4 = permute_1 = None
    view_5: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_1, [1, 128, 4096]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_9: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_5, [1, 128, 16, 256]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_4: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_9, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_4: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_3, 2);  getitem_3 = None
    unsqueeze_5: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 4);  unsqueeze_4 = None
    expand_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_5, [1, 128, 1, 32, 2]);  unsqueeze_5 = None
    clone_2: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_12: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_2, [1, 128, 1, 64]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_2: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_4, view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_30: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_4, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_30);  slice_30 = None
    unsqueeze_6: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg, 4);  neg = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_26: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_4, 3, 0, 9223372036854775807, 2);  slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_7: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_26, 4);  slice_26 = None
    cat: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_6, unsqueeze_7], 4);  unsqueeze_6 = unsqueeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_13: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat, [1, 128, 16, 64]);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_2: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_2, 2);  getitem_2 = None
    unsqueeze_3: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 4);  unsqueeze_2 = None
    expand: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_3, [1, 128, 1, 32, 2]);  unsqueeze_3 = None
    clone_1: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_11: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_1, [1, 128, 1, 64]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_3: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_13, view_11);  view_13 = view_11 = None
    add_2: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_8: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_9, 3, 64, 9223372036854775807);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_2: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_2, slice_8], 3);  add_2 = slice_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_4: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_2, [0, 2, 1, 3]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_6: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2])
    expand_5: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_6, [1, 16, 256, 128]);  permute_6 = None
    view_18: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_5, [16, 256, 128]);  expand_5 = None
    bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_17, view_18);  view_17 = view_18 = None
    view_19: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, [1, 16, 128, 128]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, view_19, full_default);  slice_48 = view_19 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where, arg287_1);  where = arg287_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div, [-1], True)
    sub_1: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div, amax);  div = amax = None
    exp: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_1: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_6: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_1, [1, 16, 128, 128]);  div_1 = None
    view_20: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_6, [16, 128, 128]);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_6: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_1, [128, 4096])
    permute_2: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    mm_2: "f32[128, 4096]" = torch.ops.aten.mm.default(view_6, permute_2);  view_6 = permute_2 = None
    view_7: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_2, [1, 128, 4096]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_10: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_7, [1, 128, 16, 256]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_3: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_7: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_3, [1, 16, 128, 256])
    view_21: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_7, [16, 128, 256]);  expand_7 = None
    bmm_1: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_20, view_21);  view_20 = view_21 = None
    view_22: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_1, [1, 16, 128, 256]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_22, [0, 2, 1, 3]);  view_22 = None
    clone_6: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_23: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_6, [1, 128, 4096]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_24: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_23, [128, 4096]);  view_23 = None
    permute_8: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    mm_3: "f32[128, 4096]" = torch.ops.aten.mm.default(view_24, permute_8);  view_24 = permute_8 = None
    view_25: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_3, [1, 128, 4096]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_26: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_1, [128, 4096]);  add_1 = None
    permute_9: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    addmm: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg8_1, view_26, permute_9);  arg8_1 = view_26 = permute_9 = None
    view_27: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm, [1, 128, 16384]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_6: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_27, 0.5)
    pow_1: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_27, 3.0)
    mul_7: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_4: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_27, mul_7);  view_27 = mul_7 = None
    mul_8: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_4, 0.7978845608028654);  add_4 = None
    tanh: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_8);  mul_8 = None
    add_5: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_9: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_6, add_5);  mul_6 = add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_28: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_9, [128, 16384]);  mul_9 = None
    permute_10: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    addmm_1: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg10_1, view_28, permute_10);  arg10_1 = view_28 = permute_10 = None
    view_29: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_1, [1, 128, 4096]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_6: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_25, view_29);  view_25 = view_29 = None
    add_7: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_6, embedding);  add_6 = embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_2: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg288_1, [1, 1, 1]);  arg288_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_14: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_3: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_14, [1, 1, 64]);  unsqueeze_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_1: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_2, 1, repeat_3);  repeat_2 = repeat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(gather_1, [32, 32], 2);  gather_1 = None
    getitem_6: "f32[1, 128, 32]" = split_with_sizes_1[0]
    getitem_7: "f32[1, 128, 32]" = split_with_sizes_1[1];  split_with_sizes_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant1: "f32[]" = self._tensor_constant1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_95: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg289_1, 2, 0, 128);  arg289_1 = None
    slice_96: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_95, 3, 0, 128);  slice_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_2: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_7, getitem_5);  getitem_5 = None
    add_8: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    mul_10: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_11: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_10, arg11_1);  mul_10 = arg11_1 = None
    add_9: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_11, arg12_1);  mul_11 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_30: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_9, [128, 4096])
    permute_11: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
    mm_4: "f32[128, 4096]" = torch.ops.aten.mm.default(view_30, permute_11);  view_30 = permute_11 = None
    view_31: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_4, [1, 128, 4096]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_36: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_31, [1, 128, 16, 256]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_60: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_36, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_23: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_7, 2)
    unsqueeze_24: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_23, 4);  unsqueeze_23 = None
    expand_11: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_24, [1, 128, 1, 32, 2]);  unsqueeze_24 = None
    clone_12: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_43: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_12, [1, 128, 1, 64]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_14: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_60, view_43);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_92: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_60, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_3: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_92);  slice_92 = None
    unsqueeze_25: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_3, 4);  neg_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_88: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_60, 3, 0, 9223372036854775807, 2);  slice_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_26: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_88, 4);  slice_88 = None
    cat_5: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_25, unsqueeze_26], 4);  unsqueeze_25 = unsqueeze_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_44: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_5, [1, 128, 16, 64]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_21: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_6, 2)
    unsqueeze_22: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_21, 4);  unsqueeze_21 = None
    expand_10: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_22, [1, 128, 1, 32, 2]);  unsqueeze_22 = None
    clone_11: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_42: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_11, [1, 128, 1, 64]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_15: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_44, view_42);  view_44 = view_42 = None
    add_11: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_64: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_36, 3, 64, 9223372036854775807);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_7: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_11, slice_64], 3);  add_11 = slice_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_16: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_7, [0, 2, 1, 3]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_12: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_16, [1, 16, 128, 256]);  permute_16 = None
    view_45: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_12, [16, 128, 256]);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_32: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_9, [128, 4096])
    permute_12: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    mm_5: "f32[128, 4096]" = torch.ops.aten.mm.default(view_32, permute_12);  view_32 = permute_12 = None
    view_33: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_5, [1, 128, 4096]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_37: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_33, [1, 128, 16, 256]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_52: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_37, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_17: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_7, 2);  getitem_7 = None
    unsqueeze_18: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 4);  unsqueeze_17 = None
    expand_9: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_18, [1, 128, 1, 32, 2]);  unsqueeze_18 = None
    clone_10: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_40: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_10, [1, 128, 1, 64]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_12: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_52, view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_78: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_52, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_2: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_78);  slice_78 = None
    unsqueeze_19: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_2, 4);  neg_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_74: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_52, 3, 0, 9223372036854775807, 2);  slice_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_20: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_74, 4);  slice_74 = None
    cat_4: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_19, unsqueeze_20], 4);  unsqueeze_19 = unsqueeze_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_41: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_4, [1, 128, 16, 64]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_15: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_6, 2);  getitem_6 = None
    unsqueeze_16: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 4);  unsqueeze_15 = None
    expand_8: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_16, [1, 128, 1, 32, 2]);  unsqueeze_16 = None
    clone_9: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_39: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_9, [1, 128, 1, 64]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_13: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_41, view_39);  view_41 = view_39 = None
    add_10: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_12, mul_13);  mul_12 = mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_56: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_37, 3, 64, 9223372036854775807);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_6: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_10, slice_56], 3);  add_10 = slice_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_15: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_6, [0, 2, 1, 3]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_17: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2])
    expand_13: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_17, [1, 16, 256, 128]);  permute_17 = None
    view_46: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_13, [16, 256, 128]);  expand_13 = None
    bmm_2: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_45, view_46);  view_45 = view_46 = None
    view_47: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [1, 16, 128, 128]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_1: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_96, view_47, full_default_1);  slice_96 = view_47 = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_2: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_1, arg290_1);  where_1 = arg290_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_2, [-1], True)
    sub_3: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_2, amax_1);  div_2 = amax_1 = None
    exp_1: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_14: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_3, [1, 16, 128, 128]);  div_3 = None
    view_48: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_14, [16, 128, 128]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_34: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_9, [128, 4096])
    permute_13: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
    mm_6: "f32[128, 4096]" = torch.ops.aten.mm.default(view_34, permute_13);  view_34 = permute_13 = None
    view_35: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_6, [1, 128, 4096]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_38: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_35, [1, 128, 16, 256]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_14: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_15: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_14, [1, 16, 128, 256])
    view_49: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_15, [16, 128, 256]);  expand_15 = None
    bmm_3: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_48, view_49);  view_48 = view_49 = None
    view_50: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_3, [1, 16, 128, 256]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
    clone_14: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_51: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_14, [1, 128, 4096]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_52: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_51, [128, 4096]);  view_51 = None
    permute_19: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    mm_7: "f32[128, 4096]" = torch.ops.aten.mm.default(view_52, permute_19);  view_52 = permute_19 = None
    view_53: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_7, [1, 128, 4096]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_54: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_9, [128, 4096]);  add_9 = None
    permute_20: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    addmm_2: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg18_1, view_54, permute_20);  arg18_1 = view_54 = permute_20 = None
    view_55: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 16384]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_16: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_55, 0.5)
    pow_2: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_55, 3.0)
    mul_17: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_12: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_55, mul_17);  view_55 = mul_17 = None
    mul_18: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_12, 0.7978845608028654);  add_12 = None
    tanh_1: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
    add_13: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_19: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_16, add_13);  mul_16 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_56: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_19, [128, 16384]);  mul_19 = None
    permute_21: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
    addmm_3: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg20_1, view_56, permute_21);  arg20_1 = view_56 = permute_21 = None
    view_57: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_3, [1, 128, 4096]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_14: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_53, view_57);  view_53 = view_57 = None
    add_15: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_14, add_7);  add_14 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_4: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg291_1, [1, 1, 1]);  arg291_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_27: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_5: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_27, [1, 1, 64]);  unsqueeze_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_2: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_4, 1, repeat_5);  repeat_4 = repeat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(gather_2, [32, 32], 2);  gather_2 = None
    getitem_10: "f32[1, 128, 32]" = split_with_sizes_2[0]
    getitem_11: "f32[1, 128, 32]" = split_with_sizes_2[1];  split_with_sizes_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant2: "f32[]" = self._tensor_constant2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_143: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg292_1, 2, 0, 128);  arg292_1 = None
    slice_144: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_143, 3, 0, 128);  slice_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_4: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_15, getitem_9);  getitem_9 = None
    add_16: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    mul_20: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = rsqrt_2 = None
    mul_21: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_20, arg21_1);  mul_20 = arg21_1 = None
    add_17: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_21, arg22_1);  mul_21 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_58: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_17, [128, 4096])
    permute_22: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    mm_8: "f32[128, 4096]" = torch.ops.aten.mm.default(view_58, permute_22);  view_58 = permute_22 = None
    view_59: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_8, [1, 128, 4096]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_64: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_59, [1, 128, 16, 256]);  view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_108: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_64, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_36: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_11, 2)
    unsqueeze_37: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, 4);  unsqueeze_36 = None
    expand_19: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_37, [1, 128, 1, 32, 2]);  unsqueeze_37 = None
    clone_20: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_71: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_20, [1, 128, 1, 64]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_24: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_108, view_71);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_140: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_108, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_5: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_140);  slice_140 = None
    unsqueeze_38: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_5, 4);  neg_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_136: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_108, 3, 0, 9223372036854775807, 2);  slice_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_39: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_136, 4);  slice_136 = None
    cat_9: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_38, unsqueeze_39], 4);  unsqueeze_38 = unsqueeze_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_72: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_9, [1, 128, 16, 64]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_34: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_10, 2)
    unsqueeze_35: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 4);  unsqueeze_34 = None
    expand_18: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_35, [1, 128, 1, 32, 2]);  unsqueeze_35 = None
    clone_19: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_70: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_19, [1, 128, 1, 64]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_25: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_72, view_70);  view_72 = view_70 = None
    add_19: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_24, mul_25);  mul_24 = mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_112: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_64, 3, 64, 9223372036854775807);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_11: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_19, slice_112], 3);  add_19 = slice_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_27: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_11, [0, 2, 1, 3]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_20: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_27, [1, 16, 128, 256]);  permute_27 = None
    view_73: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_20, [16, 128, 256]);  expand_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_60: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_17, [128, 4096])
    permute_23: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    mm_9: "f32[128, 4096]" = torch.ops.aten.mm.default(view_60, permute_23);  view_60 = permute_23 = None
    view_61: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_9, [1, 128, 4096]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_65: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_61, [1, 128, 16, 256]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_100: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_65, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_30: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_11, 2);  getitem_11 = None
    unsqueeze_31: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 4);  unsqueeze_30 = None
    expand_17: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_31, [1, 128, 1, 32, 2]);  unsqueeze_31 = None
    clone_18: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_68: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_18, [1, 128, 1, 64]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_22: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_100, view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_126: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_100, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_4: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_126);  slice_126 = None
    unsqueeze_32: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_4, 4);  neg_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_122: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_100, 3, 0, 9223372036854775807, 2);  slice_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_33: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_122, 4);  slice_122 = None
    cat_8: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_32, unsqueeze_33], 4);  unsqueeze_32 = unsqueeze_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_69: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_8, [1, 128, 16, 64]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_28: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_10, 2);  getitem_10 = None
    unsqueeze_29: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 4);  unsqueeze_28 = None
    expand_16: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_29, [1, 128, 1, 32, 2]);  unsqueeze_29 = None
    clone_17: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_67: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_17, [1, 128, 1, 64]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_23: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_69, view_67);  view_69 = view_67 = None
    add_18: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_22, mul_23);  mul_22 = mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_104: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_65, 3, 64, 9223372036854775807);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_10: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_18, slice_104], 3);  add_18 = slice_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_26: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_10, [0, 2, 1, 3]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_28: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2])
    expand_21: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_28, [1, 16, 256, 128]);  permute_28 = None
    view_74: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_21, [16, 256, 128]);  expand_21 = None
    bmm_4: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_73, view_74);  view_73 = view_74 = None
    view_75: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [1, 16, 128, 128]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_2: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_144, view_75, full_default_2);  slice_144 = view_75 = full_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_4: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_2, arg293_1);  where_2 = arg293_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_4, [-1], True)
    sub_5: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_4, amax_2);  div_4 = amax_2 = None
    exp_2: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_3: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_5: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_22: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_5, [1, 16, 128, 128]);  div_5 = None
    view_76: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_22, [16, 128, 128]);  expand_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_62: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_17, [128, 4096])
    permute_24: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    mm_10: "f32[128, 4096]" = torch.ops.aten.mm.default(view_62, permute_24);  view_62 = permute_24 = None
    view_63: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_10, [1, 128, 4096]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_66: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_63, [1, 128, 16, 256]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_25: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_66, [0, 2, 1, 3]);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_23: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_25, [1, 16, 128, 256])
    view_77: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_23, [16, 128, 256]);  expand_23 = None
    bmm_5: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_76, view_77);  view_76 = view_77 = None
    view_78: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_5, [1, 16, 128, 256]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    clone_22: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_79: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_22, [1, 128, 4096]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_80: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_79, [128, 4096]);  view_79 = None
    permute_30: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    mm_11: "f32[128, 4096]" = torch.ops.aten.mm.default(view_80, permute_30);  view_80 = permute_30 = None
    view_81: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_11, [1, 128, 4096]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_82: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_17, [128, 4096]);  add_17 = None
    permute_31: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    addmm_4: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg28_1, view_82, permute_31);  arg28_1 = view_82 = permute_31 = None
    view_83: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 16384]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_26: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_83, 0.5)
    pow_3: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_83, 3.0)
    mul_27: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_20: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_83, mul_27);  view_83 = mul_27 = None
    mul_28: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_20, 0.7978845608028654);  add_20 = None
    tanh_2: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_28);  mul_28 = None
    add_21: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_29: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_26, add_21);  mul_26 = add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_84: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_29, [128, 16384]);  mul_29 = None
    permute_32: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
    addmm_5: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg30_1, view_84, permute_32);  arg30_1 = view_84 = permute_32 = None
    view_85: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_5, [1, 128, 4096]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_22: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_81, view_85);  view_81 = view_85 = None
    add_23: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_22, add_15);  add_22 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_6: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg294_1, [1, 1, 1]);  arg294_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_40: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_7: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_40, [1, 1, 64]);  unsqueeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_3: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_6, 1, repeat_7);  repeat_6 = repeat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_3 = torch.ops.aten.split_with_sizes.default(gather_3, [32, 32], 2);  gather_3 = None
    getitem_14: "f32[1, 128, 32]" = split_with_sizes_3[0]
    getitem_15: "f32[1, 128, 32]" = split_with_sizes_3[1];  split_with_sizes_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant3: "f32[]" = self._tensor_constant3
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_191: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg295_1, 2, 0, 128);  arg295_1 = None
    slice_192: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_191, 3, 0, 128);  slice_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_6: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_23, getitem_13);  getitem_13 = None
    add_24: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    mul_30: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_3);  sub_6 = rsqrt_3 = None
    mul_31: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_30, arg31_1);  mul_30 = arg31_1 = None
    add_25: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_31, arg32_1);  mul_31 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_86: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_25, [128, 4096])
    permute_33: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    mm_12: "f32[128, 4096]" = torch.ops.aten.mm.default(view_86, permute_33);  view_86 = permute_33 = None
    view_87: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_12, [1, 128, 4096]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_92: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_87, [1, 128, 16, 256]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_156: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_92, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_49: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_15, 2)
    unsqueeze_50: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_49, 4);  unsqueeze_49 = None
    expand_27: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_50, [1, 128, 1, 32, 2]);  unsqueeze_50 = None
    clone_28: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_99: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_28, [1, 128, 1, 64]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_34: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_156, view_99);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_188: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_156, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_7: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_188);  slice_188 = None
    unsqueeze_51: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_7, 4);  neg_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_184: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_156, 3, 0, 9223372036854775807, 2);  slice_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_52: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_184, 4);  slice_184 = None
    cat_13: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_51, unsqueeze_52], 4);  unsqueeze_51 = unsqueeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_100: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_13, [1, 128, 16, 64]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_47: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_14, 2)
    unsqueeze_48: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_47, 4);  unsqueeze_47 = None
    expand_26: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_48, [1, 128, 1, 32, 2]);  unsqueeze_48 = None
    clone_27: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_98: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_27, [1, 128, 1, 64]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_35: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_100, view_98);  view_100 = view_98 = None
    add_27: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_34, mul_35);  mul_34 = mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_160: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_92, 3, 64, 9223372036854775807);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_15: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_27, slice_160], 3);  add_27 = slice_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_38: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_15, [0, 2, 1, 3]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_28: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_38, [1, 16, 128, 256]);  permute_38 = None
    view_101: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_28, [16, 128, 256]);  expand_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_88: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_25, [128, 4096])
    permute_34: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
    mm_13: "f32[128, 4096]" = torch.ops.aten.mm.default(view_88, permute_34);  view_88 = permute_34 = None
    view_89: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_13, [1, 128, 4096]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_93: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_89, [1, 128, 16, 256]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_148: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_93, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_43: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_15, 2);  getitem_15 = None
    unsqueeze_44: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_43, 4);  unsqueeze_43 = None
    expand_25: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_44, [1, 128, 1, 32, 2]);  unsqueeze_44 = None
    clone_26: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_96: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_26, [1, 128, 1, 64]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_32: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_148, view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_174: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_148, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_6: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_174);  slice_174 = None
    unsqueeze_45: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_6, 4);  neg_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_170: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_148, 3, 0, 9223372036854775807, 2);  slice_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_46: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_170, 4);  slice_170 = None
    cat_12: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_45, unsqueeze_46], 4);  unsqueeze_45 = unsqueeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_97: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_12, [1, 128, 16, 64]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_41: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_14, 2);  getitem_14 = None
    unsqueeze_42: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_41, 4);  unsqueeze_41 = None
    expand_24: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_42, [1, 128, 1, 32, 2]);  unsqueeze_42 = None
    clone_25: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_95: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_25, [1, 128, 1, 64]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_33: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_97, view_95);  view_97 = view_95 = None
    add_26: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_32, mul_33);  mul_32 = mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_152: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_93, 3, 64, 9223372036854775807);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_14: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_26, slice_152], 3);  add_26 = slice_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_37: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_14, [0, 2, 1, 3]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_39: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2])
    expand_29: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_39, [1, 16, 256, 128]);  permute_39 = None
    view_102: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_29, [16, 256, 128]);  expand_29 = None
    bmm_6: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_101, view_102);  view_101 = view_102 = None
    view_103: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [1, 16, 128, 128]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_3: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_192, view_103, full_default_3);  slice_192 = view_103 = full_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_6: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_3, arg296_1);  where_3 = arg296_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_6, [-1], True)
    sub_7: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_6, amax_3);  div_6 = amax_3 = None
    exp_3: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_4: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_7: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_30: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_7, [1, 16, 128, 128]);  div_7 = None
    view_104: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_30, [16, 128, 128]);  expand_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_90: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_25, [128, 4096])
    permute_35: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    mm_14: "f32[128, 4096]" = torch.ops.aten.mm.default(view_90, permute_35);  view_90 = permute_35 = None
    view_91: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_14, [1, 128, 4096]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_94: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_91, [1, 128, 16, 256]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_36: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_94, [0, 2, 1, 3]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_31: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_36, [1, 16, 128, 256])
    view_105: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_31, [16, 128, 256]);  expand_31 = None
    bmm_7: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_104, view_105);  view_104 = view_105 = None
    view_106: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_7, [1, 16, 128, 256]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_106, [0, 2, 1, 3]);  view_106 = None
    clone_30: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_107: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_30, [1, 128, 4096]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_108: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_107, [128, 4096]);  view_107 = None
    permute_41: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    mm_15: "f32[128, 4096]" = torch.ops.aten.mm.default(view_108, permute_41);  view_108 = permute_41 = None
    view_109: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_15, [1, 128, 4096]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_110: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_25, [128, 4096]);  add_25 = None
    permute_42: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    addmm_6: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg38_1, view_110, permute_42);  arg38_1 = view_110 = permute_42 = None
    view_111: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 16384]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_111, 0.5)
    pow_4: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_111, 3.0)
    mul_37: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_28: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_111, mul_37);  view_111 = mul_37 = None
    mul_38: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_28, 0.7978845608028654);  add_28 = None
    tanh_3: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    add_29: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_39: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_36, add_29);  mul_36 = add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_112: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_39, [128, 16384]);  mul_39 = None
    permute_43: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    addmm_7: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg40_1, view_112, permute_43);  arg40_1 = view_112 = permute_43 = None
    view_113: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 4096]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_30: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_109, view_113);  view_109 = view_113 = None
    add_31: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_30, add_23);  add_30 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_8: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg297_1, [1, 1, 1]);  arg297_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_53: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_9: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_53, [1, 1, 64]);  unsqueeze_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_4: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_8, 1, repeat_9);  repeat_8 = repeat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_4 = torch.ops.aten.split_with_sizes.default(gather_4, [32, 32], 2);  gather_4 = None
    getitem_18: "f32[1, 128, 32]" = split_with_sizes_4[0]
    getitem_19: "f32[1, 128, 32]" = split_with_sizes_4[1];  split_with_sizes_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant4: "f32[]" = self._tensor_constant4
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_239: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg298_1, 2, 0, 128);  arg298_1 = None
    slice_240: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_239, 3, 0, 128);  slice_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_8: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_31, getitem_17);  getitem_17 = None
    add_32: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    mul_40: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_4);  sub_8 = rsqrt_4 = None
    mul_41: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_40, arg41_1);  mul_40 = arg41_1 = None
    add_33: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_41, arg42_1);  mul_41 = arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_114: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_33, [128, 4096])
    permute_44: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    mm_16: "f32[128, 4096]" = torch.ops.aten.mm.default(view_114, permute_44);  view_114 = permute_44 = None
    view_115: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_16, [1, 128, 4096]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_120: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_115, [1, 128, 16, 256]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_204: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_120, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_62: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_19, 2)
    unsqueeze_63: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, 4);  unsqueeze_62 = None
    expand_35: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_63, [1, 128, 1, 32, 2]);  unsqueeze_63 = None
    clone_36: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_127: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_36, [1, 128, 1, 64]);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_44: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_204, view_127);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_236: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_204, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_9: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_236);  slice_236 = None
    unsqueeze_64: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_9, 4);  neg_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_232: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_204, 3, 0, 9223372036854775807, 2);  slice_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_65: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_232, 4);  slice_232 = None
    cat_17: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_64, unsqueeze_65], 4);  unsqueeze_64 = unsqueeze_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_128: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_17, [1, 128, 16, 64]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_60: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_18, 2)
    unsqueeze_61: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, 4);  unsqueeze_60 = None
    expand_34: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_61, [1, 128, 1, 32, 2]);  unsqueeze_61 = None
    clone_35: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_126: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_35, [1, 128, 1, 64]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_45: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_128, view_126);  view_128 = view_126 = None
    add_35: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_44, mul_45);  mul_44 = mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_208: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_120, 3, 64, 9223372036854775807);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_19: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_35, slice_208], 3);  add_35 = slice_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_49: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_19, [0, 2, 1, 3]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_36: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_49, [1, 16, 128, 256]);  permute_49 = None
    view_129: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_36, [16, 128, 256]);  expand_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_116: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_33, [128, 4096])
    permute_45: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    mm_17: "f32[128, 4096]" = torch.ops.aten.mm.default(view_116, permute_45);  view_116 = permute_45 = None
    view_117: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_17, [1, 128, 4096]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_121: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_117, [1, 128, 16, 256]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_196: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_121, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_56: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_19, 2);  getitem_19 = None
    unsqueeze_57: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, 4);  unsqueeze_56 = None
    expand_33: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_57, [1, 128, 1, 32, 2]);  unsqueeze_57 = None
    clone_34: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_124: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_34, [1, 128, 1, 64]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_42: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_196, view_124);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_222: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_196, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_8: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_222);  slice_222 = None
    unsqueeze_58: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_8, 4);  neg_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_218: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_196, 3, 0, 9223372036854775807, 2);  slice_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_59: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_218, 4);  slice_218 = None
    cat_16: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_58, unsqueeze_59], 4);  unsqueeze_58 = unsqueeze_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_125: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_16, [1, 128, 16, 64]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_54: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_18, 2);  getitem_18 = None
    unsqueeze_55: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, 4);  unsqueeze_54 = None
    expand_32: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_55, [1, 128, 1, 32, 2]);  unsqueeze_55 = None
    clone_33: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_123: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_33, [1, 128, 1, 64]);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_43: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_125, view_123);  view_125 = view_123 = None
    add_34: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_42, mul_43);  mul_42 = mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_200: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_121, 3, 64, 9223372036854775807);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_18: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_34, slice_200], 3);  add_34 = slice_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_48: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_18, [0, 2, 1, 3]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_50: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2])
    expand_37: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_50, [1, 16, 256, 128]);  permute_50 = None
    view_130: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_37, [16, 256, 128]);  expand_37 = None
    bmm_8: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_129, view_130);  view_129 = view_130 = None
    view_131: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [1, 16, 128, 128]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_4: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_4: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_240, view_131, full_default_4);  slice_240 = view_131 = full_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_8: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_4, arg299_1);  where_4 = arg299_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_8, [-1], True)
    sub_9: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_8, amax_4);  div_8 = amax_4 = None
    exp_4: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_5: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_9: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_38: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_9, [1, 16, 128, 128]);  div_9 = None
    view_132: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_38, [16, 128, 128]);  expand_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_118: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_33, [128, 4096])
    permute_46: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
    mm_18: "f32[128, 4096]" = torch.ops.aten.mm.default(view_118, permute_46);  view_118 = permute_46 = None
    view_119: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_18, [1, 128, 4096]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_122: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_119, [1, 128, 16, 256]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_47: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_39: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_47, [1, 16, 128, 256])
    view_133: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_39, [16, 128, 256]);  expand_39 = None
    bmm_9: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_132, view_133);  view_132 = view_133 = None
    view_134: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_9, [1, 16, 128, 256]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
    clone_38: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_135: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_38, [1, 128, 4096]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_136: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_135, [128, 4096]);  view_135 = None
    permute_52: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    mm_19: "f32[128, 4096]" = torch.ops.aten.mm.default(view_136, permute_52);  view_136 = permute_52 = None
    view_137: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_19, [1, 128, 4096]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_138: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_33, [128, 4096]);  add_33 = None
    permute_53: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    addmm_8: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg48_1, view_138, permute_53);  arg48_1 = view_138 = permute_53 = None
    view_139: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 16384]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_46: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_139, 0.5)
    pow_5: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_139, 3.0)
    mul_47: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_36: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_139, mul_47);  view_139 = mul_47 = None
    mul_48: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_36, 0.7978845608028654);  add_36 = None
    tanh_4: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
    add_37: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_49: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_46, add_37);  mul_46 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_140: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_49, [128, 16384]);  mul_49 = None
    permute_54: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    addmm_9: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg50_1, view_140, permute_54);  arg50_1 = view_140 = permute_54 = None
    view_141: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_9, [1, 128, 4096]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_38: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_137, view_141);  view_137 = view_141 = None
    add_39: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_38, add_31);  add_38 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_10: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg300_1, [1, 1, 1]);  arg300_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_66: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_11: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_66, [1, 1, 64]);  unsqueeze_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_5: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_10, 1, repeat_11);  repeat_10 = repeat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_5 = torch.ops.aten.split_with_sizes.default(gather_5, [32, 32], 2);  gather_5 = None
    getitem_22: "f32[1, 128, 32]" = split_with_sizes_5[0]
    getitem_23: "f32[1, 128, 32]" = split_with_sizes_5[1];  split_with_sizes_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant5: "f32[]" = self._tensor_constant5
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_287: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg301_1, 2, 0, 128);  arg301_1 = None
    slice_288: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_287, 3, 0, 128);  slice_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_10: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_39, getitem_21);  getitem_21 = None
    add_40: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    mul_50: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_5);  sub_10 = rsqrt_5 = None
    mul_51: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_50, arg51_1);  mul_50 = arg51_1 = None
    add_41: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_51, arg52_1);  mul_51 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_142: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_41, [128, 4096])
    permute_55: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    mm_20: "f32[128, 4096]" = torch.ops.aten.mm.default(view_142, permute_55);  view_142 = permute_55 = None
    view_143: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_20, [1, 128, 4096]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_148: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_143, [1, 128, 16, 256]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_252: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_148, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_75: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_23, 2)
    unsqueeze_76: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_75, 4);  unsqueeze_75 = None
    expand_43: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_76, [1, 128, 1, 32, 2]);  unsqueeze_76 = None
    clone_44: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_155: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_44, [1, 128, 1, 64]);  clone_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_54: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_252, view_155);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_284: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_252, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_11: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_284);  slice_284 = None
    unsqueeze_77: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_11, 4);  neg_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_280: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_252, 3, 0, 9223372036854775807, 2);  slice_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_78: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_280, 4);  slice_280 = None
    cat_21: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_77, unsqueeze_78], 4);  unsqueeze_77 = unsqueeze_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_156: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_21, [1, 128, 16, 64]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_73: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_22, 2)
    unsqueeze_74: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_73, 4);  unsqueeze_73 = None
    expand_42: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_74, [1, 128, 1, 32, 2]);  unsqueeze_74 = None
    clone_43: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_154: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_43, [1, 128, 1, 64]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_55: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_156, view_154);  view_156 = view_154 = None
    add_43: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_54, mul_55);  mul_54 = mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_256: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_148, 3, 64, 9223372036854775807);  view_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_23: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_43, slice_256], 3);  add_43 = slice_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_60: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_23, [0, 2, 1, 3]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_44: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_60, [1, 16, 128, 256]);  permute_60 = None
    view_157: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_44, [16, 128, 256]);  expand_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_144: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_41, [128, 4096])
    permute_56: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    mm_21: "f32[128, 4096]" = torch.ops.aten.mm.default(view_144, permute_56);  view_144 = permute_56 = None
    view_145: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_21, [1, 128, 4096]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_149: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_145, [1, 128, 16, 256]);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_244: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_149, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_69: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_23, 2);  getitem_23 = None
    unsqueeze_70: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_69, 4);  unsqueeze_69 = None
    expand_41: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_70, [1, 128, 1, 32, 2]);  unsqueeze_70 = None
    clone_42: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_152: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_42, [1, 128, 1, 64]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_52: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_244, view_152);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_270: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_244, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_10: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_270);  slice_270 = None
    unsqueeze_71: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_10, 4);  neg_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_266: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_244, 3, 0, 9223372036854775807, 2);  slice_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_72: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_266, 4);  slice_266 = None
    cat_20: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_71, unsqueeze_72], 4);  unsqueeze_71 = unsqueeze_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_153: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_20, [1, 128, 16, 64]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_67: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_22, 2);  getitem_22 = None
    unsqueeze_68: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 4);  unsqueeze_67 = None
    expand_40: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_68, [1, 128, 1, 32, 2]);  unsqueeze_68 = None
    clone_41: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_151: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_41, [1, 128, 1, 64]);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_53: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_153, view_151);  view_153 = view_151 = None
    add_42: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_52, mul_53);  mul_52 = mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_248: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_149, 3, 64, 9223372036854775807);  view_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_22: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_42, slice_248], 3);  add_42 = slice_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_59: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_22, [0, 2, 1, 3]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_61: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2])
    expand_45: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_61, [1, 16, 256, 128]);  permute_61 = None
    view_158: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_45, [16, 256, 128]);  expand_45 = None
    bmm_10: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_157, view_158);  view_157 = view_158 = None
    view_159: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [1, 16, 128, 128]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_5: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_5: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_288, view_159, full_default_5);  slice_288 = view_159 = full_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_10: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_5, arg302_1);  where_5 = arg302_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_10, [-1], True)
    sub_11: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_10, amax_5);  div_10 = amax_5 = None
    exp_5: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
    sum_6: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_11: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_46: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_11, [1, 16, 128, 128]);  div_11 = None
    view_160: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_46, [16, 128, 128]);  expand_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_146: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_41, [128, 4096])
    permute_57: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    mm_22: "f32[128, 4096]" = torch.ops.aten.mm.default(view_146, permute_57);  view_146 = permute_57 = None
    view_147: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_22, [1, 128, 4096]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_150: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_147, [1, 128, 16, 256]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_58: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_150, [0, 2, 1, 3]);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_47: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_58, [1, 16, 128, 256])
    view_161: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_47, [16, 128, 256]);  expand_47 = None
    bmm_11: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_160, view_161);  view_160 = view_161 = None
    view_162: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_11, [1, 16, 128, 256]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    clone_46: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_163: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_46, [1, 128, 4096]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_164: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_163, [128, 4096]);  view_163 = None
    permute_63: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    mm_23: "f32[128, 4096]" = torch.ops.aten.mm.default(view_164, permute_63);  view_164 = permute_63 = None
    view_165: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_23, [1, 128, 4096]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_166: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_41, [128, 4096]);  add_41 = None
    permute_64: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    addmm_10: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg58_1, view_166, permute_64);  arg58_1 = view_166 = permute_64 = None
    view_167: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 16384]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_56: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_167, 0.5)
    pow_6: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_167, 3.0)
    mul_57: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_44: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_167, mul_57);  view_167 = mul_57 = None
    mul_58: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_44, 0.7978845608028654);  add_44 = None
    tanh_5: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
    add_45: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_59: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_56, add_45);  mul_56 = add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_168: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_59, [128, 16384]);  mul_59 = None
    permute_65: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    addmm_11: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg60_1, view_168, permute_65);  arg60_1 = view_168 = permute_65 = None
    view_169: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_11, [1, 128, 4096]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_46: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_165, view_169);  view_165 = view_169 = None
    add_47: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_46, add_39);  add_46 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_12: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg303_1, [1, 1, 1]);  arg303_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_79: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_13: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_79, [1, 1, 64]);  unsqueeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_6: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_12, 1, repeat_13);  repeat_12 = repeat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_6 = torch.ops.aten.split_with_sizes.default(gather_6, [32, 32], 2);  gather_6 = None
    getitem_26: "f32[1, 128, 32]" = split_with_sizes_6[0]
    getitem_27: "f32[1, 128, 32]" = split_with_sizes_6[1];  split_with_sizes_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant6: "f32[]" = self._tensor_constant6
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_335: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg304_1, 2, 0, 128);  arg304_1 = None
    slice_336: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_335, 3, 0, 128);  slice_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_12: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_47, getitem_25);  getitem_25 = None
    add_48: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    mul_60: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_6);  sub_12 = rsqrt_6 = None
    mul_61: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_60, arg61_1);  mul_60 = arg61_1 = None
    add_49: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_61, arg62_1);  mul_61 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_170: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_49, [128, 4096])
    permute_66: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    mm_24: "f32[128, 4096]" = torch.ops.aten.mm.default(view_170, permute_66);  view_170 = permute_66 = None
    view_171: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_24, [1, 128, 4096]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_176: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_171, [1, 128, 16, 256]);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_300: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_176, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_88: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_27, 2)
    unsqueeze_89: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 4);  unsqueeze_88 = None
    expand_51: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_89, [1, 128, 1, 32, 2]);  unsqueeze_89 = None
    clone_52: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
    view_183: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_52, [1, 128, 1, 64]);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_64: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_300, view_183);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_332: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_300, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_13: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_332);  slice_332 = None
    unsqueeze_90: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_13, 4);  neg_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_328: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_300, 3, 0, 9223372036854775807, 2);  slice_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_91: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_328, 4);  slice_328 = None
    cat_25: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_90, unsqueeze_91], 4);  unsqueeze_90 = unsqueeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_184: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_25, [1, 128, 16, 64]);  cat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_86: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_26, 2)
    unsqueeze_87: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, 4);  unsqueeze_86 = None
    expand_50: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_87, [1, 128, 1, 32, 2]);  unsqueeze_87 = None
    clone_51: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    view_182: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_51, [1, 128, 1, 64]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_65: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_184, view_182);  view_184 = view_182 = None
    add_51: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_64, mul_65);  mul_64 = mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_304: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_176, 3, 64, 9223372036854775807);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_27: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_51, slice_304], 3);  add_51 = slice_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_71: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_27, [0, 2, 1, 3]);  cat_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_52: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_71, [1, 16, 128, 256]);  permute_71 = None
    view_185: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_52, [16, 128, 256]);  expand_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_172: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_49, [128, 4096])
    permute_67: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    mm_25: "f32[128, 4096]" = torch.ops.aten.mm.default(view_172, permute_67);  view_172 = permute_67 = None
    view_173: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_25, [1, 128, 4096]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_177: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_173, [1, 128, 16, 256]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_292: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_177, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_82: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_27, 2);  getitem_27 = None
    unsqueeze_83: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, 4);  unsqueeze_82 = None
    expand_49: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_83, [1, 128, 1, 32, 2]);  unsqueeze_83 = None
    clone_50: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_180: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_50, [1, 128, 1, 64]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_62: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_292, view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_318: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_292, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_12: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_318);  slice_318 = None
    unsqueeze_84: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_12, 4);  neg_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_314: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_292, 3, 0, 9223372036854775807, 2);  slice_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_85: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_314, 4);  slice_314 = None
    cat_24: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_84, unsqueeze_85], 4);  unsqueeze_84 = unsqueeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_181: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_24, [1, 128, 16, 64]);  cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_80: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_26, 2);  getitem_26 = None
    unsqueeze_81: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, 4);  unsqueeze_80 = None
    expand_48: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_81, [1, 128, 1, 32, 2]);  unsqueeze_81 = None
    clone_49: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_179: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_49, [1, 128, 1, 64]);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_63: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_181, view_179);  view_181 = view_179 = None
    add_50: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_296: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_177, 3, 64, 9223372036854775807);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_26: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_50, slice_296], 3);  add_50 = slice_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_70: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_26, [0, 2, 1, 3]);  cat_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_72: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2])
    expand_53: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_72, [1, 16, 256, 128]);  permute_72 = None
    view_186: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_53, [16, 256, 128]);  expand_53 = None
    bmm_12: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_185, view_186);  view_185 = view_186 = None
    view_187: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [1, 16, 128, 128]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_6: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_6: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_336, view_187, full_default_6);  slice_336 = view_187 = full_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_12: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_6, arg305_1);  where_6 = arg305_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_12, [-1], True)
    sub_13: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_12, amax_6);  div_12 = amax_6 = None
    exp_6: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_7: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_13: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_54: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_13, [1, 16, 128, 128]);  div_13 = None
    view_188: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_54, [16, 128, 128]);  expand_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_174: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_49, [128, 4096])
    permute_68: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    mm_26: "f32[128, 4096]" = torch.ops.aten.mm.default(view_174, permute_68);  view_174 = permute_68 = None
    view_175: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_26, [1, 128, 4096]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_178: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_175, [1, 128, 16, 256]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_69: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_55: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_69, [1, 16, 128, 256])
    view_189: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_55, [16, 128, 256]);  expand_55 = None
    bmm_13: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_188, view_189);  view_188 = view_189 = None
    view_190: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_13, [1, 16, 128, 256]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_54: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_191: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_54, [1, 128, 4096]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_192: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_191, [128, 4096]);  view_191 = None
    permute_74: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    mm_27: "f32[128, 4096]" = torch.ops.aten.mm.default(view_192, permute_74);  view_192 = permute_74 = None
    view_193: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_27, [1, 128, 4096]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_194: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_49, [128, 4096]);  add_49 = None
    permute_75: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    addmm_12: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg68_1, view_194, permute_75);  arg68_1 = view_194 = permute_75 = None
    view_195: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 16384]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_66: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    pow_7: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_195, 3.0)
    mul_67: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_52: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_195, mul_67);  view_195 = mul_67 = None
    mul_68: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_52, 0.7978845608028654);  add_52 = None
    tanh_6: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_68);  mul_68 = None
    add_53: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    mul_69: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_66, add_53);  mul_66 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_196: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_69, [128, 16384]);  mul_69 = None
    permute_76: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    addmm_13: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg70_1, view_196, permute_76);  arg70_1 = view_196 = permute_76 = None
    view_197: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_13, [1, 128, 4096]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_54: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_193, view_197);  view_193 = view_197 = None
    add_55: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_54, add_47);  add_54 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_14: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg306_1, [1, 1, 1]);  arg306_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_92: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_15: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_92, [1, 1, 64]);  unsqueeze_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_7: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_14, 1, repeat_15);  repeat_14 = repeat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_7 = torch.ops.aten.split_with_sizes.default(gather_7, [32, 32], 2);  gather_7 = None
    getitem_30: "f32[1, 128, 32]" = split_with_sizes_7[0]
    getitem_31: "f32[1, 128, 32]" = split_with_sizes_7[1];  split_with_sizes_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant7: "f32[]" = self._tensor_constant7
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_383: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg307_1, 2, 0, 128);  arg307_1 = None
    slice_384: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_383, 3, 0, 128);  slice_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_14: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_55, getitem_29);  getitem_29 = None
    add_56: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    mul_70: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_7);  sub_14 = rsqrt_7 = None
    mul_71: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_70, arg71_1);  mul_70 = arg71_1 = None
    add_57: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_71, arg72_1);  mul_71 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_198: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_57, [128, 4096])
    permute_77: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    mm_28: "f32[128, 4096]" = torch.ops.aten.mm.default(view_198, permute_77);  view_198 = permute_77 = None
    view_199: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_28, [1, 128, 4096]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_204: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_199, [1, 128, 16, 256]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_348: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_204, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_101: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_31, 2)
    unsqueeze_102: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 4);  unsqueeze_101 = None
    expand_59: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_102, [1, 128, 1, 32, 2]);  unsqueeze_102 = None
    clone_60: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
    view_211: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_60, [1, 128, 1, 64]);  clone_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_74: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_348, view_211);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_380: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_348, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_15: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_380);  slice_380 = None
    unsqueeze_103: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_15, 4);  neg_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_376: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_348, 3, 0, 9223372036854775807, 2);  slice_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_104: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_376, 4);  slice_376 = None
    cat_29: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_103, unsqueeze_104], 4);  unsqueeze_103 = unsqueeze_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_212: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_29, [1, 128, 16, 64]);  cat_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_99: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_30, 2)
    unsqueeze_100: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_99, 4);  unsqueeze_99 = None
    expand_58: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_100, [1, 128, 1, 32, 2]);  unsqueeze_100 = None
    clone_59: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
    view_210: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_59, [1, 128, 1, 64]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_75: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_212, view_210);  view_212 = view_210 = None
    add_59: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_74, mul_75);  mul_74 = mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_352: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_204, 3, 64, 9223372036854775807);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_31: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_59, slice_352], 3);  add_59 = slice_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_82: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_31, [0, 2, 1, 3]);  cat_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_60: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_82, [1, 16, 128, 256]);  permute_82 = None
    view_213: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_60, [16, 128, 256]);  expand_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_200: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_57, [128, 4096])
    permute_78: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    mm_29: "f32[128, 4096]" = torch.ops.aten.mm.default(view_200, permute_78);  view_200 = permute_78 = None
    view_201: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_29, [1, 128, 4096]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_205: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_201, [1, 128, 16, 256]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_340: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_205, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_95: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_31, 2);  getitem_31 = None
    unsqueeze_96: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_95, 4);  unsqueeze_95 = None
    expand_57: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_96, [1, 128, 1, 32, 2]);  unsqueeze_96 = None
    clone_58: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_208: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_58, [1, 128, 1, 64]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_72: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_340, view_208);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_366: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_340, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_14: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_366);  slice_366 = None
    unsqueeze_97: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_14, 4);  neg_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_362: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_340, 3, 0, 9223372036854775807, 2);  slice_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_98: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_362, 4);  slice_362 = None
    cat_28: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_97, unsqueeze_98], 4);  unsqueeze_97 = unsqueeze_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_209: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_28, [1, 128, 16, 64]);  cat_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_93: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_30, 2);  getitem_30 = None
    unsqueeze_94: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_93, 4);  unsqueeze_93 = None
    expand_56: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_94, [1, 128, 1, 32, 2]);  unsqueeze_94 = None
    clone_57: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_207: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_57, [1, 128, 1, 64]);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_73: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_209, view_207);  view_209 = view_207 = None
    add_58: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_72, mul_73);  mul_72 = mul_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_344: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_205, 3, 64, 9223372036854775807);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_30: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_58, slice_344], 3);  add_58 = slice_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_81: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_30, [0, 2, 1, 3]);  cat_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_83: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2])
    expand_61: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_83, [1, 16, 256, 128]);  permute_83 = None
    view_214: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_61, [16, 256, 128]);  expand_61 = None
    bmm_14: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_213, view_214);  view_213 = view_214 = None
    view_215: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, [1, 16, 128, 128]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_7: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_7: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_384, view_215, full_default_7);  slice_384 = view_215 = full_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_14: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_7, arg308_1);  where_7 = arg308_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_14, [-1], True)
    sub_15: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_14, amax_7);  div_14 = amax_7 = None
    exp_7: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_8: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_15: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_62: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_15, [1, 16, 128, 128]);  div_15 = None
    view_216: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_62, [16, 128, 128]);  expand_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_202: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_57, [128, 4096])
    permute_79: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    mm_30: "f32[128, 4096]" = torch.ops.aten.mm.default(view_202, permute_79);  view_202 = permute_79 = None
    view_203: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_30, [1, 128, 4096]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_206: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_203, [1, 128, 16, 256]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_80: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_63: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_80, [1, 16, 128, 256])
    view_217: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_63, [16, 128, 256]);  expand_63 = None
    bmm_15: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_216, view_217);  view_216 = view_217 = None
    view_218: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_15, [1, 16, 128, 256]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
    clone_62: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_219: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_62, [1, 128, 4096]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_220: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_219, [128, 4096]);  view_219 = None
    permute_85: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    mm_31: "f32[128, 4096]" = torch.ops.aten.mm.default(view_220, permute_85);  view_220 = permute_85 = None
    view_221: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_31, [1, 128, 4096]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_222: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_57, [128, 4096]);  add_57 = None
    permute_86: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    addmm_14: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg78_1, view_222, permute_86);  arg78_1 = view_222 = permute_86 = None
    view_223: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 16384]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_223, 0.5)
    pow_8: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_223, 3.0)
    mul_77: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_60: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_223, mul_77);  view_223 = mul_77 = None
    mul_78: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_60, 0.7978845608028654);  add_60 = None
    tanh_7: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
    add_61: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    mul_79: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_76, add_61);  mul_76 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_224: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_79, [128, 16384]);  mul_79 = None
    permute_87: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    addmm_15: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg80_1, view_224, permute_87);  arg80_1 = view_224 = permute_87 = None
    view_225: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_15, [1, 128, 4096]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_62: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_221, view_225);  view_221 = view_225 = None
    add_63: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_62, add_55);  add_62 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_16: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg309_1, [1, 1, 1]);  arg309_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_105: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_17: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_105, [1, 1, 64]);  unsqueeze_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_8: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_16, 1, repeat_17);  repeat_16 = repeat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_8 = torch.ops.aten.split_with_sizes.default(gather_8, [32, 32], 2);  gather_8 = None
    getitem_34: "f32[1, 128, 32]" = split_with_sizes_8[0]
    getitem_35: "f32[1, 128, 32]" = split_with_sizes_8[1];  split_with_sizes_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant8: "f32[]" = self._tensor_constant8
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_431: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg310_1, 2, 0, 128);  arg310_1 = None
    slice_432: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_431, 3, 0, 128);  slice_431 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_16: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_63, getitem_33);  getitem_33 = None
    add_64: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    mul_80: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_8);  sub_16 = rsqrt_8 = None
    mul_81: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_80, arg81_1);  mul_80 = arg81_1 = None
    add_65: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_81, arg82_1);  mul_81 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_226: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_65, [128, 4096])
    permute_88: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    mm_32: "f32[128, 4096]" = torch.ops.aten.mm.default(view_226, permute_88);  view_226 = permute_88 = None
    view_227: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_32, [1, 128, 4096]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_232: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_227, [1, 128, 16, 256]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_396: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_232, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_114: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_35, 2)
    unsqueeze_115: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, 4);  unsqueeze_114 = None
    expand_67: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_115, [1, 128, 1, 32, 2]);  unsqueeze_115 = None
    clone_68: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
    view_239: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_68, [1, 128, 1, 64]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_84: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_396, view_239);  view_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_428: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_396, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_17: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_428);  slice_428 = None
    unsqueeze_116: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_17, 4);  neg_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_424: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_396, 3, 0, 9223372036854775807, 2);  slice_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_117: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_424, 4);  slice_424 = None
    cat_33: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_116, unsqueeze_117], 4);  unsqueeze_116 = unsqueeze_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_240: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_33, [1, 128, 16, 64]);  cat_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_112: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_34, 2)
    unsqueeze_113: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, 4);  unsqueeze_112 = None
    expand_66: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_113, [1, 128, 1, 32, 2]);  unsqueeze_113 = None
    clone_67: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
    view_238: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_67, [1, 128, 1, 64]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_85: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_240, view_238);  view_240 = view_238 = None
    add_67: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_84, mul_85);  mul_84 = mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_400: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_232, 3, 64, 9223372036854775807);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_35: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_67, slice_400], 3);  add_67 = slice_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_93: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_35, [0, 2, 1, 3]);  cat_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_68: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_93, [1, 16, 128, 256]);  permute_93 = None
    view_241: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_68, [16, 128, 256]);  expand_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_228: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_65, [128, 4096])
    permute_89: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    mm_33: "f32[128, 4096]" = torch.ops.aten.mm.default(view_228, permute_89);  view_228 = permute_89 = None
    view_229: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_33, [1, 128, 4096]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_233: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_229, [1, 128, 16, 256]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_388: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_233, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_108: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_35, 2);  getitem_35 = None
    unsqueeze_109: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, 4);  unsqueeze_108 = None
    expand_65: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_109, [1, 128, 1, 32, 2]);  unsqueeze_109 = None
    clone_66: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_236: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_66, [1, 128, 1, 64]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_82: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_388, view_236);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_414: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_388, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_16: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_414);  slice_414 = None
    unsqueeze_110: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_16, 4);  neg_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_410: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_388, 3, 0, 9223372036854775807, 2);  slice_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_111: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_410, 4);  slice_410 = None
    cat_32: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_110, unsqueeze_111], 4);  unsqueeze_110 = unsqueeze_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_237: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_32, [1, 128, 16, 64]);  cat_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_106: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_34, 2);  getitem_34 = None
    unsqueeze_107: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, 4);  unsqueeze_106 = None
    expand_64: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_107, [1, 128, 1, 32, 2]);  unsqueeze_107 = None
    clone_65: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_235: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_65, [1, 128, 1, 64]);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_83: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_237, view_235);  view_237 = view_235 = None
    add_66: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_82, mul_83);  mul_82 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_392: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_233, 3, 64, 9223372036854775807);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_34: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_66, slice_392], 3);  add_66 = slice_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_92: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_34, [0, 2, 1, 3]);  cat_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_94: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2])
    expand_69: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_94, [1, 16, 256, 128]);  permute_94 = None
    view_242: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_69, [16, 256, 128]);  expand_69 = None
    bmm_16: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_241, view_242);  view_241 = view_242 = None
    view_243: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [1, 16, 128, 128]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_8: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_8: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_432, view_243, full_default_8);  slice_432 = view_243 = full_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_16: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_8, arg311_1);  where_8 = arg311_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_16, [-1], True)
    sub_17: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_16, amax_8);  div_16 = amax_8 = None
    exp_8: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_9: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_17: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_70: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_17, [1, 16, 128, 128]);  div_17 = None
    view_244: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_70, [16, 128, 128]);  expand_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_230: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_65, [128, 4096])
    permute_90: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    mm_34: "f32[128, 4096]" = torch.ops.aten.mm.default(view_230, permute_90);  view_230 = permute_90 = None
    view_231: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_34, [1, 128, 4096]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_234: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_231, [1, 128, 16, 256]);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_91: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_71: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_91, [1, 16, 128, 256])
    view_245: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_71, [16, 128, 256]);  expand_71 = None
    bmm_17: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_244, view_245);  view_244 = view_245 = None
    view_246: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_17, [1, 16, 128, 256]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    clone_70: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_247: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_70, [1, 128, 4096]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_248: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_247, [128, 4096]);  view_247 = None
    permute_96: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    mm_35: "f32[128, 4096]" = torch.ops.aten.mm.default(view_248, permute_96);  view_248 = permute_96 = None
    view_249: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_35, [1, 128, 4096]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_250: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_65, [128, 4096]);  add_65 = None
    permute_97: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    addmm_16: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg88_1, view_250, permute_97);  arg88_1 = view_250 = permute_97 = None
    view_251: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 16384]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_86: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_251, 0.5)
    pow_9: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_251, 3.0)
    mul_87: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_68: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_251, mul_87);  view_251 = mul_87 = None
    mul_88: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_68, 0.7978845608028654);  add_68 = None
    tanh_8: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_88);  mul_88 = None
    add_69: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    mul_89: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_86, add_69);  mul_86 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_252: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_89, [128, 16384]);  mul_89 = None
    permute_98: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    addmm_17: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg90_1, view_252, permute_98);  arg90_1 = view_252 = permute_98 = None
    view_253: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_17, [1, 128, 4096]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_70: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_249, view_253);  view_249 = view_253 = None
    add_71: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_70, add_63);  add_70 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_37: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_18: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg312_1, [1, 1, 1]);  arg312_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_118: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_19: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_118, [1, 1, 64]);  unsqueeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_9: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_18, 1, repeat_19);  repeat_18 = repeat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_9 = torch.ops.aten.split_with_sizes.default(gather_9, [32, 32], 2);  gather_9 = None
    getitem_38: "f32[1, 128, 32]" = split_with_sizes_9[0]
    getitem_39: "f32[1, 128, 32]" = split_with_sizes_9[1];  split_with_sizes_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant9: "f32[]" = self._tensor_constant9
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_479: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg313_1, 2, 0, 128);  arg313_1 = None
    slice_480: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_479, 3, 0, 128);  slice_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_18: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_71, getitem_37);  getitem_37 = None
    add_72: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    mul_90: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_9);  sub_18 = rsqrt_9 = None
    mul_91: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_90, arg91_1);  mul_90 = arg91_1 = None
    add_73: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_91, arg92_1);  mul_91 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_254: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_73, [128, 4096])
    permute_99: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    mm_36: "f32[128, 4096]" = torch.ops.aten.mm.default(view_254, permute_99);  view_254 = permute_99 = None
    view_255: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_36, [1, 128, 4096]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_260: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_255, [1, 128, 16, 256]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_444: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_260, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_127: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_39, 2)
    unsqueeze_128: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 4);  unsqueeze_127 = None
    expand_75: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_128, [1, 128, 1, 32, 2]);  unsqueeze_128 = None
    clone_76: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
    view_267: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_76, [1, 128, 1, 64]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_94: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_444, view_267);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_476: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_444, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_19: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_476);  slice_476 = None
    unsqueeze_129: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_19, 4);  neg_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_472: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_444, 3, 0, 9223372036854775807, 2);  slice_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_130: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_472, 4);  slice_472 = None
    cat_37: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_129, unsqueeze_130], 4);  unsqueeze_129 = unsqueeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_268: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_37, [1, 128, 16, 64]);  cat_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_125: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_38, 2)
    unsqueeze_126: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_125, 4);  unsqueeze_125 = None
    expand_74: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_126, [1, 128, 1, 32, 2]);  unsqueeze_126 = None
    clone_75: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    view_266: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_75, [1, 128, 1, 64]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_95: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_268, view_266);  view_268 = view_266 = None
    add_75: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_448: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_260, 3, 64, 9223372036854775807);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_39: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_75, slice_448], 3);  add_75 = slice_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_104: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_39, [0, 2, 1, 3]);  cat_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_76: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_104, [1, 16, 128, 256]);  permute_104 = None
    view_269: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_76, [16, 128, 256]);  expand_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_256: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_73, [128, 4096])
    permute_100: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    mm_37: "f32[128, 4096]" = torch.ops.aten.mm.default(view_256, permute_100);  view_256 = permute_100 = None
    view_257: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_37, [1, 128, 4096]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_261: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_257, [1, 128, 16, 256]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_436: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_261, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_121: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_39, 2);  getitem_39 = None
    unsqueeze_122: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_121, 4);  unsqueeze_121 = None
    expand_73: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_122, [1, 128, 1, 32, 2]);  unsqueeze_122 = None
    clone_74: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    view_264: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_74, [1, 128, 1, 64]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_92: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_436, view_264);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_462: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_436, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_18: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_462);  slice_462 = None
    unsqueeze_123: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_18, 4);  neg_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_458: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_436, 3, 0, 9223372036854775807, 2);  slice_436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_124: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_458, 4);  slice_458 = None
    cat_36: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_123, unsqueeze_124], 4);  unsqueeze_123 = unsqueeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_265: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_36, [1, 128, 16, 64]);  cat_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_119: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_38, 2);  getitem_38 = None
    unsqueeze_120: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_119, 4);  unsqueeze_119 = None
    expand_72: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_120, [1, 128, 1, 32, 2]);  unsqueeze_120 = None
    clone_73: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_263: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_73, [1, 128, 1, 64]);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_93: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_265, view_263);  view_265 = view_263 = None
    add_74: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_92, mul_93);  mul_92 = mul_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_440: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_261, 3, 64, 9223372036854775807);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_38: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_74, slice_440], 3);  add_74 = slice_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_103: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_38, [0, 2, 1, 3]);  cat_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_105: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2])
    expand_77: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_105, [1, 16, 256, 128]);  permute_105 = None
    view_270: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_77, [16, 256, 128]);  expand_77 = None
    bmm_18: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_269, view_270);  view_269 = view_270 = None
    view_271: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_18, [1, 16, 128, 128]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_9: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_9: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_480, view_271, full_default_9);  slice_480 = view_271 = full_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_18: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_9, arg314_1);  where_9 = arg314_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_18, [-1], True)
    sub_19: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_18, amax_9);  div_18 = amax_9 = None
    exp_9: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_10: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_19: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_78: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_19, [1, 16, 128, 128]);  div_19 = None
    view_272: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_78, [16, 128, 128]);  expand_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_258: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_73, [128, 4096])
    permute_101: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    mm_38: "f32[128, 4096]" = torch.ops.aten.mm.default(view_258, permute_101);  view_258 = permute_101 = None
    view_259: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_38, [1, 128, 4096]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_262: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_259, [1, 128, 16, 256]);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_102: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_79: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_102, [1, 16, 128, 256])
    view_273: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_79, [16, 128, 256]);  expand_79 = None
    bmm_19: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_272, view_273);  view_272 = view_273 = None
    view_274: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_19, [1, 16, 128, 256]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
    clone_78: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_275: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_78, [1, 128, 4096]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_276: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_275, [128, 4096]);  view_275 = None
    permute_107: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    mm_39: "f32[128, 4096]" = torch.ops.aten.mm.default(view_276, permute_107);  view_276 = permute_107 = None
    view_277: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_39, [1, 128, 4096]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_278: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_73, [128, 4096]);  add_73 = None
    permute_108: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    addmm_18: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg98_1, view_278, permute_108);  arg98_1 = view_278 = permute_108 = None
    view_279: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 16384]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_96: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_279, 0.5)
    pow_10: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_279, 3.0)
    mul_97: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_76: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_279, mul_97);  view_279 = mul_97 = None
    mul_98: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_76, 0.7978845608028654);  add_76 = None
    tanh_9: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_98);  mul_98 = None
    add_77: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    mul_99: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_96, add_77);  mul_96 = add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_280: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_99, [128, 16384]);  mul_99 = None
    permute_109: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
    addmm_19: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg100_1, view_280, permute_109);  arg100_1 = view_280 = permute_109 = None
    view_281: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_19, [1, 128, 4096]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_78: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_277, view_281);  view_277 = view_281 = None
    add_79: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_78, add_71);  add_78 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_41: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_20: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg315_1, [1, 1, 1]);  arg315_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_131: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_21: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_131, [1, 1, 64]);  unsqueeze_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_10: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_20, 1, repeat_21);  repeat_20 = repeat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_10 = torch.ops.aten.split_with_sizes.default(gather_10, [32, 32], 2);  gather_10 = None
    getitem_42: "f32[1, 128, 32]" = split_with_sizes_10[0]
    getitem_43: "f32[1, 128, 32]" = split_with_sizes_10[1];  split_with_sizes_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant10: "f32[]" = self._tensor_constant10
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_527: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg316_1, 2, 0, 128);  arg316_1 = None
    slice_528: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_527, 3, 0, 128);  slice_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_20: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_79, getitem_41);  getitem_41 = None
    add_80: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    mul_100: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_10);  sub_20 = rsqrt_10 = None
    mul_101: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_100, arg101_1);  mul_100 = arg101_1 = None
    add_81: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_101, arg102_1);  mul_101 = arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_282: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_81, [128, 4096])
    permute_110: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    mm_40: "f32[128, 4096]" = torch.ops.aten.mm.default(view_282, permute_110);  view_282 = permute_110 = None
    view_283: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_40, [1, 128, 4096]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_288: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_283, [1, 128, 16, 256]);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_492: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_288, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_140: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_43, 2)
    unsqueeze_141: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 4);  unsqueeze_140 = None
    expand_83: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_141, [1, 128, 1, 32, 2]);  unsqueeze_141 = None
    clone_84: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
    view_295: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_84, [1, 128, 1, 64]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_104: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_492, view_295);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_524: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_492, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_21: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_524);  slice_524 = None
    unsqueeze_142: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_21, 4);  neg_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_520: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_492, 3, 0, 9223372036854775807, 2);  slice_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_143: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_520, 4);  slice_520 = None
    cat_41: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_142, unsqueeze_143], 4);  unsqueeze_142 = unsqueeze_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_296: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_41, [1, 128, 16, 64]);  cat_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_138: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_42, 2)
    unsqueeze_139: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 4);  unsqueeze_138 = None
    expand_82: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_139, [1, 128, 1, 32, 2]);  unsqueeze_139 = None
    clone_83: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
    view_294: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_83, [1, 128, 1, 64]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_105: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_296, view_294);  view_296 = view_294 = None
    add_83: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_104, mul_105);  mul_104 = mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_496: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_288, 3, 64, 9223372036854775807);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_43: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_83, slice_496], 3);  add_83 = slice_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_115: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_43, [0, 2, 1, 3]);  cat_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_84: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_115, [1, 16, 128, 256]);  permute_115 = None
    view_297: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_84, [16, 128, 256]);  expand_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_284: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_81, [128, 4096])
    permute_111: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    mm_41: "f32[128, 4096]" = torch.ops.aten.mm.default(view_284, permute_111);  view_284 = permute_111 = None
    view_285: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_41, [1, 128, 4096]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_289: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_285, [1, 128, 16, 256]);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_484: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_289, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_134: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_43, 2);  getitem_43 = None
    unsqueeze_135: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 4);  unsqueeze_134 = None
    expand_81: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_135, [1, 128, 1, 32, 2]);  unsqueeze_135 = None
    clone_82: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_292: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_82, [1, 128, 1, 64]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_102: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_484, view_292);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_510: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_484, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_20: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_510);  slice_510 = None
    unsqueeze_136: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_20, 4);  neg_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_506: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_484, 3, 0, 9223372036854775807, 2);  slice_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_137: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_506, 4);  slice_506 = None
    cat_40: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_136, unsqueeze_137], 4);  unsqueeze_136 = unsqueeze_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_293: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_40, [1, 128, 16, 64]);  cat_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_132: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_42, 2);  getitem_42 = None
    unsqueeze_133: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, 4);  unsqueeze_132 = None
    expand_80: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_133, [1, 128, 1, 32, 2]);  unsqueeze_133 = None
    clone_81: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_291: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_81, [1, 128, 1, 64]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_103: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_293, view_291);  view_293 = view_291 = None
    add_82: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_102, mul_103);  mul_102 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_488: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_289, 3, 64, 9223372036854775807);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_42: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_82, slice_488], 3);  add_82 = slice_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_114: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_42, [0, 2, 1, 3]);  cat_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_116: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2])
    expand_85: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_116, [1, 16, 256, 128]);  permute_116 = None
    view_298: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_85, [16, 256, 128]);  expand_85 = None
    bmm_20: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_297, view_298);  view_297 = view_298 = None
    view_299: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [1, 16, 128, 128]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_10: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_10: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_528, view_299, full_default_10);  slice_528 = view_299 = full_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_20: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_10, arg317_1);  where_10 = arg317_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_20, [-1], True)
    sub_21: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_20, amax_10);  div_20 = amax_10 = None
    exp_10: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_11: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_21: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_86: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_21, [1, 16, 128, 128]);  div_21 = None
    view_300: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_86, [16, 128, 128]);  expand_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_286: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_81, [128, 4096])
    permute_112: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    mm_42: "f32[128, 4096]" = torch.ops.aten.mm.default(view_286, permute_112);  view_286 = permute_112 = None
    view_287: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_42, [1, 128, 4096]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_290: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_287, [1, 128, 16, 256]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_113: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_87: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_113, [1, 16, 128, 256])
    view_301: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_87, [16, 128, 256]);  expand_87 = None
    bmm_21: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_300, view_301);  view_300 = view_301 = None
    view_302: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_21, [1, 16, 128, 256]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
    clone_86: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_303: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_86, [1, 128, 4096]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_304: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_303, [128, 4096]);  view_303 = None
    permute_118: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    mm_43: "f32[128, 4096]" = torch.ops.aten.mm.default(view_304, permute_118);  view_304 = permute_118 = None
    view_305: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_43, [1, 128, 4096]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_306: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_81, [128, 4096]);  add_81 = None
    permute_119: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    addmm_20: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg108_1, view_306, permute_119);  arg108_1 = view_306 = permute_119 = None
    view_307: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 16384]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_106: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_307, 0.5)
    pow_11: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_307, 3.0)
    mul_107: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_84: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_307, mul_107);  view_307 = mul_107 = None
    mul_108: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_84, 0.7978845608028654);  add_84 = None
    tanh_10: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_108);  mul_108 = None
    add_85: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    mul_109: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_106, add_85);  mul_106 = add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_308: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_109, [128, 16384]);  mul_109 = None
    permute_120: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    addmm_21: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg110_1, view_308, permute_120);  arg110_1 = view_308 = permute_120 = None
    view_309: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_21, [1, 128, 4096]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_86: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_305, view_309);  view_305 = view_309 = None
    add_87: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_86, add_79);  add_86 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_45: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_22: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg318_1, [1, 1, 1]);  arg318_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_144: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_23: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_144, [1, 1, 64]);  unsqueeze_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_11: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_22, 1, repeat_23);  repeat_22 = repeat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_11 = torch.ops.aten.split_with_sizes.default(gather_11, [32, 32], 2);  gather_11 = None
    getitem_46: "f32[1, 128, 32]" = split_with_sizes_11[0]
    getitem_47: "f32[1, 128, 32]" = split_with_sizes_11[1];  split_with_sizes_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant11: "f32[]" = self._tensor_constant11
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_575: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg319_1, 2, 0, 128);  arg319_1 = None
    slice_576: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_575, 3, 0, 128);  slice_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_22: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_87, getitem_45);  getitem_45 = None
    add_88: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    mul_110: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_11);  sub_22 = rsqrt_11 = None
    mul_111: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_110, arg111_1);  mul_110 = arg111_1 = None
    add_89: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_111, arg112_1);  mul_111 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_310: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_89, [128, 4096])
    permute_121: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    mm_44: "f32[128, 4096]" = torch.ops.aten.mm.default(view_310, permute_121);  view_310 = permute_121 = None
    view_311: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_44, [1, 128, 4096]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_316: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_311, [1, 128, 16, 256]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_540: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_316, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_153: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_47, 2)
    unsqueeze_154: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 4);  unsqueeze_153 = None
    expand_91: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_154, [1, 128, 1, 32, 2]);  unsqueeze_154 = None
    clone_92: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
    view_323: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_92, [1, 128, 1, 64]);  clone_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_114: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_540, view_323);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_572: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_540, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_23: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_572);  slice_572 = None
    unsqueeze_155: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_23, 4);  neg_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_568: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_540, 3, 0, 9223372036854775807, 2);  slice_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_156: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_568, 4);  slice_568 = None
    cat_45: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_155, unsqueeze_156], 4);  unsqueeze_155 = unsqueeze_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_324: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_45, [1, 128, 16, 64]);  cat_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_151: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_46, 2)
    unsqueeze_152: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 4);  unsqueeze_151 = None
    expand_90: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_152, [1, 128, 1, 32, 2]);  unsqueeze_152 = None
    clone_91: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
    view_322: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_91, [1, 128, 1, 64]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_115: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_324, view_322);  view_324 = view_322 = None
    add_91: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_114, mul_115);  mul_114 = mul_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_544: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_316, 3, 64, 9223372036854775807);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_47: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_91, slice_544], 3);  add_91 = slice_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_126: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_47, [0, 2, 1, 3]);  cat_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_92: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_126, [1, 16, 128, 256]);  permute_126 = None
    view_325: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_92, [16, 128, 256]);  expand_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_312: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_89, [128, 4096])
    permute_122: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    mm_45: "f32[128, 4096]" = torch.ops.aten.mm.default(view_312, permute_122);  view_312 = permute_122 = None
    view_313: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_45, [1, 128, 4096]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_317: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_313, [1, 128, 16, 256]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_532: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_317, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_147: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_47, 2);  getitem_47 = None
    unsqueeze_148: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_147, 4);  unsqueeze_147 = None
    expand_89: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_148, [1, 128, 1, 32, 2]);  unsqueeze_148 = None
    clone_90: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    view_320: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_90, [1, 128, 1, 64]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_112: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_532, view_320);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_558: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_532, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_22: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_558);  slice_558 = None
    unsqueeze_149: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_22, 4);  neg_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_554: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_532, 3, 0, 9223372036854775807, 2);  slice_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_150: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_554, 4);  slice_554 = None
    cat_44: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_149, unsqueeze_150], 4);  unsqueeze_149 = unsqueeze_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_321: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_44, [1, 128, 16, 64]);  cat_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_145: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_46, 2);  getitem_46 = None
    unsqueeze_146: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 4);  unsqueeze_145 = None
    expand_88: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_146, [1, 128, 1, 32, 2]);  unsqueeze_146 = None
    clone_89: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
    view_319: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_89, [1, 128, 1, 64]);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_113: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_321, view_319);  view_321 = view_319 = None
    add_90: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_112, mul_113);  mul_112 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_536: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_317, 3, 64, 9223372036854775807);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_46: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_90, slice_536], 3);  add_90 = slice_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_125: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_46, [0, 2, 1, 3]);  cat_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_127: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2])
    expand_93: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_127, [1, 16, 256, 128]);  permute_127 = None
    view_326: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_93, [16, 256, 128]);  expand_93 = None
    bmm_22: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_325, view_326);  view_325 = view_326 = None
    view_327: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, [1, 16, 128, 128]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_11: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_11: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_576, view_327, full_default_11);  slice_576 = view_327 = full_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_22: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_11, arg320_1);  where_11 = arg320_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_22, [-1], True)
    sub_23: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_22, amax_11);  div_22 = amax_11 = None
    exp_11: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_12: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_23: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_94: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_23, [1, 16, 128, 128]);  div_23 = None
    view_328: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_94, [16, 128, 128]);  expand_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_314: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_89, [128, 4096])
    permute_123: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    mm_46: "f32[128, 4096]" = torch.ops.aten.mm.default(view_314, permute_123);  view_314 = permute_123 = None
    view_315: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_46, [1, 128, 4096]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_318: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_315, [1, 128, 16, 256]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_124: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_95: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_124, [1, 16, 128, 256])
    view_329: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_95, [16, 128, 256]);  expand_95 = None
    bmm_23: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_328, view_329);  view_328 = view_329 = None
    view_330: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_23, [1, 16, 128, 256]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    clone_94: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_331: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_94, [1, 128, 4096]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_332: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_331, [128, 4096]);  view_331 = None
    permute_129: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    mm_47: "f32[128, 4096]" = torch.ops.aten.mm.default(view_332, permute_129);  view_332 = permute_129 = None
    view_333: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_47, [1, 128, 4096]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_334: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_89, [128, 4096]);  add_89 = None
    permute_130: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    addmm_22: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg118_1, view_334, permute_130);  arg118_1 = view_334 = permute_130 = None
    view_335: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 16384]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_116: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_335, 0.5)
    pow_12: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_335, 3.0)
    mul_117: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_92: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_335, mul_117);  view_335 = mul_117 = None
    mul_118: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_92, 0.7978845608028654);  add_92 = None
    tanh_11: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
    add_93: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    mul_119: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_116, add_93);  mul_116 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_336: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_119, [128, 16384]);  mul_119 = None
    permute_131: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    addmm_23: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg120_1, view_336, permute_131);  arg120_1 = view_336 = permute_131 = None
    view_337: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_23, [1, 128, 4096]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_94: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_333, view_337);  view_333 = view_337 = None
    add_95: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_94, add_87);  add_94 = add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_49: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_24: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg321_1, [1, 1, 1]);  arg321_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_157: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_25: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_157, [1, 1, 64]);  unsqueeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_12: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_24, 1, repeat_25);  repeat_24 = repeat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_12 = torch.ops.aten.split_with_sizes.default(gather_12, [32, 32], 2);  gather_12 = None
    getitem_50: "f32[1, 128, 32]" = split_with_sizes_12[0]
    getitem_51: "f32[1, 128, 32]" = split_with_sizes_12[1];  split_with_sizes_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant12: "f32[]" = self._tensor_constant12
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_623: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg322_1, 2, 0, 128);  arg322_1 = None
    slice_624: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_623, 3, 0, 128);  slice_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_24: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_95, getitem_49);  getitem_49 = None
    add_96: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    mul_120: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_12);  sub_24 = rsqrt_12 = None
    mul_121: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_120, arg121_1);  mul_120 = arg121_1 = None
    add_97: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_121, arg122_1);  mul_121 = arg122_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_338: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_97, [128, 4096])
    permute_132: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    mm_48: "f32[128, 4096]" = torch.ops.aten.mm.default(view_338, permute_132);  view_338 = permute_132 = None
    view_339: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_48, [1, 128, 4096]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_344: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_339, [1, 128, 16, 256]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_588: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_344, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_166: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_51, 2)
    unsqueeze_167: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 4);  unsqueeze_166 = None
    expand_99: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_167, [1, 128, 1, 32, 2]);  unsqueeze_167 = None
    clone_100: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_99, memory_format = torch.contiguous_format);  expand_99 = None
    view_351: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_100, [1, 128, 1, 64]);  clone_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_124: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_588, view_351);  view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_620: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_588, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_25: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_620);  slice_620 = None
    unsqueeze_168: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_25, 4);  neg_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_616: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_588, 3, 0, 9223372036854775807, 2);  slice_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_169: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_616, 4);  slice_616 = None
    cat_49: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_168, unsqueeze_169], 4);  unsqueeze_168 = unsqueeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_352: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_49, [1, 128, 16, 64]);  cat_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_164: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_50, 2)
    unsqueeze_165: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 4);  unsqueeze_164 = None
    expand_98: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_165, [1, 128, 1, 32, 2]);  unsqueeze_165 = None
    clone_99: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
    view_350: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_99, [1, 128, 1, 64]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_125: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_352, view_350);  view_352 = view_350 = None
    add_99: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_124, mul_125);  mul_124 = mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_592: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_344, 3, 64, 9223372036854775807);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_51: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_99, slice_592], 3);  add_99 = slice_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_137: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_51, [0, 2, 1, 3]);  cat_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_100: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_137, [1, 16, 128, 256]);  permute_137 = None
    view_353: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_100, [16, 128, 256]);  expand_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_340: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_97, [128, 4096])
    permute_133: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    mm_49: "f32[128, 4096]" = torch.ops.aten.mm.default(view_340, permute_133);  view_340 = permute_133 = None
    view_341: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_49, [1, 128, 4096]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_345: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_341, [1, 128, 16, 256]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_580: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_345, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_160: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_51, 2);  getitem_51 = None
    unsqueeze_161: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, 4);  unsqueeze_160 = None
    expand_97: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_161, [1, 128, 1, 32, 2]);  unsqueeze_161 = None
    clone_98: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
    view_348: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_98, [1, 128, 1, 64]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_122: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_580, view_348);  view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_606: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_580, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_24: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_606);  slice_606 = None
    unsqueeze_162: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_24, 4);  neg_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_602: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_580, 3, 0, 9223372036854775807, 2);  slice_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_163: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_602, 4);  slice_602 = None
    cat_48: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_162, unsqueeze_163], 4);  unsqueeze_162 = unsqueeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_349: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_48, [1, 128, 16, 64]);  cat_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_158: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_50, 2);  getitem_50 = None
    unsqueeze_159: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 4);  unsqueeze_158 = None
    expand_96: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_159, [1, 128, 1, 32, 2]);  unsqueeze_159 = None
    clone_97: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
    view_347: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_97, [1, 128, 1, 64]);  clone_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_123: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_349, view_347);  view_349 = view_347 = None
    add_98: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_122, mul_123);  mul_122 = mul_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_584: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_345, 3, 64, 9223372036854775807);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_50: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_98, slice_584], 3);  add_98 = slice_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_136: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_50, [0, 2, 1, 3]);  cat_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_138: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_136, [0, 1, 3, 2])
    expand_101: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_138, [1, 16, 256, 128]);  permute_138 = None
    view_354: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_101, [16, 256, 128]);  expand_101 = None
    bmm_24: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_353, view_354);  view_353 = view_354 = None
    view_355: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [1, 16, 128, 128]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_12: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_12: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_624, view_355, full_default_12);  slice_624 = view_355 = full_default_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_24: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_12, arg323_1);  where_12 = arg323_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_24, [-1], True)
    sub_25: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_24, amax_12);  div_24 = amax_12 = None
    exp_12: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_13: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_25: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_102: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_25, [1, 16, 128, 128]);  div_25 = None
    view_356: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_102, [16, 128, 128]);  expand_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_342: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_97, [128, 4096])
    permute_134: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    mm_50: "f32[128, 4096]" = torch.ops.aten.mm.default(view_342, permute_134);  view_342 = permute_134 = None
    view_343: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_50, [1, 128, 4096]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_346: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_343, [1, 128, 16, 256]);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_135: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_103: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_135, [1, 16, 128, 256])
    view_357: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_103, [16, 128, 256]);  expand_103 = None
    bmm_25: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_356, view_357);  view_356 = view_357 = None
    view_358: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_25, [1, 16, 128, 256]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_139: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None
    clone_102: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_359: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_102, [1, 128, 4096]);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_360: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_359, [128, 4096]);  view_359 = None
    permute_140: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    mm_51: "f32[128, 4096]" = torch.ops.aten.mm.default(view_360, permute_140);  view_360 = permute_140 = None
    view_361: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_51, [1, 128, 4096]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_362: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_97, [128, 4096]);  add_97 = None
    permute_141: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    addmm_24: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg128_1, view_362, permute_141);  arg128_1 = view_362 = permute_141 = None
    view_363: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 16384]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_126: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_363, 0.5)
    pow_13: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_363, 3.0)
    mul_127: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
    add_100: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_363, mul_127);  view_363 = mul_127 = None
    mul_128: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
    tanh_12: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_128);  mul_128 = None
    add_101: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
    mul_129: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_126, add_101);  mul_126 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_364: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_129, [128, 16384]);  mul_129 = None
    permute_142: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    addmm_25: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg130_1, view_364, permute_142);  arg130_1 = view_364 = permute_142 = None
    view_365: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_25, [1, 128, 4096]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_102: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_361, view_365);  view_361 = view_365 = None
    add_103: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_102, add_95);  add_102 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_53: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_26: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg324_1, [1, 1, 1]);  arg324_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_170: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_27: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_170, [1, 1, 64]);  unsqueeze_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_13: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_26, 1, repeat_27);  repeat_26 = repeat_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_13 = torch.ops.aten.split_with_sizes.default(gather_13, [32, 32], 2);  gather_13 = None
    getitem_54: "f32[1, 128, 32]" = split_with_sizes_13[0]
    getitem_55: "f32[1, 128, 32]" = split_with_sizes_13[1];  split_with_sizes_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant13: "f32[]" = self._tensor_constant13
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_671: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg325_1, 2, 0, 128);  arg325_1 = None
    slice_672: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_671, 3, 0, 128);  slice_671 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_26: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_103, getitem_53);  getitem_53 = None
    add_104: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
    mul_130: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_13);  sub_26 = rsqrt_13 = None
    mul_131: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_130, arg131_1);  mul_130 = arg131_1 = None
    add_105: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_131, arg132_1);  mul_131 = arg132_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_366: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_105, [128, 4096])
    permute_143: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    mm_52: "f32[128, 4096]" = torch.ops.aten.mm.default(view_366, permute_143);  view_366 = permute_143 = None
    view_367: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_52, [1, 128, 4096]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_372: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_367, [1, 128, 16, 256]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_636: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_372, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_179: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_55, 2)
    unsqueeze_180: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 4);  unsqueeze_179 = None
    expand_107: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_180, [1, 128, 1, 32, 2]);  unsqueeze_180 = None
    clone_108: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
    view_379: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_108, [1, 128, 1, 64]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_134: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_636, view_379);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_668: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_636, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_27: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_668);  slice_668 = None
    unsqueeze_181: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_27, 4);  neg_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_664: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_636, 3, 0, 9223372036854775807, 2);  slice_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_182: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_664, 4);  slice_664 = None
    cat_53: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_181, unsqueeze_182], 4);  unsqueeze_181 = unsqueeze_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_380: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_53, [1, 128, 16, 64]);  cat_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_177: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_54, 2)
    unsqueeze_178: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_177, 4);  unsqueeze_177 = None
    expand_106: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_178, [1, 128, 1, 32, 2]);  unsqueeze_178 = None
    clone_107: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_106, memory_format = torch.contiguous_format);  expand_106 = None
    view_378: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_107, [1, 128, 1, 64]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_135: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_380, view_378);  view_380 = view_378 = None
    add_107: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_134, mul_135);  mul_134 = mul_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_640: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_372, 3, 64, 9223372036854775807);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_55: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_107, slice_640], 3);  add_107 = slice_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_148: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_55, [0, 2, 1, 3]);  cat_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_108: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_148, [1, 16, 128, 256]);  permute_148 = None
    view_381: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_108, [16, 128, 256]);  expand_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_368: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_105, [128, 4096])
    permute_144: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    mm_53: "f32[128, 4096]" = torch.ops.aten.mm.default(view_368, permute_144);  view_368 = permute_144 = None
    view_369: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_53, [1, 128, 4096]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_373: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_369, [1, 128, 16, 256]);  view_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_628: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_373, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_173: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_55, 2);  getitem_55 = None
    unsqueeze_174: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 4);  unsqueeze_173 = None
    expand_105: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_174, [1, 128, 1, 32, 2]);  unsqueeze_174 = None
    clone_106: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
    view_376: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_106, [1, 128, 1, 64]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_132: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_628, view_376);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_654: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_628, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_26: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_654);  slice_654 = None
    unsqueeze_175: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_26, 4);  neg_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_650: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_628, 3, 0, 9223372036854775807, 2);  slice_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_176: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_650, 4);  slice_650 = None
    cat_52: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_175, unsqueeze_176], 4);  unsqueeze_175 = unsqueeze_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_377: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_52, [1, 128, 16, 64]);  cat_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_171: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_54, 2);  getitem_54 = None
    unsqueeze_172: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 4);  unsqueeze_171 = None
    expand_104: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_172, [1, 128, 1, 32, 2]);  unsqueeze_172 = None
    clone_105: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_104, memory_format = torch.contiguous_format);  expand_104 = None
    view_375: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_105, [1, 128, 1, 64]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_133: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_377, view_375);  view_377 = view_375 = None
    add_106: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_632: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_373, 3, 64, 9223372036854775807);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_54: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_106, slice_632], 3);  add_106 = slice_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_147: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_54, [0, 2, 1, 3]);  cat_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_149: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_147, [0, 1, 3, 2])
    expand_109: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_149, [1, 16, 256, 128]);  permute_149 = None
    view_382: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_109, [16, 256, 128]);  expand_109 = None
    bmm_26: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_381, view_382);  view_381 = view_382 = None
    view_383: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_26, [1, 16, 128, 128]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_13: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_13: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_672, view_383, full_default_13);  slice_672 = view_383 = full_default_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_26: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_13, arg326_1);  where_13 = arg326_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_13: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_26, [-1], True)
    sub_27: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_26, amax_13);  div_26 = amax_13 = None
    exp_13: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_14: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_27: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_110: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_27, [1, 16, 128, 128]);  div_27 = None
    view_384: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_110, [16, 128, 128]);  expand_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_370: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_105, [128, 4096])
    permute_145: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    mm_54: "f32[128, 4096]" = torch.ops.aten.mm.default(view_370, permute_145);  view_370 = permute_145 = None
    view_371: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_54, [1, 128, 4096]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_374: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_371, [1, 128, 16, 256]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_146: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_374, [0, 2, 1, 3]);  view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_111: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_146, [1, 16, 128, 256])
    view_385: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_111, [16, 128, 256]);  expand_111 = None
    bmm_27: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_384, view_385);  view_384 = view_385 = None
    view_386: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_27, [1, 16, 128, 256]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_150: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
    clone_110: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_387: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_110, [1, 128, 4096]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_388: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_387, [128, 4096]);  view_387 = None
    permute_151: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    mm_55: "f32[128, 4096]" = torch.ops.aten.mm.default(view_388, permute_151);  view_388 = permute_151 = None
    view_389: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_55, [1, 128, 4096]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_390: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_105, [128, 4096]);  add_105 = None
    permute_152: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_26: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg138_1, view_390, permute_152);  arg138_1 = view_390 = permute_152 = None
    view_391: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 16384]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_136: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_391, 0.5)
    pow_14: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_391, 3.0)
    mul_137: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
    add_108: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_391, mul_137);  view_391 = mul_137 = None
    mul_138: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_108, 0.7978845608028654);  add_108 = None
    tanh_13: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_138);  mul_138 = None
    add_109: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
    mul_139: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_136, add_109);  mul_136 = add_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_392: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_139, [128, 16384]);  mul_139 = None
    permute_153: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    addmm_27: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg140_1, view_392, permute_153);  arg140_1 = view_392 = permute_153 = None
    view_393: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_27, [1, 128, 4096]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_110: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_389, view_393);  view_389 = view_393 = None
    add_111: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_110, add_103);  add_110 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_111, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_57: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_28: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg327_1, [1, 1, 1]);  arg327_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_183: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_29: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_183, [1, 1, 64]);  unsqueeze_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_14: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_28, 1, repeat_29);  repeat_28 = repeat_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_14 = torch.ops.aten.split_with_sizes.default(gather_14, [32, 32], 2);  gather_14 = None
    getitem_58: "f32[1, 128, 32]" = split_with_sizes_14[0]
    getitem_59: "f32[1, 128, 32]" = split_with_sizes_14[1];  split_with_sizes_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant14: "f32[]" = self._tensor_constant14
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_719: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg328_1, 2, 0, 128);  arg328_1 = None
    slice_720: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_719, 3, 0, 128);  slice_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_28: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_111, getitem_57);  getitem_57 = None
    add_112: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    mul_140: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_14);  sub_28 = rsqrt_14 = None
    mul_141: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_140, arg141_1);  mul_140 = arg141_1 = None
    add_113: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_141, arg142_1);  mul_141 = arg142_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_394: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_113, [128, 4096])
    permute_154: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    mm_56: "f32[128, 4096]" = torch.ops.aten.mm.default(view_394, permute_154);  view_394 = permute_154 = None
    view_395: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_56, [1, 128, 4096]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_400: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_395, [1, 128, 16, 256]);  view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_684: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_400, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_192: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_59, 2)
    unsqueeze_193: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 4);  unsqueeze_192 = None
    expand_115: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_193, [1, 128, 1, 32, 2]);  unsqueeze_193 = None
    clone_116: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
    view_407: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_116, [1, 128, 1, 64]);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_144: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_684, view_407);  view_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_716: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_684, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_29: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_716);  slice_716 = None
    unsqueeze_194: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_29, 4);  neg_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_712: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_684, 3, 0, 9223372036854775807, 2);  slice_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_195: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_712, 4);  slice_712 = None
    cat_57: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_194, unsqueeze_195], 4);  unsqueeze_194 = unsqueeze_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_408: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_57, [1, 128, 16, 64]);  cat_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_190: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_58, 2)
    unsqueeze_191: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 4);  unsqueeze_190 = None
    expand_114: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_191, [1, 128, 1, 32, 2]);  unsqueeze_191 = None
    clone_115: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_114, memory_format = torch.contiguous_format);  expand_114 = None
    view_406: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_115, [1, 128, 1, 64]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_145: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_408, view_406);  view_408 = view_406 = None
    add_115: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_144, mul_145);  mul_144 = mul_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_688: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_400, 3, 64, 9223372036854775807);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_59: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_115, slice_688], 3);  add_115 = slice_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_159: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_59, [0, 2, 1, 3]);  cat_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_116: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_159, [1, 16, 128, 256]);  permute_159 = None
    view_409: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_116, [16, 128, 256]);  expand_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_396: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_113, [128, 4096])
    permute_155: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    mm_57: "f32[128, 4096]" = torch.ops.aten.mm.default(view_396, permute_155);  view_396 = permute_155 = None
    view_397: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_57, [1, 128, 4096]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_401: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_397, [1, 128, 16, 256]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_676: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_401, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_186: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_59, 2);  getitem_59 = None
    unsqueeze_187: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 4);  unsqueeze_186 = None
    expand_113: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_187, [1, 128, 1, 32, 2]);  unsqueeze_187 = None
    clone_114: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
    view_404: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_114, [1, 128, 1, 64]);  clone_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_142: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_676, view_404);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_702: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_676, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_28: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_702);  slice_702 = None
    unsqueeze_188: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_28, 4);  neg_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_698: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_676, 3, 0, 9223372036854775807, 2);  slice_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_189: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_698, 4);  slice_698 = None
    cat_56: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_188, unsqueeze_189], 4);  unsqueeze_188 = unsqueeze_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_405: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_56, [1, 128, 16, 64]);  cat_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_184: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_58, 2);  getitem_58 = None
    unsqueeze_185: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 4);  unsqueeze_184 = None
    expand_112: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_185, [1, 128, 1, 32, 2]);  unsqueeze_185 = None
    clone_113: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
    view_403: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_113, [1, 128, 1, 64]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_143: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_405, view_403);  view_405 = view_403 = None
    add_114: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_142, mul_143);  mul_142 = mul_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_680: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_401, 3, 64, 9223372036854775807);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_58: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_114, slice_680], 3);  add_114 = slice_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_158: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_58, [0, 2, 1, 3]);  cat_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_160: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_158, [0, 1, 3, 2])
    expand_117: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_160, [1, 16, 256, 128]);  permute_160 = None
    view_410: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_117, [16, 256, 128]);  expand_117 = None
    bmm_28: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_409, view_410);  view_409 = view_410 = None
    view_411: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [1, 16, 128, 128]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_14: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_14: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_720, view_411, full_default_14);  slice_720 = view_411 = full_default_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_28: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_14, arg329_1);  where_14 = arg329_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_28, [-1], True)
    sub_29: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_28, amax_14);  div_28 = amax_14 = None
    exp_14: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
    sum_15: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_29: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_118: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_29, [1, 16, 128, 128]);  div_29 = None
    view_412: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_118, [16, 128, 128]);  expand_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_398: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_113, [128, 4096])
    permute_156: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    mm_58: "f32[128, 4096]" = torch.ops.aten.mm.default(view_398, permute_156);  view_398 = permute_156 = None
    view_399: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_58, [1, 128, 4096]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_402: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_399, [1, 128, 16, 256]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_157: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_119: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_157, [1, 16, 128, 256])
    view_413: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_119, [16, 128, 256]);  expand_119 = None
    bmm_29: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_412, view_413);  view_412 = view_413 = None
    view_414: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_29, [1, 16, 128, 256]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_161: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
    clone_118: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_415: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_118, [1, 128, 4096]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_416: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_415, [128, 4096]);  view_415 = None
    permute_162: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    mm_59: "f32[128, 4096]" = torch.ops.aten.mm.default(view_416, permute_162);  view_416 = permute_162 = None
    view_417: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_59, [1, 128, 4096]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_418: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_113, [128, 4096]);  add_113 = None
    permute_163: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    addmm_28: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg148_1, view_418, permute_163);  arg148_1 = view_418 = permute_163 = None
    view_419: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 16384]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_146: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_419, 0.5)
    pow_15: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_419, 3.0)
    mul_147: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
    add_116: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_419, mul_147);  view_419 = mul_147 = None
    mul_148: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_116, 0.7978845608028654);  add_116 = None
    tanh_14: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_148);  mul_148 = None
    add_117: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
    mul_149: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_146, add_117);  mul_146 = add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_420: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_149, [128, 16384]);  mul_149 = None
    permute_164: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    addmm_29: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg150_1, view_420, permute_164);  arg150_1 = view_420 = permute_164 = None
    view_421: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_29, [1, 128, 4096]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_118: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_417, view_421);  view_417 = view_421 = None
    add_119: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_118, add_111);  add_118 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_61: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_30: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg330_1, [1, 1, 1]);  arg330_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_196: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_31: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_196, [1, 1, 64]);  unsqueeze_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_15: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_30, 1, repeat_31);  repeat_30 = repeat_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_15 = torch.ops.aten.split_with_sizes.default(gather_15, [32, 32], 2);  gather_15 = None
    getitem_62: "f32[1, 128, 32]" = split_with_sizes_15[0]
    getitem_63: "f32[1, 128, 32]" = split_with_sizes_15[1];  split_with_sizes_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant15: "f32[]" = self._tensor_constant15
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_767: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg331_1, 2, 0, 128);  arg331_1 = None
    slice_768: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_767, 3, 0, 128);  slice_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_30: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_119, getitem_61);  getitem_61 = None
    add_120: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    mul_150: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_15);  sub_30 = rsqrt_15 = None
    mul_151: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_150, arg151_1);  mul_150 = arg151_1 = None
    add_121: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_151, arg152_1);  mul_151 = arg152_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_422: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_121, [128, 4096])
    permute_165: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    mm_60: "f32[128, 4096]" = torch.ops.aten.mm.default(view_422, permute_165);  view_422 = permute_165 = None
    view_423: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_60, [1, 128, 4096]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_428: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_423, [1, 128, 16, 256]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_732: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_428, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_205: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_63, 2)
    unsqueeze_206: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 4);  unsqueeze_205 = None
    expand_123: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_206, [1, 128, 1, 32, 2]);  unsqueeze_206 = None
    clone_124: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
    view_435: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_124, [1, 128, 1, 64]);  clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_154: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_732, view_435);  view_435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_764: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_732, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_31: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_764);  slice_764 = None
    unsqueeze_207: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_31, 4);  neg_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_760: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_732, 3, 0, 9223372036854775807, 2);  slice_732 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_208: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_760, 4);  slice_760 = None
    cat_61: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_207, unsqueeze_208], 4);  unsqueeze_207 = unsqueeze_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_436: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_61, [1, 128, 16, 64]);  cat_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_203: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_62, 2)
    unsqueeze_204: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 4);  unsqueeze_203 = None
    expand_122: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_204, [1, 128, 1, 32, 2]);  unsqueeze_204 = None
    clone_123: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
    view_434: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_123, [1, 128, 1, 64]);  clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_155: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_436, view_434);  view_436 = view_434 = None
    add_123: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_154, mul_155);  mul_154 = mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_736: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_428, 3, 64, 9223372036854775807);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_63: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_123, slice_736], 3);  add_123 = slice_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_170: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_63, [0, 2, 1, 3]);  cat_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_124: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_170, [1, 16, 128, 256]);  permute_170 = None
    view_437: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_124, [16, 128, 256]);  expand_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_424: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_121, [128, 4096])
    permute_166: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    mm_61: "f32[128, 4096]" = torch.ops.aten.mm.default(view_424, permute_166);  view_424 = permute_166 = None
    view_425: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_61, [1, 128, 4096]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_429: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_425, [1, 128, 16, 256]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_724: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_429, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_199: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_63, 2);  getitem_63 = None
    unsqueeze_200: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 4);  unsqueeze_199 = None
    expand_121: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_200, [1, 128, 1, 32, 2]);  unsqueeze_200 = None
    clone_122: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_121, memory_format = torch.contiguous_format);  expand_121 = None
    view_432: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_122, [1, 128, 1, 64]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_152: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_724, view_432);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_750: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_724, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_30: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_750);  slice_750 = None
    unsqueeze_201: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_30, 4);  neg_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_746: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_724, 3, 0, 9223372036854775807, 2);  slice_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_202: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_746, 4);  slice_746 = None
    cat_60: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_201, unsqueeze_202], 4);  unsqueeze_201 = unsqueeze_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_433: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_60, [1, 128, 16, 64]);  cat_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_197: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_62, 2);  getitem_62 = None
    unsqueeze_198: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 4);  unsqueeze_197 = None
    expand_120: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_198, [1, 128, 1, 32, 2]);  unsqueeze_198 = None
    clone_121: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
    view_431: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_121, [1, 128, 1, 64]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_153: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_433, view_431);  view_433 = view_431 = None
    add_122: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_728: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_429, 3, 64, 9223372036854775807);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_62: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_122, slice_728], 3);  add_122 = slice_728 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_169: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_62, [0, 2, 1, 3]);  cat_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_171: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_169, [0, 1, 3, 2])
    expand_125: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_171, [1, 16, 256, 128]);  permute_171 = None
    view_438: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_125, [16, 256, 128]);  expand_125 = None
    bmm_30: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_437, view_438);  view_437 = view_438 = None
    view_439: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_30, [1, 16, 128, 128]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_15: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_15: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_768, view_439, full_default_15);  slice_768 = view_439 = full_default_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_30: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_15, arg332_1);  where_15 = arg332_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_15: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_30, [-1], True)
    sub_31: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_30, amax_15);  div_30 = amax_15 = None
    exp_15: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_16: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_31: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_126: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_31, [1, 16, 128, 128]);  div_31 = None
    view_440: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_126, [16, 128, 128]);  expand_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_426: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_121, [128, 4096])
    permute_167: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    mm_62: "f32[128, 4096]" = torch.ops.aten.mm.default(view_426, permute_167);  view_426 = permute_167 = None
    view_427: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_62, [1, 128, 4096]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_430: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_427, [1, 128, 16, 256]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_168: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_127: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_168, [1, 16, 128, 256])
    view_441: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_127, [16, 128, 256]);  expand_127 = None
    bmm_31: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_440, view_441);  view_440 = view_441 = None
    view_442: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_31, [1, 16, 128, 256]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_172: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    clone_126: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_443: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_126, [1, 128, 4096]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_444: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_443, [128, 4096]);  view_443 = None
    permute_173: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    mm_63: "f32[128, 4096]" = torch.ops.aten.mm.default(view_444, permute_173);  view_444 = permute_173 = None
    view_445: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_63, [1, 128, 4096]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_446: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_121, [128, 4096]);  add_121 = None
    permute_174: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    addmm_30: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg158_1, view_446, permute_174);  arg158_1 = view_446 = permute_174 = None
    view_447: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 16384]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_156: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_447, 0.5)
    pow_16: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_447, 3.0)
    mul_157: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
    add_124: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_447, mul_157);  view_447 = mul_157 = None
    mul_158: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_124, 0.7978845608028654);  add_124 = None
    tanh_15: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
    add_125: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
    mul_159: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_156, add_125);  mul_156 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_448: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_159, [128, 16384]);  mul_159 = None
    permute_175: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    addmm_31: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg160_1, view_448, permute_175);  arg160_1 = view_448 = permute_175 = None
    view_449: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_31, [1, 128, 4096]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_126: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_445, view_449);  view_445 = view_449 = None
    add_127: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_126, add_119);  add_126 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_127, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_65: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_32: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg333_1, [1, 1, 1]);  arg333_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_209: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_33: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_209, [1, 1, 64]);  unsqueeze_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_16: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_32, 1, repeat_33);  repeat_32 = repeat_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_16 = torch.ops.aten.split_with_sizes.default(gather_16, [32, 32], 2);  gather_16 = None
    getitem_66: "f32[1, 128, 32]" = split_with_sizes_16[0]
    getitem_67: "f32[1, 128, 32]" = split_with_sizes_16[1];  split_with_sizes_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant16: "f32[]" = self._tensor_constant16
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_815: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg334_1, 2, 0, 128);  arg334_1 = None
    slice_816: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_815, 3, 0, 128);  slice_815 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_32: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_127, getitem_65);  getitem_65 = None
    add_128: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    mul_160: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_16);  sub_32 = rsqrt_16 = None
    mul_161: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_160, arg161_1);  mul_160 = arg161_1 = None
    add_129: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_161, arg162_1);  mul_161 = arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_450: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_129, [128, 4096])
    permute_176: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    mm_64: "f32[128, 4096]" = torch.ops.aten.mm.default(view_450, permute_176);  view_450 = permute_176 = None
    view_451: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_64, [1, 128, 4096]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_456: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_451, [1, 128, 16, 256]);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_780: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_456, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_218: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_67, 2)
    unsqueeze_219: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 4);  unsqueeze_218 = None
    expand_131: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_219, [1, 128, 1, 32, 2]);  unsqueeze_219 = None
    clone_132: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_131, memory_format = torch.contiguous_format);  expand_131 = None
    view_463: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_132, [1, 128, 1, 64]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_164: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_780, view_463);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_812: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_780, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_33: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_812);  slice_812 = None
    unsqueeze_220: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_33, 4);  neg_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_808: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_780, 3, 0, 9223372036854775807, 2);  slice_780 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_221: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_808, 4);  slice_808 = None
    cat_65: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_220, unsqueeze_221], 4);  unsqueeze_220 = unsqueeze_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_464: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_65, [1, 128, 16, 64]);  cat_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_216: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_66, 2)
    unsqueeze_217: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 4);  unsqueeze_216 = None
    expand_130: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_217, [1, 128, 1, 32, 2]);  unsqueeze_217 = None
    clone_131: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
    view_462: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_131, [1, 128, 1, 64]);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_165: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_464, view_462);  view_464 = view_462 = None
    add_131: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_164, mul_165);  mul_164 = mul_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_784: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_456, 3, 64, 9223372036854775807);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_67: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_131, slice_784], 3);  add_131 = slice_784 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_181: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_67, [0, 2, 1, 3]);  cat_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_132: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_181, [1, 16, 128, 256]);  permute_181 = None
    view_465: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_132, [16, 128, 256]);  expand_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_452: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_129, [128, 4096])
    permute_177: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    mm_65: "f32[128, 4096]" = torch.ops.aten.mm.default(view_452, permute_177);  view_452 = permute_177 = None
    view_453: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_65, [1, 128, 4096]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_457: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_453, [1, 128, 16, 256]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_772: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_457, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_212: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_67, 2);  getitem_67 = None
    unsqueeze_213: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 4);  unsqueeze_212 = None
    expand_129: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_213, [1, 128, 1, 32, 2]);  unsqueeze_213 = None
    clone_130: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_129, memory_format = torch.contiguous_format);  expand_129 = None
    view_460: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_130, [1, 128, 1, 64]);  clone_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_162: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_772, view_460);  view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_798: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_772, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_32: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_798);  slice_798 = None
    unsqueeze_214: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_32, 4);  neg_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_794: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_772, 3, 0, 9223372036854775807, 2);  slice_772 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_215: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_794, 4);  slice_794 = None
    cat_64: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_214, unsqueeze_215], 4);  unsqueeze_214 = unsqueeze_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_461: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_64, [1, 128, 16, 64]);  cat_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_210: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_66, 2);  getitem_66 = None
    unsqueeze_211: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 4);  unsqueeze_210 = None
    expand_128: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_211, [1, 128, 1, 32, 2]);  unsqueeze_211 = None
    clone_129: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
    view_459: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_129, [1, 128, 1, 64]);  clone_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_163: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_461, view_459);  view_461 = view_459 = None
    add_130: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_162, mul_163);  mul_162 = mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_776: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_457, 3, 64, 9223372036854775807);  view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_66: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_130, slice_776], 3);  add_130 = slice_776 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_180: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_66, [0, 2, 1, 3]);  cat_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_182: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2])
    expand_133: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_182, [1, 16, 256, 128]);  permute_182 = None
    view_466: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_133, [16, 256, 128]);  expand_133 = None
    bmm_32: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_465, view_466);  view_465 = view_466 = None
    view_467: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [1, 16, 128, 128]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_16: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_16: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_816, view_467, full_default_16);  slice_816 = view_467 = full_default_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_32: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_16, arg335_1);  where_16 = arg335_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_32, [-1], True)
    sub_33: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_32, amax_16);  div_32 = amax_16 = None
    exp_16: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
    sum_17: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_33: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_134: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_33, [1, 16, 128, 128]);  div_33 = None
    view_468: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_134, [16, 128, 128]);  expand_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_454: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_129, [128, 4096])
    permute_178: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    mm_66: "f32[128, 4096]" = torch.ops.aten.mm.default(view_454, permute_178);  view_454 = permute_178 = None
    view_455: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_66, [1, 128, 4096]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_458: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_455, [1, 128, 16, 256]);  view_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_179: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_135: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_179, [1, 16, 128, 256])
    view_469: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_135, [16, 128, 256]);  expand_135 = None
    bmm_33: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_468, view_469);  view_468 = view_469 = None
    view_470: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_33, [1, 16, 128, 256]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_183: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
    clone_134: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_471: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_134, [1, 128, 4096]);  clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_472: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_471, [128, 4096]);  view_471 = None
    permute_184: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    mm_67: "f32[128, 4096]" = torch.ops.aten.mm.default(view_472, permute_184);  view_472 = permute_184 = None
    view_473: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_67, [1, 128, 4096]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_474: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_129, [128, 4096]);  add_129 = None
    permute_185: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_32: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg168_1, view_474, permute_185);  arg168_1 = view_474 = permute_185 = None
    view_475: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 16384]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_166: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_475, 0.5)
    pow_17: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_475, 3.0)
    mul_167: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
    add_132: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_475, mul_167);  view_475 = mul_167 = None
    mul_168: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_132, 0.7978845608028654);  add_132 = None
    tanh_16: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_168);  mul_168 = None
    add_133: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
    mul_169: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_166, add_133);  mul_166 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_476: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_169, [128, 16384]);  mul_169 = None
    permute_186: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_33: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg170_1, view_476, permute_186);  arg170_1 = view_476 = permute_186 = None
    view_477: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_33, [1, 128, 4096]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_134: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_473, view_477);  view_473 = view_477 = None
    add_135: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_134, add_127);  add_134 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_135, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1]" = var_mean_17[0]
    getitem_69: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_34: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg336_1, [1, 1, 1]);  arg336_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_222: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_35: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_222, [1, 1, 64]);  unsqueeze_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_17: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_34, 1, repeat_35);  repeat_34 = repeat_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_17 = torch.ops.aten.split_with_sizes.default(gather_17, [32, 32], 2);  gather_17 = None
    getitem_70: "f32[1, 128, 32]" = split_with_sizes_17[0]
    getitem_71: "f32[1, 128, 32]" = split_with_sizes_17[1];  split_with_sizes_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant17: "f32[]" = self._tensor_constant17
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_863: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg337_1, 2, 0, 128);  arg337_1 = None
    slice_864: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_863, 3, 0, 128);  slice_863 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_34: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_135, getitem_69);  getitem_69 = None
    add_136: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_136);  add_136 = None
    mul_170: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_17);  sub_34 = rsqrt_17 = None
    mul_171: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_170, arg171_1);  mul_170 = arg171_1 = None
    add_137: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_171, arg172_1);  mul_171 = arg172_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_478: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_137, [128, 4096])
    permute_187: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    mm_68: "f32[128, 4096]" = torch.ops.aten.mm.default(view_478, permute_187);  view_478 = permute_187 = None
    view_479: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_68, [1, 128, 4096]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_484: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_479, [1, 128, 16, 256]);  view_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_828: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_484, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_231: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_71, 2)
    unsqueeze_232: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 4);  unsqueeze_231 = None
    expand_139: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_232, [1, 128, 1, 32, 2]);  unsqueeze_232 = None
    clone_140: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_139, memory_format = torch.contiguous_format);  expand_139 = None
    view_491: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_140, [1, 128, 1, 64]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_174: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_828, view_491);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_860: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_828, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_35: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_860);  slice_860 = None
    unsqueeze_233: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_35, 4);  neg_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_856: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_828, 3, 0, 9223372036854775807, 2);  slice_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_234: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_856, 4);  slice_856 = None
    cat_69: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_233, unsqueeze_234], 4);  unsqueeze_233 = unsqueeze_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_492: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_69, [1, 128, 16, 64]);  cat_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_229: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_70, 2)
    unsqueeze_230: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 4);  unsqueeze_229 = None
    expand_138: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_230, [1, 128, 1, 32, 2]);  unsqueeze_230 = None
    clone_139: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_138, memory_format = torch.contiguous_format);  expand_138 = None
    view_490: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_139, [1, 128, 1, 64]);  clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_175: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_492, view_490);  view_492 = view_490 = None
    add_139: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_174, mul_175);  mul_174 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_832: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_484, 3, 64, 9223372036854775807);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_71: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_139, slice_832], 3);  add_139 = slice_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_192: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_71, [0, 2, 1, 3]);  cat_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_140: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_192, [1, 16, 128, 256]);  permute_192 = None
    view_493: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_140, [16, 128, 256]);  expand_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_480: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_137, [128, 4096])
    permute_188: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    mm_69: "f32[128, 4096]" = torch.ops.aten.mm.default(view_480, permute_188);  view_480 = permute_188 = None
    view_481: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_69, [1, 128, 4096]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_485: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_481, [1, 128, 16, 256]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_820: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_485, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_225: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_71, 2);  getitem_71 = None
    unsqueeze_226: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_225, 4);  unsqueeze_225 = None
    expand_137: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_226, [1, 128, 1, 32, 2]);  unsqueeze_226 = None
    clone_138: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_137, memory_format = torch.contiguous_format);  expand_137 = None
    view_488: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_138, [1, 128, 1, 64]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_172: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_820, view_488);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_846: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_820, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_34: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_846);  slice_846 = None
    unsqueeze_227: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_34, 4);  neg_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_842: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_820, 3, 0, 9223372036854775807, 2);  slice_820 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_228: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_842, 4);  slice_842 = None
    cat_68: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_227, unsqueeze_228], 4);  unsqueeze_227 = unsqueeze_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_489: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_68, [1, 128, 16, 64]);  cat_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_223: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_70, 2);  getitem_70 = None
    unsqueeze_224: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 4);  unsqueeze_223 = None
    expand_136: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_224, [1, 128, 1, 32, 2]);  unsqueeze_224 = None
    clone_137: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_136, memory_format = torch.contiguous_format);  expand_136 = None
    view_487: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_137, [1, 128, 1, 64]);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_173: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_489, view_487);  view_489 = view_487 = None
    add_138: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_172, mul_173);  mul_172 = mul_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_824: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_485, 3, 64, 9223372036854775807);  view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_70: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_138, slice_824], 3);  add_138 = slice_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_191: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_70, [0, 2, 1, 3]);  cat_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_193: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_191, [0, 1, 3, 2])
    expand_141: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_193, [1, 16, 256, 128]);  permute_193 = None
    view_494: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_141, [16, 256, 128]);  expand_141 = None
    bmm_34: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_493, view_494);  view_493 = view_494 = None
    view_495: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_34, [1, 16, 128, 128]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_17: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_17: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_864, view_495, full_default_17);  slice_864 = view_495 = full_default_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_34: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_17, arg338_1);  where_17 = arg338_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_17: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_34, [-1], True)
    sub_35: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_34, amax_17);  div_34 = amax_17 = None
    exp_17: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_18: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_35: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_142: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_35, [1, 16, 128, 128]);  div_35 = None
    view_496: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_142, [16, 128, 128]);  expand_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_482: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_137, [128, 4096])
    permute_189: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    mm_70: "f32[128, 4096]" = torch.ops.aten.mm.default(view_482, permute_189);  view_482 = permute_189 = None
    view_483: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_70, [1, 128, 4096]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_486: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_483, [1, 128, 16, 256]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_190: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_143: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_190, [1, 16, 128, 256])
    view_497: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_143, [16, 128, 256]);  expand_143 = None
    bmm_35: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_496, view_497);  view_496 = view_497 = None
    view_498: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_35, [1, 16, 128, 256]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_194: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
    clone_142: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_499: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_142, [1, 128, 4096]);  clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_500: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_499, [128, 4096]);  view_499 = None
    permute_195: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    mm_71: "f32[128, 4096]" = torch.ops.aten.mm.default(view_500, permute_195);  view_500 = permute_195 = None
    view_501: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_71, [1, 128, 4096]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_502: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_137, [128, 4096]);  add_137 = None
    permute_196: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
    addmm_34: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg178_1, view_502, permute_196);  arg178_1 = view_502 = permute_196 = None
    view_503: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 16384]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_176: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_503, 0.5)
    pow_18: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_503, 3.0)
    mul_177: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
    add_140: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_503, mul_177);  view_503 = mul_177 = None
    mul_178: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_140, 0.7978845608028654);  add_140 = None
    tanh_17: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_178);  mul_178 = None
    add_141: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
    mul_179: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_176, add_141);  mul_176 = add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_504: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_179, [128, 16384]);  mul_179 = None
    permute_197: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    addmm_35: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg180_1, view_504, permute_197);  arg180_1 = view_504 = permute_197 = None
    view_505: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_35, [1, 128, 4096]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_142: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_501, view_505);  view_501 = view_505 = None
    add_143: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_142, add_135);  add_142 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1]" = var_mean_18[0]
    getitem_73: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_36: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg339_1, [1, 1, 1]);  arg339_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_235: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_37: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_235, [1, 1, 64]);  unsqueeze_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_18: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_36, 1, repeat_37);  repeat_36 = repeat_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_18 = torch.ops.aten.split_with_sizes.default(gather_18, [32, 32], 2);  gather_18 = None
    getitem_74: "f32[1, 128, 32]" = split_with_sizes_18[0]
    getitem_75: "f32[1, 128, 32]" = split_with_sizes_18[1];  split_with_sizes_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant18: "f32[]" = self._tensor_constant18
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_911: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg340_1, 2, 0, 128);  arg340_1 = None
    slice_912: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_911, 3, 0, 128);  slice_911 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_36: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_143, getitem_73);  getitem_73 = None
    add_144: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    mul_180: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_18);  sub_36 = rsqrt_18 = None
    mul_181: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_180, arg181_1);  mul_180 = arg181_1 = None
    add_145: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_181, arg182_1);  mul_181 = arg182_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_506: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_145, [128, 4096])
    permute_198: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    mm_72: "f32[128, 4096]" = torch.ops.aten.mm.default(view_506, permute_198);  view_506 = permute_198 = None
    view_507: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_72, [1, 128, 4096]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_512: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_507, [1, 128, 16, 256]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_876: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_512, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_244: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_75, 2)
    unsqueeze_245: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 4);  unsqueeze_244 = None
    expand_147: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_245, [1, 128, 1, 32, 2]);  unsqueeze_245 = None
    clone_148: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_147, memory_format = torch.contiguous_format);  expand_147 = None
    view_519: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_148, [1, 128, 1, 64]);  clone_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_184: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_876, view_519);  view_519 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_908: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_876, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_37: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_908);  slice_908 = None
    unsqueeze_246: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_37, 4);  neg_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_904: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_876, 3, 0, 9223372036854775807, 2);  slice_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_247: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_904, 4);  slice_904 = None
    cat_73: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_246, unsqueeze_247], 4);  unsqueeze_246 = unsqueeze_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_520: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_73, [1, 128, 16, 64]);  cat_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_242: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_74, 2)
    unsqueeze_243: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 4);  unsqueeze_242 = None
    expand_146: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_243, [1, 128, 1, 32, 2]);  unsqueeze_243 = None
    clone_147: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_146, memory_format = torch.contiguous_format);  expand_146 = None
    view_518: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_147, [1, 128, 1, 64]);  clone_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_185: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_520, view_518);  view_520 = view_518 = None
    add_147: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_184, mul_185);  mul_184 = mul_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_880: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_512, 3, 64, 9223372036854775807);  view_512 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_75: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_147, slice_880], 3);  add_147 = slice_880 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_203: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_75, [0, 2, 1, 3]);  cat_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_148: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_203, [1, 16, 128, 256]);  permute_203 = None
    view_521: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_148, [16, 128, 256]);  expand_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_508: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_145, [128, 4096])
    permute_199: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    mm_73: "f32[128, 4096]" = torch.ops.aten.mm.default(view_508, permute_199);  view_508 = permute_199 = None
    view_509: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_73, [1, 128, 4096]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_513: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_509, [1, 128, 16, 256]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_868: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_513, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_238: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_75, 2);  getitem_75 = None
    unsqueeze_239: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 4);  unsqueeze_238 = None
    expand_145: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_239, [1, 128, 1, 32, 2]);  unsqueeze_239 = None
    clone_146: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_145, memory_format = torch.contiguous_format);  expand_145 = None
    view_516: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_146, [1, 128, 1, 64]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_182: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_868, view_516);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_894: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_868, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_36: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_894);  slice_894 = None
    unsqueeze_240: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_36, 4);  neg_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_890: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_868, 3, 0, 9223372036854775807, 2);  slice_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_241: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_890, 4);  slice_890 = None
    cat_72: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_240, unsqueeze_241], 4);  unsqueeze_240 = unsqueeze_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_517: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_72, [1, 128, 16, 64]);  cat_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_236: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_74, 2);  getitem_74 = None
    unsqueeze_237: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 4);  unsqueeze_236 = None
    expand_144: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_237, [1, 128, 1, 32, 2]);  unsqueeze_237 = None
    clone_145: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_144, memory_format = torch.contiguous_format);  expand_144 = None
    view_515: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_145, [1, 128, 1, 64]);  clone_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_183: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_517, view_515);  view_517 = view_515 = None
    add_146: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_182, mul_183);  mul_182 = mul_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_872: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_513, 3, 64, 9223372036854775807);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_74: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_146, slice_872], 3);  add_146 = slice_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_202: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_74, [0, 2, 1, 3]);  cat_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_204: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_202, [0, 1, 3, 2])
    expand_149: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_204, [1, 16, 256, 128]);  permute_204 = None
    view_522: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_149, [16, 256, 128]);  expand_149 = None
    bmm_36: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_521, view_522);  view_521 = view_522 = None
    view_523: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [1, 16, 128, 128]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_18: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_18: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_912, view_523, full_default_18);  slice_912 = view_523 = full_default_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_36: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_18, arg341_1);  where_18 = arg341_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_18: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_36, [-1], True)
    sub_37: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_36, amax_18);  div_36 = amax_18 = None
    exp_18: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_19: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_37: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_150: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_37, [1, 16, 128, 128]);  div_37 = None
    view_524: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_150, [16, 128, 128]);  expand_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_510: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_145, [128, 4096])
    permute_200: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    mm_74: "f32[128, 4096]" = torch.ops.aten.mm.default(view_510, permute_200);  view_510 = permute_200 = None
    view_511: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_74, [1, 128, 4096]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_514: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_511, [1, 128, 16, 256]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_201: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_151: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_201, [1, 16, 128, 256])
    view_525: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_151, [16, 128, 256]);  expand_151 = None
    bmm_37: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_524, view_525);  view_524 = view_525 = None
    view_526: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_37, [1, 16, 128, 256]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_205: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None
    clone_150: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_527: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_150, [1, 128, 4096]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_528: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_527, [128, 4096]);  view_527 = None
    permute_206: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    mm_75: "f32[128, 4096]" = torch.ops.aten.mm.default(view_528, permute_206);  view_528 = permute_206 = None
    view_529: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_75, [1, 128, 4096]);  mm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_530: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_145, [128, 4096]);  add_145 = None
    permute_207: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    addmm_36: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg188_1, view_530, permute_207);  arg188_1 = view_530 = permute_207 = None
    view_531: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 16384]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_186: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_531, 0.5)
    pow_19: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_531, 3.0)
    mul_187: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
    add_148: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_531, mul_187);  view_531 = mul_187 = None
    mul_188: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_148, 0.7978845608028654);  add_148 = None
    tanh_18: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_188);  mul_188 = None
    add_149: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
    mul_189: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_186, add_149);  mul_186 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_532: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_189, [128, 16384]);  mul_189 = None
    permute_208: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_37: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg190_1, view_532, permute_208);  arg190_1 = view_532 = permute_208 = None
    view_533: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_37, [1, 128, 4096]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_150: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_529, view_533);  view_529 = view_533 = None
    add_151: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_150, add_143);  add_150 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_151, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1]" = var_mean_19[0]
    getitem_77: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_38: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg342_1, [1, 1, 1]);  arg342_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_248: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_39: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_248, [1, 1, 64]);  unsqueeze_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_19: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_38, 1, repeat_39);  repeat_38 = repeat_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_19 = torch.ops.aten.split_with_sizes.default(gather_19, [32, 32], 2);  gather_19 = None
    getitem_78: "f32[1, 128, 32]" = split_with_sizes_19[0]
    getitem_79: "f32[1, 128, 32]" = split_with_sizes_19[1];  split_with_sizes_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant19: "f32[]" = self._tensor_constant19
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_959: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg343_1, 2, 0, 128);  arg343_1 = None
    slice_960: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_959, 3, 0, 128);  slice_959 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_38: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_151, getitem_77);  getitem_77 = None
    add_152: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
    mul_190: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_19);  sub_38 = rsqrt_19 = None
    mul_191: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_190, arg191_1);  mul_190 = arg191_1 = None
    add_153: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_191, arg192_1);  mul_191 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_534: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_153, [128, 4096])
    permute_209: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    mm_76: "f32[128, 4096]" = torch.ops.aten.mm.default(view_534, permute_209);  view_534 = permute_209 = None
    view_535: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_76, [1, 128, 4096]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_540: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_535, [1, 128, 16, 256]);  view_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_924: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_540, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_257: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_79, 2)
    unsqueeze_258: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 4);  unsqueeze_257 = None
    expand_155: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_258, [1, 128, 1, 32, 2]);  unsqueeze_258 = None
    clone_156: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_155, memory_format = torch.contiguous_format);  expand_155 = None
    view_547: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_156, [1, 128, 1, 64]);  clone_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_194: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_924, view_547);  view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_956: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_924, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_39: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_956);  slice_956 = None
    unsqueeze_259: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_39, 4);  neg_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_952: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_924, 3, 0, 9223372036854775807, 2);  slice_924 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_260: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_952, 4);  slice_952 = None
    cat_77: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_259, unsqueeze_260], 4);  unsqueeze_259 = unsqueeze_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_548: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_77, [1, 128, 16, 64]);  cat_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_255: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_78, 2)
    unsqueeze_256: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 4);  unsqueeze_255 = None
    expand_154: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_256, [1, 128, 1, 32, 2]);  unsqueeze_256 = None
    clone_155: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_154, memory_format = torch.contiguous_format);  expand_154 = None
    view_546: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_155, [1, 128, 1, 64]);  clone_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_195: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_548, view_546);  view_548 = view_546 = None
    add_155: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_194, mul_195);  mul_194 = mul_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_928: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_540, 3, 64, 9223372036854775807);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_79: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_155, slice_928], 3);  add_155 = slice_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_214: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_79, [0, 2, 1, 3]);  cat_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_156: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_214, [1, 16, 128, 256]);  permute_214 = None
    view_549: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_156, [16, 128, 256]);  expand_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_536: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_153, [128, 4096])
    permute_210: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg194_1, [1, 0]);  arg194_1 = None
    mm_77: "f32[128, 4096]" = torch.ops.aten.mm.default(view_536, permute_210);  view_536 = permute_210 = None
    view_537: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_77, [1, 128, 4096]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_541: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_537, [1, 128, 16, 256]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_916: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_541, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_251: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_79, 2);  getitem_79 = None
    unsqueeze_252: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 4);  unsqueeze_251 = None
    expand_153: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_252, [1, 128, 1, 32, 2]);  unsqueeze_252 = None
    clone_154: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_153, memory_format = torch.contiguous_format);  expand_153 = None
    view_544: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_154, [1, 128, 1, 64]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_192: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_916, view_544);  view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_942: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_916, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_38: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_942);  slice_942 = None
    unsqueeze_253: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_38, 4);  neg_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_938: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_916, 3, 0, 9223372036854775807, 2);  slice_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_254: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_938, 4);  slice_938 = None
    cat_76: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_253, unsqueeze_254], 4);  unsqueeze_253 = unsqueeze_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_545: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_76, [1, 128, 16, 64]);  cat_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_249: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_78, 2);  getitem_78 = None
    unsqueeze_250: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 4);  unsqueeze_249 = None
    expand_152: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_250, [1, 128, 1, 32, 2]);  unsqueeze_250 = None
    clone_153: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_152, memory_format = torch.contiguous_format);  expand_152 = None
    view_543: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_153, [1, 128, 1, 64]);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_193: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_545, view_543);  view_545 = view_543 = None
    add_154: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_192, mul_193);  mul_192 = mul_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_920: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_541, 3, 64, 9223372036854775807);  view_541 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_78: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_154, slice_920], 3);  add_154 = slice_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_213: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_78, [0, 2, 1, 3]);  cat_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_215: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_213, [0, 1, 3, 2])
    expand_157: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_215, [1, 16, 256, 128]);  permute_215 = None
    view_550: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_157, [16, 256, 128]);  expand_157 = None
    bmm_38: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_549, view_550);  view_549 = view_550 = None
    view_551: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_38, [1, 16, 128, 128]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_19: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_19: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_960, view_551, full_default_19);  slice_960 = view_551 = full_default_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_38: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_19, arg344_1);  where_19 = arg344_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_19: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_38, [-1], True)
    sub_39: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_38, amax_19);  div_38 = amax_19 = None
    exp_19: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_39);  sub_39 = None
    sum_20: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_39: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_158: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_39, [1, 16, 128, 128]);  div_39 = None
    view_552: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_158, [16, 128, 128]);  expand_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_538: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_153, [128, 4096])
    permute_211: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    mm_78: "f32[128, 4096]" = torch.ops.aten.mm.default(view_538, permute_211);  view_538 = permute_211 = None
    view_539: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_78, [1, 128, 4096]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_542: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_539, [1, 128, 16, 256]);  view_539 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_212: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_542, [0, 2, 1, 3]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_159: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_212, [1, 16, 128, 256])
    view_553: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_159, [16, 128, 256]);  expand_159 = None
    bmm_39: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_552, view_553);  view_552 = view_553 = None
    view_554: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_39, [1, 16, 128, 256]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_216: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
    clone_158: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_555: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_158, [1, 128, 4096]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_556: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_555, [128, 4096]);  view_555 = None
    permute_217: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
    mm_79: "f32[128, 4096]" = torch.ops.aten.mm.default(view_556, permute_217);  view_556 = permute_217 = None
    view_557: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_79, [1, 128, 4096]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_558: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_153, [128, 4096]);  add_153 = None
    permute_218: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    addmm_38: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg198_1, view_558, permute_218);  arg198_1 = view_558 = permute_218 = None
    view_559: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 16384]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_196: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_559, 0.5)
    pow_20: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_559, 3.0)
    mul_197: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
    add_156: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_559, mul_197);  view_559 = mul_197 = None
    mul_198: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_156, 0.7978845608028654);  add_156 = None
    tanh_19: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_198);  mul_198 = None
    add_157: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
    mul_199: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_196, add_157);  mul_196 = add_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_560: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_199, [128, 16384]);  mul_199 = None
    permute_219: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
    addmm_39: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg200_1, view_560, permute_219);  arg200_1 = view_560 = permute_219 = None
    view_561: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_39, [1, 128, 4096]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_158: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_557, view_561);  view_557 = view_561 = None
    add_159: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_158, add_151);  add_158 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_159, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1]" = var_mean_20[0]
    getitem_81: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_40: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg345_1, [1, 1, 1]);  arg345_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_261: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_41: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_261, [1, 1, 64]);  unsqueeze_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_20: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_40, 1, repeat_41);  repeat_40 = repeat_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_20 = torch.ops.aten.split_with_sizes.default(gather_20, [32, 32], 2);  gather_20 = None
    getitem_82: "f32[1, 128, 32]" = split_with_sizes_20[0]
    getitem_83: "f32[1, 128, 32]" = split_with_sizes_20[1];  split_with_sizes_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant20: "f32[]" = self._tensor_constant20
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1007: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg346_1, 2, 0, 128);  arg346_1 = None
    slice_1008: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1007, 3, 0, 128);  slice_1007 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_40: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_159, getitem_81);  getitem_81 = None
    add_160: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_160);  add_160 = None
    mul_200: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_20);  sub_40 = rsqrt_20 = None
    mul_201: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_200, arg201_1);  mul_200 = arg201_1 = None
    add_161: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_201, arg202_1);  mul_201 = arg202_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_562: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_161, [128, 4096])
    permute_220: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
    mm_80: "f32[128, 4096]" = torch.ops.aten.mm.default(view_562, permute_220);  view_562 = permute_220 = None
    view_563: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_80, [1, 128, 4096]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_568: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_563, [1, 128, 16, 256]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_972: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_568, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_270: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_83, 2)
    unsqueeze_271: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 4);  unsqueeze_270 = None
    expand_163: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_271, [1, 128, 1, 32, 2]);  unsqueeze_271 = None
    clone_164: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_163, memory_format = torch.contiguous_format);  expand_163 = None
    view_575: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_164, [1, 128, 1, 64]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_204: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_972, view_575);  view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1004: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_972, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_41: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1004);  slice_1004 = None
    unsqueeze_272: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_41, 4);  neg_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1000: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_972, 3, 0, 9223372036854775807, 2);  slice_972 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_273: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1000, 4);  slice_1000 = None
    cat_81: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_272, unsqueeze_273], 4);  unsqueeze_272 = unsqueeze_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_576: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_81, [1, 128, 16, 64]);  cat_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_268: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_82, 2)
    unsqueeze_269: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 4);  unsqueeze_268 = None
    expand_162: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_269, [1, 128, 1, 32, 2]);  unsqueeze_269 = None
    clone_163: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_162, memory_format = torch.contiguous_format);  expand_162 = None
    view_574: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_163, [1, 128, 1, 64]);  clone_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_205: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_576, view_574);  view_576 = view_574 = None
    add_163: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_204, mul_205);  mul_204 = mul_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_976: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_568, 3, 64, 9223372036854775807);  view_568 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_83: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_163, slice_976], 3);  add_163 = slice_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_225: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_83, [0, 2, 1, 3]);  cat_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_164: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_225, [1, 16, 128, 256]);  permute_225 = None
    view_577: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_164, [16, 128, 256]);  expand_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_564: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_161, [128, 4096])
    permute_221: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
    mm_81: "f32[128, 4096]" = torch.ops.aten.mm.default(view_564, permute_221);  view_564 = permute_221 = None
    view_565: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_81, [1, 128, 4096]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_569: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_565, [1, 128, 16, 256]);  view_565 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_964: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_569, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_264: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_83, 2);  getitem_83 = None
    unsqueeze_265: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 4);  unsqueeze_264 = None
    expand_161: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_265, [1, 128, 1, 32, 2]);  unsqueeze_265 = None
    clone_162: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_161, memory_format = torch.contiguous_format);  expand_161 = None
    view_572: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_162, [1, 128, 1, 64]);  clone_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_202: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_964, view_572);  view_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_990: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_964, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_40: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_990);  slice_990 = None
    unsqueeze_266: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_40, 4);  neg_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_986: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_964, 3, 0, 9223372036854775807, 2);  slice_964 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_267: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_986, 4);  slice_986 = None
    cat_80: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_266, unsqueeze_267], 4);  unsqueeze_266 = unsqueeze_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_573: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_80, [1, 128, 16, 64]);  cat_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_262: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_82, 2);  getitem_82 = None
    unsqueeze_263: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 4);  unsqueeze_262 = None
    expand_160: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_263, [1, 128, 1, 32, 2]);  unsqueeze_263 = None
    clone_161: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_160, memory_format = torch.contiguous_format);  expand_160 = None
    view_571: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_161, [1, 128, 1, 64]);  clone_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_203: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_573, view_571);  view_573 = view_571 = None
    add_162: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_202, mul_203);  mul_202 = mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_968: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_569, 3, 64, 9223372036854775807);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_82: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_162, slice_968], 3);  add_162 = slice_968 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_224: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_82, [0, 2, 1, 3]);  cat_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_226: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_224, [0, 1, 3, 2])
    expand_165: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_226, [1, 16, 256, 128]);  permute_226 = None
    view_578: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_165, [16, 256, 128]);  expand_165 = None
    bmm_40: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_577, view_578);  view_577 = view_578 = None
    view_579: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [1, 16, 128, 128]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_20: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_20: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1008, view_579, full_default_20);  slice_1008 = view_579 = full_default_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_40: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_20, arg347_1);  where_20 = arg347_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_20: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_40, [-1], True)
    sub_41: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_40, amax_20);  div_40 = amax_20 = None
    exp_20: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
    sum_21: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_41: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_166: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_41, [1, 16, 128, 128]);  div_41 = None
    view_580: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_166, [16, 128, 128]);  expand_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_566: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_161, [128, 4096])
    permute_222: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
    mm_82: "f32[128, 4096]" = torch.ops.aten.mm.default(view_566, permute_222);  view_566 = permute_222 = None
    view_567: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_82, [1, 128, 4096]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_570: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_567, [1, 128, 16, 256]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_223: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_167: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_223, [1, 16, 128, 256])
    view_581: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_167, [16, 128, 256]);  expand_167 = None
    bmm_41: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_580, view_581);  view_580 = view_581 = None
    view_582: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_41, [1, 16, 128, 256]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_227: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_582, [0, 2, 1, 3]);  view_582 = None
    clone_166: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_583: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_166, [1, 128, 4096]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_584: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_583, [128, 4096]);  view_583 = None
    permute_228: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
    mm_83: "f32[128, 4096]" = torch.ops.aten.mm.default(view_584, permute_228);  view_584 = permute_228 = None
    view_585: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_83, [1, 128, 4096]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_586: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_161, [128, 4096]);  add_161 = None
    permute_229: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
    addmm_40: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg208_1, view_586, permute_229);  arg208_1 = view_586 = permute_229 = None
    view_587: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 16384]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_206: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_587, 0.5)
    pow_21: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_587, 3.0)
    mul_207: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
    add_164: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_587, mul_207);  view_587 = mul_207 = None
    mul_208: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_164, 0.7978845608028654);  add_164 = None
    tanh_20: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_208);  mul_208 = None
    add_165: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
    mul_209: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_206, add_165);  mul_206 = add_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_588: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_209, [128, 16384]);  mul_209 = None
    permute_230: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
    addmm_41: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg210_1, view_588, permute_230);  arg210_1 = view_588 = permute_230 = None
    view_589: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_41, [1, 128, 4096]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_166: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_585, view_589);  view_585 = view_589 = None
    add_167: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_166, add_159);  add_166 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_167, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 128, 1]" = var_mean_21[0]
    getitem_85: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_42: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg348_1, [1, 1, 1]);  arg348_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_274: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_43: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_274, [1, 1, 64]);  unsqueeze_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_21: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_42, 1, repeat_43);  repeat_42 = repeat_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_21 = torch.ops.aten.split_with_sizes.default(gather_21, [32, 32], 2);  gather_21 = None
    getitem_86: "f32[1, 128, 32]" = split_with_sizes_21[0]
    getitem_87: "f32[1, 128, 32]" = split_with_sizes_21[1];  split_with_sizes_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant21: "f32[]" = self._tensor_constant21
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1055: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg349_1, 2, 0, 128);  arg349_1 = None
    slice_1056: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1055, 3, 0, 128);  slice_1055 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_42: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_167, getitem_85);  getitem_85 = None
    add_168: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    mul_210: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_21);  sub_42 = rsqrt_21 = None
    mul_211: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_210, arg211_1);  mul_210 = arg211_1 = None
    add_169: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_211, arg212_1);  mul_211 = arg212_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_590: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_169, [128, 4096])
    permute_231: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
    mm_84: "f32[128, 4096]" = torch.ops.aten.mm.default(view_590, permute_231);  view_590 = permute_231 = None
    view_591: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_84, [1, 128, 4096]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_596: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_591, [1, 128, 16, 256]);  view_591 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1020: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_596, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_283: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_87, 2)
    unsqueeze_284: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 4);  unsqueeze_283 = None
    expand_171: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_284, [1, 128, 1, 32, 2]);  unsqueeze_284 = None
    clone_172: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_171, memory_format = torch.contiguous_format);  expand_171 = None
    view_603: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_172, [1, 128, 1, 64]);  clone_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_214: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1020, view_603);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1052: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1020, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_43: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1052);  slice_1052 = None
    unsqueeze_285: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_43, 4);  neg_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1048: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1020, 3, 0, 9223372036854775807, 2);  slice_1020 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_286: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1048, 4);  slice_1048 = None
    cat_85: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_285, unsqueeze_286], 4);  unsqueeze_285 = unsqueeze_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_604: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_85, [1, 128, 16, 64]);  cat_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_281: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_86, 2)
    unsqueeze_282: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 4);  unsqueeze_281 = None
    expand_170: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_282, [1, 128, 1, 32, 2]);  unsqueeze_282 = None
    clone_171: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_170, memory_format = torch.contiguous_format);  expand_170 = None
    view_602: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_171, [1, 128, 1, 64]);  clone_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_215: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_604, view_602);  view_604 = view_602 = None
    add_171: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_214, mul_215);  mul_214 = mul_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1024: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_596, 3, 64, 9223372036854775807);  view_596 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_87: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_171, slice_1024], 3);  add_171 = slice_1024 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_236: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_87, [0, 2, 1, 3]);  cat_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_172: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_236, [1, 16, 128, 256]);  permute_236 = None
    view_605: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_172, [16, 128, 256]);  expand_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_592: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_169, [128, 4096])
    permute_232: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    mm_85: "f32[128, 4096]" = torch.ops.aten.mm.default(view_592, permute_232);  view_592 = permute_232 = None
    view_593: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_85, [1, 128, 4096]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_597: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_593, [1, 128, 16, 256]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1012: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_597, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_277: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_87, 2);  getitem_87 = None
    unsqueeze_278: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 4);  unsqueeze_277 = None
    expand_169: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_278, [1, 128, 1, 32, 2]);  unsqueeze_278 = None
    clone_170: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_169, memory_format = torch.contiguous_format);  expand_169 = None
    view_600: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_170, [1, 128, 1, 64]);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_212: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1012, view_600);  view_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1038: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1012, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_42: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1038);  slice_1038 = None
    unsqueeze_279: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_42, 4);  neg_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1034: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1012, 3, 0, 9223372036854775807, 2);  slice_1012 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_280: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1034, 4);  slice_1034 = None
    cat_84: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_279, unsqueeze_280], 4);  unsqueeze_279 = unsqueeze_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_601: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_84, [1, 128, 16, 64]);  cat_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_275: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_86, 2);  getitem_86 = None
    unsqueeze_276: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 4);  unsqueeze_275 = None
    expand_168: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_276, [1, 128, 1, 32, 2]);  unsqueeze_276 = None
    clone_169: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_168, memory_format = torch.contiguous_format);  expand_168 = None
    view_599: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_169, [1, 128, 1, 64]);  clone_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_213: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_601, view_599);  view_601 = view_599 = None
    add_170: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_212, mul_213);  mul_212 = mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1016: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_597, 3, 64, 9223372036854775807);  view_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_86: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_170, slice_1016], 3);  add_170 = slice_1016 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_235: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_86, [0, 2, 1, 3]);  cat_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_237: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_235, [0, 1, 3, 2])
    expand_173: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_237, [1, 16, 256, 128]);  permute_237 = None
    view_606: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_173, [16, 256, 128]);  expand_173 = None
    bmm_42: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_605, view_606);  view_605 = view_606 = None
    view_607: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_42, [1, 16, 128, 128]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_21: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_21: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1056, view_607, full_default_21);  slice_1056 = view_607 = full_default_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_42: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_21, arg350_1);  where_21 = arg350_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_21: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_42, [-1], True)
    sub_43: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_42, amax_21);  div_42 = amax_21 = None
    exp_21: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_22: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_43: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_174: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_43, [1, 16, 128, 128]);  div_43 = None
    view_608: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_174, [16, 128, 128]);  expand_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_594: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_169, [128, 4096])
    permute_233: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
    mm_86: "f32[128, 4096]" = torch.ops.aten.mm.default(view_594, permute_233);  view_594 = permute_233 = None
    view_595: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_86, [1, 128, 4096]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_598: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_595, [1, 128, 16, 256]);  view_595 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_234: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_598, [0, 2, 1, 3]);  view_598 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_175: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_234, [1, 16, 128, 256])
    view_609: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_175, [16, 128, 256]);  expand_175 = None
    bmm_43: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_608, view_609);  view_608 = view_609 = None
    view_610: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_43, [1, 16, 128, 256]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_238: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None
    clone_174: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_611: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_174, [1, 128, 4096]);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_612: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_611, [128, 4096]);  view_611 = None
    permute_239: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
    mm_87: "f32[128, 4096]" = torch.ops.aten.mm.default(view_612, permute_239);  view_612 = permute_239 = None
    view_613: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_87, [1, 128, 4096]);  mm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_614: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_169, [128, 4096]);  add_169 = None
    permute_240: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
    addmm_42: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg218_1, view_614, permute_240);  arg218_1 = view_614 = permute_240 = None
    view_615: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 16384]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_216: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_615, 0.5)
    pow_22: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_615, 3.0)
    mul_217: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
    add_172: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_615, mul_217);  view_615 = mul_217 = None
    mul_218: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_172, 0.7978845608028654);  add_172 = None
    tanh_21: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_218);  mul_218 = None
    add_173: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
    mul_219: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_216, add_173);  mul_216 = add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_616: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_219, [128, 16384]);  mul_219 = None
    permute_241: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
    addmm_43: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg220_1, view_616, permute_241);  arg220_1 = view_616 = permute_241 = None
    view_617: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_43, [1, 128, 4096]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_174: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_613, view_617);  view_613 = view_617 = None
    add_175: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_174, add_167);  add_174 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_175, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 128, 1]" = var_mean_22[0]
    getitem_89: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_44: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg351_1, [1, 1, 1]);  arg351_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_287: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_45: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_287, [1, 1, 64]);  unsqueeze_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_22: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_44, 1, repeat_45);  repeat_44 = repeat_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_22 = torch.ops.aten.split_with_sizes.default(gather_22, [32, 32], 2);  gather_22 = None
    getitem_90: "f32[1, 128, 32]" = split_with_sizes_22[0]
    getitem_91: "f32[1, 128, 32]" = split_with_sizes_22[1];  split_with_sizes_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant22: "f32[]" = self._tensor_constant22
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1103: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg352_1, 2, 0, 128);  arg352_1 = None
    slice_1104: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1103, 3, 0, 128);  slice_1103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_44: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_175, getitem_89);  getitem_89 = None
    add_176: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_176);  add_176 = None
    mul_220: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_22);  sub_44 = rsqrt_22 = None
    mul_221: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_220, arg221_1);  mul_220 = arg221_1 = None
    add_177: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_221, arg222_1);  mul_221 = arg222_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_618: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_177, [128, 4096])
    permute_242: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
    mm_88: "f32[128, 4096]" = torch.ops.aten.mm.default(view_618, permute_242);  view_618 = permute_242 = None
    view_619: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_88, [1, 128, 4096]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_624: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_619, [1, 128, 16, 256]);  view_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1068: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_624, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_296: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_91, 2)
    unsqueeze_297: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 4);  unsqueeze_296 = None
    expand_179: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_297, [1, 128, 1, 32, 2]);  unsqueeze_297 = None
    clone_180: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_179, memory_format = torch.contiguous_format);  expand_179 = None
    view_631: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_180, [1, 128, 1, 64]);  clone_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_224: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1068, view_631);  view_631 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1100: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1068, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_45: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1100);  slice_1100 = None
    unsqueeze_298: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_45, 4);  neg_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1096: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1068, 3, 0, 9223372036854775807, 2);  slice_1068 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_299: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1096, 4);  slice_1096 = None
    cat_89: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_298, unsqueeze_299], 4);  unsqueeze_298 = unsqueeze_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_632: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_89, [1, 128, 16, 64]);  cat_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_294: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_90, 2)
    unsqueeze_295: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 4);  unsqueeze_294 = None
    expand_178: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_295, [1, 128, 1, 32, 2]);  unsqueeze_295 = None
    clone_179: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_178, memory_format = torch.contiguous_format);  expand_178 = None
    view_630: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_179, [1, 128, 1, 64]);  clone_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_225: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_632, view_630);  view_632 = view_630 = None
    add_179: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1072: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_624, 3, 64, 9223372036854775807);  view_624 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_91: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_179, slice_1072], 3);  add_179 = slice_1072 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_247: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_91, [0, 2, 1, 3]);  cat_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_180: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_247, [1, 16, 128, 256]);  permute_247 = None
    view_633: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_180, [16, 128, 256]);  expand_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_620: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_177, [128, 4096])
    permute_243: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg224_1, [1, 0]);  arg224_1 = None
    mm_89: "f32[128, 4096]" = torch.ops.aten.mm.default(view_620, permute_243);  view_620 = permute_243 = None
    view_621: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_89, [1, 128, 4096]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_625: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_621, [1, 128, 16, 256]);  view_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1060: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_625, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_290: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_91, 2);  getitem_91 = None
    unsqueeze_291: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 4);  unsqueeze_290 = None
    expand_177: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_291, [1, 128, 1, 32, 2]);  unsqueeze_291 = None
    clone_178: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_177, memory_format = torch.contiguous_format);  expand_177 = None
    view_628: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_178, [1, 128, 1, 64]);  clone_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_222: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1060, view_628);  view_628 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1086: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1060, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_44: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1086);  slice_1086 = None
    unsqueeze_292: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_44, 4);  neg_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1082: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1060, 3, 0, 9223372036854775807, 2);  slice_1060 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_293: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1082, 4);  slice_1082 = None
    cat_88: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_292, unsqueeze_293], 4);  unsqueeze_292 = unsqueeze_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_629: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_88, [1, 128, 16, 64]);  cat_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_288: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_90, 2);  getitem_90 = None
    unsqueeze_289: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 4);  unsqueeze_288 = None
    expand_176: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_289, [1, 128, 1, 32, 2]);  unsqueeze_289 = None
    clone_177: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_176, memory_format = torch.contiguous_format);  expand_176 = None
    view_627: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_177, [1, 128, 1, 64]);  clone_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_223: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_629, view_627);  view_629 = view_627 = None
    add_178: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1064: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_625, 3, 64, 9223372036854775807);  view_625 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_90: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_178, slice_1064], 3);  add_178 = slice_1064 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_246: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_90, [0, 2, 1, 3]);  cat_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_248: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_246, [0, 1, 3, 2])
    expand_181: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_248, [1, 16, 256, 128]);  permute_248 = None
    view_634: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_181, [16, 256, 128]);  expand_181 = None
    bmm_44: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_633, view_634);  view_633 = view_634 = None
    view_635: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [1, 16, 128, 128]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_22: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_22: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1104, view_635, full_default_22);  slice_1104 = view_635 = full_default_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_44: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_22, arg353_1);  where_22 = arg353_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_22: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_44, [-1], True)
    sub_45: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_44, amax_22);  div_44 = amax_22 = None
    exp_22: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_23: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_45: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_182: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_45, [1, 16, 128, 128]);  div_45 = None
    view_636: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_182, [16, 128, 128]);  expand_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_622: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_177, [128, 4096])
    permute_244: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
    mm_90: "f32[128, 4096]" = torch.ops.aten.mm.default(view_622, permute_244);  view_622 = permute_244 = None
    view_623: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_90, [1, 128, 4096]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_626: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_623, [1, 128, 16, 256]);  view_623 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_245: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_626, [0, 2, 1, 3]);  view_626 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_183: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_245, [1, 16, 128, 256])
    view_637: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_183, [16, 128, 256]);  expand_183 = None
    bmm_45: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_636, view_637);  view_636 = view_637 = None
    view_638: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_45, [1, 16, 128, 256]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_249: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_638, [0, 2, 1, 3]);  view_638 = None
    clone_182: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_639: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_182, [1, 128, 4096]);  clone_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_640: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_639, [128, 4096]);  view_639 = None
    permute_250: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
    mm_91: "f32[128, 4096]" = torch.ops.aten.mm.default(view_640, permute_250);  view_640 = permute_250 = None
    view_641: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_91, [1, 128, 4096]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_642: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_177, [128, 4096]);  add_177 = None
    permute_251: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
    addmm_44: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg228_1, view_642, permute_251);  arg228_1 = view_642 = permute_251 = None
    view_643: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 16384]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_226: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_643, 0.5)
    pow_23: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_643, 3.0)
    mul_227: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
    add_180: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_643, mul_227);  view_643 = mul_227 = None
    mul_228: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_180, 0.7978845608028654);  add_180 = None
    tanh_22: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_228);  mul_228 = None
    add_181: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
    mul_229: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_226, add_181);  mul_226 = add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_644: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_229, [128, 16384]);  mul_229 = None
    permute_252: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
    addmm_45: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg230_1, view_644, permute_252);  arg230_1 = view_644 = permute_252 = None
    view_645: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_45, [1, 128, 4096]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_182: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_641, view_645);  view_641 = view_645 = None
    add_183: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_182, add_175);  add_182 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_183, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 128, 1]" = var_mean_23[0]
    getitem_93: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_46: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg354_1, [1, 1, 1]);  arg354_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_300: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_47: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_300, [1, 1, 64]);  unsqueeze_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_23: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_46, 1, repeat_47);  repeat_46 = repeat_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_23 = torch.ops.aten.split_with_sizes.default(gather_23, [32, 32], 2);  gather_23 = None
    getitem_94: "f32[1, 128, 32]" = split_with_sizes_23[0]
    getitem_95: "f32[1, 128, 32]" = split_with_sizes_23[1];  split_with_sizes_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant23: "f32[]" = self._tensor_constant23
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1151: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg355_1, 2, 0, 128);  arg355_1 = None
    slice_1152: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1151, 3, 0, 128);  slice_1151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_46: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_183, getitem_93);  getitem_93 = None
    add_184: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
    mul_230: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_23);  sub_46 = rsqrt_23 = None
    mul_231: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_230, arg231_1);  mul_230 = arg231_1 = None
    add_185: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_231, arg232_1);  mul_231 = arg232_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_646: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_185, [128, 4096])
    permute_253: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
    mm_92: "f32[128, 4096]" = torch.ops.aten.mm.default(view_646, permute_253);  view_646 = permute_253 = None
    view_647: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_92, [1, 128, 4096]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_652: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_647, [1, 128, 16, 256]);  view_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1116: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_652, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_309: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_95, 2)
    unsqueeze_310: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 4);  unsqueeze_309 = None
    expand_187: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_310, [1, 128, 1, 32, 2]);  unsqueeze_310 = None
    clone_188: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_187, memory_format = torch.contiguous_format);  expand_187 = None
    view_659: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_188, [1, 128, 1, 64]);  clone_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_234: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1116, view_659);  view_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1148: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1116, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_47: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1148);  slice_1148 = None
    unsqueeze_311: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_47, 4);  neg_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1144: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1116, 3, 0, 9223372036854775807, 2);  slice_1116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_312: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1144, 4);  slice_1144 = None
    cat_93: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_311, unsqueeze_312], 4);  unsqueeze_311 = unsqueeze_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_660: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_93, [1, 128, 16, 64]);  cat_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_307: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_94, 2)
    unsqueeze_308: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 4);  unsqueeze_307 = None
    expand_186: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_308, [1, 128, 1, 32, 2]);  unsqueeze_308 = None
    clone_187: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_186, memory_format = torch.contiguous_format);  expand_186 = None
    view_658: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_187, [1, 128, 1, 64]);  clone_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_235: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_660, view_658);  view_660 = view_658 = None
    add_187: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_234, mul_235);  mul_234 = mul_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1120: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_652, 3, 64, 9223372036854775807);  view_652 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_95: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_187, slice_1120], 3);  add_187 = slice_1120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_258: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_95, [0, 2, 1, 3]);  cat_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_188: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_258, [1, 16, 128, 256]);  permute_258 = None
    view_661: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_188, [16, 128, 256]);  expand_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_648: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_185, [128, 4096])
    permute_254: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
    mm_93: "f32[128, 4096]" = torch.ops.aten.mm.default(view_648, permute_254);  view_648 = permute_254 = None
    view_649: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_93, [1, 128, 4096]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_653: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_649, [1, 128, 16, 256]);  view_649 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1108: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_653, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_303: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_95, 2);  getitem_95 = None
    unsqueeze_304: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 4);  unsqueeze_303 = None
    expand_185: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_304, [1, 128, 1, 32, 2]);  unsqueeze_304 = None
    clone_186: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_185, memory_format = torch.contiguous_format);  expand_185 = None
    view_656: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_186, [1, 128, 1, 64]);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_232: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1108, view_656);  view_656 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1134: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1108, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_46: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1134);  slice_1134 = None
    unsqueeze_305: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_46, 4);  neg_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1130: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1108, 3, 0, 9223372036854775807, 2);  slice_1108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_306: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1130, 4);  slice_1130 = None
    cat_92: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_305, unsqueeze_306], 4);  unsqueeze_305 = unsqueeze_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_657: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_92, [1, 128, 16, 64]);  cat_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_301: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_94, 2);  getitem_94 = None
    unsqueeze_302: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 4);  unsqueeze_301 = None
    expand_184: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_302, [1, 128, 1, 32, 2]);  unsqueeze_302 = None
    clone_185: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_184, memory_format = torch.contiguous_format);  expand_184 = None
    view_655: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_185, [1, 128, 1, 64]);  clone_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_233: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_657, view_655);  view_657 = view_655 = None
    add_186: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_232, mul_233);  mul_232 = mul_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1112: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_653, 3, 64, 9223372036854775807);  view_653 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_94: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_186, slice_1112], 3);  add_186 = slice_1112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_257: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_94, [0, 2, 1, 3]);  cat_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_259: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2])
    expand_189: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_259, [1, 16, 256, 128]);  permute_259 = None
    view_662: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_189, [16, 256, 128]);  expand_189 = None
    bmm_46: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_661, view_662);  view_661 = view_662 = None
    view_663: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, [1, 16, 128, 128]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_23: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_23: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1152, view_663, full_default_23);  slice_1152 = view_663 = full_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_46: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_23, arg356_1);  where_23 = arg356_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_23: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_46, [-1], True)
    sub_47: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_46, amax_23);  div_46 = amax_23 = None
    exp_23: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_47);  sub_47 = None
    sum_24: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_47: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_190: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_47, [1, 16, 128, 128]);  div_47 = None
    view_664: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_190, [16, 128, 128]);  expand_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_650: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_185, [128, 4096])
    permute_255: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
    mm_94: "f32[128, 4096]" = torch.ops.aten.mm.default(view_650, permute_255);  view_650 = permute_255 = None
    view_651: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_94, [1, 128, 4096]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_654: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_651, [1, 128, 16, 256]);  view_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_256: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_654, [0, 2, 1, 3]);  view_654 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_191: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_256, [1, 16, 128, 256])
    view_665: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_191, [16, 128, 256]);  expand_191 = None
    bmm_47: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_664, view_665);  view_664 = view_665 = None
    view_666: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_47, [1, 16, 128, 256]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_260: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_666, [0, 2, 1, 3]);  view_666 = None
    clone_190: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_667: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_190, [1, 128, 4096]);  clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_668: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_667, [128, 4096]);  view_667 = None
    permute_261: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
    mm_95: "f32[128, 4096]" = torch.ops.aten.mm.default(view_668, permute_261);  view_668 = permute_261 = None
    view_669: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_95, [1, 128, 4096]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_670: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_185, [128, 4096]);  add_185 = None
    permute_262: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
    addmm_46: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg238_1, view_670, permute_262);  arg238_1 = view_670 = permute_262 = None
    view_671: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 16384]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_236: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_671, 0.5)
    pow_24: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_671, 3.0)
    mul_237: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
    add_188: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_671, mul_237);  view_671 = mul_237 = None
    mul_238: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_188, 0.7978845608028654);  add_188 = None
    tanh_23: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_238);  mul_238 = None
    add_189: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
    mul_239: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_236, add_189);  mul_236 = add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_672: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_239, [128, 16384]);  mul_239 = None
    permute_263: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
    addmm_47: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg240_1, view_672, permute_263);  arg240_1 = view_672 = permute_263 = None
    view_673: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_47, [1, 128, 4096]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_190: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_669, view_673);  view_669 = view_673 = None
    add_191: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_190, add_183);  add_190 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_191, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 128, 1]" = var_mean_24[0]
    getitem_97: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_48: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg357_1, [1, 1, 1]);  arg357_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_313: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_49: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_313, [1, 1, 64]);  unsqueeze_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_24: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_48, 1, repeat_49);  repeat_48 = repeat_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_24 = torch.ops.aten.split_with_sizes.default(gather_24, [32, 32], 2);  gather_24 = None
    getitem_98: "f32[1, 128, 32]" = split_with_sizes_24[0]
    getitem_99: "f32[1, 128, 32]" = split_with_sizes_24[1];  split_with_sizes_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant24: "f32[]" = self._tensor_constant24
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1199: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg358_1, 2, 0, 128);  arg358_1 = None
    slice_1200: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1199, 3, 0, 128);  slice_1199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_48: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_191, getitem_97);  getitem_97 = None
    add_192: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    mul_240: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_24);  sub_48 = rsqrt_24 = None
    mul_241: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_240, arg241_1);  mul_240 = arg241_1 = None
    add_193: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_241, arg242_1);  mul_241 = arg242_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_674: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_193, [128, 4096])
    permute_264: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
    mm_96: "f32[128, 4096]" = torch.ops.aten.mm.default(view_674, permute_264);  view_674 = permute_264 = None
    view_675: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_96, [1, 128, 4096]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_680: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_675, [1, 128, 16, 256]);  view_675 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1164: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_680, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_322: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_99, 2)
    unsqueeze_323: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 4);  unsqueeze_322 = None
    expand_195: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_323, [1, 128, 1, 32, 2]);  unsqueeze_323 = None
    clone_196: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_195, memory_format = torch.contiguous_format);  expand_195 = None
    view_687: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_196, [1, 128, 1, 64]);  clone_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_244: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1164, view_687);  view_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1196: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1164, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_49: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1196);  slice_1196 = None
    unsqueeze_324: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_49, 4);  neg_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1192: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1164, 3, 0, 9223372036854775807, 2);  slice_1164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_325: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1192, 4);  slice_1192 = None
    cat_97: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_324, unsqueeze_325], 4);  unsqueeze_324 = unsqueeze_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_688: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_97, [1, 128, 16, 64]);  cat_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_320: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_98, 2)
    unsqueeze_321: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 4);  unsqueeze_320 = None
    expand_194: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_321, [1, 128, 1, 32, 2]);  unsqueeze_321 = None
    clone_195: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_194, memory_format = torch.contiguous_format);  expand_194 = None
    view_686: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_195, [1, 128, 1, 64]);  clone_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_245: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_688, view_686);  view_688 = view_686 = None
    add_195: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_244, mul_245);  mul_244 = mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1168: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_680, 3, 64, 9223372036854775807);  view_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_99: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_195, slice_1168], 3);  add_195 = slice_1168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_269: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_99, [0, 2, 1, 3]);  cat_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_196: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_269, [1, 16, 128, 256]);  permute_269 = None
    view_689: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_196, [16, 128, 256]);  expand_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_676: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_193, [128, 4096])
    permute_265: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
    mm_97: "f32[128, 4096]" = torch.ops.aten.mm.default(view_676, permute_265);  view_676 = permute_265 = None
    view_677: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_97, [1, 128, 4096]);  mm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_681: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_677, [1, 128, 16, 256]);  view_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1156: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_681, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_316: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_99, 2);  getitem_99 = None
    unsqueeze_317: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 4);  unsqueeze_316 = None
    expand_193: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_317, [1, 128, 1, 32, 2]);  unsqueeze_317 = None
    clone_194: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_193, memory_format = torch.contiguous_format);  expand_193 = None
    view_684: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_194, [1, 128, 1, 64]);  clone_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_242: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1156, view_684);  view_684 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1182: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1156, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_48: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1182);  slice_1182 = None
    unsqueeze_318: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_48, 4);  neg_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1178: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1156, 3, 0, 9223372036854775807, 2);  slice_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_319: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1178, 4);  slice_1178 = None
    cat_96: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_318, unsqueeze_319], 4);  unsqueeze_318 = unsqueeze_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_685: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_96, [1, 128, 16, 64]);  cat_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_314: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_98, 2);  getitem_98 = None
    unsqueeze_315: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 4);  unsqueeze_314 = None
    expand_192: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_315, [1, 128, 1, 32, 2]);  unsqueeze_315 = None
    clone_193: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_192, memory_format = torch.contiguous_format);  expand_192 = None
    view_683: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_193, [1, 128, 1, 64]);  clone_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_243: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_685, view_683);  view_685 = view_683 = None
    add_194: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_242, mul_243);  mul_242 = mul_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1160: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_681, 3, 64, 9223372036854775807);  view_681 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_98: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_194, slice_1160], 3);  add_194 = slice_1160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_268: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_98, [0, 2, 1, 3]);  cat_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_270: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_268, [0, 1, 3, 2])
    expand_197: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_270, [1, 16, 256, 128]);  permute_270 = None
    view_690: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_197, [16, 256, 128]);  expand_197 = None
    bmm_48: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_689, view_690);  view_689 = view_690 = None
    view_691: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_48, [1, 16, 128, 128]);  bmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_24: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_24: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1200, view_691, full_default_24);  slice_1200 = view_691 = full_default_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_48: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_24, arg359_1);  where_24 = arg359_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_24: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_48, [-1], True)
    sub_49: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_48, amax_24);  div_48 = amax_24 = None
    exp_24: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_25: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
    div_49: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_198: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_49, [1, 16, 128, 128]);  div_49 = None
    view_692: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_198, [16, 128, 128]);  expand_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_678: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_193, [128, 4096])
    permute_266: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
    mm_98: "f32[128, 4096]" = torch.ops.aten.mm.default(view_678, permute_266);  view_678 = permute_266 = None
    view_679: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_98, [1, 128, 4096]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_682: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_679, [1, 128, 16, 256]);  view_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_267: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_682, [0, 2, 1, 3]);  view_682 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_199: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_267, [1, 16, 128, 256])
    view_693: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_199, [16, 128, 256]);  expand_199 = None
    bmm_49: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_692, view_693);  view_692 = view_693 = None
    view_694: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_49, [1, 16, 128, 256]);  bmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_271: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_694, [0, 2, 1, 3]);  view_694 = None
    clone_198: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_695: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_198, [1, 128, 4096]);  clone_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_696: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_695, [128, 4096]);  view_695 = None
    permute_272: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg246_1, [1, 0]);  arg246_1 = None
    mm_99: "f32[128, 4096]" = torch.ops.aten.mm.default(view_696, permute_272);  view_696 = permute_272 = None
    view_697: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_99, [1, 128, 4096]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_698: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_193, [128, 4096]);  add_193 = None
    permute_273: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
    addmm_48: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg248_1, view_698, permute_273);  arg248_1 = view_698 = permute_273 = None
    view_699: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 16384]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_246: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_699, 0.5)
    pow_25: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_699, 3.0)
    mul_247: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
    add_196: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_699, mul_247);  view_699 = mul_247 = None
    mul_248: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_196, 0.7978845608028654);  add_196 = None
    tanh_24: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_248);  mul_248 = None
    add_197: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_24, 1.0);  tanh_24 = None
    mul_249: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_246, add_197);  mul_246 = add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_700: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_249, [128, 16384]);  mul_249 = None
    permute_274: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
    addmm_49: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg250_1, view_700, permute_274);  arg250_1 = view_700 = permute_274 = None
    view_701: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_49, [1, 128, 4096]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_198: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_697, view_701);  view_697 = view_701 = None
    add_199: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_198, add_191);  add_198 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_199, [2], correction = 0, keepdim = True)
    getitem_100: "f32[1, 128, 1]" = var_mean_25[0]
    getitem_101: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_50: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg360_1, [1, 1, 1]);  arg360_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_326: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_51: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_326, [1, 1, 64]);  unsqueeze_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_25: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_50, 1, repeat_51);  repeat_50 = repeat_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_25 = torch.ops.aten.split_with_sizes.default(gather_25, [32, 32], 2);  gather_25 = None
    getitem_102: "f32[1, 128, 32]" = split_with_sizes_25[0]
    getitem_103: "f32[1, 128, 32]" = split_with_sizes_25[1];  split_with_sizes_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant25: "f32[]" = self._tensor_constant25
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1247: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg361_1, 2, 0, 128);  arg361_1 = None
    slice_1248: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1247, 3, 0, 128);  slice_1247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_50: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_199, getitem_101);  getitem_101 = None
    add_200: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
    mul_250: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_25);  sub_50 = rsqrt_25 = None
    mul_251: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_250, arg251_1);  mul_250 = arg251_1 = None
    add_201: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_251, arg252_1);  mul_251 = arg252_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_702: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_201, [128, 4096])
    permute_275: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
    mm_100: "f32[128, 4096]" = torch.ops.aten.mm.default(view_702, permute_275);  view_702 = permute_275 = None
    view_703: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_100, [1, 128, 4096]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_708: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_703, [1, 128, 16, 256]);  view_703 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1212: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_708, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_335: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_103, 2)
    unsqueeze_336: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 4);  unsqueeze_335 = None
    expand_203: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_336, [1, 128, 1, 32, 2]);  unsqueeze_336 = None
    clone_204: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_203, memory_format = torch.contiguous_format);  expand_203 = None
    view_715: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_204, [1, 128, 1, 64]);  clone_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_254: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1212, view_715);  view_715 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1244: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1212, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_51: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1244);  slice_1244 = None
    unsqueeze_337: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_51, 4);  neg_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1240: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1212, 3, 0, 9223372036854775807, 2);  slice_1212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_338: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1240, 4);  slice_1240 = None
    cat_101: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_337, unsqueeze_338], 4);  unsqueeze_337 = unsqueeze_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_716: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_101, [1, 128, 16, 64]);  cat_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_333: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_102, 2)
    unsqueeze_334: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 4);  unsqueeze_333 = None
    expand_202: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_334, [1, 128, 1, 32, 2]);  unsqueeze_334 = None
    clone_203: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_202, memory_format = torch.contiguous_format);  expand_202 = None
    view_714: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_203, [1, 128, 1, 64]);  clone_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_255: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_716, view_714);  view_716 = view_714 = None
    add_203: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_254, mul_255);  mul_254 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1216: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_708, 3, 64, 9223372036854775807);  view_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_103: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_203, slice_1216], 3);  add_203 = slice_1216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_280: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_103, [0, 2, 1, 3]);  cat_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_204: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_280, [1, 16, 128, 256]);  permute_280 = None
    view_717: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_204, [16, 128, 256]);  expand_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_704: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_201, [128, 4096])
    permute_276: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    mm_101: "f32[128, 4096]" = torch.ops.aten.mm.default(view_704, permute_276);  view_704 = permute_276 = None
    view_705: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_101, [1, 128, 4096]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_709: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_705, [1, 128, 16, 256]);  view_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1204: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_709, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_329: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_103, 2);  getitem_103 = None
    unsqueeze_330: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 4);  unsqueeze_329 = None
    expand_201: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_330, [1, 128, 1, 32, 2]);  unsqueeze_330 = None
    clone_202: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_201, memory_format = torch.contiguous_format);  expand_201 = None
    view_712: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_202, [1, 128, 1, 64]);  clone_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_252: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1204, view_712);  view_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1230: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1204, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_50: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1230);  slice_1230 = None
    unsqueeze_331: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_50, 4);  neg_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1226: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1204, 3, 0, 9223372036854775807, 2);  slice_1204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_332: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1226, 4);  slice_1226 = None
    cat_100: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_331, unsqueeze_332], 4);  unsqueeze_331 = unsqueeze_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_713: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_100, [1, 128, 16, 64]);  cat_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_327: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_102, 2);  getitem_102 = None
    unsqueeze_328: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 4);  unsqueeze_327 = None
    expand_200: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_328, [1, 128, 1, 32, 2]);  unsqueeze_328 = None
    clone_201: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_200, memory_format = torch.contiguous_format);  expand_200 = None
    view_711: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_201, [1, 128, 1, 64]);  clone_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_253: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_713, view_711);  view_713 = view_711 = None
    add_202: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1208: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_709, 3, 64, 9223372036854775807);  view_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_102: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_202, slice_1208], 3);  add_202 = slice_1208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_279: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_102, [0, 2, 1, 3]);  cat_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_281: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_279, [0, 1, 3, 2])
    expand_205: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_281, [1, 16, 256, 128]);  permute_281 = None
    view_718: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_205, [16, 256, 128]);  expand_205 = None
    bmm_50: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_717, view_718);  view_717 = view_718 = None
    view_719: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_50, [1, 16, 128, 128]);  bmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_25: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_25: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1248, view_719, full_default_25);  slice_1248 = view_719 = full_default_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_50: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_25, arg362_1);  where_25 = arg362_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_25: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_50, [-1], True)
    sub_51: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_50, amax_25);  div_50 = amax_25 = None
    exp_25: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_26: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
    div_51: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_206: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_51, [1, 16, 128, 128]);  div_51 = None
    view_720: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_206, [16, 128, 128]);  expand_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_706: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_201, [128, 4096])
    permute_277: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
    mm_102: "f32[128, 4096]" = torch.ops.aten.mm.default(view_706, permute_277);  view_706 = permute_277 = None
    view_707: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_102, [1, 128, 4096]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_710: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_707, [1, 128, 16, 256]);  view_707 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_278: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_710, [0, 2, 1, 3]);  view_710 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_207: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_278, [1, 16, 128, 256])
    view_721: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_207, [16, 128, 256]);  expand_207 = None
    bmm_51: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_720, view_721);  view_720 = view_721 = None
    view_722: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_51, [1, 16, 128, 256]);  bmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_282: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_722, [0, 2, 1, 3]);  view_722 = None
    clone_206: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_723: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_206, [1, 128, 4096]);  clone_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_724: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_723, [128, 4096]);  view_723 = None
    permute_283: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg256_1, [1, 0]);  arg256_1 = None
    mm_103: "f32[128, 4096]" = torch.ops.aten.mm.default(view_724, permute_283);  view_724 = permute_283 = None
    view_725: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_103, [1, 128, 4096]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_726: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_201, [128, 4096]);  add_201 = None
    permute_284: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
    addmm_50: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg258_1, view_726, permute_284);  arg258_1 = view_726 = permute_284 = None
    view_727: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 16384]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_256: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_727, 0.5)
    pow_26: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_727, 3.0)
    mul_257: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
    add_204: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_727, mul_257);  view_727 = mul_257 = None
    mul_258: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_204, 0.7978845608028654);  add_204 = None
    tanh_25: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_258);  mul_258 = None
    add_205: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_25, 1.0);  tanh_25 = None
    mul_259: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_256, add_205);  mul_256 = add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_728: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_259, [128, 16384]);  mul_259 = None
    permute_285: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
    addmm_51: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg260_1, view_728, permute_285);  arg260_1 = view_728 = permute_285 = None
    view_729: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_51, [1, 128, 4096]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_206: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_725, view_729);  view_725 = view_729 = None
    add_207: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_206, add_199);  add_206 = add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_207, [2], correction = 0, keepdim = True)
    getitem_104: "f32[1, 128, 1]" = var_mean_26[0]
    getitem_105: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_52: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg363_1, [1, 1, 1]);  arg363_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_339: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1)
    repeat_53: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_339, [1, 1, 64]);  unsqueeze_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_26: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_52, 1, repeat_53);  repeat_52 = repeat_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_26 = torch.ops.aten.split_with_sizes.default(gather_26, [32, 32], 2);  gather_26 = None
    getitem_106: "f32[1, 128, 32]" = split_with_sizes_26[0]
    getitem_107: "f32[1, 128, 32]" = split_with_sizes_26[1];  split_with_sizes_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant26: "f32[]" = self._tensor_constant26
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1295: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg364_1, 2, 0, 128);  arg364_1 = None
    slice_1296: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1295, 3, 0, 128);  slice_1295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_52: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_207, getitem_105);  getitem_105 = None
    add_208: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-05);  getitem_104 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    mul_260: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_26);  sub_52 = rsqrt_26 = None
    mul_261: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_260, arg261_1);  mul_260 = arg261_1 = None
    add_209: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_261, arg262_1);  mul_261 = arg262_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_730: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_209, [128, 4096])
    permute_286: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
    mm_104: "f32[128, 4096]" = torch.ops.aten.mm.default(view_730, permute_286);  view_730 = permute_286 = None
    view_731: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_104, [1, 128, 4096]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_736: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_731, [1, 128, 16, 256]);  view_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1260: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_736, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_348: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_107, 2)
    unsqueeze_349: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 4);  unsqueeze_348 = None
    expand_211: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_349, [1, 128, 1, 32, 2]);  unsqueeze_349 = None
    clone_212: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_211, memory_format = torch.contiguous_format);  expand_211 = None
    view_743: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_212, [1, 128, 1, 64]);  clone_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_264: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1260, view_743);  view_743 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1292: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1260, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_53: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1292);  slice_1292 = None
    unsqueeze_350: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_53, 4);  neg_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1288: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1260, 3, 0, 9223372036854775807, 2);  slice_1260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_351: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1288, 4);  slice_1288 = None
    cat_105: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_350, unsqueeze_351], 4);  unsqueeze_350 = unsqueeze_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_744: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_105, [1, 128, 16, 64]);  cat_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_346: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_106, 2)
    unsqueeze_347: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 4);  unsqueeze_346 = None
    expand_210: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_347, [1, 128, 1, 32, 2]);  unsqueeze_347 = None
    clone_211: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_210, memory_format = torch.contiguous_format);  expand_210 = None
    view_742: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_211, [1, 128, 1, 64]);  clone_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_265: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_744, view_742);  view_744 = view_742 = None
    add_211: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_264, mul_265);  mul_264 = mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1264: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_736, 3, 64, 9223372036854775807);  view_736 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_107: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_211, slice_1264], 3);  add_211 = slice_1264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_291: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_107, [0, 2, 1, 3]);  cat_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_212: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_291, [1, 16, 128, 256]);  permute_291 = None
    view_745: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_212, [16, 128, 256]);  expand_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_732: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_209, [128, 4096])
    permute_287: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
    mm_105: "f32[128, 4096]" = torch.ops.aten.mm.default(view_732, permute_287);  view_732 = permute_287 = None
    view_733: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_105, [1, 128, 4096]);  mm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_737: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_733, [1, 128, 16, 256]);  view_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1252: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_737, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_342: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_107, 2);  getitem_107 = None
    unsqueeze_343: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 4);  unsqueeze_342 = None
    expand_209: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_343, [1, 128, 1, 32, 2]);  unsqueeze_343 = None
    clone_210: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_209, memory_format = torch.contiguous_format);  expand_209 = None
    view_740: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_210, [1, 128, 1, 64]);  clone_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_262: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1252, view_740);  view_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1278: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1252, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_52: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1278);  slice_1278 = None
    unsqueeze_344: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_52, 4);  neg_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1274: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1252, 3, 0, 9223372036854775807, 2);  slice_1252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_345: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1274, 4);  slice_1274 = None
    cat_104: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_344, unsqueeze_345], 4);  unsqueeze_344 = unsqueeze_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_741: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_104, [1, 128, 16, 64]);  cat_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_340: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_106, 2);  getitem_106 = None
    unsqueeze_341: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 4);  unsqueeze_340 = None
    expand_208: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_341, [1, 128, 1, 32, 2]);  unsqueeze_341 = None
    clone_209: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_208, memory_format = torch.contiguous_format);  expand_208 = None
    view_739: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_209, [1, 128, 1, 64]);  clone_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_263: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_741, view_739);  view_741 = view_739 = None
    add_210: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1256: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_737, 3, 64, 9223372036854775807);  view_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_106: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_210, slice_1256], 3);  add_210 = slice_1256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_290: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_106, [0, 2, 1, 3]);  cat_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_292: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_290, [0, 1, 3, 2])
    expand_213: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_292, [1, 16, 256, 128]);  permute_292 = None
    view_746: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_213, [16, 256, 128]);  expand_213 = None
    bmm_52: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_745, view_746);  view_745 = view_746 = None
    view_747: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_52, [1, 16, 128, 128]);  bmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_26: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_26: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1296, view_747, full_default_26);  slice_1296 = view_747 = full_default_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_52: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_26, arg365_1);  where_26 = arg365_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_26: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_52, [-1], True)
    sub_53: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_52, amax_26);  div_52 = amax_26 = None
    exp_26: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
    sum_27: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [-1], True)
    div_53: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_26, sum_27);  exp_26 = sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_214: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_53, [1, 16, 128, 128]);  div_53 = None
    view_748: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_214, [16, 128, 128]);  expand_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_734: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_209, [128, 4096])
    permute_288: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
    mm_106: "f32[128, 4096]" = torch.ops.aten.mm.default(view_734, permute_288);  view_734 = permute_288 = None
    view_735: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_106, [1, 128, 4096]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_738: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_735, [1, 128, 16, 256]);  view_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_289: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_738, [0, 2, 1, 3]);  view_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_215: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_289, [1, 16, 128, 256])
    view_749: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_215, [16, 128, 256]);  expand_215 = None
    bmm_53: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_748, view_749);  view_748 = view_749 = None
    view_750: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_53, [1, 16, 128, 256]);  bmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_293: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_750, [0, 2, 1, 3]);  view_750 = None
    clone_214: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_751: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_214, [1, 128, 4096]);  clone_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_752: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_751, [128, 4096]);  view_751 = None
    permute_294: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
    mm_107: "f32[128, 4096]" = torch.ops.aten.mm.default(view_752, permute_294);  view_752 = permute_294 = None
    view_753: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_107, [1, 128, 4096]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_754: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_209, [128, 4096]);  add_209 = None
    permute_295: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
    addmm_52: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg268_1, view_754, permute_295);  arg268_1 = view_754 = permute_295 = None
    view_755: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 16384]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_266: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_755, 0.5)
    pow_27: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_755, 3.0)
    mul_267: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
    add_212: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_755, mul_267);  view_755 = mul_267 = None
    mul_268: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_212, 0.7978845608028654);  add_212 = None
    tanh_26: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_268);  mul_268 = None
    add_213: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_26, 1.0);  tanh_26 = None
    mul_269: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_266, add_213);  mul_266 = add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_756: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_269, [128, 16384]);  mul_269 = None
    permute_296: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
    addmm_53: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg270_1, view_756, permute_296);  arg270_1 = view_756 = permute_296 = None
    view_757: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_53, [1, 128, 4096]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_214: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_753, view_757);  view_753 = view_757 = None
    add_215: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_214, add_207);  add_214 = add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_215, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1, 128, 1]" = var_mean_27[0]
    getitem_109: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:192, code: return embed_positions.repeat(position_ids.shape[0], 1, 1)
    repeat_54: "f32[1, 2048, 64]" = torch.ops.aten.repeat.default(arg366_1, [1, 1, 1]);  arg366_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:222, code: repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
    unsqueeze_352: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(view_1, -1);  view_1 = None
    repeat_55: "i64[1, 128, 64]" = torch.ops.aten.repeat.default(unsqueeze_352, [1, 1, 64]);  unsqueeze_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:223, code: sincos = torch.gather(embed_positions, 1, repeated_position_ids)
    gather_27: "f32[1, 128, 64]" = torch.ops.aten.gather.default(repeat_54, 1, repeat_55);  repeat_54 = repeat_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:224, code: sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
    split_with_sizes_27 = torch.ops.aten.split_with_sizes.default(gather_27, [32, 32], 2);  gather_27 = None
    getitem_110: "f32[1, 128, 32]" = split_with_sizes_27[0]
    getitem_111: "f32[1, 128, 32]" = split_with_sizes_27[1];  split_with_sizes_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant27: "f32[]" = self._tensor_constant27
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:155, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_1343: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg367_1, 2, 0, 128);  arg367_1 = None
    slice_1344: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_1343, 3, 0, 128);  slice_1343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:308, code: hidden_states = self.ln_1(hidden_states)
    sub_54: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_215, getitem_109);  getitem_109 = None
    add_216: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_216);  add_216 = None
    mul_270: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_27);  sub_54 = rsqrt_27 = None
    mul_271: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_270, arg271_1);  mul_270 = arg271_1 = None
    add_217: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_271, arg272_1);  mul_271 = arg272_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:207, code: query = self.q_proj(hidden_states)
    view_758: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_217, [128, 4096])
    permute_297: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
    mm_108: "f32[128, 4096]" = torch.ops.aten.mm.default(view_758, permute_297);  view_758 = permute_297 = None
    view_759: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_108, [1, 128, 4096]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_764: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_759, [1, 128, 16, 256]);  view_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:230, code: q_rot = query[:, :, :, : self.rotary_dim]
    slice_1308: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_764, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_361: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_111, 2)
    unsqueeze_362: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 4);  unsqueeze_361 = None
    expand_219: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_362, [1, 128, 1, 32, 2]);  unsqueeze_362 = None
    clone_220: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_219, memory_format = torch.contiguous_format);  expand_219 = None
    view_771: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_220, [1, 128, 1, 64]);  clone_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_274: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1308, view_771);  view_771 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1340: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1308, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_55: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1340);  slice_1340 = None
    unsqueeze_363: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_55, 4);  neg_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1336: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1308, 3, 0, 9223372036854775807, 2);  slice_1308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_364: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1336, 4);  slice_1336 = None
    cat_109: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_363, unsqueeze_364], 4);  unsqueeze_363 = unsqueeze_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_772: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_109, [1, 128, 16, 64]);  cat_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_359: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_110, 2)
    unsqueeze_360: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 4);  unsqueeze_359 = None
    expand_218: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_360, [1, 128, 1, 32, 2]);  unsqueeze_360 = None
    clone_219: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_218, memory_format = torch.contiguous_format);  expand_218 = None
    view_770: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_219, [1, 128, 1, 64]);  clone_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_275: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_772, view_770);  view_772 = view_770 = None
    add_219: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_274, mul_275);  mul_274 = mul_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:231, code: q_pass = query[:, :, :, self.rotary_dim :]
    slice_1312: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_764, 3, 64, 9223372036854775807);  view_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:237, code: query = torch.cat([q_rot, q_pass], dim=-1)
    cat_111: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_219, slice_1312], 3);  add_219 = slice_1312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:243, code: query = query.permute(0, 2, 1, 3)
    permute_302: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_111, [0, 2, 1, 3]);  cat_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_220: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_302, [1, 16, 128, 256]);  permute_302 = None
    view_773: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_220, [16, 128, 256]);  expand_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:208, code: key = self.k_proj(hidden_states)
    view_760: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_217, [128, 4096])
    permute_298: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg274_1, [1, 0]);  arg274_1 = None
    mm_109: "f32[128, 4096]" = torch.ops.aten.mm.default(view_760, permute_298);  view_760 = permute_298 = None
    view_761: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_109, [1, 128, 4096]);  mm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_765: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_761, [1, 128, 16, 256]);  view_761 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:227, code: k_rot = key[:, :, :, : self.rotary_dim]
    slice_1300: "f32[1, 128, 16, 64]" = torch.ops.aten.slice.Tensor(view_765, 3, 0, 64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:78, code: cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    unsqueeze_355: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_111, 2);  getitem_111 = None
    unsqueeze_356: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 4);  unsqueeze_355 = None
    expand_217: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_356, [1, 128, 1, 32, 2]);  unsqueeze_356 = None
    clone_218: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_217, memory_format = torch.contiguous_format);  expand_217 = None
    view_768: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_218, [1, 128, 1, 64]);  clone_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_272: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(slice_1300, view_768);  view_768 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:71, code: x2 = x[:, :, :, 1::2]
    slice_1326: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1300, 3, 1, 9223372036854775807, 2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    neg_54: "f32[1, 128, 16, 32]" = torch.ops.aten.neg.default(slice_1326);  slice_1326 = None
    unsqueeze_357: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(neg_54, 4);  neg_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:70, code: x1 = x[:, :, :, ::2]
    slice_1322: "f32[1, 128, 16, 32]" = torch.ops.aten.slice.Tensor(slice_1300, 3, 0, 9223372036854775807, 2);  slice_1300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:72, code: x = torch.stack((-x2, x1), dim=-1)
    unsqueeze_358: "f32[1, 128, 16, 32, 1]" = torch.ops.aten.unsqueeze.default(slice_1322, 4);  slice_1322 = None
    cat_108: "f32[1, 128, 16, 32, 2]" = torch.ops.aten.cat.default([unsqueeze_357, unsqueeze_358], 4);  unsqueeze_357 = unsqueeze_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:73, code: return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    view_769: "f32[1, 128, 16, 64]" = torch.ops.aten.reshape.default(cat_108, [1, 128, 16, 64]);  cat_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:77, code: sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    unsqueeze_353: "f32[1, 128, 1, 32]" = torch.ops.aten.unsqueeze.default(getitem_110, 2);  getitem_110 = None
    unsqueeze_354: "f32[1, 128, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 4);  unsqueeze_353 = None
    expand_216: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.expand.default(unsqueeze_354, [1, 128, 1, 32, 2]);  unsqueeze_354 = None
    clone_217: "f32[1, 128, 1, 32, 2]" = torch.ops.aten.clone.default(expand_216, memory_format = torch.contiguous_format);  expand_216 = None
    view_767: "f32[1, 128, 1, 64]" = torch.ops.aten.reshape.default(clone_217, [1, 128, 1, 64]);  clone_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:79, code: return (tensor * cos) + (rotate_every_two(tensor) * sin)
    mul_273: "f32[1, 128, 16, 64]" = torch.ops.aten.mul.Tensor(view_769, view_767);  view_769 = view_767 = None
    add_218: "f32[1, 128, 16, 64]" = torch.ops.aten.add.Tensor(mul_272, mul_273);  mul_272 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:228, code: k_pass = key[:, :, :, self.rotary_dim :]
    slice_1304: "f32[1, 128, 16, 192]" = torch.ops.aten.slice.Tensor(view_765, 3, 64, 9223372036854775807);  view_765 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:236, code: key = torch.cat([k_rot, k_pass], dim=-1)
    cat_110: "f32[1, 128, 16, 256]" = torch.ops.aten.cat.default([add_218, slice_1304], 3);  add_218 = slice_1304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:242, code: key = key.permute(0, 2, 1, 3)
    permute_301: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(cat_110, [0, 2, 1, 3]);  cat_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:161, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_303: "f32[1, 16, 256, 128]" = torch.ops.aten.permute.default(permute_301, [0, 1, 3, 2])
    expand_221: "f32[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_303, [1, 16, 256, 128]);  permute_303 = None
    view_774: "f32[16, 256, 128]" = torch.ops.aten.reshape.default(expand_221, [16, 256, 128]);  expand_221 = None
    bmm_54: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_773, view_774);  view_773 = view_774 = None
    view_775: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_54, [1, 16, 128, 128]);  bmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:166, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_27: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:167, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_27: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_1344, view_775, full_default_27);  slice_1344 = view_775 = full_default_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:169, code: attn_weights = attn_weights / self.scale_attn
    div_54: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(where_27, arg368_1);  where_27 = arg368_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:175, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_27: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(div_54, [-1], True)
    sub_55: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(div_54, amax_27);  div_54 = amax_27 = None
    exp_27: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_28: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_27, [-1], True)
    div_55: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_27, sum_28);  exp_27 = sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_222: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_55, [1, 16, 128, 128]);  div_55 = None
    view_776: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_222, [16, 128, 128]);  expand_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:209, code: value = self.v_proj(hidden_states)
    view_762: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_217, [128, 4096])
    permute_299: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
    mm_110: "f32[128, 4096]" = torch.ops.aten.mm.default(view_762, permute_299);  view_762 = permute_299 = None
    view_763: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_110, [1, 128, 4096]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:122, code: tensor = tensor.view(new_shape)
    view_766: "f32[1, 128, 16, 256]" = torch.ops.aten.reshape.default(view_763, [1, 128, 16, 256]);  view_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:128, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_300: "f32[1, 16, 128, 256]" = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:183, code: attn_output = torch.matmul(attn_weights, value)
    expand_223: "f32[1, 16, 128, 256]" = torch.ops.aten.expand.default(permute_300, [1, 16, 128, 256])
    view_777: "f32[16, 128, 256]" = torch.ops.aten.reshape.default(expand_223, [16, 128, 256]);  expand_223 = None
    bmm_55: "f32[16, 128, 256]" = torch.ops.aten.bmm.default(view_776, view_777);  view_776 = view_777 = None
    view_778: "f32[1, 16, 128, 256]" = torch.ops.aten.reshape.default(bmm_55, [1, 16, 128, 256]);  bmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:139, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_304: "f32[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_778, [0, 2, 1, 3]);  view_778 = None
    clone_222: "f32[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_304, memory_format = torch.contiguous_format);  permute_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:143, code: return tensor.view(new_shape)
    view_779: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(clone_222, [1, 128, 4096]);  clone_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:260, code: attn_output = self.out_proj(attn_output)
    view_780: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_779, [128, 4096]);  view_779 = None
    permute_305: "f32[4096, 4096]" = torch.ops.aten.permute.default(arg276_1, [1, 0]);  arg276_1 = None
    mm_111: "f32[128, 4096]" = torch.ops.aten.mm.default(view_780, permute_305);  view_780 = permute_305 = None
    view_781: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(mm_111, [1, 128, 4096]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:282, code: hidden_states = self.fc_in(hidden_states)
    view_782: "f32[128, 4096]" = torch.ops.aten.reshape.default(add_217, [128, 4096]);  add_217 = None
    permute_306: "f32[4096, 16384]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
    addmm_54: "f32[128, 16384]" = torch.ops.aten.addmm.default(arg278_1, view_782, permute_306);  arg278_1 = view_782 = permute_306 = None
    view_783: "f32[1, 128, 16384]" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 16384]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_276: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(view_783, 0.5)
    pow_28: "f32[1, 128, 16384]" = torch.ops.aten.pow.Tensor_Scalar(view_783, 3.0)
    mul_277: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(pow_28, 0.044715);  pow_28 = None
    add_220: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(view_783, mul_277);  view_783 = mul_277 = None
    mul_278: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(add_220, 0.7978845608028654);  add_220 = None
    tanh_27: "f32[1, 128, 16384]" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
    add_221: "f32[1, 128, 16384]" = torch.ops.aten.add.Tensor(tanh_27, 1.0);  tanh_27 = None
    mul_279: "f32[1, 128, 16384]" = torch.ops.aten.mul.Tensor(mul_276, add_221);  mul_276 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:284, code: hidden_states = self.fc_out(hidden_states)
    view_784: "f32[128, 16384]" = torch.ops.aten.reshape.default(mul_279, [128, 16384]);  mul_279 = None
    permute_307: "f32[16384, 4096]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
    addmm_55: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg280_1, view_784, permute_307);  arg280_1 = view_784 = permute_307 = None
    view_785: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(addmm_55, [1, 128, 4096]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:322, code: hidden_states = attn_output + feed_forward_hidden_states + residual
    add_222: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(view_781, view_785);  view_781 = view_785 = None
    add_223: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(add_222, add_215);  add_222 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:713, code: hidden_states = self.ln_f(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_223, [2], correction = 0, keepdim = True)
    getitem_112: "f32[1, 128, 1]" = var_mean_28[0]
    getitem_113: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:886, code: shift_labels = labels[..., 1:].contiguous()
    slice_1347: "i64[1, 127]" = torch.ops.aten.slice.Tensor(arg370_1, 1, 1, 9223372036854775807);  arg370_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:889, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_790: "i64[127]" = torch.ops.aten.reshape.default(slice_1347, [-1]);  slice_1347 = None
    ne_1: "b8[127]" = torch.ops.aten.ne.Scalar(view_790, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:713, code: hidden_states = self.ln_f(hidden_states)
    sub_56: "f32[1, 128, 4096]" = torch.ops.aten.sub.Tensor(add_223, getitem_113);  add_223 = getitem_113 = None
    add_224: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-05);  getitem_112 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_224);  add_224 = None
    mul_280: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_28);  sub_56 = rsqrt_28 = None
    mul_281: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_280, arg281_1);  mul_280 = arg281_1 = None
    add_225: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(mul_281, arg282_1);  mul_281 = arg282_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:715, code: hidden_states = hidden_states.view(output_shape)
    view_786: "f32[1, 128, 4096]" = torch.ops.aten.reshape.default(add_225, [-1, 128, 4096]);  add_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:878, code: lm_logits = self.lm_head(hidden_states).to(torch.float32)
    view_787: "f32[128, 4096]" = torch.ops.aten.reshape.default(view_786, [128, 4096]);  view_786 = None
    permute_308: "f32[4096, 50400]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
    addmm_56: "f32[128, 50400]" = torch.ops.aten.addmm.default(arg284_1, view_787, permute_308);  arg284_1 = view_787 = permute_308 = None
    view_788: "f32[1, 128, 50400]" = torch.ops.aten.reshape.default(addmm_56, [1, 128, 50400]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:885, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    slice_1345: "f32[1, 127, 50400]" = torch.ops.aten.slice.Tensor(view_788, 1, 0, -1)
    slice_1346: "f32[1, 127, 50400]" = torch.ops.aten.slice.Tensor(slice_1345, 2, 0, 9223372036854775807);  slice_1345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gptj/modeling_gptj.py:889, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_789: "f32[127, 50400]" = torch.ops.aten.reshape.default(slice_1346, [-1, 50400]);  slice_1346 = None
    amax_28: "f32[127, 1]" = torch.ops.aten.amax.default(view_789, [1], True)
    sub_57: "f32[127, 50400]" = torch.ops.aten.sub.Tensor(view_789, amax_28);  view_789 = amax_28 = None
    exp_28: "f32[127, 50400]" = torch.ops.aten.exp.default(sub_57)
    sum_29: "f32[127, 1]" = torch.ops.aten.sum.dim_IntList(exp_28, [1], True);  exp_28 = None
    log: "f32[127, 1]" = torch.ops.aten.log.default(sum_29);  sum_29 = None
    sub_58: "f32[127, 50400]" = torch.ops.aten.sub.Tensor(sub_57, log);  sub_57 = log = None
    ne: "b8[127]" = torch.ops.aten.ne.Scalar(view_790, -100)
    full_default_28: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_28: "i64[127]" = torch.ops.aten.where.self(ne, view_790, full_default_28);  ne = full_default_28 = None
    unsqueeze_365: "i64[127, 1]" = torch.ops.aten.unsqueeze.default(where_28, 1);  where_28 = None
    gather_28: "f32[127, 1]" = torch.ops.aten.gather.default(sub_58, 1, unsqueeze_365);  sub_58 = unsqueeze_365 = None
    squeeze: "f32[127]" = torch.ops.aten.squeeze.dim(gather_28, 1);  gather_28 = None
    neg_56: "f32[127]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_29: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_29: "f32[127]" = torch.ops.aten.where.self(ne_1, neg_56, full_default_29);  ne_1 = neg_56 = full_default_29 = None
    sum_31: "f32[]" = torch.ops.aten.sum.default(where_29);  where_29 = None
    ne_2: "b8[127]" = torch.ops.aten.ne.Scalar(view_790, -100);  view_790 = None
    sum_30: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_30, torch.float32);  sum_30 = None
    div_56: "f32[]" = torch.ops.aten.div.Tensor(sum_31, convert_element_type);  sum_31 = convert_element_type = None
    return (div_56, view_788, permute_4, permute_3, permute_15, permute_14, permute_26, permute_25, permute_37, permute_36, permute_48, permute_47, permute_59, permute_58, permute_70, permute_69, permute_81, permute_80, permute_92, permute_91, permute_103, permute_102, permute_114, permute_113, permute_125, permute_124, permute_136, permute_135, permute_147, permute_146, permute_158, permute_157, permute_169, permute_168, permute_180, permute_179, permute_191, permute_190, permute_202, permute_201, permute_213, permute_212, permute_224, permute_223, permute_235, permute_234, permute_246, permute_245, permute_257, permute_256, permute_268, permute_267, permute_279, permute_278, permute_290, permute_289, permute_301, permute_300)
    