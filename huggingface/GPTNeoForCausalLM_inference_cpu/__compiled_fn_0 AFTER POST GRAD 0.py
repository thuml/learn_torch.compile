from __future__ import annotations



def forward(self, arg0_1: "f32[50257, 2048]", arg1_1: "f32[2048, 2048]", arg2_1: "f32[2048]", arg3_1: "f32[2048]", arg4_1: "f32[2048, 2048]", arg5_1: "f32[2048, 2048]", arg6_1: "f32[2048, 2048]", arg7_1: "f32[2048, 2048]", arg8_1: "f32[2048]", arg9_1: "f32[2048]", arg10_1: "f32[2048]", arg11_1: "f32[8192, 2048]", arg12_1: "f32[8192]", arg13_1: "f32[2048, 8192]", arg14_1: "f32[2048]", arg15_1: "f32[2048]", arg16_1: "f32[2048]", arg17_1: "f32[2048, 2048]", arg18_1: "f32[2048, 2048]", arg19_1: "f32[2048, 2048]", arg20_1: "f32[2048, 2048]", arg21_1: "f32[2048]", arg22_1: "f32[2048]", arg23_1: "f32[2048]", arg24_1: "f32[8192, 2048]", arg25_1: "f32[8192]", arg26_1: "f32[2048, 8192]", arg27_1: "f32[2048]", arg28_1: "f32[2048]", arg29_1: "f32[2048]", arg30_1: "f32[2048, 2048]", arg31_1: "f32[2048, 2048]", arg32_1: "f32[2048, 2048]", arg33_1: "f32[2048, 2048]", arg34_1: "f32[2048]", arg35_1: "f32[2048]", arg36_1: "f32[2048]", arg37_1: "f32[8192, 2048]", arg38_1: "f32[8192]", arg39_1: "f32[2048, 8192]", arg40_1: "f32[2048]", arg41_1: "f32[2048]", arg42_1: "f32[2048]", arg43_1: "f32[2048, 2048]", arg44_1: "f32[2048, 2048]", arg45_1: "f32[2048, 2048]", arg46_1: "f32[2048, 2048]", arg47_1: "f32[2048]", arg48_1: "f32[2048]", arg49_1: "f32[2048]", arg50_1: "f32[8192, 2048]", arg51_1: "f32[8192]", arg52_1: "f32[2048, 8192]", arg53_1: "f32[2048]", arg54_1: "f32[2048]", arg55_1: "f32[2048]", arg56_1: "f32[2048, 2048]", arg57_1: "f32[2048, 2048]", arg58_1: "f32[2048, 2048]", arg59_1: "f32[2048, 2048]", arg60_1: "f32[2048]", arg61_1: "f32[2048]", arg62_1: "f32[2048]", arg63_1: "f32[8192, 2048]", arg64_1: "f32[8192]", arg65_1: "f32[2048, 8192]", arg66_1: "f32[2048]", arg67_1: "f32[2048]", arg68_1: "f32[2048]", arg69_1: "f32[2048, 2048]", arg70_1: "f32[2048, 2048]", arg71_1: "f32[2048, 2048]", arg72_1: "f32[2048, 2048]", arg73_1: "f32[2048]", arg74_1: "f32[2048]", arg75_1: "f32[2048]", arg76_1: "f32[8192, 2048]", arg77_1: "f32[8192]", arg78_1: "f32[2048, 8192]", arg79_1: "f32[2048]", arg80_1: "f32[2048]", arg81_1: "f32[2048]", arg82_1: "f32[2048, 2048]", arg83_1: "f32[2048, 2048]", arg84_1: "f32[2048, 2048]", arg85_1: "f32[2048, 2048]", arg86_1: "f32[2048]", arg87_1: "f32[2048]", arg88_1: "f32[2048]", arg89_1: "f32[8192, 2048]", arg90_1: "f32[8192]", arg91_1: "f32[2048, 8192]", arg92_1: "f32[2048]", arg93_1: "f32[2048]", arg94_1: "f32[2048]", arg95_1: "f32[2048, 2048]", arg96_1: "f32[2048, 2048]", arg97_1: "f32[2048, 2048]", arg98_1: "f32[2048, 2048]", arg99_1: "f32[2048]", arg100_1: "f32[2048]", arg101_1: "f32[2048]", arg102_1: "f32[8192, 2048]", arg103_1: "f32[8192]", arg104_1: "f32[2048, 8192]", arg105_1: "f32[2048]", arg106_1: "f32[2048]", arg107_1: "f32[2048]", arg108_1: "f32[2048, 2048]", arg109_1: "f32[2048, 2048]", arg110_1: "f32[2048, 2048]", arg111_1: "f32[2048, 2048]", arg112_1: "f32[2048]", arg113_1: "f32[2048]", arg114_1: "f32[2048]", arg115_1: "f32[8192, 2048]", arg116_1: "f32[8192]", arg117_1: "f32[2048, 8192]", arg118_1: "f32[2048]", arg119_1: "f32[2048]", arg120_1: "f32[2048]", arg121_1: "f32[2048, 2048]", arg122_1: "f32[2048, 2048]", arg123_1: "f32[2048, 2048]", arg124_1: "f32[2048, 2048]", arg125_1: "f32[2048]", arg126_1: "f32[2048]", arg127_1: "f32[2048]", arg128_1: "f32[8192, 2048]", arg129_1: "f32[8192]", arg130_1: "f32[2048, 8192]", arg131_1: "f32[2048]", arg132_1: "f32[2048]", arg133_1: "f32[2048]", arg134_1: "f32[2048, 2048]", arg135_1: "f32[2048, 2048]", arg136_1: "f32[2048, 2048]", arg137_1: "f32[2048, 2048]", arg138_1: "f32[2048]", arg139_1: "f32[2048]", arg140_1: "f32[2048]", arg141_1: "f32[8192, 2048]", arg142_1: "f32[8192]", arg143_1: "f32[2048, 8192]", arg144_1: "f32[2048]", arg145_1: "f32[2048]", arg146_1: "f32[2048]", arg147_1: "f32[2048, 2048]", arg148_1: "f32[2048, 2048]", arg149_1: "f32[2048, 2048]", arg150_1: "f32[2048, 2048]", arg151_1: "f32[2048]", arg152_1: "f32[2048]", arg153_1: "f32[2048]", arg154_1: "f32[8192, 2048]", arg155_1: "f32[8192]", arg156_1: "f32[2048, 8192]", arg157_1: "f32[2048]", arg158_1: "f32[2048]", arg159_1: "f32[2048]", arg160_1: "f32[2048, 2048]", arg161_1: "f32[2048, 2048]", arg162_1: "f32[2048, 2048]", arg163_1: "f32[2048, 2048]", arg164_1: "f32[2048]", arg165_1: "f32[2048]", arg166_1: "f32[2048]", arg167_1: "f32[8192, 2048]", arg168_1: "f32[8192]", arg169_1: "f32[2048, 8192]", arg170_1: "f32[2048]", arg171_1: "f32[2048]", arg172_1: "f32[2048]", arg173_1: "f32[2048, 2048]", arg174_1: "f32[2048, 2048]", arg175_1: "f32[2048, 2048]", arg176_1: "f32[2048, 2048]", arg177_1: "f32[2048]", arg178_1: "f32[2048]", arg179_1: "f32[2048]", arg180_1: "f32[8192, 2048]", arg181_1: "f32[8192]", arg182_1: "f32[2048, 8192]", arg183_1: "f32[2048]", arg184_1: "f32[2048]", arg185_1: "f32[2048]", arg186_1: "f32[2048, 2048]", arg187_1: "f32[2048, 2048]", arg188_1: "f32[2048, 2048]", arg189_1: "f32[2048, 2048]", arg190_1: "f32[2048]", arg191_1: "f32[2048]", arg192_1: "f32[2048]", arg193_1: "f32[8192, 2048]", arg194_1: "f32[8192]", arg195_1: "f32[2048, 8192]", arg196_1: "f32[2048]", arg197_1: "f32[2048]", arg198_1: "f32[2048]", arg199_1: "f32[2048, 2048]", arg200_1: "f32[2048, 2048]", arg201_1: "f32[2048, 2048]", arg202_1: "f32[2048, 2048]", arg203_1: "f32[2048]", arg204_1: "f32[2048]", arg205_1: "f32[2048]", arg206_1: "f32[8192, 2048]", arg207_1: "f32[8192]", arg208_1: "f32[2048, 8192]", arg209_1: "f32[2048]", arg210_1: "f32[2048]", arg211_1: "f32[2048]", arg212_1: "f32[2048, 2048]", arg213_1: "f32[2048, 2048]", arg214_1: "f32[2048, 2048]", arg215_1: "f32[2048, 2048]", arg216_1: "f32[2048]", arg217_1: "f32[2048]", arg218_1: "f32[2048]", arg219_1: "f32[8192, 2048]", arg220_1: "f32[8192]", arg221_1: "f32[2048, 8192]", arg222_1: "f32[2048]", arg223_1: "f32[2048]", arg224_1: "f32[2048]", arg225_1: "f32[2048, 2048]", arg226_1: "f32[2048, 2048]", arg227_1: "f32[2048, 2048]", arg228_1: "f32[2048, 2048]", arg229_1: "f32[2048]", arg230_1: "f32[2048]", arg231_1: "f32[2048]", arg232_1: "f32[8192, 2048]", arg233_1: "f32[8192]", arg234_1: "f32[2048, 8192]", arg235_1: "f32[2048]", arg236_1: "f32[2048]", arg237_1: "f32[2048]", arg238_1: "f32[2048, 2048]", arg239_1: "f32[2048, 2048]", arg240_1: "f32[2048, 2048]", arg241_1: "f32[2048, 2048]", arg242_1: "f32[2048]", arg243_1: "f32[2048]", arg244_1: "f32[2048]", arg245_1: "f32[8192, 2048]", arg246_1: "f32[8192]", arg247_1: "f32[2048, 8192]", arg248_1: "f32[2048]", arg249_1: "f32[2048]", arg250_1: "f32[2048]", arg251_1: "f32[2048, 2048]", arg252_1: "f32[2048, 2048]", arg253_1: "f32[2048, 2048]", arg254_1: "f32[2048, 2048]", arg255_1: "f32[2048]", arg256_1: "f32[2048]", arg257_1: "f32[2048]", arg258_1: "f32[8192, 2048]", arg259_1: "f32[8192]", arg260_1: "f32[2048, 8192]", arg261_1: "f32[2048]", arg262_1: "f32[2048]", arg263_1: "f32[2048]", arg264_1: "f32[2048, 2048]", arg265_1: "f32[2048, 2048]", arg266_1: "f32[2048, 2048]", arg267_1: "f32[2048, 2048]", arg268_1: "f32[2048]", arg269_1: "f32[2048]", arg270_1: "f32[2048]", arg271_1: "f32[8192, 2048]", arg272_1: "f32[8192]", arg273_1: "f32[2048, 8192]", arg274_1: "f32[2048]", arg275_1: "f32[2048]", arg276_1: "f32[2048]", arg277_1: "f32[2048, 2048]", arg278_1: "f32[2048, 2048]", arg279_1: "f32[2048, 2048]", arg280_1: "f32[2048, 2048]", arg281_1: "f32[2048]", arg282_1: "f32[2048]", arg283_1: "f32[2048]", arg284_1: "f32[8192, 2048]", arg285_1: "f32[8192]", arg286_1: "f32[2048, 8192]", arg287_1: "f32[2048]", arg288_1: "f32[2048]", arg289_1: "f32[2048]", arg290_1: "f32[2048, 2048]", arg291_1: "f32[2048, 2048]", arg292_1: "f32[2048, 2048]", arg293_1: "f32[2048, 2048]", arg294_1: "f32[2048]", arg295_1: "f32[2048]", arg296_1: "f32[2048]", arg297_1: "f32[8192, 2048]", arg298_1: "f32[8192]", arg299_1: "f32[2048, 8192]", arg300_1: "f32[2048]", arg301_1: "f32[2048]", arg302_1: "f32[2048]", arg303_1: "f32[2048, 2048]", arg304_1: "f32[2048, 2048]", arg305_1: "f32[2048, 2048]", arg306_1: "f32[2048, 2048]", arg307_1: "f32[2048]", arg308_1: "f32[2048]", arg309_1: "f32[2048]", arg310_1: "f32[8192, 2048]", arg311_1: "f32[8192]", arg312_1: "f32[2048, 8192]", arg313_1: "f32[2048]", arg314_1: "f32[2048]", arg315_1: "f32[2048]", arg316_1: "f32[50257, 2048]", arg317_1: "b8[1, 1, 2048, 2048]", arg318_1: "b8[1, 1, 2048, 2048]", arg319_1: "b8[1, 1, 2048, 2048]", arg320_1: "b8[1, 1, 2048, 2048]", arg321_1: "b8[1, 1, 2048, 2048]", arg322_1: "b8[1, 1, 2048, 2048]", arg323_1: "b8[1, 1, 2048, 2048]", arg324_1: "b8[1, 1, 2048, 2048]", arg325_1: "b8[1, 1, 2048, 2048]", arg326_1: "b8[1, 1, 2048, 2048]", arg327_1: "b8[1, 1, 2048, 2048]", arg328_1: "b8[1, 1, 2048, 2048]", arg329_1: "b8[1, 1, 2048, 2048]", arg330_1: "b8[1, 1, 2048, 2048]", arg331_1: "b8[1, 1, 2048, 2048]", arg332_1: "b8[1, 1, 2048, 2048]", arg333_1: "b8[1, 1, 2048, 2048]", arg334_1: "b8[1, 1, 2048, 2048]", arg335_1: "b8[1, 1, 2048, 2048]", arg336_1: "b8[1, 1, 2048, 2048]", arg337_1: "b8[1, 1, 2048, 2048]", arg338_1: "b8[1, 1, 2048, 2048]", arg339_1: "b8[1, 1, 2048, 2048]", arg340_1: "b8[1, 1, 2048, 2048]", arg341_1: "i64[1, 128]", arg342_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:530, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.reshape.default(arg341_1, [-1, 128]);  arg341_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:582, code: inputs_embeds = self.wte(input_ids)
    embedding: "f32[1, 128, 2048]" = torch.ops.aten.embedding.default(arg0_1, view);  arg0_1 = view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:552, code: position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:553, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    view_1: "i64[1, 128]" = torch.ops.aten.reshape.default(unsqueeze, [-1, 128]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:583, code: position_embeds = self.wpe(position_ids)
    embedding_1: "f32[1, 128, 2048]" = torch.ops.aten.embedding.default(arg1_1, view_1);  arg1_1 = view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:584, code: hidden_states = inputs_embeds + position_embeds
    add: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant0: "f32[]" = self._tensor_constant0
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_3: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg317_1, 2, 0, 128);  arg317_1 = None
    slice_4: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 128);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
    add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    mul: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul, arg2_1);  mul = arg2_1 = None
    add_2: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_2: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_2, [128, 2048])
    permute: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    mm: "f32[128, 2048]" = torch.ops.aten.mm.default(view_2, permute);  view_2 = permute = None
    view_3: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm, [1, 128, 2048]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_8: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_3, [1, 128, 16, 128]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_3: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_3, [1, 16, 128, 128]);  permute_3 = None
    view_11: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand, [16, 128, 128]);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_4: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_2, [128, 2048])
    permute_1: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    mm_1: "f32[128, 2048]" = torch.ops.aten.mm.default(view_4, permute_1);  view_4 = permute_1 = None
    view_5: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_1, [1, 128, 2048]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_9: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_5, [1, 128, 16, 128]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_4: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_6: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2])
    expand_1: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_6, [1, 16, 128, 128]);  permute_6 = None
    view_12: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_1, [16, 128, 128]);  expand_1 = None
    bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_11, view_12);  view_11 = view_12 = None
    view_13: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm, [1, 16, 128, 128]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_4, view_13, full_default);  slice_4 = view_13 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_2: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div, [1, 16, 128, 128]);  div = None
    view_14: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_2, [16, 128, 128]);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_6: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_2, [128, 2048]);  add_2 = None
    permute_2: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    mm_2: "f32[128, 2048]" = torch.ops.aten.mm.default(view_6, permute_2);  view_6 = permute_2 = None
    view_7: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_2, [1, 128, 2048]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_10: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_7, [1, 128, 16, 128]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_5: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_3: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_5, [1, 16, 128, 128])
    view_15: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_3, [16, 128, 128]);  expand_3 = None
    bmm_1: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_14, view_15);  view_14 = view_15 = None
    view_16: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_1, [1, 16, 128, 128]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_7: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    clone_2: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_17: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_2, [1, 128, 2048]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_18: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_17, [128, 2048]);  view_17 = None
    permute_8: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    addmm: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg8_1, view_18, permute_8);  arg8_1 = view_18 = permute_8 = None
    view_19: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm, [1, 128, 2048]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_3: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_19, add);  view_19 = add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_2: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  getitem_3 = None
    add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    mul_2: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_3: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
    add_5: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_3, arg10_1);  mul_3 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_20: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_5, [128, 2048]);  add_5 = None
    permute_9: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    addmm_1: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg12_1, view_20, permute_9);  arg12_1 = view_20 = permute_9 = None
    view_21: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_1, [1, 128, 8192]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_4: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_21, 0.5)
    pow_1: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_21, 3.0)
    mul_5: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
    add_6: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_21, mul_5);  view_21 = mul_5 = None
    mul_6: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_6, 0.7978845608028654);  add_6 = None
    tanh: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_6);  mul_6 = None
    add_7: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh, 1.0);  tanh = None
    mul_7: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_4, add_7);  mul_4 = add_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_22: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_7, [128, 8192]);  mul_7 = None
    permute_10: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
    addmm_2: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg14_1, view_22, permute_10);  arg14_1 = view_22 = permute_10 = None
    view_23: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_2, [1, 128, 2048]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_8: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_3, view_23);  add_3 = view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant1: "f32[]" = self._tensor_constant1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_7: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg318_1, 2, 0, 128);  arg318_1 = None
    slice_8: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_7, 3, 0, 128);  slice_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_3: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_8, getitem_5);  getitem_5 = None
    add_9: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
    mul_8: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_9: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_8, arg15_1);  mul_8 = arg15_1 = None
    add_10: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_9, arg16_1);  mul_9 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_24: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_10, [128, 2048])
    permute_11: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    mm_3: "f32[128, 2048]" = torch.ops.aten.mm.default(view_24, permute_11);  view_24 = permute_11 = None
    view_25: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_3, [1, 128, 2048]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_30: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_25, [1, 128, 16, 128]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_14: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_4: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_14, [1, 16, 128, 128]);  permute_14 = None
    view_33: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_4, [16, 128, 128]);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_26: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_10, [128, 2048])
    permute_12: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
    mm_4: "f32[128, 2048]" = torch.ops.aten.mm.default(view_26, permute_12);  view_26 = permute_12 = None
    view_27: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_4, [1, 128, 2048]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_31: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_27, [1, 128, 16, 128]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_15: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_31, [0, 2, 1, 3]);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_17: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2])
    expand_5: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_17, [1, 16, 128, 128]);  permute_17 = None
    view_34: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_5, [16, 128, 128]);  expand_5 = None
    bmm_2: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_33, view_34);  view_33 = view_34 = None
    view_35: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [1, 16, 128, 128]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_1: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_8, view_35, full_default_1);  slice_8 = view_35 = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_6: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_1, [1, 16, 128, 128]);  div_1 = None
    view_36: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_6, [16, 128, 128]);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_28: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_10, [128, 2048]);  add_10 = None
    permute_13: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
    mm_5: "f32[128, 2048]" = torch.ops.aten.mm.default(view_28, permute_13);  view_28 = permute_13 = None
    view_29: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_5, [1, 128, 2048]);  mm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_32: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_29, [1, 128, 16, 128]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_16: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_7: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_16, [1, 16, 128, 128])
    view_37: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_7, [16, 128, 128]);  expand_7 = None
    bmm_3: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_36, view_37);  view_36 = view_37 = None
    view_38: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_3, [1, 16, 128, 128]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_18: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_6: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_39: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_6, [1, 128, 2048]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_40: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_39, [128, 2048]);  view_39 = None
    permute_19: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg20_1, [1, 0]);  arg20_1 = None
    addmm_3: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg21_1, view_40, permute_19);  arg21_1 = view_40 = permute_19 = None
    view_41: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_3, [1, 128, 2048]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_11: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_41, add_8);  view_41 = add_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_5: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_11, getitem_7);  getitem_7 = None
    add_12: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    mul_10: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_11: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_10, arg22_1);  mul_10 = arg22_1 = None
    add_13: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_11, arg23_1);  mul_11 = arg23_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_42: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_13, [128, 2048]);  add_13 = None
    permute_20: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    addmm_4: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg25_1, view_42, permute_20);  arg25_1 = view_42 = permute_20 = None
    view_43: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_4, [1, 128, 8192]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_12: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    pow_2: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_43, 3.0)
    mul_13: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
    add_14: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_43, mul_13);  view_43 = mul_13 = None
    mul_14: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_14, 0.7978845608028654);  add_14 = None
    tanh_1: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
    add_15: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_1, 1.0);  tanh_1 = None
    mul_15: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_12, add_15);  mul_12 = add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_44: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_15, [128, 8192]);  mul_15 = None
    permute_21: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
    addmm_5: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg27_1, view_44, permute_21);  arg27_1 = view_44 = permute_21 = None
    view_45: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_5, [1, 128, 2048]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_16: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_11, view_45);  add_11 = view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant2: "f32[]" = self._tensor_constant2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_11: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg319_1, 2, 0, 128);  arg319_1 = None
    slice_12: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_11, 3, 0, 128);  slice_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_6: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_16, getitem_9);  getitem_9 = None
    add_17: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    mul_16: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_17: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_16, arg28_1);  mul_16 = arg28_1 = None
    add_18: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_17, arg29_1);  mul_17 = arg29_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_46: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_18, [128, 2048])
    permute_22: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    mm_6: "f32[128, 2048]" = torch.ops.aten.mm.default(view_46, permute_22);  view_46 = permute_22 = None
    view_47: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_6, [1, 128, 2048]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_52: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_47, [1, 128, 16, 128]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_25: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_8: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_25, [1, 16, 128, 128]);  permute_25 = None
    view_55: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_8, [16, 128, 128]);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_48: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_18, [128, 2048])
    permute_23: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
    mm_7: "f32[128, 2048]" = torch.ops.aten.mm.default(view_48, permute_23);  view_48 = permute_23 = None
    view_49: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_7, [1, 128, 2048]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_53: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_49, [1, 128, 16, 128]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_26: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_28: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2])
    expand_9: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_28, [1, 16, 128, 128]);  permute_28 = None
    view_56: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_9, [16, 128, 128]);  expand_9 = None
    bmm_4: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_55, view_56);  view_55 = view_56 = None
    view_57: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [1, 16, 128, 128]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_2: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_2: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_12, view_57, full_default_2);  slice_12 = view_57 = full_default_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_10: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_2, [1, 16, 128, 128]);  div_2 = None
    view_58: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_10, [16, 128, 128]);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_50: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_18, [128, 2048]);  add_18 = None
    permute_24: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
    mm_8: "f32[128, 2048]" = torch.ops.aten.mm.default(view_50, permute_24);  view_50 = permute_24 = None
    view_51: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_8, [1, 128, 2048]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_54: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_51, [1, 128, 16, 128]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_27: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_11: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_27, [1, 16, 128, 128])
    view_59: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_11, [16, 128, 128]);  expand_11 = None
    bmm_5: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_58, view_59);  view_58 = view_59 = None
    view_60: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_5, [1, 16, 128, 128]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_29: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    clone_10: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_61: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_10, [1, 128, 2048]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_62: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_61, [128, 2048]);  view_61 = None
    permute_30: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    addmm_6: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg34_1, view_62, permute_30);  arg34_1 = view_62 = permute_30 = None
    view_63: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_6, [1, 128, 2048]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_19: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_63, add_16);  view_63 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_8: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_19, getitem_11);  getitem_11 = None
    add_20: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    mul_18: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_19: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_18, arg35_1);  mul_18 = arg35_1 = None
    add_21: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_19, arg36_1);  mul_19 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_64: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_21, [128, 2048]);  add_21 = None
    permute_31: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    addmm_7: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg38_1, view_64, permute_31);  arg38_1 = view_64 = permute_31 = None
    view_65: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_7, [1, 128, 8192]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_20: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_65, 0.5)
    pow_3: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_65, 3.0)
    mul_21: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_22: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_65, mul_21);  view_65 = mul_21 = None
    mul_22: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
    tanh_2: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
    add_23: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_2, 1.0);  tanh_2 = None
    mul_23: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_20, add_23);  mul_20 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_66: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_23, [128, 8192]);  mul_23 = None
    permute_32: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    addmm_8: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg40_1, view_66, permute_32);  arg40_1 = view_66 = permute_32 = None
    view_67: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_8, [1, 128, 2048]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_24: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_19, view_67);  add_19 = view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant3: "f32[]" = self._tensor_constant3
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_15: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg320_1, 2, 0, 128);  arg320_1 = None
    slice_16: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_15, 3, 0, 128);  slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_9: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_24, getitem_13);  getitem_13 = None
    add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    mul_24: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_25: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_24, arg41_1);  mul_24 = arg41_1 = None
    add_26: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_25, arg42_1);  mul_25 = arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_68: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_26, [128, 2048])
    permute_33: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    mm_9: "f32[128, 2048]" = torch.ops.aten.mm.default(view_68, permute_33);  view_68 = permute_33 = None
    view_69: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_9, [1, 128, 2048]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_74: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_69, [1, 128, 16, 128]);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_36: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_12: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_36, [1, 16, 128, 128]);  permute_36 = None
    view_77: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_12, [16, 128, 128]);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_70: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_26, [128, 2048])
    permute_34: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg44_1, [1, 0]);  arg44_1 = None
    mm_10: "f32[128, 2048]" = torch.ops.aten.mm.default(view_70, permute_34);  view_70 = permute_34 = None
    view_71: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_10, [1, 128, 2048]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_75: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_71, [1, 128, 16, 128]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_37: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_39: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_37, [0, 1, 3, 2])
    expand_13: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_39, [1, 16, 128, 128]);  permute_39 = None
    view_78: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_13, [16, 128, 128]);  expand_13 = None
    bmm_6: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_77, view_78);  view_77 = view_78 = None
    view_79: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [1, 16, 128, 128]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_3: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_3: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_16, view_79, full_default_3);  slice_16 = view_79 = full_default_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_14: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_3, [1, 16, 128, 128]);  div_3 = None
    view_80: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_14, [16, 128, 128]);  expand_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_72: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_26, [128, 2048]);  add_26 = None
    permute_35: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
    mm_11: "f32[128, 2048]" = torch.ops.aten.mm.default(view_72, permute_35);  view_72 = permute_35 = None
    view_73: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_11, [1, 128, 2048]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_76: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_73, [1, 128, 16, 128]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_38: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_76, [0, 2, 1, 3]);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_15: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_38, [1, 16, 128, 128])
    view_81: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_15, [16, 128, 128]);  expand_15 = None
    bmm_7: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_80, view_81);  view_80 = view_81 = None
    view_82: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_7, [1, 16, 128, 128]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_40: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    clone_14: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_83: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_14, [1, 128, 2048]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_84: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_83, [128, 2048]);  view_83 = None
    permute_41: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    addmm_9: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg47_1, view_84, permute_41);  arg47_1 = view_84 = permute_41 = None
    view_85: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_9, [1, 128, 2048]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_27: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_85, add_24);  view_85 = add_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_11: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_27, getitem_15);  getitem_15 = None
    add_28: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    mul_26: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_27: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_26, arg48_1);  mul_26 = arg48_1 = None
    add_29: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_27, arg49_1);  mul_27 = arg49_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_86: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_29, [128, 2048]);  add_29 = None
    permute_42: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
    addmm_10: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg51_1, view_86, permute_42);  arg51_1 = view_86 = permute_42 = None
    view_87: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_10, [1, 128, 8192]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_28: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_87, 0.5)
    pow_4: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_87, 3.0)
    mul_29: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
    add_30: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_87, mul_29);  view_87 = mul_29 = None
    mul_30: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_30, 0.7978845608028654);  add_30 = None
    tanh_3: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_30);  mul_30 = None
    add_31: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_3, 1.0);  tanh_3 = None
    mul_31: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_28, add_31);  mul_28 = add_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_88: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_31, [128, 8192]);  mul_31 = None
    permute_43: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    addmm_11: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg53_1, view_88, permute_43);  arg53_1 = view_88 = permute_43 = None
    view_89: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_11, [1, 128, 2048]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_32: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_27, view_89);  add_27 = view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant4: "f32[]" = self._tensor_constant4
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_19: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg321_1, 2, 0, 128);  arg321_1 = None
    slice_20: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_19, 3, 0, 128);  slice_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_12: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_32, getitem_17);  getitem_17 = None
    add_33: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    mul_32: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_33: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_32, arg54_1);  mul_32 = arg54_1 = None
    add_34: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_33, arg55_1);  mul_33 = arg55_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_90: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_34, [128, 2048])
    permute_44: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    mm_12: "f32[128, 2048]" = torch.ops.aten.mm.default(view_90, permute_44);  view_90 = permute_44 = None
    view_91: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_12, [1, 128, 2048]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_96: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_91, [1, 128, 16, 128]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_47: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_16: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_47, [1, 16, 128, 128]);  permute_47 = None
    view_99: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_16, [16, 128, 128]);  expand_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_92: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_34, [128, 2048])
    permute_45: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    mm_13: "f32[128, 2048]" = torch.ops.aten.mm.default(view_92, permute_45);  view_92 = permute_45 = None
    view_93: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_13, [1, 128, 2048]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_97: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_93, [1, 128, 16, 128]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_48: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_50: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_48, [0, 1, 3, 2])
    expand_17: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_50, [1, 16, 128, 128]);  permute_50 = None
    view_100: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_17, [16, 128, 128]);  expand_17 = None
    bmm_8: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_99, view_100);  view_99 = view_100 = None
    view_101: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [1, 16, 128, 128]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_4: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_4: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_20, view_101, full_default_4);  slice_20 = view_101 = full_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_18: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_4, [1, 16, 128, 128]);  div_4 = None
    view_102: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_18, [16, 128, 128]);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_94: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_34, [128, 2048]);  add_34 = None
    permute_46: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    mm_14: "f32[128, 2048]" = torch.ops.aten.mm.default(view_94, permute_46);  view_94 = permute_46 = None
    view_95: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_14, [1, 128, 2048]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_98: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_95, [1, 128, 16, 128]);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_49: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_19: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_49, [1, 16, 128, 128])
    view_103: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_19, [16, 128, 128]);  expand_19 = None
    bmm_9: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_102, view_103);  view_102 = view_103 = None
    view_104: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_9, [1, 16, 128, 128]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_51: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    clone_18: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_105: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_18, [1, 128, 2048]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_106: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_105, [128, 2048]);  view_105 = None
    permute_52: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    addmm_12: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg60_1, view_106, permute_52);  arg60_1 = view_106 = permute_52 = None
    view_107: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_12, [1, 128, 2048]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_35: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_107, add_32);  view_107 = add_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_14: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_35, getitem_19);  getitem_19 = None
    add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    mul_34: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_35: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_34, arg61_1);  mul_34 = arg61_1 = None
    add_37: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_35, arg62_1);  mul_35 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_108: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_37, [128, 2048]);  add_37 = None
    permute_53: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    addmm_13: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg64_1, view_108, permute_53);  arg64_1 = view_108 = permute_53 = None
    view_109: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_13, [1, 128, 8192]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_36: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
    pow_5: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_109, 3.0)
    mul_37: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
    add_38: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_109, mul_37);  view_109 = mul_37 = None
    mul_38: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_38, 0.7978845608028654);  add_38 = None
    tanh_4: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_38);  mul_38 = None
    add_39: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_4, 1.0);  tanh_4 = None
    mul_39: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_36, add_39);  mul_36 = add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_110: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_39, [128, 8192]);  mul_39 = None
    permute_54: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    addmm_14: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg66_1, view_110, permute_54);  arg66_1 = view_110 = permute_54 = None
    view_111: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_14, [1, 128, 2048]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_40: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_35, view_111);  add_35 = view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant5: "f32[]" = self._tensor_constant5
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_23: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg322_1, 2, 0, 128);  arg322_1 = None
    slice_24: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_23, 3, 0, 128);  slice_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_15: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_40, getitem_21);  getitem_21 = None
    add_41: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
    mul_40: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_41: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_40, arg67_1);  mul_40 = arg67_1 = None
    add_42: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_41, arg68_1);  mul_41 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_112: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_42, [128, 2048])
    permute_55: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    mm_15: "f32[128, 2048]" = torch.ops.aten.mm.default(view_112, permute_55);  view_112 = permute_55 = None
    view_113: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_15, [1, 128, 2048]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_118: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_113, [1, 128, 16, 128]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_58: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_20: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_58, [1, 16, 128, 128]);  permute_58 = None
    view_121: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_20, [16, 128, 128]);  expand_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_114: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_42, [128, 2048])
    permute_56: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    mm_16: "f32[128, 2048]" = torch.ops.aten.mm.default(view_114, permute_56);  view_114 = permute_56 = None
    view_115: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_16, [1, 128, 2048]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_119: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_115, [1, 128, 16, 128]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_59: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_61: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_59, [0, 1, 3, 2])
    expand_21: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_61, [1, 16, 128, 128]);  permute_61 = None
    view_122: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_21, [16, 128, 128]);  expand_21 = None
    bmm_10: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_121, view_122);  view_121 = view_122 = None
    view_123: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [1, 16, 128, 128]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_5: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_5: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_24, view_123, full_default_5);  slice_24 = view_123 = full_default_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_22: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_5, [1, 16, 128, 128]);  div_5 = None
    view_124: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_22, [16, 128, 128]);  expand_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_116: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_42, [128, 2048]);  add_42 = None
    permute_57: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    mm_17: "f32[128, 2048]" = torch.ops.aten.mm.default(view_116, permute_57);  view_116 = permute_57 = None
    view_117: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_17, [1, 128, 2048]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_120: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_117, [1, 128, 16, 128]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_60: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_23: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_60, [1, 16, 128, 128])
    view_125: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_23, [16, 128, 128]);  expand_23 = None
    bmm_11: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_124, view_125);  view_124 = view_125 = None
    view_126: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_11, [1, 16, 128, 128]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_62: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    clone_22: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_127: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_22, [1, 128, 2048]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_128: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_127, [128, 2048]);  view_127 = None
    permute_63: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_15: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg73_1, view_128, permute_63);  arg73_1 = view_128 = permute_63 = None
    view_129: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_15, [1, 128, 2048]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_43: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_129, add_40);  view_129 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_17: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_43, getitem_23);  getitem_23 = None
    add_44: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    mul_42: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_43: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_42, arg74_1);  mul_42 = arg74_1 = None
    add_45: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_43, arg75_1);  mul_43 = arg75_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_130: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_45, [128, 2048]);  add_45 = None
    permute_64: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    addmm_16: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg77_1, view_130, permute_64);  arg77_1 = view_130 = permute_64 = None
    view_131: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_16, [1, 128, 8192]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_44: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
    pow_6: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 3.0)
    mul_45: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_46: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_131, mul_45);  view_131 = mul_45 = None
    mul_46: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
    tanh_5: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
    add_47: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_5, 1.0);  tanh_5 = None
    mul_47: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_44, add_47);  mul_44 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_132: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_47, [128, 8192]);  mul_47 = None
    permute_65: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_17: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg79_1, view_132, permute_65);  arg79_1 = view_132 = permute_65 = None
    view_133: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_17, [1, 128, 2048]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_48: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_43, view_133);  add_43 = view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant6: "f32[]" = self._tensor_constant6
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_27: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg323_1, 2, 0, 128);  arg323_1 = None
    slice_28: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_27, 3, 0, 128);  slice_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_18: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_48, getitem_25);  getitem_25 = None
    add_49: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    mul_48: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_49: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_48, arg80_1);  mul_48 = arg80_1 = None
    add_50: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_49, arg81_1);  mul_49 = arg81_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_134: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_50, [128, 2048])
    permute_66: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    mm_18: "f32[128, 2048]" = torch.ops.aten.mm.default(view_134, permute_66);  view_134 = permute_66 = None
    view_135: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_18, [1, 128, 2048]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_140: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_135, [1, 128, 16, 128]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_69: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_24: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_69, [1, 16, 128, 128]);  permute_69 = None
    view_143: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_24, [16, 128, 128]);  expand_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_136: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_50, [128, 2048])
    permute_67: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    mm_19: "f32[128, 2048]" = torch.ops.aten.mm.default(view_136, permute_67);  view_136 = permute_67 = None
    view_137: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_19, [1, 128, 2048]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_141: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_137, [1, 128, 16, 128]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_70: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_141, [0, 2, 1, 3]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_72: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_70, [0, 1, 3, 2])
    expand_25: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_72, [1, 16, 128, 128]);  permute_72 = None
    view_144: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_25, [16, 128, 128]);  expand_25 = None
    bmm_12: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_143, view_144);  view_143 = view_144 = None
    view_145: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [1, 16, 128, 128]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_6: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_6: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_28, view_145, full_default_6);  slice_28 = view_145 = full_default_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_19: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_6, amax_6);  where_6 = amax_6 = None
    exp_6: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_26: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_6, [1, 16, 128, 128]);  div_6 = None
    view_146: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_26, [16, 128, 128]);  expand_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_138: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_50, [128, 2048]);  add_50 = None
    permute_68: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    mm_20: "f32[128, 2048]" = torch.ops.aten.mm.default(view_138, permute_68);  view_138 = permute_68 = None
    view_139: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_20, [1, 128, 2048]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_142: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_139, [1, 128, 16, 128]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_71: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_27: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_71, [1, 16, 128, 128])
    view_147: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_27, [16, 128, 128]);  expand_27 = None
    bmm_13: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_146, view_147);  view_146 = view_147 = None
    view_148: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_13, [1, 16, 128, 128]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_73: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_26: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_149: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_26, [1, 128, 2048]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_150: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_149, [128, 2048]);  view_149 = None
    permute_74: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    addmm_18: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg86_1, view_150, permute_74);  arg86_1 = view_150 = permute_74 = None
    view_151: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_18, [1, 128, 2048]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_51: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_151, add_48);  view_151 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_20: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_51, getitem_27);  getitem_27 = None
    add_52: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    mul_50: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_51: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_50, arg87_1);  mul_50 = arg87_1 = None
    add_53: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_51, arg88_1);  mul_51 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_152: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_53, [128, 2048]);  add_53 = None
    permute_75: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    addmm_19: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg90_1, view_152, permute_75);  arg90_1 = view_152 = permute_75 = None
    view_153: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_19, [1, 128, 8192]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_153, 0.5)
    pow_7: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_153, 3.0)
    mul_53: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
    add_54: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_153, mul_53);  view_153 = mul_53 = None
    mul_54: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_54, 0.7978845608028654);  add_54 = None
    tanh_6: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
    add_55: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_6, 1.0);  tanh_6 = None
    mul_55: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_52, add_55);  mul_52 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_154: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_55, [128, 8192]);  mul_55 = None
    permute_76: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    addmm_20: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg92_1, view_154, permute_76);  arg92_1 = view_154 = permute_76 = None
    view_155: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_20, [1, 128, 2048]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_56: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_51, view_155);  add_51 = view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant7: "f32[]" = self._tensor_constant7
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_31: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg324_1, 2, 0, 128);  arg324_1 = None
    slice_32: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_31, 3, 0, 128);  slice_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_21: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_56, getitem_29);  getitem_29 = None
    add_57: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    mul_56: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_57: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_56, arg93_1);  mul_56 = arg93_1 = None
    add_58: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_57, arg94_1);  mul_57 = arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_156: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_58, [128, 2048])
    permute_77: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    mm_21: "f32[128, 2048]" = torch.ops.aten.mm.default(view_156, permute_77);  view_156 = permute_77 = None
    view_157: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_21, [1, 128, 2048]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_162: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_157, [1, 128, 16, 128]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_80: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_28: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_80, [1, 16, 128, 128]);  permute_80 = None
    view_165: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_28, [16, 128, 128]);  expand_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_158: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_58, [128, 2048])
    permute_78: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    mm_22: "f32[128, 2048]" = torch.ops.aten.mm.default(view_158, permute_78);  view_158 = permute_78 = None
    view_159: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_22, [1, 128, 2048]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_163: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_159, [1, 128, 16, 128]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_81: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_163, [0, 2, 1, 3]);  view_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_83: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_81, [0, 1, 3, 2])
    expand_29: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_83, [1, 16, 128, 128]);  permute_83 = None
    view_166: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_29, [16, 128, 128]);  expand_29 = None
    bmm_14: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_165, view_166);  view_165 = view_166 = None
    view_167: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, [1, 16, 128, 128]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_7: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_7: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_32, view_167, full_default_7);  slice_32 = view_167 = full_default_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_7, [-1], True)
    sub_22: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_7, amax_7);  where_7 = amax_7 = None
    exp_7: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_30: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_7, [1, 16, 128, 128]);  div_7 = None
    view_168: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_30, [16, 128, 128]);  expand_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_160: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_58, [128, 2048]);  add_58 = None
    permute_79: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    mm_23: "f32[128, 2048]" = torch.ops.aten.mm.default(view_160, permute_79);  view_160 = permute_79 = None
    view_161: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_23, [1, 128, 2048]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_164: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_161, [1, 128, 16, 128]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_82: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_164, [0, 2, 1, 3]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_31: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_82, [1, 16, 128, 128])
    view_169: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_31, [16, 128, 128]);  expand_31 = None
    bmm_15: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_168, view_169);  view_168 = view_169 = None
    view_170: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_15, [1, 16, 128, 128]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_84: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_30: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_171: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_30, [1, 128, 2048]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_172: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_171, [128, 2048]);  view_171 = None
    permute_85: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    addmm_21: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg99_1, view_172, permute_85);  arg99_1 = view_172 = permute_85 = None
    view_173: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_21, [1, 128, 2048]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_59: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_173, add_56);  view_173 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_23: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_59, getitem_31);  getitem_31 = None
    add_60: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    mul_58: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_59: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_58, arg100_1);  mul_58 = arg100_1 = None
    add_61: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_59, arg101_1);  mul_59 = arg101_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_174: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_61, [128, 2048]);  add_61 = None
    permute_86: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_22: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg103_1, view_174, permute_86);  arg103_1 = view_174 = permute_86 = None
    view_175: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_22, [1, 128, 8192]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_60: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_175, 0.5)
    pow_8: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_175, 3.0)
    mul_61: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
    add_62: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_175, mul_61);  view_175 = mul_61 = None
    mul_62: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_62, 0.7978845608028654);  add_62 = None
    tanh_7: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_62);  mul_62 = None
    add_63: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_7, 1.0);  tanh_7 = None
    mul_63: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_176: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_63, [128, 8192]);  mul_63 = None
    permute_87: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_23: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg105_1, view_176, permute_87);  arg105_1 = view_176 = permute_87 = None
    view_177: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_23, [1, 128, 2048]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_64: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_59, view_177);  add_59 = view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant8: "f32[]" = self._tensor_constant8
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_35: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg325_1, 2, 0, 128);  arg325_1 = None
    slice_36: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_35, 3, 0, 128);  slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_24: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_64, getitem_33);  getitem_33 = None
    add_65: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    mul_64: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_65: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_64, arg106_1);  mul_64 = arg106_1 = None
    add_66: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_65, arg107_1);  mul_65 = arg107_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_178: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_66, [128, 2048])
    permute_88: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    mm_24: "f32[128, 2048]" = torch.ops.aten.mm.default(view_178, permute_88);  view_178 = permute_88 = None
    view_179: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_24, [1, 128, 2048]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_184: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_179, [1, 128, 16, 128]);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_91: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_32: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_91, [1, 16, 128, 128]);  permute_91 = None
    view_187: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_32, [16, 128, 128]);  expand_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_180: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_66, [128, 2048])
    permute_89: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    mm_25: "f32[128, 2048]" = torch.ops.aten.mm.default(view_180, permute_89);  view_180 = permute_89 = None
    view_181: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_25, [1, 128, 2048]);  mm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_185: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_181, [1, 128, 16, 128]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_92: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_94: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_92, [0, 1, 3, 2])
    expand_33: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_94, [1, 16, 128, 128]);  permute_94 = None
    view_188: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_33, [16, 128, 128]);  expand_33 = None
    bmm_16: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_187, view_188);  view_187 = view_188 = None
    view_189: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [1, 16, 128, 128]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_8: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_8: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_36, view_189, full_default_8);  slice_36 = view_189 = full_default_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
    sub_25: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_8, amax_8);  where_8 = amax_8 = None
    exp_8: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_34: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_8, [1, 16, 128, 128]);  div_8 = None
    view_190: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_34, [16, 128, 128]);  expand_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_182: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_66, [128, 2048]);  add_66 = None
    permute_90: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    mm_26: "f32[128, 2048]" = torch.ops.aten.mm.default(view_182, permute_90);  view_182 = permute_90 = None
    view_183: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_26, [1, 128, 2048]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_186: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_183, [1, 128, 16, 128]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_93: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_186, [0, 2, 1, 3]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_35: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_93, [1, 16, 128, 128])
    view_191: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_35, [16, 128, 128]);  expand_35 = None
    bmm_17: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_190, view_191);  view_190 = view_191 = None
    view_192: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_17, [1, 16, 128, 128]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_95: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_34: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_193: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_34, [1, 128, 2048]);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_194: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_193, [128, 2048]);  view_193 = None
    permute_96: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
    addmm_24: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg112_1, view_194, permute_96);  arg112_1 = view_194 = permute_96 = None
    view_195: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_24, [1, 128, 2048]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_67: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_195, add_64);  view_195 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_26: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_67, getitem_35);  getitem_35 = None
    add_68: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    mul_66: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_67: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_66, arg113_1);  mul_66 = arg113_1 = None
    add_69: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_67, arg114_1);  mul_67 = arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_196: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_69, [128, 2048]);  add_69 = None
    permute_97: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    addmm_25: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg116_1, view_196, permute_97);  arg116_1 = view_196 = permute_97 = None
    view_197: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_25, [1, 128, 8192]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_68: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_197, 0.5)
    pow_9: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_197, 3.0)
    mul_69: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_70: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_197, mul_69);  view_197 = mul_69 = None
    mul_70: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_70, 0.7978845608028654);  add_70 = None
    tanh_8: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
    add_71: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_8, 1.0);  tanh_8 = None
    mul_71: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_68, add_71);  mul_68 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_198: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_71, [128, 8192]);  mul_71 = None
    permute_98: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    addmm_26: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg118_1, view_198, permute_98);  arg118_1 = view_198 = permute_98 = None
    view_199: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_26, [1, 128, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_72: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_67, view_199);  add_67 = view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant9: "f32[]" = self._tensor_constant9
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_39: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg326_1, 2, 0, 128);  arg326_1 = None
    slice_40: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_39, 3, 0, 128);  slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_27: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_72, getitem_37);  getitem_37 = None
    add_73: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    mul_72: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
    mul_73: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_72, arg119_1);  mul_72 = arg119_1 = None
    add_74: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_73, arg120_1);  mul_73 = arg120_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_200: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_74, [128, 2048])
    permute_99: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    mm_27: "f32[128, 2048]" = torch.ops.aten.mm.default(view_200, permute_99);  view_200 = permute_99 = None
    view_201: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_27, [1, 128, 2048]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_206: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_201, [1, 128, 16, 128]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_102: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_36: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_102, [1, 16, 128, 128]);  permute_102 = None
    view_209: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_36, [16, 128, 128]);  expand_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_202: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_74, [128, 2048])
    permute_100: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    mm_28: "f32[128, 2048]" = torch.ops.aten.mm.default(view_202, permute_100);  view_202 = permute_100 = None
    view_203: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_28, [1, 128, 2048]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_207: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_203, [1, 128, 16, 128]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_103: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_105: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_103, [0, 1, 3, 2])
    expand_37: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_105, [1, 16, 128, 128]);  permute_105 = None
    view_210: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_37, [16, 128, 128]);  expand_37 = None
    bmm_18: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_209, view_210);  view_209 = view_210 = None
    view_211: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_18, [1, 16, 128, 128]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_9: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_9: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_40, view_211, full_default_9);  slice_40 = view_211 = full_default_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_9, [-1], True)
    sub_28: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_9, amax_9);  where_9 = amax_9 = None
    exp_9: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_38: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_9, [1, 16, 128, 128]);  div_9 = None
    view_212: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_38, [16, 128, 128]);  expand_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_204: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_74, [128, 2048]);  add_74 = None
    permute_101: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    mm_29: "f32[128, 2048]" = torch.ops.aten.mm.default(view_204, permute_101);  view_204 = permute_101 = None
    view_205: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_29, [1, 128, 2048]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_208: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_205, [1, 128, 16, 128]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_104: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_39: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_104, [1, 16, 128, 128])
    view_213: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_39, [16, 128, 128]);  expand_39 = None
    bmm_19: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_212, view_213);  view_212 = view_213 = None
    view_214: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_19, [1, 16, 128, 128]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_106: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    clone_38: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_215: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_38, [1, 128, 2048]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_216: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_215, [128, 2048]);  view_215 = None
    permute_107: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    addmm_27: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg125_1, view_216, permute_107);  arg125_1 = view_216 = permute_107 = None
    view_217: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_27, [1, 128, 2048]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_75: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_217, add_72);  view_217 = add_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_29: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_75, getitem_39);  getitem_39 = None
    add_76: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    mul_74: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_75: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_74, arg126_1);  mul_74 = arg126_1 = None
    add_77: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_75, arg127_1);  mul_75 = arg127_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_218: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_77, [128, 2048]);  add_77 = None
    permute_108: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_28: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg129_1, view_218, permute_108);  arg129_1 = view_218 = permute_108 = None
    view_219: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_28, [1, 128, 8192]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_76: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_219, 0.5)
    pow_10: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_219, 3.0)
    mul_77: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
    add_78: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_219, mul_77);  view_219 = mul_77 = None
    mul_78: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_78, 0.7978845608028654);  add_78 = None
    tanh_9: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_78);  mul_78 = None
    add_79: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_9, 1.0);  tanh_9 = None
    mul_79: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_76, add_79);  mul_76 = add_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_220: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_79, [128, 8192]);  mul_79 = None
    permute_109: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_29: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg131_1, view_220, permute_109);  arg131_1 = view_220 = permute_109 = None
    view_221: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_29, [1, 128, 2048]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_80: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_75, view_221);  add_75 = view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant10: "f32[]" = self._tensor_constant10
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_43: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg327_1, 2, 0, 128);  arg327_1 = None
    slice_44: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_43, 3, 0, 128);  slice_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_30: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_80, getitem_41);  getitem_41 = None
    add_81: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    mul_80: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
    mul_81: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_80, arg132_1);  mul_80 = arg132_1 = None
    add_82: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_81, arg133_1);  mul_81 = arg133_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_222: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_82, [128, 2048])
    permute_110: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    mm_30: "f32[128, 2048]" = torch.ops.aten.mm.default(view_222, permute_110);  view_222 = permute_110 = None
    view_223: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_30, [1, 128, 2048]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_228: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_223, [1, 128, 16, 128]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_113: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_40: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_113, [1, 16, 128, 128]);  permute_113 = None
    view_231: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_40, [16, 128, 128]);  expand_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_224: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_82, [128, 2048])
    permute_111: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    mm_31: "f32[128, 2048]" = torch.ops.aten.mm.default(view_224, permute_111);  view_224 = permute_111 = None
    view_225: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_31, [1, 128, 2048]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_229: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_225, [1, 128, 16, 128]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_114: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_116: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_114, [0, 1, 3, 2])
    expand_41: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_116, [1, 16, 128, 128]);  permute_116 = None
    view_232: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_41, [16, 128, 128]);  expand_41 = None
    bmm_20: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_231, view_232);  view_231 = view_232 = None
    view_233: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [1, 16, 128, 128]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_10: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_10: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_44, view_233, full_default_10);  slice_44 = view_233 = full_default_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
    sub_31: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_10, amax_10);  where_10 = amax_10 = None
    exp_10: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_42: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_10, [1, 16, 128, 128]);  div_10 = None
    view_234: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_42, [16, 128, 128]);  expand_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_226: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_82, [128, 2048]);  add_82 = None
    permute_112: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    mm_32: "f32[128, 2048]" = torch.ops.aten.mm.default(view_226, permute_112);  view_226 = permute_112 = None
    view_227: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_32, [1, 128, 2048]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_230: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_227, [1, 128, 16, 128]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_115: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_43: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_115, [1, 16, 128, 128])
    view_235: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_43, [16, 128, 128]);  expand_43 = None
    bmm_21: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_234, view_235);  view_234 = view_235 = None
    view_236: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_21, [1, 16, 128, 128]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_117: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    clone_42: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_237: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_42, [1, 128, 2048]);  clone_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_238: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_237, [128, 2048]);  view_237 = None
    permute_118: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_30: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg138_1, view_238, permute_118);  arg138_1 = view_238 = permute_118 = None
    view_239: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_30, [1, 128, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_83: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_239, add_80);  view_239 = add_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 128, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_32: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_83, getitem_43);  getitem_43 = None
    add_84: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    mul_82: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_83: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_82, arg139_1);  mul_82 = arg139_1 = None
    add_85: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_83, arg140_1);  mul_83 = arg140_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_240: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_85, [128, 2048]);  add_85 = None
    permute_119: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    addmm_31: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg142_1, view_240, permute_119);  arg142_1 = view_240 = permute_119 = None
    view_241: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_31, [1, 128, 8192]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_84: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_241, 0.5)
    pow_11: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_241, 3.0)
    mul_85: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
    add_86: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_241, mul_85);  view_241 = mul_85 = None
    mul_86: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_86, 0.7978845608028654);  add_86 = None
    tanh_10: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_86);  mul_86 = None
    add_87: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_10, 1.0);  tanh_10 = None
    mul_87: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_84, add_87);  mul_84 = add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_242: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_87, [128, 8192]);  mul_87 = None
    permute_120: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    addmm_32: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg144_1, view_242, permute_120);  arg144_1 = view_242 = permute_120 = None
    view_243: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_32, [1, 128, 2048]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_88: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_83, view_243);  add_83 = view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant11: "f32[]" = self._tensor_constant11
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_47: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg328_1, 2, 0, 128);  arg328_1 = None
    slice_48: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_47, 3, 0, 128);  slice_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_33: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_88, getitem_45);  getitem_45 = None
    add_89: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    mul_88: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
    mul_89: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_88, arg145_1);  mul_88 = arg145_1 = None
    add_90: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_89, arg146_1);  mul_89 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_244: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_90, [128, 2048])
    permute_121: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    mm_33: "f32[128, 2048]" = torch.ops.aten.mm.default(view_244, permute_121);  view_244 = permute_121 = None
    view_245: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_33, [1, 128, 2048]);  mm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_250: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_245, [1, 128, 16, 128]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_124: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_44: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_124, [1, 16, 128, 128]);  permute_124 = None
    view_253: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_44, [16, 128, 128]);  expand_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_246: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_90, [128, 2048])
    permute_122: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    mm_34: "f32[128, 2048]" = torch.ops.aten.mm.default(view_246, permute_122);  view_246 = permute_122 = None
    view_247: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_34, [1, 128, 2048]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_251: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_247, [1, 128, 16, 128]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_125: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_127: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2])
    expand_45: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_127, [1, 16, 128, 128]);  permute_127 = None
    view_254: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_45, [16, 128, 128]);  expand_45 = None
    bmm_22: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_253, view_254);  view_253 = view_254 = None
    view_255: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, [1, 16, 128, 128]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_11: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_11: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_48, view_255, full_default_11);  slice_48 = view_255 = full_default_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
    sub_34: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_11, amax_11);  where_11 = amax_11 = None
    exp_11: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_46: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_11, [1, 16, 128, 128]);  div_11 = None
    view_256: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_46, [16, 128, 128]);  expand_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_248: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_90, [128, 2048]);  add_90 = None
    permute_123: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    mm_35: "f32[128, 2048]" = torch.ops.aten.mm.default(view_248, permute_123);  view_248 = permute_123 = None
    view_249: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_35, [1, 128, 2048]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_252: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_249, [1, 128, 16, 128]);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_126: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_47: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_126, [1, 16, 128, 128])
    view_257: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_47, [16, 128, 128]);  expand_47 = None
    bmm_23: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_256, view_257);  view_256 = view_257 = None
    view_258: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_23, [1, 16, 128, 128]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_128: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    clone_46: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_259: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_46, [1, 128, 2048]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_260: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_259, [128, 2048]);  view_259 = None
    permute_129: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_33: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg151_1, view_260, permute_129);  arg151_1 = view_260 = permute_129 = None
    view_261: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_33, [1, 128, 2048]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_91: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_261, add_88);  view_261 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_35: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_91, getitem_47);  getitem_47 = None
    add_92: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    mul_90: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_91: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_90, arg152_1);  mul_90 = arg152_1 = None
    add_93: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_91, arg153_1);  mul_91 = arg153_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_262: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_93, [128, 2048]);  add_93 = None
    permute_130: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_34: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg155_1, view_262, permute_130);  arg155_1 = view_262 = permute_130 = None
    view_263: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_34, [1, 128, 8192]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_92: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_263, 0.5)
    pow_12: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_263, 3.0)
    mul_93: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_94: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_263, mul_93);  view_263 = mul_93 = None
    mul_94: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_94, 0.7978845608028654);  add_94 = None
    tanh_11: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
    add_95: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_11, 1.0);  tanh_11 = None
    mul_95: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_92, add_95);  mul_92 = add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_264: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_95, [128, 8192]);  mul_95 = None
    permute_131: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    addmm_35: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg157_1, view_264, permute_131);  arg157_1 = view_264 = permute_131 = None
    view_265: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_35, [1, 128, 2048]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_96: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_91, view_265);  add_91 = view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant12: "f32[]" = self._tensor_constant12
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_51: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg329_1, 2, 0, 128);  arg329_1 = None
    slice_52: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_51, 3, 0, 128);  slice_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_36: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_96, getitem_49);  getitem_49 = None
    add_97: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
    mul_96: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
    mul_97: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_96, arg158_1);  mul_96 = arg158_1 = None
    add_98: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_97, arg159_1);  mul_97 = arg159_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_266: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_98, [128, 2048])
    permute_132: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    mm_36: "f32[128, 2048]" = torch.ops.aten.mm.default(view_266, permute_132);  view_266 = permute_132 = None
    view_267: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_36, [1, 128, 2048]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_272: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_267, [1, 128, 16, 128]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_135: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_48: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_135, [1, 16, 128, 128]);  permute_135 = None
    view_275: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_48, [16, 128, 128]);  expand_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_268: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_98, [128, 2048])
    permute_133: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    mm_37: "f32[128, 2048]" = torch.ops.aten.mm.default(view_268, permute_133);  view_268 = permute_133 = None
    view_269: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_37, [1, 128, 2048]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_273: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_269, [1, 128, 16, 128]);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_136: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_138: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_136, [0, 1, 3, 2])
    expand_49: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_138, [1, 16, 128, 128]);  permute_138 = None
    view_276: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_49, [16, 128, 128]);  expand_49 = None
    bmm_24: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_275, view_276);  view_275 = view_276 = None
    view_277: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [1, 16, 128, 128]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_12: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_12: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_52, view_277, full_default_12);  slice_52 = view_277 = full_default_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_12, [-1], True)
    sub_37: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_12, amax_12);  where_12 = amax_12 = None
    exp_12: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_13: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_50: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_12, [1, 16, 128, 128]);  div_12 = None
    view_278: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_50, [16, 128, 128]);  expand_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_270: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_98, [128, 2048]);  add_98 = None
    permute_134: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    mm_38: "f32[128, 2048]" = torch.ops.aten.mm.default(view_270, permute_134);  view_270 = permute_134 = None
    view_271: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_38, [1, 128, 2048]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_274: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_271, [1, 128, 16, 128]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_137: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_51: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_137, [1, 16, 128, 128])
    view_279: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_51, [16, 128, 128]);  expand_51 = None
    bmm_25: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_278, view_279);  view_278 = view_279 = None
    view_280: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_25, [1, 16, 128, 128]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_139: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    clone_50: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_281: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_50, [1, 128, 2048]);  clone_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_282: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_281, [128, 2048]);  view_281 = None
    permute_140: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    addmm_36: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg164_1, view_282, permute_140);  arg164_1 = view_282 = permute_140 = None
    view_283: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_36, [1, 128, 2048]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_99: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_283, add_96);  view_283 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
    sub_38: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_99, getitem_51);  getitem_51 = None
    add_100: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    mul_98: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
    mul_99: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_98, arg165_1);  mul_98 = arg165_1 = None
    add_101: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_99, arg166_1);  mul_99 = arg166_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_284: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_101, [128, 2048]);  add_101 = None
    permute_141: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_37: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg168_1, view_284, permute_141);  arg168_1 = view_284 = permute_141 = None
    view_285: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_37, [1, 128, 8192]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_100: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_285, 0.5)
    pow_13: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_285, 3.0)
    mul_101: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
    add_102: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_285, mul_101);  view_285 = mul_101 = None
    mul_102: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_102, 0.7978845608028654);  add_102 = None
    tanh_12: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_102);  mul_102 = None
    add_103: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_12, 1.0);  tanh_12 = None
    mul_103: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_100, add_103);  mul_100 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_286: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_103, [128, 8192]);  mul_103 = None
    permute_142: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_38: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg170_1, view_286, permute_142);  arg170_1 = view_286 = permute_142 = None
    view_287: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_38, [1, 128, 2048]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_104: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_99, view_287);  add_99 = view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_104, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant13: "f32[]" = self._tensor_constant13
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_55: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg330_1, 2, 0, 128);  arg330_1 = None
    slice_56: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_55, 3, 0, 128);  slice_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_39: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_104, getitem_53);  getitem_53 = None
    add_105: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    mul_104: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
    mul_105: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_104, arg171_1);  mul_104 = arg171_1 = None
    add_106: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_105, arg172_1);  mul_105 = arg172_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_288: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_106, [128, 2048])
    permute_143: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    mm_39: "f32[128, 2048]" = torch.ops.aten.mm.default(view_288, permute_143);  view_288 = permute_143 = None
    view_289: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_39, [1, 128, 2048]);  mm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_294: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_289, [1, 128, 16, 128]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_146: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_52: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_146, [1, 16, 128, 128]);  permute_146 = None
    view_297: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_52, [16, 128, 128]);  expand_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_290: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_106, [128, 2048])
    permute_144: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    mm_40: "f32[128, 2048]" = torch.ops.aten.mm.default(view_290, permute_144);  view_290 = permute_144 = None
    view_291: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_40, [1, 128, 2048]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_295: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_291, [1, 128, 16, 128]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_147: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_295, [0, 2, 1, 3]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_149: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_147, [0, 1, 3, 2])
    expand_53: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_149, [1, 16, 128, 128]);  permute_149 = None
    view_298: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_53, [16, 128, 128]);  expand_53 = None
    bmm_26: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_297, view_298);  view_297 = view_298 = None
    view_299: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_26, [1, 16, 128, 128]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_13: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_13: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_56, view_299, full_default_13);  slice_56 = view_299 = full_default_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_13: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_13, [-1], True)
    sub_40: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_13, amax_13);  where_13 = amax_13 = None
    exp_13: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_14: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_54: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_13, [1, 16, 128, 128]);  div_13 = None
    view_300: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_54, [16, 128, 128]);  expand_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_292: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_106, [128, 2048]);  add_106 = None
    permute_145: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    mm_41: "f32[128, 2048]" = torch.ops.aten.mm.default(view_292, permute_145);  view_292 = permute_145 = None
    view_293: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_41, [1, 128, 2048]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_296: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_293, [1, 128, 16, 128]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_148: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_55: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_148, [1, 16, 128, 128])
    view_301: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_55, [16, 128, 128]);  expand_55 = None
    bmm_27: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_300, view_301);  view_300 = view_301 = None
    view_302: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_27, [1, 16, 128, 128]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_150: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
    clone_54: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_303: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_54, [1, 128, 2048]);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_304: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_303, [128, 2048]);  view_303 = None
    permute_151: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_39: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg177_1, view_304, permute_151);  arg177_1 = view_304 = permute_151 = None
    view_305: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_39, [1, 128, 2048]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_107: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_305, add_104);  view_305 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 128, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
    sub_41: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_107, getitem_55);  getitem_55 = None
    add_108: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    mul_106: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
    mul_107: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_106, arg178_1);  mul_106 = arg178_1 = None
    add_109: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_107, arg179_1);  mul_107 = arg179_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_306: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_109, [128, 2048]);  add_109 = None
    permute_152: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    addmm_40: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg181_1, view_306, permute_152);  arg181_1 = view_306 = permute_152 = None
    view_307: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_40, [1, 128, 8192]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_108: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_307, 0.5)
    pow_14: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_307, 3.0)
    mul_109: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
    add_110: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_307, mul_109);  view_307 = mul_109 = None
    mul_110: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_110, 0.7978845608028654);  add_110 = None
    tanh_13: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_110);  mul_110 = None
    add_111: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_13, 1.0);  tanh_13 = None
    mul_111: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_108, add_111);  mul_108 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_308: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_111, [128, 8192]);  mul_111 = None
    permute_153: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    addmm_41: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg183_1, view_308, permute_153);  arg183_1 = view_308 = permute_153 = None
    view_309: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_41, [1, 128, 2048]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_112: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_107, view_309);  add_107 = view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant14: "f32[]" = self._tensor_constant14
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_59: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg331_1, 2, 0, 128);  arg331_1 = None
    slice_60: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_59, 3, 0, 128);  slice_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_42: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_112, getitem_57);  getitem_57 = None
    add_113: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    mul_112: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
    mul_113: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_112, arg184_1);  mul_112 = arg184_1 = None
    add_114: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_113, arg185_1);  mul_113 = arg185_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_310: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_114, [128, 2048])
    permute_154: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    mm_42: "f32[128, 2048]" = torch.ops.aten.mm.default(view_310, permute_154);  view_310 = permute_154 = None
    view_311: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_42, [1, 128, 2048]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_316: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_311, [1, 128, 16, 128]);  view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_157: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_316, [0, 2, 1, 3]);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_56: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_157, [1, 16, 128, 128]);  permute_157 = None
    view_319: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_56, [16, 128, 128]);  expand_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_312: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_114, [128, 2048])
    permute_155: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    mm_43: "f32[128, 2048]" = torch.ops.aten.mm.default(view_312, permute_155);  view_312 = permute_155 = None
    view_313: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_43, [1, 128, 2048]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_317: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_313, [1, 128, 16, 128]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_158: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_317, [0, 2, 1, 3]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_160: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_158, [0, 1, 3, 2])
    expand_57: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_160, [1, 16, 128, 128]);  permute_160 = None
    view_320: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_57, [16, 128, 128]);  expand_57 = None
    bmm_28: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_319, view_320);  view_319 = view_320 = None
    view_321: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [1, 16, 128, 128]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_14: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_14: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_60, view_321, full_default_14);  slice_60 = view_321 = full_default_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_14, [-1], True)
    sub_43: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_14, amax_14);  where_14 = amax_14 = None
    exp_14: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_15: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_58: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_14, [1, 16, 128, 128]);  div_14 = None
    view_322: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_58, [16, 128, 128]);  expand_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_314: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_114, [128, 2048]);  add_114 = None
    permute_156: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    mm_44: "f32[128, 2048]" = torch.ops.aten.mm.default(view_314, permute_156);  view_314 = permute_156 = None
    view_315: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_44, [1, 128, 2048]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_318: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_315, [1, 128, 16, 128]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_159: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_59: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_159, [1, 16, 128, 128])
    view_323: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_59, [16, 128, 128]);  expand_59 = None
    bmm_29: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_322, view_323);  view_322 = view_323 = None
    view_324: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_29, [1, 16, 128, 128]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_161: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
    clone_58: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_325: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_58, [1, 128, 2048]);  clone_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_326: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_325, [128, 2048]);  view_325 = None
    permute_162: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_42: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg190_1, view_326, permute_162);  arg190_1 = view_326 = permute_162 = None
    view_327: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_42, [1, 128, 2048]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_115: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_327, add_112);  view_327 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 128, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 128, 1]" = var_mean_29[1];  var_mean_29 = None
    sub_44: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_115, getitem_59);  getitem_59 = None
    add_116: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    mul_114: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
    mul_115: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_114, arg191_1);  mul_114 = arg191_1 = None
    add_117: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_115, arg192_1);  mul_115 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_328: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_117, [128, 2048]);  add_117 = None
    permute_163: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    addmm_43: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg194_1, view_328, permute_163);  arg194_1 = view_328 = permute_163 = None
    view_329: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_43, [1, 128, 8192]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_116: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_329, 0.5)
    pow_15: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_329, 3.0)
    mul_117: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
    add_118: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_329, mul_117);  view_329 = mul_117 = None
    mul_118: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_118, 0.7978845608028654);  add_118 = None
    tanh_14: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
    add_119: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_14, 1.0);  tanh_14 = None
    mul_119: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_116, add_119);  mul_116 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_330: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_119, [128, 8192]);  mul_119 = None
    permute_164: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    addmm_44: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg196_1, view_330, permute_164);  arg196_1 = view_330 = permute_164 = None
    view_331: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_44, [1, 128, 2048]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_120: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_115, view_331);  add_115 = view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_120, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 128, 1]" = var_mean_30[1];  var_mean_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant15: "f32[]" = self._tensor_constant15
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_63: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg332_1, 2, 0, 128);  arg332_1 = None
    slice_64: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_63, 3, 0, 128);  slice_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_45: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_120, getitem_61);  getitem_61 = None
    add_121: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    mul_120: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = rsqrt_30 = None
    mul_121: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, arg197_1);  mul_120 = arg197_1 = None
    add_122: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_121, arg198_1);  mul_121 = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_332: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_122, [128, 2048])
    permute_165: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
    mm_45: "f32[128, 2048]" = torch.ops.aten.mm.default(view_332, permute_165);  view_332 = permute_165 = None
    view_333: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_45, [1, 128, 2048]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_338: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_333, [1, 128, 16, 128]);  view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_168: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_60: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_168, [1, 16, 128, 128]);  permute_168 = None
    view_341: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_60, [16, 128, 128]);  expand_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_334: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_122, [128, 2048])
    permute_166: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg200_1, [1, 0]);  arg200_1 = None
    mm_46: "f32[128, 2048]" = torch.ops.aten.mm.default(view_334, permute_166);  view_334 = permute_166 = None
    view_335: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_46, [1, 128, 2048]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_339: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_335, [1, 128, 16, 128]);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_169: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_171: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_169, [0, 1, 3, 2])
    expand_61: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_171, [1, 16, 128, 128]);  permute_171 = None
    view_342: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_61, [16, 128, 128]);  expand_61 = None
    bmm_30: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_341, view_342);  view_341 = view_342 = None
    view_343: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_30, [1, 16, 128, 128]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_15: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_15: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_64, view_343, full_default_15);  slice_64 = view_343 = full_default_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_15: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_15, [-1], True)
    sub_46: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_15, amax_15);  where_15 = amax_15 = None
    exp_15: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_16: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_62: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_15, [1, 16, 128, 128]);  div_15 = None
    view_344: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_62, [16, 128, 128]);  expand_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_336: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_122, [128, 2048]);  add_122 = None
    permute_167: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
    mm_47: "f32[128, 2048]" = torch.ops.aten.mm.default(view_336, permute_167);  view_336 = permute_167 = None
    view_337: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_47, [1, 128, 2048]);  mm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_340: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_337, [1, 128, 16, 128]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_170: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_63: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_170, [1, 16, 128, 128])
    view_345: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_63, [16, 128, 128]);  expand_63 = None
    bmm_31: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_344, view_345);  view_344 = view_345 = None
    view_346: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_31, [1, 16, 128, 128]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_172: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
    clone_62: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_347: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_62, [1, 128, 2048]);  clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_348: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_347, [128, 2048]);  view_347 = None
    permute_173: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
    addmm_45: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg203_1, view_348, permute_173);  arg203_1 = view_348 = permute_173 = None
    view_349: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_45, [1, 128, 2048]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_123: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_349, add_120);  view_349 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_123, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 128, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 128, 1]" = var_mean_31[1];  var_mean_31 = None
    sub_47: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_123, getitem_63);  getitem_63 = None
    add_124: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    mul_122: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
    mul_123: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_122, arg204_1);  mul_122 = arg204_1 = None
    add_125: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_123, arg205_1);  mul_123 = arg205_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_350: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_125, [128, 2048]);  add_125 = None
    permute_174: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg206_1, [1, 0]);  arg206_1 = None
    addmm_46: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg207_1, view_350, permute_174);  arg207_1 = view_350 = permute_174 = None
    view_351: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_46, [1, 128, 8192]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_124: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_351, 0.5)
    pow_16: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_351, 3.0)
    mul_125: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
    add_126: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_351, mul_125);  view_351 = mul_125 = None
    mul_126: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_126, 0.7978845608028654);  add_126 = None
    tanh_15: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_126);  mul_126 = None
    add_127: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_15, 1.0);  tanh_15 = None
    mul_127: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_124, add_127);  mul_124 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_352: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_127, [128, 8192]);  mul_127 = None
    permute_175: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
    addmm_47: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg209_1, view_352, permute_175);  arg209_1 = view_352 = permute_175 = None
    view_353: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_47, [1, 128, 2048]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_128: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_123, view_353);  add_123 = view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 128, 1]" = var_mean_32[1];  var_mean_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant16: "f32[]" = self._tensor_constant16
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_67: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg333_1, 2, 0, 128);  arg333_1 = None
    slice_68: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_67, 3, 0, 128);  slice_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_48: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_128, getitem_65);  getitem_65 = None
    add_129: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
    mul_128: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = rsqrt_32 = None
    mul_129: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_128, arg210_1);  mul_128 = arg210_1 = None
    add_130: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_129, arg211_1);  mul_129 = arg211_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_354: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_130, [128, 2048])
    permute_176: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg212_1, [1, 0]);  arg212_1 = None
    mm_48: "f32[128, 2048]" = torch.ops.aten.mm.default(view_354, permute_176);  view_354 = permute_176 = None
    view_355: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_48, [1, 128, 2048]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_360: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_355, [1, 128, 16, 128]);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_179: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_64: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_179, [1, 16, 128, 128]);  permute_179 = None
    view_363: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_64, [16, 128, 128]);  expand_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_356: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_130, [128, 2048])
    permute_177: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
    mm_49: "f32[128, 2048]" = torch.ops.aten.mm.default(view_356, permute_177);  view_356 = permute_177 = None
    view_357: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_49, [1, 128, 2048]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_361: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_357, [1, 128, 16, 128]);  view_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_180: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_182: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_180, [0, 1, 3, 2])
    expand_65: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_182, [1, 16, 128, 128]);  permute_182 = None
    view_364: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_65, [16, 128, 128]);  expand_65 = None
    bmm_32: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_363, view_364);  view_363 = view_364 = None
    view_365: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [1, 16, 128, 128]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_16: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_16: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_68, view_365, full_default_16);  slice_68 = view_365 = full_default_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_16, [-1], True)
    sub_49: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_16, amax_16);  where_16 = amax_16 = None
    exp_16: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_17: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_66: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_16, [1, 16, 128, 128]);  div_16 = None
    view_366: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_66, [16, 128, 128]);  expand_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_358: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_130, [128, 2048]);  add_130 = None
    permute_178: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    mm_50: "f32[128, 2048]" = torch.ops.aten.mm.default(view_358, permute_178);  view_358 = permute_178 = None
    view_359: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_50, [1, 128, 2048]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_362: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_359, [1, 128, 16, 128]);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_181: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_362, [0, 2, 1, 3]);  view_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_67: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_181, [1, 16, 128, 128])
    view_367: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_67, [16, 128, 128]);  expand_67 = None
    bmm_33: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_366, view_367);  view_366 = view_367 = None
    view_368: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_33, [1, 16, 128, 128]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_183: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    clone_66: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_369: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_66, [1, 128, 2048]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_370: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_369, [128, 2048]);  view_369 = None
    permute_184: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
    addmm_48: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg216_1, view_370, permute_184);  arg216_1 = view_370 = permute_184 = None
    view_371: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_48, [1, 128, 2048]);  addmm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_131: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_371, add_128);  view_371 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_131, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 128, 1]" = var_mean_33[1];  var_mean_33 = None
    sub_50: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_131, getitem_67);  getitem_67 = None
    add_132: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_132);  add_132 = None
    mul_130: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
    mul_131: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_130, arg217_1);  mul_130 = arg217_1 = None
    add_133: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_131, arg218_1);  mul_131 = arg218_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_372: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_133, [128, 2048]);  add_133 = None
    permute_185: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
    addmm_49: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg220_1, view_372, permute_185);  arg220_1 = view_372 = permute_185 = None
    view_373: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_49, [1, 128, 8192]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_132: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_373, 0.5)
    pow_17: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_373, 3.0)
    mul_133: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
    add_134: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_373, mul_133);  view_373 = mul_133 = None
    mul_134: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_134, 0.7978845608028654);  add_134 = None
    tanh_16: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_134);  mul_134 = None
    add_135: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_16, 1.0);  tanh_16 = None
    mul_135: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_132, add_135);  mul_132 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_374: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_135, [128, 8192]);  mul_135 = None
    permute_186: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
    addmm_50: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg222_1, view_374, permute_186);  arg222_1 = view_374 = permute_186 = None
    view_375: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_50, [1, 128, 2048]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_136: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_131, view_375);  add_131 = view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 128, 1]" = var_mean_34[1];  var_mean_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant17: "f32[]" = self._tensor_constant17
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_71: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg334_1, 2, 0, 128);  arg334_1 = None
    slice_72: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_71, 3, 0, 128);  slice_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_51: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_136, getitem_69);  getitem_69 = None
    add_137: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    mul_136: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = rsqrt_34 = None
    mul_137: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_136, arg223_1);  mul_136 = arg223_1 = None
    add_138: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_137, arg224_1);  mul_137 = arg224_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_376: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_138, [128, 2048])
    permute_187: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
    mm_51: "f32[128, 2048]" = torch.ops.aten.mm.default(view_376, permute_187);  view_376 = permute_187 = None
    view_377: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_51, [1, 128, 2048]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_382: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_377, [1, 128, 16, 128]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_190: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_382, [0, 2, 1, 3]);  view_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_68: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_190, [1, 16, 128, 128]);  permute_190 = None
    view_385: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_68, [16, 128, 128]);  expand_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_378: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_138, [128, 2048])
    permute_188: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
    mm_52: "f32[128, 2048]" = torch.ops.aten.mm.default(view_378, permute_188);  view_378 = permute_188 = None
    view_379: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_52, [1, 128, 2048]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_383: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_379, [1, 128, 16, 128]);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_191: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_193: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_191, [0, 1, 3, 2])
    expand_69: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_193, [1, 16, 128, 128]);  permute_193 = None
    view_386: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_69, [16, 128, 128]);  expand_69 = None
    bmm_34: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_385, view_386);  view_385 = view_386 = None
    view_387: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_34, [1, 16, 128, 128]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_17: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_17: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_72, view_387, full_default_17);  slice_72 = view_387 = full_default_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_17: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_17, [-1], True)
    sub_52: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_17, amax_17);  where_17 = amax_17 = None
    exp_17: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_18: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_70: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_17, [1, 16, 128, 128]);  div_17 = None
    view_388: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_70, [16, 128, 128]);  expand_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_380: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_138, [128, 2048]);  add_138 = None
    permute_189: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
    mm_53: "f32[128, 2048]" = torch.ops.aten.mm.default(view_380, permute_189);  view_380 = permute_189 = None
    view_381: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_53, [1, 128, 2048]);  mm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_384: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_381, [1, 128, 16, 128]);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_192: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_71: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_192, [1, 16, 128, 128])
    view_389: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_71, [16, 128, 128]);  expand_71 = None
    bmm_35: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_388, view_389);  view_388 = view_389 = None
    view_390: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_35, [1, 16, 128, 128]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_194: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
    clone_70: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_391: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_70, [1, 128, 2048]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_392: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_391, [128, 2048]);  view_391 = None
    permute_195: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg228_1, [1, 0]);  arg228_1 = None
    addmm_51: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg229_1, view_392, permute_195);  arg229_1 = view_392 = permute_195 = None
    view_393: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_51, [1, 128, 2048]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_139: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_393, add_136);  view_393 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 128, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 128, 1]" = var_mean_35[1];  var_mean_35 = None
    sub_53: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_139, getitem_71);  getitem_71 = None
    add_140: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    mul_138: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
    mul_139: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_138, arg230_1);  mul_138 = arg230_1 = None
    add_141: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_139, arg231_1);  mul_139 = arg231_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_394: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_141, [128, 2048]);  add_141 = None
    permute_196: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg232_1, [1, 0]);  arg232_1 = None
    addmm_52: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg233_1, view_394, permute_196);  arg233_1 = view_394 = permute_196 = None
    view_395: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_52, [1, 128, 8192]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_140: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_395, 0.5)
    pow_18: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_395, 3.0)
    mul_141: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
    add_142: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_395, mul_141);  view_395 = mul_141 = None
    mul_142: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_142, 0.7978845608028654);  add_142 = None
    tanh_17: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
    add_143: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_17, 1.0);  tanh_17 = None
    mul_143: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_140, add_143);  mul_140 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_396: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_143, [128, 8192]);  mul_143 = None
    permute_197: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
    addmm_53: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg235_1, view_396, permute_197);  arg235_1 = view_396 = permute_197 = None
    view_397: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_53, [1, 128, 2048]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_144: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_139, view_397);  add_139 = view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 128, 1]" = var_mean_36[1];  var_mean_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant18: "f32[]" = self._tensor_constant18
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_75: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg335_1, 2, 0, 128);  arg335_1 = None
    slice_76: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_75, 3, 0, 128);  slice_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_54: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_144, getitem_73);  getitem_73 = None
    add_145: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    mul_144: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = rsqrt_36 = None
    mul_145: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_144, arg236_1);  mul_144 = arg236_1 = None
    add_146: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_145, arg237_1);  mul_145 = arg237_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_398: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_146, [128, 2048])
    permute_198: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
    mm_54: "f32[128, 2048]" = torch.ops.aten.mm.default(view_398, permute_198);  view_398 = permute_198 = None
    view_399: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_54, [1, 128, 2048]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_404: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_399, [1, 128, 16, 128]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_201: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_72: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_201, [1, 16, 128, 128]);  permute_201 = None
    view_407: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_72, [16, 128, 128]);  expand_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_400: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_146, [128, 2048])
    permute_199: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
    mm_55: "f32[128, 2048]" = torch.ops.aten.mm.default(view_400, permute_199);  view_400 = permute_199 = None
    view_401: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_55, [1, 128, 2048]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_405: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_401, [1, 128, 16, 128]);  view_401 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_202: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_204: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_202, [0, 1, 3, 2])
    expand_73: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_204, [1, 16, 128, 128]);  permute_204 = None
    view_408: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_73, [16, 128, 128]);  expand_73 = None
    bmm_36: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_407, view_408);  view_407 = view_408 = None
    view_409: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [1, 16, 128, 128]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_18: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_18: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_76, view_409, full_default_18);  slice_76 = view_409 = full_default_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_18: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_18, [-1], True)
    sub_55: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_18, amax_18);  where_18 = amax_18 = None
    exp_18: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_19: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_74: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_18, [1, 16, 128, 128]);  div_18 = None
    view_410: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_74, [16, 128, 128]);  expand_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_402: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_146, [128, 2048]);  add_146 = None
    permute_200: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
    mm_56: "f32[128, 2048]" = torch.ops.aten.mm.default(view_402, permute_200);  view_402 = permute_200 = None
    view_403: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_56, [1, 128, 2048]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_406: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_403, [1, 128, 16, 128]);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_203: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_75: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_203, [1, 16, 128, 128])
    view_411: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_75, [16, 128, 128]);  expand_75 = None
    bmm_37: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_410, view_411);  view_410 = view_411 = None
    view_412: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_37, [1, 16, 128, 128]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_205: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
    clone_74: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_413: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_74, [1, 128, 2048]);  clone_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_414: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_413, [128, 2048]);  view_413 = None
    permute_206: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
    addmm_54: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg242_1, view_414, permute_206);  arg242_1 = view_414 = permute_206 = None
    view_415: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_54, [1, 128, 2048]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_147: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_415, add_144);  view_415 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 128, 1]" = var_mean_37[1];  var_mean_37 = None
    sub_56: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_147, getitem_75);  getitem_75 = None
    add_148: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    mul_146: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
    mul_147: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_146, arg243_1);  mul_146 = arg243_1 = None
    add_149: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_147, arg244_1);  mul_147 = arg244_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_416: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_149, [128, 2048]);  add_149 = None
    permute_207: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
    addmm_55: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg246_1, view_416, permute_207);  arg246_1 = view_416 = permute_207 = None
    view_417: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_55, [1, 128, 8192]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_148: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_417, 0.5)
    pow_19: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_417, 3.0)
    mul_149: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
    add_150: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_417, mul_149);  view_417 = mul_149 = None
    mul_150: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_150, 0.7978845608028654);  add_150 = None
    tanh_18: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_150);  mul_150 = None
    add_151: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_18, 1.0);  tanh_18 = None
    mul_151: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_148, add_151);  mul_148 = add_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_418: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_151, [128, 8192]);  mul_151 = None
    permute_208: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
    addmm_56: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg248_1, view_418, permute_208);  arg248_1 = view_418 = permute_208 = None
    view_419: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_56, [1, 128, 2048]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_152: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_147, view_419);  add_147 = view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_152, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 128, 1]" = var_mean_38[1];  var_mean_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant19: "f32[]" = self._tensor_constant19
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_79: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg336_1, 2, 0, 128);  arg336_1 = None
    slice_80: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_79, 3, 0, 128);  slice_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_57: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_152, getitem_77);  getitem_77 = None
    add_153: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    mul_152: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = rsqrt_38 = None
    mul_153: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_152, arg249_1);  mul_152 = arg249_1 = None
    add_154: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_153, arg250_1);  mul_153 = arg250_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_420: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_154, [128, 2048])
    permute_209: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
    mm_57: "f32[128, 2048]" = torch.ops.aten.mm.default(view_420, permute_209);  view_420 = permute_209 = None
    view_421: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_57, [1, 128, 2048]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_426: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_421, [1, 128, 16, 128]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_212: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_76: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_212, [1, 16, 128, 128]);  permute_212 = None
    view_429: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_76, [16, 128, 128]);  expand_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_422: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_154, [128, 2048])
    permute_210: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
    mm_58: "f32[128, 2048]" = torch.ops.aten.mm.default(view_422, permute_210);  view_422 = permute_210 = None
    view_423: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_58, [1, 128, 2048]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_427: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_423, [1, 128, 16, 128]);  view_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_213: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_215: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_213, [0, 1, 3, 2])
    expand_77: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_215, [1, 16, 128, 128]);  permute_215 = None
    view_430: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_77, [16, 128, 128]);  expand_77 = None
    bmm_38: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_429, view_430);  view_429 = view_430 = None
    view_431: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_38, [1, 16, 128, 128]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_19: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_19: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_80, view_431, full_default_19);  slice_80 = view_431 = full_default_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_19: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_19, [-1], True)
    sub_58: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_19, amax_19);  where_19 = amax_19 = None
    exp_19: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_20: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_78: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_19, [1, 16, 128, 128]);  div_19 = None
    view_432: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_78, [16, 128, 128]);  expand_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_424: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_154, [128, 2048]);  add_154 = None
    permute_211: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
    mm_59: "f32[128, 2048]" = torch.ops.aten.mm.default(view_424, permute_211);  view_424 = permute_211 = None
    view_425: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_59, [1, 128, 2048]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_428: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_425, [1, 128, 16, 128]);  view_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_214: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_428, [0, 2, 1, 3]);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_79: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_214, [1, 16, 128, 128])
    view_433: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_79, [16, 128, 128]);  expand_79 = None
    bmm_39: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_432, view_433);  view_432 = view_433 = None
    view_434: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_39, [1, 16, 128, 128]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_216: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
    clone_78: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_435: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_78, [1, 128, 2048]);  clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_436: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_435, [128, 2048]);  view_435 = None
    permute_217: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    addmm_57: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg255_1, view_436, permute_217);  arg255_1 = view_436 = permute_217 = None
    view_437: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_57, [1, 128, 2048]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_155: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_437, add_152);  view_437 = add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_155, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 128, 1]" = var_mean_39[1];  var_mean_39 = None
    sub_59: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_155, getitem_79);  getitem_79 = None
    add_156: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_156);  add_156 = None
    mul_154: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
    mul_155: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_154, arg256_1);  mul_154 = arg256_1 = None
    add_157: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_155, arg257_1);  mul_155 = arg257_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_438: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_157, [128, 2048]);  add_157 = None
    permute_218: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
    addmm_58: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg259_1, view_438, permute_218);  arg259_1 = view_438 = permute_218 = None
    view_439: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_58, [1, 128, 8192]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_156: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_439, 0.5)
    pow_20: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_439, 3.0)
    mul_157: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
    add_158: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_439, mul_157);  view_439 = mul_157 = None
    mul_158: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_158, 0.7978845608028654);  add_158 = None
    tanh_19: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_158);  mul_158 = None
    add_159: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_19, 1.0);  tanh_19 = None
    mul_159: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_156, add_159);  mul_156 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_440: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_159, [128, 8192]);  mul_159 = None
    permute_219: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg260_1, [1, 0]);  arg260_1 = None
    addmm_59: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg261_1, view_440, permute_219);  arg261_1 = view_440 = permute_219 = None
    view_441: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_59, [1, 128, 2048]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_160: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_155, view_441);  add_155 = view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 128, 1]" = var_mean_40[1];  var_mean_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant20: "f32[]" = self._tensor_constant20
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_83: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg337_1, 2, 0, 128);  arg337_1 = None
    slice_84: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_83, 3, 0, 128);  slice_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_60: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_160, getitem_81);  getitem_81 = None
    add_161: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    mul_160: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = rsqrt_40 = None
    mul_161: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_160, arg262_1);  mul_160 = arg262_1 = None
    add_162: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_161, arg263_1);  mul_161 = arg263_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_442: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_162, [128, 2048])
    permute_220: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
    mm_60: "f32[128, 2048]" = torch.ops.aten.mm.default(view_442, permute_220);  view_442 = permute_220 = None
    view_443: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_60, [1, 128, 2048]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_448: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_443, [1, 128, 16, 128]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_223: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_80: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_223, [1, 16, 128, 128]);  permute_223 = None
    view_451: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_80, [16, 128, 128]);  expand_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_444: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_162, [128, 2048])
    permute_221: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
    mm_61: "f32[128, 2048]" = torch.ops.aten.mm.default(view_444, permute_221);  view_444 = permute_221 = None
    view_445: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_61, [1, 128, 2048]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_449: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_445, [1, 128, 16, 128]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_224: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_449, [0, 2, 1, 3]);  view_449 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_226: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_224, [0, 1, 3, 2])
    expand_81: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_226, [1, 16, 128, 128]);  permute_226 = None
    view_452: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_81, [16, 128, 128]);  expand_81 = None
    bmm_40: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_451, view_452);  view_451 = view_452 = None
    view_453: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [1, 16, 128, 128]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_20: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_20: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_84, view_453, full_default_20);  slice_84 = view_453 = full_default_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_20: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_20, [-1], True)
    sub_61: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_20, amax_20);  where_20 = amax_20 = None
    exp_20: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_21: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_82: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_20, [1, 16, 128, 128]);  div_20 = None
    view_454: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_82, [16, 128, 128]);  expand_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_446: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_162, [128, 2048]);  add_162 = None
    permute_222: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
    mm_62: "f32[128, 2048]" = torch.ops.aten.mm.default(view_446, permute_222);  view_446 = permute_222 = None
    view_447: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_62, [1, 128, 2048]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_450: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_447, [1, 128, 16, 128]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_225: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_83: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_225, [1, 16, 128, 128])
    view_455: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_83, [16, 128, 128]);  expand_83 = None
    bmm_41: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_454, view_455);  view_454 = view_455 = None
    view_456: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_41, [1, 16, 128, 128]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_227: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
    clone_82: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_457: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_82, [1, 128, 2048]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_458: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_457, [128, 2048]);  view_457 = None
    permute_228: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
    addmm_60: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg268_1, view_458, permute_228);  arg268_1 = view_458 = permute_228 = None
    view_459: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_60, [1, 128, 2048]);  addmm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_163: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_459, add_160);  view_459 = add_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_163, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 128, 1]" = var_mean_41[1];  var_mean_41 = None
    sub_62: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_163, getitem_83);  getitem_83 = None
    add_164: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    mul_162: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
    mul_163: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_162, arg269_1);  mul_162 = arg269_1 = None
    add_165: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_163, arg270_1);  mul_163 = arg270_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_460: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_165, [128, 2048]);  add_165 = None
    permute_229: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
    addmm_61: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg272_1, view_460, permute_229);  arg272_1 = view_460 = permute_229 = None
    view_461: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_61, [1, 128, 8192]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_164: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_461, 0.5)
    pow_21: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_461, 3.0)
    mul_165: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
    add_166: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_461, mul_165);  view_461 = mul_165 = None
    mul_166: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_166, 0.7978845608028654);  add_166 = None
    tanh_20: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
    add_167: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_20, 1.0);  tanh_20 = None
    mul_167: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_164, add_167);  mul_164 = add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_462: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_167, [128, 8192]);  mul_167 = None
    permute_230: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
    addmm_62: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg274_1, view_462, permute_230);  arg274_1 = view_462 = permute_230 = None
    view_463: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_62, [1, 128, 2048]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_168: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_163, view_463);  add_163 = view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_168, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 128, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 128, 1]" = var_mean_42[1];  var_mean_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant21: "f32[]" = self._tensor_constant21
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_87: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg338_1, 2, 0, 128);  arg338_1 = None
    slice_88: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_87, 3, 0, 128);  slice_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_63: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_168, getitem_85);  getitem_85 = None
    add_169: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    mul_168: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = rsqrt_42 = None
    mul_169: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_168, arg275_1);  mul_168 = arg275_1 = None
    add_170: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_169, arg276_1);  mul_169 = arg276_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_464: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_170, [128, 2048])
    permute_231: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
    mm_63: "f32[128, 2048]" = torch.ops.aten.mm.default(view_464, permute_231);  view_464 = permute_231 = None
    view_465: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_63, [1, 128, 2048]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_470: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_465, [1, 128, 16, 128]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_234: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_470, [0, 2, 1, 3]);  view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_84: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_234, [1, 16, 128, 128]);  permute_234 = None
    view_473: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_84, [16, 128, 128]);  expand_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_466: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_170, [128, 2048])
    permute_232: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg278_1, [1, 0]);  arg278_1 = None
    mm_64: "f32[128, 2048]" = torch.ops.aten.mm.default(view_466, permute_232);  view_466 = permute_232 = None
    view_467: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_64, [1, 128, 2048]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_471: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_467, [1, 128, 16, 128]);  view_467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_235: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_237: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_235, [0, 1, 3, 2])
    expand_85: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_237, [1, 16, 128, 128]);  permute_237 = None
    view_474: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_85, [16, 128, 128]);  expand_85 = None
    bmm_42: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_473, view_474);  view_473 = view_474 = None
    view_475: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_42, [1, 16, 128, 128]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_21: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_21: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_88, view_475, full_default_21);  slice_88 = view_475 = full_default_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_21: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_21, [-1], True)
    sub_64: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_21, amax_21);  where_21 = amax_21 = None
    exp_21: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_22: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_21: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_86: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_21, [1, 16, 128, 128]);  div_21 = None
    view_476: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_86, [16, 128, 128]);  expand_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_468: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_170, [128, 2048]);  add_170 = None
    permute_233: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
    mm_65: "f32[128, 2048]" = torch.ops.aten.mm.default(view_468, permute_233);  view_468 = permute_233 = None
    view_469: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_65, [1, 128, 2048]);  mm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_472: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_469, [1, 128, 16, 128]);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_236: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_87: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_236, [1, 16, 128, 128])
    view_477: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_87, [16, 128, 128]);  expand_87 = None
    bmm_43: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_476, view_477);  view_476 = view_477 = None
    view_478: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_43, [1, 16, 128, 128]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_238: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
    clone_86: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_479: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_86, [1, 128, 2048]);  clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_480: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_479, [128, 2048]);  view_479 = None
    permute_239: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg280_1, [1, 0]);  arg280_1 = None
    addmm_63: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg281_1, view_480, permute_239);  arg281_1 = view_480 = permute_239 = None
    view_481: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_63, [1, 128, 2048]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_171: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_481, add_168);  view_481 = add_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_171, [2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 128, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 128, 1]" = var_mean_43[1];  var_mean_43 = None
    sub_65: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_171, getitem_87);  getitem_87 = None
    add_172: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    mul_170: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
    mul_171: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_170, arg282_1);  mul_170 = arg282_1 = None
    add_173: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_171, arg283_1);  mul_171 = arg283_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_482: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_173, [128, 2048]);  add_173 = None
    permute_240: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg284_1, [1, 0]);  arg284_1 = None
    addmm_64: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg285_1, view_482, permute_240);  arg285_1 = view_482 = permute_240 = None
    view_483: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_64, [1, 128, 8192]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_172: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_483, 0.5)
    pow_22: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_483, 3.0)
    mul_173: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
    add_174: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_483, mul_173);  view_483 = mul_173 = None
    mul_174: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_174, 0.7978845608028654);  add_174 = None
    tanh_21: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_174);  mul_174 = None
    add_175: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_21, 1.0);  tanh_21 = None
    mul_175: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_172, add_175);  mul_172 = add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_484: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_175, [128, 8192]);  mul_175 = None
    permute_241: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg286_1, [1, 0]);  arg286_1 = None
    addmm_65: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg287_1, view_484, permute_241);  arg287_1 = view_484 = permute_241 = None
    view_485: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_65, [1, 128, 2048]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_176: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_171, view_485);  add_171 = view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_176, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 128, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 128, 1]" = var_mean_44[1];  var_mean_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant22: "f32[]" = self._tensor_constant22
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_91: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg339_1, 2, 0, 128);  arg339_1 = None
    slice_92: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_91, 3, 0, 128);  slice_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_66: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_176, getitem_89);  getitem_89 = None
    add_177: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    mul_176: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = rsqrt_44 = None
    mul_177: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_176, arg288_1);  mul_176 = arg288_1 = None
    add_178: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_177, arg289_1);  mul_177 = arg289_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_486: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_178, [128, 2048])
    permute_242: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg290_1, [1, 0]);  arg290_1 = None
    mm_66: "f32[128, 2048]" = torch.ops.aten.mm.default(view_486, permute_242);  view_486 = permute_242 = None
    view_487: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_66, [1, 128, 2048]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_492: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_487, [1, 128, 16, 128]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_245: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_88: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_245, [1, 16, 128, 128]);  permute_245 = None
    view_495: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_88, [16, 128, 128]);  expand_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_488: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_178, [128, 2048])
    permute_243: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
    mm_67: "f32[128, 2048]" = torch.ops.aten.mm.default(view_488, permute_243);  view_488 = permute_243 = None
    view_489: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_67, [1, 128, 2048]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_493: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_489, [1, 128, 16, 128]);  view_489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_246: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_248: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_246, [0, 1, 3, 2])
    expand_89: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_248, [1, 16, 128, 128]);  permute_248 = None
    view_496: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_89, [16, 128, 128]);  expand_89 = None
    bmm_44: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_495, view_496);  view_495 = view_496 = None
    view_497: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [1, 16, 128, 128]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_22: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_22: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_92, view_497, full_default_22);  slice_92 = view_497 = full_default_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_22: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_22, [-1], True)
    sub_67: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_22, amax_22);  where_22 = amax_22 = None
    exp_22: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
    sum_23: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_90: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_22, [1, 16, 128, 128]);  div_22 = None
    view_498: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_90, [16, 128, 128]);  expand_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_490: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_178, [128, 2048]);  add_178 = None
    permute_244: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg292_1, [1, 0]);  arg292_1 = None
    mm_68: "f32[128, 2048]" = torch.ops.aten.mm.default(view_490, permute_244);  view_490 = permute_244 = None
    view_491: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_68, [1, 128, 2048]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_494: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_491, [1, 128, 16, 128]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_247: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_91: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_247, [1, 16, 128, 128])
    view_499: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_91, [16, 128, 128]);  expand_91 = None
    bmm_45: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_498, view_499);  view_498 = view_499 = None
    view_500: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_45, [1, 16, 128, 128]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_249: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
    clone_90: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_501: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_90, [1, 128, 2048]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_502: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_501, [128, 2048]);  view_501 = None
    permute_250: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
    addmm_66: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg294_1, view_502, permute_250);  arg294_1 = view_502 = permute_250 = None
    view_503: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_66, [1, 128, 2048]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_179: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_503, add_176);  view_503 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_179, [2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 128, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 128, 1]" = var_mean_45[1];  var_mean_45 = None
    sub_68: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_179, getitem_91);  getitem_91 = None
    add_180: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_180);  add_180 = None
    mul_178: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
    mul_179: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_178, arg295_1);  mul_178 = arg295_1 = None
    add_181: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_179, arg296_1);  mul_179 = arg296_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_504: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_181, [128, 2048]);  add_181 = None
    permute_251: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
    addmm_67: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg298_1, view_504, permute_251);  arg298_1 = view_504 = permute_251 = None
    view_505: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_67, [1, 128, 8192]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_180: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_505, 0.5)
    pow_23: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_505, 3.0)
    mul_181: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
    add_182: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_505, mul_181);  view_505 = mul_181 = None
    mul_182: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_182, 0.7978845608028654);  add_182 = None
    tanh_22: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_182);  mul_182 = None
    add_183: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_22, 1.0);  tanh_22 = None
    mul_183: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_180, add_183);  mul_180 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_506: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_183, [128, 8192]);  mul_183 = None
    permute_252: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
    addmm_68: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg300_1, view_506, permute_252);  arg300_1 = view_506 = permute_252 = None
    view_507: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_68, [1, 128, 2048]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_184: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_179, view_507);  add_179 = view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_184, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 128, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 128, 1]" = var_mean_46[1];  var_mean_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    _tensor_constant23: "f32[]" = self._tensor_constant23
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:192, code: causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]
    slice_95: "b8[1, 1, 128, 2048]" = torch.ops.aten.slice.Tensor(arg340_1, 2, 0, 128);  arg340_1 = None
    slice_96: "b8[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_95, 3, 0, 128);  slice_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:327, code: hidden_states = self.ln_1(hidden_states)
    sub_69: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_184, getitem_93);  getitem_93 = None
    add_185: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    mul_184: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = rsqrt_46 = None
    mul_185: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_184, arg301_1);  mul_184 = arg301_1 = None
    add_186: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_185, arg302_1);  mul_185 = arg302_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:224, code: query = self.q_proj(hidden_states)
    view_508: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_186, [128, 2048])
    permute_253: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
    mm_69: "f32[128, 2048]" = torch.ops.aten.mm.default(view_508, permute_253);  view_508 = permute_253 = None
    view_509: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_69, [1, 128, 2048]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_514: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_509, [1, 128, 16, 128]);  view_509 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_256: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    expand_92: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_256, [1, 16, 128, 128]);  permute_256 = None
    view_517: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_92, [16, 128, 128]);  expand_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:225, code: key = self.k_proj(hidden_states)
    view_510: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_186, [128, 2048])
    permute_254: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg304_1, [1, 0]);  arg304_1 = None
    mm_70: "f32[128, 2048]" = torch.ops.aten.mm.default(view_510, permute_254);  view_510 = permute_254 = None
    view_511: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_70, [1, 128, 2048]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_515: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_511, [1, 128, 16, 128]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_257: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_515, [0, 2, 1, 3]);  view_515 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:189, code: attn_weights = torch.matmul(query, key.transpose(-1, -2))
    permute_259: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2])
    expand_93: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_259, [1, 16, 128, 128]);  permute_259 = None
    view_518: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_93, [16, 128, 128]);  expand_93 = None
    bmm_46: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_517, view_518);  view_517 = view_518 = None
    view_519: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, [1, 16, 128, 128]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:196, code: mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
    full_default_23: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:197, code: attn_weights = torch.where(causal_mask, attn_weights, mask_value)
    where_23: "f32[1, 16, 128, 128]" = torch.ops.aten.where.self(slice_96, view_519, full_default_23);  slice_96 = view_519 = full_default_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:203, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_23: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(where_23, [-1], True)
    sub_70: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(where_23, amax_23);  where_23 = amax_23 = None
    exp_23: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
    sum_24: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_23: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_94: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(div_23, [1, 16, 128, 128]);  div_23 = None
    view_520: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_94, [16, 128, 128]);  expand_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:226, code: value = self.v_proj(hidden_states)
    view_512: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_186, [128, 2048]);  add_186 = None
    permute_255: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
    mm_71: "f32[128, 2048]" = torch.ops.aten.mm.default(view_512, permute_255);  view_512 = permute_255 = None
    view_513: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(mm_71, [1, 128, 2048]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:173, code: tensor = tensor.view(new_shape)
    view_516: "f32[1, 128, 16, 128]" = torch.ops.aten.reshape.default(view_513, [1, 128, 16, 128]);  view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:174, code: return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    permute_258: "f32[1, 16, 128, 128]" = torch.ops.aten.permute.default(view_516, [0, 2, 1, 3]);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:211, code: attn_output = torch.matmul(attn_weights, value)
    expand_95: "f32[1, 16, 128, 128]" = torch.ops.aten.expand.default(permute_258, [1, 16, 128, 128])
    view_521: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(expand_95, [16, 128, 128]);  expand_95 = None
    bmm_47: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_520, view_521);  view_520 = view_521 = None
    view_522: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_47, [1, 16, 128, 128]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:180, code: tensor = tensor.permute(0, 2, 1, 3).contiguous()
    permute_260: "f32[1, 128, 16, 128]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
    clone_94: "f32[1, 128, 16, 128]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:182, code: return tensor.view(new_shape)
    view_523: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(clone_94, [1, 128, 2048]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:246, code: attn_output = self.out_proj(attn_output)
    view_524: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_523, [128, 2048]);  view_523 = None
    permute_261: "f32[2048, 2048]" = torch.ops.aten.permute.default(arg306_1, [1, 0]);  arg306_1 = None
    addmm_69: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg307_1, view_524, permute_261);  arg307_1 = view_524 = permute_261 = None
    view_525: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_69, [1, 128, 2048]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:339, code: hidden_states = attn_output + residual
    add_187: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(view_525, add_184);  view_525 = add_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:342, code: hidden_states = self.ln_2(hidden_states)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_187, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 128, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 128, 1]" = var_mean_47[1];  var_mean_47 = None
    sub_71: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_187, getitem_95);  getitem_95 = None
    add_188: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    mul_186: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
    mul_187: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_186, arg308_1);  mul_186 = arg308_1 = None
    add_189: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_187, arg309_1);  mul_187 = arg309_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:300, code: hidden_states = self.c_fc(hidden_states)
    view_526: "f32[128, 2048]" = torch.ops.aten.reshape.default(add_189, [128, 2048]);  add_189 = None
    permute_262: "f32[2048, 8192]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
    addmm_70: "f32[128, 8192]" = torch.ops.aten.addmm.default(arg311_1, view_526, permute_262);  arg311_1 = view_526 = permute_262 = None
    view_527: "f32[1, 128, 8192]" = torch.ops.aten.reshape.default(addmm_70, [1, 128, 8192]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_188: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(view_527, 0.5)
    pow_24: "f32[1, 128, 8192]" = torch.ops.aten.pow.Tensor_Scalar(view_527, 3.0)
    mul_189: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
    add_190: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(view_527, mul_189);  view_527 = mul_189 = None
    mul_190: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(add_190, 0.7978845608028654);  add_190 = None
    tanh_23: "f32[1, 128, 8192]" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
    add_191: "f32[1, 128, 8192]" = torch.ops.aten.add.Tensor(tanh_23, 1.0);  tanh_23 = None
    mul_191: "f32[1, 128, 8192]" = torch.ops.aten.mul.Tensor(mul_188, add_191);  mul_188 = add_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:302, code: hidden_states = self.c_proj(hidden_states)
    view_528: "f32[128, 8192]" = torch.ops.aten.reshape.default(mul_191, [128, 8192]);  mul_191 = None
    permute_263: "f32[8192, 2048]" = torch.ops.aten.permute.default(arg312_1, [1, 0]);  arg312_1 = None
    addmm_71: "f32[128, 2048]" = torch.ops.aten.addmm.default(arg313_1, view_528, permute_263);  arg313_1 = view_528 = permute_263 = None
    view_529: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(addmm_71, [1, 128, 2048]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:345, code: hidden_states = residual + feed_forward_hidden_states
    add_192: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(add_187, view_529);  add_187 = view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:641, code: hidden_states = self.ln_f(hidden_states)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_192, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 128, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 128, 1]" = var_mean_48[1];  var_mean_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:775, code: shift_labels = labels[..., 1:].contiguous()
    slice_99: "i64[1, 127]" = torch.ops.aten.slice.Tensor(arg342_1, 1, 1, 9223372036854775807);  arg342_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:778, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_534: "i64[127]" = torch.ops.aten.reshape.default(slice_99, [-1]);  slice_99 = None
    ne_1: "b8[127]" = torch.ops.aten.ne.Scalar(view_534, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:641, code: hidden_states = self.ln_f(hidden_states)
    sub_72: "f32[1, 128, 2048]" = torch.ops.aten.sub.Tensor(add_192, getitem_97);  add_192 = getitem_97 = None
    add_193: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    mul_192: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = rsqrt_48 = None
    mul_193: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_192, arg314_1);  mul_192 = arg314_1 = None
    add_194: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(mul_193, arg315_1);  mul_193 = arg315_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:643, code: hidden_states = hidden_states.view(output_shape)
    view_530: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_194, [-1, 128, 2048]);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:763, code: lm_logits = self.lm_head(hidden_states)
    view_531: "f32[128, 2048]" = torch.ops.aten.reshape.default(view_530, [128, 2048]);  view_530 = None
    permute_264: "f32[2048, 50257]" = torch.ops.aten.permute.default(arg316_1, [1, 0]);  arg316_1 = None
    mm_72: "f32[128, 50257]" = torch.ops.aten.mm.default(view_531, permute_264);  view_531 = permute_264 = None
    view_532: "f32[1, 128, 50257]" = torch.ops.aten.reshape.default(mm_72, [1, 128, 50257]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:774, code: shift_logits = lm_logits[..., :-1, :].contiguous()
    slice_97: "f32[1, 127, 50257]" = torch.ops.aten.slice.Tensor(view_532, 1, 0, -1)
    slice_98: "f32[1, 127, 50257]" = torch.ops.aten.slice.Tensor(slice_97, 2, 0, 9223372036854775807);  slice_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/gpt_neo/modeling_gpt_neo.py:778, code: loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    view_533: "f32[127, 50257]" = torch.ops.aten.reshape.default(slice_98, [-1, 50257]);  slice_98 = None
    amax_24: "f32[127, 1]" = torch.ops.aten.amax.default(view_533, [1], True)
    sub_73: "f32[127, 50257]" = torch.ops.aten.sub.Tensor(view_533, amax_24);  view_533 = amax_24 = None
    exp_24: "f32[127, 50257]" = torch.ops.aten.exp.default(sub_73)
    sum_25: "f32[127, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[127, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_74: "f32[127, 50257]" = torch.ops.aten.sub.Tensor(sub_73, log);  sub_73 = log = None
    ne: "b8[127]" = torch.ops.aten.ne.Scalar(view_534, -100)
    full_default_24: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_24: "i64[127]" = torch.ops.aten.where.self(ne, view_534, full_default_24);  ne = full_default_24 = None
    unsqueeze_1: "i64[127, 1]" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
    gather: "f32[127, 1]" = torch.ops.aten.gather.default(sub_74, 1, unsqueeze_1);  sub_74 = unsqueeze_1 = None
    squeeze: "f32[127]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[127]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_25: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_25: "f32[127]" = torch.ops.aten.where.self(ne_1, neg, full_default_25);  ne_1 = neg = full_default_25 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_25);  where_25 = None
    ne_2: "b8[127]" = torch.ops.aten.ne.Scalar(view_534, -100);  view_534 = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
    return (div_24, view_532, permute_4, permute_5, permute_15, permute_16, permute_26, permute_27, permute_37, permute_38, permute_48, permute_49, permute_59, permute_60, permute_70, permute_71, permute_81, permute_82, permute_92, permute_93, permute_103, permute_104, permute_114, permute_115, permute_125, permute_126, permute_136, permute_137, permute_147, permute_148, permute_158, permute_159, permute_169, permute_170, permute_180, permute_181, permute_191, permute_192, permute_202, permute_203, permute_213, permute_214, permute_224, permute_225, permute_235, permute_236, permute_246, permute_247, permute_257, permute_258)
    