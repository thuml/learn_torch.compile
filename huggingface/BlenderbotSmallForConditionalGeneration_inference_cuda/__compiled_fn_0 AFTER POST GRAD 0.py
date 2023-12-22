from __future__ import annotations



def forward(self, arg0_1: "f32[512, 512]", arg1_1: "f32[512, 512]", arg2_1: "f32[50265, 512]", arg3_1: "f32[512]", arg4_1: "f32[512]", arg5_1: "f32[512, 512]", arg6_1: "f32[512]", arg7_1: "f32[512, 512]", arg8_1: "f32[512]", arg9_1: "f32[512, 512]", arg10_1: "f32[512]", arg11_1: "f32[512, 512]", arg12_1: "f32[512]", arg13_1: "f32[512]", arg14_1: "f32[512]", arg15_1: "f32[2048, 512]", arg16_1: "f32[2048]", arg17_1: "f32[512, 2048]", arg18_1: "f32[512]", arg19_1: "f32[512]", arg20_1: "f32[512]", arg21_1: "f32[512, 512]", arg22_1: "f32[512]", arg23_1: "f32[512, 512]", arg24_1: "f32[512]", arg25_1: "f32[512, 512]", arg26_1: "f32[512]", arg27_1: "f32[512, 512]", arg28_1: "f32[512]", arg29_1: "f32[512]", arg30_1: "f32[512]", arg31_1: "f32[2048, 512]", arg32_1: "f32[2048]", arg33_1: "f32[512, 2048]", arg34_1: "f32[512]", arg35_1: "f32[512]", arg36_1: "f32[512]", arg37_1: "f32[512, 512]", arg38_1: "f32[512]", arg39_1: "f32[512, 512]", arg40_1: "f32[512]", arg41_1: "f32[512, 512]", arg42_1: "f32[512]", arg43_1: "f32[512, 512]", arg44_1: "f32[512]", arg45_1: "f32[512]", arg46_1: "f32[512]", arg47_1: "f32[2048, 512]", arg48_1: "f32[2048]", arg49_1: "f32[512, 2048]", arg50_1: "f32[512]", arg51_1: "f32[512]", arg52_1: "f32[512]", arg53_1: "f32[512, 512]", arg54_1: "f32[512]", arg55_1: "f32[512, 512]", arg56_1: "f32[512]", arg57_1: "f32[512, 512]", arg58_1: "f32[512]", arg59_1: "f32[512, 512]", arg60_1: "f32[512]", arg61_1: "f32[512]", arg62_1: "f32[512]", arg63_1: "f32[2048, 512]", arg64_1: "f32[2048]", arg65_1: "f32[512, 2048]", arg66_1: "f32[512]", arg67_1: "f32[512]", arg68_1: "f32[512]", arg69_1: "f32[512, 512]", arg70_1: "f32[512]", arg71_1: "f32[512, 512]", arg72_1: "f32[512]", arg73_1: "f32[512, 512]", arg74_1: "f32[512]", arg75_1: "f32[512, 512]", arg76_1: "f32[512]", arg77_1: "f32[512]", arg78_1: "f32[512]", arg79_1: "f32[2048, 512]", arg80_1: "f32[2048]", arg81_1: "f32[512, 2048]", arg82_1: "f32[512]", arg83_1: "f32[512]", arg84_1: "f32[512]", arg85_1: "f32[512, 512]", arg86_1: "f32[512]", arg87_1: "f32[512, 512]", arg88_1: "f32[512]", arg89_1: "f32[512, 512]", arg90_1: "f32[512]", arg91_1: "f32[512, 512]", arg92_1: "f32[512]", arg93_1: "f32[512]", arg94_1: "f32[512]", arg95_1: "f32[2048, 512]", arg96_1: "f32[2048]", arg97_1: "f32[512, 2048]", arg98_1: "f32[512]", arg99_1: "f32[512]", arg100_1: "f32[512]", arg101_1: "f32[512, 512]", arg102_1: "f32[512]", arg103_1: "f32[512, 512]", arg104_1: "f32[512]", arg105_1: "f32[512, 512]", arg106_1: "f32[512]", arg107_1: "f32[512, 512]", arg108_1: "f32[512]", arg109_1: "f32[512]", arg110_1: "f32[512]", arg111_1: "f32[2048, 512]", arg112_1: "f32[2048]", arg113_1: "f32[512, 2048]", arg114_1: "f32[512]", arg115_1: "f32[512]", arg116_1: "f32[512]", arg117_1: "f32[512, 512]", arg118_1: "f32[512]", arg119_1: "f32[512, 512]", arg120_1: "f32[512]", arg121_1: "f32[512, 512]", arg122_1: "f32[512]", arg123_1: "f32[512, 512]", arg124_1: "f32[512]", arg125_1: "f32[512]", arg126_1: "f32[512]", arg127_1: "f32[2048, 512]", arg128_1: "f32[2048]", arg129_1: "f32[512, 2048]", arg130_1: "f32[512]", arg131_1: "f32[512]", arg132_1: "f32[512]", arg133_1: "f32[512]", arg134_1: "f32[512]", arg135_1: "f32[512, 512]", arg136_1: "f32[512]", arg137_1: "f32[512, 512]", arg138_1: "f32[512]", arg139_1: "f32[512, 512]", arg140_1: "f32[512]", arg141_1: "f32[512, 512]", arg142_1: "f32[512]", arg143_1: "f32[512]", arg144_1: "f32[512]", arg145_1: "f32[512, 512]", arg146_1: "f32[512]", arg147_1: "f32[512, 512]", arg148_1: "f32[512]", arg149_1: "f32[512, 512]", arg150_1: "f32[512]", arg151_1: "f32[512, 512]", arg152_1: "f32[512]", arg153_1: "f32[512]", arg154_1: "f32[512]", arg155_1: "f32[2048, 512]", arg156_1: "f32[2048]", arg157_1: "f32[512, 2048]", arg158_1: "f32[512]", arg159_1: "f32[512]", arg160_1: "f32[512]", arg161_1: "f32[512, 512]", arg162_1: "f32[512]", arg163_1: "f32[512, 512]", arg164_1: "f32[512]", arg165_1: "f32[512, 512]", arg166_1: "f32[512]", arg167_1: "f32[512, 512]", arg168_1: "f32[512]", arg169_1: "f32[512]", arg170_1: "f32[512]", arg171_1: "f32[512, 512]", arg172_1: "f32[512]", arg173_1: "f32[512, 512]", arg174_1: "f32[512]", arg175_1: "f32[512, 512]", arg176_1: "f32[512]", arg177_1: "f32[512, 512]", arg178_1: "f32[512]", arg179_1: "f32[512]", arg180_1: "f32[512]", arg181_1: "f32[2048, 512]", arg182_1: "f32[2048]", arg183_1: "f32[512, 2048]", arg184_1: "f32[512]", arg185_1: "f32[512]", arg186_1: "f32[512]", arg187_1: "f32[512, 512]", arg188_1: "f32[512]", arg189_1: "f32[512, 512]", arg190_1: "f32[512]", arg191_1: "f32[512, 512]", arg192_1: "f32[512]", arg193_1: "f32[512, 512]", arg194_1: "f32[512]", arg195_1: "f32[512]", arg196_1: "f32[512]", arg197_1: "f32[512, 512]", arg198_1: "f32[512]", arg199_1: "f32[512, 512]", arg200_1: "f32[512]", arg201_1: "f32[512, 512]", arg202_1: "f32[512]", arg203_1: "f32[512, 512]", arg204_1: "f32[512]", arg205_1: "f32[512]", arg206_1: "f32[512]", arg207_1: "f32[2048, 512]", arg208_1: "f32[2048]", arg209_1: "f32[512, 2048]", arg210_1: "f32[512]", arg211_1: "f32[512]", arg212_1: "f32[512]", arg213_1: "f32[512, 512]", arg214_1: "f32[512]", arg215_1: "f32[512, 512]", arg216_1: "f32[512]", arg217_1: "f32[512, 512]", arg218_1: "f32[512]", arg219_1: "f32[512, 512]", arg220_1: "f32[512]", arg221_1: "f32[512]", arg222_1: "f32[512]", arg223_1: "f32[512, 512]", arg224_1: "f32[512]", arg225_1: "f32[512, 512]", arg226_1: "f32[512]", arg227_1: "f32[512, 512]", arg228_1: "f32[512]", arg229_1: "f32[512, 512]", arg230_1: "f32[512]", arg231_1: "f32[512]", arg232_1: "f32[512]", arg233_1: "f32[2048, 512]", arg234_1: "f32[2048]", arg235_1: "f32[512, 2048]", arg236_1: "f32[512]", arg237_1: "f32[512]", arg238_1: "f32[512]", arg239_1: "f32[512, 512]", arg240_1: "f32[512]", arg241_1: "f32[512, 512]", arg242_1: "f32[512]", arg243_1: "f32[512, 512]", arg244_1: "f32[512]", arg245_1: "f32[512, 512]", arg246_1: "f32[512]", arg247_1: "f32[512]", arg248_1: "f32[512]", arg249_1: "f32[512, 512]", arg250_1: "f32[512]", arg251_1: "f32[512, 512]", arg252_1: "f32[512]", arg253_1: "f32[512, 512]", arg254_1: "f32[512]", arg255_1: "f32[512, 512]", arg256_1: "f32[512]", arg257_1: "f32[512]", arg258_1: "f32[512]", arg259_1: "f32[2048, 512]", arg260_1: "f32[2048]", arg261_1: "f32[512, 2048]", arg262_1: "f32[512]", arg263_1: "f32[512]", arg264_1: "f32[512]", arg265_1: "f32[512, 512]", arg266_1: "f32[512]", arg267_1: "f32[512, 512]", arg268_1: "f32[512]", arg269_1: "f32[512, 512]", arg270_1: "f32[512]", arg271_1: "f32[512, 512]", arg272_1: "f32[512]", arg273_1: "f32[512]", arg274_1: "f32[512]", arg275_1: "f32[512, 512]", arg276_1: "f32[512]", arg277_1: "f32[512, 512]", arg278_1: "f32[512]", arg279_1: "f32[512, 512]", arg280_1: "f32[512]", arg281_1: "f32[512, 512]", arg282_1: "f32[512]", arg283_1: "f32[512]", arg284_1: "f32[512]", arg285_1: "f32[2048, 512]", arg286_1: "f32[2048]", arg287_1: "f32[512, 2048]", arg288_1: "f32[512]", arg289_1: "f32[512]", arg290_1: "f32[512]", arg291_1: "f32[512, 512]", arg292_1: "f32[512]", arg293_1: "f32[512, 512]", arg294_1: "f32[512]", arg295_1: "f32[512, 512]", arg296_1: "f32[512]", arg297_1: "f32[512, 512]", arg298_1: "f32[512]", arg299_1: "f32[512]", arg300_1: "f32[512]", arg301_1: "f32[512, 512]", arg302_1: "f32[512]", arg303_1: "f32[512, 512]", arg304_1: "f32[512]", arg305_1: "f32[512, 512]", arg306_1: "f32[512]", arg307_1: "f32[512, 512]", arg308_1: "f32[512]", arg309_1: "f32[512]", arg310_1: "f32[512]", arg311_1: "f32[2048, 512]", arg312_1: "f32[2048]", arg313_1: "f32[512, 2048]", arg314_1: "f32[512]", arg315_1: "f32[512]", arg316_1: "f32[512]", arg317_1: "f32[512, 512]", arg318_1: "f32[512]", arg319_1: "f32[512, 512]", arg320_1: "f32[512]", arg321_1: "f32[512, 512]", arg322_1: "f32[512]", arg323_1: "f32[512, 512]", arg324_1: "f32[512]", arg325_1: "f32[512]", arg326_1: "f32[512]", arg327_1: "f32[512, 512]", arg328_1: "f32[512]", arg329_1: "f32[512, 512]", arg330_1: "f32[512]", arg331_1: "f32[512, 512]", arg332_1: "f32[512]", arg333_1: "f32[512, 512]", arg334_1: "f32[512]", arg335_1: "f32[512]", arg336_1: "f32[512]", arg337_1: "f32[2048, 512]", arg338_1: "f32[2048]", arg339_1: "f32[512, 2048]", arg340_1: "f32[512]", arg341_1: "f32[512]", arg342_1: "f32[512]", arg343_1: "f32[50265, 512]", arg344_1: "f32[1, 50265]", arg345_1: "i64[1, 128]", arg346_1: "i64[1, 128]", arg347_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:734, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.reshape.default(arg347_1, [-1, 128]);  arg347_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:741, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg2_1, view, 0);  view = None
    mul: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:119, code: positions = torch.arange(
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_1: "f32[128, 512]" = torch.ops.aten.embedding.default(arg0_1, iota);  arg0_1 = iota = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:745, code: hidden_states = inputs_embeds + embed_pos
    add: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:746, code: hidden_states = self.layernorm_embedding(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    sub: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
    add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    mul_1: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_2: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, arg3_1);  mul_1 = arg3_1 = None
    add_2: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_2, arg4_1);  mul_2 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_1: "f32[128, 512]" = torch.ops.aten.reshape.default(add_2, [128, 512])
    permute: "f32[512, 512]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    
    # No stacktrace found for following nodes
    mm_default_127: "f32[128, 512]" = torch.ops.aten.mm.default(view_1, permute);  view_1 = permute = None
    add_tensor_127: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_127, arg6_1);  mm_default_127 = arg6_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_2: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_127, [1, 128, 512]);  add_tensor_127 = None
    mul_3: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_2, 0.1767766952966369);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_9: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_3, [1, 128, 16, 32]);  mul_3 = None
    permute_5: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    clone_3: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_10: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_3, [16, -1, 32]);  clone_3 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_45: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_10, 0);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_3: "f32[128, 512]" = torch.ops.aten.reshape.default(add_2, [128, 512])
    permute_1: "f32[512, 512]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    
    # No stacktrace found for following nodes
    mm_default_126: "f32[128, 512]" = torch.ops.aten.mm.default(view_3, permute_1);  view_3 = permute_1 = None
    add_tensor_126: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_126, arg8_1);  mm_default_126 = arg8_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_4: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_126, [1, 128, 512]);  add_tensor_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_5: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_4, [1, -1, 16, 32]);  view_4 = None
    permute_2: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    clone_1: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_11: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_1, [16, -1, 32]);  clone_1 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_46: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_11, 0);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_6: "f32[128, 512]" = torch.ops.aten.reshape.default(add_2, [128, 512])
    permute_3: "f32[512, 512]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    
    # No stacktrace found for following nodes
    mm_default_125: "f32[128, 512]" = torch.ops.aten.mm.default(view_6, permute_3);  view_6 = permute_3 = None
    add_tensor_125: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_125, arg10_1);  mm_default_125 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_7: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_125, [1, 128, 512]);  add_tensor_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_8: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_7, [1, -1, 16, 32]);  view_7 = None
    permute_4: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    clone_2: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_12: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_2, [16, -1, 32]);  clone_2 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_47: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_12, 0);  view_12 = None
    _scaled_dot_product_efficient_attention_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_45, unsqueeze_default_46, unsqueeze_default_47, None, True, scale = 1.0);  unsqueeze_default_45 = unsqueeze_default_46 = unsqueeze_default_47 = None
    getitem_99: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_15[0];  _scaled_dot_product_efficient_attention_default_15 = None
    squeeze_dim_15: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_99, 0);  getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_13: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_15, [1, 16, 128, 32]);  squeeze_dim_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_14: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_5, [1, 128, 512]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_15: "f32[128, 512]" = torch.ops.aten.reshape.default(view_14, [128, 512]);  view_14 = None
    permute_8: "f32[512, 512]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
    
    # No stacktrace found for following nodes
    mm_default_124: "f32[128, 512]" = torch.ops.aten.mm.default(view_15, permute_8);  view_15 = permute_8 = None
    add_tensor_124: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_124, arg12_1);  mm_default_124 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_16: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_124, [1, 128, 512]);  add_tensor_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:324, code: hidden_states = residual + hidden_states
    add_3: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_2, view_16);  add_2 = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:325, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    sub_2: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_3, getitem_3);  add_3 = getitem_3 = None
    add_4: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    mul_4: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_5: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_4, arg13_1);  mul_4 = arg13_1 = None
    add_5: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_17: "f32[128, 512]" = torch.ops.aten.reshape.default(add_5, [128, 512])
    permute_9: "f32[512, 2048]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
    
    # No stacktrace found for following nodes
    mm_default_123: "f32[128, 2048]" = torch.ops.aten.mm.default(view_17, permute_9);  view_17 = permute_9 = None
    add_tensor_123: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_123, arg16_1);  mm_default_123 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_18: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_123, [1, 128, 2048]);  add_tensor_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_7: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
    erf: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_6: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_6, add_6);  mul_6 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_19: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_8, [128, 2048]);  mul_8 = None
    permute_10: "f32[2048, 512]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
    
    # No stacktrace found for following nodes
    mm_default_122: "f32[128, 512]" = torch.ops.aten.mm.default(view_19, permute_10);  view_19 = permute_10 = None
    add_tensor_122: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_122, arg18_1);  mm_default_122 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_20: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_122, [1, 128, 512]);  add_tensor_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:332, code: hidden_states = residual + hidden_states
    add_7: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_5, view_20);  add_5 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:333, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_3: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_7, getitem_5);  add_7 = getitem_5 = None
    add_8: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    mul_9: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_10: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_9, arg19_1);  mul_9 = arg19_1 = None
    add_9: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_21: "f32[128, 512]" = torch.ops.aten.reshape.default(add_9, [128, 512])
    permute_11: "f32[512, 512]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
    
    # No stacktrace found for following nodes
    mm_default_121: "f32[128, 512]" = torch.ops.aten.mm.default(view_21, permute_11);  view_21 = permute_11 = None
    add_tensor_121: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_121, arg22_1);  mm_default_121 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_22: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_121, [1, 128, 512]);  add_tensor_121 = None
    mul_11: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_22, 0.1767766952966369);  view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_29: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_11, [1, 128, 16, 32]);  mul_11 = None
    permute_16: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_11: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_30: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_11, [16, -1, 32]);  clone_11 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_42: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_30, 0);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_23: "f32[128, 512]" = torch.ops.aten.reshape.default(add_9, [128, 512])
    permute_12: "f32[512, 512]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    
    # No stacktrace found for following nodes
    mm_default_120: "f32[128, 512]" = torch.ops.aten.mm.default(view_23, permute_12);  view_23 = permute_12 = None
    add_tensor_120: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_120, arg24_1);  mm_default_120 = arg24_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_24: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_120, [1, 128, 512]);  add_tensor_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_25: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_24, [1, -1, 16, 32]);  view_24 = None
    permute_13: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_25, [0, 2, 1, 3]);  view_25 = None
    clone_9: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_31: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_9, [16, -1, 32]);  clone_9 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_43: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_31, 0);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_26: "f32[128, 512]" = torch.ops.aten.reshape.default(add_9, [128, 512])
    permute_14: "f32[512, 512]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    
    # No stacktrace found for following nodes
    mm_default_119: "f32[128, 512]" = torch.ops.aten.mm.default(view_26, permute_14);  view_26 = permute_14 = None
    add_tensor_119: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_119, arg26_1);  mm_default_119 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_27: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_119, [1, 128, 512]);  add_tensor_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_28: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_27, [1, -1, 16, 32]);  view_27 = None
    permute_15: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_28, [0, 2, 1, 3]);  view_28 = None
    clone_10: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_32: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_10, [16, -1, 32]);  clone_10 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_44: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_32, 0);  view_32 = None
    _scaled_dot_product_efficient_attention_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_42, unsqueeze_default_43, unsqueeze_default_44, None, True, scale = 1.0);  unsqueeze_default_42 = unsqueeze_default_43 = unsqueeze_default_44 = None
    getitem_98: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_14[0];  _scaled_dot_product_efficient_attention_default_14 = None
    squeeze_dim_14: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_98, 0);  getitem_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_33: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_14, [1, 16, 128, 32]);  squeeze_dim_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_34: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_13, [1, 128, 512]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_35: "f32[128, 512]" = torch.ops.aten.reshape.default(view_34, [128, 512]);  view_34 = None
    permute_19: "f32[512, 512]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
    
    # No stacktrace found for following nodes
    mm_default_118: "f32[128, 512]" = torch.ops.aten.mm.default(view_35, permute_19);  view_35 = permute_19 = None
    add_tensor_118: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_118, arg28_1);  mm_default_118 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_36: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_118, [1, 128, 512]);  add_tensor_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:324, code: hidden_states = residual + hidden_states
    add_10: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_9, view_36);  add_9 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:325, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_5: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_10, getitem_7);  add_10 = getitem_7 = None
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    mul_12: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_13: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_12, arg29_1);  mul_12 = arg29_1 = None
    add_12: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_13, arg30_1);  mul_13 = arg30_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_37: "f32[128, 512]" = torch.ops.aten.reshape.default(add_12, [128, 512])
    permute_20: "f32[512, 2048]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
    
    # No stacktrace found for following nodes
    mm_default_117: "f32[128, 2048]" = torch.ops.aten.mm.default(view_37, permute_20);  view_37 = permute_20 = None
    add_tensor_117: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_117, arg32_1);  mm_default_117 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_38: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_117, [1, 128, 2048]);  add_tensor_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_15: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
    erf_1: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_13: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_14, add_13);  mul_14 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_39: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_16, [128, 2048]);  mul_16 = None
    permute_21: "f32[2048, 512]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
    
    # No stacktrace found for following nodes
    mm_default_116: "f32[128, 512]" = torch.ops.aten.mm.default(view_39, permute_21);  view_39 = permute_21 = None
    add_tensor_116: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_116, arg34_1);  mm_default_116 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_40: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_116, [1, 128, 512]);  add_tensor_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:332, code: hidden_states = residual + hidden_states
    add_14: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_12, view_40);  add_12 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:333, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    sub_6: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_14, getitem_9);  add_14 = getitem_9 = None
    add_15: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    mul_17: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_18: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_17, arg35_1);  mul_17 = arg35_1 = None
    add_16: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_18, arg36_1);  mul_18 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_41: "f32[128, 512]" = torch.ops.aten.reshape.default(add_16, [128, 512])
    permute_22: "f32[512, 512]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    
    # No stacktrace found for following nodes
    mm_default_115: "f32[128, 512]" = torch.ops.aten.mm.default(view_41, permute_22);  view_41 = permute_22 = None
    add_tensor_115: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_115, arg38_1);  mm_default_115 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_42: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_115, [1, 128, 512]);  add_tensor_115 = None
    mul_19: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_42, 0.1767766952966369);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_49: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_19, [1, 128, 16, 32]);  mul_19 = None
    permute_27: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    clone_19: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_50: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_19, [16, -1, 32]);  clone_19 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_39: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_50, 0);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_43: "f32[128, 512]" = torch.ops.aten.reshape.default(add_16, [128, 512])
    permute_23: "f32[512, 512]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    
    # No stacktrace found for following nodes
    mm_default_114: "f32[128, 512]" = torch.ops.aten.mm.default(view_43, permute_23);  view_43 = permute_23 = None
    add_tensor_114: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_114, arg40_1);  mm_default_114 = arg40_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_44: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_114, [1, 128, 512]);  add_tensor_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_45: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_44, [1, -1, 16, 32]);  view_44 = None
    permute_24: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    clone_17: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_51: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_17, [16, -1, 32]);  clone_17 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_40: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_51, 0);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_46: "f32[128, 512]" = torch.ops.aten.reshape.default(add_16, [128, 512])
    permute_25: "f32[512, 512]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    
    # No stacktrace found for following nodes
    mm_default_113: "f32[128, 512]" = torch.ops.aten.mm.default(view_46, permute_25);  view_46 = permute_25 = None
    add_tensor_113: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_113, arg42_1);  mm_default_113 = arg42_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_47: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_113, [1, 128, 512]);  add_tensor_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_48: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_47, [1, -1, 16, 32]);  view_47 = None
    permute_26: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
    clone_18: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_52: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_18, [16, -1, 32]);  clone_18 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_41: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_52, 0);  view_52 = None
    _scaled_dot_product_efficient_attention_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_39, unsqueeze_default_40, unsqueeze_default_41, None, True, scale = 1.0);  unsqueeze_default_39 = unsqueeze_default_40 = unsqueeze_default_41 = None
    getitem_97: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_13[0];  _scaled_dot_product_efficient_attention_default_13 = None
    squeeze_dim_13: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_97, 0);  getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_53: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_13, [1, 16, 128, 32]);  squeeze_dim_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_53, [0, 2, 1, 3]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_21: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_54: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_21, [1, 128, 512]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_55: "f32[128, 512]" = torch.ops.aten.reshape.default(view_54, [128, 512]);  view_54 = None
    permute_30: "f32[512, 512]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    
    # No stacktrace found for following nodes
    mm_default_112: "f32[128, 512]" = torch.ops.aten.mm.default(view_55, permute_30);  view_55 = permute_30 = None
    add_tensor_112: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_112, arg44_1);  mm_default_112 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_56: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_112, [1, 128, 512]);  add_tensor_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:324, code: hidden_states = residual + hidden_states
    add_17: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_16, view_56);  add_16 = view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:325, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_8: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_17, getitem_11);  add_17 = getitem_11 = None
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    mul_20: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_21: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_20, arg45_1);  mul_20 = arg45_1 = None
    add_19: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_21, arg46_1);  mul_21 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_57: "f32[128, 512]" = torch.ops.aten.reshape.default(add_19, [128, 512])
    permute_31: "f32[512, 2048]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    
    # No stacktrace found for following nodes
    mm_default_111: "f32[128, 2048]" = torch.ops.aten.mm.default(view_57, permute_31);  view_57 = permute_31 = None
    add_tensor_111: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_111, arg48_1);  mm_default_111 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_58: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_111, [1, 128, 2048]);  add_tensor_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_23: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
    erf_2: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_20: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_22, add_20);  mul_22 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_59: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_24, [128, 2048]);  mul_24 = None
    permute_32: "f32[2048, 512]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    
    # No stacktrace found for following nodes
    mm_default_110: "f32[128, 512]" = torch.ops.aten.mm.default(view_59, permute_32);  view_59 = permute_32 = None
    add_tensor_110: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_110, arg50_1);  mm_default_110 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_60: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_110, [1, 128, 512]);  add_tensor_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:332, code: hidden_states = residual + hidden_states
    add_21: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_19, view_60);  add_19 = view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:333, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    sub_9: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_21, getitem_13);  add_21 = getitem_13 = None
    add_22: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    mul_25: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_26: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_25, arg51_1);  mul_25 = arg51_1 = None
    add_23: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_26, arg52_1);  mul_26 = arg52_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_61: "f32[128, 512]" = torch.ops.aten.reshape.default(add_23, [128, 512])
    permute_33: "f32[512, 512]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    
    # No stacktrace found for following nodes
    mm_default_109: "f32[128, 512]" = torch.ops.aten.mm.default(view_61, permute_33);  view_61 = permute_33 = None
    add_tensor_109: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_109, arg54_1);  mm_default_109 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_62: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_109, [1, 128, 512]);  add_tensor_109 = None
    mul_27: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_62, 0.1767766952966369);  view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_69: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_27, [1, 128, 16, 32]);  mul_27 = None
    permute_38: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
    clone_27: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_70: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_27, [16, -1, 32]);  clone_27 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_36: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_70, 0);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_63: "f32[128, 512]" = torch.ops.aten.reshape.default(add_23, [128, 512])
    permute_34: "f32[512, 512]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    
    # No stacktrace found for following nodes
    mm_default_108: "f32[128, 512]" = torch.ops.aten.mm.default(view_63, permute_34);  view_63 = permute_34 = None
    add_tensor_108: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_108, arg56_1);  mm_default_108 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_64: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_108, [1, 128, 512]);  add_tensor_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_65: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_64, [1, -1, 16, 32]);  view_64 = None
    permute_35: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_25: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_71: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_25, [16, -1, 32]);  clone_25 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_37: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_71, 0);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_66: "f32[128, 512]" = torch.ops.aten.reshape.default(add_23, [128, 512])
    permute_36: "f32[512, 512]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    
    # No stacktrace found for following nodes
    mm_default_107: "f32[128, 512]" = torch.ops.aten.mm.default(view_66, permute_36);  view_66 = permute_36 = None
    add_tensor_107: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_107, arg58_1);  mm_default_107 = arg58_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_67: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_107, [1, 128, 512]);  add_tensor_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_68: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_67, [1, -1, 16, 32]);  view_67 = None
    permute_37: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    clone_26: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_72: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_26, [16, -1, 32]);  clone_26 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_38: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_72, 0);  view_72 = None
    _scaled_dot_product_efficient_attention_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_36, unsqueeze_default_37, unsqueeze_default_38, None, True, scale = 1.0);  unsqueeze_default_36 = unsqueeze_default_37 = unsqueeze_default_38 = None
    getitem_96: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_12[0];  _scaled_dot_product_efficient_attention_default_12 = None
    squeeze_dim_12: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_96, 0);  getitem_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_73: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_12, [1, 16, 128, 32]);  squeeze_dim_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_73, [0, 2, 1, 3]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_29: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_74: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_29, [1, 128, 512]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_75: "f32[128, 512]" = torch.ops.aten.reshape.default(view_74, [128, 512]);  view_74 = None
    permute_41: "f32[512, 512]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    
    # No stacktrace found for following nodes
    mm_default_106: "f32[128, 512]" = torch.ops.aten.mm.default(view_75, permute_41);  view_75 = permute_41 = None
    add_tensor_106: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_106, arg60_1);  mm_default_106 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_76: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_106, [1, 128, 512]);  add_tensor_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:324, code: hidden_states = residual + hidden_states
    add_24: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_23, view_76);  add_23 = view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:325, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_11: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_24, getitem_15);  add_24 = getitem_15 = None
    add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    mul_28: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_29: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_28, arg61_1);  mul_28 = arg61_1 = None
    add_26: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_29, arg62_1);  mul_29 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_77: "f32[128, 512]" = torch.ops.aten.reshape.default(add_26, [128, 512])
    permute_42: "f32[512, 2048]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    
    # No stacktrace found for following nodes
    mm_default_105: "f32[128, 2048]" = torch.ops.aten.mm.default(view_77, permute_42);  view_77 = permute_42 = None
    add_tensor_105: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_105, arg64_1);  mm_default_105 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_78: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_105, [1, 128, 2048]);  add_tensor_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_31: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
    erf_3: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_27: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_30, add_27);  mul_30 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_79: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_32, [128, 2048]);  mul_32 = None
    permute_43: "f32[2048, 512]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    
    # No stacktrace found for following nodes
    mm_default_104: "f32[128, 512]" = torch.ops.aten.mm.default(view_79, permute_43);  view_79 = permute_43 = None
    add_tensor_104: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_104, arg66_1);  mm_default_104 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_80: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_104, [1, 128, 512]);  add_tensor_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:332, code: hidden_states = residual + hidden_states
    add_28: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_26, view_80);  add_26 = view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:333, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_12: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_28, getitem_17);  add_28 = getitem_17 = None
    add_29: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    mul_33: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_34: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_33, arg67_1);  mul_33 = arg67_1 = None
    add_30: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_34, arg68_1);  mul_34 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_81: "f32[128, 512]" = torch.ops.aten.reshape.default(add_30, [128, 512])
    permute_44: "f32[512, 512]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    
    # No stacktrace found for following nodes
    mm_default_103: "f32[128, 512]" = torch.ops.aten.mm.default(view_81, permute_44);  view_81 = permute_44 = None
    add_tensor_103: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_103, arg70_1);  mm_default_103 = arg70_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_82: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_103, [1, 128, 512]);  add_tensor_103 = None
    mul_35: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_82, 0.1767766952966369);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_89: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_35, [1, 128, 16, 32]);  mul_35 = None
    permute_49: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    clone_35: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_90: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_35, [16, -1, 32]);  clone_35 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_33: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_90, 0);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_83: "f32[128, 512]" = torch.ops.aten.reshape.default(add_30, [128, 512])
    permute_45: "f32[512, 512]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    
    # No stacktrace found for following nodes
    mm_default_102: "f32[128, 512]" = torch.ops.aten.mm.default(view_83, permute_45);  view_83 = permute_45 = None
    add_tensor_102: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_102, arg72_1);  mm_default_102 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_84: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_102, [1, 128, 512]);  add_tensor_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_85: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_84, [1, -1, 16, 32]);  view_84 = None
    permute_46: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
    clone_33: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_91: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_33, [16, -1, 32]);  clone_33 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_34: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_91, 0);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_86: "f32[128, 512]" = torch.ops.aten.reshape.default(add_30, [128, 512])
    permute_47: "f32[512, 512]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    
    # No stacktrace found for following nodes
    mm_default_101: "f32[128, 512]" = torch.ops.aten.mm.default(view_86, permute_47);  view_86 = permute_47 = None
    add_tensor_101: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_101, arg74_1);  mm_default_101 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_87: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_101, [1, 128, 512]);  add_tensor_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_88: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_87, [1, -1, 16, 32]);  view_87 = None
    permute_48: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_88, [0, 2, 1, 3]);  view_88 = None
    clone_34: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_92: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_34, [16, -1, 32]);  clone_34 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_35: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_92, 0);  view_92 = None
    _scaled_dot_product_efficient_attention_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_33, unsqueeze_default_34, unsqueeze_default_35, None, True, scale = 1.0);  unsqueeze_default_33 = unsqueeze_default_34 = unsqueeze_default_35 = None
    getitem_95: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_11[0];  _scaled_dot_product_efficient_attention_default_11 = None
    squeeze_dim_11: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_95, 0);  getitem_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_93: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_11, [1, 16, 128, 32]);  squeeze_dim_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_37: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_94: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_37, [1, 128, 512]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_95: "f32[128, 512]" = torch.ops.aten.reshape.default(view_94, [128, 512]);  view_94 = None
    permute_52: "f32[512, 512]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    
    # No stacktrace found for following nodes
    mm_default_100: "f32[128, 512]" = torch.ops.aten.mm.default(view_95, permute_52);  view_95 = permute_52 = None
    add_tensor_100: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_100, arg76_1);  mm_default_100 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_96: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_100, [1, 128, 512]);  add_tensor_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:324, code: hidden_states = residual + hidden_states
    add_31: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_30, view_96);  add_30 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:325, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    sub_14: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_31, getitem_19);  add_31 = getitem_19 = None
    add_32: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    mul_36: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_37: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_36, arg77_1);  mul_36 = arg77_1 = None
    add_33: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_37, arg78_1);  mul_37 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_97: "f32[128, 512]" = torch.ops.aten.reshape.default(add_33, [128, 512])
    permute_53: "f32[512, 2048]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    
    # No stacktrace found for following nodes
    mm_default_99: "f32[128, 2048]" = torch.ops.aten.mm.default(view_97, permute_53);  view_97 = permute_53 = None
    add_tensor_99: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_99, arg80_1);  mm_default_99 = arg80_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_98: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_99, [1, 128, 2048]);  add_tensor_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_39: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
    erf_4: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_34: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_38, add_34);  mul_38 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_99: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_40, [128, 2048]);  mul_40 = None
    permute_54: "f32[2048, 512]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    
    # No stacktrace found for following nodes
    mm_default_98: "f32[128, 512]" = torch.ops.aten.mm.default(view_99, permute_54);  view_99 = permute_54 = None
    add_tensor_98: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_98, arg82_1);  mm_default_98 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_100: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_98, [1, 128, 512]);  add_tensor_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:332, code: hidden_states = residual + hidden_states
    add_35: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_33, view_100);  add_33 = view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:333, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_15: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_35, getitem_21);  add_35 = getitem_21 = None
    add_36: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    mul_41: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_42: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_41, arg83_1);  mul_41 = arg83_1 = None
    add_37: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_42, arg84_1);  mul_42 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_101: "f32[128, 512]" = torch.ops.aten.reshape.default(add_37, [128, 512])
    permute_55: "f32[512, 512]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    
    # No stacktrace found for following nodes
    mm_default_97: "f32[128, 512]" = torch.ops.aten.mm.default(view_101, permute_55);  view_101 = permute_55 = None
    add_tensor_97: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_97, arg86_1);  mm_default_97 = arg86_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_102: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_97, [1, 128, 512]);  add_tensor_97 = None
    mul_43: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_102, 0.1767766952966369);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_109: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_43, [1, 128, 16, 32]);  mul_43 = None
    permute_60: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
    clone_43: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_110: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_43, [16, -1, 32]);  clone_43 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_30: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_110, 0);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_103: "f32[128, 512]" = torch.ops.aten.reshape.default(add_37, [128, 512])
    permute_56: "f32[512, 512]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    
    # No stacktrace found for following nodes
    mm_default_96: "f32[128, 512]" = torch.ops.aten.mm.default(view_103, permute_56);  view_103 = permute_56 = None
    add_tensor_96: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_96, arg88_1);  mm_default_96 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_104: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_96, [1, 128, 512]);  add_tensor_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_105: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_104, [1, -1, 16, 32]);  view_104 = None
    permute_57: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
    clone_41: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_111: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_41, [16, -1, 32]);  clone_41 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_31: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_111, 0);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_106: "f32[128, 512]" = torch.ops.aten.reshape.default(add_37, [128, 512])
    permute_58: "f32[512, 512]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    
    # No stacktrace found for following nodes
    mm_default_95: "f32[128, 512]" = torch.ops.aten.mm.default(view_106, permute_58);  view_106 = permute_58 = None
    add_tensor_95: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_95, arg90_1);  mm_default_95 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_107: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_95, [1, 128, 512]);  add_tensor_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_108: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_107, [1, -1, 16, 32]);  view_107 = None
    permute_59: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_108, [0, 2, 1, 3]);  view_108 = None
    clone_42: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_112: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_42, [16, -1, 32]);  clone_42 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_32: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_112, 0);  view_112 = None
    _scaled_dot_product_efficient_attention_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_30, unsqueeze_default_31, unsqueeze_default_32, None, True, scale = 1.0);  unsqueeze_default_30 = unsqueeze_default_31 = unsqueeze_default_32 = None
    getitem_94: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_10[0];  _scaled_dot_product_efficient_attention_default_10 = None
    squeeze_dim_10: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_94, 0);  getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_113: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_10, [1, 16, 128, 32]);  squeeze_dim_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_45: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_114: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_45, [1, 128, 512]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_115: "f32[128, 512]" = torch.ops.aten.reshape.default(view_114, [128, 512]);  view_114 = None
    permute_63: "f32[512, 512]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    
    # No stacktrace found for following nodes
    mm_default_94: "f32[128, 512]" = torch.ops.aten.mm.default(view_115, permute_63);  view_115 = permute_63 = None
    add_tensor_94: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_94, arg92_1);  mm_default_94 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_116: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_94, [1, 128, 512]);  add_tensor_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:324, code: hidden_states = residual + hidden_states
    add_38: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_37, view_116);  add_37 = view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:325, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    sub_17: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_38, getitem_23);  add_38 = getitem_23 = None
    add_39: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    mul_44: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_45: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_44, arg93_1);  mul_44 = arg93_1 = None
    add_40: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_45, arg94_1);  mul_45 = arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_117: "f32[128, 512]" = torch.ops.aten.reshape.default(add_40, [128, 512])
    permute_64: "f32[512, 2048]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    
    # No stacktrace found for following nodes
    mm_default_93: "f32[128, 2048]" = torch.ops.aten.mm.default(view_117, permute_64);  view_117 = permute_64 = None
    add_tensor_93: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_93, arg96_1);  mm_default_93 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_118: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_93, [1, 128, 2048]);  add_tensor_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_47: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
    erf_5: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_41: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_46, add_41);  mul_46 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_119: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_48, [128, 2048]);  mul_48 = None
    permute_65: "f32[2048, 512]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    
    # No stacktrace found for following nodes
    mm_default_92: "f32[128, 512]" = torch.ops.aten.mm.default(view_119, permute_65);  view_119 = permute_65 = None
    add_tensor_92: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_92, arg98_1);  mm_default_92 = arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_120: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_92, [1, 128, 512]);  add_tensor_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:332, code: hidden_states = residual + hidden_states
    add_42: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_40, view_120);  add_40 = view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:333, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_18: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_42, getitem_25);  add_42 = getitem_25 = None
    add_43: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    mul_49: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_50: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_49, arg99_1);  mul_49 = arg99_1 = None
    add_44: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_50, arg100_1);  mul_50 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_121: "f32[128, 512]" = torch.ops.aten.reshape.default(add_44, [128, 512])
    permute_66: "f32[512, 512]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
    
    # No stacktrace found for following nodes
    mm_default_91: "f32[128, 512]" = torch.ops.aten.mm.default(view_121, permute_66);  view_121 = permute_66 = None
    add_tensor_91: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_91, arg102_1);  mm_default_91 = arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_122: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_91, [1, 128, 512]);  add_tensor_91 = None
    mul_51: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_122, 0.1767766952966369);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_129: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_51, [1, 128, 16, 32]);  mul_51 = None
    permute_71: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_129, [0, 2, 1, 3]);  view_129 = None
    clone_51: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_130: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_51, [16, -1, 32]);  clone_51 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_27: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_130, 0);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_123: "f32[128, 512]" = torch.ops.aten.reshape.default(add_44, [128, 512])
    permute_67: "f32[512, 512]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    
    # No stacktrace found for following nodes
    mm_default_90: "f32[128, 512]" = torch.ops.aten.mm.default(view_123, permute_67);  view_123 = permute_67 = None
    add_tensor_90: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_90, arg104_1);  mm_default_90 = arg104_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_124: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_90, [1, 128, 512]);  add_tensor_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_125: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_124, [1, -1, 16, 32]);  view_124 = None
    permute_68: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_125, [0, 2, 1, 3]);  view_125 = None
    clone_49: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_131: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_49, [16, -1, 32]);  clone_49 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_28: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_131, 0);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_126: "f32[128, 512]" = torch.ops.aten.reshape.default(add_44, [128, 512])
    permute_69: "f32[512, 512]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    
    # No stacktrace found for following nodes
    mm_default_89: "f32[128, 512]" = torch.ops.aten.mm.default(view_126, permute_69);  view_126 = permute_69 = None
    add_tensor_89: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_89, arg106_1);  mm_default_89 = arg106_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_127: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_89, [1, 128, 512]);  add_tensor_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_128: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_127, [1, -1, 16, 32]);  view_127 = None
    permute_70: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
    clone_50: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_132: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_50, [16, -1, 32]);  clone_50 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_29: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_132, 0);  view_132 = None
    _scaled_dot_product_efficient_attention_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_27, unsqueeze_default_28, unsqueeze_default_29, None, True, scale = 1.0);  unsqueeze_default_27 = unsqueeze_default_28 = unsqueeze_default_29 = None
    getitem_93: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_9[0];  _scaled_dot_product_efficient_attention_default_9 = None
    squeeze_dim_9: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_93, 0);  getitem_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_133: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_9, [1, 16, 128, 32]);  squeeze_dim_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_53: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_134: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_53, [1, 128, 512]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_135: "f32[128, 512]" = torch.ops.aten.reshape.default(view_134, [128, 512]);  view_134 = None
    permute_74: "f32[512, 512]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
    
    # No stacktrace found for following nodes
    mm_default_88: "f32[128, 512]" = torch.ops.aten.mm.default(view_135, permute_74);  view_135 = permute_74 = None
    add_tensor_88: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_88, arg108_1);  mm_default_88 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_136: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_88, [1, 128, 512]);  add_tensor_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:324, code: hidden_states = residual + hidden_states
    add_45: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_44, view_136);  add_44 = view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:325, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_20: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_45, getitem_27);  add_45 = getitem_27 = None
    add_46: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    mul_52: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_53: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_52, arg109_1);  mul_52 = arg109_1 = None
    add_47: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_53, arg110_1);  mul_53 = arg110_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_137: "f32[128, 512]" = torch.ops.aten.reshape.default(add_47, [128, 512])
    permute_75: "f32[512, 2048]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
    
    # No stacktrace found for following nodes
    mm_default_87: "f32[128, 2048]" = torch.ops.aten.mm.default(view_137, permute_75);  view_137 = permute_75 = None
    add_tensor_87: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_87, arg112_1);  mm_default_87 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_138: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_87, [1, 128, 2048]);  add_tensor_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_138, 0.5)
    mul_55: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476);  view_138 = None
    erf_6: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_48: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_56: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_54, add_48);  mul_54 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_139: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_56, [128, 2048]);  mul_56 = None
    permute_76: "f32[2048, 512]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    
    # No stacktrace found for following nodes
    mm_default_86: "f32[128, 512]" = torch.ops.aten.mm.default(view_139, permute_76);  view_139 = permute_76 = None
    add_tensor_86: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_86, arg114_1);  mm_default_86 = arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_140: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_86, [1, 128, 512]);  add_tensor_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:332, code: hidden_states = residual + hidden_states
    add_49: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_47, view_140);  add_47 = view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:333, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    sub_21: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_49, getitem_29);  add_49 = getitem_29 = None
    add_50: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    mul_57: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_58: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_57, arg115_1);  mul_57 = arg115_1 = None
    add_51: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_58, arg116_1);  mul_58 = arg116_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_141: "f32[128, 512]" = torch.ops.aten.reshape.default(add_51, [128, 512])
    permute_77: "f32[512, 512]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    
    # No stacktrace found for following nodes
    mm_default_85: "f32[128, 512]" = torch.ops.aten.mm.default(view_141, permute_77);  view_141 = permute_77 = None
    add_tensor_85: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_85, arg118_1);  mm_default_85 = arg118_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_142: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_85, [1, 128, 512]);  add_tensor_85 = None
    mul_59: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_142, 0.1767766952966369);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_149: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_59, [1, 128, 16, 32]);  mul_59 = None
    permute_82: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_59: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_150: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_59, [16, -1, 32]);  clone_59 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_24: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_150, 0);  view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_143: "f32[128, 512]" = torch.ops.aten.reshape.default(add_51, [128, 512])
    permute_78: "f32[512, 512]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    
    # No stacktrace found for following nodes
    mm_default_84: "f32[128, 512]" = torch.ops.aten.mm.default(view_143, permute_78);  view_143 = permute_78 = None
    add_tensor_84: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_84, arg120_1);  mm_default_84 = arg120_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_144: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_84, [1, 128, 512]);  add_tensor_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_145: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_144, [1, -1, 16, 32]);  view_144 = None
    permute_79: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
    clone_57: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_151: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_57, [16, -1, 32]);  clone_57 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_25: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_151, 0);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_146: "f32[128, 512]" = torch.ops.aten.reshape.default(add_51, [128, 512])
    permute_80: "f32[512, 512]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    
    # No stacktrace found for following nodes
    mm_default_83: "f32[128, 512]" = torch.ops.aten.mm.default(view_146, permute_80);  view_146 = permute_80 = None
    add_tensor_83: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_83, arg122_1);  mm_default_83 = arg122_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_147: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_83, [1, 128, 512]);  add_tensor_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_148: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_147, [1, -1, 16, 32]);  view_147 = None
    permute_81: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_58: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_152: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_58, [16, -1, 32]);  clone_58 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_26: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_152, 0);  view_152 = None
    _scaled_dot_product_efficient_attention_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_24, unsqueeze_default_25, unsqueeze_default_26, None, True, scale = 1.0);  unsqueeze_default_24 = unsqueeze_default_25 = unsqueeze_default_26 = None
    getitem_92: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_8[0];  _scaled_dot_product_efficient_attention_default_8 = None
    squeeze_dim_8: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_92, 0);  getitem_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_153: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_8, [1, 16, 128, 32]);  squeeze_dim_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_84: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_61: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_154: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_61, [1, 128, 512]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_155: "f32[128, 512]" = torch.ops.aten.reshape.default(view_154, [128, 512]);  view_154 = None
    permute_85: "f32[512, 512]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
    
    # No stacktrace found for following nodes
    mm_default_82: "f32[128, 512]" = torch.ops.aten.mm.default(view_155, permute_85);  view_155 = permute_85 = None
    add_tensor_82: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_82, arg124_1);  mm_default_82 = arg124_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_156: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_82, [1, 128, 512]);  add_tensor_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:324, code: hidden_states = residual + hidden_states
    add_52: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_51, view_156);  add_51 = view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:325, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_23: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_52, getitem_31);  add_52 = getitem_31 = None
    add_53: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    mul_60: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_61: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_60, arg125_1);  mul_60 = arg125_1 = None
    add_54: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_61, arg126_1);  mul_61 = arg126_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_157: "f32[128, 512]" = torch.ops.aten.reshape.default(add_54, [128, 512])
    permute_86: "f32[512, 2048]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    
    # No stacktrace found for following nodes
    mm_default_81: "f32[128, 2048]" = torch.ops.aten.mm.default(view_157, permute_86);  view_157 = permute_86 = None
    add_tensor_81: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_81, arg128_1);  mm_default_81 = arg128_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:328, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_158: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_81, [1, 128, 2048]);  add_tensor_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
    mul_63: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_158, 0.7071067811865476);  view_158 = None
    erf_7: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_55: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_64: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_62, add_55);  mul_62 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_159: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_64, [128, 2048]);  mul_64 = None
    permute_87: "f32[2048, 512]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
    
    # No stacktrace found for following nodes
    mm_default_80: "f32[128, 512]" = torch.ops.aten.mm.default(view_159, permute_87);  view_159 = permute_87 = None
    add_tensor_80: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_80, arg130_1);  mm_default_80 = arg130_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:330, code: hidden_states = self.fc2(hidden_states)
    view_160: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_80, [1, 128, 512]);  add_tensor_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:332, code: hidden_states = residual + hidden_states
    add_56: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_54, view_160);  add_54 = view_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:333, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:969, code: input_ids = input_ids.view(-1, input_shape[-1])
    view_161: "i64[1, 128]" = torch.ops.aten.reshape.default(arg346_1, [-1, 128]);  arg346_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:979, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding_2: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg2_1, view_161, 0);  arg2_1 = view_161 = None
    mul_67: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(embedding_2, 1.0);  embedding_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:994, code: inputs_embeds = self.layernorm_embedding(inputs_embeds)
    var_mean_17 = torch.ops.aten.var_mean.correction(mul_67, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_25: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(mul_67, getitem_35);  mul_67 = getitem_35 = None
    add_60: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    mul_68: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_17);  sub_25 = rsqrt_17 = None
    mul_69: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_68, arg133_1);  mul_68 = arg133_1 = None
    add_61: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_69, arg134_1);  mul_69 = arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:119, code: positions = torch.arange(
    iota_2: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/sparse.py:163, code: return F.embedding(
    embedding_3: "f32[128, 512]" = torch.ops.aten.embedding.default(arg1_1, iota_2);  arg1_1 = iota_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:995, code: hidden_states = inputs_embeds + positions
    add_62: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_61, embedding_3);  add_61 = embedding_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_163: "f32[128, 512]" = torch.ops.aten.reshape.default(add_62, [128, 512])
    permute_88: "f32[512, 512]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    
    # No stacktrace found for following nodes
    mm_default_79: "f32[128, 512]" = torch.ops.aten.mm.default(view_163, permute_88);  view_163 = permute_88 = None
    add_tensor_79: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_79, arg136_1);  mm_default_79 = arg136_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_164: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_79, [1, 128, 512]);  add_tensor_79 = None
    mul_70: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_164, 0.1767766952966369);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_171: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_70, [1, 128, 16, 32]);  mul_70 = None
    permute_93: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
    clone_68: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_172: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_68, [16, -1, 32]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_165: "f32[128, 512]" = torch.ops.aten.reshape.default(add_62, [128, 512])
    permute_89: "f32[512, 512]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    
    # No stacktrace found for following nodes
    mm_default_78: "f32[128, 512]" = torch.ops.aten.mm.default(view_165, permute_89);  view_165 = permute_89 = None
    add_tensor_78: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_78, arg138_1);  mm_default_78 = arg138_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_166: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_78, [1, 128, 512]);  add_tensor_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_167: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_166, [1, -1, 16, 32]);  view_166 = None
    permute_90: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
    clone_66: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_173: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_66, [16, -1, 32]);  clone_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_94: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    bmm_16: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_172, permute_94);  view_172 = permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_175: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [1, 16, 128, 128]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:83, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:84, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add_59: "i64[128]" = torch.ops.aten.add.Tensor(iota_1, 1)
    view_162: "i64[128, 1]" = torch.ops.aten.reshape.default(add_59, [128, 1]);  add_59 = None
    lt: "b8[128, 128]" = torch.ops.aten.lt.Tensor(iota_1, view_162);  iota_1 = view_162 = None
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:82, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full_default: "f32[128, 128]" = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:84, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    where: "f32[128, 128]" = torch.ops.aten.where.self(lt, full_default_1, full_default);  lt = full_default_1 = full_default = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    unsqueeze_2: "f32[1, 128, 128]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_3: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, 1);  unsqueeze_2 = None
    expand_1: "f32[1, 1, 128, 128]" = torch.ops.aten.expand.default(unsqueeze_3, [1, 1, 128, 128]);  unsqueeze_3 = None
    add_63: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_175, expand_1);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_176: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_63, [16, 128, 128]);  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_176, [-1], True)
    sub_26: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_176, amax_8);  view_176 = amax_8 = None
    exp_8: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_9: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_168: "f32[128, 512]" = torch.ops.aten.reshape.default(add_62, [128, 512])
    permute_91: "f32[512, 512]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
    
    # No stacktrace found for following nodes
    mm_default_77: "f32[128, 512]" = torch.ops.aten.mm.default(view_168, permute_91);  view_168 = permute_91 = None
    add_tensor_77: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_77, arg140_1);  mm_default_77 = arg140_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_169: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_77, [1, 128, 512]);  add_tensor_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_170: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_169, [1, -1, 16, 32]);  view_169 = None
    permute_92: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_67: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_174: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_67, [16, -1, 32]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_17: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(div_8, view_174);  div_8 = view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_177: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_17, [1, 16, 128, 32]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_95: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_70: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_178: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_70, [1, 128, 512]);  clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_179: "f32[128, 512]" = torch.ops.aten.reshape.default(view_178, [128, 512]);  view_178 = None
    permute_96: "f32[512, 512]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    
    # No stacktrace found for following nodes
    mm_default_76: "f32[128, 512]" = torch.ops.aten.mm.default(view_179, permute_96);  view_179 = permute_96 = None
    add_tensor_76: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_76, arg142_1);  mm_default_76 = arg142_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_180: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_76, [1, 128, 512]);  add_tensor_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_64: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_62, view_180);  add_62 = view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_27: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_64, getitem_37);  add_64 = getitem_37 = None
    add_65: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    mul_71: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
    mul_72: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_71, arg143_1);  mul_71 = arg143_1 = None
    add_66: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_72, arg144_1);  mul_72 = arg144_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_181: "f32[128, 512]" = torch.ops.aten.reshape.default(add_66, [128, 512])
    permute_97: "f32[512, 512]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
    
    # No stacktrace found for following nodes
    mm_default_75: "f32[128, 512]" = torch.ops.aten.mm.default(view_181, permute_97);  view_181 = permute_97 = None
    add_tensor_75: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_75, arg146_1);  mm_default_75 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_182: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_75, [1, 128, 512]);  add_tensor_75 = None
    mul_73: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_182, 0.1767766952966369);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_189: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_73, [1, 128, 16, 32]);  mul_73 = None
    permute_102: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
    clone_74: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_102, memory_format = torch.contiguous_format);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_190: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_74, [16, -1, 32]);  clone_74 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_21: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_190, 0);  view_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:333, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_24: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_56, getitem_33);  add_56 = getitem_33 = None
    add_57: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    mul_65: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_66: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_65, arg131_1);  mul_65 = arg131_1 = None
    add_58: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_66, arg132_1);  mul_66 = arg132_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_183: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_98: "f32[512, 512]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    
    # No stacktrace found for following nodes
    mm_default_74: "f32[128, 512]" = torch.ops.aten.mm.default(view_183, permute_98);  view_183 = permute_98 = None
    add_tensor_74: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_74, arg148_1);  mm_default_74 = arg148_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_184: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_74, [1, 128, 512]);  add_tensor_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_185: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_184, [1, -1, 16, 32]);  view_184 = None
    permute_99: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    clone_72: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_99, memory_format = torch.contiguous_format);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_191: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_72, [16, -1, 32]);  clone_72 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_22: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_191, 0);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_186: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_100: "f32[512, 512]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    
    # No stacktrace found for following nodes
    mm_default_73: "f32[128, 512]" = torch.ops.aten.mm.default(view_186, permute_100);  view_186 = permute_100 = None
    add_tensor_73: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_73, arg150_1);  mm_default_73 = arg150_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_187: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_73, [1, 128, 512]);  add_tensor_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_188: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_187, [1, -1, 16, 32]);  view_187 = None
    permute_101: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
    clone_73: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_192: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_73, [16, -1, 32]);  clone_73 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_23: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_192, 0);  view_192 = None
    _scaled_dot_product_efficient_attention_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_21, unsqueeze_default_22, unsqueeze_default_23, None, True, scale = 1.0);  unsqueeze_default_21 = unsqueeze_default_22 = unsqueeze_default_23 = None
    getitem_91: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_7[0];  _scaled_dot_product_efficient_attention_default_7 = None
    squeeze_dim_7: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_91, 0);  getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_193: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_7, [1, 16, 128, 32]);  squeeze_dim_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_104: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_76: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_194: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_76, [1, 128, 512]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_195: "f32[128, 512]" = torch.ops.aten.reshape.default(view_194, [128, 512]);  view_194 = None
    permute_105: "f32[512, 512]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    
    # No stacktrace found for following nodes
    mm_default_72: "f32[128, 512]" = torch.ops.aten.mm.default(view_195, permute_105);  view_195 = permute_105 = None
    add_tensor_72: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_72, arg152_1);  mm_default_72 = arg152_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_196: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_72, [1, 128, 512]);  add_tensor_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:441, code: hidden_states = residual + hidden_states
    add_67: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_66, view_196);  add_66 = view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:442, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
    sub_29: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_67, getitem_39);  add_67 = getitem_39 = None
    add_68: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    mul_74: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_75: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_74, arg153_1);  mul_74 = arg153_1 = None
    add_69: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_75, arg154_1);  mul_75 = arg154_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_197: "f32[128, 512]" = torch.ops.aten.reshape.default(add_69, [128, 512])
    permute_106: "f32[512, 2048]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    
    # No stacktrace found for following nodes
    mm_default_71: "f32[128, 2048]" = torch.ops.aten.mm.default(view_197, permute_106);  view_197 = permute_106 = None
    add_tensor_71: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_71, arg156_1);  mm_default_71 = arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_198: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_71, [1, 128, 2048]);  add_tensor_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_76: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_198, 0.5)
    mul_77: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_198, 0.7071067811865476);  view_198 = None
    erf_8: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
    add_70: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_78: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_76, add_70);  mul_76 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_199: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_78, [128, 2048]);  mul_78 = None
    permute_107: "f32[2048, 512]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    
    # No stacktrace found for following nodes
    mm_default_70: "f32[128, 512]" = torch.ops.aten.mm.default(view_199, permute_107);  view_199 = permute_107 = None
    add_tensor_70: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_70, arg158_1);  mm_default_70 = arg158_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_200: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_70, [1, 128, 512]);  add_tensor_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_71: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_69, view_200);  add_69 = view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_30: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_71, getitem_41);  add_71 = getitem_41 = None
    add_72: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    mul_79: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
    mul_80: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_79, arg159_1);  mul_79 = arg159_1 = None
    add_73: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_80, arg160_1);  mul_80 = arg160_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_201: "f32[128, 512]" = torch.ops.aten.reshape.default(add_73, [128, 512])
    permute_108: "f32[512, 512]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    
    # No stacktrace found for following nodes
    mm_default_69: "f32[128, 512]" = torch.ops.aten.mm.default(view_201, permute_108);  view_201 = permute_108 = None
    add_tensor_69: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_69, arg162_1);  mm_default_69 = arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_202: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_69, [1, 128, 512]);  add_tensor_69 = None
    mul_81: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_202, 0.1767766952966369);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_209: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_81, [1, 128, 16, 32]);  mul_81 = None
    permute_113: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
    clone_82: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_210: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_82, [16, -1, 32]);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_203: "f32[128, 512]" = torch.ops.aten.reshape.default(add_73, [128, 512])
    permute_109: "f32[512, 512]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    
    # No stacktrace found for following nodes
    mm_default_68: "f32[128, 512]" = torch.ops.aten.mm.default(view_203, permute_109);  view_203 = permute_109 = None
    add_tensor_68: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_68, arg164_1);  mm_default_68 = arg164_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_204: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_68, [1, 128, 512]);  add_tensor_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_205: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_204, [1, -1, 16, 32]);  view_204 = None
    permute_110: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_205, [0, 2, 1, 3]);  view_205 = None
    clone_80: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_211: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_80, [16, -1, 32]);  clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_114: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
    bmm_20: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_210, permute_114);  view_210 = permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_213: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [1, 16, 128, 128]);  bmm_20 = None
    add_74: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_213, expand_1);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_214: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_74, [16, 128, 128]);  add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_214, [-1], True)
    sub_31: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_214, amax_10);  view_214 = amax_10 = None
    exp_10: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_206: "f32[128, 512]" = torch.ops.aten.reshape.default(add_73, [128, 512])
    permute_111: "f32[512, 512]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    
    # No stacktrace found for following nodes
    mm_default_67: "f32[128, 512]" = torch.ops.aten.mm.default(view_206, permute_111);  view_206 = permute_111 = None
    add_tensor_67: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_67, arg166_1);  mm_default_67 = arg166_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_207: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_67, [1, 128, 512]);  add_tensor_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_208: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_207, [1, -1, 16, 32]);  view_207 = None
    permute_112: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_208, [0, 2, 1, 3]);  view_208 = None
    clone_81: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_212: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_81, [16, -1, 32]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_21: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(div_10, view_212);  div_10 = view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_215: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_21, [1, 16, 128, 32]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_115: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_84: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    view_216: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_84, [1, 128, 512]);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_217: "f32[128, 512]" = torch.ops.aten.reshape.default(view_216, [128, 512]);  view_216 = None
    permute_116: "f32[512, 512]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    
    # No stacktrace found for following nodes
    mm_default_66: "f32[128, 512]" = torch.ops.aten.mm.default(view_217, permute_116);  view_217 = permute_116 = None
    add_tensor_66: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_66, arg168_1);  mm_default_66 = arg168_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_218: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_66, [1, 128, 512]);  add_tensor_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_75: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_73, view_218);  add_73 = view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 128, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
    sub_32: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_75, getitem_43);  add_75 = getitem_43 = None
    add_76: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    mul_82: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_83: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_82, arg169_1);  mul_82 = arg169_1 = None
    add_77: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_83, arg170_1);  mul_83 = arg170_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_219: "f32[128, 512]" = torch.ops.aten.reshape.default(add_77, [128, 512])
    permute_117: "f32[512, 512]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    
    # No stacktrace found for following nodes
    mm_default_65: "f32[128, 512]" = torch.ops.aten.mm.default(view_219, permute_117);  view_219 = permute_117 = None
    add_tensor_65: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_65, arg172_1);  mm_default_65 = arg172_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_220: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_65, [1, 128, 512]);  add_tensor_65 = None
    mul_84: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_220, 0.1767766952966369);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_227: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_84, [1, 128, 16, 32]);  mul_84 = None
    permute_122: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_227, [0, 2, 1, 3]);  view_227 = None
    clone_88: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_122, memory_format = torch.contiguous_format);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_228: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_88, [16, -1, 32]);  clone_88 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_18: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_228, 0);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_221: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_118: "f32[512, 512]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    
    # No stacktrace found for following nodes
    mm_default_64: "f32[128, 512]" = torch.ops.aten.mm.default(view_221, permute_118);  view_221 = permute_118 = None
    add_tensor_64: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_64, arg174_1);  mm_default_64 = arg174_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_222: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_64, [1, 128, 512]);  add_tensor_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_223: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_222, [1, -1, 16, 32]);  view_222 = None
    permute_119: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
    clone_86: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_119, memory_format = torch.contiguous_format);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_229: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_86, [16, -1, 32]);  clone_86 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_19: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_229, 0);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_224: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_120: "f32[512, 512]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    
    # No stacktrace found for following nodes
    mm_default_63: "f32[128, 512]" = torch.ops.aten.mm.default(view_224, permute_120);  view_224 = permute_120 = None
    add_tensor_63: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_63, arg176_1);  mm_default_63 = arg176_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_225: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_63, [1, 128, 512]);  add_tensor_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_226: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_225, [1, -1, 16, 32]);  view_225 = None
    permute_121: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_87: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_230: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_87, [16, -1, 32]);  clone_87 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_20: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_230, 0);  view_230 = None
    _scaled_dot_product_efficient_attention_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_18, unsqueeze_default_19, unsqueeze_default_20, None, True, scale = 1.0);  unsqueeze_default_18 = unsqueeze_default_19 = unsqueeze_default_20 = None
    getitem_90: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_6[0];  _scaled_dot_product_efficient_attention_default_6 = None
    squeeze_dim_6: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_90, 0);  getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_231: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_6, [1, 16, 128, 32]);  squeeze_dim_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_124: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_90: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
    view_232: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_90, [1, 128, 512]);  clone_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_233: "f32[128, 512]" = torch.ops.aten.reshape.default(view_232, [128, 512]);  view_232 = None
    permute_125: "f32[512, 512]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
    
    # No stacktrace found for following nodes
    mm_default_62: "f32[128, 512]" = torch.ops.aten.mm.default(view_233, permute_125);  view_233 = permute_125 = None
    add_tensor_62: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_62, arg178_1);  mm_default_62 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_234: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_62, [1, 128, 512]);  add_tensor_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:441, code: hidden_states = residual + hidden_states
    add_78: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_77, view_234);  add_77 = view_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:442, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_34: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_78, getitem_45);  add_78 = getitem_45 = None
    add_79: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    mul_85: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = rsqrt_22 = None
    mul_86: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_85, arg179_1);  mul_85 = arg179_1 = None
    add_80: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_86, arg180_1);  mul_86 = arg180_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_235: "f32[128, 512]" = torch.ops.aten.reshape.default(add_80, [128, 512])
    permute_126: "f32[512, 2048]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    
    # No stacktrace found for following nodes
    mm_default_61: "f32[128, 2048]" = torch.ops.aten.mm.default(view_235, permute_126);  view_235 = permute_126 = None
    add_tensor_61: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_61, arg182_1);  mm_default_61 = arg182_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_236: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_61, [1, 128, 2048]);  add_tensor_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_87: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_236, 0.5)
    mul_88: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_236, 0.7071067811865476);  view_236 = None
    erf_9: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_88);  mul_88 = None
    add_81: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_89: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_87, add_81);  mul_87 = add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_237: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_89, [128, 2048]);  mul_89 = None
    permute_127: "f32[2048, 512]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    
    # No stacktrace found for following nodes
    mm_default_60: "f32[128, 512]" = torch.ops.aten.mm.default(view_237, permute_127);  view_237 = permute_127 = None
    add_tensor_60: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_60, arg184_1);  mm_default_60 = arg184_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_238: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_60, [1, 128, 512]);  add_tensor_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_82: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_80, view_238);  add_80 = view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_35: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_82, getitem_47);  add_82 = getitem_47 = None
    add_83: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    mul_90: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_91: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_90, arg185_1);  mul_90 = arg185_1 = None
    add_84: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_91, arg186_1);  mul_91 = arg186_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_239: "f32[128, 512]" = torch.ops.aten.reshape.default(add_84, [128, 512])
    permute_128: "f32[512, 512]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    
    # No stacktrace found for following nodes
    mm_default_59: "f32[128, 512]" = torch.ops.aten.mm.default(view_239, permute_128);  view_239 = permute_128 = None
    add_tensor_59: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_59, arg188_1);  mm_default_59 = arg188_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_240: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_59, [1, 128, 512]);  add_tensor_59 = None
    mul_92: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_240, 0.1767766952966369);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_247: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_92, [1, 128, 16, 32]);  mul_92 = None
    permute_133: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
    clone_96: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_248: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_96, [16, -1, 32]);  clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_241: "f32[128, 512]" = torch.ops.aten.reshape.default(add_84, [128, 512])
    permute_129: "f32[512, 512]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    
    # No stacktrace found for following nodes
    mm_default_58: "f32[128, 512]" = torch.ops.aten.mm.default(view_241, permute_129);  view_241 = permute_129 = None
    add_tensor_58: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_58, arg190_1);  mm_default_58 = arg190_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_242: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_58, [1, 128, 512]);  add_tensor_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_243: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_242, [1, -1, 16, 32]);  view_242 = None
    permute_130: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
    clone_94: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_249: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_94, [16, -1, 32]);  clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_134: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None
    bmm_24: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_248, permute_134);  view_248 = permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_251: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [1, 16, 128, 128]);  bmm_24 = None
    add_85: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_251, expand_1);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_252: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_85, [16, 128, 128]);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_252, [-1], True)
    sub_36: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_252, amax_12);  view_252 = amax_12 = None
    exp_12: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_36);  sub_36 = None
    sum_13: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_244: "f32[128, 512]" = torch.ops.aten.reshape.default(add_84, [128, 512])
    permute_131: "f32[512, 512]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
    
    # No stacktrace found for following nodes
    mm_default_57: "f32[128, 512]" = torch.ops.aten.mm.default(view_244, permute_131);  view_244 = permute_131 = None
    add_tensor_57: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_57, arg192_1);  mm_default_57 = arg192_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_245: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_57, [1, 128, 512]);  add_tensor_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_246: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_245, [1, -1, 16, 32]);  view_245 = None
    permute_132: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    clone_95: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_250: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_95, [16, -1, 32]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_25: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(div_12, view_250);  div_12 = view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_253: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_25, [1, 16, 128, 32]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_135: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_98: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
    view_254: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_98, [1, 128, 512]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_255: "f32[128, 512]" = torch.ops.aten.reshape.default(view_254, [128, 512]);  view_254 = None
    permute_136: "f32[512, 512]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    
    # No stacktrace found for following nodes
    mm_default_56: "f32[128, 512]" = torch.ops.aten.mm.default(view_255, permute_136);  view_255 = permute_136 = None
    add_tensor_56: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_56, arg194_1);  mm_default_56 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_256: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_56, [1, 128, 512]);  add_tensor_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_86: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_84, view_256);  add_84 = view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
    sub_37: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_86, getitem_49);  add_86 = getitem_49 = None
    add_87: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    mul_93: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = rsqrt_24 = None
    mul_94: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_93, arg195_1);  mul_93 = arg195_1 = None
    add_88: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_94, arg196_1);  mul_94 = arg196_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_257: "f32[128, 512]" = torch.ops.aten.reshape.default(add_88, [128, 512])
    permute_137: "f32[512, 512]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    
    # No stacktrace found for following nodes
    mm_default_55: "f32[128, 512]" = torch.ops.aten.mm.default(view_257, permute_137);  view_257 = permute_137 = None
    add_tensor_55: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_55, arg198_1);  mm_default_55 = arg198_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_258: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_55, [1, 128, 512]);  add_tensor_55 = None
    mul_95: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_258, 0.1767766952966369);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_265: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_95, [1, 128, 16, 32]);  mul_95 = None
    permute_142: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
    clone_102: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_266: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_102, [16, -1, 32]);  clone_102 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_15: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_266, 0);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_259: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_138: "f32[512, 512]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
    
    # No stacktrace found for following nodes
    mm_default_54: "f32[128, 512]" = torch.ops.aten.mm.default(view_259, permute_138);  view_259 = permute_138 = None
    add_tensor_54: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_54, arg200_1);  mm_default_54 = arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_260: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_54, [1, 128, 512]);  add_tensor_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_261: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_260, [1, -1, 16, 32]);  view_260 = None
    permute_139: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
    clone_100: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_267: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_100, [16, -1, 32]);  clone_100 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_16: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_267, 0);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_262: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_140: "f32[512, 512]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
    
    # No stacktrace found for following nodes
    mm_default_53: "f32[128, 512]" = torch.ops.aten.mm.default(view_262, permute_140);  view_262 = permute_140 = None
    add_tensor_53: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_53, arg202_1);  mm_default_53 = arg202_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_263: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_53, [1, 128, 512]);  add_tensor_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_264: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_263, [1, -1, 16, 32]);  view_263 = None
    permute_141: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
    clone_101: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_141, memory_format = torch.contiguous_format);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_268: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_101, [16, -1, 32]);  clone_101 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_17: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_268, 0);  view_268 = None
    _scaled_dot_product_efficient_attention_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_15, unsqueeze_default_16, unsqueeze_default_17, None, True, scale = 1.0);  unsqueeze_default_15 = unsqueeze_default_16 = unsqueeze_default_17 = None
    getitem_89: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_5[0];  _scaled_dot_product_efficient_attention_default_5 = None
    squeeze_dim_5: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_89, 0);  getitem_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_269: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_5, [1, 16, 128, 32]);  squeeze_dim_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_144: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_104: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    view_270: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_104, [1, 128, 512]);  clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_271: "f32[128, 512]" = torch.ops.aten.reshape.default(view_270, [128, 512]);  view_270 = None
    permute_145: "f32[512, 512]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
    
    # No stacktrace found for following nodes
    mm_default_52: "f32[128, 512]" = torch.ops.aten.mm.default(view_271, permute_145);  view_271 = permute_145 = None
    add_tensor_52: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_52, arg204_1);  mm_default_52 = arg204_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_272: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_52, [1, 128, 512]);  add_tensor_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:441, code: hidden_states = residual + hidden_states
    add_89: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_88, view_272);  add_88 = view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:442, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
    sub_39: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_89, getitem_51);  add_89 = getitem_51 = None
    add_90: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    mul_96: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_25);  sub_39 = rsqrt_25 = None
    mul_97: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_96, arg205_1);  mul_96 = arg205_1 = None
    add_91: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_97, arg206_1);  mul_97 = arg206_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_273: "f32[128, 512]" = torch.ops.aten.reshape.default(add_91, [128, 512])
    permute_146: "f32[512, 2048]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
    
    # No stacktrace found for following nodes
    mm_default_51: "f32[128, 2048]" = torch.ops.aten.mm.default(view_273, permute_146);  view_273 = permute_146 = None
    add_tensor_51: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_51, arg208_1);  mm_default_51 = arg208_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_274: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_51, [1, 128, 2048]);  add_tensor_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_98: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_274, 0.5)
    mul_99: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_274, 0.7071067811865476);  view_274 = None
    erf_10: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_99);  mul_99 = None
    add_92: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_100: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_98, add_92);  mul_98 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_275: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_100, [128, 2048]);  mul_100 = None
    permute_147: "f32[2048, 512]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
    
    # No stacktrace found for following nodes
    mm_default_50: "f32[128, 512]" = torch.ops.aten.mm.default(view_275, permute_147);  view_275 = permute_147 = None
    add_tensor_50: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_50, arg210_1);  mm_default_50 = arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_276: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_50, [1, 128, 512]);  add_tensor_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_93: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_91, view_276);  add_91 = view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_93, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
    sub_40: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_93, getitem_53);  add_93 = getitem_53 = None
    add_94: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    mul_101: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = rsqrt_26 = None
    mul_102: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_101, arg211_1);  mul_101 = arg211_1 = None
    add_95: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_102, arg212_1);  mul_102 = arg212_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_277: "f32[128, 512]" = torch.ops.aten.reshape.default(add_95, [128, 512])
    permute_148: "f32[512, 512]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
    
    # No stacktrace found for following nodes
    mm_default_49: "f32[128, 512]" = torch.ops.aten.mm.default(view_277, permute_148);  view_277 = permute_148 = None
    add_tensor_49: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_49, arg214_1);  mm_default_49 = arg214_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_278: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_49, [1, 128, 512]);  add_tensor_49 = None
    mul_103: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_278, 0.1767766952966369);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_285: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_103, [1, 128, 16, 32]);  mul_103 = None
    permute_153: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
    clone_110: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_153, memory_format = torch.contiguous_format);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_286: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_110, [16, -1, 32]);  clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_279: "f32[128, 512]" = torch.ops.aten.reshape.default(add_95, [128, 512])
    permute_149: "f32[512, 512]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
    
    # No stacktrace found for following nodes
    mm_default_48: "f32[128, 512]" = torch.ops.aten.mm.default(view_279, permute_149);  view_279 = permute_149 = None
    add_tensor_48: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_48, arg216_1);  mm_default_48 = arg216_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_280: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_48, [1, 128, 512]);  add_tensor_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_281: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_280, [1, -1, 16, 32]);  view_280 = None
    permute_150: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None
    clone_108: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_287: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_108, [16, -1, 32]);  clone_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_154: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None
    bmm_28: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_286, permute_154);  view_286 = permute_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_289: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [1, 16, 128, 128]);  bmm_28 = None
    add_96: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_289, expand_1);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_290: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_96, [16, 128, 128]);  add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_290, [-1], True)
    sub_41: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_290, amax_14);  view_290 = amax_14 = None
    exp_14: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
    sum_15: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_282: "f32[128, 512]" = torch.ops.aten.reshape.default(add_95, [128, 512])
    permute_151: "f32[512, 512]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
    
    # No stacktrace found for following nodes
    mm_default_47: "f32[128, 512]" = torch.ops.aten.mm.default(view_282, permute_151);  view_282 = permute_151 = None
    add_tensor_47: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_47, arg218_1);  mm_default_47 = arg218_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_283: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_47, [1, 128, 512]);  add_tensor_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_284: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_283, [1, -1, 16, 32]);  view_283 = None
    permute_152: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
    clone_109: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_288: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_109, [16, -1, 32]);  clone_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_29: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(div_14, view_288);  div_14 = view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_291: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_29, [1, 16, 128, 32]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_155: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_291, [0, 2, 1, 3]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_112: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_155, memory_format = torch.contiguous_format);  permute_155 = None
    view_292: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_112, [1, 128, 512]);  clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_293: "f32[128, 512]" = torch.ops.aten.reshape.default(view_292, [128, 512]);  view_292 = None
    permute_156: "f32[512, 512]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
    
    # No stacktrace found for following nodes
    mm_default_46: "f32[128, 512]" = torch.ops.aten.mm.default(view_293, permute_156);  view_293 = permute_156 = None
    add_tensor_46: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_46, arg220_1);  mm_default_46 = arg220_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_294: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_46, [1, 128, 512]);  add_tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_97: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_95, view_294);  add_95 = view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_97, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 128, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
    sub_42: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_97, getitem_55);  add_97 = getitem_55 = None
    add_98: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    mul_104: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_27);  sub_42 = rsqrt_27 = None
    mul_105: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_104, arg221_1);  mul_104 = arg221_1 = None
    add_99: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_105, arg222_1);  mul_105 = arg222_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_295: "f32[128, 512]" = torch.ops.aten.reshape.default(add_99, [128, 512])
    permute_157: "f32[512, 512]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
    
    # No stacktrace found for following nodes
    mm_default_45: "f32[128, 512]" = torch.ops.aten.mm.default(view_295, permute_157);  view_295 = permute_157 = None
    add_tensor_45: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_45, arg224_1);  mm_default_45 = arg224_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_296: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_45, [1, 128, 512]);  add_tensor_45 = None
    mul_106: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_296, 0.1767766952966369);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_303: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_106, [1, 128, 16, 32]);  mul_106 = None
    permute_162: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
    clone_116: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_304: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_116, [16, -1, 32]);  clone_116 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_12: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_304, 0);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_297: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_158: "f32[512, 512]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
    
    # No stacktrace found for following nodes
    mm_default_44: "f32[128, 512]" = torch.ops.aten.mm.default(view_297, permute_158);  view_297 = permute_158 = None
    add_tensor_44: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_44, arg226_1);  mm_default_44 = arg226_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_298: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_44, [1, 128, 512]);  add_tensor_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_299: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_298, [1, -1, 16, 32]);  view_298 = None
    permute_159: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_299, [0, 2, 1, 3]);  view_299 = None
    clone_114: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_305: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_114, [16, -1, 32]);  clone_114 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_13: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_305, 0);  view_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_300: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_160: "f32[512, 512]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
    
    # No stacktrace found for following nodes
    mm_default_43: "f32[128, 512]" = torch.ops.aten.mm.default(view_300, permute_160);  view_300 = permute_160 = None
    add_tensor_43: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_43, arg228_1);  mm_default_43 = arg228_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_301: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_43, [1, 128, 512]);  add_tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_302: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_301, [1, -1, 16, 32]);  view_301 = None
    permute_161: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
    clone_115: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_306: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_115, [16, -1, 32]);  clone_115 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_14: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_306, 0);  view_306 = None
    _scaled_dot_product_efficient_attention_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_12, unsqueeze_default_13, unsqueeze_default_14, None, True, scale = 1.0);  unsqueeze_default_12 = unsqueeze_default_13 = unsqueeze_default_14 = None
    getitem_88: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_4[0];  _scaled_dot_product_efficient_attention_default_4 = None
    squeeze_dim_4: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_88, 0);  getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_307: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_4, [1, 16, 128, 32]);  squeeze_dim_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_164: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_118: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    view_308: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_118, [1, 128, 512]);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_309: "f32[128, 512]" = torch.ops.aten.reshape.default(view_308, [128, 512]);  view_308 = None
    permute_165: "f32[512, 512]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
    
    # No stacktrace found for following nodes
    mm_default_42: "f32[128, 512]" = torch.ops.aten.mm.default(view_309, permute_165);  view_309 = permute_165 = None
    add_tensor_42: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_42, arg230_1);  mm_default_42 = arg230_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_310: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_42, [1, 128, 512]);  add_tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:441, code: hidden_states = residual + hidden_states
    add_100: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_99, view_310);  add_99 = view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:442, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_100, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
    sub_44: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_100, getitem_57);  add_100 = getitem_57 = None
    add_101: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    mul_107: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_28);  sub_44 = rsqrt_28 = None
    mul_108: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_107, arg231_1);  mul_107 = arg231_1 = None
    add_102: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_108, arg232_1);  mul_108 = arg232_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_311: "f32[128, 512]" = torch.ops.aten.reshape.default(add_102, [128, 512])
    permute_166: "f32[512, 2048]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
    
    # No stacktrace found for following nodes
    mm_default_41: "f32[128, 2048]" = torch.ops.aten.mm.default(view_311, permute_166);  view_311 = permute_166 = None
    add_tensor_41: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_41, arg234_1);  mm_default_41 = arg234_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_312: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_41, [1, 128, 2048]);  add_tensor_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_109: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_312, 0.5)
    mul_110: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_312, 0.7071067811865476);  view_312 = None
    erf_11: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_110);  mul_110 = None
    add_103: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_111: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_109, add_103);  mul_109 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_313: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_111, [128, 2048]);  mul_111 = None
    permute_167: "f32[2048, 512]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
    
    # No stacktrace found for following nodes
    mm_default_40: "f32[128, 512]" = torch.ops.aten.mm.default(view_313, permute_167);  view_313 = permute_167 = None
    add_tensor_40: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_40, arg236_1);  mm_default_40 = arg236_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_314: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_40, [1, 128, 512]);  add_tensor_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_104: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_102, view_314);  add_102 = view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_104, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 128, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 128, 1]" = var_mean_29[1];  var_mean_29 = None
    sub_45: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_104, getitem_59);  add_104 = getitem_59 = None
    add_105: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    mul_112: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_29);  sub_45 = rsqrt_29 = None
    mul_113: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_112, arg237_1);  mul_112 = arg237_1 = None
    add_106: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_113, arg238_1);  mul_113 = arg238_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_315: "f32[128, 512]" = torch.ops.aten.reshape.default(add_106, [128, 512])
    permute_168: "f32[512, 512]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
    
    # No stacktrace found for following nodes
    mm_default_39: "f32[128, 512]" = torch.ops.aten.mm.default(view_315, permute_168);  view_315 = permute_168 = None
    add_tensor_39: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_39, arg240_1);  mm_default_39 = arg240_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_316: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_39, [1, 128, 512]);  add_tensor_39 = None
    mul_114: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_316, 0.1767766952966369);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_323: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_114, [1, 128, 16, 32]);  mul_114 = None
    permute_173: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None
    clone_124: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_173, memory_format = torch.contiguous_format);  permute_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_324: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_124, [16, -1, 32]);  clone_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_317: "f32[128, 512]" = torch.ops.aten.reshape.default(add_106, [128, 512])
    permute_169: "f32[512, 512]" = torch.ops.aten.permute.default(arg241_1, [1, 0]);  arg241_1 = None
    
    # No stacktrace found for following nodes
    mm_default_38: "f32[128, 512]" = torch.ops.aten.mm.default(view_317, permute_169);  view_317 = permute_169 = None
    add_tensor_38: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_38, arg242_1);  mm_default_38 = arg242_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_318: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_38, [1, 128, 512]);  add_tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_319: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_318, [1, -1, 16, 32]);  view_318 = None
    permute_170: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_319, [0, 2, 1, 3]);  view_319 = None
    clone_122: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_325: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_122, [16, -1, 32]);  clone_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_174: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None
    bmm_32: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_324, permute_174);  view_324 = permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_327: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [1, 16, 128, 128]);  bmm_32 = None
    add_107: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_327, expand_1);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_328: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_107, [16, 128, 128]);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_328, [-1], True)
    sub_46: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_328, amax_16);  view_328 = amax_16 = None
    exp_16: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_17: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_320: "f32[128, 512]" = torch.ops.aten.reshape.default(add_106, [128, 512])
    permute_171: "f32[512, 512]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
    
    # No stacktrace found for following nodes
    mm_default_37: "f32[128, 512]" = torch.ops.aten.mm.default(view_320, permute_171);  view_320 = permute_171 = None
    add_tensor_37: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_37, arg244_1);  mm_default_37 = arg244_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_321: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_37, [1, 128, 512]);  add_tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_322: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_321, [1, -1, 16, 32]);  view_321 = None
    permute_172: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_322, [0, 2, 1, 3]);  view_322 = None
    clone_123: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_326: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_123, [16, -1, 32]);  clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_33: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(div_16, view_326);  div_16 = view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_329: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_33, [1, 16, 128, 32]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_175: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_329, [0, 2, 1, 3]);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_126: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_175, memory_format = torch.contiguous_format);  permute_175 = None
    view_330: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_126, [1, 128, 512]);  clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_331: "f32[128, 512]" = torch.ops.aten.reshape.default(view_330, [128, 512]);  view_330 = None
    permute_176: "f32[512, 512]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[128, 512]" = torch.ops.aten.mm.default(view_331, permute_176);  view_331 = permute_176 = None
    add_tensor_36: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_36, arg246_1);  mm_default_36 = arg246_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_332: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_36, [1, 128, 512]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_108: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_106, view_332);  add_106 = view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_108, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 128, 1]" = var_mean_30[1];  var_mean_30 = None
    sub_47: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_108, getitem_61);  add_108 = getitem_61 = None
    add_109: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_109);  add_109 = None
    mul_115: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_30);  sub_47 = rsqrt_30 = None
    mul_116: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_115, arg247_1);  mul_115 = arg247_1 = None
    add_110: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_116, arg248_1);  mul_116 = arg248_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_333: "f32[128, 512]" = torch.ops.aten.reshape.default(add_110, [128, 512])
    permute_177: "f32[512, 512]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[128, 512]" = torch.ops.aten.mm.default(view_333, permute_177);  view_333 = permute_177 = None
    add_tensor_35: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_35, arg250_1);  mm_default_35 = arg250_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_334: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_35, [1, 128, 512]);  add_tensor_35 = None
    mul_117: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_334, 0.1767766952966369);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_341: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_117, [1, 128, 16, 32]);  mul_117 = None
    permute_182: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
    clone_130: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_182, memory_format = torch.contiguous_format);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_342: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_130, [16, -1, 32]);  clone_130 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_9: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_342, 0);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_335: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_178: "f32[512, 512]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[128, 512]" = torch.ops.aten.mm.default(view_335, permute_178);  view_335 = permute_178 = None
    add_tensor_34: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_34, arg252_1);  mm_default_34 = arg252_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_336: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_34, [1, 128, 512]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_337: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_336, [1, -1, 16, 32]);  view_336 = None
    permute_179: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_337, [0, 2, 1, 3]);  view_337 = None
    clone_128: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_179, memory_format = torch.contiguous_format);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_343: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_128, [16, -1, 32]);  clone_128 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_10: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_343, 0);  view_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_338: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_180: "f32[512, 512]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[128, 512]" = torch.ops.aten.mm.default(view_338, permute_180);  view_338 = permute_180 = None
    add_tensor_33: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_33, arg254_1);  mm_default_33 = arg254_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_339: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_33, [1, 128, 512]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_340: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_339, [1, -1, 16, 32]);  view_339 = None
    permute_181: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
    clone_129: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_344: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_129, [16, -1, 32]);  clone_129 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_11: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_344, 0);  view_344 = None
    _scaled_dot_product_efficient_attention_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_9, unsqueeze_default_10, unsqueeze_default_11, None, True, scale = 1.0);  unsqueeze_default_9 = unsqueeze_default_10 = unsqueeze_default_11 = None
    getitem_87: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_3[0];  _scaled_dot_product_efficient_attention_default_3 = None
    squeeze_dim_3: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_87, 0);  getitem_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_345: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_3, [1, 16, 128, 32]);  squeeze_dim_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_184: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_132: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_184, memory_format = torch.contiguous_format);  permute_184 = None
    view_346: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_132, [1, 128, 512]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_347: "f32[128, 512]" = torch.ops.aten.reshape.default(view_346, [128, 512]);  view_346 = None
    permute_185: "f32[512, 512]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[128, 512]" = torch.ops.aten.mm.default(view_347, permute_185);  view_347 = permute_185 = None
    add_tensor_32: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_32, arg256_1);  mm_default_32 = arg256_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_348: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_32, [1, 128, 512]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:441, code: hidden_states = residual + hidden_states
    add_111: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_110, view_348);  add_110 = view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:442, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_111, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 128, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 128, 1]" = var_mean_31[1];  var_mean_31 = None
    sub_49: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_111, getitem_63);  add_111 = getitem_63 = None
    add_112: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_112);  add_112 = None
    mul_118: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_31);  sub_49 = rsqrt_31 = None
    mul_119: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_118, arg257_1);  mul_118 = arg257_1 = None
    add_113: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_119, arg258_1);  mul_119 = arg258_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_349: "f32[128, 512]" = torch.ops.aten.reshape.default(add_113, [128, 512])
    permute_186: "f32[512, 2048]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[128, 2048]" = torch.ops.aten.mm.default(view_349, permute_186);  view_349 = permute_186 = None
    add_tensor_31: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_31, arg260_1);  mm_default_31 = arg260_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_350: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_31, [1, 128, 2048]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_120: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_350, 0.5)
    mul_121: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_350, 0.7071067811865476);  view_350 = None
    erf_12: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_121);  mul_121 = None
    add_114: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_122: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_120, add_114);  mul_120 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_351: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_122, [128, 2048]);  mul_122 = None
    permute_187: "f32[2048, 512]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[128, 512]" = torch.ops.aten.mm.default(view_351, permute_187);  view_351 = permute_187 = None
    add_tensor_30: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_30, arg262_1);  mm_default_30 = arg262_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_352: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_30, [1, 128, 512]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_115: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_113, view_352);  add_113 = view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_115, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 128, 1]" = var_mean_32[1];  var_mean_32 = None
    sub_50: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_115, getitem_65);  add_115 = getitem_65 = None
    add_116: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_116);  add_116 = None
    mul_123: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_32);  sub_50 = rsqrt_32 = None
    mul_124: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_123, arg263_1);  mul_123 = arg263_1 = None
    add_117: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_124, arg264_1);  mul_124 = arg264_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_353: "f32[128, 512]" = torch.ops.aten.reshape.default(add_117, [128, 512])
    permute_188: "f32[512, 512]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[128, 512]" = torch.ops.aten.mm.default(view_353, permute_188);  view_353 = permute_188 = None
    add_tensor_29: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_29, arg266_1);  mm_default_29 = arg266_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_354: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_29, [1, 128, 512]);  add_tensor_29 = None
    mul_125: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_354, 0.1767766952966369);  view_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_361: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_125, [1, 128, 16, 32]);  mul_125 = None
    permute_193: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
    clone_138: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_362: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_138, [16, -1, 32]);  clone_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_355: "f32[128, 512]" = torch.ops.aten.reshape.default(add_117, [128, 512])
    permute_189: "f32[512, 512]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[128, 512]" = torch.ops.aten.mm.default(view_355, permute_189);  view_355 = permute_189 = None
    add_tensor_28: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_28, arg268_1);  mm_default_28 = arg268_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_356: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_28, [1, 128, 512]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_357: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_356, [1, -1, 16, 32]);  view_356 = None
    permute_190: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None
    clone_136: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_363: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_136, [16, -1, 32]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_194: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None
    bmm_36: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_362, permute_194);  view_362 = permute_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_365: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [1, 16, 128, 128]);  bmm_36 = None
    add_118: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_365, expand_1);  view_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_366: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_118, [16, 128, 128]);  add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_18: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_366, [-1], True)
    sub_51: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_366, amax_18);  view_366 = amax_18 = None
    exp_18: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_51);  sub_51 = None
    sum_19: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_358: "f32[128, 512]" = torch.ops.aten.reshape.default(add_117, [128, 512])
    permute_191: "f32[512, 512]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[128, 512]" = torch.ops.aten.mm.default(view_358, permute_191);  view_358 = permute_191 = None
    add_tensor_27: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_27, arg270_1);  mm_default_27 = arg270_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_359: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_27, [1, 128, 512]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_360: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_359, [1, -1, 16, 32]);  view_359 = None
    permute_192: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_360, [0, 2, 1, 3]);  view_360 = None
    clone_137: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_364: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_137, [16, -1, 32]);  clone_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_37: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(div_18, view_364);  div_18 = view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_367: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_37, [1, 16, 128, 32]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_195: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_140: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_195, memory_format = torch.contiguous_format);  permute_195 = None
    view_368: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_140, [1, 128, 512]);  clone_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_369: "f32[128, 512]" = torch.ops.aten.reshape.default(view_368, [128, 512]);  view_368 = None
    permute_196: "f32[512, 512]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[128, 512]" = torch.ops.aten.mm.default(view_369, permute_196);  view_369 = permute_196 = None
    add_tensor_26: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_26, arg272_1);  mm_default_26 = arg272_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_370: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_26, [1, 128, 512]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_119: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_117, view_370);  add_117 = view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 128, 1]" = var_mean_33[1];  var_mean_33 = None
    sub_52: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_119, getitem_67);  add_119 = getitem_67 = None
    add_120: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    mul_126: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_33);  sub_52 = rsqrt_33 = None
    mul_127: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_126, arg273_1);  mul_126 = arg273_1 = None
    add_121: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_127, arg274_1);  mul_127 = arg274_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_371: "f32[128, 512]" = torch.ops.aten.reshape.default(add_121, [128, 512])
    permute_197: "f32[512, 512]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[128, 512]" = torch.ops.aten.mm.default(view_371, permute_197);  view_371 = permute_197 = None
    add_tensor_25: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_25, arg276_1);  mm_default_25 = arg276_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_372: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_25, [1, 128, 512]);  add_tensor_25 = None
    mul_128: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_372, 0.1767766952966369);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_379: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_128, [1, 128, 16, 32]);  mul_128 = None
    permute_202: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
    clone_144: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_380: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_144, [16, -1, 32]);  clone_144 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_6: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_380, 0);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_373: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_198: "f32[512, 512]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[128, 512]" = torch.ops.aten.mm.default(view_373, permute_198);  view_373 = permute_198 = None
    add_tensor_24: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_24, arg278_1);  mm_default_24 = arg278_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_374: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_24, [1, 128, 512]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_375: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_374, [1, -1, 16, 32]);  view_374 = None
    permute_199: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
    clone_142: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_381: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_142, [16, -1, 32]);  clone_142 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_7: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_381, 0);  view_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_376: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_200: "f32[512, 512]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[128, 512]" = torch.ops.aten.mm.default(view_376, permute_200);  view_376 = permute_200 = None
    add_tensor_23: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_23, arg280_1);  mm_default_23 = arg280_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_377: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_23, [1, 128, 512]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_378: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_377, [1, -1, 16, 32]);  view_377 = None
    permute_201: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
    clone_143: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_201, memory_format = torch.contiguous_format);  permute_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_382: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_143, [16, -1, 32]);  clone_143 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_8: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_382, 0);  view_382 = None
    _scaled_dot_product_efficient_attention_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_6, unsqueeze_default_7, unsqueeze_default_8, None, True, scale = 1.0);  unsqueeze_default_6 = unsqueeze_default_7 = unsqueeze_default_8 = None
    getitem_86: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_2[0];  _scaled_dot_product_efficient_attention_default_2 = None
    squeeze_dim_2: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_86, 0);  getitem_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_383: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_2, [1, 16, 128, 32]);  squeeze_dim_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_204: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_146: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
    view_384: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_146, [1, 128, 512]);  clone_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_385: "f32[128, 512]" = torch.ops.aten.reshape.default(view_384, [128, 512]);  view_384 = None
    permute_205: "f32[512, 512]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[128, 512]" = torch.ops.aten.mm.default(view_385, permute_205);  view_385 = permute_205 = None
    add_tensor_22: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_22, arg282_1);  mm_default_22 = arg282_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_386: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_22, [1, 128, 512]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:441, code: hidden_states = residual + hidden_states
    add_122: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_121, view_386);  add_121 = view_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:442, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 128, 1]" = var_mean_34[1];  var_mean_34 = None
    sub_54: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_122, getitem_69);  add_122 = getitem_69 = None
    add_123: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    mul_129: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_34);  sub_54 = rsqrt_34 = None
    mul_130: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_129, arg283_1);  mul_129 = arg283_1 = None
    add_124: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_130, arg284_1);  mul_130 = arg284_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_387: "f32[128, 512]" = torch.ops.aten.reshape.default(add_124, [128, 512])
    permute_206: "f32[512, 2048]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[128, 2048]" = torch.ops.aten.mm.default(view_387, permute_206);  view_387 = permute_206 = None
    add_tensor_21: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_21, arg286_1);  mm_default_21 = arg286_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_388: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_21, [1, 128, 2048]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_131: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_388, 0.5)
    mul_132: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_388, 0.7071067811865476);  view_388 = None
    erf_13: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_132);  mul_132 = None
    add_125: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_133: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_131, add_125);  mul_131 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_389: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_133, [128, 2048]);  mul_133 = None
    permute_207: "f32[2048, 512]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[128, 512]" = torch.ops.aten.mm.default(view_389, permute_207);  view_389 = permute_207 = None
    add_tensor_20: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_20, arg288_1);  mm_default_20 = arg288_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_390: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_20, [1, 128, 512]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_126: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_124, view_390);  add_124 = view_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_126, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 128, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 128, 1]" = var_mean_35[1];  var_mean_35 = None
    sub_55: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_126, getitem_71);  add_126 = getitem_71 = None
    add_127: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    mul_134: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_35);  sub_55 = rsqrt_35 = None
    mul_135: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_134, arg289_1);  mul_134 = arg289_1 = None
    add_128: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_135, arg290_1);  mul_135 = arg290_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_391: "f32[128, 512]" = torch.ops.aten.reshape.default(add_128, [128, 512])
    permute_208: "f32[512, 512]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[128, 512]" = torch.ops.aten.mm.default(view_391, permute_208);  view_391 = permute_208 = None
    add_tensor_19: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_19, arg292_1);  mm_default_19 = arg292_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_392: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_19, [1, 128, 512]);  add_tensor_19 = None
    mul_136: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_392, 0.1767766952966369);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_399: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_136, [1, 128, 16, 32]);  mul_136 = None
    permute_213: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
    clone_152: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_400: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_152, [16, -1, 32]);  clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_393: "f32[128, 512]" = torch.ops.aten.reshape.default(add_128, [128, 512])
    permute_209: "f32[512, 512]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[128, 512]" = torch.ops.aten.mm.default(view_393, permute_209);  view_393 = permute_209 = None
    add_tensor_18: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_18, arg294_1);  mm_default_18 = arg294_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_394: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_18, [1, 128, 512]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_395: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_394, [1, -1, 16, 32]);  view_394 = None
    permute_210: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_395, [0, 2, 1, 3]);  view_395 = None
    clone_150: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_401: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_150, [16, -1, 32]);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_214: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_401, [0, 2, 1]);  view_401 = None
    bmm_40: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_400, permute_214);  view_400 = permute_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_403: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [1, 16, 128, 128]);  bmm_40 = None
    add_129: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_403, expand_1);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_404: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_129, [16, 128, 128]);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_20: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_404, [-1], True)
    sub_56: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_404, amax_20);  view_404 = amax_20 = None
    exp_20: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
    sum_21: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_396: "f32[128, 512]" = torch.ops.aten.reshape.default(add_128, [128, 512])
    permute_211: "f32[512, 512]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[128, 512]" = torch.ops.aten.mm.default(view_396, permute_211);  view_396 = permute_211 = None
    add_tensor_17: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_17, arg296_1);  mm_default_17 = arg296_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_397: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_17, [1, 128, 512]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_398: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_397, [1, -1, 16, 32]);  view_397 = None
    permute_212: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None
    clone_151: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_212, memory_format = torch.contiguous_format);  permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_402: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_151, [16, -1, 32]);  clone_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_41: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(div_20, view_402);  div_20 = view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_405: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_41, [1, 16, 128, 32]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_215: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_154: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_215, memory_format = torch.contiguous_format);  permute_215 = None
    view_406: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_154, [1, 128, 512]);  clone_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_407: "f32[128, 512]" = torch.ops.aten.reshape.default(view_406, [128, 512]);  view_406 = None
    permute_216: "f32[512, 512]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[128, 512]" = torch.ops.aten.mm.default(view_407, permute_216);  view_407 = permute_216 = None
    add_tensor_16: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_16, arg298_1);  mm_default_16 = arg298_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_408: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_16, [1, 128, 512]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_130: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_128, view_408);  add_128 = view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_130, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 128, 1]" = var_mean_36[1];  var_mean_36 = None
    sub_57: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_130, getitem_73);  add_130 = getitem_73 = None
    add_131: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    mul_137: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_36);  sub_57 = rsqrt_36 = None
    mul_138: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_137, arg299_1);  mul_137 = arg299_1 = None
    add_132: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_138, arg300_1);  mul_138 = arg300_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_409: "f32[128, 512]" = torch.ops.aten.reshape.default(add_132, [128, 512])
    permute_217: "f32[512, 512]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[128, 512]" = torch.ops.aten.mm.default(view_409, permute_217);  view_409 = permute_217 = None
    add_tensor_15: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_15, arg302_1);  mm_default_15 = arg302_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_410: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_15, [1, 128, 512]);  add_tensor_15 = None
    mul_139: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_410, 0.1767766952966369);  view_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_417: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_139, [1, 128, 16, 32]);  mul_139 = None
    permute_222: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_417, [0, 2, 1, 3]);  view_417 = None
    clone_158: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_418: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_158, [16, -1, 32]);  clone_158 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_3: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_418, 0);  view_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_411: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_218: "f32[512, 512]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[128, 512]" = torch.ops.aten.mm.default(view_411, permute_218);  view_411 = permute_218 = None
    add_tensor_14: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_14, arg304_1);  mm_default_14 = arg304_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_412: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_14, [1, 128, 512]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_413: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_412, [1, -1, 16, 32]);  view_412 = None
    permute_219: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
    clone_156: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_419: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_156, [16, -1, 32]);  clone_156 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_4: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_419, 0);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_414: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_220: "f32[512, 512]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[128, 512]" = torch.ops.aten.mm.default(view_414, permute_220);  view_414 = permute_220 = None
    add_tensor_13: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_13, arg306_1);  mm_default_13 = arg306_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_415: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_13, [1, 128, 512]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_416: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_415, [1, -1, 16, 32]);  view_415 = None
    permute_221: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
    clone_157: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_221, memory_format = torch.contiguous_format);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_420: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_157, [16, -1, 32]);  clone_157 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_5: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_420, 0);  view_420 = None
    _scaled_dot_product_efficient_attention_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default_3, unsqueeze_default_4, unsqueeze_default_5, None, True, scale = 1.0);  unsqueeze_default_3 = unsqueeze_default_4 = unsqueeze_default_5 = None
    getitem_85: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default_1[0];  _scaled_dot_product_efficient_attention_default_1 = None
    squeeze_dim_1: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_85, 0);  getitem_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_421: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim_1, [1, 16, 128, 32]);  squeeze_dim_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_224: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_160: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
    view_422: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_160, [1, 128, 512]);  clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_423: "f32[128, 512]" = torch.ops.aten.reshape.default(view_422, [128, 512]);  view_422 = None
    permute_225: "f32[512, 512]" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[128, 512]" = torch.ops.aten.mm.default(view_423, permute_225);  view_423 = permute_225 = None
    add_tensor_12: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_12, arg308_1);  mm_default_12 = arg308_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_424: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_12, [1, 128, 512]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:441, code: hidden_states = residual + hidden_states
    add_133: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_132, view_424);  add_132 = view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:442, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 128, 1]" = var_mean_37[1];  var_mean_37 = None
    sub_59: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_133, getitem_75);  add_133 = getitem_75 = None
    add_134: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    mul_140: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_37);  sub_59 = rsqrt_37 = None
    mul_141: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_140, arg309_1);  mul_140 = arg309_1 = None
    add_135: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_141, arg310_1);  mul_141 = arg310_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_425: "f32[128, 512]" = torch.ops.aten.reshape.default(add_135, [128, 512])
    permute_226: "f32[512, 2048]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[128, 2048]" = torch.ops.aten.mm.default(view_425, permute_226);  view_425 = permute_226 = None
    add_tensor_11: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_11, arg312_1);  mm_default_11 = arg312_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_426: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_11, [1, 128, 2048]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_142: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_426, 0.5)
    mul_143: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_426, 0.7071067811865476);  view_426 = None
    erf_14: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_143);  mul_143 = None
    add_136: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_144: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_142, add_136);  mul_142 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_427: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_144, [128, 2048]);  mul_144 = None
    permute_227: "f32[2048, 512]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[128, 512]" = torch.ops.aten.mm.default(view_427, permute_227);  view_427 = permute_227 = None
    add_tensor_10: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_10, arg314_1);  mm_default_10 = arg314_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_428: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_10, [1, 128, 512]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_137: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_135, view_428);  add_135 = view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_137, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 128, 1]" = var_mean_38[1];  var_mean_38 = None
    sub_60: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_137, getitem_77);  add_137 = getitem_77 = None
    add_138: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    mul_145: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_38);  sub_60 = rsqrt_38 = None
    mul_146: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_145, arg315_1);  mul_145 = arg315_1 = None
    add_139: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_146, arg316_1);  mul_146 = arg316_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_429: "f32[128, 512]" = torch.ops.aten.reshape.default(add_139, [128, 512])
    permute_228: "f32[512, 512]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[128, 512]" = torch.ops.aten.mm.default(view_429, permute_228);  view_429 = permute_228 = None
    add_tensor_9: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_9, arg318_1);  mm_default_9 = arg318_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_430: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_9, [1, 128, 512]);  add_tensor_9 = None
    mul_147: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_430, 0.1767766952966369);  view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_437: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_147, [1, 128, 16, 32]);  mul_147 = None
    permute_233: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None
    clone_166: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_438: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_166, [16, -1, 32]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_431: "f32[128, 512]" = torch.ops.aten.reshape.default(add_139, [128, 512])
    permute_229: "f32[512, 512]" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[128, 512]" = torch.ops.aten.mm.default(view_431, permute_229);  view_431 = permute_229 = None
    add_tensor_8: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_8, arg320_1);  mm_default_8 = arg320_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:202, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_432: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_8, [1, 128, 512]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_433: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_432, [1, -1, 16, 32]);  view_432 = None
    permute_230: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_433, [0, 2, 1, 3]);  view_433 = None
    clone_164: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_439: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_164, [16, -1, 32]);  clone_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:221, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_234: "f32[16, 32, 128]" = torch.ops.aten.permute.default(view_439, [0, 2, 1]);  view_439 = None
    bmm_44: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_438, permute_234);  view_438 = permute_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:234, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_441: "f32[1, 16, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [1, 16, 128, 128]);  bmm_44 = None
    add_140: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_441, expand_1);  view_441 = expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:235, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_442: "f32[16, 128, 128]" = torch.ops.aten.reshape.default(add_140, [16, 128, 128]);  add_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:237, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_22: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_442, [-1], True)
    sub_61: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_442, amax_22);  view_442 = amax_22 = None
    exp_22: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_23: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_434: "f32[128, 512]" = torch.ops.aten.reshape.default(add_139, [128, 512])
    permute_231: "f32[512, 512]" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[128, 512]" = torch.ops.aten.mm.default(view_434, permute_231);  view_434 = permute_231 = None
    add_tensor_7: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_7, arg322_1);  mm_default_7 = arg322_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:203, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_435: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_7, [1, 128, 512]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_436: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_435, [1, -1, 16, 32]);  view_435 = None
    permute_232: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_436, [0, 2, 1, 3]);  view_436 = None
    clone_165: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_232, memory_format = torch.contiguous_format);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_440: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_165, [16, -1, 32]);  clone_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:260, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_45: "f32[16, 128, 32]" = torch.ops.aten.bmm.default(div_22, view_440);  div_22 = view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_443: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(bmm_45, [1, 16, 128, 32]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_235: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_168: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
    view_444: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_168, [1, 128, 512]);  clone_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_445: "f32[128, 512]" = torch.ops.aten.reshape.default(view_444, [128, 512]);  view_444 = None
    permute_236: "f32[512, 512]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[128, 512]" = torch.ops.aten.mm.default(view_445, permute_236);  view_445 = permute_236 = None
    add_tensor_6: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_6, arg324_1);  mm_default_6 = arg324_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_446: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_6, [1, 128, 512]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:421, code: hidden_states = residual + hidden_states
    add_141: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_139, view_446);  add_139 = view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:422, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_141, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 128, 1]" = var_mean_39[1];  var_mean_39 = None
    sub_62: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_141, getitem_79);  add_141 = getitem_79 = None
    add_142: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_142);  add_142 = None
    mul_148: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_39);  sub_62 = rsqrt_39 = None
    mul_149: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_148, arg325_1);  mul_148 = arg325_1 = None
    add_143: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_149, arg326_1);  mul_149 = arg326_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_447: "f32[128, 512]" = torch.ops.aten.reshape.default(add_143, [128, 512])
    permute_237: "f32[512, 512]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[128, 512]" = torch.ops.aten.mm.default(view_447, permute_237);  view_447 = permute_237 = None
    add_tensor_5: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_5, arg328_1);  mm_default_5 = arg328_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:177, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_448: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_5, [1, 128, 512]);  add_tensor_5 = None
    mul_150: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(view_448, 0.1767766952966369);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_455: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(mul_150, [1, 128, 16, 32]);  mul_150 = None
    permute_242: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_455, [0, 2, 1, 3]);  view_455 = None
    clone_172: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_242, memory_format = torch.contiguous_format);  permute_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:216, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_456: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_172, [16, -1, 32]);  clone_172 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_456, 0);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_449: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_238: "f32[512, 512]" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[128, 512]" = torch.ops.aten.mm.default(view_449, permute_238);  view_449 = permute_238 = None
    add_tensor_4: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_4, arg330_1);  mm_default_4 = arg330_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:192, code: key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
    view_450: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_4, [1, 128, 512]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_451: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_450, [1, -1, 16, 32]);  view_450 = None
    permute_239: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_451, [0, 2, 1, 3]);  view_451 = None
    clone_170: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_239, memory_format = torch.contiguous_format);  permute_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:217, code: key_states = key_states.reshape(*proj_shape)
    view_457: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_170, [16, -1, 32]);  clone_170 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_1: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_457, 0);  view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_452: "f32[128, 512]" = torch.ops.aten.reshape.default(add_58, [128, 512])
    permute_240: "f32[512, 512]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[128, 512]" = torch.ops.aten.mm.default(view_452, permute_240);  view_452 = permute_240 = None
    add_tensor_3: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_3, arg332_1);  mm_default_3 = arg332_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:193, code: value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
    view_453: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_3, [1, 128, 512]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:157, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_454: "f32[1, 128, 16, 32]" = torch.ops.aten.reshape.default(view_453, [1, -1, 16, 32]);  view_453 = None
    permute_241: "f32[1, 16, 128, 32]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None
    clone_171: "f32[1, 16, 128, 32]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:218, code: value_states = value_states.reshape(*proj_shape)
    view_458: "f32[16, 128, 32]" = torch.ops.aten.reshape.default(clone_171, [16, -1, 32]);  clone_171 = None
    
    # No stacktrace found for following nodes
    unsqueeze_default_2: "f32[1, 16, 128, 32]" = torch.ops.aten.unsqueeze.default(view_458, 0);  view_458 = None
    _scaled_dot_product_efficient_attention_default = torch.ops.aten._scaled_dot_product_efficient_attention.default(unsqueeze_default, unsqueeze_default_1, unsqueeze_default_2, None, True, scale = 1.0);  unsqueeze_default = unsqueeze_default_1 = unsqueeze_default_2 = None
    getitem_84: "f32[1, 16, 128, 32]" = _scaled_dot_product_efficient_attention_default[0];  _scaled_dot_product_efficient_attention_default = None
    squeeze_dim: "f32[16, 128, 32]" = torch.ops.aten.squeeze.dim(getitem_84, 0);  getitem_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:268, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_459: "f32[1, 16, 128, 32]" = torch.ops.aten.reshape.default(squeeze_dim, [1, 16, 128, 32]);  squeeze_dim = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:269, code: attn_output = attn_output.transpose(1, 2)
    permute_244: "f32[1, 128, 16, 32]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:273, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_174: "f32[1, 128, 16, 32]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
    view_460: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(clone_174, [1, 128, 512]);  clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_461: "f32[128, 512]" = torch.ops.aten.reshape.default(view_460, [128, 512]);  view_460 = None
    permute_245: "f32[512, 512]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[128, 512]" = torch.ops.aten.mm.default(view_461, permute_245);  view_461 = permute_245 = None
    add_tensor_2: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default_2, arg334_1);  mm_default_2 = arg334_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:275, code: attn_output = self.out_proj(attn_output)
    view_462: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor_2, [1, 128, 512]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:441, code: hidden_states = residual + hidden_states
    add_144: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_143, view_462);  add_143 = view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:442, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_144, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 128, 1]" = var_mean_40[1];  var_mean_40 = None
    sub_64: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_144, getitem_81);  add_144 = getitem_81 = None
    add_145: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    mul_151: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_40);  sub_64 = rsqrt_40 = None
    mul_152: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_151, arg335_1);  mul_151 = arg335_1 = None
    add_146: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_152, arg336_1);  mul_152 = arg336_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_463: "f32[128, 512]" = torch.ops.aten.reshape.default(add_146, [128, 512])
    permute_246: "f32[512, 2048]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[128, 2048]" = torch.ops.aten.mm.default(view_463, permute_246);  view_463 = permute_246 = None
    add_tensor_1: "f32[128, 2048]" = torch.ops.aten.add.Tensor(mm_default_1, arg338_1);  mm_default_1 = arg338_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:449, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_464: "f32[1, 128, 2048]" = torch.ops.aten.reshape.default(add_tensor_1, [1, 128, 2048]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_153: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_464, 0.5)
    mul_154: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(view_464, 0.7071067811865476);  view_464 = None
    erf_15: "f32[1, 128, 2048]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
    add_147: "f32[1, 128, 2048]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_155: "f32[1, 128, 2048]" = torch.ops.aten.mul.Tensor(mul_153, add_147);  mul_153 = add_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_465: "f32[128, 2048]" = torch.ops.aten.reshape.default(mul_155, [128, 2048]);  mul_155 = None
    permute_247: "f32[2048, 512]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[128, 512]" = torch.ops.aten.mm.default(view_465, permute_247);  view_465 = permute_247 = None
    add_tensor: "f32[128, 512]" = torch.ops.aten.add.Tensor(mm_default, arg340_1);  mm_default = arg340_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:451, code: hidden_states = self.fc2(hidden_states)
    view_466: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(add_tensor, [1, 128, 512]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:453, code: hidden_states = residual + hidden_states
    add_148: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_146, view_466);  add_146 = view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_148, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 128, 1]" = var_mean_41[1];  var_mean_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:1330, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_470: "i64[128]" = torch.ops.aten.reshape.default(arg345_1, [-1]);  arg345_1 = None
    ne_1: "b8[128]" = torch.ops.aten.ne.Scalar(view_470, -100)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:454, code: hidden_states = self.final_layer_norm(hidden_states)
    sub_65: "f32[1, 128, 512]" = torch.ops.aten.sub.Tensor(add_148, getitem_83);  add_148 = getitem_83 = None
    add_149: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_149);  add_149 = None
    mul_156: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_41);  sub_65 = rsqrt_41 = None
    mul_157: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(mul_156, arg341_1);  mul_156 = arg341_1 = None
    add_150: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(mul_157, arg342_1);  mul_157 = arg342_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:1325, code: lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
    view_467: "f32[128, 512]" = torch.ops.aten.reshape.default(add_150, [128, 512]);  add_150 = None
    permute_248: "f32[512, 50265]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
    mm: "f32[128, 50265]" = torch.ops.aten.mm.default(view_467, permute_248);  view_467 = permute_248 = None
    view_468: "f32[1, 128, 50265]" = torch.ops.aten.reshape.default(mm, [1, 128, 50265]);  mm = None
    add_151: "f32[1, 128, 50265]" = torch.ops.aten.add.Tensor(view_468, arg344_1);  view_468 = arg344_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/blenderbot_small/modeling_blenderbot_small.py:1330, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
    view_469: "f32[128, 50265]" = torch.ops.aten.reshape.default(add_151, [-1, 50265])
    amax_24: "f32[128, 1]" = torch.ops.aten.amax.default(view_469, [1], True)
    sub_66: "f32[128, 50265]" = torch.ops.aten.sub.Tensor(view_469, amax_24);  view_469 = amax_24 = None
    exp_24: "f32[128, 50265]" = torch.ops.aten.exp.default(sub_66)
    sum_25: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_67: "f32[128, 50265]" = torch.ops.aten.sub.Tensor(sub_66, log);  sub_66 = log = None
    ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_470, -100)
    full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "i64[128]" = torch.ops.aten.where.self(ne, view_470, full_default_2);  ne = full_default_2 = None
    unsqueeze_4: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_67, 1, unsqueeze_4);  sub_67 = unsqueeze_4 = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_2: "f32[128]" = torch.ops.aten.where.self(ne_1, neg, full_default_3);  ne_1 = neg = full_default_3 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
    ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_470, -100);  view_470 = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
    return (div_24, add_151, add_58)
    