from __future__ import annotations



def forward(self, arg0_1: "f32[256008, 1024]", arg1_1: "f32[1024]", arg2_1: "f32[1024]", arg3_1: "f32[1024, 1024]", arg4_1: "f32[1024]", arg5_1: "f32[1024, 1024]", arg6_1: "f32[1024]", arg7_1: "f32[1024, 1024]", arg8_1: "f32[1024]", arg9_1: "f32[1024, 1024]", arg10_1: "f32[1024]", arg11_1: "f32[1024]", arg12_1: "f32[1024]", arg13_1: "f32[4096, 1024]", arg14_1: "f32[4096]", arg15_1: "f32[1024, 4096]", arg16_1: "f32[1024]", arg17_1: "f32[1024]", arg18_1: "f32[1024]", arg19_1: "f32[1024, 1024]", arg20_1: "f32[1024]", arg21_1: "f32[1024, 1024]", arg22_1: "f32[1024]", arg23_1: "f32[1024, 1024]", arg24_1: "f32[1024]", arg25_1: "f32[1024, 1024]", arg26_1: "f32[1024]", arg27_1: "f32[1024]", arg28_1: "f32[1024]", arg29_1: "f32[4096, 1024]", arg30_1: "f32[4096]", arg31_1: "f32[1024, 4096]", arg32_1: "f32[1024]", arg33_1: "f32[1024]", arg34_1: "f32[1024]", arg35_1: "f32[1024, 1024]", arg36_1: "f32[1024]", arg37_1: "f32[1024, 1024]", arg38_1: "f32[1024]", arg39_1: "f32[1024, 1024]", arg40_1: "f32[1024]", arg41_1: "f32[1024, 1024]", arg42_1: "f32[1024]", arg43_1: "f32[1024]", arg44_1: "f32[1024]", arg45_1: "f32[4096, 1024]", arg46_1: "f32[4096]", arg47_1: "f32[1024, 4096]", arg48_1: "f32[1024]", arg49_1: "f32[1024]", arg50_1: "f32[1024]", arg51_1: "f32[1024, 1024]", arg52_1: "f32[1024]", arg53_1: "f32[1024, 1024]", arg54_1: "f32[1024]", arg55_1: "f32[1024, 1024]", arg56_1: "f32[1024]", arg57_1: "f32[1024, 1024]", arg58_1: "f32[1024]", arg59_1: "f32[1024]", arg60_1: "f32[1024]", arg61_1: "f32[4096, 1024]", arg62_1: "f32[4096]", arg63_1: "f32[1024, 4096]", arg64_1: "f32[1024]", arg65_1: "f32[1024]", arg66_1: "f32[1024]", arg67_1: "f32[1024, 1024]", arg68_1: "f32[1024]", arg69_1: "f32[1024, 1024]", arg70_1: "f32[1024]", arg71_1: "f32[1024, 1024]", arg72_1: "f32[1024]", arg73_1: "f32[1024, 1024]", arg74_1: "f32[1024]", arg75_1: "f32[1024]", arg76_1: "f32[1024]", arg77_1: "f32[4096, 1024]", arg78_1: "f32[4096]", arg79_1: "f32[1024, 4096]", arg80_1: "f32[1024]", arg81_1: "f32[1024]", arg82_1: "f32[1024]", arg83_1: "f32[1024, 1024]", arg84_1: "f32[1024]", arg85_1: "f32[1024, 1024]", arg86_1: "f32[1024]", arg87_1: "f32[1024, 1024]", arg88_1: "f32[1024]", arg89_1: "f32[1024, 1024]", arg90_1: "f32[1024]", arg91_1: "f32[1024]", arg92_1: "f32[1024]", arg93_1: "f32[4096, 1024]", arg94_1: "f32[4096]", arg95_1: "f32[1024, 4096]", arg96_1: "f32[1024]", arg97_1: "f32[1024]", arg98_1: "f32[1024]", arg99_1: "f32[1024, 1024]", arg100_1: "f32[1024]", arg101_1: "f32[1024, 1024]", arg102_1: "f32[1024]", arg103_1: "f32[1024, 1024]", arg104_1: "f32[1024]", arg105_1: "f32[1024, 1024]", arg106_1: "f32[1024]", arg107_1: "f32[1024]", arg108_1: "f32[1024]", arg109_1: "f32[4096, 1024]", arg110_1: "f32[4096]", arg111_1: "f32[1024, 4096]", arg112_1: "f32[1024]", arg113_1: "f32[1024]", arg114_1: "f32[1024]", arg115_1: "f32[1024, 1024]", arg116_1: "f32[1024]", arg117_1: "f32[1024, 1024]", arg118_1: "f32[1024]", arg119_1: "f32[1024, 1024]", arg120_1: "f32[1024]", arg121_1: "f32[1024, 1024]", arg122_1: "f32[1024]", arg123_1: "f32[1024]", arg124_1: "f32[1024]", arg125_1: "f32[4096, 1024]", arg126_1: "f32[4096]", arg127_1: "f32[1024, 4096]", arg128_1: "f32[1024]", arg129_1: "f32[1024]", arg130_1: "f32[1024]", arg131_1: "f32[1024, 1024]", arg132_1: "f32[1024]", arg133_1: "f32[1024, 1024]", arg134_1: "f32[1024]", arg135_1: "f32[1024, 1024]", arg136_1: "f32[1024]", arg137_1: "f32[1024, 1024]", arg138_1: "f32[1024]", arg139_1: "f32[1024]", arg140_1: "f32[1024]", arg141_1: "f32[4096, 1024]", arg142_1: "f32[4096]", arg143_1: "f32[1024, 4096]", arg144_1: "f32[1024]", arg145_1: "f32[1024]", arg146_1: "f32[1024]", arg147_1: "f32[1024, 1024]", arg148_1: "f32[1024]", arg149_1: "f32[1024, 1024]", arg150_1: "f32[1024]", arg151_1: "f32[1024, 1024]", arg152_1: "f32[1024]", arg153_1: "f32[1024, 1024]", arg154_1: "f32[1024]", arg155_1: "f32[1024]", arg156_1: "f32[1024]", arg157_1: "f32[4096, 1024]", arg158_1: "f32[4096]", arg159_1: "f32[1024, 4096]", arg160_1: "f32[1024]", arg161_1: "f32[1024]", arg162_1: "f32[1024]", arg163_1: "f32[1024, 1024]", arg164_1: "f32[1024]", arg165_1: "f32[1024, 1024]", arg166_1: "f32[1024]", arg167_1: "f32[1024, 1024]", arg168_1: "f32[1024]", arg169_1: "f32[1024, 1024]", arg170_1: "f32[1024]", arg171_1: "f32[1024]", arg172_1: "f32[1024]", arg173_1: "f32[4096, 1024]", arg174_1: "f32[4096]", arg175_1: "f32[1024, 4096]", arg176_1: "f32[1024]", arg177_1: "f32[1024]", arg178_1: "f32[1024]", arg179_1: "f32[1024, 1024]", arg180_1: "f32[1024]", arg181_1: "f32[1024, 1024]", arg182_1: "f32[1024]", arg183_1: "f32[1024, 1024]", arg184_1: "f32[1024]", arg185_1: "f32[1024, 1024]", arg186_1: "f32[1024]", arg187_1: "f32[1024]", arg188_1: "f32[1024]", arg189_1: "f32[4096, 1024]", arg190_1: "f32[4096]", arg191_1: "f32[1024, 4096]", arg192_1: "f32[1024]", arg193_1: "f32[1024]", arg194_1: "f32[1024]", arg195_1: "f32[1024, 1024]", arg196_1: "f32[1024]", arg197_1: "f32[1024, 1024]", arg198_1: "f32[1024]", arg199_1: "f32[1024, 1024]", arg200_1: "f32[1024]", arg201_1: "f32[1024, 1024]", arg202_1: "f32[1024]", arg203_1: "f32[1024]", arg204_1: "f32[1024]", arg205_1: "f32[4096, 1024]", arg206_1: "f32[4096]", arg207_1: "f32[1024, 4096]", arg208_1: "f32[1024]", arg209_1: "f32[1024]", arg210_1: "f32[1024]", arg211_1: "f32[1024, 1024]", arg212_1: "f32[1024]", arg213_1: "f32[1024, 1024]", arg214_1: "f32[1024]", arg215_1: "f32[1024, 1024]", arg216_1: "f32[1024]", arg217_1: "f32[1024, 1024]", arg218_1: "f32[1024]", arg219_1: "f32[1024]", arg220_1: "f32[1024]", arg221_1: "f32[4096, 1024]", arg222_1: "f32[4096]", arg223_1: "f32[1024, 4096]", arg224_1: "f32[1024]", arg225_1: "f32[1024]", arg226_1: "f32[1024]", arg227_1: "f32[1024, 1024]", arg228_1: "f32[1024]", arg229_1: "f32[1024, 1024]", arg230_1: "f32[1024]", arg231_1: "f32[1024, 1024]", arg232_1: "f32[1024]", arg233_1: "f32[1024, 1024]", arg234_1: "f32[1024]", arg235_1: "f32[1024]", arg236_1: "f32[1024]", arg237_1: "f32[4096, 1024]", arg238_1: "f32[4096]", arg239_1: "f32[1024, 4096]", arg240_1: "f32[1024]", arg241_1: "f32[1024]", arg242_1: "f32[1024]", arg243_1: "f32[1024, 1024]", arg244_1: "f32[1024]", arg245_1: "f32[1024, 1024]", arg246_1: "f32[1024]", arg247_1: "f32[1024, 1024]", arg248_1: "f32[1024]", arg249_1: "f32[1024, 1024]", arg250_1: "f32[1024]", arg251_1: "f32[1024]", arg252_1: "f32[1024]", arg253_1: "f32[4096, 1024]", arg254_1: "f32[4096]", arg255_1: "f32[1024, 4096]", arg256_1: "f32[1024]", arg257_1: "f32[1024]", arg258_1: "f32[1024]", arg259_1: "f32[1024, 1024]", arg260_1: "f32[1024]", arg261_1: "f32[1024, 1024]", arg262_1: "f32[1024]", arg263_1: "f32[1024, 1024]", arg264_1: "f32[1024]", arg265_1: "f32[1024, 1024]", arg266_1: "f32[1024]", arg267_1: "f32[1024]", arg268_1: "f32[1024]", arg269_1: "f32[4096, 1024]", arg270_1: "f32[4096]", arg271_1: "f32[1024, 4096]", arg272_1: "f32[1024]", arg273_1: "f32[1024]", arg274_1: "f32[1024]", arg275_1: "f32[1024, 1024]", arg276_1: "f32[1024]", arg277_1: "f32[1024, 1024]", arg278_1: "f32[1024]", arg279_1: "f32[1024, 1024]", arg280_1: "f32[1024]", arg281_1: "f32[1024, 1024]", arg282_1: "f32[1024]", arg283_1: "f32[1024]", arg284_1: "f32[1024]", arg285_1: "f32[4096, 1024]", arg286_1: "f32[4096]", arg287_1: "f32[1024, 4096]", arg288_1: "f32[1024]", arg289_1: "f32[1024]", arg290_1: "f32[1024]", arg291_1: "f32[1024, 1024]", arg292_1: "f32[1024]", arg293_1: "f32[1024, 1024]", arg294_1: "f32[1024]", arg295_1: "f32[1024, 1024]", arg296_1: "f32[1024]", arg297_1: "f32[1024, 1024]", arg298_1: "f32[1024]", arg299_1: "f32[1024]", arg300_1: "f32[1024]", arg301_1: "f32[4096, 1024]", arg302_1: "f32[4096]", arg303_1: "f32[1024, 4096]", arg304_1: "f32[1024]", arg305_1: "f32[1024]", arg306_1: "f32[1024]", arg307_1: "f32[1024, 1024]", arg308_1: "f32[1024]", arg309_1: "f32[1024, 1024]", arg310_1: "f32[1024]", arg311_1: "f32[1024, 1024]", arg312_1: "f32[1024]", arg313_1: "f32[1024, 1024]", arg314_1: "f32[1024]", arg315_1: "f32[1024]", arg316_1: "f32[1024]", arg317_1: "f32[4096, 1024]", arg318_1: "f32[4096]", arg319_1: "f32[1024, 4096]", arg320_1: "f32[1024]", arg321_1: "f32[1024]", arg322_1: "f32[1024]", arg323_1: "f32[1024, 1024]", arg324_1: "f32[1024]", arg325_1: "f32[1024, 1024]", arg326_1: "f32[1024]", arg327_1: "f32[1024, 1024]", arg328_1: "f32[1024]", arg329_1: "f32[1024, 1024]", arg330_1: "f32[1024]", arg331_1: "f32[1024]", arg332_1: "f32[1024]", arg333_1: "f32[4096, 1024]", arg334_1: "f32[4096]", arg335_1: "f32[1024, 4096]", arg336_1: "f32[1024]", arg337_1: "f32[1024]", arg338_1: "f32[1024]", arg339_1: "f32[1024, 1024]", arg340_1: "f32[1024]", arg341_1: "f32[1024, 1024]", arg342_1: "f32[1024]", arg343_1: "f32[1024, 1024]", arg344_1: "f32[1024]", arg345_1: "f32[1024, 1024]", arg346_1: "f32[1024]", arg347_1: "f32[1024]", arg348_1: "f32[1024]", arg349_1: "f32[4096, 1024]", arg350_1: "f32[4096]", arg351_1: "f32[1024, 4096]", arg352_1: "f32[1024]", arg353_1: "f32[1024]", arg354_1: "f32[1024]", arg355_1: "f32[1024, 1024]", arg356_1: "f32[1024]", arg357_1: "f32[1024, 1024]", arg358_1: "f32[1024]", arg359_1: "f32[1024, 1024]", arg360_1: "f32[1024]", arg361_1: "f32[1024, 1024]", arg362_1: "f32[1024]", arg363_1: "f32[1024]", arg364_1: "f32[1024]", arg365_1: "f32[4096, 1024]", arg366_1: "f32[4096]", arg367_1: "f32[1024, 4096]", arg368_1: "f32[1024]", arg369_1: "f32[1024]", arg370_1: "f32[1024]", arg371_1: "f32[1024, 1024]", arg372_1: "f32[1024]", arg373_1: "f32[1024, 1024]", arg374_1: "f32[1024]", arg375_1: "f32[1024, 1024]", arg376_1: "f32[1024]", arg377_1: "f32[1024, 1024]", arg378_1: "f32[1024]", arg379_1: "f32[1024]", arg380_1: "f32[1024]", arg381_1: "f32[4096, 1024]", arg382_1: "f32[4096]", arg383_1: "f32[1024, 4096]", arg384_1: "f32[1024]", arg385_1: "f32[1024]", arg386_1: "f32[1024]", arg387_1: "f32[256008, 1024]", arg388_1: "f32[2050, 1024]", arg389_1: "i64[1, 128]", arg390_1: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:611, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.view.default(arg389_1, [-1, 128]);  arg389_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:620, code: position_ids = torch.arange(
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:626, code: position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
    unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    view_1: "i64[1, 128]" = torch.ops.aten.view.default(unsqueeze, [-1, 128]);  unsqueeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:631, code: inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
    embedding: "f32[1, 128, 1024]" = torch.ops.aten.embedding.default(arg0_1, view, 1);  arg0_1 = view = None
    mul: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(embedding, 32.0);  embedding = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:138, code: mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    full: "f32[128, 128]" = torch.ops.aten.full.default([128, 128], -3.4028234663852886e+38, dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:139, code: mask_cond = torch.arange(mask.size(-1), device=device)
    iota_1: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:140, code: mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    add: "i64[128]" = torch.ops.aten.add.Tensor(iota_1, 1)
    view_2: "i64[128, 1]" = torch.ops.aten.view.default(add, [128, 1]);  add = None
    lt: "b8[128, 128]" = torch.ops.aten.lt.Tensor(iota_1, view_2);  iota_1 = view_2 = None
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[128, 128]" = torch.ops.aten.where.self(lt, scalar_tensor, full);  lt = scalar_tensor = full = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:205, code: position_ids += self.offset
    add_1: "i64[1, 128]" = torch.ops.aten.add.Tensor(view_1, 2);  view_1 = None
    view_3: "i64[1, 128]" = torch.ops.aten.view.default(add_1, [1, 128]);  add_1 = None
    squeeze: "i64[128]" = torch.ops.aten.squeeze.dim(view_3, 0);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:212, code: return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
    unsqueeze_5: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(squeeze, 0);  squeeze = None
    view_6: "i64[1, 128]" = torch.ops.aten.view.default(unsqueeze_5, [-1, 128]);  unsqueeze_5 = None
    view_7: "i64[128]" = torch.ops.aten.view.default(view_6, [-1]);  view_6 = None
    index: "f32[128, 1024]" = torch.ops.aten.index.Tensor(arg388_1, [view_7]);  arg388_1 = view_7 = None
    view_8: "f32[1, 128, 1024]" = torch.ops.aten.view.default(index, [1, 128, 1024]);  index = None
    alias: "f32[1, 128, 1024]" = torch.ops.aten.alias.default(view_8);  view_8 = None
    alias_1: "f32[1, 128, 1024]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:642, code: hidden_states = inputs_embeds + self.embed_positions(position_ids, past_key_values_length)
    add_2: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul, alias_1);  mul = alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:643, code: hidden_states = nn.functional.dropout(hidden_states, p=float(self.dropout), training=self.training)
    clone: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(add_2);  add_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[1, 128, 1]" = var_mean[0]
    getitem_1: "f32[1, 128, 1]" = var_mean[1];  var_mean = None
    add_3: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  getitem_1 = None
    mul_1: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_2: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1, arg1_1);  mul_1 = arg1_1 = None
    add_4: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_2, arg2_1);  mul_2 = arg2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_9: "f32[128, 1024]" = torch.ops.aten.view.default(add_4, [128, 1024])
    permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
    addmm: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg4_1, view_9, permute);  arg4_1 = view_9 = permute = None
    view_10: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm, [1, 128, 1024]);  addmm = None
    mul_3: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_10, 0.125);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_11: "f32[128, 1024]" = torch.ops.aten.view.default(add_4, [128, 1024])
    permute_1: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    addmm_1: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg6_1, view_11, permute_1);  arg6_1 = view_11 = permute_1 = None
    view_12: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_1, [1, 128, 1024]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_13: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_12, [1, -1, 16, 64]);  view_12 = None
    permute_2: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_13, [0, 2, 1, 3]);  view_13 = None
    clone_1: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_14: "f32[128, 1024]" = torch.ops.aten.view.default(add_4, [128, 1024]);  add_4 = None
    permute_3: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    addmm_2: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg8_1, view_14, permute_3);  arg8_1 = view_14 = permute_3 = None
    view_15: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_2, [1, 128, 1024]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_16: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_15, [1, -1, 16, 64]);  view_15 = None
    permute_4: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
    clone_2: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_4, memory_format = torch.contiguous_format);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_17: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_3, [1, 128, 16, 64]);  mul_3 = None
    permute_5: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
    clone_3: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_18: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_3, [16, -1, 64]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_19: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_1, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_20: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_2, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_6: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_19, [0, 2, 1]);  view_19 = None
    bmm: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_18, permute_6);  view_18 = permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_21: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm, [1, 16, 128, 128]);  bmm = None
    unsqueeze_6: "f32[1, 128, 128]" = torch.ops.aten.unsqueeze.default(where, 0);  where = None
    unsqueeze_7: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
    slice_3: "f32[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(unsqueeze_7, 2, 0, 9223372036854775807);  unsqueeze_7 = None
    slice_4: "f32[1, 1, 128, 128]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 9223372036854775807);  slice_3 = None
    expand_1: "f32[1, 1, 128, 128]" = torch.ops.aten.expand.default(slice_4, [1, 1, 128, 128]);  slice_4 = None
    add_5: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_21, expand_1);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_5, lift_fresh_copy);  add_5 = lift_fresh_copy = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_22: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum, [16, 128, 128]);  maximum = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_22, [-1], True)
    sub_1: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_22, amax);  view_22 = amax = None
    exp: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_4: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_1: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_4, view_20);  clone_4 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_23: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_1, [1, 16, 128, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_7: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_23, [0, 2, 1, 3]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_5: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_24: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_5, [1, 128, 1024]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_25: "f32[128, 1024]" = torch.ops.aten.view.default(view_24, [128, 1024]);  view_24 = None
    permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
    addmm_3: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg10_1, view_25, permute_8);  arg10_1 = view_25 = permute_8 = None
    view_26: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_3, [1, 128, 1024]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_6: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_26);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_6: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(clone, clone_6);  clone = clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_2: "f32[1, 128, 1]" = var_mean_1[0]
    getitem_3: "f32[1, 128, 1]" = var_mean_1[1];  var_mean_1 = None
    add_7: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_2: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_6, getitem_3);  getitem_3 = None
    mul_4: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
    mul_5: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_4, arg11_1);  mul_4 = arg11_1 = None
    add_8: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_5, arg12_1);  mul_5 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_27: "f32[128, 1024]" = torch.ops.aten.view.default(add_8, [128, 1024]);  add_8 = None
    permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
    addmm_4: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg14_1, view_27, permute_9);  arg14_1 = view_27 = permute_9 = None
    view_28: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_4, [1, 128, 4096]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_6: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_28, 0.5)
    mul_7: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476);  view_28 = None
    erf: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_7);  mul_7 = None
    add_9: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_8: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_6, add_9);  mul_6 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_7: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_8);  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_29: "f32[128, 4096]" = torch.ops.aten.view.default(clone_7, [128, 4096]);  clone_7 = None
    permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
    addmm_5: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg16_1, view_29, permute_10);  arg16_1 = view_29 = permute_10 = None
    view_30: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_5, [1, 128, 1024]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_8: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_30);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_10: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_6, clone_8);  add_6 = clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1, 128, 1]" = var_mean_2[0]
    getitem_5: "f32[1, 128, 1]" = var_mean_2[1];  var_mean_2 = None
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_3: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_10, getitem_5);  getitem_5 = None
    mul_9: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
    mul_10: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_9, arg17_1);  mul_9 = arg17_1 = None
    add_12: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_10, arg18_1);  mul_10 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_31: "f32[128, 1024]" = torch.ops.aten.view.default(add_12, [128, 1024])
    permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
    addmm_6: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg20_1, view_31, permute_11);  arg20_1 = view_31 = permute_11 = None
    view_32: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_6, [1, 128, 1024]);  addmm_6 = None
    mul_11: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_32, 0.125);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_33: "f32[128, 1024]" = torch.ops.aten.view.default(add_12, [128, 1024])
    permute_12: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
    addmm_7: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg22_1, view_33, permute_12);  arg22_1 = view_33 = permute_12 = None
    view_34: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_7, [1, 128, 1024]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_35: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_34, [1, -1, 16, 64]);  view_34 = None
    permute_13: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
    clone_9: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_36: "f32[128, 1024]" = torch.ops.aten.view.default(add_12, [128, 1024]);  add_12 = None
    permute_14: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg23_1, [1, 0]);  arg23_1 = None
    addmm_8: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg24_1, view_36, permute_14);  arg24_1 = view_36 = permute_14 = None
    view_37: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_8, [1, 128, 1024]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_38: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_37, [1, -1, 16, 64]);  view_37 = None
    permute_15: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_38, [0, 2, 1, 3]);  view_38 = None
    clone_10: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_15, memory_format = torch.contiguous_format);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_39: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_11, [1, 128, 16, 64]);  mul_11 = None
    permute_16: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
    clone_11: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_16, memory_format = torch.contiguous_format);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_40: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_11, [16, -1, 64]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_41: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_9, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_42: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_10, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_17: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
    bmm_2: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_40, permute_17);  view_40 = permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_43: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_2, [1, 16, 128, 128]);  bmm_2 = None
    add_13: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_43, expand_1);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant1 = self._tensor_constant1
    lift_fresh_copy_1: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_1: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_13, lift_fresh_copy_1);  add_13 = lift_fresh_copy_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_44: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_1, [16, 128, 128]);  maximum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_1: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_44, [-1], True)
    sub_4: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_44, amax_1);  view_44 = amax_1 = None
    exp_1: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_12: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_3: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_12, view_42);  clone_12 = view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_45: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_3, [1, 16, 128, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_18: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_45, [0, 2, 1, 3]);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_13: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_46: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_13, [1, 128, 1024]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_47: "f32[128, 1024]" = torch.ops.aten.view.default(view_46, [128, 1024]);  view_46 = None
    permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    addmm_9: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg26_1, view_47, permute_19);  arg26_1 = view_47 = permute_19 = None
    view_48: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_9, [1, 128, 1024]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_14: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_14: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_10, clone_14);  add_10 = clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_6: "f32[1, 128, 1]" = var_mean_3[0]
    getitem_7: "f32[1, 128, 1]" = var_mean_3[1];  var_mean_3 = None
    add_15: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_5: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_14, getitem_7);  getitem_7 = None
    mul_12: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
    mul_13: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, arg27_1);  mul_12 = arg27_1 = None
    add_16: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_13, arg28_1);  mul_13 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_49: "f32[128, 1024]" = torch.ops.aten.view.default(add_16, [128, 1024]);  add_16 = None
    permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
    addmm_10: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg30_1, view_49, permute_20);  arg30_1 = view_49 = permute_20 = None
    view_50: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_10, [1, 128, 4096]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_14: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_50, 0.5)
    mul_15: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_50, 0.7071067811865476);  view_50 = None
    erf_1: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_17: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_14, add_17);  mul_14 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_15: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_16);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_51: "f32[128, 4096]" = torch.ops.aten.view.default(clone_15, [128, 4096]);  clone_15 = None
    permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
    addmm_11: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg32_1, view_51, permute_21);  arg32_1 = view_51 = permute_21 = None
    view_52: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_11, [1, 128, 1024]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_16: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_52);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_18: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_14, clone_16);  add_14 = clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1, 128, 1]" = var_mean_4[0]
    getitem_9: "f32[1, 128, 1]" = var_mean_4[1];  var_mean_4 = None
    add_19: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
    sub_6: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_18, getitem_9);  getitem_9 = None
    mul_17: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
    mul_18: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_17, arg33_1);  mul_17 = arg33_1 = None
    add_20: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_18, arg34_1);  mul_18 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_53: "f32[128, 1024]" = torch.ops.aten.view.default(add_20, [128, 1024])
    permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    addmm_12: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg36_1, view_53, permute_22);  arg36_1 = view_53 = permute_22 = None
    view_54: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_12, [1, 128, 1024]);  addmm_12 = None
    mul_19: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_54, 0.125);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_55: "f32[128, 1024]" = torch.ops.aten.view.default(add_20, [128, 1024])
    permute_23: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    addmm_13: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg38_1, view_55, permute_23);  arg38_1 = view_55 = permute_23 = None
    view_56: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_13, [1, 128, 1024]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_57: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_56, [1, -1, 16, 64]);  view_56 = None
    permute_24: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_57, [0, 2, 1, 3]);  view_57 = None
    clone_17: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_58: "f32[128, 1024]" = torch.ops.aten.view.default(add_20, [128, 1024]);  add_20 = None
    permute_25: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
    addmm_14: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg40_1, view_58, permute_25);  arg40_1 = view_58 = permute_25 = None
    view_59: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_14, [1, 128, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_60: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_59, [1, -1, 16, 64]);  view_59 = None
    permute_26: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1, 3]);  view_60 = None
    clone_18: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_26, memory_format = torch.contiguous_format);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_61: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_19, [1, 128, 16, 64]);  mul_19 = None
    permute_27: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
    clone_19: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_62: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_19, [16, -1, 64]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_63: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_17, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_64: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_18, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_28: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    bmm_4: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_62, permute_28);  view_62 = permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_65: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_4, [1, 16, 128, 128]);  bmm_4 = None
    add_21: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_65, expand_1);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant2 = self._tensor_constant2
    lift_fresh_copy_2: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant2);  _tensor_constant2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_2: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_21, lift_fresh_copy_2);  add_21 = lift_fresh_copy_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_66: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_2, [16, 128, 128]);  maximum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_2: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_66, [-1], True)
    sub_7: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_66, amax_2);  view_66 = amax_2 = None
    exp_2: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_20: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_5: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_20, view_64);  clone_20 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_67: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_5, [1, 16, 128, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_29: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_67, [0, 2, 1, 3]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_21: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_68: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_21, [1, 128, 1024]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_69: "f32[128, 1024]" = torch.ops.aten.view.default(view_68, [128, 1024]);  view_68 = None
    permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    addmm_15: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg42_1, view_69, permute_30);  arg42_1 = view_69 = permute_30 = None
    view_70: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_15, [1, 128, 1024]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_22: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_70);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_22: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_18, clone_22);  add_18 = clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_22, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1, 128, 1]" = var_mean_5[0]
    getitem_11: "f32[1, 128, 1]" = var_mean_5[1];  var_mean_5 = None
    add_23: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_8: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_22, getitem_11);  getitem_11 = None
    mul_20: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = rsqrt_5 = None
    mul_21: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_20, arg43_1);  mul_20 = arg43_1 = None
    add_24: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_21, arg44_1);  mul_21 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_71: "f32[128, 1024]" = torch.ops.aten.view.default(add_24, [128, 1024]);  add_24 = None
    permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
    addmm_16: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg46_1, view_71, permute_31);  arg46_1 = view_71 = permute_31 = None
    view_72: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_16, [1, 128, 4096]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_22: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_72, 0.5)
    mul_23: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_72, 0.7071067811865476);  view_72 = None
    erf_2: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_25: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_24: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_22, add_25);  mul_22 = add_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_23: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_24);  mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_73: "f32[128, 4096]" = torch.ops.aten.view.default(clone_23, [128, 4096]);  clone_23 = None
    permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    addmm_17: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg48_1, view_73, permute_32);  arg48_1 = view_73 = permute_32 = None
    view_74: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_17, [1, 128, 1024]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_24: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_74);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_26: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_22, clone_24);  add_22 = clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
    getitem_12: "f32[1, 128, 1]" = var_mean_6[0]
    getitem_13: "f32[1, 128, 1]" = var_mean_6[1];  var_mean_6 = None
    add_27: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
    sub_9: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_26, getitem_13);  getitem_13 = None
    mul_25: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = rsqrt_6 = None
    mul_26: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_25, arg49_1);  mul_25 = arg49_1 = None
    add_28: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_26, arg50_1);  mul_26 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_75: "f32[128, 1024]" = torch.ops.aten.view.default(add_28, [128, 1024])
    permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
    addmm_18: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg52_1, view_75, permute_33);  arg52_1 = view_75 = permute_33 = None
    view_76: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_18, [1, 128, 1024]);  addmm_18 = None
    mul_27: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_76, 0.125);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_77: "f32[128, 1024]" = torch.ops.aten.view.default(add_28, [128, 1024])
    permute_34: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    addmm_19: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg54_1, view_77, permute_34);  arg54_1 = view_77 = permute_34 = None
    view_78: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_19, [1, 128, 1024]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_79: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_78, [1, -1, 16, 64]);  view_78 = None
    permute_35: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1, 3]);  view_79 = None
    clone_25: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_80: "f32[128, 1024]" = torch.ops.aten.view.default(add_28, [128, 1024]);  add_28 = None
    permute_36: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
    addmm_20: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg56_1, view_80, permute_36);  arg56_1 = view_80 = permute_36 = None
    view_81: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_20, [1, 128, 1024]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_82: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_81, [1, -1, 16, 64]);  view_81 = None
    permute_37: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_82, [0, 2, 1, 3]);  view_82 = None
    clone_26: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_83: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_27, [1, 128, 16, 64]);  mul_27 = None
    permute_38: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
    clone_27: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_84: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_27, [16, -1, 64]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_85: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_25, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_86: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_26, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_39: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
    bmm_6: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_84, permute_39);  view_84 = permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_87: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_6, [1, 16, 128, 128]);  bmm_6 = None
    add_29: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_87, expand_1);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant3 = self._tensor_constant3
    lift_fresh_copy_3: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant3);  _tensor_constant3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_3: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_29, lift_fresh_copy_3);  add_29 = lift_fresh_copy_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_88: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_3, [16, 128, 128]);  maximum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_3: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_88, [-1], True)
    sub_10: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_88, amax_3);  view_88 = amax_3 = None
    exp_3: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_28: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_7: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_28, view_86);  clone_28 = view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_89: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_7, [1, 16, 128, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_40: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1, 3]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_29: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_90: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_29, [1, 128, 1024]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_91: "f32[128, 1024]" = torch.ops.aten.view.default(view_90, [128, 1024]);  view_90 = None
    permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    addmm_21: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg58_1, view_91, permute_41);  arg58_1 = view_91 = permute_41 = None
    view_92: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_21, [1, 128, 1024]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_30: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_92);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_30: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_26, clone_30);  add_26 = clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
    getitem_14: "f32[1, 128, 1]" = var_mean_7[0]
    getitem_15: "f32[1, 128, 1]" = var_mean_7[1];  var_mean_7 = None
    add_31: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-05);  getitem_14 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_11: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_30, getitem_15);  getitem_15 = None
    mul_28: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = rsqrt_7 = None
    mul_29: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_28, arg59_1);  mul_28 = arg59_1 = None
    add_32: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_29, arg60_1);  mul_29 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_93: "f32[128, 1024]" = torch.ops.aten.view.default(add_32, [128, 1024]);  add_32 = None
    permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
    addmm_22: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg62_1, view_93, permute_42);  arg62_1 = view_93 = permute_42 = None
    view_94: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_22, [1, 128, 4096]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_30: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
    mul_31: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_94, 0.7071067811865476);  view_94 = None
    erf_3: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
    add_33: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_32: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_30, add_33);  mul_30 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_31: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_95: "f32[128, 4096]" = torch.ops.aten.view.default(clone_31, [128, 4096]);  clone_31 = None
    permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    addmm_23: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg64_1, view_95, permute_43);  arg64_1 = view_95 = permute_43 = None
    view_96: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_23, [1, 128, 1024]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_32: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_34: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_30, clone_32);  add_30 = clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
    getitem_16: "f32[1, 128, 1]" = var_mean_8[0]
    getitem_17: "f32[1, 128, 1]" = var_mean_8[1];  var_mean_8 = None
    add_35: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_12: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_34, getitem_17);  getitem_17 = None
    mul_33: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = rsqrt_8 = None
    mul_34: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_33, arg65_1);  mul_33 = arg65_1 = None
    add_36: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_34, arg66_1);  mul_34 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_97: "f32[128, 1024]" = torch.ops.aten.view.default(add_36, [128, 1024])
    permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
    addmm_24: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg68_1, view_97, permute_44);  arg68_1 = view_97 = permute_44 = None
    view_98: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_24, [1, 128, 1024]);  addmm_24 = None
    mul_35: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_98, 0.125);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_99: "f32[128, 1024]" = torch.ops.aten.view.default(add_36, [128, 1024])
    permute_45: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    addmm_25: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg70_1, view_99, permute_45);  arg70_1 = view_99 = permute_45 = None
    view_100: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_25, [1, 128, 1024]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_101: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_100, [1, -1, 16, 64]);  view_100 = None
    permute_46: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
    clone_33: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_102: "f32[128, 1024]" = torch.ops.aten.view.default(add_36, [128, 1024]);  add_36 = None
    permute_47: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    addmm_26: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg72_1, view_102, permute_47);  arg72_1 = view_102 = permute_47 = None
    view_103: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_26, [1, 128, 1024]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_104: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_103, [1, -1, 16, 64]);  view_103 = None
    permute_48: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
    clone_34: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_48, memory_format = torch.contiguous_format);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_105: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_35, [1, 128, 16, 64]);  mul_35 = None
    permute_49: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_105, [0, 2, 1, 3]);  view_105 = None
    clone_35: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_106: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_35, [16, -1, 64]);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_107: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_33, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_108: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_34, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_50: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
    bmm_8: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_106, permute_50);  view_106 = permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_109: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_8, [1, 16, 128, 128]);  bmm_8 = None
    add_37: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_109, expand_1);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant4 = self._tensor_constant4
    lift_fresh_copy_4: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant4);  _tensor_constant4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_4: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_37, lift_fresh_copy_4);  add_37 = lift_fresh_copy_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_110: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_4, [16, 128, 128]);  maximum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_4: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_110, [-1], True)
    sub_13: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_110, amax_4);  view_110 = amax_4 = None
    exp_4: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_36: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_9: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_36, view_108);  clone_36 = view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_111: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_9, [1, 16, 128, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_51: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_111, [0, 2, 1, 3]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_37: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_112: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_37, [1, 128, 1024]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_113: "f32[128, 1024]" = torch.ops.aten.view.default(view_112, [128, 1024]);  view_112 = None
    permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
    addmm_27: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg74_1, view_113, permute_52);  arg74_1 = view_113 = permute_52 = None
    view_114: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_27, [1, 128, 1024]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_38: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_114);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_38: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_34, clone_38);  add_34 = clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1, 128, 1]" = var_mean_9[0]
    getitem_19: "f32[1, 128, 1]" = var_mean_9[1];  var_mean_9 = None
    add_39: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_14: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_38, getitem_19);  getitem_19 = None
    mul_36: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = rsqrt_9 = None
    mul_37: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_36, arg75_1);  mul_36 = arg75_1 = None
    add_40: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_37, arg76_1);  mul_37 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_115: "f32[128, 1024]" = torch.ops.aten.view.default(add_40, [128, 1024]);  add_40 = None
    permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    addmm_28: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg78_1, view_115, permute_53);  arg78_1 = view_115 = permute_53 = None
    view_116: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_28, [1, 128, 4096]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_38: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_116, 0.5)
    mul_39: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476);  view_116 = None
    erf_4: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_39);  mul_39 = None
    add_41: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_40: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_38, add_41);  mul_38 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_39: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_40);  mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_117: "f32[128, 4096]" = torch.ops.aten.view.default(clone_39, [128, 4096]);  clone_39 = None
    permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
    addmm_29: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg80_1, view_117, permute_54);  arg80_1 = view_117 = permute_54 = None
    view_118: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_29, [1, 128, 1024]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_40: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_118);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_42: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_38, clone_40);  add_38 = clone_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_20: "f32[1, 128, 1]" = var_mean_10[0]
    getitem_21: "f32[1, 128, 1]" = var_mean_10[1];  var_mean_10 = None
    add_43: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_15: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_42, getitem_21);  getitem_21 = None
    mul_41: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = rsqrt_10 = None
    mul_42: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_41, arg81_1);  mul_41 = arg81_1 = None
    add_44: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_42, arg82_1);  mul_42 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_119: "f32[128, 1024]" = torch.ops.aten.view.default(add_44, [128, 1024])
    permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
    addmm_30: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg84_1, view_119, permute_55);  arg84_1 = view_119 = permute_55 = None
    view_120: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_30, [1, 128, 1024]);  addmm_30 = None
    mul_43: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_120, 0.125);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_121: "f32[128, 1024]" = torch.ops.aten.view.default(add_44, [128, 1024])
    permute_56: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    addmm_31: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg86_1, view_121, permute_56);  arg86_1 = view_121 = permute_56 = None
    view_122: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_31, [1, 128, 1024]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_123: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_122, [1, -1, 16, 64]);  view_122 = None
    permute_57: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1, 3]);  view_123 = None
    clone_41: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_124: "f32[128, 1024]" = torch.ops.aten.view.default(add_44, [128, 1024]);  add_44 = None
    permute_58: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    addmm_32: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg88_1, view_124, permute_58);  arg88_1 = view_124 = permute_58 = None
    view_125: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_32, [1, 128, 1024]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_126: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_125, [1, -1, 16, 64]);  view_125 = None
    permute_59: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
    clone_42: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_127: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_43, [1, 128, 16, 64]);  mul_43 = None
    permute_60: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_127, [0, 2, 1, 3]);  view_127 = None
    clone_43: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_60, memory_format = torch.contiguous_format);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_128: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_43, [16, -1, 64]);  clone_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_129: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_41, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_130: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_42, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_61: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_129, [0, 2, 1]);  view_129 = None
    bmm_10: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_128, permute_61);  view_128 = permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_131: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_10, [1, 16, 128, 128]);  bmm_10 = None
    add_45: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_131, expand_1);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant5 = self._tensor_constant5
    lift_fresh_copy_5: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant5);  _tensor_constant5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_5: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_45, lift_fresh_copy_5);  add_45 = lift_fresh_copy_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_132: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_5, [16, 128, 128]);  maximum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_5: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_132, [-1], True)
    sub_16: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_132, amax_5);  view_132 = amax_5 = None
    exp_5: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_44: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_11: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_44, view_130);  clone_44 = view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_133: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_11, [1, 16, 128, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_62: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_45: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_134: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_45, [1, 128, 1024]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_135: "f32[128, 1024]" = torch.ops.aten.view.default(view_134, [128, 1024]);  view_134 = None
    permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
    addmm_33: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg90_1, view_135, permute_63);  arg90_1 = view_135 = permute_63 = None
    view_136: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_33, [1, 128, 1024]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_46: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_136);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_46: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_42, clone_46);  add_42 = clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1, 128, 1]" = var_mean_11[0]
    getitem_23: "f32[1, 128, 1]" = var_mean_11[1];  var_mean_11 = None
    add_47: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_17: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_46, getitem_23);  getitem_23 = None
    mul_44: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = rsqrt_11 = None
    mul_45: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_44, arg91_1);  mul_44 = arg91_1 = None
    add_48: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_45, arg92_1);  mul_45 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_137: "f32[128, 1024]" = torch.ops.aten.view.default(add_48, [128, 1024]);  add_48 = None
    permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    addmm_34: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg94_1, view_137, permute_64);  arg94_1 = view_137 = permute_64 = None
    view_138: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_34, [1, 128, 4096]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_46: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_138, 0.5)
    mul_47: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_138, 0.7071067811865476);  view_138 = None
    erf_5: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_49: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_48: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_47: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_139: "f32[128, 4096]" = torch.ops.aten.view.default(clone_47, [128, 4096]);  clone_47 = None
    permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
    addmm_35: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg96_1, view_139, permute_65);  arg96_1 = view_139 = permute_65 = None
    view_140: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_35, [1, 128, 1024]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_48: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_50: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_46, clone_48);  add_46 = clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1, 128, 1]" = var_mean_12[0]
    getitem_25: "f32[1, 128, 1]" = var_mean_12[1];  var_mean_12 = None
    add_51: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_18: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_50, getitem_25);  getitem_25 = None
    mul_49: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = rsqrt_12 = None
    mul_50: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_49, arg97_1);  mul_49 = arg97_1 = None
    add_52: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_50, arg98_1);  mul_50 = arg98_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_141: "f32[128, 1024]" = torch.ops.aten.view.default(add_52, [128, 1024])
    permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
    addmm_36: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg100_1, view_141, permute_66);  arg100_1 = view_141 = permute_66 = None
    view_142: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_36, [1, 128, 1024]);  addmm_36 = None
    mul_51: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_142, 0.125);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_143: "f32[128, 1024]" = torch.ops.aten.view.default(add_52, [128, 1024])
    permute_67: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
    addmm_37: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg102_1, view_143, permute_67);  arg102_1 = view_143 = permute_67 = None
    view_144: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_37, [1, 128, 1024]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_145: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_144, [1, -1, 16, 64]);  view_144 = None
    permute_68: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_145, [0, 2, 1, 3]);  view_145 = None
    clone_49: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_146: "f32[128, 1024]" = torch.ops.aten.view.default(add_52, [128, 1024]);  add_52 = None
    permute_69: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    addmm_38: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg104_1, view_146, permute_69);  arg104_1 = view_146 = permute_69 = None
    view_147: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_38, [1, 128, 1024]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_148: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_147, [1, -1, 16, 64]);  view_147 = None
    permute_70: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    clone_50: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_149: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_51, [1, 128, 16, 64]);  mul_51 = None
    permute_71: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_149, [0, 2, 1, 3]);  view_149 = None
    clone_51: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_71, memory_format = torch.contiguous_format);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_150: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_51, [16, -1, 64]);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_151: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_49, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_152: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_50, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_72: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
    bmm_12: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_150, permute_72);  view_150 = permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_153: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_12, [1, 16, 128, 128]);  bmm_12 = None
    add_53: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_153, expand_1);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant6 = self._tensor_constant6
    lift_fresh_copy_6: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant6);  _tensor_constant6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_6: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_53, lift_fresh_copy_6);  add_53 = lift_fresh_copy_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_154: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_6, [16, 128, 128]);  maximum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_6: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_154, [-1], True)
    sub_19: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_154, amax_6);  view_154 = amax_6 = None
    exp_6: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_52: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_13: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_52, view_152);  clone_52 = view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_155: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_13, [1, 16, 128, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_73: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_53: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_156: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_53, [1, 128, 1024]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_157: "f32[128, 1024]" = torch.ops.aten.view.default(view_156, [128, 1024]);  view_156 = None
    permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    addmm_39: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg106_1, view_157, permute_74);  arg106_1 = view_157 = permute_74 = None
    view_158: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_39, [1, 128, 1024]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_54: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_158);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_54: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_50, clone_54);  add_50 = clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_54, [2], correction = 0, keepdim = True)
    getitem_26: "f32[1, 128, 1]" = var_mean_13[0]
    getitem_27: "f32[1, 128, 1]" = var_mean_13[1];  var_mean_13 = None
    add_55: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_20: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_54, getitem_27);  getitem_27 = None
    mul_52: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = rsqrt_13 = None
    mul_53: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_52, arg107_1);  mul_52 = arg107_1 = None
    add_56: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_53, arg108_1);  mul_53 = arg108_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_159: "f32[128, 1024]" = torch.ops.aten.view.default(add_56, [128, 1024]);  add_56 = None
    permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    addmm_40: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg110_1, view_159, permute_75);  arg110_1 = view_159 = permute_75 = None
    view_160: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_40, [1, 128, 4096]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_54: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_160, 0.5)
    mul_55: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_160, 0.7071067811865476);  view_160 = None
    erf_6: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_55);  mul_55 = None
    add_57: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_56: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_54, add_57);  mul_54 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_55: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_56);  mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_161: "f32[128, 4096]" = torch.ops.aten.view.default(clone_55, [128, 4096]);  clone_55 = None
    permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
    addmm_41: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg112_1, view_161, permute_76);  arg112_1 = view_161 = permute_76 = None
    view_162: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_41, [1, 128, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_56: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_162);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_58: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_54, clone_56);  add_54 = clone_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_58, [2], correction = 0, keepdim = True)
    getitem_28: "f32[1, 128, 1]" = var_mean_14[0]
    getitem_29: "f32[1, 128, 1]" = var_mean_14[1];  var_mean_14 = None
    add_59: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_21: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_58, getitem_29);  getitem_29 = None
    mul_57: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = rsqrt_14 = None
    mul_58: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_57, arg113_1);  mul_57 = arg113_1 = None
    add_60: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_58, arg114_1);  mul_58 = arg114_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_163: "f32[128, 1024]" = torch.ops.aten.view.default(add_60, [128, 1024])
    permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    addmm_42: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg116_1, view_163, permute_77);  arg116_1 = view_163 = permute_77 = None
    view_164: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_42, [1, 128, 1024]);  addmm_42 = None
    mul_59: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_164, 0.125);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_165: "f32[128, 1024]" = torch.ops.aten.view.default(add_60, [128, 1024])
    permute_78: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg117_1, [1, 0]);  arg117_1 = None
    addmm_43: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg118_1, view_165, permute_78);  arg118_1 = view_165 = permute_78 = None
    view_166: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_43, [1, 128, 1024]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_167: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_166, [1, -1, 16, 64]);  view_166 = None
    permute_79: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1, 3]);  view_167 = None
    clone_57: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_79, memory_format = torch.contiguous_format);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_168: "f32[128, 1024]" = torch.ops.aten.view.default(add_60, [128, 1024]);  add_60 = None
    permute_80: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    addmm_44: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg120_1, view_168, permute_80);  arg120_1 = view_168 = permute_80 = None
    view_169: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_44, [1, 128, 1024]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_170: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_169, [1, -1, 16, 64]);  view_169 = None
    permute_81: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
    clone_58: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_171: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_59, [1, 128, 16, 64]);  mul_59 = None
    permute_82: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_171, [0, 2, 1, 3]);  view_171 = None
    clone_59: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_172: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_59, [16, -1, 64]);  clone_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_173: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_57, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_174: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_58, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_83: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None
    bmm_14: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_172, permute_83);  view_172 = permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_175: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_14, [1, 16, 128, 128]);  bmm_14 = None
    add_61: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_175, expand_1);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant7 = self._tensor_constant7
    lift_fresh_copy_7: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant7);  _tensor_constant7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_7: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_61, lift_fresh_copy_7);  add_61 = lift_fresh_copy_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_176: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_7, [16, 128, 128]);  maximum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_7: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_176, [-1], True)
    sub_22: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_176, amax_7);  view_176 = amax_7 = None
    exp_7: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_60: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_7);  div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_15: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_60, view_174);  clone_60 = view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_177: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_15, [1, 16, 128, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_84: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_61: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_178: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_61, [1, 128, 1024]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_179: "f32[128, 1024]" = torch.ops.aten.view.default(view_178, [128, 1024]);  view_178 = None
    permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    addmm_45: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg122_1, view_179, permute_85);  arg122_1 = view_179 = permute_85 = None
    view_180: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_45, [1, 128, 1024]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_62: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_62: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_58, clone_62);  add_58 = clone_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
    getitem_30: "f32[1, 128, 1]" = var_mean_15[0]
    getitem_31: "f32[1, 128, 1]" = var_mean_15[1];  var_mean_15 = None
    add_63: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_23: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_62, getitem_31);  getitem_31 = None
    mul_60: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = rsqrt_15 = None
    mul_61: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_60, arg123_1);  mul_60 = arg123_1 = None
    add_64: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_61, arg124_1);  mul_61 = arg124_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_181: "f32[128, 1024]" = torch.ops.aten.view.default(add_64, [128, 1024]);  add_64 = None
    permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    addmm_46: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg126_1, view_181, permute_86);  arg126_1 = view_181 = permute_86 = None
    view_182: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_46, [1, 128, 4096]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_62: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_182, 0.5)
    mul_63: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_182, 0.7071067811865476);  view_182 = None
    erf_7: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_65: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_64: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_62, add_65);  mul_62 = add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_63: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_64);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_183: "f32[128, 4096]" = torch.ops.aten.view.default(clone_63, [128, 4096]);  clone_63 = None
    permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    addmm_47: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg128_1, view_183, permute_87);  arg128_1 = view_183 = permute_87 = None
    view_184: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_47, [1, 128, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_64: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_184);  view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_66: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_62, clone_64);  add_62 = clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1, 128, 1]" = var_mean_16[0]
    getitem_33: "f32[1, 128, 1]" = var_mean_16[1];  var_mean_16 = None
    add_67: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_24: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_66, getitem_33);  getitem_33 = None
    mul_65: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = rsqrt_16 = None
    mul_66: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_65, arg129_1);  mul_65 = arg129_1 = None
    add_68: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_66, arg130_1);  mul_66 = arg130_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_185: "f32[128, 1024]" = torch.ops.aten.view.default(add_68, [128, 1024])
    permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    addmm_48: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg132_1, view_185, permute_88);  arg132_1 = view_185 = permute_88 = None
    view_186: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_48, [1, 128, 1024]);  addmm_48 = None
    mul_67: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_186, 0.125);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_187: "f32[128, 1024]" = torch.ops.aten.view.default(add_68, [128, 1024])
    permute_89: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    addmm_49: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg134_1, view_187, permute_89);  arg134_1 = view_187 = permute_89 = None
    view_188: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_49, [1, 128, 1024]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_189: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_188, [1, -1, 16, 64]);  view_188 = None
    permute_90: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_189, [0, 2, 1, 3]);  view_189 = None
    clone_65: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_190: "f32[128, 1024]" = torch.ops.aten.view.default(add_68, [128, 1024]);  add_68 = None
    permute_91: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
    addmm_50: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg136_1, view_190, permute_91);  arg136_1 = view_190 = permute_91 = None
    view_191: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_50, [1, 128, 1024]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_192: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_191, [1, -1, 16, 64]);  view_191 = None
    permute_92: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    clone_66: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_193: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_67, [1, 128, 16, 64]);  mul_67 = None
    permute_93: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_193, [0, 2, 1, 3]);  view_193 = None
    clone_67: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_194: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_67, [16, -1, 64]);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_195: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_65, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_196: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_66, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_94: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
    bmm_16: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_194, permute_94);  view_194 = permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_197: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_16, [1, 16, 128, 128]);  bmm_16 = None
    add_69: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_197, expand_1);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant8 = self._tensor_constant8
    lift_fresh_copy_8: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant8);  _tensor_constant8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_8: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_69, lift_fresh_copy_8);  add_69 = lift_fresh_copy_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_198: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_8, [16, 128, 128]);  maximum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_8: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_198, [-1], True)
    sub_25: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_198, amax_8);  view_198 = amax_8 = None
    exp_8: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_68: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_8);  div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_17: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_68, view_196);  clone_68 = view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_199: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_17, [1, 16, 128, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_95: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_69: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_200: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_69, [1, 128, 1024]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_201: "f32[128, 1024]" = torch.ops.aten.view.default(view_200, [128, 1024]);  view_200 = None
    permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    addmm_51: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg138_1, view_201, permute_96);  arg138_1 = view_201 = permute_96 = None
    view_202: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_51, [1, 128, 1024]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_70: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_202);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_70: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_66, clone_70);  add_66 = clone_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_34: "f32[1, 128, 1]" = var_mean_17[0]
    getitem_35: "f32[1, 128, 1]" = var_mean_17[1];  var_mean_17 = None
    add_71: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_34, 1e-05);  getitem_34 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_26: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_70, getitem_35);  getitem_35 = None
    mul_68: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = rsqrt_17 = None
    mul_69: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_68, arg139_1);  mul_68 = arg139_1 = None
    add_72: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_69, arg140_1);  mul_69 = arg140_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_203: "f32[128, 1024]" = torch.ops.aten.view.default(add_72, [128, 1024]);  add_72 = None
    permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    addmm_52: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg142_1, view_203, permute_97);  arg142_1 = view_203 = permute_97 = None
    view_204: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_52, [1, 128, 4096]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_70: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_204, 0.5)
    mul_71: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_204, 0.7071067811865476);  view_204 = None
    erf_8: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_71);  mul_71 = None
    add_73: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_72: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_70, add_73);  mul_70 = add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_71: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_72);  mul_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_205: "f32[128, 4096]" = torch.ops.aten.view.default(clone_71, [128, 4096]);  clone_71 = None
    permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    addmm_53: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg144_1, view_205, permute_98);  arg144_1 = view_205 = permute_98 = None
    view_206: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_53, [1, 128, 1024]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_72: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_206);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_74: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_70, clone_72);  add_70 = clone_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1, 128, 1]" = var_mean_18[0]
    getitem_37: "f32[1, 128, 1]" = var_mean_18[1];  var_mean_18 = None
    add_75: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    sub_27: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_74, getitem_37);  getitem_37 = None
    mul_73: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = rsqrt_18 = None
    mul_74: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_73, arg145_1);  mul_73 = arg145_1 = None
    add_76: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_74, arg146_1);  mul_74 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_207: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    addmm_54: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg148_1, view_207, permute_99);  arg148_1 = view_207 = permute_99 = None
    view_208: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_54, [1, 128, 1024]);  addmm_54 = None
    mul_75: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_208, 0.125);  view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_209: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024])
    permute_100: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    addmm_55: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg150_1, view_209, permute_100);  arg150_1 = view_209 = permute_100 = None
    view_210: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_55, [1, 128, 1024]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_211: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_210, [1, -1, 16, 64]);  view_210 = None
    permute_101: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_211, [0, 2, 1, 3]);  view_211 = None
    clone_73: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_101, memory_format = torch.contiguous_format);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_212: "f32[128, 1024]" = torch.ops.aten.view.default(add_76, [128, 1024]);  add_76 = None
    permute_102: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
    addmm_56: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg152_1, view_212, permute_102);  arg152_1 = view_212 = permute_102 = None
    view_213: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_56, [1, 128, 1024]);  addmm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_214: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_213, [1, -1, 16, 64]);  view_213 = None
    permute_103: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_214, [0, 2, 1, 3]);  view_214 = None
    clone_74: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_103, memory_format = torch.contiguous_format);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_215: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_75, [1, 128, 16, 64]);  mul_75 = None
    permute_104: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
    clone_75: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_216: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_75, [16, -1, 64]);  clone_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_217: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_73, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_218: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_74, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_105: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None
    bmm_18: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_216, permute_105);  view_216 = permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_219: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_18, [1, 16, 128, 128]);  bmm_18 = None
    add_77: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_219, expand_1);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant9 = self._tensor_constant9
    lift_fresh_copy_9: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant9);  _tensor_constant9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_9: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_77, lift_fresh_copy_9);  add_77 = lift_fresh_copy_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_220: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_9, [16, 128, 128]);  maximum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_9: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_220, [-1], True)
    sub_28: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_220, amax_9);  view_220 = amax_9 = None
    exp_9: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_76: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_9);  div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_19: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_76, view_218);  clone_76 = view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_221: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_19, [1, 16, 128, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_106: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_77: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    view_222: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_77, [1, 128, 1024]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_223: "f32[128, 1024]" = torch.ops.aten.view.default(view_222, [128, 1024]);  view_222 = None
    permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    addmm_57: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg154_1, view_223, permute_107);  arg154_1 = view_223 = permute_107 = None
    view_224: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_57, [1, 128, 1024]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_78: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_224);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_78: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_74, clone_78);  add_74 = clone_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_78, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1, 128, 1]" = var_mean_19[0]
    getitem_39: "f32[1, 128, 1]" = var_mean_19[1];  var_mean_19 = None
    add_79: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_29: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_78, getitem_39);  getitem_39 = None
    mul_76: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = rsqrt_19 = None
    mul_77: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_76, arg155_1);  mul_76 = arg155_1 = None
    add_80: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_77, arg156_1);  mul_77 = arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_225: "f32[128, 1024]" = torch.ops.aten.view.default(add_80, [128, 1024]);  add_80 = None
    permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
    addmm_58: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg158_1, view_225, permute_108);  arg158_1 = view_225 = permute_108 = None
    view_226: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_58, [1, 128, 4096]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_78: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_226, 0.5)
    mul_79: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_226, 0.7071067811865476);  view_226 = None
    erf_9: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_79);  mul_79 = None
    add_81: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_80: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_78, add_81);  mul_78 = add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_79: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_80);  mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_227: "f32[128, 4096]" = torch.ops.aten.view.default(clone_79, [128, 4096]);  clone_79 = None
    permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    addmm_59: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg160_1, view_227, permute_109);  arg160_1 = view_227 = permute_109 = None
    view_228: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_59, [1, 128, 1024]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_80: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_228);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_82: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_78, clone_80);  add_78 = clone_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
    getitem_40: "f32[1, 128, 1]" = var_mean_20[0]
    getitem_41: "f32[1, 128, 1]" = var_mean_20[1];  var_mean_20 = None
    add_83: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_30: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_82, getitem_41);  getitem_41 = None
    mul_81: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = rsqrt_20 = None
    mul_82: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_81, arg161_1);  mul_81 = arg161_1 = None
    add_84: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_82, arg162_1);  mul_82 = arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_229: "f32[128, 1024]" = torch.ops.aten.view.default(add_84, [128, 1024])
    permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg163_1, [1, 0]);  arg163_1 = None
    addmm_60: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg164_1, view_229, permute_110);  arg164_1 = view_229 = permute_110 = None
    view_230: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_60, [1, 128, 1024]);  addmm_60 = None
    mul_83: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_230, 0.125);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_231: "f32[128, 1024]" = torch.ops.aten.view.default(add_84, [128, 1024])
    permute_111: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    addmm_61: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg166_1, view_231, permute_111);  arg166_1 = view_231 = permute_111 = None
    view_232: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_61, [1, 128, 1024]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_233: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_232, [1, -1, 16, 64]);  view_232 = None
    permute_112: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
    clone_81: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_112, memory_format = torch.contiguous_format);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_234: "f32[128, 1024]" = torch.ops.aten.view.default(add_84, [128, 1024]);  add_84 = None
    permute_113: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
    addmm_62: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg168_1, view_234, permute_113);  arg168_1 = view_234 = permute_113 = None
    view_235: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_62, [1, 128, 1024]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_236: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_235, [1, -1, 16, 64]);  view_235 = None
    permute_114: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    clone_82: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_237: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_83, [1, 128, 16, 64]);  mul_83 = None
    permute_115: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
    clone_83: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_115, memory_format = torch.contiguous_format);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_238: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_83, [16, -1, 64]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_239: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_81, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_240: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_82, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_116: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
    bmm_20: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_238, permute_116);  view_238 = permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_241: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_20, [1, 16, 128, 128]);  bmm_20 = None
    add_85: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_241, expand_1);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant10 = self._tensor_constant10
    lift_fresh_copy_10: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant10);  _tensor_constant10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_10: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_85, lift_fresh_copy_10);  add_85 = lift_fresh_copy_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_242: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_10, [16, 128, 128]);  maximum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_10: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_242, [-1], True)
    sub_31: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_242, amax_10);  view_242 = amax_10 = None
    exp_10: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_84: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_21: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_84, view_240);  clone_84 = view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_243: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_21, [1, 16, 128, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_117: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_85: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_244: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_85, [1, 128, 1024]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_245: "f32[128, 1024]" = torch.ops.aten.view.default(view_244, [128, 1024]);  view_244 = None
    permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    addmm_63: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg170_1, view_245, permute_118);  arg170_1 = view_245 = permute_118 = None
    view_246: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_63, [1, 128, 1024]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_86: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_246);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_86: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_82, clone_86);  add_82 = clone_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
    getitem_42: "f32[1, 128, 1]" = var_mean_21[0]
    getitem_43: "f32[1, 128, 1]" = var_mean_21[1];  var_mean_21 = None
    add_87: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-05);  getitem_42 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_32: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_86, getitem_43);  getitem_43 = None
    mul_84: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = rsqrt_21 = None
    mul_85: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_84, arg171_1);  mul_84 = arg171_1 = None
    add_88: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_85, arg172_1);  mul_85 = arg172_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_247: "f32[128, 1024]" = torch.ops.aten.view.default(add_88, [128, 1024]);  add_88 = None
    permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
    addmm_64: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg174_1, view_247, permute_119);  arg174_1 = view_247 = permute_119 = None
    view_248: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_64, [1, 128, 4096]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_86: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_248, 0.5)
    mul_87: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_248, 0.7071067811865476);  view_248 = None
    erf_10: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_89: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_88: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_86, add_89);  mul_86 = add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_87: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_249: "f32[128, 4096]" = torch.ops.aten.view.default(clone_87, [128, 4096]);  clone_87 = None
    permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    addmm_65: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg176_1, view_249, permute_120);  arg176_1 = view_249 = permute_120 = None
    view_250: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_65, [1, 128, 1024]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_88: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_250);  view_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_90: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_86, clone_88);  add_86 = clone_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_90, [2], correction = 0, keepdim = True)
    getitem_44: "f32[1, 128, 1]" = var_mean_22[0]
    getitem_45: "f32[1, 128, 1]" = var_mean_22[1];  var_mean_22 = None
    add_91: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    sub_33: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_90, getitem_45);  getitem_45 = None
    mul_89: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = rsqrt_22 = None
    mul_90: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_89, arg177_1);  mul_89 = arg177_1 = None
    add_92: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_90, arg178_1);  mul_90 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_251: "f32[128, 1024]" = torch.ops.aten.view.default(add_92, [128, 1024])
    permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
    addmm_66: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg180_1, view_251, permute_121);  arg180_1 = view_251 = permute_121 = None
    view_252: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_66, [1, 128, 1024]);  addmm_66 = None
    mul_91: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_252, 0.125);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_253: "f32[128, 1024]" = torch.ops.aten.view.default(add_92, [128, 1024])
    permute_122: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    addmm_67: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg182_1, view_253, permute_122);  arg182_1 = view_253 = permute_122 = None
    view_254: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_67, [1, 128, 1024]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_255: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_254, [1, -1, 16, 64]);  view_254 = None
    permute_123: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    clone_89: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_256: "f32[128, 1024]" = torch.ops.aten.view.default(add_92, [128, 1024]);  add_92 = None
    permute_124: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    addmm_68: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg184_1, view_256, permute_124);  arg184_1 = view_256 = permute_124 = None
    view_257: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_68, [1, 128, 1024]);  addmm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_258: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_257, [1, -1, 16, 64]);  view_257 = None
    permute_125: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    clone_90: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_259: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_91, [1, 128, 16, 64]);  mul_91 = None
    permute_126: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_259, [0, 2, 1, 3]);  view_259 = None
    clone_91: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_260: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_91, [16, -1, 64]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_261: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_89, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_262: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_90, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_127: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_261, [0, 2, 1]);  view_261 = None
    bmm_22: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_260, permute_127);  view_260 = permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_263: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_22, [1, 16, 128, 128]);  bmm_22 = None
    add_93: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_263, expand_1);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant11 = self._tensor_constant11
    lift_fresh_copy_11: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant11);  _tensor_constant11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_11: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_93, lift_fresh_copy_11);  add_93 = lift_fresh_copy_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_264: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_11, [16, 128, 128]);  maximum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_11: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_264, [-1], True)
    sub_34: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_264, amax_11);  view_264 = amax_11 = None
    exp_11: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_92: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_11);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_23: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_92, view_262);  clone_92 = view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_265: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_23, [1, 16, 128, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_128: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_265, [0, 2, 1, 3]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_93: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_266: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_93, [1, 128, 1024]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_267: "f32[128, 1024]" = torch.ops.aten.view.default(view_266, [128, 1024]);  view_266 = None
    permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
    addmm_69: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg186_1, view_267, permute_129);  arg186_1 = view_267 = permute_129 = None
    view_268: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_69, [1, 128, 1024]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_94: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_268);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_94: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_90, clone_94);  add_90 = clone_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_94, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1, 128, 1]" = var_mean_23[0]
    getitem_47: "f32[1, 128, 1]" = var_mean_23[1];  var_mean_23 = None
    add_95: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    sub_35: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_94, getitem_47);  getitem_47 = None
    mul_92: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = rsqrt_23 = None
    mul_93: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_92, arg187_1);  mul_92 = arg187_1 = None
    add_96: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_93, arg188_1);  mul_93 = arg188_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_269: "f32[128, 1024]" = torch.ops.aten.view.default(add_96, [128, 1024]);  add_96 = None
    permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    addmm_70: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg190_1, view_269, permute_130);  arg190_1 = view_269 = permute_130 = None
    view_270: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_70, [1, 128, 4096]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_94: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_270, 0.5)
    mul_95: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_270, 0.7071067811865476);  view_270 = None
    erf_11: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_95);  mul_95 = None
    add_97: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_96: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_94, add_97);  mul_94 = add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_95: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_96);  mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_271: "f32[128, 4096]" = torch.ops.aten.view.default(clone_95, [128, 4096]);  clone_95 = None
    permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
    addmm_71: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg192_1, view_271, permute_131);  arg192_1 = view_271 = permute_131 = None
    view_272: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_71, [1, 128, 1024]);  addmm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_96: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_272);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_98: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_94, clone_96);  add_94 = clone_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_48: "f32[1, 128, 1]" = var_mean_24[0]
    getitem_49: "f32[1, 128, 1]" = var_mean_24[1];  var_mean_24 = None
    add_99: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_36: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_98, getitem_49);  getitem_49 = None
    mul_97: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = rsqrt_24 = None
    mul_98: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_97, arg193_1);  mul_97 = arg193_1 = None
    add_100: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_98, arg194_1);  mul_98 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_273: "f32[128, 1024]" = torch.ops.aten.view.default(add_100, [128, 1024])
    permute_132: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg195_1, [1, 0]);  arg195_1 = None
    addmm_72: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg196_1, view_273, permute_132);  arg196_1 = view_273 = permute_132 = None
    view_274: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_72, [1, 128, 1024]);  addmm_72 = None
    mul_99: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_274, 0.125);  view_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_275: "f32[128, 1024]" = torch.ops.aten.view.default(add_100, [128, 1024])
    permute_133: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    addmm_73: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg198_1, view_275, permute_133);  arg198_1 = view_275 = permute_133 = None
    view_276: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_73, [1, 128, 1024]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_277: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_276, [1, -1, 16, 64]);  view_276 = None
    permute_134: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
    clone_97: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_134, memory_format = torch.contiguous_format);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_278: "f32[128, 1024]" = torch.ops.aten.view.default(add_100, [128, 1024]);  add_100 = None
    permute_135: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
    addmm_74: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg200_1, view_278, permute_135);  arg200_1 = view_278 = permute_135 = None
    view_279: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_74, [1, 128, 1024]);  addmm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_280: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_279, [1, -1, 16, 64]);  view_279 = None
    permute_136: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_280, [0, 2, 1, 3]);  view_280 = None
    clone_98: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_136, memory_format = torch.contiguous_format);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_281: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_99, [1, 128, 16, 64]);  mul_99 = None
    permute_137: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None
    clone_99: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_137, memory_format = torch.contiguous_format);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_282: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_99, [16, -1, 64]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_283: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_97, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_284: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_98, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_138: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_283, [0, 2, 1]);  view_283 = None
    bmm_24: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_282, permute_138);  view_282 = permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_285: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_24, [1, 16, 128, 128]);  bmm_24 = None
    add_101: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_285, expand_1);  view_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant12 = self._tensor_constant12
    lift_fresh_copy_12: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant12);  _tensor_constant12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_12: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_101, lift_fresh_copy_12);  add_101 = lift_fresh_copy_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_286: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_12, [16, 128, 128]);  maximum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_12: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_286, [-1], True)
    sub_37: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_286, amax_12);  view_286 = amax_12 = None
    exp_12: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
    sum_13: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_100: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_12);  div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_25: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_100, view_284);  clone_100 = view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_287: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_25, [1, 16, 128, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_139: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_287, [0, 2, 1, 3]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_101: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_139, memory_format = torch.contiguous_format);  permute_139 = None
    view_288: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_101, [1, 128, 1024]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_289: "f32[128, 1024]" = torch.ops.aten.view.default(view_288, [128, 1024]);  view_288 = None
    permute_140: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
    addmm_75: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg202_1, view_289, permute_140);  arg202_1 = view_289 = permute_140 = None
    view_290: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_75, [1, 128, 1024]);  addmm_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_102: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_290);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_102: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_98, clone_102);  add_98 = clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_25 = torch.ops.aten.var_mean.correction(add_102, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1, 128, 1]" = var_mean_25[0]
    getitem_51: "f32[1, 128, 1]" = var_mean_25[1];  var_mean_25 = None
    add_103: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    sub_38: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_102, getitem_51);  getitem_51 = None
    mul_100: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = rsqrt_25 = None
    mul_101: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_100, arg203_1);  mul_100 = arg203_1 = None
    add_104: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_101, arg204_1);  mul_101 = arg204_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_291: "f32[128, 1024]" = torch.ops.aten.view.default(add_104, [128, 1024]);  add_104 = None
    permute_141: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
    addmm_76: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg206_1, view_291, permute_141);  arg206_1 = view_291 = permute_141 = None
    view_292: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_76, [1, 128, 4096]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_102: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_292, 0.5)
    mul_103: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_292, 0.7071067811865476);  view_292 = None
    erf_12: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
    add_105: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_104: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_102, add_105);  mul_102 = add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_103: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_104);  mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_293: "f32[128, 4096]" = torch.ops.aten.view.default(clone_103, [128, 4096]);  clone_103 = None
    permute_142: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
    addmm_77: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg208_1, view_293, permute_142);  arg208_1 = view_293 = permute_142 = None
    view_294: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_77, [1, 128, 1024]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_104: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_294);  view_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_106: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_102, clone_104);  add_102 = clone_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_26 = torch.ops.aten.var_mean.correction(add_106, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1, 128, 1]" = var_mean_26[0]
    getitem_53: "f32[1, 128, 1]" = var_mean_26[1];  var_mean_26 = None
    add_107: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
    sub_39: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_106, getitem_53);  getitem_53 = None
    mul_105: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_26);  sub_39 = rsqrt_26 = None
    mul_106: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_105, arg209_1);  mul_105 = arg209_1 = None
    add_108: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_106, arg210_1);  mul_106 = arg210_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_295: "f32[128, 1024]" = torch.ops.aten.view.default(add_108, [128, 1024])
    permute_143: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
    addmm_78: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg212_1, view_295, permute_143);  arg212_1 = view_295 = permute_143 = None
    view_296: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_78, [1, 128, 1024]);  addmm_78 = None
    mul_107: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_296, 0.125);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_297: "f32[128, 1024]" = torch.ops.aten.view.default(add_108, [128, 1024])
    permute_144: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
    addmm_79: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg214_1, view_297, permute_144);  arg214_1 = view_297 = permute_144 = None
    view_298: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_79, [1, 128, 1024]);  addmm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_299: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_298, [1, -1, 16, 64]);  view_298 = None
    permute_145: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_299, [0, 2, 1, 3]);  view_299 = None
    clone_105: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_300: "f32[128, 1024]" = torch.ops.aten.view.default(add_108, [128, 1024]);  add_108 = None
    permute_146: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
    addmm_80: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg216_1, view_300, permute_146);  arg216_1 = view_300 = permute_146 = None
    view_301: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_80, [1, 128, 1024]);  addmm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_302: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_301, [1, -1, 16, 64]);  view_301 = None
    permute_147: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None
    clone_106: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_303: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_107, [1, 128, 16, 64]);  mul_107 = None
    permute_148: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
    clone_107: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_304: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_107, [16, -1, 64]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_305: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_105, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_306: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_106, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_149: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_305, [0, 2, 1]);  view_305 = None
    bmm_26: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_304, permute_149);  view_304 = permute_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_307: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_26, [1, 16, 128, 128]);  bmm_26 = None
    add_109: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_307, expand_1);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant13 = self._tensor_constant13
    lift_fresh_copy_13: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant13);  _tensor_constant13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_13: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_109, lift_fresh_copy_13);  add_109 = lift_fresh_copy_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_308: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_13, [16, 128, 128]);  maximum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_13: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_308, [-1], True)
    sub_40: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_308, amax_13);  view_308 = amax_13 = None
    exp_13: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_40);  sub_40 = None
    sum_14: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_108: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_13);  div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_27: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_108, view_306);  clone_108 = view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_309: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_27, [1, 16, 128, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_150: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_109: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    view_310: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_109, [1, 128, 1024]);  clone_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_311: "f32[128, 1024]" = torch.ops.aten.view.default(view_310, [128, 1024]);  view_310 = None
    permute_151: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
    addmm_81: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg218_1, view_311, permute_151);  arg218_1 = view_311 = permute_151 = None
    view_312: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_81, [1, 128, 1024]);  addmm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_110: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_312);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_110: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_106, clone_110);  add_106 = clone_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_27 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
    getitem_54: "f32[1, 128, 1]" = var_mean_27[0]
    getitem_55: "f32[1, 128, 1]" = var_mean_27[1];  var_mean_27 = None
    add_111: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    sub_41: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_110, getitem_55);  getitem_55 = None
    mul_108: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = rsqrt_27 = None
    mul_109: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_108, arg219_1);  mul_108 = arg219_1 = None
    add_112: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_109, arg220_1);  mul_109 = arg220_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_313: "f32[128, 1024]" = torch.ops.aten.view.default(add_112, [128, 1024]);  add_112 = None
    permute_152: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
    addmm_82: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg222_1, view_313, permute_152);  arg222_1 = view_313 = permute_152 = None
    view_314: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_82, [1, 128, 4096]);  addmm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_110: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_314, 0.5)
    mul_111: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_314, 0.7071067811865476);  view_314 = None
    erf_13: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_111);  mul_111 = None
    add_113: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_112: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_110, add_113);  mul_110 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_111: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_112);  mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_315: "f32[128, 4096]" = torch.ops.aten.view.default(clone_111, [128, 4096]);  clone_111 = None
    permute_153: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
    addmm_83: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg224_1, view_315, permute_153);  arg224_1 = view_315 = permute_153 = None
    view_316: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_83, [1, 128, 1024]);  addmm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_112: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_316);  view_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_114: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_110, clone_112);  add_110 = clone_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_28 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
    getitem_56: "f32[1, 128, 1]" = var_mean_28[0]
    getitem_57: "f32[1, 128, 1]" = var_mean_28[1];  var_mean_28 = None
    add_115: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    sub_42: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_114, getitem_57);  getitem_57 = None
    mul_113: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_28);  sub_42 = rsqrt_28 = None
    mul_114: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_113, arg225_1);  mul_113 = arg225_1 = None
    add_116: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_114, arg226_1);  mul_114 = arg226_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_317: "f32[128, 1024]" = torch.ops.aten.view.default(add_116, [128, 1024])
    permute_154: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
    addmm_84: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg228_1, view_317, permute_154);  arg228_1 = view_317 = permute_154 = None
    view_318: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_84, [1, 128, 1024]);  addmm_84 = None
    mul_115: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_318, 0.125);  view_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_319: "f32[128, 1024]" = torch.ops.aten.view.default(add_116, [128, 1024])
    permute_155: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
    addmm_85: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg230_1, view_319, permute_155);  arg230_1 = view_319 = permute_155 = None
    view_320: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_85, [1, 128, 1024]);  addmm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_321: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_320, [1, -1, 16, 64]);  view_320 = None
    permute_156: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_321, [0, 2, 1, 3]);  view_321 = None
    clone_113: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_322: "f32[128, 1024]" = torch.ops.aten.view.default(add_116, [128, 1024]);  add_116 = None
    permute_157: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
    addmm_86: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg232_1, view_322, permute_157);  arg232_1 = view_322 = permute_157 = None
    view_323: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_86, [1, 128, 1024]);  addmm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_324: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_323, [1, -1, 16, 64]);  view_323 = None
    permute_158: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
    clone_114: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_158, memory_format = torch.contiguous_format);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_325: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_115, [1, 128, 16, 64]);  mul_115 = None
    permute_159: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
    clone_115: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_159, memory_format = torch.contiguous_format);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_326: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_115, [16, -1, 64]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_327: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_113, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_328: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_114, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_160: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_327, [0, 2, 1]);  view_327 = None
    bmm_28: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_326, permute_160);  view_326 = permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_329: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_28, [1, 16, 128, 128]);  bmm_28 = None
    add_117: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_329, expand_1);  view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant14 = self._tensor_constant14
    lift_fresh_copy_14: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant14);  _tensor_constant14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_14: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_117, lift_fresh_copy_14);  add_117 = lift_fresh_copy_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_330: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_14, [16, 128, 128]);  maximum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_14: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_330, [-1], True)
    sub_43: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_330, amax_14);  view_330 = amax_14 = None
    exp_14: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_43);  sub_43 = None
    sum_15: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_116: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_29: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_116, view_328);  clone_116 = view_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_331: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_29, [1, 16, 128, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_161: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_331, [0, 2, 1, 3]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_117: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_332: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_117, [1, 128, 1024]);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_333: "f32[128, 1024]" = torch.ops.aten.view.default(view_332, [128, 1024]);  view_332 = None
    permute_162: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
    addmm_87: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg234_1, view_333, permute_162);  arg234_1 = view_333 = permute_162 = None
    view_334: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_87, [1, 128, 1024]);  addmm_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_118: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_334);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_118: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_114, clone_118);  add_114 = clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_29 = torch.ops.aten.var_mean.correction(add_118, [2], correction = 0, keepdim = True)
    getitem_58: "f32[1, 128, 1]" = var_mean_29[0]
    getitem_59: "f32[1, 128, 1]" = var_mean_29[1];  var_mean_29 = None
    add_119: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_29: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
    sub_44: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_118, getitem_59);  getitem_59 = None
    mul_116: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = rsqrt_29 = None
    mul_117: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_116, arg235_1);  mul_116 = arg235_1 = None
    add_120: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_117, arg236_1);  mul_117 = arg236_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_335: "f32[128, 1024]" = torch.ops.aten.view.default(add_120, [128, 1024]);  add_120 = None
    permute_163: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
    addmm_88: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg238_1, view_335, permute_163);  arg238_1 = view_335 = permute_163 = None
    view_336: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_88, [1, 128, 4096]);  addmm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_118: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_336, 0.5)
    mul_119: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_336, 0.7071067811865476);  view_336 = None
    erf_14: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
    add_121: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_120: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_118, add_121);  mul_118 = add_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_119: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_120);  mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_337: "f32[128, 4096]" = torch.ops.aten.view.default(clone_119, [128, 4096]);  clone_119 = None
    permute_164: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
    addmm_89: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg240_1, view_337, permute_164);  arg240_1 = view_337 = permute_164 = None
    view_338: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_89, [1, 128, 1024]);  addmm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_120: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_338);  view_338 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_122: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_118, clone_120);  add_118 = clone_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_30 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1, 128, 1]" = var_mean_30[0]
    getitem_61: "f32[1, 128, 1]" = var_mean_30[1];  var_mean_30 = None
    add_123: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_30: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    sub_45: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_122, getitem_61);  getitem_61 = None
    mul_121: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_45, rsqrt_30);  sub_45 = rsqrt_30 = None
    mul_122: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_121, arg241_1);  mul_121 = arg241_1 = None
    add_124: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_122, arg242_1);  mul_122 = arg242_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_339: "f32[128, 1024]" = torch.ops.aten.view.default(add_124, [128, 1024])
    permute_165: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
    addmm_90: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg244_1, view_339, permute_165);  arg244_1 = view_339 = permute_165 = None
    view_340: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_90, [1, 128, 1024]);  addmm_90 = None
    mul_123: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_340, 0.125);  view_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_341: "f32[128, 1024]" = torch.ops.aten.view.default(add_124, [128, 1024])
    permute_166: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
    addmm_91: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg246_1, view_341, permute_166);  arg246_1 = view_341 = permute_166 = None
    view_342: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_91, [1, 128, 1024]);  addmm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_343: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_342, [1, -1, 16, 64]);  view_342 = None
    permute_167: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_343, [0, 2, 1, 3]);  view_343 = None
    clone_121: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_344: "f32[128, 1024]" = torch.ops.aten.view.default(add_124, [128, 1024]);  add_124 = None
    permute_168: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
    addmm_92: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg248_1, view_344, permute_168);  arg248_1 = view_344 = permute_168 = None
    view_345: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_92, [1, 128, 1024]);  addmm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_346: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_345, [1, -1, 16, 64]);  view_345 = None
    permute_169: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_346, [0, 2, 1, 3]);  view_346 = None
    clone_122: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_347: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_123, [1, 128, 16, 64]);  mul_123 = None
    permute_170: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_347, [0, 2, 1, 3]);  view_347 = None
    clone_123: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_348: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_123, [16, -1, 64]);  clone_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_349: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_121, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_350: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_122, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_171: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
    bmm_30: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_348, permute_171);  view_348 = permute_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_351: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_30, [1, 16, 128, 128]);  bmm_30 = None
    add_125: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_351, expand_1);  view_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant15 = self._tensor_constant15
    lift_fresh_copy_15: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant15);  _tensor_constant15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_15: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_125, lift_fresh_copy_15);  add_125 = lift_fresh_copy_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_352: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_15, [16, 128, 128]);  maximum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_15: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_352, [-1], True)
    sub_46: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_352, amax_15);  view_352 = amax_15 = None
    exp_15: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_46);  sub_46 = None
    sum_16: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_124: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_15);  div_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_31: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_124, view_350);  clone_124 = view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_353: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_31, [1, 16, 128, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_172: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_125: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_172, memory_format = torch.contiguous_format);  permute_172 = None
    view_354: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_125, [1, 128, 1024]);  clone_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_355: "f32[128, 1024]" = torch.ops.aten.view.default(view_354, [128, 1024]);  view_354 = None
    permute_173: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
    addmm_93: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg250_1, view_355, permute_173);  arg250_1 = view_355 = permute_173 = None
    view_356: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_93, [1, 128, 1024]);  addmm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_126: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_356);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_126: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_122, clone_126);  add_122 = clone_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_31 = torch.ops.aten.var_mean.correction(add_126, [2], correction = 0, keepdim = True)
    getitem_62: "f32[1, 128, 1]" = var_mean_31[0]
    getitem_63: "f32[1, 128, 1]" = var_mean_31[1];  var_mean_31 = None
    add_127: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-05);  getitem_62 = None
    rsqrt_31: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_47: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_126, getitem_63);  getitem_63 = None
    mul_124: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = rsqrt_31 = None
    mul_125: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_124, arg251_1);  mul_124 = arg251_1 = None
    add_128: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_125, arg252_1);  mul_125 = arg252_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_357: "f32[128, 1024]" = torch.ops.aten.view.default(add_128, [128, 1024]);  add_128 = None
    permute_174: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
    addmm_94: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg254_1, view_357, permute_174);  arg254_1 = view_357 = permute_174 = None
    view_358: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_94, [1, 128, 4096]);  addmm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_126: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_358, 0.5)
    mul_127: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_358, 0.7071067811865476);  view_358 = None
    erf_15: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_127);  mul_127 = None
    add_129: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_128: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_126, add_129);  mul_126 = add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_127: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_128);  mul_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_359: "f32[128, 4096]" = torch.ops.aten.view.default(clone_127, [128, 4096]);  clone_127 = None
    permute_175: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
    addmm_95: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg256_1, view_359, permute_175);  arg256_1 = view_359 = permute_175 = None
    view_360: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_95, [1, 128, 1024]);  addmm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_128: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_360);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_130: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_126, clone_128);  add_126 = clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_32 = torch.ops.aten.var_mean.correction(add_130, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1, 128, 1]" = var_mean_32[0]
    getitem_65: "f32[1, 128, 1]" = var_mean_32[1];  var_mean_32 = None
    add_131: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_32: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    sub_48: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_130, getitem_65);  getitem_65 = None
    mul_129: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_32);  sub_48 = rsqrt_32 = None
    mul_130: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_129, arg257_1);  mul_129 = arg257_1 = None
    add_132: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_130, arg258_1);  mul_130 = arg258_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_361: "f32[128, 1024]" = torch.ops.aten.view.default(add_132, [128, 1024])
    permute_176: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
    addmm_96: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg260_1, view_361, permute_176);  arg260_1 = view_361 = permute_176 = None
    view_362: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_96, [1, 128, 1024]);  addmm_96 = None
    mul_131: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_362, 0.125);  view_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_363: "f32[128, 1024]" = torch.ops.aten.view.default(add_132, [128, 1024])
    permute_177: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
    addmm_97: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg262_1, view_363, permute_177);  arg262_1 = view_363 = permute_177 = None
    view_364: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_97, [1, 128, 1024]);  addmm_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_365: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_364, [1, -1, 16, 64]);  view_364 = None
    permute_178: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_365, [0, 2, 1, 3]);  view_365 = None
    clone_129: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_178, memory_format = torch.contiguous_format);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_366: "f32[128, 1024]" = torch.ops.aten.view.default(add_132, [128, 1024]);  add_132 = None
    permute_179: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
    addmm_98: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg264_1, view_366, permute_179);  arg264_1 = view_366 = permute_179 = None
    view_367: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_98, [1, 128, 1024]);  addmm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_368: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_367, [1, -1, 16, 64]);  view_367 = None
    permute_180: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_368, [0, 2, 1, 3]);  view_368 = None
    clone_130: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_369: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_131, [1, 128, 16, 64]);  mul_131 = None
    permute_181: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None
    clone_131: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_370: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_131, [16, -1, 64]);  clone_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_371: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_129, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_372: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_130, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_182: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_371, [0, 2, 1]);  view_371 = None
    bmm_32: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_370, permute_182);  view_370 = permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_373: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_32, [1, 16, 128, 128]);  bmm_32 = None
    add_133: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_373, expand_1);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant16 = self._tensor_constant16
    lift_fresh_copy_16: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant16);  _tensor_constant16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_16: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_133, lift_fresh_copy_16);  add_133 = lift_fresh_copy_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_374: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_16, [16, 128, 128]);  maximum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_16: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_374, [-1], True)
    sub_49: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_374, amax_16);  view_374 = amax_16 = None
    exp_16: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_17: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_132: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_16);  div_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_33: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_132, view_372);  clone_132 = view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_375: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_33, [1, 16, 128, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_183: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_375, [0, 2, 1, 3]);  view_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_133: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    view_376: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_133, [1, 128, 1024]);  clone_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_377: "f32[128, 1024]" = torch.ops.aten.view.default(view_376, [128, 1024]);  view_376 = None
    permute_184: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
    addmm_99: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg266_1, view_377, permute_184);  arg266_1 = view_377 = permute_184 = None
    view_378: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_99, [1, 128, 1024]);  addmm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_134: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_378);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_134: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_130, clone_134);  add_130 = clone_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_33 = torch.ops.aten.var_mean.correction(add_134, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1, 128, 1]" = var_mean_33[0]
    getitem_67: "f32[1, 128, 1]" = var_mean_33[1];  var_mean_33 = None
    add_135: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_33: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    sub_50: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_134, getitem_67);  getitem_67 = None
    mul_132: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = rsqrt_33 = None
    mul_133: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_132, arg267_1);  mul_132 = arg267_1 = None
    add_136: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_133, arg268_1);  mul_133 = arg268_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_379: "f32[128, 1024]" = torch.ops.aten.view.default(add_136, [128, 1024]);  add_136 = None
    permute_185: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
    addmm_100: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg270_1, view_379, permute_185);  arg270_1 = view_379 = permute_185 = None
    view_380: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_100, [1, 128, 4096]);  addmm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_134: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_380, 0.5)
    mul_135: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_380, 0.7071067811865476);  view_380 = None
    erf_16: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_135);  mul_135 = None
    add_137: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_136: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_134, add_137);  mul_134 = add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_135: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_136);  mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_381: "f32[128, 4096]" = torch.ops.aten.view.default(clone_135, [128, 4096]);  clone_135 = None
    permute_186: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
    addmm_101: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg272_1, view_381, permute_186);  arg272_1 = view_381 = permute_186 = None
    view_382: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_101, [1, 128, 1024]);  addmm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_136: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_382);  view_382 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_138: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_134, clone_136);  add_134 = clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_34 = torch.ops.aten.var_mean.correction(add_138, [2], correction = 0, keepdim = True)
    getitem_68: "f32[1, 128, 1]" = var_mean_34[0]
    getitem_69: "f32[1, 128, 1]" = var_mean_34[1];  var_mean_34 = None
    add_139: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_34: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    sub_51: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_138, getitem_69);  getitem_69 = None
    mul_137: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_34);  sub_51 = rsqrt_34 = None
    mul_138: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_137, arg273_1);  mul_137 = arg273_1 = None
    add_140: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_138, arg274_1);  mul_138 = arg274_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_383: "f32[128, 1024]" = torch.ops.aten.view.default(add_140, [128, 1024])
    permute_187: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
    addmm_102: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg276_1, view_383, permute_187);  arg276_1 = view_383 = permute_187 = None
    view_384: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_102, [1, 128, 1024]);  addmm_102 = None
    mul_139: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_384, 0.125);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_385: "f32[128, 1024]" = torch.ops.aten.view.default(add_140, [128, 1024])
    permute_188: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
    addmm_103: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg278_1, view_385, permute_188);  arg278_1 = view_385 = permute_188 = None
    view_386: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_103, [1, 128, 1024]);  addmm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_387: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_386, [1, -1, 16, 64]);  view_386 = None
    permute_189: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
    clone_137: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_388: "f32[128, 1024]" = torch.ops.aten.view.default(add_140, [128, 1024]);  add_140 = None
    permute_190: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
    addmm_104: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg280_1, view_388, permute_190);  arg280_1 = view_388 = permute_190 = None
    view_389: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_104, [1, 128, 1024]);  addmm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_390: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_389, [1, -1, 16, 64]);  view_389 = None
    permute_191: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_390, [0, 2, 1, 3]);  view_390 = None
    clone_138: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_191, memory_format = torch.contiguous_format);  permute_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_391: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_139, [1, 128, 16, 64]);  mul_139 = None
    permute_192: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None
    clone_139: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_192, memory_format = torch.contiguous_format);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_392: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_139, [16, -1, 64]);  clone_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_393: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_137, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_394: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_138, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_193: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_393, [0, 2, 1]);  view_393 = None
    bmm_34: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_392, permute_193);  view_392 = permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_395: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_34, [1, 16, 128, 128]);  bmm_34 = None
    add_141: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_395, expand_1);  view_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant17 = self._tensor_constant17
    lift_fresh_copy_17: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant17);  _tensor_constant17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_17: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_141, lift_fresh_copy_17);  add_141 = lift_fresh_copy_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_396: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_17, [16, 128, 128]);  maximum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_17: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_396, [-1], True)
    sub_52: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_396, amax_17);  view_396 = amax_17 = None
    exp_17: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_18: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_140: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_17);  div_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_35: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_140, view_394);  clone_140 = view_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_397: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_35, [1, 16, 128, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_194: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_141: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_194, memory_format = torch.contiguous_format);  permute_194 = None
    view_398: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_141, [1, 128, 1024]);  clone_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_399: "f32[128, 1024]" = torch.ops.aten.view.default(view_398, [128, 1024]);  view_398 = None
    permute_195: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
    addmm_105: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg282_1, view_399, permute_195);  arg282_1 = view_399 = permute_195 = None
    view_400: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_105, [1, 128, 1024]);  addmm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_142: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_400);  view_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_142: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_138, clone_142);  add_138 = clone_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_35 = torch.ops.aten.var_mean.correction(add_142, [2], correction = 0, keepdim = True)
    getitem_70: "f32[1, 128, 1]" = var_mean_35[0]
    getitem_71: "f32[1, 128, 1]" = var_mean_35[1];  var_mean_35 = None
    add_143: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
    rsqrt_35: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_143);  add_143 = None
    sub_53: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_142, getitem_71);  getitem_71 = None
    mul_140: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = rsqrt_35 = None
    mul_141: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_140, arg283_1);  mul_140 = arg283_1 = None
    add_144: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_141, arg284_1);  mul_141 = arg284_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_401: "f32[128, 1024]" = torch.ops.aten.view.default(add_144, [128, 1024]);  add_144 = None
    permute_196: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
    addmm_106: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg286_1, view_401, permute_196);  arg286_1 = view_401 = permute_196 = None
    view_402: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_106, [1, 128, 4096]);  addmm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_142: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_402, 0.5)
    mul_143: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_402, 0.7071067811865476);  view_402 = None
    erf_17: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_143);  mul_143 = None
    add_145: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_144: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_142, add_145);  mul_142 = add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_143: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_144);  mul_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_403: "f32[128, 4096]" = torch.ops.aten.view.default(clone_143, [128, 4096]);  clone_143 = None
    permute_197: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
    addmm_107: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg288_1, view_403, permute_197);  arg288_1 = view_403 = permute_197 = None
    view_404: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_107, [1, 128, 1024]);  addmm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_144: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_404);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_146: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_142, clone_144);  add_142 = clone_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_36 = torch.ops.aten.var_mean.correction(add_146, [2], correction = 0, keepdim = True)
    getitem_72: "f32[1, 128, 1]" = var_mean_36[0]
    getitem_73: "f32[1, 128, 1]" = var_mean_36[1];  var_mean_36 = None
    add_147: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_36: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_54: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_146, getitem_73);  getitem_73 = None
    mul_145: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_36);  sub_54 = rsqrt_36 = None
    mul_146: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_145, arg289_1);  mul_145 = arg289_1 = None
    add_148: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_146, arg290_1);  mul_146 = arg290_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_405: "f32[128, 1024]" = torch.ops.aten.view.default(add_148, [128, 1024])
    permute_198: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
    addmm_108: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg292_1, view_405, permute_198);  arg292_1 = view_405 = permute_198 = None
    view_406: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_108, [1, 128, 1024]);  addmm_108 = None
    mul_147: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_406, 0.125);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_407: "f32[128, 1024]" = torch.ops.aten.view.default(add_148, [128, 1024])
    permute_199: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
    addmm_109: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg294_1, view_407, permute_199);  arg294_1 = view_407 = permute_199 = None
    view_408: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_109, [1, 128, 1024]);  addmm_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_409: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_408, [1, -1, 16, 64]);  view_408 = None
    permute_200: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
    clone_145: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_200, memory_format = torch.contiguous_format);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_410: "f32[128, 1024]" = torch.ops.aten.view.default(add_148, [128, 1024]);  add_148 = None
    permute_201: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
    addmm_110: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg296_1, view_410, permute_201);  arg296_1 = view_410 = permute_201 = None
    view_411: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_110, [1, 128, 1024]);  addmm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_412: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_411, [1, -1, 16, 64]);  view_411 = None
    permute_202: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_412, [0, 2, 1, 3]);  view_412 = None
    clone_146: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_202, memory_format = torch.contiguous_format);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_413: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_147, [1, 128, 16, 64]);  mul_147 = None
    permute_203: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_413, [0, 2, 1, 3]);  view_413 = None
    clone_147: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_203, memory_format = torch.contiguous_format);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_414: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_147, [16, -1, 64]);  clone_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_415: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_145, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_416: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_146, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_204: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_415, [0, 2, 1]);  view_415 = None
    bmm_36: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_414, permute_204);  view_414 = permute_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_417: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_36, [1, 16, 128, 128]);  bmm_36 = None
    add_149: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_417, expand_1);  view_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant18 = self._tensor_constant18
    lift_fresh_copy_18: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant18);  _tensor_constant18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_18: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_149, lift_fresh_copy_18);  add_149 = lift_fresh_copy_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_418: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_18, [16, 128, 128]);  maximum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_18: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_418, [-1], True)
    sub_55: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_418, amax_18);  view_418 = amax_18 = None
    exp_18: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_55);  sub_55 = None
    sum_19: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_148: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_18);  div_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_37: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_148, view_416);  clone_148 = view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_419: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_37, [1, 16, 128, 64]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_205: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_419, [0, 2, 1, 3]);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_149: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_205, memory_format = torch.contiguous_format);  permute_205 = None
    view_420: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_149, [1, 128, 1024]);  clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_421: "f32[128, 1024]" = torch.ops.aten.view.default(view_420, [128, 1024]);  view_420 = None
    permute_206: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
    addmm_111: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg298_1, view_421, permute_206);  arg298_1 = view_421 = permute_206 = None
    view_422: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_111, [1, 128, 1024]);  addmm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_150: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_422);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_150: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_146, clone_150);  add_146 = clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_37 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1, 128, 1]" = var_mean_37[0]
    getitem_75: "f32[1, 128, 1]" = var_mean_37[1];  var_mean_37 = None
    add_151: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_37: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_56: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_150, getitem_75);  getitem_75 = None
    mul_148: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = rsqrt_37 = None
    mul_149: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_148, arg299_1);  mul_148 = arg299_1 = None
    add_152: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_149, arg300_1);  mul_149 = arg300_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_423: "f32[128, 1024]" = torch.ops.aten.view.default(add_152, [128, 1024]);  add_152 = None
    permute_207: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
    addmm_112: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg302_1, view_423, permute_207);  arg302_1 = view_423 = permute_207 = None
    view_424: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_112, [1, 128, 4096]);  addmm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_150: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_424, 0.5)
    mul_151: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_424, 0.7071067811865476);  view_424 = None
    erf_18: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_151);  mul_151 = None
    add_153: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_152: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_150, add_153);  mul_150 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_151: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_152);  mul_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_425: "f32[128, 4096]" = torch.ops.aten.view.default(clone_151, [128, 4096]);  clone_151 = None
    permute_208: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
    addmm_113: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg304_1, view_425, permute_208);  arg304_1 = view_425 = permute_208 = None
    view_426: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_113, [1, 128, 1024]);  addmm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_152: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_426);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_154: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_150, clone_152);  add_150 = clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_38 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
    getitem_76: "f32[1, 128, 1]" = var_mean_38[0]
    getitem_77: "f32[1, 128, 1]" = var_mean_38[1];  var_mean_38 = None
    add_155: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
    rsqrt_38: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
    sub_57: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_154, getitem_77);  getitem_77 = None
    mul_153: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_38);  sub_57 = rsqrt_38 = None
    mul_154: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_153, arg305_1);  mul_153 = arg305_1 = None
    add_156: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_154, arg306_1);  mul_154 = arg306_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_427: "f32[128, 1024]" = torch.ops.aten.view.default(add_156, [128, 1024])
    permute_209: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
    addmm_114: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg308_1, view_427, permute_209);  arg308_1 = view_427 = permute_209 = None
    view_428: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_114, [1, 128, 1024]);  addmm_114 = None
    mul_155: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_428, 0.125);  view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_429: "f32[128, 1024]" = torch.ops.aten.view.default(add_156, [128, 1024])
    permute_210: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
    addmm_115: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg310_1, view_429, permute_210);  arg310_1 = view_429 = permute_210 = None
    view_430: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_115, [1, 128, 1024]);  addmm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_431: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_430, [1, -1, 16, 64]);  view_430 = None
    permute_211: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_431, [0, 2, 1, 3]);  view_431 = None
    clone_153: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_211, memory_format = torch.contiguous_format);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_432: "f32[128, 1024]" = torch.ops.aten.view.default(add_156, [128, 1024]);  add_156 = None
    permute_212: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
    addmm_116: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg312_1, view_432, permute_212);  arg312_1 = view_432 = permute_212 = None
    view_433: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_116, [1, 128, 1024]);  addmm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_434: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_433, [1, -1, 16, 64]);  view_433 = None
    permute_213: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_434, [0, 2, 1, 3]);  view_434 = None
    clone_154: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_213, memory_format = torch.contiguous_format);  permute_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_435: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_155, [1, 128, 16, 64]);  mul_155 = None
    permute_214: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_435, [0, 2, 1, 3]);  view_435 = None
    clone_155: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_214, memory_format = torch.contiguous_format);  permute_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_436: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_155, [16, -1, 64]);  clone_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_437: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_153, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_438: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_154, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_215: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_437, [0, 2, 1]);  view_437 = None
    bmm_38: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_436, permute_215);  view_436 = permute_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_439: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_38, [1, 16, 128, 128]);  bmm_38 = None
    add_157: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_439, expand_1);  view_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant19 = self._tensor_constant19
    lift_fresh_copy_19: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant19);  _tensor_constant19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_19: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_157, lift_fresh_copy_19);  add_157 = lift_fresh_copy_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_440: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_19, [16, 128, 128]);  maximum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_19: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_440, [-1], True)
    sub_58: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_440, amax_19);  view_440 = amax_19 = None
    exp_19: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_58);  sub_58 = None
    sum_20: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_156: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_19);  div_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_39: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_156, view_438);  clone_156 = view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_441: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_39, [1, 16, 128, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_216: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_441, [0, 2, 1, 3]);  view_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_157: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    view_442: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_157, [1, 128, 1024]);  clone_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_443: "f32[128, 1024]" = torch.ops.aten.view.default(view_442, [128, 1024]);  view_442 = None
    permute_217: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
    addmm_117: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg314_1, view_443, permute_217);  arg314_1 = view_443 = permute_217 = None
    view_444: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_117, [1, 128, 1024]);  addmm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_158: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_444);  view_444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_158: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_154, clone_158);  add_154 = clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_39 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1, 128, 1]" = var_mean_39[0]
    getitem_79: "f32[1, 128, 1]" = var_mean_39[1];  var_mean_39 = None
    add_159: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_39: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    sub_59: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_158, getitem_79);  getitem_79 = None
    mul_156: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = rsqrt_39 = None
    mul_157: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_156, arg315_1);  mul_156 = arg315_1 = None
    add_160: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_157, arg316_1);  mul_157 = arg316_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_445: "f32[128, 1024]" = torch.ops.aten.view.default(add_160, [128, 1024]);  add_160 = None
    permute_218: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
    addmm_118: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg318_1, view_445, permute_218);  arg318_1 = view_445 = permute_218 = None
    view_446: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_118, [1, 128, 4096]);  addmm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_158: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_446, 0.5)
    mul_159: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_446, 0.7071067811865476);  view_446 = None
    erf_19: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_159);  mul_159 = None
    add_161: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_160: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_158, add_161);  mul_158 = add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_159: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_160);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_447: "f32[128, 4096]" = torch.ops.aten.view.default(clone_159, [128, 4096]);  clone_159 = None
    permute_219: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg319_1, [1, 0]);  arg319_1 = None
    addmm_119: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg320_1, view_447, permute_219);  arg320_1 = view_447 = permute_219 = None
    view_448: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_119, [1, 128, 1024]);  addmm_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_160: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_448);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_162: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_158, clone_160);  add_158 = clone_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_40 = torch.ops.aten.var_mean.correction(add_162, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1, 128, 1]" = var_mean_40[0]
    getitem_81: "f32[1, 128, 1]" = var_mean_40[1];  var_mean_40 = None
    add_163: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_40: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
    sub_60: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_162, getitem_81);  getitem_81 = None
    mul_161: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_40);  sub_60 = rsqrt_40 = None
    mul_162: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_161, arg321_1);  mul_161 = arg321_1 = None
    add_164: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_162, arg322_1);  mul_162 = arg322_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_449: "f32[128, 1024]" = torch.ops.aten.view.default(add_164, [128, 1024])
    permute_220: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
    addmm_120: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg324_1, view_449, permute_220);  arg324_1 = view_449 = permute_220 = None
    view_450: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_120, [1, 128, 1024]);  addmm_120 = None
    mul_163: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_450, 0.125);  view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_451: "f32[128, 1024]" = torch.ops.aten.view.default(add_164, [128, 1024])
    permute_221: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg325_1, [1, 0]);  arg325_1 = None
    addmm_121: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg326_1, view_451, permute_221);  arg326_1 = view_451 = permute_221 = None
    view_452: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_121, [1, 128, 1024]);  addmm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_453: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_452, [1, -1, 16, 64]);  view_452 = None
    permute_222: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
    clone_161: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_454: "f32[128, 1024]" = torch.ops.aten.view.default(add_164, [128, 1024]);  add_164 = None
    permute_223: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
    addmm_122: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg328_1, view_454, permute_223);  arg328_1 = view_454 = permute_223 = None
    view_455: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_122, [1, 128, 1024]);  addmm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_456: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_455, [1, -1, 16, 64]);  view_455 = None
    permute_224: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
    clone_162: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_224, memory_format = torch.contiguous_format);  permute_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_457: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_163, [1, 128, 16, 64]);  mul_163 = None
    permute_225: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_457, [0, 2, 1, 3]);  view_457 = None
    clone_163: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_225, memory_format = torch.contiguous_format);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_458: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_163, [16, -1, 64]);  clone_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_459: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_161, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_460: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_162, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_226: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_459, [0, 2, 1]);  view_459 = None
    bmm_40: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_458, permute_226);  view_458 = permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_461: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_40, [1, 16, 128, 128]);  bmm_40 = None
    add_165: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_461, expand_1);  view_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant20 = self._tensor_constant20
    lift_fresh_copy_20: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant20);  _tensor_constant20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_20: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_165, lift_fresh_copy_20);  add_165 = lift_fresh_copy_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_462: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_20, [16, 128, 128]);  maximum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_20: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_462, [-1], True)
    sub_61: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_462, amax_20);  view_462 = amax_20 = None
    exp_20: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
    sum_21: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_164: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_20);  div_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_41: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_164, view_460);  clone_164 = view_460 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_463: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_41, [1, 16, 128, 64]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_227: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_463, [0, 2, 1, 3]);  view_463 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_165: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None
    view_464: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_165, [1, 128, 1024]);  clone_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_465: "f32[128, 1024]" = torch.ops.aten.view.default(view_464, [128, 1024]);  view_464 = None
    permute_228: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
    addmm_123: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg330_1, view_465, permute_228);  arg330_1 = view_465 = permute_228 = None
    view_466: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_123, [1, 128, 1024]);  addmm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_166: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_466);  view_466 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_166: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_162, clone_166);  add_162 = clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_41 = torch.ops.aten.var_mean.correction(add_166, [2], correction = 0, keepdim = True)
    getitem_82: "f32[1, 128, 1]" = var_mean_41[0]
    getitem_83: "f32[1, 128, 1]" = var_mean_41[1];  var_mean_41 = None
    add_167: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_41: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    sub_62: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_166, getitem_83);  getitem_83 = None
    mul_164: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = rsqrt_41 = None
    mul_165: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_164, arg331_1);  mul_164 = arg331_1 = None
    add_168: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_165, arg332_1);  mul_165 = arg332_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_467: "f32[128, 1024]" = torch.ops.aten.view.default(add_168, [128, 1024]);  add_168 = None
    permute_229: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
    addmm_124: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg334_1, view_467, permute_229);  arg334_1 = view_467 = permute_229 = None
    view_468: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_124, [1, 128, 4096]);  addmm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_166: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_468, 0.5)
    mul_167: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_468, 0.7071067811865476);  view_468 = None
    erf_20: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_169: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_168: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_166, add_169);  mul_166 = add_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_167: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_168);  mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_469: "f32[128, 4096]" = torch.ops.aten.view.default(clone_167, [128, 4096]);  clone_167 = None
    permute_230: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
    addmm_125: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg336_1, view_469, permute_230);  arg336_1 = view_469 = permute_230 = None
    view_470: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_125, [1, 128, 1024]);  addmm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_168: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_470);  view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_170: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_166, clone_168);  add_166 = clone_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_42 = torch.ops.aten.var_mean.correction(add_170, [2], correction = 0, keepdim = True)
    getitem_84: "f32[1, 128, 1]" = var_mean_42[0]
    getitem_85: "f32[1, 128, 1]" = var_mean_42[1];  var_mean_42 = None
    add_171: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-05);  getitem_84 = None
    rsqrt_42: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    sub_63: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_170, getitem_85);  getitem_85 = None
    mul_169: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_42);  sub_63 = rsqrt_42 = None
    mul_170: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_169, arg337_1);  mul_169 = arg337_1 = None
    add_172: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_170, arg338_1);  mul_170 = arg338_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_471: "f32[128, 1024]" = torch.ops.aten.view.default(add_172, [128, 1024])
    permute_231: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
    addmm_126: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg340_1, view_471, permute_231);  arg340_1 = view_471 = permute_231 = None
    view_472: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_126, [1, 128, 1024]);  addmm_126 = None
    mul_171: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_472, 0.125);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_473: "f32[128, 1024]" = torch.ops.aten.view.default(add_172, [128, 1024])
    permute_232: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
    addmm_127: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg342_1, view_473, permute_232);  arg342_1 = view_473 = permute_232 = None
    view_474: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_127, [1, 128, 1024]);  addmm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_475: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_474, [1, -1, 16, 64]);  view_474 = None
    permute_233: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_475, [0, 2, 1, 3]);  view_475 = None
    clone_169: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_476: "f32[128, 1024]" = torch.ops.aten.view.default(add_172, [128, 1024]);  add_172 = None
    permute_234: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
    addmm_128: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg344_1, view_476, permute_234);  arg344_1 = view_476 = permute_234 = None
    view_477: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_128, [1, 128, 1024]);  addmm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_478: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_477, [1, -1, 16, 64]);  view_477 = None
    permute_235: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
    clone_170: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_479: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_171, [1, 128, 16, 64]);  mul_171 = None
    permute_236: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_479, [0, 2, 1, 3]);  view_479 = None
    clone_171: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_236, memory_format = torch.contiguous_format);  permute_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_480: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_171, [16, -1, 64]);  clone_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_481: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_169, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_482: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_170, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_237: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_481, [0, 2, 1]);  view_481 = None
    bmm_42: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_480, permute_237);  view_480 = permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_483: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_42, [1, 16, 128, 128]);  bmm_42 = None
    add_173: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_483, expand_1);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant21 = self._tensor_constant21
    lift_fresh_copy_21: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant21);  _tensor_constant21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_21: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_173, lift_fresh_copy_21);  add_173 = lift_fresh_copy_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_484: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_21, [16, 128, 128]);  maximum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_21: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_484, [-1], True)
    sub_64: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_484, amax_21);  view_484 = amax_21 = None
    exp_21: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_64);  sub_64 = None
    sum_22: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_21: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_172: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_21);  div_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_43: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_172, view_482);  clone_172 = view_482 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_485: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_43, [1, 16, 128, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_238: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_485, [0, 2, 1, 3]);  view_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_173: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_238, memory_format = torch.contiguous_format);  permute_238 = None
    view_486: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_173, [1, 128, 1024]);  clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_487: "f32[128, 1024]" = torch.ops.aten.view.default(view_486, [128, 1024]);  view_486 = None
    permute_239: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
    addmm_129: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg346_1, view_487, permute_239);  arg346_1 = view_487 = permute_239 = None
    view_488: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_129, [1, 128, 1024]);  addmm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_174: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_488);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_174: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_170, clone_174);  add_170 = clone_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_43 = torch.ops.aten.var_mean.correction(add_174, [2], correction = 0, keepdim = True)
    getitem_86: "f32[1, 128, 1]" = var_mean_43[0]
    getitem_87: "f32[1, 128, 1]" = var_mean_43[1];  var_mean_43 = None
    add_175: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_43: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_65: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_174, getitem_87);  getitem_87 = None
    mul_172: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = rsqrt_43 = None
    mul_173: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_172, arg347_1);  mul_172 = arg347_1 = None
    add_176: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_173, arg348_1);  mul_173 = arg348_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_489: "f32[128, 1024]" = torch.ops.aten.view.default(add_176, [128, 1024]);  add_176 = None
    permute_240: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
    addmm_130: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg350_1, view_489, permute_240);  arg350_1 = view_489 = permute_240 = None
    view_490: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_130, [1, 128, 4096]);  addmm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_174: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_490, 0.5)
    mul_175: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_490, 0.7071067811865476);  view_490 = None
    erf_21: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_175);  mul_175 = None
    add_177: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_176: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_174, add_177);  mul_174 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_175: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_176);  mul_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_491: "f32[128, 4096]" = torch.ops.aten.view.default(clone_175, [128, 4096]);  clone_175 = None
    permute_241: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
    addmm_131: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg352_1, view_491, permute_241);  arg352_1 = view_491 = permute_241 = None
    view_492: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_131, [1, 128, 1024]);  addmm_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_176: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_492);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_178: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_174, clone_176);  add_174 = clone_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_44 = torch.ops.aten.var_mean.correction(add_178, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1, 128, 1]" = var_mean_44[0]
    getitem_89: "f32[1, 128, 1]" = var_mean_44[1];  var_mean_44 = None
    add_179: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_44: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
    sub_66: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_178, getitem_89);  getitem_89 = None
    mul_177: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_44);  sub_66 = rsqrt_44 = None
    mul_178: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_177, arg353_1);  mul_177 = arg353_1 = None
    add_180: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_178, arg354_1);  mul_178 = arg354_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_493: "f32[128, 1024]" = torch.ops.aten.view.default(add_180, [128, 1024])
    permute_242: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
    addmm_132: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg356_1, view_493, permute_242);  arg356_1 = view_493 = permute_242 = None
    view_494: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_132, [1, 128, 1024]);  addmm_132 = None
    mul_179: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_494, 0.125);  view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_495: "f32[128, 1024]" = torch.ops.aten.view.default(add_180, [128, 1024])
    permute_243: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
    addmm_133: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg358_1, view_495, permute_243);  arg358_1 = view_495 = permute_243 = None
    view_496: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_133, [1, 128, 1024]);  addmm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_497: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_496, [1, -1, 16, 64]);  view_496 = None
    permute_244: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_497, [0, 2, 1, 3]);  view_497 = None
    clone_177: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_244, memory_format = torch.contiguous_format);  permute_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_498: "f32[128, 1024]" = torch.ops.aten.view.default(add_180, [128, 1024]);  add_180 = None
    permute_245: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
    addmm_134: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg360_1, view_498, permute_245);  arg360_1 = view_498 = permute_245 = None
    view_499: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_134, [1, 128, 1024]);  addmm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_500: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_499, [1, -1, 16, 64]);  view_499 = None
    permute_246: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None
    clone_178: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_501: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_179, [1, 128, 16, 64]);  mul_179 = None
    permute_247: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_501, [0, 2, 1, 3]);  view_501 = None
    clone_179: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_502: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_179, [16, -1, 64]);  clone_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_503: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_177, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_504: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_178, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_248: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_503, [0, 2, 1]);  view_503 = None
    bmm_44: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_502, permute_248);  view_502 = permute_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_505: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_44, [1, 16, 128, 128]);  bmm_44 = None
    add_181: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_505, expand_1);  view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant22 = self._tensor_constant22
    lift_fresh_copy_22: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant22);  _tensor_constant22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_22: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_181, lift_fresh_copy_22);  add_181 = lift_fresh_copy_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_506: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_22, [16, 128, 128]);  maximum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_22: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_506, [-1], True)
    sub_67: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_506, amax_22);  view_506 = amax_22 = None
    exp_22: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_67);  sub_67 = None
    sum_23: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_180: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_22);  div_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_45: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_180, view_504);  clone_180 = view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_507: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_45, [1, 16, 128, 64]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_249: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_507, [0, 2, 1, 3]);  view_507 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_181: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    view_508: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_181, [1, 128, 1024]);  clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_509: "f32[128, 1024]" = torch.ops.aten.view.default(view_508, [128, 1024]);  view_508 = None
    permute_250: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
    addmm_135: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg362_1, view_509, permute_250);  arg362_1 = view_509 = permute_250 = None
    view_510: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_135, [1, 128, 1024]);  addmm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_182: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_510);  view_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_182: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_178, clone_182);  add_178 = clone_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_45 = torch.ops.aten.var_mean.correction(add_182, [2], correction = 0, keepdim = True)
    getitem_90: "f32[1, 128, 1]" = var_mean_45[0]
    getitem_91: "f32[1, 128, 1]" = var_mean_45[1];  var_mean_45 = None
    add_183: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
    rsqrt_45: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    sub_68: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_182, getitem_91);  getitem_91 = None
    mul_180: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = rsqrt_45 = None
    mul_181: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_180, arg363_1);  mul_180 = arg363_1 = None
    add_184: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_181, arg364_1);  mul_181 = arg364_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_511: "f32[128, 1024]" = torch.ops.aten.view.default(add_184, [128, 1024]);  add_184 = None
    permute_251: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg365_1, [1, 0]);  arg365_1 = None
    addmm_136: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg366_1, view_511, permute_251);  arg366_1 = view_511 = permute_251 = None
    view_512: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_136, [1, 128, 4096]);  addmm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_182: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_512, 0.5)
    mul_183: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_512, 0.7071067811865476);  view_512 = None
    erf_22: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_183);  mul_183 = None
    add_185: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_184: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_182, add_185);  mul_182 = add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_183: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_184);  mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_513: "f32[128, 4096]" = torch.ops.aten.view.default(clone_183, [128, 4096]);  clone_183 = None
    permute_252: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
    addmm_137: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg368_1, view_513, permute_252);  arg368_1 = view_513 = permute_252 = None
    view_514: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_137, [1, 128, 1024]);  addmm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_184: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_514);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_186: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_182, clone_184);  add_182 = clone_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:430, code: hidden_states = self.self_attn_layer_norm(hidden_states)
    var_mean_46 = torch.ops.aten.var_mean.correction(add_186, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1, 128, 1]" = var_mean_46[0]
    getitem_93: "f32[1, 128, 1]" = var_mean_46[1];  var_mean_46 = None
    add_187: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_46: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    sub_69: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_186, getitem_93);  getitem_93 = None
    mul_185: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_46);  sub_69 = rsqrt_46 = None
    mul_186: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_185, arg369_1);  mul_185 = arg369_1 = None
    add_188: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_186, arg370_1);  mul_186 = arg370_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:266, code: query_states = self.q_proj(hidden_states) * self.scaling
    view_515: "f32[128, 1024]" = torch.ops.aten.view.default(add_188, [128, 1024])
    permute_253: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg371_1, [1, 0]);  arg371_1 = None
    addmm_138: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg372_1, view_515, permute_253);  arg372_1 = view_515 = permute_253 = None
    view_516: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_138, [1, 128, 1024]);  addmm_138 = None
    mul_187: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_516, 0.125);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:284, code: key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
    view_517: "f32[128, 1024]" = torch.ops.aten.view.default(add_188, [128, 1024])
    permute_254: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
    addmm_139: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg374_1, view_517, permute_254);  arg374_1 = view_517 = permute_254 = None
    view_518: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_139, [1, 128, 1024]);  addmm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_519: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_518, [1, -1, 16, 64]);  view_518 = None
    permute_255: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_519, [0, 2, 1, 3]);  view_519 = None
    clone_185: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_255, memory_format = torch.contiguous_format);  permute_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:285, code: value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
    view_520: "f32[128, 1024]" = torch.ops.aten.view.default(add_188, [128, 1024]);  add_188 = None
    permute_256: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg375_1, [1, 0]);  arg375_1 = None
    addmm_140: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg376_1, view_520, permute_256);  arg376_1 = view_520 = permute_256 = None
    view_521: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_140, [1, 128, 1024]);  addmm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_522: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(view_521, [1, -1, 16, 64]);  view_521 = None
    permute_257: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
    clone_186: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_257, memory_format = torch.contiguous_format);  permute_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:246, code: return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    view_523: "f32[1, 128, 16, 64]" = torch.ops.aten.view.default(mul_187, [1, 128, 16, 64]);  mul_187 = None
    permute_258: "f32[1, 16, 128, 64]" = torch.ops.aten.permute.default(view_523, [0, 2, 1, 3]);  view_523 = None
    clone_187: "f32[1, 16, 128, 64]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:298, code: query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
    view_524: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_187, [16, -1, 64]);  clone_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:299, code: key_states = key_states.view(*proj_shape)
    view_525: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_185, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:300, code: value_states = value_states.view(*proj_shape)
    view_526: "f32[16, 128, 64]" = torch.ops.aten.view.default(clone_186, [16, -1, 64])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:303, code: attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
    permute_259: "f32[16, 64, 128]" = torch.ops.aten.permute.default(view_525, [0, 2, 1]);  view_525 = None
    bmm_46: "f32[16, 128, 128]" = torch.ops.aten.bmm.default(view_524, permute_259);  view_524 = permute_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:316, code: attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
    view_527: "f32[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm_46, [1, 16, 128, 128]);  bmm_46 = None
    add_189: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(view_527, expand_1);  view_527 = expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:318, code: attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
    _tensor_constant23 = self._tensor_constant23
    lift_fresh_copy_23: "f32[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant23);  _tensor_constant23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:317, code: attn_weights = torch.max(
    maximum_23: "f32[1, 16, 128, 128]" = torch.ops.aten.maximum.default(add_189, lift_fresh_copy_23);  add_189 = lift_fresh_copy_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:320, code: attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
    view_528: "f32[16, 128, 128]" = torch.ops.aten.view.default(maximum_23, [16, 128, 128]);  maximum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:326, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    amax_23: "f32[16, 128, 1]" = torch.ops.aten.amax.default(view_528, [-1], True)
    sub_70: "f32[16, 128, 128]" = torch.ops.aten.sub.Tensor(view_528, amax_23);  view_528 = amax_23 = None
    exp_23: "f32[16, 128, 128]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
    sum_24: "f32[16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_23: "f32[16, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:347, code: attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    clone_188: "f32[16, 128, 128]" = torch.ops.aten.clone.default(div_23);  div_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:349, code: attn_output = torch.bmm(attn_probs, value_states)
    bmm_47: "f32[16, 128, 64]" = torch.ops.aten.bmm.default(clone_188, view_526);  clone_188 = view_526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:357, code: attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
    view_529: "f32[1, 16, 128, 64]" = torch.ops.aten.view.default(bmm_47, [1, 16, 128, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:358, code: attn_output = attn_output.transpose(1, 2)
    permute_260: "f32[1, 128, 16, 64]" = torch.ops.aten.permute.default(view_529, [0, 2, 1, 3]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:362, code: attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)
    clone_189: "f32[1, 128, 16, 64]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_530: "f32[1, 128, 1024]" = torch.ops.aten.view.default(clone_189, [1, 128, 1024]);  clone_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:364, code: attn_output = self.out_proj(attn_output)
    view_531: "f32[128, 1024]" = torch.ops.aten.view.default(view_530, [128, 1024]);  view_530 = None
    permute_261: "f32[1024, 1024]" = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
    addmm_141: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg378_1, view_531, permute_261);  arg378_1 = view_531 = permute_261 = None
    view_532: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_141, [1, 128, 1024]);  addmm_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:443, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_190: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_532);  view_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:444, code: hidden_states = residual + hidden_states
    add_190: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_186, clone_190);  add_186 = clone_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:471, code: hidden_states = self.final_layer_norm(hidden_states)
    var_mean_47 = torch.ops.aten.var_mean.correction(add_190, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1, 128, 1]" = var_mean_47[0]
    getitem_95: "f32[1, 128, 1]" = var_mean_47[1];  var_mean_47 = None
    add_191: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_47: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    sub_71: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_190, getitem_95);  getitem_95 = None
    mul_188: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = rsqrt_47 = None
    mul_189: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_188, arg379_1);  mul_188 = arg379_1 = None
    add_192: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_189, arg380_1);  mul_189 = arg380_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:472, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
    view_533: "f32[128, 1024]" = torch.ops.aten.view.default(add_192, [128, 1024]);  add_192 = None
    permute_262: "f32[1024, 4096]" = torch.ops.aten.permute.default(arg381_1, [1, 0]);  arg381_1 = None
    addmm_142: "f32[128, 4096]" = torch.ops.aten.addmm.default(arg382_1, view_533, permute_262);  arg382_1 = view_533 = permute_262 = None
    view_534: "f32[1, 128, 4096]" = torch.ops.aten.view.default(addmm_142, [1, 128, 4096]);  addmm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_190: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_534, 0.5)
    mul_191: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_534, 0.7071067811865476);  view_534 = None
    erf_23: "f32[1, 128, 4096]" = torch.ops.aten.erf.default(mul_191);  mul_191 = None
    add_193: "f32[1, 128, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_192: "f32[1, 128, 4096]" = torch.ops.aten.mul.Tensor(mul_190, add_193);  mul_190 = add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:473, code: hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
    clone_191: "f32[1, 128, 4096]" = torch.ops.aten.clone.default(mul_192);  mul_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:474, code: hidden_states = self.fc2(hidden_states)
    view_535: "f32[128, 4096]" = torch.ops.aten.view.default(clone_191, [128, 4096]);  clone_191 = None
    permute_263: "f32[4096, 1024]" = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
    addmm_143: "f32[128, 1024]" = torch.ops.aten.addmm.default(arg384_1, view_535, permute_263);  arg384_1 = view_535 = permute_263 = None
    view_536: "f32[1, 128, 1024]" = torch.ops.aten.view.default(addmm_143, [1, 128, 1024]);  addmm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:475, code: hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
    clone_192: "f32[1, 128, 1024]" = torch.ops.aten.clone.default(view_536);  view_536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:476, code: hidden_states = residual + hidden_states
    add_194: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(add_190, clone_192);  add_190 = clone_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:722, code: hidden_states = self.layer_norm(hidden_states)
    var_mean_48 = torch.ops.aten.var_mean.correction(add_194, [2], correction = 0, keepdim = True)
    getitem_96: "f32[1, 128, 1]" = var_mean_48[0]
    getitem_97: "f32[1, 128, 1]" = var_mean_48[1];  var_mean_48 = None
    add_195: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_48: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
    sub_72: "f32[1, 128, 1024]" = torch.ops.aten.sub.Tensor(add_194, getitem_97);  add_194 = getitem_97 = None
    mul_193: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_48);  sub_72 = rsqrt_48 = None
    mul_194: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_193, arg385_1);  mul_193 = arg385_1 = None
    add_196: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(mul_194, arg386_1);  mul_194 = arg386_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:828, code: logits = self.lm_head(outputs[0])
    permute_264: "f32[1024, 256008]" = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
    view_537: "f32[128, 1024]" = torch.ops.aten.view.default(add_196, [128, 1024]);  add_196 = None
    mm: "f32[128, 256008]" = torch.ops.aten.mm.default(view_537, permute_264);  view_537 = permute_264 = None
    view_538: "f32[1, 128, 256008]" = torch.ops.aten.view.default(mm, [1, 128, 256008]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:833, code: shift_labels = labels.new_zeros(labels.shape)
    full_1: "i64[1, 128]" = torch.ops.aten.full.default([1, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:834, code: shift_labels[:, :-1] = labels[:, 1:].clone()
    slice_5: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg390_1, 0, 0, 9223372036854775807);  arg390_1 = None
    slice_6: "i64[1, 127]" = torch.ops.aten.slice.Tensor(slice_5, 1, 1, 9223372036854775807);  slice_5 = None
    clone_193: "i64[1, 127]" = torch.ops.aten.clone.default(slice_6);  slice_6 = None
    slice_7: "i64[1, 128]" = torch.ops.aten.slice.Tensor(full_1, 0, 0, 9223372036854775807)
    slice_8: "i64[1, 127]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, -1);  slice_7 = None
    copy: "i64[1, 127]" = torch.ops.aten.copy.default(slice_8, clone_193);  slice_8 = clone_193 = None
    slice_9: "i64[1, 128]" = torch.ops.aten.slice.Tensor(full_1, 0, 0, 9223372036854775807)
    slice_scatter: "i64[1, 128]" = torch.ops.aten.slice_scatter.default(slice_9, copy, 1, 0, -1);  slice_9 = copy = None
    slice_scatter_1: "i64[1, 128]" = torch.ops.aten.slice_scatter.default(full_1, slice_scatter, 0, 0, 9223372036854775807);  full_1 = slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:835, code: shift_labels[:, -1] = self.config.pad_token_id
    _tensor_constant24 = self._tensor_constant24
    lift_fresh_copy_24: "i64[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant24);  _tensor_constant24 = None
    slice_13: "i64[1, 128]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0, 9223372036854775807)
    select_1: "i64[1]" = torch.ops.aten.select.int(slice_13, 1, -1);  slice_13 = None
    copy_1: "i64[1]" = torch.ops.aten.copy.default(select_1, lift_fresh_copy_24);  select_1 = lift_fresh_copy_24 = None
    slice_14: "i64[1, 128]" = torch.ops.aten.slice.Tensor(slice_scatter_1, 0, 0, 9223372036854775807)
    select_scatter: "i64[1, 128]" = torch.ops.aten.select_scatter.default(slice_14, copy_1, 1, -1);  slice_14 = copy_1 = None
    slice_scatter_2: "i64[1, 128]" = torch.ops.aten.slice_scatter.default(slice_scatter_1, select_scatter, 0, 0, 9223372036854775807);  slice_scatter_1 = select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xglm/modeling_xglm.py:838, code: loss = loss_fct(logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
    view_539: "f32[128, 256008]" = torch.ops.aten.view.default(view_538, [-1, 256008])
    amax_24: "f32[128, 1]" = torch.ops.aten.amax.default(view_539, [1], True)
    sub_73: "f32[128, 256008]" = torch.ops.aten.sub.Tensor(view_539, amax_24);  view_539 = amax_24 = None
    exp_24: "f32[128, 256008]" = torch.ops.aten.exp.default(sub_73)
    sum_25: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log: "f32[128, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_74: "f32[128, 256008]" = torch.ops.aten.sub.Tensor(sub_73, log);  sub_73 = log = None
    view_541: "i64[128]" = torch.ops.aten.view.default(slice_scatter_2, [-1]);  slice_scatter_2 = None
    ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_541, -100)
    scalar_tensor_1: "i64[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "i64[128]" = torch.ops.aten.where.self(ne, view_541, scalar_tensor_1);  ne = scalar_tensor_1 = None
    unsqueeze_8: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_1, 1);  where_1 = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_74, 1, unsqueeze_8);  sub_74 = unsqueeze_8 = None
    squeeze_1: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg: "f32[128]" = torch.ops.aten.neg.default(squeeze_1);  squeeze_1 = None
    ne_1: "b8[128]" = torch.ops.aten.ne.Scalar(view_541, -100)
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[128]" = torch.ops.aten.where.self(ne_1, neg, scalar_tensor_2);  ne_1 = neg = scalar_tensor_2 = None
    ne_2: "b8[128]" = torch.ops.aten.ne.Scalar(view_541, -100);  view_541 = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
    convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_2);  where_2 = None
    div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type);  sum_27 = convert_element_type = None
    return (div_24, view_538, clone_1, clone_2, clone_9, clone_10, clone_17, clone_18, clone_25, clone_26, clone_33, clone_34, clone_41, clone_42, clone_49, clone_50, clone_57, clone_58, clone_65, clone_66, clone_73, clone_74, clone_81, clone_82, clone_89, clone_90, clone_97, clone_98, clone_105, clone_106, clone_113, clone_114, clone_121, clone_122, clone_129, clone_130, clone_137, clone_138, clone_145, clone_146, clone_153, clone_154, clone_161, clone_162, clone_169, clone_170, clone_177, clone_178, clone_185, clone_186)
    