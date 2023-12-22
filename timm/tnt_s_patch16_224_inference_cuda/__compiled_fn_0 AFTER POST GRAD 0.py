from __future__ import annotations



def forward(self, arg0_1: "f32[1, 24, 4, 4]", arg1_1: "f32[1, 1, 384]", arg2_1: "f32[1, 197, 384]", arg3_1: "f32[24, 3, 7, 7]", arg4_1: "f32[24]", arg5_1: "f32[384]", arg6_1: "f32[384]", arg7_1: "f32[384, 384]", arg8_1: "f32[384]", arg9_1: "f32[384]", arg10_1: "f32[384]", arg11_1: "f32[24]", arg12_1: "f32[24]", arg13_1: "f32[48, 24]", arg14_1: "f32[24, 24]", arg15_1: "f32[24, 24]", arg16_1: "f32[24]", arg17_1: "f32[24]", arg18_1: "f32[24]", arg19_1: "f32[96, 24]", arg20_1: "f32[96]", arg21_1: "f32[24, 96]", arg22_1: "f32[24]", arg23_1: "f32[24]", arg24_1: "f32[24]", arg25_1: "f32[384, 384]", arg26_1: "f32[384]", arg27_1: "f32[384]", arg28_1: "f32[384]", arg29_1: "f32[768, 384]", arg30_1: "f32[384, 384]", arg31_1: "f32[384, 384]", arg32_1: "f32[384]", arg33_1: "f32[384]", arg34_1: "f32[384]", arg35_1: "f32[1536, 384]", arg36_1: "f32[1536]", arg37_1: "f32[384, 1536]", arg38_1: "f32[384]", arg39_1: "f32[24]", arg40_1: "f32[24]", arg41_1: "f32[48, 24]", arg42_1: "f32[24, 24]", arg43_1: "f32[24, 24]", arg44_1: "f32[24]", arg45_1: "f32[24]", arg46_1: "f32[24]", arg47_1: "f32[96, 24]", arg48_1: "f32[96]", arg49_1: "f32[24, 96]", arg50_1: "f32[24]", arg51_1: "f32[24]", arg52_1: "f32[24]", arg53_1: "f32[384, 384]", arg54_1: "f32[384]", arg55_1: "f32[384]", arg56_1: "f32[384]", arg57_1: "f32[768, 384]", arg58_1: "f32[384, 384]", arg59_1: "f32[384, 384]", arg60_1: "f32[384]", arg61_1: "f32[384]", arg62_1: "f32[384]", arg63_1: "f32[1536, 384]", arg64_1: "f32[1536]", arg65_1: "f32[384, 1536]", arg66_1: "f32[384]", arg67_1: "f32[24]", arg68_1: "f32[24]", arg69_1: "f32[48, 24]", arg70_1: "f32[24, 24]", arg71_1: "f32[24, 24]", arg72_1: "f32[24]", arg73_1: "f32[24]", arg74_1: "f32[24]", arg75_1: "f32[96, 24]", arg76_1: "f32[96]", arg77_1: "f32[24, 96]", arg78_1: "f32[24]", arg79_1: "f32[24]", arg80_1: "f32[24]", arg81_1: "f32[384, 384]", arg82_1: "f32[384]", arg83_1: "f32[384]", arg84_1: "f32[384]", arg85_1: "f32[768, 384]", arg86_1: "f32[384, 384]", arg87_1: "f32[384, 384]", arg88_1: "f32[384]", arg89_1: "f32[384]", arg90_1: "f32[384]", arg91_1: "f32[1536, 384]", arg92_1: "f32[1536]", arg93_1: "f32[384, 1536]", arg94_1: "f32[384]", arg95_1: "f32[24]", arg96_1: "f32[24]", arg97_1: "f32[48, 24]", arg98_1: "f32[24, 24]", arg99_1: "f32[24, 24]", arg100_1: "f32[24]", arg101_1: "f32[24]", arg102_1: "f32[24]", arg103_1: "f32[96, 24]", arg104_1: "f32[96]", arg105_1: "f32[24, 96]", arg106_1: "f32[24]", arg107_1: "f32[24]", arg108_1: "f32[24]", arg109_1: "f32[384, 384]", arg110_1: "f32[384]", arg111_1: "f32[384]", arg112_1: "f32[384]", arg113_1: "f32[768, 384]", arg114_1: "f32[384, 384]", arg115_1: "f32[384, 384]", arg116_1: "f32[384]", arg117_1: "f32[384]", arg118_1: "f32[384]", arg119_1: "f32[1536, 384]", arg120_1: "f32[1536]", arg121_1: "f32[384, 1536]", arg122_1: "f32[384]", arg123_1: "f32[24]", arg124_1: "f32[24]", arg125_1: "f32[48, 24]", arg126_1: "f32[24, 24]", arg127_1: "f32[24, 24]", arg128_1: "f32[24]", arg129_1: "f32[24]", arg130_1: "f32[24]", arg131_1: "f32[96, 24]", arg132_1: "f32[96]", arg133_1: "f32[24, 96]", arg134_1: "f32[24]", arg135_1: "f32[24]", arg136_1: "f32[24]", arg137_1: "f32[384, 384]", arg138_1: "f32[384]", arg139_1: "f32[384]", arg140_1: "f32[384]", arg141_1: "f32[768, 384]", arg142_1: "f32[384, 384]", arg143_1: "f32[384, 384]", arg144_1: "f32[384]", arg145_1: "f32[384]", arg146_1: "f32[384]", arg147_1: "f32[1536, 384]", arg148_1: "f32[1536]", arg149_1: "f32[384, 1536]", arg150_1: "f32[384]", arg151_1: "f32[24]", arg152_1: "f32[24]", arg153_1: "f32[48, 24]", arg154_1: "f32[24, 24]", arg155_1: "f32[24, 24]", arg156_1: "f32[24]", arg157_1: "f32[24]", arg158_1: "f32[24]", arg159_1: "f32[96, 24]", arg160_1: "f32[96]", arg161_1: "f32[24, 96]", arg162_1: "f32[24]", arg163_1: "f32[24]", arg164_1: "f32[24]", arg165_1: "f32[384, 384]", arg166_1: "f32[384]", arg167_1: "f32[384]", arg168_1: "f32[384]", arg169_1: "f32[768, 384]", arg170_1: "f32[384, 384]", arg171_1: "f32[384, 384]", arg172_1: "f32[384]", arg173_1: "f32[384]", arg174_1: "f32[384]", arg175_1: "f32[1536, 384]", arg176_1: "f32[1536]", arg177_1: "f32[384, 1536]", arg178_1: "f32[384]", arg179_1: "f32[24]", arg180_1: "f32[24]", arg181_1: "f32[48, 24]", arg182_1: "f32[24, 24]", arg183_1: "f32[24, 24]", arg184_1: "f32[24]", arg185_1: "f32[24]", arg186_1: "f32[24]", arg187_1: "f32[96, 24]", arg188_1: "f32[96]", arg189_1: "f32[24, 96]", arg190_1: "f32[24]", arg191_1: "f32[24]", arg192_1: "f32[24]", arg193_1: "f32[384, 384]", arg194_1: "f32[384]", arg195_1: "f32[384]", arg196_1: "f32[384]", arg197_1: "f32[768, 384]", arg198_1: "f32[384, 384]", arg199_1: "f32[384, 384]", arg200_1: "f32[384]", arg201_1: "f32[384]", arg202_1: "f32[384]", arg203_1: "f32[1536, 384]", arg204_1: "f32[1536]", arg205_1: "f32[384, 1536]", arg206_1: "f32[384]", arg207_1: "f32[24]", arg208_1: "f32[24]", arg209_1: "f32[48, 24]", arg210_1: "f32[24, 24]", arg211_1: "f32[24, 24]", arg212_1: "f32[24]", arg213_1: "f32[24]", arg214_1: "f32[24]", arg215_1: "f32[96, 24]", arg216_1: "f32[96]", arg217_1: "f32[24, 96]", arg218_1: "f32[24]", arg219_1: "f32[24]", arg220_1: "f32[24]", arg221_1: "f32[384, 384]", arg222_1: "f32[384]", arg223_1: "f32[384]", arg224_1: "f32[384]", arg225_1: "f32[768, 384]", arg226_1: "f32[384, 384]", arg227_1: "f32[384, 384]", arg228_1: "f32[384]", arg229_1: "f32[384]", arg230_1: "f32[384]", arg231_1: "f32[1536, 384]", arg232_1: "f32[1536]", arg233_1: "f32[384, 1536]", arg234_1: "f32[384]", arg235_1: "f32[24]", arg236_1: "f32[24]", arg237_1: "f32[48, 24]", arg238_1: "f32[24, 24]", arg239_1: "f32[24, 24]", arg240_1: "f32[24]", arg241_1: "f32[24]", arg242_1: "f32[24]", arg243_1: "f32[96, 24]", arg244_1: "f32[96]", arg245_1: "f32[24, 96]", arg246_1: "f32[24]", arg247_1: "f32[24]", arg248_1: "f32[24]", arg249_1: "f32[384, 384]", arg250_1: "f32[384]", arg251_1: "f32[384]", arg252_1: "f32[384]", arg253_1: "f32[768, 384]", arg254_1: "f32[384, 384]", arg255_1: "f32[384, 384]", arg256_1: "f32[384]", arg257_1: "f32[384]", arg258_1: "f32[384]", arg259_1: "f32[1536, 384]", arg260_1: "f32[1536]", arg261_1: "f32[384, 1536]", arg262_1: "f32[384]", arg263_1: "f32[24]", arg264_1: "f32[24]", arg265_1: "f32[48, 24]", arg266_1: "f32[24, 24]", arg267_1: "f32[24, 24]", arg268_1: "f32[24]", arg269_1: "f32[24]", arg270_1: "f32[24]", arg271_1: "f32[96, 24]", arg272_1: "f32[96]", arg273_1: "f32[24, 96]", arg274_1: "f32[24]", arg275_1: "f32[24]", arg276_1: "f32[24]", arg277_1: "f32[384, 384]", arg278_1: "f32[384]", arg279_1: "f32[384]", arg280_1: "f32[384]", arg281_1: "f32[768, 384]", arg282_1: "f32[384, 384]", arg283_1: "f32[384, 384]", arg284_1: "f32[384]", arg285_1: "f32[384]", arg286_1: "f32[384]", arg287_1: "f32[1536, 384]", arg288_1: "f32[1536]", arg289_1: "f32[384, 1536]", arg290_1: "f32[384]", arg291_1: "f32[24]", arg292_1: "f32[24]", arg293_1: "f32[48, 24]", arg294_1: "f32[24, 24]", arg295_1: "f32[24, 24]", arg296_1: "f32[24]", arg297_1: "f32[24]", arg298_1: "f32[24]", arg299_1: "f32[96, 24]", arg300_1: "f32[96]", arg301_1: "f32[24, 96]", arg302_1: "f32[24]", arg303_1: "f32[24]", arg304_1: "f32[24]", arg305_1: "f32[384, 384]", arg306_1: "f32[384]", arg307_1: "f32[384]", arg308_1: "f32[384]", arg309_1: "f32[768, 384]", arg310_1: "f32[384, 384]", arg311_1: "f32[384, 384]", arg312_1: "f32[384]", arg313_1: "f32[384]", arg314_1: "f32[384]", arg315_1: "f32[1536, 384]", arg316_1: "f32[1536]", arg317_1: "f32[384, 1536]", arg318_1: "f32[384]", arg319_1: "f32[24]", arg320_1: "f32[24]", arg321_1: "f32[48, 24]", arg322_1: "f32[24, 24]", arg323_1: "f32[24, 24]", arg324_1: "f32[24]", arg325_1: "f32[24]", arg326_1: "f32[24]", arg327_1: "f32[96, 24]", arg328_1: "f32[96]", arg329_1: "f32[24, 96]", arg330_1: "f32[24]", arg331_1: "f32[24]", arg332_1: "f32[24]", arg333_1: "f32[384, 384]", arg334_1: "f32[384]", arg335_1: "f32[384]", arg336_1: "f32[384]", arg337_1: "f32[768, 384]", arg338_1: "f32[384, 384]", arg339_1: "f32[384, 384]", arg340_1: "f32[384]", arg341_1: "f32[384]", arg342_1: "f32[384]", arg343_1: "f32[1536, 384]", arg344_1: "f32[1536]", arg345_1: "f32[384, 1536]", arg346_1: "f32[384]", arg347_1: "f32[384]", arg348_1: "f32[384]", arg349_1: "f32[1000, 384]", arg350_1: "f32[1000]", arg351_1: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:181, code: x = self.proj(x)
    convolution: "f32[8, 24, 56, 56]" = torch.ops.aten.convolution.default(arg351_1, arg3_1, arg4_1, [4, 4], [3, 3], [1, 1], False, [0, 0], 1);  arg351_1 = arg3_1 = arg4_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:182, code: x = self.unfold(x)
    iota: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 4, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    iota_1: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_1: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_1, -1);  iota_1 = None
    add: "i64[4, 14]" = torch.ops.aten.add.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
    unsqueeze_4: "i64[4, 14, 1]" = torch.ops.aten.unsqueeze.default(add, -1);  add = None
    unsqueeze_5: "i64[4, 14, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
    iota_2: "i64[14]" = torch.ops.prims.iota.default(14, start = 0, step = 4, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_2: "i64[1, 14]" = torch.ops.aten.unsqueeze.default(iota_2, 0);  iota_2 = None
    iota_3: "i64[4]" = torch.ops.prims.iota.default(4, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    unsqueeze_3: "i64[4, 1]" = torch.ops.aten.unsqueeze.default(iota_3, -1);  iota_3 = None
    add_1: "i64[4, 14]" = torch.ops.aten.add.Tensor(unsqueeze_2, unsqueeze_3);  unsqueeze_2 = unsqueeze_3 = None
    index: "f32[8, 24, 4, 14, 4, 14]" = torch.ops.aten.index.Tensor(convolution, [None, None, unsqueeze_5, add_1]);  convolution = unsqueeze_5 = add_1 = None
    permute: "f32[8, 24, 4, 4, 14, 14]" = torch.ops.aten.permute.default(index, [0, 1, 2, 4, 3, 5]);  index = None
    clone: "f32[8, 24, 4, 4, 14, 14]" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None
    view: "f32[8, 384, 196]" = torch.ops.aten.reshape.default(clone, [8, 384, 196]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:183, code: x = x.transpose(1, 2).reshape(B * self.num_patches, self.in_dim, self.new_patch_size[0], self.new_patch_size[1])
    permute_1: "f32[8, 196, 384]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    clone_1: "f32[8, 196, 384]" = torch.ops.aten.clone.default(permute_1, memory_format = torch.contiguous_format);  permute_1 = None
    view_1: "f32[1568, 24, 4, 4]" = torch.ops.aten.reshape.default(clone_1, [1568, 24, 4, 4]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:184, code: x = x + pixel_pos
    add_2: "f32[1568, 24, 4, 4]" = torch.ops.aten.add.Tensor(view_1, arg0_1);  view_1 = arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:185, code: x = x.reshape(B * self.num_patches, self.in_dim, -1).transpose(1, 2)
    view_2: "f32[1568, 24, 16]" = torch.ops.aten.reshape.default(add_2, [1568, 24, -1]);  add_2 = None
    permute_2: "f32[1568, 16, 24]" = torch.ops.aten.permute.default(view_2, [0, 2, 1]);  view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:311, code: patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
    clone_2: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format)
    view_3: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(clone_2, [8, 196, 384]);  clone_2 = None
    var_mean = torch.ops.aten.var_mean.correction(view_3, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 196, 1]" = var_mean[0]
    getitem_1: "f32[8, 196, 1]" = var_mean[1];  var_mean = None
    sub: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(view_3, getitem_1);  view_3 = getitem_1 = None
    add_3: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
    rsqrt: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    mul: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
    mul_1: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul, arg5_1);  mul = arg5_1 = None
    add_4: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_1, arg6_1);  mul_1 = arg6_1 = None
    view_4: "f32[1568, 384]" = torch.ops.aten.reshape.default(add_4, [1568, 384]);  add_4 = None
    permute_3: "f32[384, 384]" = torch.ops.aten.permute.default(arg7_1, [1, 0]);  arg7_1 = None
    addmm: "f32[1568, 384]" = torch.ops.aten.addmm.default(arg8_1, view_4, permute_3);  arg8_1 = view_4 = permute_3 = None
    view_5: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(addmm, [8, 196, 384]);  addmm = None
    var_mean_1 = torch.ops.aten.var_mean.correction(view_5, [2], correction = 0, keepdim = True)
    getitem_2: "f32[8, 196, 1]" = var_mean_1[0]
    getitem_3: "f32[8, 196, 1]" = var_mean_1[1];  var_mean_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    clone_4: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(permute_2, memory_format = torch.contiguous_format)
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_4, [2], correction = 0, keepdim = True)
    getitem_4: "f32[1568, 16, 1]" = var_mean_2[0]
    getitem_5: "f32[1568, 16, 1]" = var_mean_2[1];  var_mean_2 = None
    sub_2: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_4, getitem_5);  clone_4 = getitem_5 = None
    add_8: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
    rsqrt_2: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    mul_4: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = rsqrt_2 = None
    mul_5: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_4, arg11_1);  mul_4 = arg11_1 = None
    add_9: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_5, arg12_1);  mul_5 = arg12_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_6: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_9, [25088, 24])
    permute_4: "f32[24, 48]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
    mm: "f32[25088, 48]" = torch.ops.aten.mm.default(view_6, permute_4);  view_6 = permute_4 = None
    view_7: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm, [1568, 16, 48]);  mm = None
    view_8: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_7, [1568, 16, 2, 4, 6]);  view_7 = None
    permute_5: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_8, [2, 0, 3, 1, 4]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind = torch.ops.aten.unbind.int(permute_5);  permute_5 = None
    getitem_6: "f32[1568, 4, 16, 6]" = unbind[0]
    getitem_7: "f32[1568, 4, 16, 6]" = unbind[1];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_1: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_6, [1568, 4, 16, 6]);  getitem_6 = None
    clone_5: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_12: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_5, [6272, 16, 6]);  clone_5 = None
    permute_8: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_7, [0, 1, 3, 2]);  getitem_7 = None
    expand_2: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_8, [1568, 4, 6, 16]);  permute_8 = None
    clone_6: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_13: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_6, [6272, 6, 16]);  clone_6 = None
    bmm: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
    view_14: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm, [1568, 4, 16, 16]);  bmm = None
    mul_6: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_14, 0.408248290463863);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_6, [-1], True)
    sub_3: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_6, amax);  mul_6 = amax = None
    exp: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_1: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_3: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div, [1568, 4, 16, 16]);  div = None
    view_15: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_3, [6272, 16, 16]);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_9: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_9, [25088, 24]);  add_9 = None
    permute_6: "f32[24, 24]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    mm_1: "f32[25088, 24]" = torch.ops.aten.mm.default(view_9, permute_6);  view_9 = permute_6 = None
    view_10: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_1, [1568, 16, 24]);  mm_1 = None
    view_11: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_10, [1568, 16, 4, -1]);  view_10 = None
    permute_7: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_4: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_7, [1568, 4, 16, 6]);  permute_7 = None
    clone_7: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_16: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_7, [6272, 16, 6]);  clone_7 = None
    bmm_1: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_15, view_16);  view_15 = view_16 = None
    view_17: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_1, [1568, 4, 16, 6]);  bmm_1 = None
    permute_9: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
    clone_8: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
    view_18: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_8, [1568, 16, 24]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_19: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_18, [25088, 24]);  view_18 = None
    permute_10: "f32[24, 24]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
    
    # No stacktrace found for following nodes
    mm_default_83: "f32[25088, 24]" = torch.ops.aten.mm.default(view_19, permute_10);  view_19 = permute_10 = None
    add_tensor_83: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_83, arg16_1);  mm_default_83 = arg16_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_20: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_83, [1568, 16, 24]);  add_tensor_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_10: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(permute_2, view_20);  permute_2 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_9: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_10, memory_format = torch.contiguous_format)
    var_mean_3 = torch.ops.aten.var_mean.correction(clone_9, [2], correction = 0, keepdim = True)
    getitem_8: "f32[1568, 16, 1]" = var_mean_3[0]
    getitem_9: "f32[1568, 16, 1]" = var_mean_3[1];  var_mean_3 = None
    sub_4: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_9, getitem_9);  clone_9 = getitem_9 = None
    add_11: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
    rsqrt_3: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    mul_7: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = rsqrt_3 = None
    mul_8: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_7, arg17_1);  mul_7 = arg17_1 = None
    add_12: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_8, arg18_1);  mul_8 = arg18_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_21: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_12, [25088, 24]);  add_12 = None
    permute_11: "f32[24, 96]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
    
    # No stacktrace found for following nodes
    mm_default_82: "f32[25088, 96]" = torch.ops.aten.mm.default(view_21, permute_11);  view_21 = permute_11 = None
    add_tensor_82: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_82, arg20_1);  mm_default_82 = arg20_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_22: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_82, [1568, 16, 96]);  add_tensor_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_9: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
    mul_10: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476);  view_22 = None
    erf: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_10);  mul_10 = None
    add_13: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_11: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_9, add_13);  mul_9 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_23: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_11, [25088, 96]);  mul_11 = None
    permute_12: "f32[96, 24]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
    
    # No stacktrace found for following nodes
    mm_default_81: "f32[25088, 24]" = torch.ops.aten.mm.default(view_23, permute_12);  view_23 = permute_12 = None
    add_tensor_81: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_81, arg22_1);  mm_default_81 = arg22_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_24: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_81, [1568, 16, 24]);  add_tensor_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_14: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_10, view_24);  add_10 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_12: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_14, memory_format = torch.contiguous_format)
    var_mean_4 = torch.ops.aten.var_mean.correction(clone_12, [2], correction = 0, keepdim = True)
    getitem_10: "f32[1568, 16, 1]" = var_mean_4[0]
    getitem_11: "f32[1568, 16, 1]" = var_mean_4[1];  var_mean_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:312, code: patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
    expand: "f32[8, 1, 384]" = torch.ops.aten.expand.default(arg1_1, [8, -1, -1]);  arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:311, code: patch_embed = self.norm2_proj(self.proj(self.norm1_proj(pixel_embed.reshape(B, self.num_patches, -1))))
    sub_1: "f32[8, 196, 384]" = torch.ops.aten.sub.Tensor(view_5, getitem_3);  view_5 = getitem_3 = None
    add_5: "f32[8, 196, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
    rsqrt_1: "f32[8, 196, 1]" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
    mul_2: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = rsqrt_1 = None
    mul_3: "f32[8, 196, 384]" = torch.ops.aten.mul.Tensor(mul_2, arg9_1);  mul_2 = arg9_1 = None
    add_6: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(mul_3, arg10_1);  mul_3 = arg10_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:312, code: patch_embed = torch.cat((self.cls_token.expand(B, -1, -1), patch_embed), dim=1)
    cat: "f32[8, 197, 384]" = torch.ops.aten.cat.default([expand, add_6], 1);  expand = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:313, code: patch_embed = patch_embed + self.patch_pos
    add_7: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat, arg2_1);  cat = arg2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_4: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_7, 1, 0, 1)
    slice_6: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_7, 1, 1, 9223372036854775807);  add_7 = None
    sub_5: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_12, getitem_11);  clone_12 = getitem_11 = None
    add_15: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
    rsqrt_4: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    mul_12: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_4);  sub_5 = rsqrt_4 = None
    mul_13: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_12, arg23_1);  mul_12 = arg23_1 = None
    add_16: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_13, arg24_1);  mul_13 = arg24_1 = None
    view_25: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_16, [8, 196, -1]);  add_16 = None
    view_26: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_25, [1568, 384]);  view_25 = None
    permute_13: "f32[384, 384]" = torch.ops.aten.permute.default(arg25_1, [1, 0]);  arg25_1 = None
    
    # No stacktrace found for following nodes
    mm_default_80: "f32[1568, 384]" = torch.ops.aten.mm.default(view_26, permute_13);  view_26 = permute_13 = None
    add_tensor_80: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_80, arg26_1);  mm_default_80 = arg26_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    view_27: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_80, [8, 196, 384]);  add_tensor_80 = None
    add_17: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_6, view_27);  slice_6 = view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_1: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_4, add_17], 1);  slice_4 = add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_5 = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 197, 1]" = var_mean_5[0]
    getitem_13: "f32[8, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    sub_6: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_1, getitem_13);  getitem_13 = None
    add_18: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-05);  getitem_12 = None
    rsqrt_5: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    mul_14: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_5);  sub_6 = rsqrt_5 = None
    mul_15: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_14, arg27_1);  mul_14 = arg27_1 = None
    add_19: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_15, arg28_1);  mul_15 = arg28_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_28: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_19, [1576, 384])
    permute_14: "f32[384, 768]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
    mm_2: "f32[1576, 768]" = torch.ops.aten.mm.default(view_28, permute_14);  view_28 = permute_14 = None
    view_29: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_2, [8, 197, 768]);  mm_2 = None
    view_30: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_29, [8, 197, 2, 6, 64]);  view_29 = None
    permute_15: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_30, [2, 0, 3, 1, 4]);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_1 = torch.ops.aten.unbind.int(permute_15);  permute_15 = None
    getitem_14: "f32[8, 6, 197, 64]" = unbind_1[0]
    getitem_15: "f32[8, 6, 197, 64]" = unbind_1[1];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_5: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_14, [8, 6, 197, 64]);  getitem_14 = None
    clone_13: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_34: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_13, [48, 197, 64]);  clone_13 = None
    permute_18: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_15, [0, 1, 3, 2]);  getitem_15 = None
    expand_6: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_18, [8, 6, 64, 197]);  permute_18 = None
    clone_14: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_35: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_14, [48, 64, 197]);  clone_14 = None
    bmm_2: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
    view_36: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_2, [8, 6, 197, 197]);  bmm_2 = None
    mul_16: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_36, 0.125);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_16, [-1], True)
    sub_7: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_16, amax_1);  mul_16 = amax_1 = None
    exp_1: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_2: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_7: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_1, [8, 6, 197, 197]);  div_1 = None
    view_37: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_7, [48, 197, 197]);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_31: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_19, [1576, 384]);  add_19 = None
    permute_16: "f32[384, 384]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    mm_3: "f32[1576, 384]" = torch.ops.aten.mm.default(view_31, permute_16);  view_31 = permute_16 = None
    view_32: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_3, [8, 197, 384]);  mm_3 = None
    view_33: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_32, [8, 197, 6, -1]);  view_32 = None
    permute_17: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_33, [0, 2, 1, 3]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_8: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_17, [8, 6, 197, 64]);  permute_17 = None
    clone_15: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_38: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_15, [48, 197, 64]);  clone_15 = None
    bmm_3: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_37, view_38);  view_37 = view_38 = None
    view_39: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_3, [8, 6, 197, 64]);  bmm_3 = None
    permute_19: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_39, [0, 2, 1, 3]);  view_39 = None
    clone_16: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    view_40: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_16, [8, 197, 384]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_41: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_40, [1576, 384]);  view_40 = None
    permute_20: "f32[384, 384]" = torch.ops.aten.permute.default(arg31_1, [1, 0]);  arg31_1 = None
    
    # No stacktrace found for following nodes
    mm_default_79: "f32[1576, 384]" = torch.ops.aten.mm.default(view_41, permute_20);  view_41 = permute_20 = None
    add_tensor_79: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_79, arg32_1);  mm_default_79 = arg32_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_42: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_79, [8, 197, 384]);  add_tensor_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_20: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_1, view_42);  cat_1 = view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_6 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 197, 1]" = var_mean_6[0]
    getitem_17: "f32[8, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    clone_19: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_14, memory_format = torch.contiguous_format)
    var_mean_7 = torch.ops.aten.var_mean.correction(clone_19, [2], correction = 0, keepdim = True)
    getitem_18: "f32[1568, 16, 1]" = var_mean_7[0]
    getitem_19: "f32[1568, 16, 1]" = var_mean_7[1];  var_mean_7 = None
    sub_9: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_19, getitem_19);  clone_19 = getitem_19 = None
    add_25: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
    rsqrt_7: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    mul_22: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_7);  sub_9 = rsqrt_7 = None
    mul_23: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_22, arg39_1);  mul_22 = arg39_1 = None
    add_26: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_23, arg40_1);  mul_23 = arg40_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_47: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_26, [25088, 24])
    permute_23: "f32[24, 48]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
    mm_4: "f32[25088, 48]" = torch.ops.aten.mm.default(view_47, permute_23);  view_47 = permute_23 = None
    view_48: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_4, [1568, 16, 48]);  mm_4 = None
    view_49: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_48, [1568, 16, 2, 4, 6]);  view_48 = None
    permute_24: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_49, [2, 0, 3, 1, 4]);  view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_2 = torch.ops.aten.unbind.int(permute_24);  permute_24 = None
    getitem_20: "f32[1568, 4, 16, 6]" = unbind_2[0]
    getitem_21: "f32[1568, 4, 16, 6]" = unbind_2[1];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_9: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_20, [1568, 4, 16, 6]);  getitem_20 = None
    clone_20: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_53: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_20, [6272, 16, 6]);  clone_20 = None
    permute_27: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_21, [0, 1, 3, 2]);  getitem_21 = None
    expand_10: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_27, [1568, 4, 6, 16]);  permute_27 = None
    clone_21: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_54: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_21, [6272, 6, 16]);  clone_21 = None
    bmm_4: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_53, view_54);  view_53 = view_54 = None
    view_55: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_4, [1568, 4, 16, 16]);  bmm_4 = None
    mul_24: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_55, 0.408248290463863);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_24, [-1], True)
    sub_10: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_24, amax_2);  mul_24 = amax_2 = None
    exp_2: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_3: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_11: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_2, [1568, 4, 16, 16]);  div_2 = None
    view_56: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_11, [6272, 16, 16]);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_50: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_26, [25088, 24]);  add_26 = None
    permute_25: "f32[24, 24]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    mm_5: "f32[25088, 24]" = torch.ops.aten.mm.default(view_50, permute_25);  view_50 = permute_25 = None
    view_51: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_5, [1568, 16, 24]);  mm_5 = None
    view_52: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_51, [1568, 16, 4, -1]);  view_51 = None
    permute_26: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_12: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_26, [1568, 4, 16, 6]);  permute_26 = None
    clone_22: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_57: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_22, [6272, 16, 6]);  clone_22 = None
    bmm_5: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_56, view_57);  view_56 = view_57 = None
    view_58: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_5, [1568, 4, 16, 6]);  bmm_5 = None
    permute_28: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    clone_23: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_28, memory_format = torch.contiguous_format);  permute_28 = None
    view_59: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_23, [1568, 16, 24]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_60: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_59, [25088, 24]);  view_59 = None
    permute_29: "f32[24, 24]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
    
    # No stacktrace found for following nodes
    mm_default_78: "f32[25088, 24]" = torch.ops.aten.mm.default(view_60, permute_29);  view_60 = permute_29 = None
    add_tensor_78: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_78, arg44_1);  mm_default_78 = arg44_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_61: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_78, [1568, 16, 24]);  add_tensor_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_27: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_14, view_61);  add_14 = view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_24: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_27, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_24, [2], correction = 0, keepdim = True)
    getitem_22: "f32[1568, 16, 1]" = var_mean_8[0]
    getitem_23: "f32[1568, 16, 1]" = var_mean_8[1];  var_mean_8 = None
    sub_11: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_24, getitem_23);  clone_24 = getitem_23 = None
    add_28: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-05);  getitem_22 = None
    rsqrt_8: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    mul_25: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = rsqrt_8 = None
    mul_26: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_25, arg45_1);  mul_25 = arg45_1 = None
    add_29: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_26, arg46_1);  mul_26 = arg46_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_62: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_29, [25088, 24]);  add_29 = None
    permute_30: "f32[24, 96]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
    
    # No stacktrace found for following nodes
    mm_default_77: "f32[25088, 96]" = torch.ops.aten.mm.default(view_62, permute_30);  view_62 = permute_30 = None
    add_tensor_77: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_77, arg48_1);  mm_default_77 = arg48_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_63: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_77, [1568, 16, 96]);  add_tensor_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_27: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_28: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
    erf_2: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_30: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_29: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_27, add_30);  mul_27 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_64: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_29, [25088, 96]);  mul_29 = None
    permute_31: "f32[96, 24]" = torch.ops.aten.permute.default(arg49_1, [1, 0]);  arg49_1 = None
    
    # No stacktrace found for following nodes
    mm_default_76: "f32[25088, 24]" = torch.ops.aten.mm.default(view_64, permute_31);  view_64 = permute_31 = None
    add_tensor_76: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_76, arg50_1);  mm_default_76 = arg50_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_65: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_76, [1568, 16, 24]);  add_tensor_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_31: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_27, view_65);  add_27 = view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_27: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_31, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_27, [2], correction = 0, keepdim = True)
    getitem_24: "f32[1568, 16, 1]" = var_mean_9[0]
    getitem_25: "f32[1568, 16, 1]" = var_mean_9[1];  var_mean_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    sub_8: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_20, getitem_17);  getitem_17 = None
    add_21: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
    rsqrt_6: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    mul_17: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
    mul_18: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_17, arg33_1);  mul_17 = arg33_1 = None
    add_22: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_18, arg34_1);  mul_18 = arg34_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_43: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_22, [1576, 384]);  add_22 = None
    permute_21: "f32[384, 1536]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
    
    # No stacktrace found for following nodes
    mm_default_75: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_43, permute_21);  view_43 = permute_21 = None
    add_tensor_75: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_75, arg36_1);  mm_default_75 = arg36_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_44: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_75, [8, 197, 1536]);  add_tensor_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_19: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.5)
    mul_20: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_44, 0.7071067811865476);  view_44 = None
    erf_1: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_23: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_21: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_19, add_23);  mul_19 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_45: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_21, [1576, 1536]);  mul_21 = None
    permute_22: "f32[1536, 384]" = torch.ops.aten.permute.default(arg37_1, [1, 0]);  arg37_1 = None
    
    # No stacktrace found for following nodes
    mm_default_74: "f32[1576, 384]" = torch.ops.aten.mm.default(view_45, permute_22);  view_45 = permute_22 = None
    add_tensor_74: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_74, arg38_1);  mm_default_74 = arg38_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_46: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_74, [8, 197, 384]);  add_tensor_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_24: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_20, view_46);  add_20 = view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_8: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_24, 1, 0, 1)
    slice_10: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_24, 1, 1, 9223372036854775807);  add_24 = None
    sub_12: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_27, getitem_25);  clone_27 = getitem_25 = None
    add_32: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_24, 1e-05);  getitem_24 = None
    rsqrt_9: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    mul_30: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_9);  sub_12 = rsqrt_9 = None
    mul_31: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_30, arg51_1);  mul_30 = arg51_1 = None
    add_33: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_31, arg52_1);  mul_31 = arg52_1 = None
    view_66: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_33, [8, 196, -1]);  add_33 = None
    view_67: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_66, [1568, 384]);  view_66 = None
    permute_32: "f32[384, 384]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
    
    # No stacktrace found for following nodes
    mm_default_73: "f32[1568, 384]" = torch.ops.aten.mm.default(view_67, permute_32);  view_67 = permute_32 = None
    add_tensor_73: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_73, arg54_1);  mm_default_73 = arg54_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    view_68: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_73, [8, 196, 384]);  add_tensor_73 = None
    add_34: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_10, view_68);  slice_10 = view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_2: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_8, add_34], 1);  slice_8 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_10 = torch.ops.aten.var_mean.correction(cat_2, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 197, 1]" = var_mean_10[0]
    getitem_27: "f32[8, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    sub_13: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_2, getitem_27);  getitem_27 = None
    add_35: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
    rsqrt_10: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    mul_32: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_10);  sub_13 = rsqrt_10 = None
    mul_33: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_32, arg55_1);  mul_32 = arg55_1 = None
    add_36: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_33, arg56_1);  mul_33 = arg56_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_69: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_36, [1576, 384])
    permute_33: "f32[384, 768]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
    mm_6: "f32[1576, 768]" = torch.ops.aten.mm.default(view_69, permute_33);  view_69 = permute_33 = None
    view_70: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_6, [8, 197, 768]);  mm_6 = None
    view_71: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_70, [8, 197, 2, 6, 64]);  view_70 = None
    permute_34: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_71, [2, 0, 3, 1, 4]);  view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_3 = torch.ops.aten.unbind.int(permute_34);  permute_34 = None
    getitem_28: "f32[8, 6, 197, 64]" = unbind_3[0]
    getitem_29: "f32[8, 6, 197, 64]" = unbind_3[1];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_13: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_28, [8, 6, 197, 64]);  getitem_28 = None
    clone_28: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_75: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_28, [48, 197, 64]);  clone_28 = None
    permute_37: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_29, [0, 1, 3, 2]);  getitem_29 = None
    expand_14: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_37, [8, 6, 64, 197]);  permute_37 = None
    clone_29: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_76: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_29, [48, 64, 197]);  clone_29 = None
    bmm_6: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_75, view_76);  view_75 = view_76 = None
    view_77: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_6, [8, 6, 197, 197]);  bmm_6 = None
    mul_34: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_77, 0.125);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_34, [-1], True)
    sub_14: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_34, amax_3);  mul_34 = amax_3 = None
    exp_3: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_4: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_15: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_3, [8, 6, 197, 197]);  div_3 = None
    view_78: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_15, [48, 197, 197]);  expand_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_72: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_36, [1576, 384]);  add_36 = None
    permute_35: "f32[384, 384]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    mm_7: "f32[1576, 384]" = torch.ops.aten.mm.default(view_72, permute_35);  view_72 = permute_35 = None
    view_73: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_7, [8, 197, 384]);  mm_7 = None
    view_74: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_73, [8, 197, 6, -1]);  view_73 = None
    permute_36: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_16: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_36, [8, 6, 197, 64]);  permute_36 = None
    clone_30: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_79: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_30, [48, 197, 64]);  clone_30 = None
    bmm_7: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
    view_80: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_7, [8, 6, 197, 64]);  bmm_7 = None
    permute_38: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    clone_31: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    view_81: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_31, [8, 197, 384]);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_82: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_81, [1576, 384]);  view_81 = None
    permute_39: "f32[384, 384]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
    
    # No stacktrace found for following nodes
    mm_default_72: "f32[1576, 384]" = torch.ops.aten.mm.default(view_82, permute_39);  view_82 = permute_39 = None
    add_tensor_72: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_72, arg60_1);  mm_default_72 = arg60_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_83: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_72, [8, 197, 384]);  add_tensor_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_37: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_2, view_83);  cat_2 = view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_11 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 197, 1]" = var_mean_11[0]
    getitem_31: "f32[8, 197, 1]" = var_mean_11[1];  var_mean_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    clone_34: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_31, memory_format = torch.contiguous_format)
    var_mean_12 = torch.ops.aten.var_mean.correction(clone_34, [2], correction = 0, keepdim = True)
    getitem_32: "f32[1568, 16, 1]" = var_mean_12[0]
    getitem_33: "f32[1568, 16, 1]" = var_mean_12[1];  var_mean_12 = None
    sub_16: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_34, getitem_33);  clone_34 = getitem_33 = None
    add_42: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-05);  getitem_32 = None
    rsqrt_12: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    mul_40: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_12);  sub_16 = rsqrt_12 = None
    mul_41: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_40, arg67_1);  mul_40 = arg67_1 = None
    add_43: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_41, arg68_1);  mul_41 = arg68_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_88: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_43, [25088, 24])
    permute_42: "f32[24, 48]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
    mm_8: "f32[25088, 48]" = torch.ops.aten.mm.default(view_88, permute_42);  view_88 = permute_42 = None
    view_89: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_8, [1568, 16, 48]);  mm_8 = None
    view_90: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_89, [1568, 16, 2, 4, 6]);  view_89 = None
    permute_43: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_90, [2, 0, 3, 1, 4]);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_4 = torch.ops.aten.unbind.int(permute_43);  permute_43 = None
    getitem_34: "f32[1568, 4, 16, 6]" = unbind_4[0]
    getitem_35: "f32[1568, 4, 16, 6]" = unbind_4[1];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_17: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_34, [1568, 4, 16, 6]);  getitem_34 = None
    clone_35: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_94: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_35, [6272, 16, 6]);  clone_35 = None
    permute_46: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_35, [0, 1, 3, 2]);  getitem_35 = None
    expand_18: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_46, [1568, 4, 6, 16]);  permute_46 = None
    clone_36: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_95: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_36, [6272, 6, 16]);  clone_36 = None
    bmm_8: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_94, view_95);  view_94 = view_95 = None
    view_96: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_8, [1568, 4, 16, 16]);  bmm_8 = None
    mul_42: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_96, 0.408248290463863);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_42, [-1], True)
    sub_17: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_42, amax_4);  mul_42 = amax_4 = None
    exp_4: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_5: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_19: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_4, [1568, 4, 16, 16]);  div_4 = None
    view_97: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_19, [6272, 16, 16]);  expand_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_91: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_43, [25088, 24]);  add_43 = None
    permute_44: "f32[24, 24]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    mm_9: "f32[25088, 24]" = torch.ops.aten.mm.default(view_91, permute_44);  view_91 = permute_44 = None
    view_92: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_9, [1568, 16, 24]);  mm_9 = None
    view_93: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_92, [1568, 16, 4, -1]);  view_92 = None
    permute_45: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_20: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_45, [1568, 4, 16, 6]);  permute_45 = None
    clone_37: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_98: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_37, [6272, 16, 6]);  clone_37 = None
    bmm_9: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
    view_99: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_9, [1568, 4, 16, 6]);  bmm_9 = None
    permute_47: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_99, [0, 2, 1, 3]);  view_99 = None
    clone_38: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_47, memory_format = torch.contiguous_format);  permute_47 = None
    view_100: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_38, [1568, 16, 24]);  clone_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_101: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_100, [25088, 24]);  view_100 = None
    permute_48: "f32[24, 24]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
    
    # No stacktrace found for following nodes
    mm_default_71: "f32[25088, 24]" = torch.ops.aten.mm.default(view_101, permute_48);  view_101 = permute_48 = None
    add_tensor_71: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_71, arg72_1);  mm_default_71 = arg72_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_102: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_71, [1568, 16, 24]);  add_tensor_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_44: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_31, view_102);  add_31 = view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_39: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_44, memory_format = torch.contiguous_format)
    var_mean_13 = torch.ops.aten.var_mean.correction(clone_39, [2], correction = 0, keepdim = True)
    getitem_36: "f32[1568, 16, 1]" = var_mean_13[0]
    getitem_37: "f32[1568, 16, 1]" = var_mean_13[1];  var_mean_13 = None
    sub_18: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_39, getitem_37);  clone_39 = getitem_37 = None
    add_45: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
    rsqrt_13: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    mul_43: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = rsqrt_13 = None
    mul_44: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_43, arg73_1);  mul_43 = arg73_1 = None
    add_46: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_44, arg74_1);  mul_44 = arg74_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_103: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_46, [25088, 24]);  add_46 = None
    permute_49: "f32[24, 96]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
    
    # No stacktrace found for following nodes
    mm_default_70: "f32[25088, 96]" = torch.ops.aten.mm.default(view_103, permute_49);  view_103 = permute_49 = None
    add_tensor_70: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_70, arg76_1);  mm_default_70 = arg76_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_104: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_70, [1568, 16, 96]);  add_tensor_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_45: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_104, 0.5)
    mul_46: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_104, 0.7071067811865476);  view_104 = None
    erf_4: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_46);  mul_46 = None
    add_47: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_47: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_45, add_47);  mul_45 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_105: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_47, [25088, 96]);  mul_47 = None
    permute_50: "f32[96, 24]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
    
    # No stacktrace found for following nodes
    mm_default_69: "f32[25088, 24]" = torch.ops.aten.mm.default(view_105, permute_50);  view_105 = permute_50 = None
    add_tensor_69: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_69, arg78_1);  mm_default_69 = arg78_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_106: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_69, [1568, 16, 24]);  add_tensor_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_48: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_44, view_106);  add_44 = view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_42: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_48, memory_format = torch.contiguous_format)
    var_mean_14 = torch.ops.aten.var_mean.correction(clone_42, [2], correction = 0, keepdim = True)
    getitem_38: "f32[1568, 16, 1]" = var_mean_14[0]
    getitem_39: "f32[1568, 16, 1]" = var_mean_14[1];  var_mean_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    sub_15: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_37, getitem_31);  getitem_31 = None
    add_38: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
    rsqrt_11: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    mul_35: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = rsqrt_11 = None
    mul_36: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_35, arg61_1);  mul_35 = arg61_1 = None
    add_39: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_36, arg62_1);  mul_36 = arg62_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_84: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_39, [1576, 384]);  add_39 = None
    permute_40: "f32[384, 1536]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
    
    # No stacktrace found for following nodes
    mm_default_68: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_84, permute_40);  view_84 = permute_40 = None
    add_tensor_68: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_68, arg64_1);  mm_default_68 = arg64_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_85: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_68, [8, 197, 1536]);  add_tensor_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_37: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_38: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
    erf_3: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_40: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_39: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_37, add_40);  mul_37 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_86: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_39, [1576, 1536]);  mul_39 = None
    permute_41: "f32[1536, 384]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
    
    # No stacktrace found for following nodes
    mm_default_67: "f32[1576, 384]" = torch.ops.aten.mm.default(view_86, permute_41);  view_86 = permute_41 = None
    add_tensor_67: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_67, arg66_1);  mm_default_67 = arg66_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_87: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_67, [8, 197, 384]);  add_tensor_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_41: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_37, view_87);  add_37 = view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_12: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_41, 1, 0, 1)
    slice_14: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_41, 1, 1, 9223372036854775807);  add_41 = None
    sub_19: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_42, getitem_39);  clone_42 = getitem_39 = None
    add_49: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
    rsqrt_14: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    mul_48: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_14);  sub_19 = rsqrt_14 = None
    mul_49: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_48, arg79_1);  mul_48 = arg79_1 = None
    add_50: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_49, arg80_1);  mul_49 = arg80_1 = None
    view_107: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_50, [8, 196, -1]);  add_50 = None
    view_108: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_107, [1568, 384]);  view_107 = None
    permute_51: "f32[384, 384]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
    
    # No stacktrace found for following nodes
    mm_default_66: "f32[1568, 384]" = torch.ops.aten.mm.default(view_108, permute_51);  view_108 = permute_51 = None
    add_tensor_66: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_66, arg82_1);  mm_default_66 = arg82_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    view_109: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_66, [8, 196, 384]);  add_tensor_66 = None
    add_51: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_14, view_109);  slice_14 = view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_3: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_12, add_51], 1);  slice_12 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_15 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 197, 1]" = var_mean_15[0]
    getitem_41: "f32[8, 197, 1]" = var_mean_15[1];  var_mean_15 = None
    sub_20: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_3, getitem_41);  getitem_41 = None
    add_52: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
    rsqrt_15: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    mul_50: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_15);  sub_20 = rsqrt_15 = None
    mul_51: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_50, arg83_1);  mul_50 = arg83_1 = None
    add_53: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_51, arg84_1);  mul_51 = arg84_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_110: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_53, [1576, 384])
    permute_52: "f32[384, 768]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
    mm_10: "f32[1576, 768]" = torch.ops.aten.mm.default(view_110, permute_52);  view_110 = permute_52 = None
    view_111: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_10, [8, 197, 768]);  mm_10 = None
    view_112: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_111, [8, 197, 2, 6, 64]);  view_111 = None
    permute_53: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_112, [2, 0, 3, 1, 4]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_5 = torch.ops.aten.unbind.int(permute_53);  permute_53 = None
    getitem_42: "f32[8, 6, 197, 64]" = unbind_5[0]
    getitem_43: "f32[8, 6, 197, 64]" = unbind_5[1];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_21: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_42, [8, 6, 197, 64]);  getitem_42 = None
    clone_43: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_116: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_43, [48, 197, 64]);  clone_43 = None
    permute_56: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_43, [0, 1, 3, 2]);  getitem_43 = None
    expand_22: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_56, [8, 6, 64, 197]);  permute_56 = None
    clone_44: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_117: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_44, [48, 64, 197]);  clone_44 = None
    bmm_10: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_116, view_117);  view_116 = view_117 = None
    view_118: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_10, [8, 6, 197, 197]);  bmm_10 = None
    mul_52: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_118, 0.125);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_52, [-1], True)
    sub_21: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_52, amax_5);  mul_52 = amax_5 = None
    exp_5: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_6: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_23: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_5, [8, 6, 197, 197]);  div_5 = None
    view_119: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_23, [48, 197, 197]);  expand_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_113: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_53, [1576, 384]);  add_53 = None
    permute_54: "f32[384, 384]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    mm_11: "f32[1576, 384]" = torch.ops.aten.mm.default(view_113, permute_54);  view_113 = permute_54 = None
    view_114: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_11, [8, 197, 384]);  mm_11 = None
    view_115: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_114, [8, 197, 6, -1]);  view_114 = None
    permute_55: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_24: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_55, [8, 6, 197, 64]);  permute_55 = None
    clone_45: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_120: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_45, [48, 197, 64]);  clone_45 = None
    bmm_11: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
    view_121: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_11, [8, 6, 197, 64]);  bmm_11 = None
    permute_57: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
    clone_46: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_57, memory_format = torch.contiguous_format);  permute_57 = None
    view_122: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_46, [8, 197, 384]);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_123: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_122, [1576, 384]);  view_122 = None
    permute_58: "f32[384, 384]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
    
    # No stacktrace found for following nodes
    mm_default_65: "f32[1576, 384]" = torch.ops.aten.mm.default(view_123, permute_58);  view_123 = permute_58 = None
    add_tensor_65: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_65, arg88_1);  mm_default_65 = arg88_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_124: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_65, [8, 197, 384]);  add_tensor_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_54: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_3, view_124);  cat_3 = view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_16 = torch.ops.aten.var_mean.correction(add_54, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 197, 1]" = var_mean_16[0]
    getitem_45: "f32[8, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    clone_49: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_48, memory_format = torch.contiguous_format)
    var_mean_17 = torch.ops.aten.var_mean.correction(clone_49, [2], correction = 0, keepdim = True)
    getitem_46: "f32[1568, 16, 1]" = var_mean_17[0]
    getitem_47: "f32[1568, 16, 1]" = var_mean_17[1];  var_mean_17 = None
    sub_23: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_49, getitem_47);  clone_49 = getitem_47 = None
    add_59: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
    rsqrt_17: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    mul_58: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_17);  sub_23 = rsqrt_17 = None
    mul_59: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_58, arg95_1);  mul_58 = arg95_1 = None
    add_60: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_59, arg96_1);  mul_59 = arg96_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_129: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_60, [25088, 24])
    permute_61: "f32[24, 48]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
    mm_12: "f32[25088, 48]" = torch.ops.aten.mm.default(view_129, permute_61);  view_129 = permute_61 = None
    view_130: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_12, [1568, 16, 48]);  mm_12 = None
    view_131: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_130, [1568, 16, 2, 4, 6]);  view_130 = None
    permute_62: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_131, [2, 0, 3, 1, 4]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_6 = torch.ops.aten.unbind.int(permute_62);  permute_62 = None
    getitem_48: "f32[1568, 4, 16, 6]" = unbind_6[0]
    getitem_49: "f32[1568, 4, 16, 6]" = unbind_6[1];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_25: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_48, [1568, 4, 16, 6]);  getitem_48 = None
    clone_50: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_135: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_50, [6272, 16, 6]);  clone_50 = None
    permute_65: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_49, [0, 1, 3, 2]);  getitem_49 = None
    expand_26: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_65, [1568, 4, 6, 16]);  permute_65 = None
    clone_51: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_136: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_51, [6272, 6, 16]);  clone_51 = None
    bmm_12: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_135, view_136);  view_135 = view_136 = None
    view_137: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_12, [1568, 4, 16, 16]);  bmm_12 = None
    mul_60: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_137, 0.408248290463863);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_6: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_60, [-1], True)
    sub_24: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_60, amax_6);  mul_60 = amax_6 = None
    exp_6: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_7: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_27: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_6, [1568, 4, 16, 16]);  div_6 = None
    view_138: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_27, [6272, 16, 16]);  expand_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_132: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_60, [25088, 24]);  add_60 = None
    permute_63: "f32[24, 24]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    mm_13: "f32[25088, 24]" = torch.ops.aten.mm.default(view_132, permute_63);  view_132 = permute_63 = None
    view_133: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_13, [1568, 16, 24]);  mm_13 = None
    view_134: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_133, [1568, 16, 4, -1]);  view_133 = None
    permute_64: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_28: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_64, [1568, 4, 16, 6]);  permute_64 = None
    clone_52: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_139: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_52, [6272, 16, 6]);  clone_52 = None
    bmm_13: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_138, view_139);  view_138 = view_139 = None
    view_140: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_13, [1568, 4, 16, 6]);  bmm_13 = None
    permute_66: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    clone_53: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    view_141: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_53, [1568, 16, 24]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_142: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_141, [25088, 24]);  view_141 = None
    permute_67: "f32[24, 24]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
    
    # No stacktrace found for following nodes
    mm_default_64: "f32[25088, 24]" = torch.ops.aten.mm.default(view_142, permute_67);  view_142 = permute_67 = None
    add_tensor_64: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_64, arg100_1);  mm_default_64 = arg100_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_143: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_64, [1568, 16, 24]);  add_tensor_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_61: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_48, view_143);  add_48 = view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_54: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_61, memory_format = torch.contiguous_format)
    var_mean_18 = torch.ops.aten.var_mean.correction(clone_54, [2], correction = 0, keepdim = True)
    getitem_50: "f32[1568, 16, 1]" = var_mean_18[0]
    getitem_51: "f32[1568, 16, 1]" = var_mean_18[1];  var_mean_18 = None
    sub_25: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_54, getitem_51);  clone_54 = getitem_51 = None
    add_62: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
    rsqrt_18: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    mul_61: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_18);  sub_25 = rsqrt_18 = None
    mul_62: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_61, arg101_1);  mul_61 = arg101_1 = None
    add_63: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_62, arg102_1);  mul_62 = arg102_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_144: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_63, [25088, 24]);  add_63 = None
    permute_68: "f32[24, 96]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
    
    # No stacktrace found for following nodes
    mm_default_63: "f32[25088, 96]" = torch.ops.aten.mm.default(view_144, permute_68);  view_144 = permute_68 = None
    add_tensor_63: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_63, arg104_1);  mm_default_63 = arg104_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_145: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_63, [1568, 16, 96]);  add_tensor_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_63: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_145, 0.5)
    mul_64: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_145, 0.7071067811865476);  view_145 = None
    erf_6: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
    add_64: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_65: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_63, add_64);  mul_63 = add_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_146: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_65, [25088, 96]);  mul_65 = None
    permute_69: "f32[96, 24]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
    
    # No stacktrace found for following nodes
    mm_default_62: "f32[25088, 24]" = torch.ops.aten.mm.default(view_146, permute_69);  view_146 = permute_69 = None
    add_tensor_62: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_62, arg106_1);  mm_default_62 = arg106_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_147: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_62, [1568, 16, 24]);  add_tensor_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_65: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_61, view_147);  add_61 = view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_57: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_65, memory_format = torch.contiguous_format)
    var_mean_19 = torch.ops.aten.var_mean.correction(clone_57, [2], correction = 0, keepdim = True)
    getitem_52: "f32[1568, 16, 1]" = var_mean_19[0]
    getitem_53: "f32[1568, 16, 1]" = var_mean_19[1];  var_mean_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    sub_22: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_54, getitem_45);  getitem_45 = None
    add_55: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-05);  getitem_44 = None
    rsqrt_16: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    mul_53: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_16);  sub_22 = rsqrt_16 = None
    mul_54: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_53, arg89_1);  mul_53 = arg89_1 = None
    add_56: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_54, arg90_1);  mul_54 = arg90_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_125: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_56, [1576, 384]);  add_56 = None
    permute_59: "f32[384, 1536]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
    
    # No stacktrace found for following nodes
    mm_default_61: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_125, permute_59);  view_125 = permute_59 = None
    add_tensor_61: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_61, arg92_1);  mm_default_61 = arg92_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_126: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_61, [8, 197, 1536]);  add_tensor_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_55: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_126, 0.5)
    mul_56: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_126, 0.7071067811865476);  view_126 = None
    erf_5: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_56);  mul_56 = None
    add_57: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_57: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_55, add_57);  mul_55 = add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_127: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_57, [1576, 1536]);  mul_57 = None
    permute_60: "f32[1536, 384]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
    
    # No stacktrace found for following nodes
    mm_default_60: "f32[1576, 384]" = torch.ops.aten.mm.default(view_127, permute_60);  view_127 = permute_60 = None
    add_tensor_60: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_60, arg94_1);  mm_default_60 = arg94_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_128: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_60, [8, 197, 384]);  add_tensor_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_58: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_54, view_128);  add_54 = view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_16: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_58, 1, 0, 1)
    slice_18: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_58, 1, 1, 9223372036854775807);  add_58 = None
    sub_26: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_57, getitem_53);  clone_57 = getitem_53 = None
    add_66: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-05);  getitem_52 = None
    rsqrt_19: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    mul_66: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_19);  sub_26 = rsqrt_19 = None
    mul_67: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_66, arg107_1);  mul_66 = arg107_1 = None
    add_67: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_67, arg108_1);  mul_67 = arg108_1 = None
    view_148: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_67, [8, 196, -1]);  add_67 = None
    view_149: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_148, [1568, 384]);  view_148 = None
    permute_70: "f32[384, 384]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
    
    # No stacktrace found for following nodes
    mm_default_59: "f32[1568, 384]" = torch.ops.aten.mm.default(view_149, permute_70);  view_149 = permute_70 = None
    add_tensor_59: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_59, arg110_1);  mm_default_59 = arg110_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    view_150: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_59, [8, 196, 384]);  add_tensor_59 = None
    add_68: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_18, view_150);  slice_18 = view_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_4: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_16, add_68], 1);  slice_16 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_20 = torch.ops.aten.var_mean.correction(cat_4, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_55: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    sub_27: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_4, getitem_55);  getitem_55 = None
    add_69: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-05);  getitem_54 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    mul_68: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_20);  sub_27 = rsqrt_20 = None
    mul_69: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_68, arg111_1);  mul_68 = arg111_1 = None
    add_70: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_69, arg112_1);  mul_69 = arg112_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_151: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_70, [1576, 384])
    permute_71: "f32[384, 768]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
    mm_14: "f32[1576, 768]" = torch.ops.aten.mm.default(view_151, permute_71);  view_151 = permute_71 = None
    view_152: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_14, [8, 197, 768]);  mm_14 = None
    view_153: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_152, [8, 197, 2, 6, 64]);  view_152 = None
    permute_72: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_153, [2, 0, 3, 1, 4]);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_7 = torch.ops.aten.unbind.int(permute_72);  permute_72 = None
    getitem_56: "f32[8, 6, 197, 64]" = unbind_7[0]
    getitem_57: "f32[8, 6, 197, 64]" = unbind_7[1];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_29: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_56, [8, 6, 197, 64]);  getitem_56 = None
    clone_58: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_157: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_58, [48, 197, 64]);  clone_58 = None
    permute_75: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_57, [0, 1, 3, 2]);  getitem_57 = None
    expand_30: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_75, [8, 6, 64, 197]);  permute_75 = None
    clone_59: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_158: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_59, [48, 64, 197]);  clone_59 = None
    bmm_14: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_157, view_158);  view_157 = view_158 = None
    view_159: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_14, [8, 6, 197, 197]);  bmm_14 = None
    mul_70: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_159, 0.125);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_7: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_70, [-1], True)
    sub_28: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_70, amax_7);  mul_70 = amax_7 = None
    exp_7: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_8: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_31: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_7, [8, 6, 197, 197]);  div_7 = None
    view_160: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_31, [48, 197, 197]);  expand_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_154: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_70, [1576, 384]);  add_70 = None
    permute_73: "f32[384, 384]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    mm_15: "f32[1576, 384]" = torch.ops.aten.mm.default(view_154, permute_73);  view_154 = permute_73 = None
    view_155: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_15, [8, 197, 384]);  mm_15 = None
    view_156: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_155, [8, 197, 6, -1]);  view_155 = None
    permute_74: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_32: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_74, [8, 6, 197, 64]);  permute_74 = None
    clone_60: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_161: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_60, [48, 197, 64]);  clone_60 = None
    bmm_15: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_160, view_161);  view_160 = view_161 = None
    view_162: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_15, [8, 6, 197, 64]);  bmm_15 = None
    permute_76: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    clone_61: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_76, memory_format = torch.contiguous_format);  permute_76 = None
    view_163: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_61, [8, 197, 384]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_164: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_163, [1576, 384]);  view_163 = None
    permute_77: "f32[384, 384]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
    
    # No stacktrace found for following nodes
    mm_default_58: "f32[1576, 384]" = torch.ops.aten.mm.default(view_164, permute_77);  view_164 = permute_77 = None
    add_tensor_58: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_58, arg116_1);  mm_default_58 = arg116_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_165: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_58, [8, 197, 384]);  add_tensor_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_71: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_4, view_165);  cat_4 = view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_21 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_58: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_59: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    clone_64: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_65, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_64, [2], correction = 0, keepdim = True)
    getitem_60: "f32[1568, 16, 1]" = var_mean_22[0]
    getitem_61: "f32[1568, 16, 1]" = var_mean_22[1];  var_mean_22 = None
    sub_30: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_64, getitem_61);  clone_64 = getitem_61 = None
    add_76: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
    rsqrt_22: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    mul_76: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_22);  sub_30 = rsqrt_22 = None
    mul_77: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_76, arg123_1);  mul_76 = arg123_1 = None
    add_77: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_77, arg124_1);  mul_77 = arg124_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_170: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_77, [25088, 24])
    permute_80: "f32[24, 48]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
    mm_16: "f32[25088, 48]" = torch.ops.aten.mm.default(view_170, permute_80);  view_170 = permute_80 = None
    view_171: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_16, [1568, 16, 48]);  mm_16 = None
    view_172: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_171, [1568, 16, 2, 4, 6]);  view_171 = None
    permute_81: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_172, [2, 0, 3, 1, 4]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_8 = torch.ops.aten.unbind.int(permute_81);  permute_81 = None
    getitem_62: "f32[1568, 4, 16, 6]" = unbind_8[0]
    getitem_63: "f32[1568, 4, 16, 6]" = unbind_8[1];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_33: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_62, [1568, 4, 16, 6]);  getitem_62 = None
    clone_65: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_176: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_65, [6272, 16, 6]);  clone_65 = None
    permute_84: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_63, [0, 1, 3, 2]);  getitem_63 = None
    expand_34: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_84, [1568, 4, 6, 16]);  permute_84 = None
    clone_66: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_177: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_66, [6272, 6, 16]);  clone_66 = None
    bmm_16: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_176, view_177);  view_176 = view_177 = None
    view_178: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_16, [1568, 4, 16, 16]);  bmm_16 = None
    mul_78: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_178, 0.408248290463863);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_8: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_78, [-1], True)
    sub_31: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_78, amax_8);  mul_78 = amax_8 = None
    exp_8: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_9: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_35: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_8, [1568, 4, 16, 16]);  div_8 = None
    view_179: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_35, [6272, 16, 16]);  expand_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_173: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_77, [25088, 24]);  add_77 = None
    permute_82: "f32[24, 24]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    mm_17: "f32[25088, 24]" = torch.ops.aten.mm.default(view_173, permute_82);  view_173 = permute_82 = None
    view_174: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_17, [1568, 16, 24]);  mm_17 = None
    view_175: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_174, [1568, 16, 4, -1]);  view_174 = None
    permute_83: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_175, [0, 2, 1, 3]);  view_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_36: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_83, [1568, 4, 16, 6]);  permute_83 = None
    clone_67: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_180: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_67, [6272, 16, 6]);  clone_67 = None
    bmm_17: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_179, view_180);  view_179 = view_180 = None
    view_181: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_17, [1568, 4, 16, 6]);  bmm_17 = None
    permute_85: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
    clone_68: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_182: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_68, [1568, 16, 24]);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_183: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_182, [25088, 24]);  view_182 = None
    permute_86: "f32[24, 24]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
    
    # No stacktrace found for following nodes
    mm_default_57: "f32[25088, 24]" = torch.ops.aten.mm.default(view_183, permute_86);  view_183 = permute_86 = None
    add_tensor_57: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_57, arg128_1);  mm_default_57 = arg128_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_184: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_57, [1568, 16, 24]);  add_tensor_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_78: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_65, view_184);  add_65 = view_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_69: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_78, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_69, [2], correction = 0, keepdim = True)
    getitem_64: "f32[1568, 16, 1]" = var_mean_23[0]
    getitem_65: "f32[1568, 16, 1]" = var_mean_23[1];  var_mean_23 = None
    sub_32: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_69, getitem_65);  clone_69 = getitem_65 = None
    add_79: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-05);  getitem_64 = None
    rsqrt_23: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    mul_79: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_23);  sub_32 = rsqrt_23 = None
    mul_80: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_79, arg129_1);  mul_79 = arg129_1 = None
    add_80: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_80, arg130_1);  mul_80 = arg130_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_185: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_80, [25088, 24]);  add_80 = None
    permute_87: "f32[24, 96]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
    
    # No stacktrace found for following nodes
    mm_default_56: "f32[25088, 96]" = torch.ops.aten.mm.default(view_185, permute_87);  view_185 = permute_87 = None
    add_tensor_56: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_56, arg132_1);  mm_default_56 = arg132_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_186: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_56, [1568, 16, 96]);  add_tensor_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_81: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_186, 0.5)
    mul_82: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_186, 0.7071067811865476);  view_186 = None
    erf_8: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
    add_81: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_83: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_81, add_81);  mul_81 = add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_187: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_83, [25088, 96]);  mul_83 = None
    permute_88: "f32[96, 24]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
    
    # No stacktrace found for following nodes
    mm_default_55: "f32[25088, 24]" = torch.ops.aten.mm.default(view_187, permute_88);  view_187 = permute_88 = None
    add_tensor_55: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_55, arg134_1);  mm_default_55 = arg134_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_188: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_55, [1568, 16, 24]);  add_tensor_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_82: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_78, view_188);  add_78 = view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_72: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_82, memory_format = torch.contiguous_format)
    var_mean_24 = torch.ops.aten.var_mean.correction(clone_72, [2], correction = 0, keepdim = True)
    getitem_66: "f32[1568, 16, 1]" = var_mean_24[0]
    getitem_67: "f32[1568, 16, 1]" = var_mean_24[1];  var_mean_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    sub_29: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_71, getitem_59);  getitem_59 = None
    add_72: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    mul_71: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_21);  sub_29 = rsqrt_21 = None
    mul_72: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_71, arg117_1);  mul_71 = arg117_1 = None
    add_73: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_72, arg118_1);  mul_72 = arg118_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_166: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_73, [1576, 384]);  add_73 = None
    permute_78: "f32[384, 1536]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
    
    # No stacktrace found for following nodes
    mm_default_54: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_166, permute_78);  view_166 = permute_78 = None
    add_tensor_54: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_54, arg120_1);  mm_default_54 = arg120_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_167: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_54, [8, 197, 1536]);  add_tensor_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_73: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_167, 0.5)
    mul_74: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_167, 0.7071067811865476);  view_167 = None
    erf_7: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
    add_74: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_75: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_73, add_74);  mul_73 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_168: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_75, [1576, 1536]);  mul_75 = None
    permute_79: "f32[1536, 384]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
    
    # No stacktrace found for following nodes
    mm_default_53: "f32[1576, 384]" = torch.ops.aten.mm.default(view_168, permute_79);  view_168 = permute_79 = None
    add_tensor_53: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_53, arg122_1);  mm_default_53 = arg122_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_169: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_53, [8, 197, 384]);  add_tensor_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_75: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_71, view_169);  add_71 = view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_20: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_75, 1, 0, 1)
    slice_22: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_75, 1, 1, 9223372036854775807);  add_75 = None
    sub_33: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_72, getitem_67);  clone_72 = getitem_67 = None
    add_83: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
    rsqrt_24: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    mul_84: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_24);  sub_33 = rsqrt_24 = None
    mul_85: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_84, arg135_1);  mul_84 = arg135_1 = None
    add_84: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_85, arg136_1);  mul_85 = arg136_1 = None
    view_189: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_84, [8, 196, -1]);  add_84 = None
    view_190: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_189, [1568, 384]);  view_189 = None
    permute_89: "f32[384, 384]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
    
    # No stacktrace found for following nodes
    mm_default_52: "f32[1568, 384]" = torch.ops.aten.mm.default(view_190, permute_89);  view_190 = permute_89 = None
    add_tensor_52: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_52, arg138_1);  mm_default_52 = arg138_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    view_191: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_52, [8, 196, 384]);  add_tensor_52 = None
    add_85: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_22, view_191);  slice_22 = view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_5: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_20, add_85], 1);  slice_20 = add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_25 = torch.ops.aten.var_mean.correction(cat_5, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 197, 1]" = var_mean_25[0]
    getitem_69: "f32[8, 197, 1]" = var_mean_25[1];  var_mean_25 = None
    sub_34: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_5, getitem_69);  getitem_69 = None
    add_86: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
    rsqrt_25: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    mul_86: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_25);  sub_34 = rsqrt_25 = None
    mul_87: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_86, arg139_1);  mul_86 = arg139_1 = None
    add_87: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_87, arg140_1);  mul_87 = arg140_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_192: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_87, [1576, 384])
    permute_90: "f32[384, 768]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
    mm_18: "f32[1576, 768]" = torch.ops.aten.mm.default(view_192, permute_90);  view_192 = permute_90 = None
    view_193: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_18, [8, 197, 768]);  mm_18 = None
    view_194: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_193, [8, 197, 2, 6, 64]);  view_193 = None
    permute_91: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_194, [2, 0, 3, 1, 4]);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_9 = torch.ops.aten.unbind.int(permute_91);  permute_91 = None
    getitem_70: "f32[8, 6, 197, 64]" = unbind_9[0]
    getitem_71: "f32[8, 6, 197, 64]" = unbind_9[1];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_37: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_70, [8, 6, 197, 64]);  getitem_70 = None
    clone_73: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_198: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_73, [48, 197, 64]);  clone_73 = None
    permute_94: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_71, [0, 1, 3, 2]);  getitem_71 = None
    expand_38: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_94, [8, 6, 64, 197]);  permute_94 = None
    clone_74: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_199: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_74, [48, 64, 197]);  clone_74 = None
    bmm_18: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_198, view_199);  view_198 = view_199 = None
    view_200: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_18, [8, 6, 197, 197]);  bmm_18 = None
    mul_88: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_200, 0.125);  view_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_9: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_88, [-1], True)
    sub_35: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_88, amax_9);  mul_88 = amax_9 = None
    exp_9: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_35);  sub_35 = None
    sum_10: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_39: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_9, [8, 6, 197, 197]);  div_9 = None
    view_201: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_39, [48, 197, 197]);  expand_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_195: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_87, [1576, 384]);  add_87 = None
    permute_92: "f32[384, 384]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    mm_19: "f32[1576, 384]" = torch.ops.aten.mm.default(view_195, permute_92);  view_195 = permute_92 = None
    view_196: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_19, [8, 197, 384]);  mm_19 = None
    view_197: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_196, [8, 197, 6, -1]);  view_196 = None
    permute_93: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_197, [0, 2, 1, 3]);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_40: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_93, [8, 6, 197, 64]);  permute_93 = None
    clone_75: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_202: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_75, [48, 197, 64]);  clone_75 = None
    bmm_19: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_201, view_202);  view_201 = view_202 = None
    view_203: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_19, [8, 6, 197, 64]);  bmm_19 = None
    permute_95: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    clone_76: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_204: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_76, [8, 197, 384]);  clone_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_205: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_204, [1576, 384]);  view_204 = None
    permute_96: "f32[384, 384]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
    
    # No stacktrace found for following nodes
    mm_default_51: "f32[1576, 384]" = torch.ops.aten.mm.default(view_205, permute_96);  view_205 = permute_96 = None
    add_tensor_51: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_51, arg144_1);  mm_default_51 = arg144_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_206: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_51, [8, 197, 384]);  add_tensor_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_88: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_5, view_206);  cat_5 = view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_26 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 197, 1]" = var_mean_26[0]
    getitem_73: "f32[8, 197, 1]" = var_mean_26[1];  var_mean_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    clone_79: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_82, memory_format = torch.contiguous_format)
    var_mean_27 = torch.ops.aten.var_mean.correction(clone_79, [2], correction = 0, keepdim = True)
    getitem_74: "f32[1568, 16, 1]" = var_mean_27[0]
    getitem_75: "f32[1568, 16, 1]" = var_mean_27[1];  var_mean_27 = None
    sub_37: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_79, getitem_75);  clone_79 = getitem_75 = None
    add_93: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-05);  getitem_74 = None
    rsqrt_27: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    mul_94: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_27);  sub_37 = rsqrt_27 = None
    mul_95: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_94, arg151_1);  mul_94 = arg151_1 = None
    add_94: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_95, arg152_1);  mul_95 = arg152_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_211: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_94, [25088, 24])
    permute_99: "f32[24, 48]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
    mm_20: "f32[25088, 48]" = torch.ops.aten.mm.default(view_211, permute_99);  view_211 = permute_99 = None
    view_212: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_20, [1568, 16, 48]);  mm_20 = None
    view_213: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_212, [1568, 16, 2, 4, 6]);  view_212 = None
    permute_100: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_213, [2, 0, 3, 1, 4]);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_10 = torch.ops.aten.unbind.int(permute_100);  permute_100 = None
    getitem_76: "f32[1568, 4, 16, 6]" = unbind_10[0]
    getitem_77: "f32[1568, 4, 16, 6]" = unbind_10[1];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_41: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_76, [1568, 4, 16, 6]);  getitem_76 = None
    clone_80: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_217: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_80, [6272, 16, 6]);  clone_80 = None
    permute_103: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_77, [0, 1, 3, 2]);  getitem_77 = None
    expand_42: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_103, [1568, 4, 6, 16]);  permute_103 = None
    clone_81: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_218: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_81, [6272, 6, 16]);  clone_81 = None
    bmm_20: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_217, view_218);  view_217 = view_218 = None
    view_219: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_20, [1568, 4, 16, 16]);  bmm_20 = None
    mul_96: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_219, 0.408248290463863);  view_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_10: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_96, [-1], True)
    sub_38: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_96, amax_10);  mul_96 = amax_10 = None
    exp_10: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_38);  sub_38 = None
    sum_11: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_43: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_10, [1568, 4, 16, 16]);  div_10 = None
    view_220: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_43, [6272, 16, 16]);  expand_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_214: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_94, [25088, 24]);  add_94 = None
    permute_101: "f32[24, 24]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    mm_21: "f32[25088, 24]" = torch.ops.aten.mm.default(view_214, permute_101);  view_214 = permute_101 = None
    view_215: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_21, [1568, 16, 24]);  mm_21 = None
    view_216: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_215, [1568, 16, 4, -1]);  view_215 = None
    permute_102: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_44: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_102, [1568, 4, 16, 6]);  permute_102 = None
    clone_82: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_221: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_82, [6272, 16, 6]);  clone_82 = None
    bmm_21: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_220, view_221);  view_220 = view_221 = None
    view_222: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_21, [1568, 4, 16, 6]);  bmm_21 = None
    permute_104: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    clone_83: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_104, memory_format = torch.contiguous_format);  permute_104 = None
    view_223: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_83, [1568, 16, 24]);  clone_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_224: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_223, [25088, 24]);  view_223 = None
    permute_105: "f32[24, 24]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
    
    # No stacktrace found for following nodes
    mm_default_50: "f32[25088, 24]" = torch.ops.aten.mm.default(view_224, permute_105);  view_224 = permute_105 = None
    add_tensor_50: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_50, arg156_1);  mm_default_50 = arg156_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_225: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_50, [1568, 16, 24]);  add_tensor_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_95: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_82, view_225);  add_82 = view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_84: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_95, memory_format = torch.contiguous_format)
    var_mean_28 = torch.ops.aten.var_mean.correction(clone_84, [2], correction = 0, keepdim = True)
    getitem_78: "f32[1568, 16, 1]" = var_mean_28[0]
    getitem_79: "f32[1568, 16, 1]" = var_mean_28[1];  var_mean_28 = None
    sub_39: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_84, getitem_79);  clone_84 = getitem_79 = None
    add_96: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
    rsqrt_28: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    mul_97: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_28);  sub_39 = rsqrt_28 = None
    mul_98: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_97, arg157_1);  mul_97 = arg157_1 = None
    add_97: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_98, arg158_1);  mul_98 = arg158_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_226: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_97, [25088, 24]);  add_97 = None
    permute_106: "f32[24, 96]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
    
    # No stacktrace found for following nodes
    mm_default_49: "f32[25088, 96]" = torch.ops.aten.mm.default(view_226, permute_106);  view_226 = permute_106 = None
    add_tensor_49: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_49, arg160_1);  mm_default_49 = arg160_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_227: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_49, [1568, 16, 96]);  add_tensor_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_99: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_227, 0.5)
    mul_100: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_227, 0.7071067811865476);  view_227 = None
    erf_10: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_100);  mul_100 = None
    add_98: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_101: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_99, add_98);  mul_99 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_228: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_101, [25088, 96]);  mul_101 = None
    permute_107: "f32[96, 24]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
    
    # No stacktrace found for following nodes
    mm_default_48: "f32[25088, 24]" = torch.ops.aten.mm.default(view_228, permute_107);  view_228 = permute_107 = None
    add_tensor_48: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_48, arg162_1);  mm_default_48 = arg162_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_229: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_48, [1568, 16, 24]);  add_tensor_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_99: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_95, view_229);  add_95 = view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_87: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_99, memory_format = torch.contiguous_format)
    var_mean_29 = torch.ops.aten.var_mean.correction(clone_87, [2], correction = 0, keepdim = True)
    getitem_80: "f32[1568, 16, 1]" = var_mean_29[0]
    getitem_81: "f32[1568, 16, 1]" = var_mean_29[1];  var_mean_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    sub_36: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_88, getitem_73);  getitem_73 = None
    add_89: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-05);  getitem_72 = None
    rsqrt_26: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    mul_89: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_26);  sub_36 = rsqrt_26 = None
    mul_90: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_89, arg145_1);  mul_89 = arg145_1 = None
    add_90: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_90, arg146_1);  mul_90 = arg146_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_207: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_90, [1576, 384]);  add_90 = None
    permute_97: "f32[384, 1536]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
    
    # No stacktrace found for following nodes
    mm_default_47: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_207, permute_97);  view_207 = permute_97 = None
    add_tensor_47: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_47, arg148_1);  mm_default_47 = arg148_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_208: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_47, [8, 197, 1536]);  add_tensor_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_91: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_208, 0.5)
    mul_92: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_208, 0.7071067811865476);  view_208 = None
    erf_9: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
    add_91: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_93: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_91, add_91);  mul_91 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_209: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_93, [1576, 1536]);  mul_93 = None
    permute_98: "f32[1536, 384]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
    
    # No stacktrace found for following nodes
    mm_default_46: "f32[1576, 384]" = torch.ops.aten.mm.default(view_209, permute_98);  view_209 = permute_98 = None
    add_tensor_46: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_46, arg150_1);  mm_default_46 = arg150_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_210: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_46, [8, 197, 384]);  add_tensor_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_92: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_88, view_210);  add_88 = view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_24: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_92, 1, 0, 1)
    slice_26: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_92, 1, 1, 9223372036854775807);  add_92 = None
    sub_40: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_87, getitem_81);  clone_87 = getitem_81 = None
    add_100: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
    rsqrt_29: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
    mul_102: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_29);  sub_40 = rsqrt_29 = None
    mul_103: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_102, arg163_1);  mul_102 = arg163_1 = None
    add_101: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_103, arg164_1);  mul_103 = arg164_1 = None
    view_230: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_101, [8, 196, -1]);  add_101 = None
    view_231: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_230, [1568, 384]);  view_230 = None
    permute_108: "f32[384, 384]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
    
    # No stacktrace found for following nodes
    mm_default_45: "f32[1568, 384]" = torch.ops.aten.mm.default(view_231, permute_108);  view_231 = permute_108 = None
    add_tensor_45: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_45, arg166_1);  mm_default_45 = arg166_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    view_232: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_45, [8, 196, 384]);  add_tensor_45 = None
    add_102: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_26, view_232);  slice_26 = view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_6: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_24, add_102], 1);  slice_24 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_30 = torch.ops.aten.var_mean.correction(cat_6, [2], correction = 0, keepdim = True)
    getitem_82: "f32[8, 197, 1]" = var_mean_30[0]
    getitem_83: "f32[8, 197, 1]" = var_mean_30[1];  var_mean_30 = None
    sub_41: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_6, getitem_83);  getitem_83 = None
    add_103: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-05);  getitem_82 = None
    rsqrt_30: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    mul_104: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_30);  sub_41 = rsqrt_30 = None
    mul_105: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_104, arg167_1);  mul_104 = arg167_1 = None
    add_104: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_105, arg168_1);  mul_105 = arg168_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_233: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_104, [1576, 384])
    permute_109: "f32[384, 768]" = torch.ops.aten.permute.default(arg169_1, [1, 0]);  arg169_1 = None
    mm_22: "f32[1576, 768]" = torch.ops.aten.mm.default(view_233, permute_109);  view_233 = permute_109 = None
    view_234: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_22, [8, 197, 768]);  mm_22 = None
    view_235: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_234, [8, 197, 2, 6, 64]);  view_234 = None
    permute_110: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_235, [2, 0, 3, 1, 4]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_11 = torch.ops.aten.unbind.int(permute_110);  permute_110 = None
    getitem_84: "f32[8, 6, 197, 64]" = unbind_11[0]
    getitem_85: "f32[8, 6, 197, 64]" = unbind_11[1];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_45: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_84, [8, 6, 197, 64]);  getitem_84 = None
    clone_88: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_239: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_88, [48, 197, 64]);  clone_88 = None
    permute_113: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_85, [0, 1, 3, 2]);  getitem_85 = None
    expand_46: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_113, [8, 6, 64, 197]);  permute_113 = None
    clone_89: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_240: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_89, [48, 64, 197]);  clone_89 = None
    bmm_22: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_239, view_240);  view_239 = view_240 = None
    view_241: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_22, [8, 6, 197, 197]);  bmm_22 = None
    mul_106: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_241, 0.125);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_11: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_106, [-1], True)
    sub_42: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_106, amax_11);  mul_106 = amax_11 = None
    exp_11: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_42);  sub_42 = None
    sum_12: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_47: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_11, [8, 6, 197, 197]);  div_11 = None
    view_242: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_47, [48, 197, 197]);  expand_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_236: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_104, [1576, 384]);  add_104 = None
    permute_111: "f32[384, 384]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    mm_23: "f32[1576, 384]" = torch.ops.aten.mm.default(view_236, permute_111);  view_236 = permute_111 = None
    view_237: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_23, [8, 197, 384]);  mm_23 = None
    view_238: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_237, [8, 197, 6, -1]);  view_237 = None
    permute_112: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_48: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_112, [8, 6, 197, 64]);  permute_112 = None
    clone_90: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_243: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_90, [48, 197, 64]);  clone_90 = None
    bmm_23: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_242, view_243);  view_242 = view_243 = None
    view_244: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_23, [8, 6, 197, 64]);  bmm_23 = None
    permute_114: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
    clone_91: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_245: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_91, [8, 197, 384]);  clone_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_246: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_245, [1576, 384]);  view_245 = None
    permute_115: "f32[384, 384]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
    
    # No stacktrace found for following nodes
    mm_default_44: "f32[1576, 384]" = torch.ops.aten.mm.default(view_246, permute_115);  view_246 = permute_115 = None
    add_tensor_44: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_44, arg172_1);  mm_default_44 = arg172_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_247: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_44, [8, 197, 384]);  add_tensor_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_105: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_6, view_247);  cat_6 = view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_31 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 197, 1]" = var_mean_31[0]
    getitem_87: "f32[8, 197, 1]" = var_mean_31[1];  var_mean_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    clone_94: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_99, memory_format = torch.contiguous_format)
    var_mean_32 = torch.ops.aten.var_mean.correction(clone_94, [2], correction = 0, keepdim = True)
    getitem_88: "f32[1568, 16, 1]" = var_mean_32[0]
    getitem_89: "f32[1568, 16, 1]" = var_mean_32[1];  var_mean_32 = None
    sub_44: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_94, getitem_89);  clone_94 = getitem_89 = None
    add_110: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
    rsqrt_32: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    mul_112: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_32);  sub_44 = rsqrt_32 = None
    mul_113: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_112, arg179_1);  mul_112 = arg179_1 = None
    add_111: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_113, arg180_1);  mul_113 = arg180_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_252: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_111, [25088, 24])
    permute_118: "f32[24, 48]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
    mm_24: "f32[25088, 48]" = torch.ops.aten.mm.default(view_252, permute_118);  view_252 = permute_118 = None
    view_253: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_24, [1568, 16, 48]);  mm_24 = None
    view_254: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_253, [1568, 16, 2, 4, 6]);  view_253 = None
    permute_119: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_254, [2, 0, 3, 1, 4]);  view_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_12 = torch.ops.aten.unbind.int(permute_119);  permute_119 = None
    getitem_90: "f32[1568, 4, 16, 6]" = unbind_12[0]
    getitem_91: "f32[1568, 4, 16, 6]" = unbind_12[1];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_49: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_90, [1568, 4, 16, 6]);  getitem_90 = None
    clone_95: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
    view_258: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_95, [6272, 16, 6]);  clone_95 = None
    permute_122: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_91, [0, 1, 3, 2]);  getitem_91 = None
    expand_50: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_122, [1568, 4, 6, 16]);  permute_122 = None
    clone_96: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
    view_259: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_96, [6272, 6, 16]);  clone_96 = None
    bmm_24: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_258, view_259);  view_258 = view_259 = None
    view_260: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_24, [1568, 4, 16, 16]);  bmm_24 = None
    mul_114: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_260, 0.408248290463863);  view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_12: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_114, [-1], True)
    sub_45: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_114, amax_12);  mul_114 = amax_12 = None
    exp_12: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
    sum_13: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_12: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_51: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_12, [1568, 4, 16, 16]);  div_12 = None
    view_261: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_51, [6272, 16, 16]);  expand_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_255: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_111, [25088, 24]);  add_111 = None
    permute_120: "f32[24, 24]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    mm_25: "f32[25088, 24]" = torch.ops.aten.mm.default(view_255, permute_120);  view_255 = permute_120 = None
    view_256: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_25, [1568, 16, 24]);  mm_25 = None
    view_257: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_256, [1568, 16, 4, -1]);  view_256 = None
    permute_121: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_257, [0, 2, 1, 3]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_52: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_121, [1568, 4, 16, 6]);  permute_121 = None
    clone_97: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
    view_262: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_97, [6272, 16, 6]);  clone_97 = None
    bmm_25: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_261, view_262);  view_261 = view_262 = None
    view_263: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_25, [1568, 4, 16, 6]);  bmm_25 = None
    permute_123: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_263, [0, 2, 1, 3]);  view_263 = None
    clone_98: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_123, memory_format = torch.contiguous_format);  permute_123 = None
    view_264: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_98, [1568, 16, 24]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_265: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_264, [25088, 24]);  view_264 = None
    permute_124: "f32[24, 24]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
    
    # No stacktrace found for following nodes
    mm_default_43: "f32[25088, 24]" = torch.ops.aten.mm.default(view_265, permute_124);  view_265 = permute_124 = None
    add_tensor_43: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_43, arg184_1);  mm_default_43 = arg184_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_266: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_43, [1568, 16, 24]);  add_tensor_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_112: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_99, view_266);  add_99 = view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_99: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_112, memory_format = torch.contiguous_format)
    var_mean_33 = torch.ops.aten.var_mean.correction(clone_99, [2], correction = 0, keepdim = True)
    getitem_92: "f32[1568, 16, 1]" = var_mean_33[0]
    getitem_93: "f32[1568, 16, 1]" = var_mean_33[1];  var_mean_33 = None
    sub_46: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_99, getitem_93);  clone_99 = getitem_93 = None
    add_113: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-05);  getitem_92 = None
    rsqrt_33: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    mul_115: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_33);  sub_46 = rsqrt_33 = None
    mul_116: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_115, arg185_1);  mul_115 = arg185_1 = None
    add_114: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_116, arg186_1);  mul_116 = arg186_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_267: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_114, [25088, 24]);  add_114 = None
    permute_125: "f32[24, 96]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
    
    # No stacktrace found for following nodes
    mm_default_42: "f32[25088, 96]" = torch.ops.aten.mm.default(view_267, permute_125);  view_267 = permute_125 = None
    add_tensor_42: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_42, arg188_1);  mm_default_42 = arg188_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_268: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_42, [1568, 16, 96]);  add_tensor_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_117: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_268, 0.5)
    mul_118: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_268, 0.7071067811865476);  view_268 = None
    erf_12: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_115: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_119: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_117, add_115);  mul_117 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_269: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_119, [25088, 96]);  mul_119 = None
    permute_126: "f32[96, 24]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
    
    # No stacktrace found for following nodes
    mm_default_41: "f32[25088, 24]" = torch.ops.aten.mm.default(view_269, permute_126);  view_269 = permute_126 = None
    add_tensor_41: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_41, arg190_1);  mm_default_41 = arg190_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_270: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_41, [1568, 16, 24]);  add_tensor_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_116: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_112, view_270);  add_112 = view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_102: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_116, memory_format = torch.contiguous_format)
    var_mean_34 = torch.ops.aten.var_mean.correction(clone_102, [2], correction = 0, keepdim = True)
    getitem_94: "f32[1568, 16, 1]" = var_mean_34[0]
    getitem_95: "f32[1568, 16, 1]" = var_mean_34[1];  var_mean_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    sub_43: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_105, getitem_87);  getitem_87 = None
    add_106: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
    rsqrt_31: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    mul_107: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_31);  sub_43 = rsqrt_31 = None
    mul_108: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_107, arg173_1);  mul_107 = arg173_1 = None
    add_107: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_108, arg174_1);  mul_108 = arg174_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_248: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_107, [1576, 384]);  add_107 = None
    permute_116: "f32[384, 1536]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
    
    # No stacktrace found for following nodes
    mm_default_40: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_248, permute_116);  view_248 = permute_116 = None
    add_tensor_40: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_40, arg176_1);  mm_default_40 = arg176_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_249: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_40, [8, 197, 1536]);  add_tensor_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_109: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_249, 0.5)
    mul_110: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_249, 0.7071067811865476);  view_249 = None
    erf_11: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_110);  mul_110 = None
    add_108: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_111: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_109, add_108);  mul_109 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_250: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_111, [1576, 1536]);  mul_111 = None
    permute_117: "f32[1536, 384]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
    
    # No stacktrace found for following nodes
    mm_default_39: "f32[1576, 384]" = torch.ops.aten.mm.default(view_250, permute_117);  view_250 = permute_117 = None
    add_tensor_39: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_39, arg178_1);  mm_default_39 = arg178_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_251: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_39, [8, 197, 384]);  add_tensor_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_109: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_105, view_251);  add_105 = view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_28: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_109, 1, 0, 1)
    slice_30: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_109, 1, 1, 9223372036854775807);  add_109 = None
    sub_47: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_102, getitem_95);  clone_102 = getitem_95 = None
    add_117: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-05);  getitem_94 = None
    rsqrt_34: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    mul_120: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_34);  sub_47 = rsqrt_34 = None
    mul_121: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_120, arg191_1);  mul_120 = arg191_1 = None
    add_118: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_121, arg192_1);  mul_121 = arg192_1 = None
    view_271: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_118, [8, 196, -1]);  add_118 = None
    view_272: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_271, [1568, 384]);  view_271 = None
    permute_127: "f32[384, 384]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
    
    # No stacktrace found for following nodes
    mm_default_38: "f32[1568, 384]" = torch.ops.aten.mm.default(view_272, permute_127);  view_272 = permute_127 = None
    add_tensor_38: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_38, arg194_1);  mm_default_38 = arg194_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    view_273: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_38, [8, 196, 384]);  add_tensor_38 = None
    add_119: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_30, view_273);  slice_30 = view_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_7: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_28, add_119], 1);  slice_28 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_35 = torch.ops.aten.var_mean.correction(cat_7, [2], correction = 0, keepdim = True)
    getitem_96: "f32[8, 197, 1]" = var_mean_35[0]
    getitem_97: "f32[8, 197, 1]" = var_mean_35[1];  var_mean_35 = None
    sub_48: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_7, getitem_97);  getitem_97 = None
    add_120: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
    rsqrt_35: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    mul_122: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_35);  sub_48 = rsqrt_35 = None
    mul_123: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_122, arg195_1);  mul_122 = arg195_1 = None
    add_121: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_123, arg196_1);  mul_123 = arg196_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_274: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_121, [1576, 384])
    permute_128: "f32[384, 768]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
    mm_26: "f32[1576, 768]" = torch.ops.aten.mm.default(view_274, permute_128);  view_274 = permute_128 = None
    view_275: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_26, [8, 197, 768]);  mm_26 = None
    view_276: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_275, [8, 197, 2, 6, 64]);  view_275 = None
    permute_129: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_276, [2, 0, 3, 1, 4]);  view_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_13 = torch.ops.aten.unbind.int(permute_129);  permute_129 = None
    getitem_98: "f32[8, 6, 197, 64]" = unbind_13[0]
    getitem_99: "f32[8, 6, 197, 64]" = unbind_13[1];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_53: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_98, [8, 6, 197, 64]);  getitem_98 = None
    clone_103: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
    view_280: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_103, [48, 197, 64]);  clone_103 = None
    permute_132: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_99, [0, 1, 3, 2]);  getitem_99 = None
    expand_54: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_132, [8, 6, 64, 197]);  permute_132 = None
    clone_104: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_54, memory_format = torch.contiguous_format);  expand_54 = None
    view_281: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_104, [48, 64, 197]);  clone_104 = None
    bmm_26: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_280, view_281);  view_280 = view_281 = None
    view_282: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_26, [8, 6, 197, 197]);  bmm_26 = None
    mul_124: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_282, 0.125);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_13: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_124, [-1], True)
    sub_49: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_124, amax_13);  mul_124 = amax_13 = None
    exp_13: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
    sum_14: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_13: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_55: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_13, [8, 6, 197, 197]);  div_13 = None
    view_283: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_55, [48, 197, 197]);  expand_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_277: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_121, [1576, 384]);  add_121 = None
    permute_130: "f32[384, 384]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
    mm_27: "f32[1576, 384]" = torch.ops.aten.mm.default(view_277, permute_130);  view_277 = permute_130 = None
    view_278: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_27, [8, 197, 384]);  mm_27 = None
    view_279: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_278, [8, 197, 6, -1]);  view_278 = None
    permute_131: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_279, [0, 2, 1, 3]);  view_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_56: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_131, [8, 6, 197, 64]);  permute_131 = None
    clone_105: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
    view_284: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_105, [48, 197, 64]);  clone_105 = None
    bmm_27: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_283, view_284);  view_283 = view_284 = None
    view_285: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_27, [8, 6, 197, 64]);  bmm_27 = None
    permute_133: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_285, [0, 2, 1, 3]);  view_285 = None
    clone_106: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_133, memory_format = torch.contiguous_format);  permute_133 = None
    view_286: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_106, [8, 197, 384]);  clone_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_287: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_286, [1576, 384]);  view_286 = None
    permute_134: "f32[384, 384]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
    
    # No stacktrace found for following nodes
    mm_default_37: "f32[1576, 384]" = torch.ops.aten.mm.default(view_287, permute_134);  view_287 = permute_134 = None
    add_tensor_37: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_37, arg200_1);  mm_default_37 = arg200_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_288: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_37, [8, 197, 384]);  add_tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_122: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_7, view_288);  cat_7 = view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_36 = torch.ops.aten.var_mean.correction(add_122, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 197, 1]" = var_mean_36[0]
    getitem_101: "f32[8, 197, 1]" = var_mean_36[1];  var_mean_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    clone_109: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_116, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_109, [2], correction = 0, keepdim = True)
    getitem_102: "f32[1568, 16, 1]" = var_mean_37[0]
    getitem_103: "f32[1568, 16, 1]" = var_mean_37[1];  var_mean_37 = None
    sub_51: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_109, getitem_103);  clone_109 = getitem_103 = None
    add_127: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-05);  getitem_102 = None
    rsqrt_37: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    mul_130: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_37);  sub_51 = rsqrt_37 = None
    mul_131: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_130, arg207_1);  mul_130 = arg207_1 = None
    add_128: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_131, arg208_1);  mul_131 = arg208_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_293: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_128, [25088, 24])
    permute_137: "f32[24, 48]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
    mm_28: "f32[25088, 48]" = torch.ops.aten.mm.default(view_293, permute_137);  view_293 = permute_137 = None
    view_294: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_28, [1568, 16, 48]);  mm_28 = None
    view_295: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_294, [1568, 16, 2, 4, 6]);  view_294 = None
    permute_138: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_295, [2, 0, 3, 1, 4]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_14 = torch.ops.aten.unbind.int(permute_138);  permute_138 = None
    getitem_104: "f32[1568, 4, 16, 6]" = unbind_14[0]
    getitem_105: "f32[1568, 4, 16, 6]" = unbind_14[1];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_57: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_104, [1568, 4, 16, 6]);  getitem_104 = None
    clone_110: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
    view_299: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_110, [6272, 16, 6]);  clone_110 = None
    permute_141: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_105, [0, 1, 3, 2]);  getitem_105 = None
    expand_58: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_141, [1568, 4, 6, 16]);  permute_141 = None
    clone_111: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
    view_300: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_111, [6272, 6, 16]);  clone_111 = None
    bmm_28: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_299, view_300);  view_299 = view_300 = None
    view_301: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_28, [1568, 4, 16, 16]);  bmm_28 = None
    mul_132: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_301, 0.408248290463863);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_14: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_132, [-1], True)
    sub_52: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_132, amax_14);  mul_132 = amax_14 = None
    exp_14: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_52);  sub_52 = None
    sum_15: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_14: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_59: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_14, [1568, 4, 16, 16]);  div_14 = None
    view_302: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_59, [6272, 16, 16]);  expand_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_296: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_128, [25088, 24]);  add_128 = None
    permute_139: "f32[24, 24]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
    mm_29: "f32[25088, 24]" = torch.ops.aten.mm.default(view_296, permute_139);  view_296 = permute_139 = None
    view_297: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_29, [1568, 16, 24]);  mm_29 = None
    view_298: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_297, [1568, 16, 4, -1]);  view_297 = None
    permute_140: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_60: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_140, [1568, 4, 16, 6]);  permute_140 = None
    clone_112: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
    view_303: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_112, [6272, 16, 6]);  clone_112 = None
    bmm_29: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_302, view_303);  view_302 = view_303 = None
    view_304: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_29, [1568, 4, 16, 6]);  bmm_29 = None
    permute_142: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
    clone_113: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_142, memory_format = torch.contiguous_format);  permute_142 = None
    view_305: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_113, [1568, 16, 24]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_306: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_305, [25088, 24]);  view_305 = None
    permute_143: "f32[24, 24]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[25088, 24]" = torch.ops.aten.mm.default(view_306, permute_143);  view_306 = permute_143 = None
    add_tensor_36: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_36, arg212_1);  mm_default_36 = arg212_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_307: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_36, [1568, 16, 24]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_129: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_116, view_307);  add_116 = view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_114: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_129, memory_format = torch.contiguous_format)
    var_mean_38 = torch.ops.aten.var_mean.correction(clone_114, [2], correction = 0, keepdim = True)
    getitem_106: "f32[1568, 16, 1]" = var_mean_38[0]
    getitem_107: "f32[1568, 16, 1]" = var_mean_38[1];  var_mean_38 = None
    sub_53: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_114, getitem_107);  clone_114 = getitem_107 = None
    add_130: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
    rsqrt_38: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    mul_133: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_38);  sub_53 = rsqrt_38 = None
    mul_134: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_133, arg213_1);  mul_133 = arg213_1 = None
    add_131: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_134, arg214_1);  mul_134 = arg214_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_308: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_131, [25088, 24]);  add_131 = None
    permute_144: "f32[24, 96]" = torch.ops.aten.permute.default(arg215_1, [1, 0]);  arg215_1 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[25088, 96]" = torch.ops.aten.mm.default(view_308, permute_144);  view_308 = permute_144 = None
    add_tensor_35: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_35, arg216_1);  mm_default_35 = arg216_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_309: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_35, [1568, 16, 96]);  add_tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_135: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_309, 0.5)
    mul_136: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_309, 0.7071067811865476);  view_309 = None
    erf_14: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_136);  mul_136 = None
    add_132: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_137: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_135, add_132);  mul_135 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_310: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_137, [25088, 96]);  mul_137 = None
    permute_145: "f32[96, 24]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[25088, 24]" = torch.ops.aten.mm.default(view_310, permute_145);  view_310 = permute_145 = None
    add_tensor_34: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_34, arg218_1);  mm_default_34 = arg218_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_311: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_34, [1568, 16, 24]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_133: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_129, view_311);  add_129 = view_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_117: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_133, memory_format = torch.contiguous_format)
    var_mean_39 = torch.ops.aten.var_mean.correction(clone_117, [2], correction = 0, keepdim = True)
    getitem_108: "f32[1568, 16, 1]" = var_mean_39[0]
    getitem_109: "f32[1568, 16, 1]" = var_mean_39[1];  var_mean_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    sub_50: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_122, getitem_101);  getitem_101 = None
    add_123: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
    rsqrt_36: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_123);  add_123 = None
    mul_125: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_36);  sub_50 = rsqrt_36 = None
    mul_126: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_125, arg201_1);  mul_125 = arg201_1 = None
    add_124: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_126, arg202_1);  mul_126 = arg202_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_289: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_124, [1576, 384]);  add_124 = None
    permute_135: "f32[384, 1536]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_289, permute_135);  view_289 = permute_135 = None
    add_tensor_33: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_33, arg204_1);  mm_default_33 = arg204_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_290: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 197, 1536]);  add_tensor_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_127: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_290, 0.5)
    mul_128: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_290, 0.7071067811865476);  view_290 = None
    erf_13: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_128);  mul_128 = None
    add_125: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_129: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_127, add_125);  mul_127 = add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_291: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_129, [1576, 1536]);  mul_129 = None
    permute_136: "f32[1536, 384]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[1576, 384]" = torch.ops.aten.mm.default(view_291, permute_136);  view_291 = permute_136 = None
    add_tensor_32: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_32, arg206_1);  mm_default_32 = arg206_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_292: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 197, 384]);  add_tensor_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_126: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_122, view_292);  add_122 = view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_32: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_126, 1, 0, 1)
    slice_34: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_126, 1, 1, 9223372036854775807);  add_126 = None
    sub_54: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_117, getitem_109);  clone_117 = getitem_109 = None
    add_134: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
    rsqrt_39: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    mul_138: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_39);  sub_54 = rsqrt_39 = None
    mul_139: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_138, arg219_1);  mul_138 = arg219_1 = None
    add_135: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_139, arg220_1);  mul_139 = arg220_1 = None
    view_312: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_135, [8, 196, -1]);  add_135 = None
    view_313: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_312, [1568, 384]);  view_312 = None
    permute_146: "f32[384, 384]" = torch.ops.aten.permute.default(arg221_1, [1, 0]);  arg221_1 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[1568, 384]" = torch.ops.aten.mm.default(view_313, permute_146);  view_313 = permute_146 = None
    add_tensor_31: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_31, arg222_1);  mm_default_31 = arg222_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    view_314: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 196, 384]);  add_tensor_31 = None
    add_136: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_34, view_314);  slice_34 = view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_8: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_32, add_136], 1);  slice_32 = add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_40 = torch.ops.aten.var_mean.correction(cat_8, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 197, 1]" = var_mean_40[0]
    getitem_111: "f32[8, 197, 1]" = var_mean_40[1];  var_mean_40 = None
    sub_55: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_8, getitem_111);  getitem_111 = None
    add_137: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
    rsqrt_40: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    mul_140: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_40);  sub_55 = rsqrt_40 = None
    mul_141: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_140, arg223_1);  mul_140 = arg223_1 = None
    add_138: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_141, arg224_1);  mul_141 = arg224_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_315: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_138, [1576, 384])
    permute_147: "f32[384, 768]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
    mm_30: "f32[1576, 768]" = torch.ops.aten.mm.default(view_315, permute_147);  view_315 = permute_147 = None
    view_316: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_30, [8, 197, 768]);  mm_30 = None
    view_317: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_316, [8, 197, 2, 6, 64]);  view_316 = None
    permute_148: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_317, [2, 0, 3, 1, 4]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_15 = torch.ops.aten.unbind.int(permute_148);  permute_148 = None
    getitem_112: "f32[8, 6, 197, 64]" = unbind_15[0]
    getitem_113: "f32[8, 6, 197, 64]" = unbind_15[1];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_61: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_112, [8, 6, 197, 64]);  getitem_112 = None
    clone_118: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
    view_321: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_118, [48, 197, 64]);  clone_118 = None
    permute_151: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_113, [0, 1, 3, 2]);  getitem_113 = None
    expand_62: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_151, [8, 6, 64, 197]);  permute_151 = None
    clone_119: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
    view_322: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_119, [48, 64, 197]);  clone_119 = None
    bmm_30: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_321, view_322);  view_321 = view_322 = None
    view_323: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_30, [8, 6, 197, 197]);  bmm_30 = None
    mul_142: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_323, 0.125);  view_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_15: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_142, [-1], True)
    sub_56: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_142, amax_15);  mul_142 = amax_15 = None
    exp_15: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_56);  sub_56 = None
    sum_16: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_15: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_63: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_15, [8, 6, 197, 197]);  div_15 = None
    view_324: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_63, [48, 197, 197]);  expand_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_318: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_138, [1576, 384]);  add_138 = None
    permute_149: "f32[384, 384]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
    mm_31: "f32[1576, 384]" = torch.ops.aten.mm.default(view_318, permute_149);  view_318 = permute_149 = None
    view_319: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_31, [8, 197, 384]);  mm_31 = None
    view_320: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_319, [8, 197, 6, -1]);  view_319 = None
    permute_150: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_320, [0, 2, 1, 3]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_64: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_150, [8, 6, 197, 64]);  permute_150 = None
    clone_120: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
    view_325: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_120, [48, 197, 64]);  clone_120 = None
    bmm_31: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_324, view_325);  view_324 = view_325 = None
    view_326: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_31, [8, 6, 197, 64]);  bmm_31 = None
    permute_152: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
    clone_121: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_152, memory_format = torch.contiguous_format);  permute_152 = None
    view_327: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_121, [8, 197, 384]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_328: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_327, [1576, 384]);  view_327 = None
    permute_153: "f32[384, 384]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[1576, 384]" = torch.ops.aten.mm.default(view_328, permute_153);  view_328 = permute_153 = None
    add_tensor_30: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_30, arg228_1);  mm_default_30 = arg228_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_329: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 197, 384]);  add_tensor_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_139: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_8, view_329);  cat_8 = view_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_41 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
    getitem_114: "f32[8, 197, 1]" = var_mean_41[0]
    getitem_115: "f32[8, 197, 1]" = var_mean_41[1];  var_mean_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    clone_124: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_133, memory_format = torch.contiguous_format)
    var_mean_42 = torch.ops.aten.var_mean.correction(clone_124, [2], correction = 0, keepdim = True)
    getitem_116: "f32[1568, 16, 1]" = var_mean_42[0]
    getitem_117: "f32[1568, 16, 1]" = var_mean_42[1];  var_mean_42 = None
    sub_58: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_124, getitem_117);  clone_124 = getitem_117 = None
    add_144: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
    rsqrt_42: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    mul_148: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_42);  sub_58 = rsqrt_42 = None
    mul_149: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_148, arg235_1);  mul_148 = arg235_1 = None
    add_145: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_149, arg236_1);  mul_149 = arg236_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_334: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_145, [25088, 24])
    permute_156: "f32[24, 48]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
    mm_32: "f32[25088, 48]" = torch.ops.aten.mm.default(view_334, permute_156);  view_334 = permute_156 = None
    view_335: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_32, [1568, 16, 48]);  mm_32 = None
    view_336: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_335, [1568, 16, 2, 4, 6]);  view_335 = None
    permute_157: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_336, [2, 0, 3, 1, 4]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_16 = torch.ops.aten.unbind.int(permute_157);  permute_157 = None
    getitem_118: "f32[1568, 4, 16, 6]" = unbind_16[0]
    getitem_119: "f32[1568, 4, 16, 6]" = unbind_16[1];  unbind_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_65: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_118, [1568, 4, 16, 6]);  getitem_118 = None
    clone_125: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
    view_340: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_125, [6272, 16, 6]);  clone_125 = None
    permute_160: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_119, [0, 1, 3, 2]);  getitem_119 = None
    expand_66: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_160, [1568, 4, 6, 16]);  permute_160 = None
    clone_126: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_66, memory_format = torch.contiguous_format);  expand_66 = None
    view_341: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_126, [6272, 6, 16]);  clone_126 = None
    bmm_32: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_340, view_341);  view_340 = view_341 = None
    view_342: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_32, [1568, 4, 16, 16]);  bmm_32 = None
    mul_150: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_342, 0.408248290463863);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_16: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_150, [-1], True)
    sub_59: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_150, amax_16);  mul_150 = amax_16 = None
    exp_16: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_59);  sub_59 = None
    sum_17: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_16: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_67: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_16, [1568, 4, 16, 16]);  div_16 = None
    view_343: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_67, [6272, 16, 16]);  expand_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_337: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_145, [25088, 24]);  add_145 = None
    permute_158: "f32[24, 24]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
    mm_33: "f32[25088, 24]" = torch.ops.aten.mm.default(view_337, permute_158);  view_337 = permute_158 = None
    view_338: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_33, [1568, 16, 24]);  mm_33 = None
    view_339: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_338, [1568, 16, 4, -1]);  view_338 = None
    permute_159: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_68: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_159, [1568, 4, 16, 6]);  permute_159 = None
    clone_127: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
    view_344: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_127, [6272, 16, 6]);  clone_127 = None
    bmm_33: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_343, view_344);  view_343 = view_344 = None
    view_345: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_33, [1568, 4, 16, 6]);  bmm_33 = None
    permute_161: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
    clone_128: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_161, memory_format = torch.contiguous_format);  permute_161 = None
    view_346: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_128, [1568, 16, 24]);  clone_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_347: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_346, [25088, 24]);  view_346 = None
    permute_162: "f32[24, 24]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[25088, 24]" = torch.ops.aten.mm.default(view_347, permute_162);  view_347 = permute_162 = None
    add_tensor_29: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_29, arg240_1);  mm_default_29 = arg240_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_348: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_29, [1568, 16, 24]);  add_tensor_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_146: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_133, view_348);  add_133 = view_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_129: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_146, memory_format = torch.contiguous_format)
    var_mean_43 = torch.ops.aten.var_mean.correction(clone_129, [2], correction = 0, keepdim = True)
    getitem_120: "f32[1568, 16, 1]" = var_mean_43[0]
    getitem_121: "f32[1568, 16, 1]" = var_mean_43[1];  var_mean_43 = None
    sub_60: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_129, getitem_121);  clone_129 = getitem_121 = None
    add_147: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
    rsqrt_43: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    mul_151: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_43);  sub_60 = rsqrt_43 = None
    mul_152: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_151, arg241_1);  mul_151 = arg241_1 = None
    add_148: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_152, arg242_1);  mul_152 = arg242_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_349: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_148, [25088, 24]);  add_148 = None
    permute_163: "f32[24, 96]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[25088, 96]" = torch.ops.aten.mm.default(view_349, permute_163);  view_349 = permute_163 = None
    add_tensor_28: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_28, arg244_1);  mm_default_28 = arg244_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_350: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_28, [1568, 16, 96]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_153: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_350, 0.5)
    mul_154: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_350, 0.7071067811865476);  view_350 = None
    erf_16: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_154);  mul_154 = None
    add_149: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_155: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_153, add_149);  mul_153 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_351: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_155, [25088, 96]);  mul_155 = None
    permute_164: "f32[96, 24]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[25088, 24]" = torch.ops.aten.mm.default(view_351, permute_164);  view_351 = permute_164 = None
    add_tensor_27: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_27, arg246_1);  mm_default_27 = arg246_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_352: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_27, [1568, 16, 24]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_150: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_146, view_352);  add_146 = view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_132: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_150, memory_format = torch.contiguous_format)
    var_mean_44 = torch.ops.aten.var_mean.correction(clone_132, [2], correction = 0, keepdim = True)
    getitem_122: "f32[1568, 16, 1]" = var_mean_44[0]
    getitem_123: "f32[1568, 16, 1]" = var_mean_44[1];  var_mean_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    sub_57: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_139, getitem_115);  getitem_115 = None
    add_140: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_114, 1e-05);  getitem_114 = None
    rsqrt_41: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
    mul_143: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_57, rsqrt_41);  sub_57 = rsqrt_41 = None
    mul_144: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_143, arg229_1);  mul_143 = arg229_1 = None
    add_141: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_144, arg230_1);  mul_144 = arg230_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_330: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_141, [1576, 384]);  add_141 = None
    permute_154: "f32[384, 1536]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_330, permute_154);  view_330 = permute_154 = None
    add_tensor_26: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_26, arg232_1);  mm_default_26 = arg232_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_331: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 197, 1536]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_145: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_331, 0.5)
    mul_146: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_331, 0.7071067811865476);  view_331 = None
    erf_15: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_146);  mul_146 = None
    add_142: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_147: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_145, add_142);  mul_145 = add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_332: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_147, [1576, 1536]);  mul_147 = None
    permute_155: "f32[1536, 384]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[1576, 384]" = torch.ops.aten.mm.default(view_332, permute_155);  view_332 = permute_155 = None
    add_tensor_25: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_25, arg234_1);  mm_default_25 = arg234_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_333: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 197, 384]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_143: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_139, view_333);  add_139 = view_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_36: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_143, 1, 0, 1)
    slice_38: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_143, 1, 1, 9223372036854775807);  add_143 = None
    sub_61: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_132, getitem_123);  clone_132 = getitem_123 = None
    add_151: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-05);  getitem_122 = None
    rsqrt_44: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    mul_156: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_44);  sub_61 = rsqrt_44 = None
    mul_157: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_156, arg247_1);  mul_156 = arg247_1 = None
    add_152: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_157, arg248_1);  mul_157 = arg248_1 = None
    view_353: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_152, [8, 196, -1]);  add_152 = None
    view_354: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_353, [1568, 384]);  view_353 = None
    permute_165: "f32[384, 384]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[1568, 384]" = torch.ops.aten.mm.default(view_354, permute_165);  view_354 = permute_165 = None
    add_tensor_24: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_24, arg250_1);  mm_default_24 = arg250_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    view_355: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 196, 384]);  add_tensor_24 = None
    add_153: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_38, view_355);  slice_38 = view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_9: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_36, add_153], 1);  slice_36 = add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_45 = torch.ops.aten.var_mean.correction(cat_9, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 197, 1]" = var_mean_45[0]
    getitem_125: "f32[8, 197, 1]" = var_mean_45[1];  var_mean_45 = None
    sub_62: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_9, getitem_125);  getitem_125 = None
    add_154: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-05);  getitem_124 = None
    rsqrt_45: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    mul_158: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_45);  sub_62 = rsqrt_45 = None
    mul_159: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_158, arg251_1);  mul_158 = arg251_1 = None
    add_155: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_159, arg252_1);  mul_159 = arg252_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_356: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_155, [1576, 384])
    permute_166: "f32[384, 768]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
    mm_34: "f32[1576, 768]" = torch.ops.aten.mm.default(view_356, permute_166);  view_356 = permute_166 = None
    view_357: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_34, [8, 197, 768]);  mm_34 = None
    view_358: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_357, [8, 197, 2, 6, 64]);  view_357 = None
    permute_167: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_358, [2, 0, 3, 1, 4]);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_17 = torch.ops.aten.unbind.int(permute_167);  permute_167 = None
    getitem_126: "f32[8, 6, 197, 64]" = unbind_17[0]
    getitem_127: "f32[8, 6, 197, 64]" = unbind_17[1];  unbind_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_69: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_126, [8, 6, 197, 64]);  getitem_126 = None
    clone_133: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
    view_362: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_133, [48, 197, 64]);  clone_133 = None
    permute_170: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_127, [0, 1, 3, 2]);  getitem_127 = None
    expand_70: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_170, [8, 6, 64, 197]);  permute_170 = None
    clone_134: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
    view_363: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_134, [48, 64, 197]);  clone_134 = None
    bmm_34: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_362, view_363);  view_362 = view_363 = None
    view_364: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_34, [8, 6, 197, 197]);  bmm_34 = None
    mul_160: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_364, 0.125);  view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_17: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_160, [-1], True)
    sub_63: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_160, amax_17);  mul_160 = amax_17 = None
    exp_17: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_63);  sub_63 = None
    sum_18: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_17: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_71: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_17, [8, 6, 197, 197]);  div_17 = None
    view_365: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_71, [48, 197, 197]);  expand_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_359: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_155, [1576, 384]);  add_155 = None
    permute_168: "f32[384, 384]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    mm_35: "f32[1576, 384]" = torch.ops.aten.mm.default(view_359, permute_168);  view_359 = permute_168 = None
    view_360: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_35, [8, 197, 384]);  mm_35 = None
    view_361: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_360, [8, 197, 6, -1]);  view_360 = None
    permute_169: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_361, [0, 2, 1, 3]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_72: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_169, [8, 6, 197, 64]);  permute_169 = None
    clone_135: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
    view_366: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_135, [48, 197, 64]);  clone_135 = None
    bmm_35: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_365, view_366);  view_365 = view_366 = None
    view_367: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_35, [8, 6, 197, 64]);  bmm_35 = None
    permute_171: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_367, [0, 2, 1, 3]);  view_367 = None
    clone_136: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_171, memory_format = torch.contiguous_format);  permute_171 = None
    view_368: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_136, [8, 197, 384]);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_369: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_368, [1576, 384]);  view_368 = None
    permute_172: "f32[384, 384]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[1576, 384]" = torch.ops.aten.mm.default(view_369, permute_172);  view_369 = permute_172 = None
    add_tensor_23: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_23, arg256_1);  mm_default_23 = arg256_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_370: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 197, 384]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_156: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_9, view_370);  cat_9 = view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_46 = torch.ops.aten.var_mean.correction(add_156, [2], correction = 0, keepdim = True)
    getitem_128: "f32[8, 197, 1]" = var_mean_46[0]
    getitem_129: "f32[8, 197, 1]" = var_mean_46[1];  var_mean_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    clone_139: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_150, memory_format = torch.contiguous_format)
    var_mean_47 = torch.ops.aten.var_mean.correction(clone_139, [2], correction = 0, keepdim = True)
    getitem_130: "f32[1568, 16, 1]" = var_mean_47[0]
    getitem_131: "f32[1568, 16, 1]" = var_mean_47[1];  var_mean_47 = None
    sub_65: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_139, getitem_131);  clone_139 = getitem_131 = None
    add_161: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
    rsqrt_47: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    mul_166: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_47);  sub_65 = rsqrt_47 = None
    mul_167: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_166, arg263_1);  mul_166 = arg263_1 = None
    add_162: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_167, arg264_1);  mul_167 = arg264_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_375: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_162, [25088, 24])
    permute_175: "f32[24, 48]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
    mm_36: "f32[25088, 48]" = torch.ops.aten.mm.default(view_375, permute_175);  view_375 = permute_175 = None
    view_376: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_36, [1568, 16, 48]);  mm_36 = None
    view_377: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_376, [1568, 16, 2, 4, 6]);  view_376 = None
    permute_176: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_377, [2, 0, 3, 1, 4]);  view_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_18 = torch.ops.aten.unbind.int(permute_176);  permute_176 = None
    getitem_132: "f32[1568, 4, 16, 6]" = unbind_18[0]
    getitem_133: "f32[1568, 4, 16, 6]" = unbind_18[1];  unbind_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_73: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_132, [1568, 4, 16, 6]);  getitem_132 = None
    clone_140: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
    view_381: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_140, [6272, 16, 6]);  clone_140 = None
    permute_179: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_133, [0, 1, 3, 2]);  getitem_133 = None
    expand_74: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_179, [1568, 4, 6, 16]);  permute_179 = None
    clone_141: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_74, memory_format = torch.contiguous_format);  expand_74 = None
    view_382: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_141, [6272, 6, 16]);  clone_141 = None
    bmm_36: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_381, view_382);  view_381 = view_382 = None
    view_383: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_36, [1568, 4, 16, 16]);  bmm_36 = None
    mul_168: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_383, 0.408248290463863);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_18: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_168, [-1], True)
    sub_66: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_168, amax_18);  mul_168 = amax_18 = None
    exp_18: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_66);  sub_66 = None
    sum_19: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_18: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_75: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_18, [1568, 4, 16, 16]);  div_18 = None
    view_384: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_75, [6272, 16, 16]);  expand_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_378: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_162, [25088, 24]);  add_162 = None
    permute_177: "f32[24, 24]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
    mm_37: "f32[25088, 24]" = torch.ops.aten.mm.default(view_378, permute_177);  view_378 = permute_177 = None
    view_379: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_37, [1568, 16, 24]);  mm_37 = None
    view_380: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_379, [1568, 16, 4, -1]);  view_379 = None
    permute_178: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_380, [0, 2, 1, 3]);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_76: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_178, [1568, 4, 16, 6]);  permute_178 = None
    clone_142: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
    view_385: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_142, [6272, 16, 6]);  clone_142 = None
    bmm_37: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_384, view_385);  view_384 = view_385 = None
    view_386: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_37, [1568, 4, 16, 6]);  bmm_37 = None
    permute_180: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None
    clone_143: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_180, memory_format = torch.contiguous_format);  permute_180 = None
    view_387: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_143, [1568, 16, 24]);  clone_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_388: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_387, [25088, 24]);  view_387 = None
    permute_181: "f32[24, 24]" = torch.ops.aten.permute.default(arg267_1, [1, 0]);  arg267_1 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[25088, 24]" = torch.ops.aten.mm.default(view_388, permute_181);  view_388 = permute_181 = None
    add_tensor_22: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_22, arg268_1);  mm_default_22 = arg268_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_389: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_22, [1568, 16, 24]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_163: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_150, view_389);  add_150 = view_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_144: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_163, memory_format = torch.contiguous_format)
    var_mean_48 = torch.ops.aten.var_mean.correction(clone_144, [2], correction = 0, keepdim = True)
    getitem_134: "f32[1568, 16, 1]" = var_mean_48[0]
    getitem_135: "f32[1568, 16, 1]" = var_mean_48[1];  var_mean_48 = None
    sub_67: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_144, getitem_135);  clone_144 = getitem_135 = None
    add_164: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-05);  getitem_134 = None
    rsqrt_48: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    mul_169: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_48);  sub_67 = rsqrt_48 = None
    mul_170: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_169, arg269_1);  mul_169 = arg269_1 = None
    add_165: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_170, arg270_1);  mul_170 = arg270_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_390: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_165, [25088, 24]);  add_165 = None
    permute_182: "f32[24, 96]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[25088, 96]" = torch.ops.aten.mm.default(view_390, permute_182);  view_390 = permute_182 = None
    add_tensor_21: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_21, arg272_1);  mm_default_21 = arg272_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_391: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_21, [1568, 16, 96]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_171: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_391, 0.5)
    mul_172: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_391, 0.7071067811865476);  view_391 = None
    erf_18: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
    add_166: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_173: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_171, add_166);  mul_171 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_392: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_173, [25088, 96]);  mul_173 = None
    permute_183: "f32[96, 24]" = torch.ops.aten.permute.default(arg273_1, [1, 0]);  arg273_1 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[25088, 24]" = torch.ops.aten.mm.default(view_392, permute_183);  view_392 = permute_183 = None
    add_tensor_20: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_20, arg274_1);  mm_default_20 = arg274_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_393: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_20, [1568, 16, 24]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_167: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_163, view_393);  add_163 = view_393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_147: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format)
    var_mean_49 = torch.ops.aten.var_mean.correction(clone_147, [2], correction = 0, keepdim = True)
    getitem_136: "f32[1568, 16, 1]" = var_mean_49[0]
    getitem_137: "f32[1568, 16, 1]" = var_mean_49[1];  var_mean_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    sub_64: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_156, getitem_129);  getitem_129 = None
    add_157: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
    rsqrt_46: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    mul_161: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_46);  sub_64 = rsqrt_46 = None
    mul_162: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_161, arg257_1);  mul_161 = arg257_1 = None
    add_158: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_162, arg258_1);  mul_162 = arg258_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_371: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_158, [1576, 384]);  add_158 = None
    permute_173: "f32[384, 1536]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_371, permute_173);  view_371 = permute_173 = None
    add_tensor_19: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_19, arg260_1);  mm_default_19 = arg260_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_372: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 197, 1536]);  add_tensor_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_163: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_372, 0.5)
    mul_164: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_372, 0.7071067811865476);  view_372 = None
    erf_17: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_164);  mul_164 = None
    add_159: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_165: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_163, add_159);  mul_163 = add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_373: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_165, [1576, 1536]);  mul_165 = None
    permute_174: "f32[1536, 384]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[1576, 384]" = torch.ops.aten.mm.default(view_373, permute_174);  view_373 = permute_174 = None
    add_tensor_18: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_18, arg262_1);  mm_default_18 = arg262_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_374: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 197, 384]);  add_tensor_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_160: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_156, view_374);  add_156 = view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_40: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_160, 1, 0, 1)
    slice_42: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_160, 1, 1, 9223372036854775807);  add_160 = None
    sub_68: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_147, getitem_137);  clone_147 = getitem_137 = None
    add_168: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
    rsqrt_49: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    mul_174: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_49);  sub_68 = rsqrt_49 = None
    mul_175: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_174, arg275_1);  mul_174 = arg275_1 = None
    add_169: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_175, arg276_1);  mul_175 = arg276_1 = None
    view_394: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_169, [8, 196, -1]);  add_169 = None
    view_395: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_394, [1568, 384]);  view_394 = None
    permute_184: "f32[384, 384]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[1568, 384]" = torch.ops.aten.mm.default(view_395, permute_184);  view_395 = permute_184 = None
    add_tensor_17: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_17, arg278_1);  mm_default_17 = arg278_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    view_396: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 196, 384]);  add_tensor_17 = None
    add_170: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_42, view_396);  slice_42 = view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_10: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_40, add_170], 1);  slice_40 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_50 = torch.ops.aten.var_mean.correction(cat_10, [2], correction = 0, keepdim = True)
    getitem_138: "f32[8, 197, 1]" = var_mean_50[0]
    getitem_139: "f32[8, 197, 1]" = var_mean_50[1];  var_mean_50 = None
    sub_69: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_10, getitem_139);  getitem_139 = None
    add_171: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
    rsqrt_50: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    mul_176: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_50);  sub_69 = rsqrt_50 = None
    mul_177: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_176, arg279_1);  mul_176 = arg279_1 = None
    add_172: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_177, arg280_1);  mul_177 = arg280_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_397: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_172, [1576, 384])
    permute_185: "f32[384, 768]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
    mm_38: "f32[1576, 768]" = torch.ops.aten.mm.default(view_397, permute_185);  view_397 = permute_185 = None
    view_398: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_38, [8, 197, 768]);  mm_38 = None
    view_399: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_398, [8, 197, 2, 6, 64]);  view_398 = None
    permute_186: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_399, [2, 0, 3, 1, 4]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_19 = torch.ops.aten.unbind.int(permute_186);  permute_186 = None
    getitem_140: "f32[8, 6, 197, 64]" = unbind_19[0]
    getitem_141: "f32[8, 6, 197, 64]" = unbind_19[1];  unbind_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_77: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_140, [8, 6, 197, 64]);  getitem_140 = None
    clone_148: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
    view_403: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_148, [48, 197, 64]);  clone_148 = None
    permute_189: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_141, [0, 1, 3, 2]);  getitem_141 = None
    expand_78: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_189, [8, 6, 64, 197]);  permute_189 = None
    clone_149: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
    view_404: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_149, [48, 64, 197]);  clone_149 = None
    bmm_38: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_403, view_404);  view_403 = view_404 = None
    view_405: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_38, [8, 6, 197, 197]);  bmm_38 = None
    mul_178: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_405, 0.125);  view_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_19: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_178, [-1], True)
    sub_70: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_178, amax_19);  mul_178 = amax_19 = None
    exp_19: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_70);  sub_70 = None
    sum_20: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_19: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_79: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_19, [8, 6, 197, 197]);  div_19 = None
    view_406: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_79, [48, 197, 197]);  expand_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_400: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_172, [1576, 384]);  add_172 = None
    permute_187: "f32[384, 384]" = torch.ops.aten.permute.default(arg282_1, [1, 0]);  arg282_1 = None
    mm_39: "f32[1576, 384]" = torch.ops.aten.mm.default(view_400, permute_187);  view_400 = permute_187 = None
    view_401: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_39, [8, 197, 384]);  mm_39 = None
    view_402: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_401, [8, 197, 6, -1]);  view_401 = None
    permute_188: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_80: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_188, [8, 6, 197, 64]);  permute_188 = None
    clone_150: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
    view_407: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_150, [48, 197, 64]);  clone_150 = None
    bmm_39: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_406, view_407);  view_406 = view_407 = None
    view_408: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_39, [8, 6, 197, 64]);  bmm_39 = None
    permute_190: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None
    clone_151: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
    view_409: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_151, [8, 197, 384]);  clone_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_410: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_409, [1576, 384]);  view_409 = None
    permute_191: "f32[384, 384]" = torch.ops.aten.permute.default(arg283_1, [1, 0]);  arg283_1 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[1576, 384]" = torch.ops.aten.mm.default(view_410, permute_191);  view_410 = permute_191 = None
    add_tensor_16: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_16, arg284_1);  mm_default_16 = arg284_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_411: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 197, 384]);  add_tensor_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_173: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_10, view_411);  cat_10 = view_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_51 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
    getitem_142: "f32[8, 197, 1]" = var_mean_51[0]
    getitem_143: "f32[8, 197, 1]" = var_mean_51[1];  var_mean_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    clone_154: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_167, memory_format = torch.contiguous_format)
    var_mean_52 = torch.ops.aten.var_mean.correction(clone_154, [2], correction = 0, keepdim = True)
    getitem_144: "f32[1568, 16, 1]" = var_mean_52[0]
    getitem_145: "f32[1568, 16, 1]" = var_mean_52[1];  var_mean_52 = None
    sub_72: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_154, getitem_145);  clone_154 = getitem_145 = None
    add_178: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-05);  getitem_144 = None
    rsqrt_52: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    mul_184: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_52);  sub_72 = rsqrt_52 = None
    mul_185: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_184, arg291_1);  mul_184 = arg291_1 = None
    add_179: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_185, arg292_1);  mul_185 = arg292_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_416: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_179, [25088, 24])
    permute_194: "f32[24, 48]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
    mm_40: "f32[25088, 48]" = torch.ops.aten.mm.default(view_416, permute_194);  view_416 = permute_194 = None
    view_417: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_40, [1568, 16, 48]);  mm_40 = None
    view_418: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_417, [1568, 16, 2, 4, 6]);  view_417 = None
    permute_195: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_418, [2, 0, 3, 1, 4]);  view_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_20 = torch.ops.aten.unbind.int(permute_195);  permute_195 = None
    getitem_146: "f32[1568, 4, 16, 6]" = unbind_20[0]
    getitem_147: "f32[1568, 4, 16, 6]" = unbind_20[1];  unbind_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_81: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_146, [1568, 4, 16, 6]);  getitem_146 = None
    clone_155: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
    view_422: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_155, [6272, 16, 6]);  clone_155 = None
    permute_198: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_147, [0, 1, 3, 2]);  getitem_147 = None
    expand_82: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_198, [1568, 4, 6, 16]);  permute_198 = None
    clone_156: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
    view_423: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_156, [6272, 6, 16]);  clone_156 = None
    bmm_40: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_422, view_423);  view_422 = view_423 = None
    view_424: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_40, [1568, 4, 16, 16]);  bmm_40 = None
    mul_186: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_424, 0.408248290463863);  view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_20: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_186, [-1], True)
    sub_73: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_186, amax_20);  mul_186 = amax_20 = None
    exp_20: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
    sum_21: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_20: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_83: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_20, [1568, 4, 16, 16]);  div_20 = None
    view_425: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_83, [6272, 16, 16]);  expand_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_419: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_179, [25088, 24]);  add_179 = None
    permute_196: "f32[24, 24]" = torch.ops.aten.permute.default(arg294_1, [1, 0]);  arg294_1 = None
    mm_41: "f32[25088, 24]" = torch.ops.aten.mm.default(view_419, permute_196);  view_419 = permute_196 = None
    view_420: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_41, [1568, 16, 24]);  mm_41 = None
    view_421: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_420, [1568, 16, 4, -1]);  view_420 = None
    permute_197: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_84: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_197, [1568, 4, 16, 6]);  permute_197 = None
    clone_157: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
    view_426: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_157, [6272, 16, 6]);  clone_157 = None
    bmm_41: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_425, view_426);  view_425 = view_426 = None
    view_427: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_41, [1568, 4, 16, 6]);  bmm_41 = None
    permute_199: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_427, [0, 2, 1, 3]);  view_427 = None
    clone_158: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_428: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_158, [1568, 16, 24]);  clone_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_429: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_428, [25088, 24]);  view_428 = None
    permute_200: "f32[24, 24]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[25088, 24]" = torch.ops.aten.mm.default(view_429, permute_200);  view_429 = permute_200 = None
    add_tensor_15: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_15, arg296_1);  mm_default_15 = arg296_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_430: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_15, [1568, 16, 24]);  add_tensor_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_180: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_167, view_430);  add_167 = view_430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_159: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_180, memory_format = torch.contiguous_format)
    var_mean_53 = torch.ops.aten.var_mean.correction(clone_159, [2], correction = 0, keepdim = True)
    getitem_148: "f32[1568, 16, 1]" = var_mean_53[0]
    getitem_149: "f32[1568, 16, 1]" = var_mean_53[1];  var_mean_53 = None
    sub_74: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_159, getitem_149);  clone_159 = getitem_149 = None
    add_181: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
    rsqrt_53: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    mul_187: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_53);  sub_74 = rsqrt_53 = None
    mul_188: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_187, arg297_1);  mul_187 = arg297_1 = None
    add_182: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_188, arg298_1);  mul_188 = arg298_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_431: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_182, [25088, 24]);  add_182 = None
    permute_201: "f32[24, 96]" = torch.ops.aten.permute.default(arg299_1, [1, 0]);  arg299_1 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[25088, 96]" = torch.ops.aten.mm.default(view_431, permute_201);  view_431 = permute_201 = None
    add_tensor_14: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_14, arg300_1);  mm_default_14 = arg300_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_432: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_14, [1568, 16, 96]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_189: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_432, 0.5)
    mul_190: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_432, 0.7071067811865476);  view_432 = None
    erf_20: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_190);  mul_190 = None
    add_183: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_191: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_189, add_183);  mul_189 = add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_433: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_191, [25088, 96]);  mul_191 = None
    permute_202: "f32[96, 24]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[25088, 24]" = torch.ops.aten.mm.default(view_433, permute_202);  view_433 = permute_202 = None
    add_tensor_13: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_13, arg302_1);  mm_default_13 = arg302_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_434: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_13, [1568, 16, 24]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_184: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_180, view_434);  add_180 = view_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_162: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_184, memory_format = torch.contiguous_format)
    var_mean_54 = torch.ops.aten.var_mean.correction(clone_162, [2], correction = 0, keepdim = True)
    getitem_150: "f32[1568, 16, 1]" = var_mean_54[0]
    getitem_151: "f32[1568, 16, 1]" = var_mean_54[1];  var_mean_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    sub_71: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_173, getitem_143);  getitem_143 = None
    add_174: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-05);  getitem_142 = None
    rsqrt_51: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
    mul_179: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_51);  sub_71 = rsqrt_51 = None
    mul_180: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_179, arg285_1);  mul_179 = arg285_1 = None
    add_175: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_180, arg286_1);  mul_180 = arg286_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_412: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_175, [1576, 384]);  add_175 = None
    permute_192: "f32[384, 1536]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_412, permute_192);  view_412 = permute_192 = None
    add_tensor_12: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_12, arg288_1);  mm_default_12 = arg288_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_413: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 197, 1536]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_181: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_413, 0.5)
    mul_182: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_413, 0.7071067811865476);  view_413 = None
    erf_19: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_182);  mul_182 = None
    add_176: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_183: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_181, add_176);  mul_181 = add_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_414: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_183, [1576, 1536]);  mul_183 = None
    permute_193: "f32[1536, 384]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[1576, 384]" = torch.ops.aten.mm.default(view_414, permute_193);  view_414 = permute_193 = None
    add_tensor_11: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_11, arg290_1);  mm_default_11 = arg290_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_415: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 197, 384]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_177: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_173, view_415);  add_173 = view_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_44: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_177, 1, 0, 1)
    slice_46: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_177, 1, 1, 9223372036854775807);  add_177 = None
    sub_75: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_162, getitem_151);  clone_162 = getitem_151 = None
    add_185: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05);  getitem_150 = None
    rsqrt_54: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    mul_192: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_54);  sub_75 = rsqrt_54 = None
    mul_193: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_192, arg303_1);  mul_192 = arg303_1 = None
    add_186: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_193, arg304_1);  mul_193 = arg304_1 = None
    view_435: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_186, [8, 196, -1]);  add_186 = None
    view_436: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_435, [1568, 384]);  view_435 = None
    permute_203: "f32[384, 384]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[1568, 384]" = torch.ops.aten.mm.default(view_436, permute_203);  view_436 = permute_203 = None
    add_tensor_10: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_10, arg306_1);  mm_default_10 = arg306_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    view_437: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 196, 384]);  add_tensor_10 = None
    add_187: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_46, view_437);  slice_46 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_11: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_44, add_187], 1);  slice_44 = add_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_55 = torch.ops.aten.var_mean.correction(cat_11, [2], correction = 0, keepdim = True)
    getitem_152: "f32[8, 197, 1]" = var_mean_55[0]
    getitem_153: "f32[8, 197, 1]" = var_mean_55[1];  var_mean_55 = None
    sub_76: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_11, getitem_153);  getitem_153 = None
    add_188: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-05);  getitem_152 = None
    rsqrt_55: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
    mul_194: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_55);  sub_76 = rsqrt_55 = None
    mul_195: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_194, arg307_1);  mul_194 = arg307_1 = None
    add_189: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_195, arg308_1);  mul_195 = arg308_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_438: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_189, [1576, 384])
    permute_204: "f32[384, 768]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
    mm_42: "f32[1576, 768]" = torch.ops.aten.mm.default(view_438, permute_204);  view_438 = permute_204 = None
    view_439: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_42, [8, 197, 768]);  mm_42 = None
    view_440: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_439, [8, 197, 2, 6, 64]);  view_439 = None
    permute_205: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_440, [2, 0, 3, 1, 4]);  view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_21 = torch.ops.aten.unbind.int(permute_205);  permute_205 = None
    getitem_154: "f32[8, 6, 197, 64]" = unbind_21[0]
    getitem_155: "f32[8, 6, 197, 64]" = unbind_21[1];  unbind_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_85: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_154, [8, 6, 197, 64]);  getitem_154 = None
    clone_163: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
    view_444: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_163, [48, 197, 64]);  clone_163 = None
    permute_208: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_155, [0, 1, 3, 2]);  getitem_155 = None
    expand_86: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_208, [8, 6, 64, 197]);  permute_208 = None
    clone_164: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_86, memory_format = torch.contiguous_format);  expand_86 = None
    view_445: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_164, [48, 64, 197]);  clone_164 = None
    bmm_42: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_444, view_445);  view_444 = view_445 = None
    view_446: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_42, [8, 6, 197, 197]);  bmm_42 = None
    mul_196: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_446, 0.125);  view_446 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_21: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_196, [-1], True)
    sub_77: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_196, amax_21);  mul_196 = amax_21 = None
    exp_21: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_77);  sub_77 = None
    sum_22: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_21: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_87: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_21, [8, 6, 197, 197]);  div_21 = None
    view_447: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_87, [48, 197, 197]);  expand_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_441: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_189, [1576, 384]);  add_189 = None
    permute_206: "f32[384, 384]" = torch.ops.aten.permute.default(arg310_1, [1, 0]);  arg310_1 = None
    mm_43: "f32[1576, 384]" = torch.ops.aten.mm.default(view_441, permute_206);  view_441 = permute_206 = None
    view_442: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_43, [8, 197, 384]);  mm_43 = None
    view_443: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_442, [8, 197, 6, -1]);  view_442 = None
    permute_207: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_443, [0, 2, 1, 3]);  view_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_88: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_207, [8, 6, 197, 64]);  permute_207 = None
    clone_165: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
    view_448: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_165, [48, 197, 64]);  clone_165 = None
    bmm_43: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_447, view_448);  view_447 = view_448 = None
    view_449: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_43, [8, 6, 197, 64]);  bmm_43 = None
    permute_209: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_449, [0, 2, 1, 3]);  view_449 = None
    clone_166: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
    view_450: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_166, [8, 197, 384]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_451: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_450, [1576, 384]);  view_450 = None
    permute_210: "f32[384, 384]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[1576, 384]" = torch.ops.aten.mm.default(view_451, permute_210);  view_451 = permute_210 = None
    add_tensor_9: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_9, arg312_1);  mm_default_9 = arg312_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_452: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 197, 384]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_190: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_11, view_452);  cat_11 = view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_56 = torch.ops.aten.var_mean.correction(add_190, [2], correction = 0, keepdim = True)
    getitem_156: "f32[8, 197, 1]" = var_mean_56[0]
    getitem_157: "f32[8, 197, 1]" = var_mean_56[1];  var_mean_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    clone_169: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_184, memory_format = torch.contiguous_format)
    var_mean_57 = torch.ops.aten.var_mean.correction(clone_169, [2], correction = 0, keepdim = True)
    getitem_158: "f32[1568, 16, 1]" = var_mean_57[0]
    getitem_159: "f32[1568, 16, 1]" = var_mean_57[1];  var_mean_57 = None
    sub_79: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_169, getitem_159);  clone_169 = getitem_159 = None
    add_195: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
    rsqrt_57: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
    mul_202: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_57);  sub_79 = rsqrt_57 = None
    mul_203: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_202, arg319_1);  mul_202 = arg319_1 = None
    add_196: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_203, arg320_1);  mul_203 = arg320_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_457: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_196, [25088, 24])
    permute_213: "f32[24, 48]" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
    mm_44: "f32[25088, 48]" = torch.ops.aten.mm.default(view_457, permute_213);  view_457 = permute_213 = None
    view_458: "f32[1568, 16, 48]" = torch.ops.aten.reshape.default(mm_44, [1568, 16, 48]);  mm_44 = None
    view_459: "f32[1568, 16, 2, 4, 6]" = torch.ops.aten.reshape.default(view_458, [1568, 16, 2, 4, 6]);  view_458 = None
    permute_214: "f32[2, 1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_459, [2, 0, 3, 1, 4]);  view_459 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_22 = torch.ops.aten.unbind.int(permute_214);  permute_214 = None
    getitem_160: "f32[1568, 4, 16, 6]" = unbind_22[0]
    getitem_161: "f32[1568, 4, 16, 6]" = unbind_22[1];  unbind_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_89: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(getitem_160, [1568, 4, 16, 6]);  getitem_160 = None
    clone_170: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
    view_463: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_170, [6272, 16, 6]);  clone_170 = None
    permute_217: "f32[1568, 4, 6, 16]" = torch.ops.aten.permute.default(getitem_161, [0, 1, 3, 2]);  getitem_161 = None
    expand_90: "f32[1568, 4, 6, 16]" = torch.ops.aten.expand.default(permute_217, [1568, 4, 6, 16]);  permute_217 = None
    clone_171: "f32[1568, 4, 6, 16]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
    view_464: "f32[6272, 6, 16]" = torch.ops.aten.reshape.default(clone_171, [6272, 6, 16]);  clone_171 = None
    bmm_44: "f32[6272, 16, 16]" = torch.ops.aten.bmm.default(view_463, view_464);  view_463 = view_464 = None
    view_465: "f32[1568, 4, 16, 16]" = torch.ops.aten.reshape.default(bmm_44, [1568, 4, 16, 16]);  bmm_44 = None
    mul_204: "f32[1568, 4, 16, 16]" = torch.ops.aten.mul.Tensor(view_465, 0.408248290463863);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_22: "f32[1568, 4, 16, 1]" = torch.ops.aten.amax.default(mul_204, [-1], True)
    sub_80: "f32[1568, 4, 16, 16]" = torch.ops.aten.sub.Tensor(mul_204, amax_22);  mul_204 = amax_22 = None
    exp_22: "f32[1568, 4, 16, 16]" = torch.ops.aten.exp.default(sub_80);  sub_80 = None
    sum_23: "f32[1568, 4, 16, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_22: "f32[1568, 4, 16, 16]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_91: "f32[1568, 4, 16, 16]" = torch.ops.aten.expand.default(div_22, [1568, 4, 16, 16]);  div_22 = None
    view_466: "f32[6272, 16, 16]" = torch.ops.aten.reshape.default(expand_91, [6272, 16, 16]);  expand_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_460: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_196, [25088, 24]);  add_196 = None
    permute_215: "f32[24, 24]" = torch.ops.aten.permute.default(arg322_1, [1, 0]);  arg322_1 = None
    mm_45: "f32[25088, 24]" = torch.ops.aten.mm.default(view_460, permute_215);  view_460 = permute_215 = None
    view_461: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(mm_45, [1568, 16, 24]);  mm_45 = None
    view_462: "f32[1568, 16, 4, 6]" = torch.ops.aten.reshape.default(view_461, [1568, 16, 4, -1]);  view_461 = None
    permute_216: "f32[1568, 4, 16, 6]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_92: "f32[1568, 4, 16, 6]" = torch.ops.aten.expand.default(permute_216, [1568, 4, 16, 6]);  permute_216 = None
    clone_172: "f32[1568, 4, 16, 6]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
    view_467: "f32[6272, 16, 6]" = torch.ops.aten.reshape.default(clone_172, [6272, 16, 6]);  clone_172 = None
    bmm_45: "f32[6272, 16, 6]" = torch.ops.aten.bmm.default(view_466, view_467);  view_466 = view_467 = None
    view_468: "f32[1568, 4, 16, 6]" = torch.ops.aten.reshape.default(bmm_45, [1568, 4, 16, 6]);  bmm_45 = None
    permute_218: "f32[1568, 16, 4, 6]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
    clone_173: "f32[1568, 16, 4, 6]" = torch.ops.aten.clone.default(permute_218, memory_format = torch.contiguous_format);  permute_218 = None
    view_469: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(clone_173, [1568, 16, 24]);  clone_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_470: "f32[25088, 24]" = torch.ops.aten.reshape.default(view_469, [25088, 24]);  view_469 = None
    permute_219: "f32[24, 24]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[25088, 24]" = torch.ops.aten.mm.default(view_470, permute_219);  view_470 = permute_219 = None
    add_tensor_8: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_8, arg324_1);  mm_default_8 = arg324_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_471: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_8, [1568, 16, 24]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:144, code: pixel_embed = pixel_embed + self.drop_path(self.attn_in(self.norm_in(pixel_embed)))
    add_197: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_184, view_471);  add_184 = view_471 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    clone_174: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_197, memory_format = torch.contiguous_format)
    var_mean_58 = torch.ops.aten.var_mean.correction(clone_174, [2], correction = 0, keepdim = True)
    getitem_162: "f32[1568, 16, 1]" = var_mean_58[0]
    getitem_163: "f32[1568, 16, 1]" = var_mean_58[1];  var_mean_58 = None
    sub_81: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_174, getitem_163);  clone_174 = getitem_163 = None
    add_198: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-05);  getitem_162 = None
    rsqrt_58: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    mul_205: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_58);  sub_81 = rsqrt_58 = None
    mul_206: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_205, arg325_1);  mul_205 = arg325_1 = None
    add_199: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_206, arg326_1);  mul_206 = arg326_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_472: "f32[25088, 24]" = torch.ops.aten.reshape.default(add_199, [25088, 24]);  add_199 = None
    permute_220: "f32[24, 96]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[25088, 96]" = torch.ops.aten.mm.default(view_472, permute_220);  view_472 = permute_220 = None
    add_tensor_7: "f32[25088, 96]" = torch.ops.aten.add.Tensor(mm_default_7, arg328_1);  mm_default_7 = arg328_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_473: "f32[1568, 16, 96]" = torch.ops.aten.reshape.default(add_tensor_7, [1568, 16, 96]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_207: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_473, 0.5)
    mul_208: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(view_473, 0.7071067811865476);  view_473 = None
    erf_22: "f32[1568, 16, 96]" = torch.ops.aten.erf.default(mul_208);  mul_208 = None
    add_200: "f32[1568, 16, 96]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_209: "f32[1568, 16, 96]" = torch.ops.aten.mul.Tensor(mul_207, add_200);  mul_207 = add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_474: "f32[25088, 96]" = torch.ops.aten.reshape.default(mul_209, [25088, 96]);  mul_209 = None
    permute_221: "f32[96, 24]" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[25088, 24]" = torch.ops.aten.mm.default(view_474, permute_221);  view_474 = permute_221 = None
    add_tensor_6: "f32[25088, 24]" = torch.ops.aten.add.Tensor(mm_default_6, arg330_1);  mm_default_6 = arg330_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_475: "f32[1568, 16, 24]" = torch.ops.aten.reshape.default(add_tensor_6, [1568, 16, 24]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:145, code: pixel_embed = pixel_embed + self.drop_path(self.mlp_in(self.norm_mlp_in(pixel_embed)))
    add_201: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(add_197, view_475);  add_197 = view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    clone_177: "f32[1568, 16, 24]" = torch.ops.aten.clone.default(add_201, memory_format = torch.contiguous_format);  add_201 = None
    var_mean_59 = torch.ops.aten.var_mean.correction(clone_177, [2], correction = 0, keepdim = True)
    getitem_164: "f32[1568, 16, 1]" = var_mean_59[0]
    getitem_165: "f32[1568, 16, 1]" = var_mean_59[1];  var_mean_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    sub_78: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_190, getitem_157);  getitem_157 = None
    add_191: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
    rsqrt_56: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_191);  add_191 = None
    mul_197: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_56);  sub_78 = rsqrt_56 = None
    mul_198: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_197, arg313_1);  mul_197 = arg313_1 = None
    add_192: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_198, arg314_1);  mul_198 = arg314_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_453: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_192, [1576, 384]);  add_192 = None
    permute_211: "f32[384, 1536]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_453, permute_211);  view_453 = permute_211 = None
    add_tensor_5: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_5, arg316_1);  mm_default_5 = arg316_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_454: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 197, 1536]);  add_tensor_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_199: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_454, 0.5)
    mul_200: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_454, 0.7071067811865476);  view_454 = None
    erf_21: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_200);  mul_200 = None
    add_193: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_201: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_199, add_193);  mul_199 = add_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_455: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_201, [1576, 1536]);  mul_201 = None
    permute_212: "f32[1536, 384]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[1576, 384]" = torch.ops.aten.mm.default(view_455, permute_212);  view_455 = permute_212 = None
    add_tensor_4: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_4, arg318_1);  mm_default_4 = arg318_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_456: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 197, 384]);  add_tensor_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_194: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_190, view_456);  add_190 = view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    slice_48: "f32[8, 1, 384]" = torch.ops.aten.slice.Tensor(add_194, 1, 0, 1)
    slice_50: "f32[8, 196, 384]" = torch.ops.aten.slice.Tensor(add_194, 1, 1, 9223372036854775807);  add_194 = None
    sub_82: "f32[1568, 16, 24]" = torch.ops.aten.sub.Tensor(clone_177, getitem_165);  clone_177 = getitem_165 = None
    add_202: "f32[1568, 16, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-05);  getitem_164 = None
    rsqrt_59: "f32[1568, 16, 1]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    mul_210: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_59);  sub_82 = rsqrt_59 = None
    mul_211: "f32[1568, 16, 24]" = torch.ops.aten.mul.Tensor(mul_210, arg331_1);  mul_210 = arg331_1 = None
    add_203: "f32[1568, 16, 24]" = torch.ops.aten.add.Tensor(mul_211, arg332_1);  mul_211 = arg332_1 = None
    view_476: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_203, [8, 196, -1]);  add_203 = None
    view_477: "f32[1568, 384]" = torch.ops.aten.reshape.default(view_476, [1568, 384]);  view_476 = None
    permute_222: "f32[384, 384]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[1568, 384]" = torch.ops.aten.mm.default(view_477, permute_222);  view_477 = permute_222 = None
    add_tensor_3: "f32[1568, 384]" = torch.ops.aten.add.Tensor(mm_default_3, arg334_1);  mm_default_3 = arg334_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:149, code: [patch_embed[:, 0:1], patch_embed[:, 1:] + self.proj(self.norm1_proj(pixel_embed).reshape(B, N - 1, -1))],
    view_478: "f32[8, 196, 384]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 196, 384]);  add_tensor_3 = None
    add_204: "f32[8, 196, 384]" = torch.ops.aten.add.Tensor(slice_50, view_478);  slice_50 = view_478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:148, code: patch_embed = torch.cat(
    cat_12: "f32[8, 197, 384]" = torch.ops.aten.cat.default([slice_48, add_204], 1);  slice_48 = add_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    var_mean_60 = torch.ops.aten.var_mean.correction(cat_12, [2], correction = 0, keepdim = True)
    getitem_166: "f32[8, 197, 1]" = var_mean_60[0]
    getitem_167: "f32[8, 197, 1]" = var_mean_60[1];  var_mean_60 = None
    sub_83: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(cat_12, getitem_167);  getitem_167 = None
    add_205: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
    rsqrt_60: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    mul_212: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_60);  sub_83 = rsqrt_60 = None
    mul_213: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_212, arg335_1);  mul_212 = arg335_1 = None
    add_206: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_213, arg336_1);  mul_213 = arg336_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:65, code: qk = self.qk(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_479: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_206, [1576, 384])
    permute_223: "f32[384, 768]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
    mm_46: "f32[1576, 768]" = torch.ops.aten.mm.default(view_479, permute_223);  view_479 = permute_223 = None
    view_480: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(mm_46, [8, 197, 768]);  mm_46 = None
    view_481: "f32[8, 197, 2, 6, 64]" = torch.ops.aten.reshape.default(view_480, [8, 197, 2, 6, 64]);  view_480 = None
    permute_224: "f32[2, 8, 6, 197, 64]" = torch.ops.aten.permute.default(view_481, [2, 0, 3, 1, 4]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:66, code: q, k = qk.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
    unbind_23 = torch.ops.aten.unbind.int(permute_224);  permute_224 = None
    getitem_168: "f32[8, 6, 197, 64]" = unbind_23[0]
    getitem_169: "f32[8, 6, 197, 64]" = unbind_23[1];  unbind_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:69, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    expand_93: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(getitem_168, [8, 6, 197, 64]);  getitem_168 = None
    clone_178: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
    view_485: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_178, [48, 197, 64]);  clone_178 = None
    permute_227: "f32[8, 6, 64, 197]" = torch.ops.aten.permute.default(getitem_169, [0, 1, 3, 2]);  getitem_169 = None
    expand_94: "f32[8, 6, 64, 197]" = torch.ops.aten.expand.default(permute_227, [8, 6, 64, 197]);  permute_227 = None
    clone_179: "f32[8, 6, 64, 197]" = torch.ops.aten.clone.default(expand_94, memory_format = torch.contiguous_format);  expand_94 = None
    view_486: "f32[48, 64, 197]" = torch.ops.aten.reshape.default(clone_179, [48, 64, 197]);  clone_179 = None
    bmm_46: "f32[48, 197, 197]" = torch.ops.aten.bmm.default(view_485, view_486);  view_485 = view_486 = None
    view_487: "f32[8, 6, 197, 197]" = torch.ops.aten.reshape.default(bmm_46, [8, 6, 197, 197]);  bmm_46 = None
    mul_214: "f32[8, 6, 197, 197]" = torch.ops.aten.mul.Tensor(view_487, 0.125);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:70, code: attn = attn.softmax(dim=-1)
    amax_23: "f32[8, 6, 197, 1]" = torch.ops.aten.amax.default(mul_214, [-1], True)
    sub_84: "f32[8, 6, 197, 197]" = torch.ops.aten.sub.Tensor(mul_214, amax_23);  mul_214 = amax_23 = None
    exp_23: "f32[8, 6, 197, 197]" = torch.ops.aten.exp.default(sub_84);  sub_84 = None
    sum_24: "f32[8, 6, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_23: "f32[8, 6, 197, 197]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_95: "f32[8, 6, 197, 197]" = torch.ops.aten.expand.default(div_23, [8, 6, 197, 197]);  div_23 = None
    view_488: "f32[48, 197, 197]" = torch.ops.aten.reshape.default(expand_95, [48, 197, 197]);  expand_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:67, code: v = self.v(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
    view_482: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_206, [1576, 384]);  add_206 = None
    permute_225: "f32[384, 384]" = torch.ops.aten.permute.default(arg338_1, [1, 0]);  arg338_1 = None
    mm_47: "f32[1576, 384]" = torch.ops.aten.mm.default(view_482, permute_225);  view_482 = permute_225 = None
    view_483: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(mm_47, [8, 197, 384]);  mm_47 = None
    view_484: "f32[8, 197, 6, 64]" = torch.ops.aten.reshape.default(view_483, [8, 197, 6, -1]);  view_483 = None
    permute_226: "f32[8, 6, 197, 64]" = torch.ops.aten.permute.default(view_484, [0, 2, 1, 3]);  view_484 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:73, code: x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    expand_96: "f32[8, 6, 197, 64]" = torch.ops.aten.expand.default(permute_226, [8, 6, 197, 64]);  permute_226 = None
    clone_180: "f32[8, 6, 197, 64]" = torch.ops.aten.clone.default(expand_96, memory_format = torch.contiguous_format);  expand_96 = None
    view_489: "f32[48, 197, 64]" = torch.ops.aten.reshape.default(clone_180, [48, 197, 64]);  clone_180 = None
    bmm_47: "f32[48, 197, 64]" = torch.ops.aten.bmm.default(view_488, view_489);  view_488 = view_489 = None
    view_490: "f32[8, 6, 197, 64]" = torch.ops.aten.reshape.default(bmm_47, [8, 6, 197, 64]);  bmm_47 = None
    permute_228: "f32[8, 197, 6, 64]" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None
    clone_181: "f32[8, 197, 6, 64]" = torch.ops.aten.clone.default(permute_228, memory_format = torch.contiguous_format);  permute_228 = None
    view_491: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(clone_181, [8, 197, 384]);  clone_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_492: "f32[1576, 384]" = torch.ops.aten.reshape.default(view_491, [1576, 384]);  view_491 = None
    permute_229: "f32[384, 384]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[1576, 384]" = torch.ops.aten.mm.default(view_492, permute_229);  view_492 = permute_229 = None
    add_tensor_2: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default_2, arg340_1);  mm_default_2 = arg340_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:74, code: x = self.proj(x)
    view_493: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 197, 384]);  add_tensor_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:151, code: patch_embed = patch_embed + self.drop_path(self.attn_out(self.norm_out(patch_embed)))
    add_207: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(cat_12, view_493);  cat_12 = view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    var_mean_61 = torch.ops.aten.var_mean.correction(add_207, [2], correction = 0, keepdim = True)
    getitem_170: "f32[8, 197, 1]" = var_mean_61[0]
    getitem_171: "f32[8, 197, 1]" = var_mean_61[1];  var_mean_61 = None
    sub_85: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_207, getitem_171);  getitem_171 = None
    add_208: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-05);  getitem_170 = None
    rsqrt_61: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    mul_215: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_61);  sub_85 = rsqrt_61 = None
    mul_216: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_215, arg341_1);  mul_215 = arg341_1 = None
    add_209: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_216, arg342_1);  mul_216 = arg342_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_494: "f32[1576, 384]" = torch.ops.aten.reshape.default(add_209, [1576, 384]);  add_209 = None
    permute_230: "f32[384, 1536]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[1576, 1536]" = torch.ops.aten.mm.default(view_494, permute_230);  view_494 = permute_230 = None
    add_tensor_1: "f32[1576, 1536]" = torch.ops.aten.add.Tensor(mm_default_1, arg344_1);  mm_default_1 = arg344_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_495: "f32[8, 197, 1536]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 197, 1536]);  add_tensor_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_217: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_495, 0.5)
    mul_218: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(view_495, 0.7071067811865476);  view_495 = None
    erf_23: "f32[8, 197, 1536]" = torch.ops.aten.erf.default(mul_218);  mul_218 = None
    add_210: "f32[8, 197, 1536]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_219: "f32[8, 197, 1536]" = torch.ops.aten.mul.Tensor(mul_217, add_210);  mul_217 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_496: "f32[1576, 1536]" = torch.ops.aten.reshape.default(mul_219, [1576, 1536]);  mul_219 = None
    permute_231: "f32[1536, 384]" = torch.ops.aten.permute.default(arg345_1, [1, 0]);  arg345_1 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[1576, 384]" = torch.ops.aten.mm.default(view_496, permute_231);  view_496 = permute_231 = None
    add_tensor: "f32[1576, 384]" = torch.ops.aten.add.Tensor(mm_default, arg346_1);  mm_default = arg346_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_497: "f32[8, 197, 384]" = torch.ops.aten.reshape.default(add_tensor, [8, 197, 384]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:152, code: patch_embed = patch_embed + self.drop_path(self.mlp(self.norm_mlp(patch_embed)))
    add_211: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(add_207, view_497);  add_207 = view_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:323, code: patch_embed = self.norm(patch_embed)
    var_mean_62 = torch.ops.aten.var_mean.correction(add_211, [2], correction = 0, keepdim = True)
    getitem_172: "f32[8, 197, 1]" = var_mean_62[0]
    getitem_173: "f32[8, 197, 1]" = var_mean_62[1];  var_mean_62 = None
    sub_86: "f32[8, 197, 384]" = torch.ops.aten.sub.Tensor(add_211, getitem_173);  add_211 = getitem_173 = None
    add_212: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_172, 1e-05);  getitem_172 = None
    rsqrt_62: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    mul_220: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_62);  sub_86 = rsqrt_62 = None
    mul_221: "f32[8, 197, 384]" = torch.ops.aten.mul.Tensor(mul_220, arg347_1);  mul_220 = arg347_1 = None
    add_213: "f32[8, 197, 384]" = torch.ops.aten.add.Tensor(mul_221, arg348_1);  mul_221 = arg348_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:328, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    select: "f32[8, 384]" = torch.ops.aten.select.int(add_213, 1, 0);  add_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:329, code: x = self.head_drop(x)
    clone_184: "f32[8, 384]" = torch.ops.aten.clone.default(select);  select = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/tnt.py:330, code: return x if pre_logits else self.head(x)
    permute_232: "f32[384, 1000]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
    addmm_85: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg350_1, clone_184, permute_232);  arg350_1 = clone_184 = permute_232 = None
    return (addmm_85,)
    