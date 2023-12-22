from __future__ import annotations



def forward(self, arg0_1: "f32[1, 1, 128]", arg1_1: "f32[1, 401, 128]", arg2_1: "f32[1, 1, 256]", arg3_1: "f32[1, 197, 256]", arg4_1: "f32[128, 3, 12, 12]", arg5_1: "f32[128]", arg6_1: "f32[256, 3, 16, 16]", arg7_1: "f32[256]", arg8_1: "f32[128]", arg9_1: "f32[128]", arg10_1: "f32[384, 128]", arg11_1: "f32[384]", arg12_1: "f32[128, 128]", arg13_1: "f32[128]", arg14_1: "f32[128]", arg15_1: "f32[128]", arg16_1: "f32[384, 128]", arg17_1: "f32[384]", arg18_1: "f32[128, 384]", arg19_1: "f32[128]", arg20_1: "f32[256]", arg21_1: "f32[256]", arg22_1: "f32[768, 256]", arg23_1: "f32[768]", arg24_1: "f32[256, 256]", arg25_1: "f32[256]", arg26_1: "f32[256]", arg27_1: "f32[256]", arg28_1: "f32[768, 256]", arg29_1: "f32[768]", arg30_1: "f32[256, 768]", arg31_1: "f32[256]", arg32_1: "f32[256]", arg33_1: "f32[256]", arg34_1: "f32[768, 256]", arg35_1: "f32[768]", arg36_1: "f32[256, 256]", arg37_1: "f32[256]", arg38_1: "f32[256]", arg39_1: "f32[256]", arg40_1: "f32[768, 256]", arg41_1: "f32[768]", arg42_1: "f32[256, 768]", arg43_1: "f32[256]", arg44_1: "f32[256]", arg45_1: "f32[256]", arg46_1: "f32[768, 256]", arg47_1: "f32[768]", arg48_1: "f32[256, 256]", arg49_1: "f32[256]", arg50_1: "f32[256]", arg51_1: "f32[256]", arg52_1: "f32[768, 256]", arg53_1: "f32[768]", arg54_1: "f32[256, 768]", arg55_1: "f32[256]", arg56_1: "f32[128]", arg57_1: "f32[128]", arg58_1: "f32[256, 128]", arg59_1: "f32[256]", arg60_1: "f32[256]", arg61_1: "f32[256]", arg62_1: "f32[128, 256]", arg63_1: "f32[128]", arg64_1: "f32[256]", arg65_1: "f32[256]", arg66_1: "f32[256, 256]", arg67_1: "f32[256]", arg68_1: "f32[256, 256]", arg69_1: "f32[256]", arg70_1: "f32[256, 256]", arg71_1: "f32[256]", arg72_1: "f32[256, 256]", arg73_1: "f32[256]", arg74_1: "f32[256]", arg75_1: "f32[256]", arg76_1: "f32[128, 256]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "f32[128]", arg80_1: "f32[128, 128]", arg81_1: "f32[128]", arg82_1: "f32[128, 128]", arg83_1: "f32[128]", arg84_1: "f32[128, 128]", arg85_1: "f32[128]", arg86_1: "f32[128, 128]", arg87_1: "f32[128]", arg88_1: "f32[128]", arg89_1: "f32[128]", arg90_1: "f32[256, 128]", arg91_1: "f32[256]", arg92_1: "f32[128]", arg93_1: "f32[128]", arg94_1: "f32[384, 128]", arg95_1: "f32[384]", arg96_1: "f32[128, 128]", arg97_1: "f32[128]", arg98_1: "f32[128]", arg99_1: "f32[128]", arg100_1: "f32[384, 128]", arg101_1: "f32[384]", arg102_1: "f32[128, 384]", arg103_1: "f32[128]", arg104_1: "f32[256]", arg105_1: "f32[256]", arg106_1: "f32[768, 256]", arg107_1: "f32[768]", arg108_1: "f32[256, 256]", arg109_1: "f32[256]", arg110_1: "f32[256]", arg111_1: "f32[256]", arg112_1: "f32[768, 256]", arg113_1: "f32[768]", arg114_1: "f32[256, 768]", arg115_1: "f32[256]", arg116_1: "f32[256]", arg117_1: "f32[256]", arg118_1: "f32[768, 256]", arg119_1: "f32[768]", arg120_1: "f32[256, 256]", arg121_1: "f32[256]", arg122_1: "f32[256]", arg123_1: "f32[256]", arg124_1: "f32[768, 256]", arg125_1: "f32[768]", arg126_1: "f32[256, 768]", arg127_1: "f32[256]", arg128_1: "f32[256]", arg129_1: "f32[256]", arg130_1: "f32[768, 256]", arg131_1: "f32[768]", arg132_1: "f32[256, 256]", arg133_1: "f32[256]", arg134_1: "f32[256]", arg135_1: "f32[256]", arg136_1: "f32[768, 256]", arg137_1: "f32[768]", arg138_1: "f32[256, 768]", arg139_1: "f32[256]", arg140_1: "f32[128]", arg141_1: "f32[128]", arg142_1: "f32[256, 128]", arg143_1: "f32[256]", arg144_1: "f32[256]", arg145_1: "f32[256]", arg146_1: "f32[128, 256]", arg147_1: "f32[128]", arg148_1: "f32[256]", arg149_1: "f32[256]", arg150_1: "f32[256, 256]", arg151_1: "f32[256]", arg152_1: "f32[256, 256]", arg153_1: "f32[256]", arg154_1: "f32[256, 256]", arg155_1: "f32[256]", arg156_1: "f32[256, 256]", arg157_1: "f32[256]", arg158_1: "f32[256]", arg159_1: "f32[256]", arg160_1: "f32[128, 256]", arg161_1: "f32[128]", arg162_1: "f32[128]", arg163_1: "f32[128]", arg164_1: "f32[128, 128]", arg165_1: "f32[128]", arg166_1: "f32[128, 128]", arg167_1: "f32[128]", arg168_1: "f32[128, 128]", arg169_1: "f32[128]", arg170_1: "f32[128, 128]", arg171_1: "f32[128]", arg172_1: "f32[128]", arg173_1: "f32[128]", arg174_1: "f32[256, 128]", arg175_1: "f32[256]", arg176_1: "f32[128]", arg177_1: "f32[128]", arg178_1: "f32[384, 128]", arg179_1: "f32[384]", arg180_1: "f32[128, 128]", arg181_1: "f32[128]", arg182_1: "f32[128]", arg183_1: "f32[128]", arg184_1: "f32[384, 128]", arg185_1: "f32[384]", arg186_1: "f32[128, 384]", arg187_1: "f32[128]", arg188_1: "f32[256]", arg189_1: "f32[256]", arg190_1: "f32[768, 256]", arg191_1: "f32[768]", arg192_1: "f32[256, 256]", arg193_1: "f32[256]", arg194_1: "f32[256]", arg195_1: "f32[256]", arg196_1: "f32[768, 256]", arg197_1: "f32[768]", arg198_1: "f32[256, 768]", arg199_1: "f32[256]", arg200_1: "f32[256]", arg201_1: "f32[256]", arg202_1: "f32[768, 256]", arg203_1: "f32[768]", arg204_1: "f32[256, 256]", arg205_1: "f32[256]", arg206_1: "f32[256]", arg207_1: "f32[256]", arg208_1: "f32[768, 256]", arg209_1: "f32[768]", arg210_1: "f32[256, 768]", arg211_1: "f32[256]", arg212_1: "f32[256]", arg213_1: "f32[256]", arg214_1: "f32[768, 256]", arg215_1: "f32[768]", arg216_1: "f32[256, 256]", arg217_1: "f32[256]", arg218_1: "f32[256]", arg219_1: "f32[256]", arg220_1: "f32[768, 256]", arg221_1: "f32[768]", arg222_1: "f32[256, 768]", arg223_1: "f32[256]", arg224_1: "f32[128]", arg225_1: "f32[128]", arg226_1: "f32[256, 128]", arg227_1: "f32[256]", arg228_1: "f32[256]", arg229_1: "f32[256]", arg230_1: "f32[128, 256]", arg231_1: "f32[128]", arg232_1: "f32[256]", arg233_1: "f32[256]", arg234_1: "f32[256, 256]", arg235_1: "f32[256]", arg236_1: "f32[256, 256]", arg237_1: "f32[256]", arg238_1: "f32[256, 256]", arg239_1: "f32[256]", arg240_1: "f32[256, 256]", arg241_1: "f32[256]", arg242_1: "f32[256]", arg243_1: "f32[256]", arg244_1: "f32[128, 256]", arg245_1: "f32[128]", arg246_1: "f32[128]", arg247_1: "f32[128]", arg248_1: "f32[128, 128]", arg249_1: "f32[128]", arg250_1: "f32[128, 128]", arg251_1: "f32[128]", arg252_1: "f32[128, 128]", arg253_1: "f32[128]", arg254_1: "f32[128, 128]", arg255_1: "f32[128]", arg256_1: "f32[128]", arg257_1: "f32[128]", arg258_1: "f32[256, 128]", arg259_1: "f32[256]", arg260_1: "f32[128]", arg261_1: "f32[128]", arg262_1: "f32[256]", arg263_1: "f32[256]", arg264_1: "f32[1000, 128]", arg265_1: "f32[1000]", arg266_1: "f32[1000, 256]", arg267_1: "f32[1000]", arg268_1: "f32[8, 3, 240, 240]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:84, code: x = self.proj(x).flatten(2).transpose(1, 2)
    convolution: "f32[8, 128, 20, 20]" = torch.ops.aten.convolution.default(arg268_1, arg4_1, arg5_1, [12, 12], [0, 0], [1, 1], False, [0, 0], 1);  arg4_1 = arg5_1 = None
    view: "f32[8, 128, 400]" = torch.ops.aten.view.default(convolution, [8, 128, 400]);  convolution = None
    permute: "f32[8, 400, 128]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:440, code: cls_tokens = cls_tokens.expand(B, -1, -1)
    expand: "f32[8, 1, 128]" = torch.ops.aten.expand.default(arg0_1, [8, -1, -1]);  arg0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:441, code: x_ = torch.cat((cls_tokens, x_), dim=1)
    cat: "f32[8, 401, 128]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:443, code: x_ = x_ + pos_embed
    add: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(cat, arg1_1);  cat = arg1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:444, code: x_ = self.pos_drop(x_)
    clone: "f32[8, 401, 128]" = torch.ops.aten.clone.default(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:292, code: x = torch.nn.functional.interpolate(x, size=ss, mode='bicubic', align_corners=False)
    iota: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    view_1: "i64[8, 1, 1, 1]" = torch.ops.aten.view.default(iota, [8, 1, 1, 1]);  iota = None
    iota_1: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    view_2: "i64[1, 3, 1, 1]" = torch.ops.aten.view.default(iota_1, [1, 3, 1, 1]);  iota_1 = None
    iota_2: "i64[224]" = torch.ops.prims.iota.default(224, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    view_3: "i64[1, 1, 224, 1]" = torch.ops.aten.view.default(iota_2, [1, 1, 224, 1]);  iota_2 = None
    iota_3: "i64[224]" = torch.ops.prims.iota.default(224, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    view_4: "i64[1, 1, 1, 224]" = torch.ops.aten.view.default(iota_3, [1, 1, 1, 224]);  iota_3 = None
    add_1: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(view_4, 0.5);  view_4 = None
    mul: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_1, 1.0714285714285714);  add_1 = None
    sub: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul, 0.5);  mul = None
    floor: "f32[1, 1, 1, 224]" = torch.ops.aten.floor.default(sub)
    sub_1: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(sub, floor);  sub = None
    convert_element_type: "i64[1, 1, 1, 224]" = torch.ops.prims.convert_element_type.default(floor, torch.int64);  floor = None
    add_2: "f32[1, 1, 224, 1]" = torch.ops.aten.add.Tensor(view_3, 0.5);  view_3 = None
    mul_1: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(add_2, 1.0714285714285714);  add_2 = None
    sub_2: "f32[1, 1, 224, 1]" = torch.ops.aten.sub.Tensor(mul_1, 0.5);  mul_1 = None
    floor_1: "f32[1, 1, 224, 1]" = torch.ops.aten.floor.default(sub_2)
    sub_3: "f32[1, 1, 224, 1]" = torch.ops.aten.sub.Tensor(sub_2, floor_1);  sub_2 = None
    convert_element_type_1: "i64[1, 1, 224, 1]" = torch.ops.prims.convert_element_type.default(floor_1, torch.int64);  floor_1 = None
    sub_4: "i64[1, 1, 224, 1]" = torch.ops.aten.sub.Tensor(convert_element_type_1, 1)
    add_3: "i64[1, 1, 224, 1]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1)
    add_4: "i64[1, 1, 224, 1]" = torch.ops.aten.add.Tensor(convert_element_type_1, 2)
    sub_5: "i64[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(convert_element_type, 1)
    add_5: "i64[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(convert_element_type, 1)
    add_6: "i64[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(convert_element_type, 2)
    clamp_min: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(sub_4, 0)
    clamp_max: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min, 239);  clamp_min = None
    clamp_min_1: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(sub_5, 0)
    clamp_max_1: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_1, 239);  clamp_min_1 = None
    _unsafe_index: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max, clamp_max_1]);  clamp_max = clamp_max_1 = None
    clamp_min_2: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(sub_4, 0)
    clamp_max_2: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_2, 239);  clamp_min_2 = None
    clamp_min_3: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(convert_element_type, 0)
    clamp_max_3: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_3, 239);  clamp_min_3 = None
    _unsafe_index_1: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_2, clamp_max_3]);  clamp_max_2 = clamp_max_3 = None
    clamp_min_4: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(sub_4, 0)
    clamp_max_4: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_4, 239);  clamp_min_4 = None
    clamp_min_5: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_5, 0)
    clamp_max_5: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_5, 239);  clamp_min_5 = None
    _unsafe_index_2: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_4, clamp_max_5]);  clamp_max_4 = clamp_max_5 = None
    clamp_min_6: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(sub_4, 0);  sub_4 = None
    clamp_max_6: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_6, 239);  clamp_min_6 = None
    clamp_min_7: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_6, 0)
    clamp_max_7: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_7, 239);  clamp_min_7 = None
    _unsafe_index_3: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_6, clamp_max_7]);  clamp_max_6 = clamp_max_7 = None
    add_7: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(sub_1, 1.0)
    mul_2: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_7, -0.75)
    sub_6: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_2, -3.75);  mul_2 = None
    mul_3: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_6, add_7);  sub_6 = None
    add_8: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_3, -6.0);  mul_3 = None
    mul_4: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_8, add_7);  add_8 = add_7 = None
    sub_7: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_4, -3.0);  mul_4 = None
    mul_5: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_1, 1.25)
    sub_8: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_5, 2.25);  mul_5 = None
    mul_6: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_8, sub_1);  sub_8 = None
    mul_7: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_6, sub_1);  mul_6 = None
    add_9: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_7, 1);  mul_7 = None
    sub_9: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(1.0, sub_1)
    mul_8: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_9, 1.25)
    sub_10: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_8, 2.25);  mul_8 = None
    mul_9: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_10, sub_9);  sub_10 = None
    mul_10: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_9, sub_9);  mul_9 = sub_9 = None
    add_10: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_10, 1);  mul_10 = None
    sub_11: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(2.0, sub_1)
    mul_11: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_11, -0.75)
    sub_12: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_11, -3.75);  mul_11 = None
    mul_12: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_12, sub_11);  sub_12 = None
    add_11: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_12, -6.0);  mul_12 = None
    mul_13: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_11, sub_11);  add_11 = sub_11 = None
    sub_13: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_13, -3.0);  mul_13 = None
    mul_14: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index, sub_7);  _unsafe_index = sub_7 = None
    mul_15: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_1, add_9);  _unsafe_index_1 = add_9 = None
    add_12: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None
    mul_16: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_2, add_10);  _unsafe_index_2 = add_10 = None
    add_13: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_12, mul_16);  add_12 = mul_16 = None
    mul_17: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_3, sub_13);  _unsafe_index_3 = sub_13 = None
    add_14: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_13, mul_17);  add_13 = mul_17 = None
    clamp_min_8: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(convert_element_type_1, 0)
    clamp_max_8: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_8, 239);  clamp_min_8 = None
    clamp_min_9: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(sub_5, 0)
    clamp_max_9: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_9, 239);  clamp_min_9 = None
    _unsafe_index_4: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_8, clamp_max_9]);  clamp_max_8 = clamp_max_9 = None
    clamp_min_10: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(convert_element_type_1, 0)
    clamp_max_10: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_10, 239);  clamp_min_10 = None
    clamp_min_11: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(convert_element_type, 0)
    clamp_max_11: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_11, 239);  clamp_min_11 = None
    _unsafe_index_5: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_10, clamp_max_11]);  clamp_max_10 = clamp_max_11 = None
    clamp_min_12: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(convert_element_type_1, 0)
    clamp_max_12: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_12, 239);  clamp_min_12 = None
    clamp_min_13: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_5, 0)
    clamp_max_13: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_13, 239);  clamp_min_13 = None
    _unsafe_index_6: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_12, clamp_max_13]);  clamp_max_12 = clamp_max_13 = None
    clamp_min_14: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(convert_element_type_1, 0);  convert_element_type_1 = None
    clamp_max_14: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_14, 239);  clamp_min_14 = None
    clamp_min_15: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_6, 0)
    clamp_max_15: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_15, 239);  clamp_min_15 = None
    _unsafe_index_7: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_14, clamp_max_15]);  clamp_max_14 = clamp_max_15 = None
    add_15: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(sub_1, 1.0)
    mul_18: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_15, -0.75)
    sub_14: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_18, -3.75);  mul_18 = None
    mul_19: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_14, add_15);  sub_14 = None
    add_16: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_19, -6.0);  mul_19 = None
    mul_20: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_16, add_15);  add_16 = add_15 = None
    sub_15: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_20, -3.0);  mul_20 = None
    mul_21: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_1, 1.25)
    sub_16: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_21, 2.25);  mul_21 = None
    mul_22: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_16, sub_1);  sub_16 = None
    mul_23: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_22, sub_1);  mul_22 = None
    add_17: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_23, 1);  mul_23 = None
    sub_17: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(1.0, sub_1)
    mul_24: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_17, 1.25)
    sub_18: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_24, 2.25);  mul_24 = None
    mul_25: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_18, sub_17);  sub_18 = None
    mul_26: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_25, sub_17);  mul_25 = sub_17 = None
    add_18: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_26, 1);  mul_26 = None
    sub_19: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(2.0, sub_1)
    mul_27: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_19, -0.75)
    sub_20: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_27, -3.75);  mul_27 = None
    mul_28: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_20, sub_19);  sub_20 = None
    add_19: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_28, -6.0);  mul_28 = None
    mul_29: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_19, sub_19);  add_19 = sub_19 = None
    sub_21: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_29, -3.0);  mul_29 = None
    mul_30: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_4, sub_15);  _unsafe_index_4 = sub_15 = None
    mul_31: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_5, add_17);  _unsafe_index_5 = add_17 = None
    add_20: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
    mul_32: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_6, add_18);  _unsafe_index_6 = add_18 = None
    add_21: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_20, mul_32);  add_20 = mul_32 = None
    mul_33: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_7, sub_21);  _unsafe_index_7 = sub_21 = None
    add_22: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_21, mul_33);  add_21 = mul_33 = None
    clamp_min_16: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_3, 0)
    clamp_max_16: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_16, 239);  clamp_min_16 = None
    clamp_min_17: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(sub_5, 0)
    clamp_max_17: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_17, 239);  clamp_min_17 = None
    _unsafe_index_8: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_16, clamp_max_17]);  clamp_max_16 = clamp_max_17 = None
    clamp_min_18: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_3, 0)
    clamp_max_18: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_18, 239);  clamp_min_18 = None
    clamp_min_19: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(convert_element_type, 0)
    clamp_max_19: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_19, 239);  clamp_min_19 = None
    _unsafe_index_9: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_18, clamp_max_19]);  clamp_max_18 = clamp_max_19 = None
    clamp_min_20: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_3, 0)
    clamp_max_20: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_20, 239);  clamp_min_20 = None
    clamp_min_21: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_5, 0)
    clamp_max_21: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_21, 239);  clamp_min_21 = None
    _unsafe_index_10: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_20, clamp_max_21]);  clamp_max_20 = clamp_max_21 = None
    clamp_min_22: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
    clamp_max_22: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_22, 239);  clamp_min_22 = None
    clamp_min_23: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_6, 0)
    clamp_max_23: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_23, 239);  clamp_min_23 = None
    _unsafe_index_11: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_22, clamp_max_23]);  clamp_max_22 = clamp_max_23 = None
    add_23: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(sub_1, 1.0)
    mul_34: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_23, -0.75)
    sub_22: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_34, -3.75);  mul_34 = None
    mul_35: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_22, add_23);  sub_22 = None
    add_24: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_35, -6.0);  mul_35 = None
    mul_36: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_24, add_23);  add_24 = add_23 = None
    sub_23: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_36, -3.0);  mul_36 = None
    mul_37: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_1, 1.25)
    sub_24: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_37, 2.25);  mul_37 = None
    mul_38: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_24, sub_1);  sub_24 = None
    mul_39: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_38, sub_1);  mul_38 = None
    add_25: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_39, 1);  mul_39 = None
    sub_25: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(1.0, sub_1)
    mul_40: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_25, 1.25)
    sub_26: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_40, 2.25);  mul_40 = None
    mul_41: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_26, sub_25);  sub_26 = None
    mul_42: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_41, sub_25);  mul_41 = sub_25 = None
    add_26: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_42, 1);  mul_42 = None
    sub_27: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(2.0, sub_1)
    mul_43: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_27, -0.75)
    sub_28: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_43, -3.75);  mul_43 = None
    mul_44: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_28, sub_27);  sub_28 = None
    add_27: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_44, -6.0);  mul_44 = None
    mul_45: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_27, sub_27);  add_27 = sub_27 = None
    sub_29: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_45, -3.0);  mul_45 = None
    mul_46: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_8, sub_23);  _unsafe_index_8 = sub_23 = None
    mul_47: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_9, add_25);  _unsafe_index_9 = add_25 = None
    add_28: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    mul_48: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_10, add_26);  _unsafe_index_10 = add_26 = None
    add_29: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_28, mul_48);  add_28 = mul_48 = None
    mul_49: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_11, sub_29);  _unsafe_index_11 = sub_29 = None
    add_30: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_29, mul_49);  add_29 = mul_49 = None
    clamp_min_24: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_4, 0)
    clamp_max_24: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_24, 239);  clamp_min_24 = None
    clamp_min_25: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(sub_5, 0);  sub_5 = None
    clamp_max_25: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_25, 239);  clamp_min_25 = None
    _unsafe_index_12: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_24, clamp_max_25]);  clamp_max_24 = clamp_max_25 = None
    clamp_min_26: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_4, 0)
    clamp_max_26: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_26, 239);  clamp_min_26 = None
    clamp_min_27: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(convert_element_type, 0);  convert_element_type = None
    clamp_max_27: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_27, 239);  clamp_min_27 = None
    _unsafe_index_13: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_26, clamp_max_27]);  clamp_max_26 = clamp_max_27 = None
    clamp_min_28: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_4, 0)
    clamp_max_28: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_28, 239);  clamp_min_28 = None
    clamp_min_29: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_29: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_29, 239);  clamp_min_29 = None
    _unsafe_index_14: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_28, clamp_max_29]);  clamp_max_28 = clamp_max_29 = None
    clamp_min_30: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_4, 0);  add_4 = None
    clamp_max_30: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_30, 239);  clamp_min_30 = None
    clamp_min_31: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_6, 0);  add_6 = None
    clamp_max_31: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_31, 239);  clamp_min_31 = None
    _unsafe_index_15: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(arg268_1, [view_1, view_2, clamp_max_30, clamp_max_31]);  arg268_1 = view_1 = view_2 = clamp_max_30 = clamp_max_31 = None
    add_31: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(sub_1, 1.0)
    mul_50: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_31, -0.75)
    sub_30: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_50, -3.75);  mul_50 = None
    mul_51: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_30, add_31);  sub_30 = None
    add_32: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_51, -6.0);  mul_51 = None
    mul_52: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_32, add_31);  add_32 = add_31 = None
    sub_31: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_52, -3.0);  mul_52 = None
    mul_53: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_1, 1.25)
    sub_32: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_53, 2.25);  mul_53 = None
    mul_54: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_32, sub_1);  sub_32 = None
    mul_55: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_54, sub_1);  mul_54 = None
    add_33: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_55, 1);  mul_55 = None
    sub_33: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(1.0, sub_1)
    mul_56: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_33, 1.25)
    sub_34: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_56, 2.25);  mul_56 = None
    mul_57: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_34, sub_33);  sub_34 = None
    mul_58: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_57, sub_33);  mul_57 = sub_33 = None
    add_34: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_58, 1);  mul_58 = None
    sub_35: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(2.0, sub_1);  sub_1 = None
    mul_59: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_35, -0.75)
    sub_36: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_59, -3.75);  mul_59 = None
    mul_60: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_36, sub_35);  sub_36 = None
    add_35: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_60, -6.0);  mul_60 = None
    mul_61: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_35, sub_35);  add_35 = sub_35 = None
    sub_37: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_61, -3.0);  mul_61 = None
    mul_62: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_12, sub_31);  _unsafe_index_12 = sub_31 = None
    mul_63: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_13, add_33);  _unsafe_index_13 = add_33 = None
    add_36: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
    mul_64: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_14, add_34);  _unsafe_index_14 = add_34 = None
    add_37: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_36, mul_64);  add_36 = mul_64 = None
    mul_65: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_15, sub_37);  _unsafe_index_15 = sub_37 = None
    add_38: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_37, mul_65);  add_37 = mul_65 = None
    add_39: "f32[1, 1, 224, 1]" = torch.ops.aten.add.Tensor(sub_3, 1.0)
    mul_66: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(add_39, -0.75)
    sub_38: "f32[1, 1, 224, 1]" = torch.ops.aten.sub.Tensor(mul_66, -3.75);  mul_66 = None
    mul_67: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(sub_38, add_39);  sub_38 = None
    add_40: "f32[1, 1, 224, 1]" = torch.ops.aten.add.Tensor(mul_67, -6.0);  mul_67 = None
    mul_68: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(add_40, add_39);  add_40 = add_39 = None
    sub_39: "f32[1, 1, 224, 1]" = torch.ops.aten.sub.Tensor(mul_68, -3.0);  mul_68 = None
    mul_69: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(sub_3, 1.25)
    sub_40: "f32[1, 1, 224, 1]" = torch.ops.aten.sub.Tensor(mul_69, 2.25);  mul_69 = None
    mul_70: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(sub_40, sub_3);  sub_40 = None
    mul_71: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(mul_70, sub_3);  mul_70 = None
    add_41: "f32[1, 1, 224, 1]" = torch.ops.aten.add.Tensor(mul_71, 1);  mul_71 = None
    sub_41: "f32[1, 1, 224, 1]" = torch.ops.aten.sub.Tensor(1.0, sub_3)
    mul_72: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(sub_41, 1.25)
    sub_42: "f32[1, 1, 224, 1]" = torch.ops.aten.sub.Tensor(mul_72, 2.25);  mul_72 = None
    mul_73: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(sub_42, sub_41);  sub_42 = None
    mul_74: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(mul_73, sub_41);  mul_73 = sub_41 = None
    add_42: "f32[1, 1, 224, 1]" = torch.ops.aten.add.Tensor(mul_74, 1);  mul_74 = None
    sub_43: "f32[1, 1, 224, 1]" = torch.ops.aten.sub.Tensor(2.0, sub_3);  sub_3 = None
    mul_75: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(sub_43, -0.75)
    sub_44: "f32[1, 1, 224, 1]" = torch.ops.aten.sub.Tensor(mul_75, -3.75);  mul_75 = None
    mul_76: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(sub_44, sub_43);  sub_44 = None
    add_43: "f32[1, 1, 224, 1]" = torch.ops.aten.add.Tensor(mul_76, -6.0);  mul_76 = None
    mul_77: "f32[1, 1, 224, 1]" = torch.ops.aten.mul.Tensor(add_43, sub_43);  add_43 = sub_43 = None
    sub_45: "f32[1, 1, 224, 1]" = torch.ops.aten.sub.Tensor(mul_77, -3.0);  mul_77 = None
    mul_78: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(add_14, sub_39);  add_14 = sub_39 = None
    mul_79: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(add_22, add_41);  add_22 = add_41 = None
    add_44: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_78, mul_79);  mul_78 = mul_79 = None
    mul_80: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(add_30, add_42);  add_30 = add_42 = None
    add_45: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_44, mul_80);  add_44 = mul_80 = None
    mul_81: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(add_38, sub_45);  add_38 = sub_45 = None
    add_46: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_45, mul_81);  add_45 = mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:84, code: x = self.proj(x).flatten(2).transpose(1, 2)
    convolution_1: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(add_46, arg6_1, arg7_1, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  add_46 = arg6_1 = arg7_1 = None
    view_5: "f32[8, 256, 196]" = torch.ops.aten.view.default(convolution_1, [8, 256, 196]);  convolution_1 = None
    permute_1: "f32[8, 196, 256]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:440, code: cls_tokens = cls_tokens.expand(B, -1, -1)
    expand_1: "f32[8, 1, 256]" = torch.ops.aten.expand.default(arg2_1, [8, -1, -1]);  arg2_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:441, code: x_ = torch.cat((cls_tokens, x_), dim=1)
    cat_1: "f32[8, 197, 256]" = torch.ops.aten.cat.default([expand_1, permute_1], 1);  expand_1 = permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:443, code: x_ = x_ + pos_embed
    add_47: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(cat_1, arg3_1);  cat_1 = arg3_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:444, code: x_ = self.pos_drop(x_)
    clone_1: "f32[8, 197, 256]" = torch.ops.aten.clone.default(add_47);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 401, 1]" = var_mean[0]
    getitem_1: "f32[8, 401, 1]" = var_mean[1];  var_mean = None
    add_48: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_46: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  getitem_1 = None
    mul_82: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt);  sub_46 = rsqrt = None
    mul_83: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_82, arg8_1);  mul_82 = arg8_1 = None
    add_49: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_83, arg9_1);  mul_83 = arg9_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_6: "f32[3208, 128]" = torch.ops.aten.view.default(add_49, [3208, 128]);  add_49 = None
    permute_2: "f32[128, 384]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    addmm: "f32[3208, 384]" = torch.ops.aten.addmm.default(arg11_1, view_6, permute_2);  arg11_1 = view_6 = permute_2 = None
    view_7: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm, [8, 401, 384]);  addmm = None
    view_8: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.view.default(view_7, [8, 401, 3, 4, 32]);  view_7 = None
    permute_3: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.permute.default(view_8, [2, 0, 3, 1, 4]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
    getitem_2: "f32[8, 4, 401, 32]" = unbind[0]
    getitem_3: "f32[8, 4, 401, 32]" = unbind[1]
    getitem_4: "f32[8, 4, 401, 32]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_2, getitem_3, getitem_4, None, False);  getitem_2 = getitem_3 = getitem_4 = None
    getitem_5: "f32[8, 4, 401, 32]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_4: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    view_9: "f32[8, 401, 128]" = torch.ops.aten.view.default(permute_4, [8, 401, 128]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_10: "f32[3208, 128]" = torch.ops.aten.view.default(view_9, [3208, 128]);  view_9 = None
    permute_5: "f32[128, 128]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
    addmm_1: "f32[3208, 128]" = torch.ops.aten.addmm.default(arg13_1, view_10, permute_5);  arg13_1 = view_10 = permute_5 = None
    view_11: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_1, [8, 401, 128]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_2: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_11);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_50: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(clone, clone_2);  clone = clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_1 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_9: "f32[8, 401, 1]" = var_mean_1[0]
    getitem_10: "f32[8, 401, 1]" = var_mean_1[1];  var_mean_1 = None
    add_51: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-06);  getitem_9 = None
    rsqrt_1: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_47: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add_50, getitem_10);  getitem_10 = None
    mul_84: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_1);  sub_47 = rsqrt_1 = None
    mul_85: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_84, arg14_1);  mul_84 = arg14_1 = None
    add_52: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_85, arg15_1);  mul_85 = arg15_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_12: "f32[3208, 128]" = torch.ops.aten.view.default(add_52, [3208, 128]);  add_52 = None
    permute_6: "f32[128, 384]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
    addmm_2: "f32[3208, 384]" = torch.ops.aten.addmm.default(arg17_1, view_12, permute_6);  arg17_1 = view_12 = permute_6 = None
    view_13: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_2, [8, 401, 384]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
    mul_87: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
    erf: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_53: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_88: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_86, add_53);  mul_86 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_3: "f32[8, 401, 384]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_14: "f32[3208, 384]" = torch.ops.aten.view.default(clone_3, [3208, 384]);  clone_3 = None
    permute_7: "f32[384, 128]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
    addmm_3: "f32[3208, 128]" = torch.ops.aten.addmm.default(arg19_1, view_14, permute_7);  arg19_1 = view_14 = permute_7 = None
    view_15: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_3, [8, 401, 128]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_4: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_15);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_54: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_50, clone_4);  add_50 = clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_1, [2], correction = 0, keepdim = True)
    getitem_11: "f32[8, 197, 1]" = var_mean_2[0]
    getitem_12: "f32[8, 197, 1]" = var_mean_2[1];  var_mean_2 = None
    add_55: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-06);  getitem_11 = None
    rsqrt_2: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_48: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(clone_1, getitem_12);  getitem_12 = None
    mul_89: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_2);  sub_48 = rsqrt_2 = None
    mul_90: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_89, arg20_1);  mul_89 = arg20_1 = None
    add_56: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_90, arg21_1);  mul_90 = arg21_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_16: "f32[1576, 256]" = torch.ops.aten.view.default(add_56, [1576, 256]);  add_56 = None
    permute_8: "f32[256, 768]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
    addmm_4: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg23_1, view_16, permute_8);  arg23_1 = view_16 = permute_8 = None
    view_17: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_4, [8, 197, 768]);  addmm_4 = None
    view_18: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_17, [8, 197, 3, 4, 64]);  view_17 = None
    permute_9: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_18, [2, 0, 3, 1, 4]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
    getitem_13: "f32[8, 4, 197, 64]" = unbind_1[0]
    getitem_14: "f32[8, 4, 197, 64]" = unbind_1[1]
    getitem_15: "f32[8, 4, 197, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_13, getitem_14, getitem_15, None, False);  getitem_13 = getitem_14 = getitem_15 = None
    getitem_16: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_10: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
    view_19: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_10, [8, 197, 256]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_20: "f32[1576, 256]" = torch.ops.aten.view.default(view_19, [1576, 256]);  view_19 = None
    permute_11: "f32[256, 256]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
    addmm_5: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg25_1, view_20, permute_11);  arg25_1 = view_20 = permute_11 = None
    view_21: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_5, [8, 197, 256]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_5: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_21);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_57: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(clone_1, clone_5);  clone_1 = clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_3 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 197, 1]" = var_mean_3[0]
    getitem_21: "f32[8, 197, 1]" = var_mean_3[1];  var_mean_3 = None
    add_58: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_3: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_49: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_57, getitem_21);  getitem_21 = None
    mul_91: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_3);  sub_49 = rsqrt_3 = None
    mul_92: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_91, arg26_1);  mul_91 = arg26_1 = None
    add_59: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_92, arg27_1);  mul_92 = arg27_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_22: "f32[1576, 256]" = torch.ops.aten.view.default(add_59, [1576, 256]);  add_59 = None
    permute_12: "f32[256, 768]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
    addmm_6: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg29_1, view_22, permute_12);  arg29_1 = view_22 = permute_12 = None
    view_23: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_6, [8, 197, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_93: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_23, 0.5)
    mul_94: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_23, 0.7071067811865476);  view_23 = None
    erf_1: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_60: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_95: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_93, add_60);  mul_93 = add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_6: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_95);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_24: "f32[1576, 768]" = torch.ops.aten.view.default(clone_6, [1576, 768]);  clone_6 = None
    permute_13: "f32[768, 256]" = torch.ops.aten.permute.default(arg30_1, [1, 0]);  arg30_1 = None
    addmm_7: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg31_1, view_24, permute_13);  arg31_1 = view_24 = permute_13 = None
    view_25: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_7, [8, 197, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_7: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_25);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_61: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_57, clone_7);  add_57 = clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_4 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 197, 1]" = var_mean_4[0]
    getitem_23: "f32[8, 197, 1]" = var_mean_4[1];  var_mean_4 = None
    add_62: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_4: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_50: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_61, getitem_23);  getitem_23 = None
    mul_96: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_4);  sub_50 = rsqrt_4 = None
    mul_97: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_96, arg32_1);  mul_96 = arg32_1 = None
    add_63: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_97, arg33_1);  mul_97 = arg33_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_26: "f32[1576, 256]" = torch.ops.aten.view.default(add_63, [1576, 256]);  add_63 = None
    permute_14: "f32[256, 768]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
    addmm_8: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg35_1, view_26, permute_14);  arg35_1 = view_26 = permute_14 = None
    view_27: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_8, [8, 197, 768]);  addmm_8 = None
    view_28: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_27, [8, 197, 3, 4, 64]);  view_27 = None
    permute_15: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_28, [2, 0, 3, 1, 4]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_15);  permute_15 = None
    getitem_24: "f32[8, 4, 197, 64]" = unbind_2[0]
    getitem_25: "f32[8, 4, 197, 64]" = unbind_2[1]
    getitem_26: "f32[8, 4, 197, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_24, getitem_25, getitem_26, None, False);  getitem_24 = getitem_25 = getitem_26 = None
    getitem_27: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_16: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
    view_29: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_16, [8, 197, 256]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_30: "f32[1576, 256]" = torch.ops.aten.view.default(view_29, [1576, 256]);  view_29 = None
    permute_17: "f32[256, 256]" = torch.ops.aten.permute.default(arg36_1, [1, 0]);  arg36_1 = None
    addmm_9: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg37_1, view_30, permute_17);  arg37_1 = view_30 = permute_17 = None
    view_31: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_9, [8, 197, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_8: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_31);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_64: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_61, clone_8);  add_61 = clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_5 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_31: "f32[8, 197, 1]" = var_mean_5[0]
    getitem_32: "f32[8, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    add_65: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-06);  getitem_31 = None
    rsqrt_5: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_51: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_64, getitem_32);  getitem_32 = None
    mul_98: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_5);  sub_51 = rsqrt_5 = None
    mul_99: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_98, arg38_1);  mul_98 = arg38_1 = None
    add_66: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_99, arg39_1);  mul_99 = arg39_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_32: "f32[1576, 256]" = torch.ops.aten.view.default(add_66, [1576, 256]);  add_66 = None
    permute_18: "f32[256, 768]" = torch.ops.aten.permute.default(arg40_1, [1, 0]);  arg40_1 = None
    addmm_10: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg41_1, view_32, permute_18);  arg41_1 = view_32 = permute_18 = None
    view_33: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_10, [8, 197, 768]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
    mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476);  view_33 = None
    erf_2: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
    add_67: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_102: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_100, add_67);  mul_100 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_9: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_102);  mul_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_34: "f32[1576, 768]" = torch.ops.aten.view.default(clone_9, [1576, 768]);  clone_9 = None
    permute_19: "f32[768, 256]" = torch.ops.aten.permute.default(arg42_1, [1, 0]);  arg42_1 = None
    addmm_11: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg43_1, view_34, permute_19);  arg43_1 = view_34 = permute_19 = None
    view_35: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_11, [8, 197, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_10: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_35);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_68: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_64, clone_10);  add_64 = clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_6 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
    getitem_33: "f32[8, 197, 1]" = var_mean_6[0]
    getitem_34: "f32[8, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    add_69: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
    rsqrt_6: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_52: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_68, getitem_34);  getitem_34 = None
    mul_103: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_6);  sub_52 = rsqrt_6 = None
    mul_104: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_103, arg44_1);  mul_103 = arg44_1 = None
    add_70: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_104, arg45_1);  mul_104 = arg45_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_36: "f32[1576, 256]" = torch.ops.aten.view.default(add_70, [1576, 256]);  add_70 = None
    permute_20: "f32[256, 768]" = torch.ops.aten.permute.default(arg46_1, [1, 0]);  arg46_1 = None
    addmm_12: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg47_1, view_36, permute_20);  arg47_1 = view_36 = permute_20 = None
    view_37: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_12, [8, 197, 768]);  addmm_12 = None
    view_38: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_37, [8, 197, 3, 4, 64]);  view_37 = None
    permute_21: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_38, [2, 0, 3, 1, 4]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_21);  permute_21 = None
    getitem_35: "f32[8, 4, 197, 64]" = unbind_3[0]
    getitem_36: "f32[8, 4, 197, 64]" = unbind_3[1]
    getitem_37: "f32[8, 4, 197, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_35, getitem_36, getitem_37, None, False);  getitem_35 = getitem_36 = getitem_37 = None
    getitem_38: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_22: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
    view_39: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_22, [8, 197, 256]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_40: "f32[1576, 256]" = torch.ops.aten.view.default(view_39, [1576, 256]);  view_39 = None
    permute_23: "f32[256, 256]" = torch.ops.aten.permute.default(arg48_1, [1, 0]);  arg48_1 = None
    addmm_13: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg49_1, view_40, permute_23);  arg49_1 = view_40 = permute_23 = None
    view_41: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_13, [8, 197, 256]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_11: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_41);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_71: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_68, clone_11);  add_68 = clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_7 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 197, 1]" = var_mean_7[0]
    getitem_43: "f32[8, 197, 1]" = var_mean_7[1];  var_mean_7 = None
    add_72: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_7: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_53: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_71, getitem_43);  getitem_43 = None
    mul_105: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_7);  sub_53 = rsqrt_7 = None
    mul_106: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_105, arg50_1);  mul_105 = arg50_1 = None
    add_73: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_106, arg51_1);  mul_106 = arg51_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_42: "f32[1576, 256]" = torch.ops.aten.view.default(add_73, [1576, 256]);  add_73 = None
    permute_24: "f32[256, 768]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    addmm_14: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg53_1, view_42, permute_24);  arg53_1 = view_42 = permute_24 = None
    view_43: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_14, [8, 197, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_107: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_108: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
    erf_3: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
    add_74: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_107, add_74);  mul_107 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_12: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_109);  mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_44: "f32[1576, 768]" = torch.ops.aten.view.default(clone_12, [1576, 768]);  clone_12 = None
    permute_25: "f32[768, 256]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_15: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg55_1, view_44, permute_25);  arg55_1 = view_44 = permute_25 = None
    view_45: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_15, [8, 197, 256]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_13: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_45);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_75: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_71, clone_13);  add_71 = clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    slice_1: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_54, 0, 0, 9223372036854775807)
    slice_2: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 1);  slice_1 = None
    clone_14: "f32[8, 1, 128]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_14, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 1, 1]" = var_mean_8[0]
    getitem_45: "f32[8, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_76: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_8: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_54: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(clone_14, getitem_45);  clone_14 = getitem_45 = None
    mul_110: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_8);  sub_54 = rsqrt_8 = None
    mul_111: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_110, arg56_1);  mul_110 = arg56_1 = None
    add_77: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_111, arg57_1);  mul_111 = arg57_1 = None
    mul_112: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_77, 0.5)
    mul_113: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_77, 0.7071067811865476);  add_77 = None
    erf_4: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_113);  mul_113 = None
    add_78: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_114: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_112, add_78);  mul_112 = add_78 = None
    view_46: "f32[8, 128]" = torch.ops.aten.view.default(mul_114, [8, 128]);  mul_114 = None
    permute_26: "f32[128, 256]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_16: "f32[8, 256]" = torch.ops.aten.addmm.default(arg59_1, view_46, permute_26);  arg59_1 = view_46 = permute_26 = None
    view_47: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_16, [8, 1, 256]);  addmm_16 = None
    slice_3: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_75, 0, 0, 9223372036854775807)
    slice_4: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 1);  slice_3 = None
    clone_15: "f32[8, 1, 256]" = torch.ops.aten.clone.default(slice_4, memory_format = torch.contiguous_format);  slice_4 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_15, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 1, 1]" = var_mean_9[0]
    getitem_47: "f32[8, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_79: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_9: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_55: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(clone_15, getitem_47);  clone_15 = getitem_47 = None
    mul_115: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_9);  sub_55 = rsqrt_9 = None
    mul_116: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_115, arg60_1);  mul_115 = arg60_1 = None
    add_80: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_116, arg61_1);  mul_116 = arg61_1 = None
    mul_117: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_80, 0.5)
    mul_118: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_80, 0.7071067811865476);  add_80 = None
    erf_5: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_81: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_119: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_117, add_81);  mul_117 = add_81 = None
    view_48: "f32[8, 256]" = torch.ops.aten.view.default(mul_119, [8, 256]);  mul_119 = None
    permute_27: "f32[256, 128]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    addmm_17: "f32[8, 128]" = torch.ops.aten.addmm.default(arg63_1, view_48, permute_27);  arg63_1 = view_48 = permute_27 = None
    view_49: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_17, [8, 1, 128]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_5: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_75, 0, 0, 9223372036854775807)
    slice_6: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(slice_5, 1, 1, 9223372036854775807);  slice_5 = None
    cat_2: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_47, slice_6], 1);  view_47 = slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_7: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(cat_2, 0, 0, 9223372036854775807)
    slice_8: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 1);  slice_7 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(cat_2, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 197, 1]" = var_mean_10[0]
    getitem_49: "f32[8, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    add_82: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_10: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_56: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_2, getitem_49);  cat_2 = getitem_49 = None
    mul_120: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_10);  sub_56 = rsqrt_10 = None
    mul_121: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_120, arg64_1);  mul_120 = arg64_1 = None
    add_83: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_121, arg65_1);  mul_121 = arg65_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_9: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_83, 0, 0, 9223372036854775807)
    slice_10: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 1);  slice_9 = None
    permute_28: "f32[256, 256]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    view_50: "f32[8, 256]" = torch.ops.aten.view.default(slice_10, [8, 256]);  slice_10 = None
    mm: "f32[8, 256]" = torch.ops.aten.mm.default(view_50, permute_28);  view_50 = permute_28 = None
    view_51: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm, [8, 1, 256]);  mm = None
    add_84: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(view_51, arg67_1);  view_51 = arg67_1 = None
    view_52: "f32[8, 1, 4, 64]" = torch.ops.aten.view.default(add_84, [8, 1, 4, 64]);  add_84 = None
    permute_29: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_53: "f32[1576, 256]" = torch.ops.aten.view.default(add_83, [1576, 256])
    permute_30: "f32[256, 256]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    addmm_18: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg69_1, view_53, permute_30);  arg69_1 = view_53 = permute_30 = None
    view_54: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_18, [8, 197, 256]);  addmm_18 = None
    view_55: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_54, [8, 197, 4, 64]);  view_54 = None
    permute_31: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_56: "f32[1576, 256]" = torch.ops.aten.view.default(add_83, [1576, 256]);  add_83 = None
    permute_32: "f32[256, 256]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_19: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg71_1, view_56, permute_32);  arg71_1 = view_56 = permute_32 = None
    view_57: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_19, [8, 197, 256]);  addmm_19 = None
    view_58: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_57, [8, 197, 4, 64]);  view_57 = None
    permute_33: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_34: "f32[8, 4, 64, 197]" = torch.ops.aten.permute.default(permute_31, [0, 1, 3, 2]);  permute_31 = None
    expand_2: "f32[8, 4, 1, 64]" = torch.ops.aten.expand.default(permute_29, [8, 4, 1, 64]);  permute_29 = None
    view_59: "f32[32, 1, 64]" = torch.ops.aten.view.default(expand_2, [32, 1, 64]);  expand_2 = None
    expand_3: "f32[8, 4, 64, 197]" = torch.ops.aten.expand.default(permute_34, [8, 4, 64, 197]);  permute_34 = None
    clone_16: "f32[8, 4, 64, 197]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_60: "f32[32, 64, 197]" = torch.ops.aten.view.default(clone_16, [32, 64, 197]);  clone_16 = None
    bmm: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_59, view_60);  view_59 = view_60 = None
    view_61: "f32[8, 4, 1, 197]" = torch.ops.aten.view.default(bmm, [8, 4, 1, 197]);  bmm = None
    mul_122: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(view_61, 0.125);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    amax: "f32[8, 4, 1, 1]" = torch.ops.aten.amax.default(mul_122, [-1], True)
    sub_57: "f32[8, 4, 1, 197]" = torch.ops.aten.sub.Tensor(mul_122, amax);  mul_122 = amax = None
    exp: "f32[8, 4, 1, 197]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_1: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 4, 1, 197]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    clone_17: "f32[8, 4, 1, 197]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    expand_4: "f32[8, 4, 1, 197]" = torch.ops.aten.expand.default(clone_17, [8, 4, 1, 197]);  clone_17 = None
    view_62: "f32[32, 1, 197]" = torch.ops.aten.view.default(expand_4, [32, 1, 197]);  expand_4 = None
    expand_5: "f32[8, 4, 197, 64]" = torch.ops.aten.expand.default(permute_33, [8, 4, 197, 64]);  permute_33 = None
    clone_18: "f32[8, 4, 197, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_63: "f32[32, 197, 64]" = torch.ops.aten.view.default(clone_18, [32, 197, 64]);  clone_18 = None
    bmm_1: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_62, view_63);  view_62 = view_63 = None
    view_64: "f32[8, 4, 1, 64]" = torch.ops.aten.view.default(bmm_1, [8, 4, 1, 64]);  bmm_1 = None
    permute_35: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
    view_65: "f32[8, 1, 256]" = torch.ops.aten.view.default(permute_35, [8, 1, 256]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_66: "f32[8, 256]" = torch.ops.aten.view.default(view_65, [8, 256]);  view_65 = None
    permute_36: "f32[256, 256]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_20: "f32[8, 256]" = torch.ops.aten.addmm.default(arg73_1, view_66, permute_36);  arg73_1 = view_66 = permute_36 = None
    view_67: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_20, [8, 1, 256]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    clone_19: "f32[8, 1, 256]" = torch.ops.aten.clone.default(view_67);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_85: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(slice_8, clone_19);  slice_8 = clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    slice_11: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_85, 0, 0, 9223372036854775807);  add_85 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(slice_11, [2], correction = 0, keepdim = True)
    getitem_50: "f32[8, 1, 1]" = var_mean_11[0]
    getitem_51: "f32[8, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_86: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_11: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_58: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(slice_11, getitem_51);  slice_11 = getitem_51 = None
    mul_123: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_11);  sub_58 = rsqrt_11 = None
    mul_124: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_123, arg74_1);  mul_123 = arg74_1 = None
    add_87: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_124, arg75_1);  mul_124 = arg75_1 = None
    mul_125: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_87, 0.5)
    mul_126: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_87, 0.7071067811865476);  add_87 = None
    erf_6: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
    add_88: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_127: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_125, add_88);  mul_125 = add_88 = None
    view_68: "f32[8, 256]" = torch.ops.aten.view.default(mul_127, [8, 256]);  mul_127 = None
    permute_37: "f32[256, 128]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    addmm_21: "f32[8, 128]" = torch.ops.aten.addmm.default(arg77_1, view_68, permute_37);  arg77_1 = view_68 = permute_37 = None
    view_69: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_21, [8, 1, 128]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_12: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_54, 0, 0, 9223372036854775807)
    slice_13: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(slice_12, 1, 1, 9223372036854775807);  slice_12 = None
    cat_3: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_69, slice_13], 1);  view_69 = slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_14: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_54, 0, 0, 9223372036854775807);  add_54 = None
    slice_15: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(slice_14, 1, 1, 9223372036854775807);  slice_14 = None
    cat_4: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_49, slice_15], 1);  view_49 = slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_16: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(cat_4, 0, 0, 9223372036854775807)
    slice_17: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 1);  slice_16 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(cat_4, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 401, 1]" = var_mean_12[0]
    getitem_53: "f32[8, 401, 1]" = var_mean_12[1];  var_mean_12 = None
    add_89: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_12: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_59: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_4, getitem_53);  cat_4 = getitem_53 = None
    mul_128: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_12);  sub_59 = rsqrt_12 = None
    mul_129: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_128, arg78_1);  mul_128 = arg78_1 = None
    add_90: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_129, arg79_1);  mul_129 = arg79_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_18: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_90, 0, 0, 9223372036854775807)
    slice_19: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_18, 1, 0, 1);  slice_18 = None
    permute_38: "f32[128, 128]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    view_70: "f32[8, 128]" = torch.ops.aten.view.default(slice_19, [8, 128]);  slice_19 = None
    mm_1: "f32[8, 128]" = torch.ops.aten.mm.default(view_70, permute_38);  view_70 = permute_38 = None
    view_71: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_1, [8, 1, 128]);  mm_1 = None
    add_91: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(view_71, arg81_1);  view_71 = arg81_1 = None
    view_72: "f32[8, 1, 4, 32]" = torch.ops.aten.view.default(add_91, [8, 1, 4, 32]);  add_91 = None
    permute_39: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_73: "f32[3208, 128]" = torch.ops.aten.view.default(add_90, [3208, 128])
    permute_40: "f32[128, 128]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    addmm_22: "f32[3208, 128]" = torch.ops.aten.addmm.default(arg83_1, view_73, permute_40);  arg83_1 = view_73 = permute_40 = None
    view_74: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_22, [8, 401, 128]);  addmm_22 = None
    view_75: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_74, [8, 401, 4, 32]);  view_74 = None
    permute_41: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_76: "f32[3208, 128]" = torch.ops.aten.view.default(add_90, [3208, 128]);  add_90 = None
    permute_42: "f32[128, 128]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_23: "f32[3208, 128]" = torch.ops.aten.addmm.default(arg85_1, view_76, permute_42);  arg85_1 = view_76 = permute_42 = None
    view_77: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_23, [8, 401, 128]);  addmm_23 = None
    view_78: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_77, [8, 401, 4, 32]);  view_77 = None
    permute_43: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_44: "f32[8, 4, 32, 401]" = torch.ops.aten.permute.default(permute_41, [0, 1, 3, 2]);  permute_41 = None
    expand_6: "f32[8, 4, 1, 32]" = torch.ops.aten.expand.default(permute_39, [8, 4, 1, 32]);  permute_39 = None
    view_79: "f32[32, 1, 32]" = torch.ops.aten.view.default(expand_6, [32, 1, 32]);  expand_6 = None
    expand_7: "f32[8, 4, 32, 401]" = torch.ops.aten.expand.default(permute_44, [8, 4, 32, 401]);  permute_44 = None
    clone_20: "f32[8, 4, 32, 401]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_80: "f32[32, 32, 401]" = torch.ops.aten.view.default(clone_20, [32, 32, 401]);  clone_20 = None
    bmm_2: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_79, view_80);  view_79 = view_80 = None
    view_81: "f32[8, 4, 1, 401]" = torch.ops.aten.view.default(bmm_2, [8, 4, 1, 401]);  bmm_2 = None
    mul_130: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(view_81, 0.1767766952966369);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[8, 4, 1, 1]" = torch.ops.aten.amax.default(mul_130, [-1], True)
    sub_60: "f32[8, 4, 1, 401]" = torch.ops.aten.sub.Tensor(mul_130, amax_1);  mul_130 = amax_1 = None
    exp_1: "f32[8, 4, 1, 401]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    sum_2: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 4, 1, 401]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    clone_21: "f32[8, 4, 1, 401]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    expand_8: "f32[8, 4, 1, 401]" = torch.ops.aten.expand.default(clone_21, [8, 4, 1, 401]);  clone_21 = None
    view_82: "f32[32, 1, 401]" = torch.ops.aten.view.default(expand_8, [32, 1, 401]);  expand_8 = None
    expand_9: "f32[8, 4, 401, 32]" = torch.ops.aten.expand.default(permute_43, [8, 4, 401, 32]);  permute_43 = None
    clone_22: "f32[8, 4, 401, 32]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_83: "f32[32, 401, 32]" = torch.ops.aten.view.default(clone_22, [32, 401, 32]);  clone_22 = None
    bmm_3: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_82, view_83);  view_82 = view_83 = None
    view_84: "f32[8, 4, 1, 32]" = torch.ops.aten.view.default(bmm_3, [8, 4, 1, 32]);  bmm_3 = None
    permute_45: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    view_85: "f32[8, 1, 128]" = torch.ops.aten.view.default(permute_45, [8, 1, 128]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_86: "f32[8, 128]" = torch.ops.aten.view.default(view_85, [8, 128]);  view_85 = None
    permute_46: "f32[128, 128]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_24: "f32[8, 128]" = torch.ops.aten.addmm.default(arg87_1, view_86, permute_46);  arg87_1 = view_86 = permute_46 = None
    view_87: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_24, [8, 1, 128]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    clone_23: "f32[8, 1, 128]" = torch.ops.aten.clone.default(view_87);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_92: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(slice_17, clone_23);  slice_17 = clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    slice_20: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_92, 0, 0, 9223372036854775807);  add_92 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(slice_20, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 1, 1]" = var_mean_13[0]
    getitem_55: "f32[8, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_93: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_13: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_61: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(slice_20, getitem_55);  slice_20 = getitem_55 = None
    mul_131: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_13);  sub_61 = rsqrt_13 = None
    mul_132: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_131, arg88_1);  mul_131 = arg88_1 = None
    add_94: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_132, arg89_1);  mul_132 = arg89_1 = None
    mul_133: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_94, 0.5)
    mul_134: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_94, 0.7071067811865476);  add_94 = None
    erf_7: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_134);  mul_134 = None
    add_95: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_135: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_133, add_95);  mul_133 = add_95 = None
    view_88: "f32[8, 128]" = torch.ops.aten.view.default(mul_135, [8, 128]);  mul_135 = None
    permute_47: "f32[128, 256]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_25: "f32[8, 256]" = torch.ops.aten.addmm.default(arg91_1, view_88, permute_47);  arg91_1 = view_88 = permute_47 = None
    view_89: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_25, [8, 1, 256]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_21: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_75, 0, 0, 9223372036854775807);  add_75 = None
    slice_22: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(slice_21, 1, 1, 9223372036854775807);  slice_21 = None
    cat_5: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_89, slice_22], 1);  view_89 = slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_14 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
    getitem_56: "f32[8, 401, 1]" = var_mean_14[0]
    getitem_57: "f32[8, 401, 1]" = var_mean_14[1];  var_mean_14 = None
    add_96: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_14: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_62: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_3, getitem_57);  getitem_57 = None
    mul_136: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_14);  sub_62 = rsqrt_14 = None
    mul_137: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_136, arg92_1);  mul_136 = arg92_1 = None
    add_97: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_137, arg93_1);  mul_137 = arg93_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_90: "f32[3208, 128]" = torch.ops.aten.view.default(add_97, [3208, 128]);  add_97 = None
    permute_48: "f32[128, 384]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    addmm_26: "f32[3208, 384]" = torch.ops.aten.addmm.default(arg95_1, view_90, permute_48);  arg95_1 = view_90 = permute_48 = None
    view_91: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_26, [8, 401, 384]);  addmm_26 = None
    view_92: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.view.default(view_91, [8, 401, 3, 4, 32]);  view_91 = None
    permute_49: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.permute.default(view_92, [2, 0, 3, 1, 4]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_49);  permute_49 = None
    getitem_58: "f32[8, 4, 401, 32]" = unbind_4[0]
    getitem_59: "f32[8, 4, 401, 32]" = unbind_4[1]
    getitem_60: "f32[8, 4, 401, 32]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_58, getitem_59, getitem_60, None, False);  getitem_58 = getitem_59 = getitem_60 = None
    getitem_61: "f32[8, 4, 401, 32]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_50: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(getitem_61, [0, 2, 1, 3]);  getitem_61 = None
    view_93: "f32[8, 401, 128]" = torch.ops.aten.view.default(permute_50, [8, 401, 128]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_94: "f32[3208, 128]" = torch.ops.aten.view.default(view_93, [3208, 128]);  view_93 = None
    permute_51: "f32[128, 128]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_27: "f32[3208, 128]" = torch.ops.aten.addmm.default(arg97_1, view_94, permute_51);  arg97_1 = view_94 = permute_51 = None
    view_95: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_27, [8, 401, 128]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_24: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_95);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_98: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(cat_3, clone_24);  cat_3 = clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_15 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_65: "f32[8, 401, 1]" = var_mean_15[0]
    getitem_66: "f32[8, 401, 1]" = var_mean_15[1];  var_mean_15 = None
    add_99: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_65, 1e-06);  getitem_65 = None
    rsqrt_15: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_63: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add_98, getitem_66);  getitem_66 = None
    mul_138: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_15);  sub_63 = rsqrt_15 = None
    mul_139: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_138, arg98_1);  mul_138 = arg98_1 = None
    add_100: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_139, arg99_1);  mul_139 = arg99_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_96: "f32[3208, 128]" = torch.ops.aten.view.default(add_100, [3208, 128]);  add_100 = None
    permute_52: "f32[128, 384]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_28: "f32[3208, 384]" = torch.ops.aten.addmm.default(arg101_1, view_96, permute_52);  arg101_1 = view_96 = permute_52 = None
    view_97: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_28, [8, 401, 384]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_140: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_97, 0.5)
    mul_141: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_97, 0.7071067811865476);  view_97 = None
    erf_8: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_101: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_142: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_140, add_101);  mul_140 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_25: "f32[8, 401, 384]" = torch.ops.aten.clone.default(mul_142);  mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_98: "f32[3208, 384]" = torch.ops.aten.view.default(clone_25, [3208, 384]);  clone_25 = None
    permute_53: "f32[384, 128]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_29: "f32[3208, 128]" = torch.ops.aten.addmm.default(arg103_1, view_98, permute_53);  arg103_1 = view_98 = permute_53 = None
    view_99: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_29, [8, 401, 128]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_26: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_99);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_102: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_98, clone_26);  add_98 = clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_16 = torch.ops.aten.var_mean.correction(cat_5, [2], correction = 0, keepdim = True)
    getitem_67: "f32[8, 197, 1]" = var_mean_16[0]
    getitem_68: "f32[8, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    add_103: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_67, 1e-06);  getitem_67 = None
    rsqrt_16: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    sub_64: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_5, getitem_68);  getitem_68 = None
    mul_143: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_16);  sub_64 = rsqrt_16 = None
    mul_144: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_143, arg104_1);  mul_143 = arg104_1 = None
    add_104: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_144, arg105_1);  mul_144 = arg105_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_100: "f32[1576, 256]" = torch.ops.aten.view.default(add_104, [1576, 256]);  add_104 = None
    permute_54: "f32[256, 768]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_30: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg107_1, view_100, permute_54);  arg107_1 = view_100 = permute_54 = None
    view_101: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_30, [8, 197, 768]);  addmm_30 = None
    view_102: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_101, [8, 197, 3, 4, 64]);  view_101 = None
    permute_55: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_102, [2, 0, 3, 1, 4]);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_55);  permute_55 = None
    getitem_69: "f32[8, 4, 197, 64]" = unbind_5[0]
    getitem_70: "f32[8, 4, 197, 64]" = unbind_5[1]
    getitem_71: "f32[8, 4, 197, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_69, getitem_70, getitem_71, None, False);  getitem_69 = getitem_70 = getitem_71 = None
    getitem_72: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_56: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None
    view_103: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_56, [8, 197, 256]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_104: "f32[1576, 256]" = torch.ops.aten.view.default(view_103, [1576, 256]);  view_103 = None
    permute_57: "f32[256, 256]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    addmm_31: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg109_1, view_104, permute_57);  arg109_1 = view_104 = permute_57 = None
    view_105: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_31, [8, 197, 256]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_27: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_105);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_105: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(cat_5, clone_27);  cat_5 = clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_17 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 197, 1]" = var_mean_17[0]
    getitem_77: "f32[8, 197, 1]" = var_mean_17[1];  var_mean_17 = None
    add_106: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_17: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_65: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_105, getitem_77);  getitem_77 = None
    mul_145: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_17);  sub_65 = rsqrt_17 = None
    mul_146: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_145, arg110_1);  mul_145 = arg110_1 = None
    add_107: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_146, arg111_1);  mul_146 = arg111_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_106: "f32[1576, 256]" = torch.ops.aten.view.default(add_107, [1576, 256]);  add_107 = None
    permute_58: "f32[256, 768]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_32: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg113_1, view_106, permute_58);  arg113_1 = view_106 = permute_58 = None
    view_107: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_32, [8, 197, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_147: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_148: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
    erf_9: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
    add_108: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_149: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_147, add_108);  mul_147 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_28: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_149);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_108: "f32[1576, 768]" = torch.ops.aten.view.default(clone_28, [1576, 768]);  clone_28 = None
    permute_59: "f32[768, 256]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm_33: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg115_1, view_108, permute_59);  arg115_1 = view_108 = permute_59 = None
    view_109: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_33, [8, 197, 256]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_29: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_109);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_109: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_105, clone_29);  add_105 = clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_18 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_78: "f32[8, 197, 1]" = var_mean_18[0]
    getitem_79: "f32[8, 197, 1]" = var_mean_18[1];  var_mean_18 = None
    add_110: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_18: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_66: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_109, getitem_79);  getitem_79 = None
    mul_150: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_18);  sub_66 = rsqrt_18 = None
    mul_151: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_150, arg116_1);  mul_150 = arg116_1 = None
    add_111: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_151, arg117_1);  mul_151 = arg117_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_110: "f32[1576, 256]" = torch.ops.aten.view.default(add_111, [1576, 256]);  add_111 = None
    permute_60: "f32[256, 768]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_34: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg119_1, view_110, permute_60);  arg119_1 = view_110 = permute_60 = None
    view_111: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_34, [8, 197, 768]);  addmm_34 = None
    view_112: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_111, [8, 197, 3, 4, 64]);  view_111 = None
    permute_61: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_112, [2, 0, 3, 1, 4]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_61);  permute_61 = None
    getitem_80: "f32[8, 4, 197, 64]" = unbind_6[0]
    getitem_81: "f32[8, 4, 197, 64]" = unbind_6[1]
    getitem_82: "f32[8, 4, 197, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_80, getitem_81, getitem_82, None, False);  getitem_80 = getitem_81 = getitem_82 = None
    getitem_83: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_62: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_83, [0, 2, 1, 3]);  getitem_83 = None
    view_113: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_62, [8, 197, 256]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_114: "f32[1576, 256]" = torch.ops.aten.view.default(view_113, [1576, 256]);  view_113 = None
    permute_63: "f32[256, 256]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_35: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg121_1, view_114, permute_63);  arg121_1 = view_114 = permute_63 = None
    view_115: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_35, [8, 197, 256]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_30: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_115);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_112: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_109, clone_30);  add_109 = clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
    getitem_87: "f32[8, 197, 1]" = var_mean_19[0]
    getitem_88: "f32[8, 197, 1]" = var_mean_19[1];  var_mean_19 = None
    add_113: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_87, 1e-06);  getitem_87 = None
    rsqrt_19: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_67: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_112, getitem_88);  getitem_88 = None
    mul_152: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_19);  sub_67 = rsqrt_19 = None
    mul_153: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_152, arg122_1);  mul_152 = arg122_1 = None
    add_114: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_153, arg123_1);  mul_153 = arg123_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_116: "f32[1576, 256]" = torch.ops.aten.view.default(add_114, [1576, 256]);  add_114 = None
    permute_64: "f32[256, 768]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    addmm_36: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg125_1, view_116, permute_64);  arg125_1 = view_116 = permute_64 = None
    view_117: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_36, [8, 197, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_154: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_117, 0.5)
    mul_155: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_117, 0.7071067811865476);  view_117 = None
    erf_10: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_115: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_156: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_154, add_115);  mul_154 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_31: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_156);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_118: "f32[1576, 768]" = torch.ops.aten.view.default(clone_31, [1576, 768]);  clone_31 = None
    permute_65: "f32[768, 256]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    addmm_37: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg127_1, view_118, permute_65);  arg127_1 = view_118 = permute_65 = None
    view_119: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_37, [8, 197, 256]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_119);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_116: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_112, clone_32);  add_112 = clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_20 = torch.ops.aten.var_mean.correction(add_116, [2], correction = 0, keepdim = True)
    getitem_89: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_90: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_117: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_89, 1e-06);  getitem_89 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_68: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_116, getitem_90);  getitem_90 = None
    mul_157: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_20);  sub_68 = rsqrt_20 = None
    mul_158: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_157, arg128_1);  mul_157 = arg128_1 = None
    add_118: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_158, arg129_1);  mul_158 = arg129_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_120: "f32[1576, 256]" = torch.ops.aten.view.default(add_118, [1576, 256]);  add_118 = None
    permute_66: "f32[256, 768]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_38: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg131_1, view_120, permute_66);  arg131_1 = view_120 = permute_66 = None
    view_121: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_38, [8, 197, 768]);  addmm_38 = None
    view_122: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_121, [8, 197, 3, 4, 64]);  view_121 = None
    permute_67: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_122, [2, 0, 3, 1, 4]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
    getitem_91: "f32[8, 4, 197, 64]" = unbind_7[0]
    getitem_92: "f32[8, 4, 197, 64]" = unbind_7[1]
    getitem_93: "f32[8, 4, 197, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_91, getitem_92, getitem_93, None, False);  getitem_91 = getitem_92 = getitem_93 = None
    getitem_94: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_68: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_94, [0, 2, 1, 3]);  getitem_94 = None
    view_123: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_68, [8, 197, 256]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_124: "f32[1576, 256]" = torch.ops.aten.view.default(view_123, [1576, 256]);  view_123 = None
    permute_69: "f32[256, 256]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    addmm_39: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg133_1, view_124, permute_69);  arg133_1 = view_124 = permute_69 = None
    view_125: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_39, [8, 197, 256]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_33: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_125);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_119: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_116, clone_33);  add_116 = clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_21 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
    getitem_98: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_99: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_120: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-06);  getitem_98 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    sub_69: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_119, getitem_99);  getitem_99 = None
    mul_159: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_21);  sub_69 = rsqrt_21 = None
    mul_160: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_159, arg134_1);  mul_159 = arg134_1 = None
    add_121: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_160, arg135_1);  mul_160 = arg135_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_126: "f32[1576, 256]" = torch.ops.aten.view.default(add_121, [1576, 256]);  add_121 = None
    permute_70: "f32[256, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    addmm_40: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg137_1, view_126, permute_70);  arg137_1 = view_126 = permute_70 = None
    view_127: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_40, [8, 197, 768]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_161: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, 0.5)
    mul_162: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, 0.7071067811865476);  view_127 = None
    erf_11: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_162);  mul_162 = None
    add_122: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_163: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_161, add_122);  mul_161 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_34: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_163);  mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_128: "f32[1576, 768]" = torch.ops.aten.view.default(clone_34, [1576, 768]);  clone_34 = None
    permute_71: "f32[768, 256]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    addmm_41: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg139_1, view_128, permute_71);  arg139_1 = view_128 = permute_71 = None
    view_129: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_41, [8, 197, 256]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_35: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_129);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_123: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_119, clone_35);  add_119 = clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    slice_23: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_102, 0, 0, 9223372036854775807)
    slice_24: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_23, 1, 0, 1);  slice_23 = None
    clone_36: "f32[8, 1, 128]" = torch.ops.aten.clone.default(slice_24, memory_format = torch.contiguous_format);  slice_24 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_36, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 1, 1]" = var_mean_22[0]
    getitem_101: "f32[8, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_124: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
    rsqrt_22: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    sub_70: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(clone_36, getitem_101);  clone_36 = getitem_101 = None
    mul_164: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_22);  sub_70 = rsqrt_22 = None
    mul_165: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_164, arg140_1);  mul_164 = arg140_1 = None
    add_125: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_165, arg141_1);  mul_165 = arg141_1 = None
    mul_166: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_125, 0.5)
    mul_167: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_125, 0.7071067811865476);  add_125 = None
    erf_12: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_126: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_168: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_166, add_126);  mul_166 = add_126 = None
    view_130: "f32[8, 128]" = torch.ops.aten.view.default(mul_168, [8, 128]);  mul_168 = None
    permute_72: "f32[128, 256]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_42: "f32[8, 256]" = torch.ops.aten.addmm.default(arg143_1, view_130, permute_72);  arg143_1 = view_130 = permute_72 = None
    view_131: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_42, [8, 1, 256]);  addmm_42 = None
    slice_25: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_123, 0, 0, 9223372036854775807)
    slice_26: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_25, 1, 0, 1);  slice_25 = None
    clone_37: "f32[8, 1, 256]" = torch.ops.aten.clone.default(slice_26, memory_format = torch.contiguous_format);  slice_26 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_37, [2], correction = 0, keepdim = True)
    getitem_102: "f32[8, 1, 1]" = var_mean_23[0]
    getitem_103: "f32[8, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_127: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
    rsqrt_23: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_71: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(clone_37, getitem_103);  clone_37 = getitem_103 = None
    mul_169: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_23);  sub_71 = rsqrt_23 = None
    mul_170: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_169, arg144_1);  mul_169 = arg144_1 = None
    add_128: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_170, arg145_1);  mul_170 = arg145_1 = None
    mul_171: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_128, 0.5)
    mul_172: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_128, 0.7071067811865476);  add_128 = None
    erf_13: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
    add_129: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_173: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_171, add_129);  mul_171 = add_129 = None
    view_132: "f32[8, 256]" = torch.ops.aten.view.default(mul_173, [8, 256]);  mul_173 = None
    permute_73: "f32[256, 128]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    addmm_43: "f32[8, 128]" = torch.ops.aten.addmm.default(arg147_1, view_132, permute_73);  arg147_1 = view_132 = permute_73 = None
    view_133: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_43, [8, 1, 128]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_27: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_123, 0, 0, 9223372036854775807)
    slice_28: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(slice_27, 1, 1, 9223372036854775807);  slice_27 = None
    cat_6: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_131, slice_28], 1);  view_131 = slice_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_29: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(cat_6, 0, 0, 9223372036854775807)
    slice_30: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_29, 1, 0, 1);  slice_29 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(cat_6, [2], correction = 0, keepdim = True)
    getitem_104: "f32[8, 197, 1]" = var_mean_24[0]
    getitem_105: "f32[8, 197, 1]" = var_mean_24[1];  var_mean_24 = None
    add_130: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
    rsqrt_24: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_72: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_6, getitem_105);  cat_6 = getitem_105 = None
    mul_174: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_24);  sub_72 = rsqrt_24 = None
    mul_175: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_174, arg148_1);  mul_174 = arg148_1 = None
    add_131: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_175, arg149_1);  mul_175 = arg149_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_31: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_131, 0, 0, 9223372036854775807)
    slice_32: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_31, 1, 0, 1);  slice_31 = None
    permute_74: "f32[256, 256]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    view_134: "f32[8, 256]" = torch.ops.aten.view.default(slice_32, [8, 256]);  slice_32 = None
    mm_2: "f32[8, 256]" = torch.ops.aten.mm.default(view_134, permute_74);  view_134 = permute_74 = None
    view_135: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_2, [8, 1, 256]);  mm_2 = None
    add_132: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(view_135, arg151_1);  view_135 = arg151_1 = None
    view_136: "f32[8, 1, 4, 64]" = torch.ops.aten.view.default(add_132, [8, 1, 4, 64]);  add_132 = None
    permute_75: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_137: "f32[1576, 256]" = torch.ops.aten.view.default(add_131, [1576, 256])
    permute_76: "f32[256, 256]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    addmm_44: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg153_1, view_137, permute_76);  arg153_1 = view_137 = permute_76 = None
    view_138: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_44, [8, 197, 256]);  addmm_44 = None
    view_139: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_138, [8, 197, 4, 64]);  view_138 = None
    permute_77: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_140: "f32[1576, 256]" = torch.ops.aten.view.default(add_131, [1576, 256]);  add_131 = None
    permute_78: "f32[256, 256]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_45: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg155_1, view_140, permute_78);  arg155_1 = view_140 = permute_78 = None
    view_141: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_45, [8, 197, 256]);  addmm_45 = None
    view_142: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_141, [8, 197, 4, 64]);  view_141 = None
    permute_79: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_80: "f32[8, 4, 64, 197]" = torch.ops.aten.permute.default(permute_77, [0, 1, 3, 2]);  permute_77 = None
    expand_10: "f32[8, 4, 1, 64]" = torch.ops.aten.expand.default(permute_75, [8, 4, 1, 64]);  permute_75 = None
    view_143: "f32[32, 1, 64]" = torch.ops.aten.view.default(expand_10, [32, 1, 64]);  expand_10 = None
    expand_11: "f32[8, 4, 64, 197]" = torch.ops.aten.expand.default(permute_80, [8, 4, 64, 197]);  permute_80 = None
    clone_38: "f32[8, 4, 64, 197]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_144: "f32[32, 64, 197]" = torch.ops.aten.view.default(clone_38, [32, 64, 197]);  clone_38 = None
    bmm_4: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_143, view_144);  view_143 = view_144 = None
    view_145: "f32[8, 4, 1, 197]" = torch.ops.aten.view.default(bmm_4, [8, 4, 1, 197]);  bmm_4 = None
    mul_176: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(view_145, 0.125);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[8, 4, 1, 1]" = torch.ops.aten.amax.default(mul_176, [-1], True)
    sub_73: "f32[8, 4, 1, 197]" = torch.ops.aten.sub.Tensor(mul_176, amax_2);  mul_176 = amax_2 = None
    exp_2: "f32[8, 4, 1, 197]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
    sum_3: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 4, 1, 197]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    clone_39: "f32[8, 4, 1, 197]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    expand_12: "f32[8, 4, 1, 197]" = torch.ops.aten.expand.default(clone_39, [8, 4, 1, 197]);  clone_39 = None
    view_146: "f32[32, 1, 197]" = torch.ops.aten.view.default(expand_12, [32, 1, 197]);  expand_12 = None
    expand_13: "f32[8, 4, 197, 64]" = torch.ops.aten.expand.default(permute_79, [8, 4, 197, 64]);  permute_79 = None
    clone_40: "f32[8, 4, 197, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_147: "f32[32, 197, 64]" = torch.ops.aten.view.default(clone_40, [32, 197, 64]);  clone_40 = None
    bmm_5: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_146, view_147);  view_146 = view_147 = None
    view_148: "f32[8, 4, 1, 64]" = torch.ops.aten.view.default(bmm_5, [8, 4, 1, 64]);  bmm_5 = None
    permute_81: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    view_149: "f32[8, 1, 256]" = torch.ops.aten.view.default(permute_81, [8, 1, 256]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_150: "f32[8, 256]" = torch.ops.aten.view.default(view_149, [8, 256]);  view_149 = None
    permute_82: "f32[256, 256]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    addmm_46: "f32[8, 256]" = torch.ops.aten.addmm.default(arg157_1, view_150, permute_82);  arg157_1 = view_150 = permute_82 = None
    view_151: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_46, [8, 1, 256]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    clone_41: "f32[8, 1, 256]" = torch.ops.aten.clone.default(view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_133: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(slice_30, clone_41);  slice_30 = clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    slice_33: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_133, 0, 0, 9223372036854775807);  add_133 = None
    var_mean_25 = torch.ops.aten.var_mean.correction(slice_33, [2], correction = 0, keepdim = True)
    getitem_106: "f32[8, 1, 1]" = var_mean_25[0]
    getitem_107: "f32[8, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_134: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
    rsqrt_25: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_74: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(slice_33, getitem_107);  slice_33 = getitem_107 = None
    mul_177: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_25);  sub_74 = rsqrt_25 = None
    mul_178: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_177, arg158_1);  mul_177 = arg158_1 = None
    add_135: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_178, arg159_1);  mul_178 = arg159_1 = None
    mul_179: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_135, 0.5)
    mul_180: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_135, 0.7071067811865476);  add_135 = None
    erf_14: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_180);  mul_180 = None
    add_136: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_181: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_179, add_136);  mul_179 = add_136 = None
    view_152: "f32[8, 256]" = torch.ops.aten.view.default(mul_181, [8, 256]);  mul_181 = None
    permute_83: "f32[256, 128]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_47: "f32[8, 128]" = torch.ops.aten.addmm.default(arg161_1, view_152, permute_83);  arg161_1 = view_152 = permute_83 = None
    view_153: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_47, [8, 1, 128]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_34: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_102, 0, 0, 9223372036854775807)
    slice_35: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(slice_34, 1, 1, 9223372036854775807);  slice_34 = None
    cat_7: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_153, slice_35], 1);  view_153 = slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_36: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_102, 0, 0, 9223372036854775807);  add_102 = None
    slice_37: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(slice_36, 1, 1, 9223372036854775807);  slice_36 = None
    cat_8: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_133, slice_37], 1);  view_133 = slice_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_38: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(cat_8, 0, 0, 9223372036854775807)
    slice_39: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_38, 1, 0, 1);  slice_38 = None
    var_mean_26 = torch.ops.aten.var_mean.correction(cat_8, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 401, 1]" = var_mean_26[0]
    getitem_109: "f32[8, 401, 1]" = var_mean_26[1];  var_mean_26 = None
    add_137: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_26: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_75: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_8, getitem_109);  cat_8 = getitem_109 = None
    mul_182: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_26);  sub_75 = rsqrt_26 = None
    mul_183: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_182, arg162_1);  mul_182 = arg162_1 = None
    add_138: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_183, arg163_1);  mul_183 = arg163_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_40: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_138, 0, 0, 9223372036854775807)
    slice_41: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_40, 1, 0, 1);  slice_40 = None
    permute_84: "f32[128, 128]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    view_154: "f32[8, 128]" = torch.ops.aten.view.default(slice_41, [8, 128]);  slice_41 = None
    mm_3: "f32[8, 128]" = torch.ops.aten.mm.default(view_154, permute_84);  view_154 = permute_84 = None
    view_155: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_3, [8, 1, 128]);  mm_3 = None
    add_139: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(view_155, arg165_1);  view_155 = arg165_1 = None
    view_156: "f32[8, 1, 4, 32]" = torch.ops.aten.view.default(add_139, [8, 1, 4, 32]);  add_139 = None
    permute_85: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_157: "f32[3208, 128]" = torch.ops.aten.view.default(add_138, [3208, 128])
    permute_86: "f32[128, 128]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    addmm_48: "f32[3208, 128]" = torch.ops.aten.addmm.default(arg167_1, view_157, permute_86);  arg167_1 = view_157 = permute_86 = None
    view_158: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_48, [8, 401, 128]);  addmm_48 = None
    view_159: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_158, [8, 401, 4, 32]);  view_158 = None
    permute_87: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_160: "f32[3208, 128]" = torch.ops.aten.view.default(add_138, [3208, 128]);  add_138 = None
    permute_88: "f32[128, 128]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    addmm_49: "f32[3208, 128]" = torch.ops.aten.addmm.default(arg169_1, view_160, permute_88);  arg169_1 = view_160 = permute_88 = None
    view_161: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_49, [8, 401, 128]);  addmm_49 = None
    view_162: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_161, [8, 401, 4, 32]);  view_161 = None
    permute_89: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_90: "f32[8, 4, 32, 401]" = torch.ops.aten.permute.default(permute_87, [0, 1, 3, 2]);  permute_87 = None
    expand_14: "f32[8, 4, 1, 32]" = torch.ops.aten.expand.default(permute_85, [8, 4, 1, 32]);  permute_85 = None
    view_163: "f32[32, 1, 32]" = torch.ops.aten.view.default(expand_14, [32, 1, 32]);  expand_14 = None
    expand_15: "f32[8, 4, 32, 401]" = torch.ops.aten.expand.default(permute_90, [8, 4, 32, 401]);  permute_90 = None
    clone_42: "f32[8, 4, 32, 401]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_164: "f32[32, 32, 401]" = torch.ops.aten.view.default(clone_42, [32, 32, 401]);  clone_42 = None
    bmm_6: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_163, view_164);  view_163 = view_164 = None
    view_165: "f32[8, 4, 1, 401]" = torch.ops.aten.view.default(bmm_6, [8, 4, 1, 401]);  bmm_6 = None
    mul_184: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(view_165, 0.1767766952966369);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[8, 4, 1, 1]" = torch.ops.aten.amax.default(mul_184, [-1], True)
    sub_76: "f32[8, 4, 1, 401]" = torch.ops.aten.sub.Tensor(mul_184, amax_3);  mul_184 = amax_3 = None
    exp_3: "f32[8, 4, 1, 401]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
    sum_4: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 4, 1, 401]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    clone_43: "f32[8, 4, 1, 401]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    expand_16: "f32[8, 4, 1, 401]" = torch.ops.aten.expand.default(clone_43, [8, 4, 1, 401]);  clone_43 = None
    view_166: "f32[32, 1, 401]" = torch.ops.aten.view.default(expand_16, [32, 1, 401]);  expand_16 = None
    expand_17: "f32[8, 4, 401, 32]" = torch.ops.aten.expand.default(permute_89, [8, 4, 401, 32]);  permute_89 = None
    clone_44: "f32[8, 4, 401, 32]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_167: "f32[32, 401, 32]" = torch.ops.aten.view.default(clone_44, [32, 401, 32]);  clone_44 = None
    bmm_7: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
    view_168: "f32[8, 4, 1, 32]" = torch.ops.aten.view.default(bmm_7, [8, 4, 1, 32]);  bmm_7 = None
    permute_91: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    view_169: "f32[8, 1, 128]" = torch.ops.aten.view.default(permute_91, [8, 1, 128]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_170: "f32[8, 128]" = torch.ops.aten.view.default(view_169, [8, 128]);  view_169 = None
    permute_92: "f32[128, 128]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_50: "f32[8, 128]" = torch.ops.aten.addmm.default(arg171_1, view_170, permute_92);  arg171_1 = view_170 = permute_92 = None
    view_171: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_50, [8, 1, 128]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    clone_45: "f32[8, 1, 128]" = torch.ops.aten.clone.default(view_171);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_140: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(slice_39, clone_45);  slice_39 = clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    slice_42: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_140, 0, 0, 9223372036854775807);  add_140 = None
    var_mean_27 = torch.ops.aten.var_mean.correction(slice_42, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 1, 1]" = var_mean_27[0]
    getitem_111: "f32[8, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_141: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_27: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_77: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(slice_42, getitem_111);  slice_42 = getitem_111 = None
    mul_185: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_27);  sub_77 = rsqrt_27 = None
    mul_186: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_185, arg172_1);  mul_185 = arg172_1 = None
    add_142: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_186, arg173_1);  mul_186 = arg173_1 = None
    mul_187: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_142, 0.5)
    mul_188: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_142, 0.7071067811865476);  add_142 = None
    erf_15: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_188);  mul_188 = None
    add_143: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_189: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_187, add_143);  mul_187 = add_143 = None
    view_172: "f32[8, 128]" = torch.ops.aten.view.default(mul_189, [8, 128]);  mul_189 = None
    permute_93: "f32[128, 256]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_51: "f32[8, 256]" = torch.ops.aten.addmm.default(arg175_1, view_172, permute_93);  arg175_1 = view_172 = permute_93 = None
    view_173: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_51, [8, 1, 256]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_43: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_123, 0, 0, 9223372036854775807);  add_123 = None
    slice_44: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(slice_43, 1, 1, 9223372036854775807);  slice_43 = None
    cat_9: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_173, slice_44], 1);  view_173 = slice_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_28 = torch.ops.aten.var_mean.correction(cat_7, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 401, 1]" = var_mean_28[0]
    getitem_113: "f32[8, 401, 1]" = var_mean_28[1];  var_mean_28 = None
    add_144: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
    rsqrt_28: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_78: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_7, getitem_113);  getitem_113 = None
    mul_190: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_28);  sub_78 = rsqrt_28 = None
    mul_191: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_190, arg176_1);  mul_190 = arg176_1 = None
    add_145: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_191, arg177_1);  mul_191 = arg177_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_174: "f32[3208, 128]" = torch.ops.aten.view.default(add_145, [3208, 128]);  add_145 = None
    permute_94: "f32[128, 384]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    addmm_52: "f32[3208, 384]" = torch.ops.aten.addmm.default(arg179_1, view_174, permute_94);  arg179_1 = view_174 = permute_94 = None
    view_175: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_52, [8, 401, 384]);  addmm_52 = None
    view_176: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.view.default(view_175, [8, 401, 3, 4, 32]);  view_175 = None
    permute_95: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.permute.default(view_176, [2, 0, 3, 1, 4]);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_95);  permute_95 = None
    getitem_114: "f32[8, 4, 401, 32]" = unbind_8[0]
    getitem_115: "f32[8, 4, 401, 32]" = unbind_8[1]
    getitem_116: "f32[8, 4, 401, 32]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_114, getitem_115, getitem_116, None, False);  getitem_114 = getitem_115 = getitem_116 = None
    getitem_117: "f32[8, 4, 401, 32]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_96: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
    view_177: "f32[8, 401, 128]" = torch.ops.aten.view.default(permute_96, [8, 401, 128]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_178: "f32[3208, 128]" = torch.ops.aten.view.default(view_177, [3208, 128]);  view_177 = None
    permute_97: "f32[128, 128]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    addmm_53: "f32[3208, 128]" = torch.ops.aten.addmm.default(arg181_1, view_178, permute_97);  arg181_1 = view_178 = permute_97 = None
    view_179: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_53, [8, 401, 128]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_46: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_179);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_146: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(cat_7, clone_46);  cat_7 = clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_29 = torch.ops.aten.var_mean.correction(add_146, [2], correction = 0, keepdim = True)
    getitem_121: "f32[8, 401, 1]" = var_mean_29[0]
    getitem_122: "f32[8, 401, 1]" = var_mean_29[1];  var_mean_29 = None
    add_147: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-06);  getitem_121 = None
    rsqrt_29: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_79: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add_146, getitem_122);  getitem_122 = None
    mul_192: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_29);  sub_79 = rsqrt_29 = None
    mul_193: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_192, arg182_1);  mul_192 = arg182_1 = None
    add_148: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_193, arg183_1);  mul_193 = arg183_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_180: "f32[3208, 128]" = torch.ops.aten.view.default(add_148, [3208, 128]);  add_148 = None
    permute_98: "f32[128, 384]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    addmm_54: "f32[3208, 384]" = torch.ops.aten.addmm.default(arg185_1, view_180, permute_98);  arg185_1 = view_180 = permute_98 = None
    view_181: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_54, [8, 401, 384]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_194: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_181, 0.5)
    mul_195: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476);  view_181 = None
    erf_16: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_149: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_196: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_194, add_149);  mul_194 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_47: "f32[8, 401, 384]" = torch.ops.aten.clone.default(mul_196);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_182: "f32[3208, 384]" = torch.ops.aten.view.default(clone_47, [3208, 384]);  clone_47 = None
    permute_99: "f32[384, 128]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    addmm_55: "f32[3208, 128]" = torch.ops.aten.addmm.default(arg187_1, view_182, permute_99);  arg187_1 = view_182 = permute_99 = None
    view_183: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_55, [8, 401, 128]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_183);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_150: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_146, clone_48);  add_146 = clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_30 = torch.ops.aten.var_mean.correction(cat_9, [2], correction = 0, keepdim = True)
    getitem_123: "f32[8, 197, 1]" = var_mean_30[0]
    getitem_124: "f32[8, 197, 1]" = var_mean_30[1];  var_mean_30 = None
    add_151: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_123, 1e-06);  getitem_123 = None
    rsqrt_30: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_80: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_9, getitem_124);  getitem_124 = None
    mul_197: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_30);  sub_80 = rsqrt_30 = None
    mul_198: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_197, arg188_1);  mul_197 = arg188_1 = None
    add_152: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_198, arg189_1);  mul_198 = arg189_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_184: "f32[1576, 256]" = torch.ops.aten.view.default(add_152, [1576, 256]);  add_152 = None
    permute_100: "f32[256, 768]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
    addmm_56: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg191_1, view_184, permute_100);  arg191_1 = view_184 = permute_100 = None
    view_185: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_56, [8, 197, 768]);  addmm_56 = None
    view_186: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_185, [8, 197, 3, 4, 64]);  view_185 = None
    permute_101: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_186, [2, 0, 3, 1, 4]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_101);  permute_101 = None
    getitem_125: "f32[8, 4, 197, 64]" = unbind_9[0]
    getitem_126: "f32[8, 4, 197, 64]" = unbind_9[1]
    getitem_127: "f32[8, 4, 197, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_125, getitem_126, getitem_127, None, False);  getitem_125 = getitem_126 = getitem_127 = None
    getitem_128: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_102: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_128, [0, 2, 1, 3]);  getitem_128 = None
    view_187: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_102, [8, 197, 256]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_188: "f32[1576, 256]" = torch.ops.aten.view.default(view_187, [1576, 256]);  view_187 = None
    permute_103: "f32[256, 256]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
    addmm_57: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg193_1, view_188, permute_103);  arg193_1 = view_188 = permute_103 = None
    view_189: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_57, [8, 197, 256]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_49: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_189);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_153: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(cat_9, clone_49);  cat_9 = clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_31 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 197, 1]" = var_mean_31[0]
    getitem_133: "f32[8, 197, 1]" = var_mean_31[1];  var_mean_31 = None
    add_154: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_31: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_81: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_153, getitem_133);  getitem_133 = None
    mul_199: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_31);  sub_81 = rsqrt_31 = None
    mul_200: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_199, arg194_1);  mul_199 = arg194_1 = None
    add_155: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_200, arg195_1);  mul_200 = arg195_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_190: "f32[1576, 256]" = torch.ops.aten.view.default(add_155, [1576, 256]);  add_155 = None
    permute_104: "f32[256, 768]" = torch.ops.aten.permute.default(arg196_1, [1, 0]);  arg196_1 = None
    addmm_58: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg197_1, view_190, permute_104);  arg197_1 = view_190 = permute_104 = None
    view_191: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_58, [8, 197, 768]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_201: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_191, 0.5)
    mul_202: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_191, 0.7071067811865476);  view_191 = None
    erf_17: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_202);  mul_202 = None
    add_156: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_203: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_201, add_156);  mul_201 = add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_50: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_203);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_192: "f32[1576, 768]" = torch.ops.aten.view.default(clone_50, [1576, 768]);  clone_50 = None
    permute_105: "f32[768, 256]" = torch.ops.aten.permute.default(arg198_1, [1, 0]);  arg198_1 = None
    addmm_59: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg199_1, view_192, permute_105);  arg199_1 = view_192 = permute_105 = None
    view_193: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_59, [8, 197, 256]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_51: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_193);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_157: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_153, clone_51);  add_153 = clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_32 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
    getitem_134: "f32[8, 197, 1]" = var_mean_32[0]
    getitem_135: "f32[8, 197, 1]" = var_mean_32[1];  var_mean_32 = None
    add_158: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
    rsqrt_32: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_82: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_157, getitem_135);  getitem_135 = None
    mul_204: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_32);  sub_82 = rsqrt_32 = None
    mul_205: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_204, arg200_1);  mul_204 = arg200_1 = None
    add_159: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_205, arg201_1);  mul_205 = arg201_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_194: "f32[1576, 256]" = torch.ops.aten.view.default(add_159, [1576, 256]);  add_159 = None
    permute_106: "f32[256, 768]" = torch.ops.aten.permute.default(arg202_1, [1, 0]);  arg202_1 = None
    addmm_60: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg203_1, view_194, permute_106);  arg203_1 = view_194 = permute_106 = None
    view_195: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_60, [8, 197, 768]);  addmm_60 = None
    view_196: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_195, [8, 197, 3, 4, 64]);  view_195 = None
    permute_107: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_196, [2, 0, 3, 1, 4]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_107);  permute_107 = None
    getitem_136: "f32[8, 4, 197, 64]" = unbind_10[0]
    getitem_137: "f32[8, 4, 197, 64]" = unbind_10[1]
    getitem_138: "f32[8, 4, 197, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_136, getitem_137, getitem_138, None, False);  getitem_136 = getitem_137 = getitem_138 = None
    getitem_139: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_108: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
    view_197: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_108, [8, 197, 256]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_198: "f32[1576, 256]" = torch.ops.aten.view.default(view_197, [1576, 256]);  view_197 = None
    permute_109: "f32[256, 256]" = torch.ops.aten.permute.default(arg204_1, [1, 0]);  arg204_1 = None
    addmm_61: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg205_1, view_198, permute_109);  arg205_1 = view_198 = permute_109 = None
    view_199: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_61, [8, 197, 256]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_52: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_199);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_160: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_157, clone_52);  add_157 = clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_33 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
    getitem_143: "f32[8, 197, 1]" = var_mean_33[0]
    getitem_144: "f32[8, 197, 1]" = var_mean_33[1];  var_mean_33 = None
    add_161: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_143, 1e-06);  getitem_143 = None
    rsqrt_33: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_83: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_160, getitem_144);  getitem_144 = None
    mul_206: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_33);  sub_83 = rsqrt_33 = None
    mul_207: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_206, arg206_1);  mul_206 = arg206_1 = None
    add_162: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_207, arg207_1);  mul_207 = arg207_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_200: "f32[1576, 256]" = torch.ops.aten.view.default(add_162, [1576, 256]);  add_162 = None
    permute_110: "f32[256, 768]" = torch.ops.aten.permute.default(arg208_1, [1, 0]);  arg208_1 = None
    addmm_62: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg209_1, view_200, permute_110);  arg209_1 = view_200 = permute_110 = None
    view_201: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_62, [8, 197, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_208: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_201, 0.5)
    mul_209: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_201, 0.7071067811865476);  view_201 = None
    erf_18: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_209);  mul_209 = None
    add_163: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_210: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_208, add_163);  mul_208 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_53: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_210);  mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_202: "f32[1576, 768]" = torch.ops.aten.view.default(clone_53, [1576, 768]);  clone_53 = None
    permute_111: "f32[768, 256]" = torch.ops.aten.permute.default(arg210_1, [1, 0]);  arg210_1 = None
    addmm_63: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg211_1, view_202, permute_111);  arg211_1 = view_202 = permute_111 = None
    view_203: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_63, [8, 197, 256]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_54: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_203);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_164: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_160, clone_54);  add_160 = clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_34 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
    getitem_145: "f32[8, 197, 1]" = var_mean_34[0]
    getitem_146: "f32[8, 197, 1]" = var_mean_34[1];  var_mean_34 = None
    add_165: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_145, 1e-06);  getitem_145 = None
    rsqrt_34: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_84: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_164, getitem_146);  getitem_146 = None
    mul_211: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_34);  sub_84 = rsqrt_34 = None
    mul_212: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_211, arg212_1);  mul_211 = arg212_1 = None
    add_166: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_212, arg213_1);  mul_212 = arg213_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_204: "f32[1576, 256]" = torch.ops.aten.view.default(add_166, [1576, 256]);  add_166 = None
    permute_112: "f32[256, 768]" = torch.ops.aten.permute.default(arg214_1, [1, 0]);  arg214_1 = None
    addmm_64: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg215_1, view_204, permute_112);  arg215_1 = view_204 = permute_112 = None
    view_205: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_64, [8, 197, 768]);  addmm_64 = None
    view_206: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_205, [8, 197, 3, 4, 64]);  view_205 = None
    permute_113: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_206, [2, 0, 3, 1, 4]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_113);  permute_113 = None
    getitem_147: "f32[8, 4, 197, 64]" = unbind_11[0]
    getitem_148: "f32[8, 4, 197, 64]" = unbind_11[1]
    getitem_149: "f32[8, 4, 197, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_147, getitem_148, getitem_149, None, False);  getitem_147 = getitem_148 = getitem_149 = None
    getitem_150: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_114: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_150, [0, 2, 1, 3]);  getitem_150 = None
    view_207: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_114, [8, 197, 256]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_208: "f32[1576, 256]" = torch.ops.aten.view.default(view_207, [1576, 256]);  view_207 = None
    permute_115: "f32[256, 256]" = torch.ops.aten.permute.default(arg216_1, [1, 0]);  arg216_1 = None
    addmm_65: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg217_1, view_208, permute_115);  arg217_1 = view_208 = permute_115 = None
    view_209: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_65, [8, 197, 256]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_55: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_209);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_167: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_164, clone_55);  add_164 = clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_35 = torch.ops.aten.var_mean.correction(add_167, [2], correction = 0, keepdim = True)
    getitem_154: "f32[8, 197, 1]" = var_mean_35[0]
    getitem_155: "f32[8, 197, 1]" = var_mean_35[1];  var_mean_35 = None
    add_168: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
    rsqrt_35: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_85: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_167, getitem_155);  getitem_155 = None
    mul_213: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_35);  sub_85 = rsqrt_35 = None
    mul_214: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_213, arg218_1);  mul_213 = arg218_1 = None
    add_169: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_214, arg219_1);  mul_214 = arg219_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_210: "f32[1576, 256]" = torch.ops.aten.view.default(add_169, [1576, 256]);  add_169 = None
    permute_116: "f32[256, 768]" = torch.ops.aten.permute.default(arg220_1, [1, 0]);  arg220_1 = None
    addmm_66: "f32[1576, 768]" = torch.ops.aten.addmm.default(arg221_1, view_210, permute_116);  arg221_1 = view_210 = permute_116 = None
    view_211: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_66, [8, 197, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_215: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, 0.5)
    mul_216: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, 0.7071067811865476);  view_211 = None
    erf_19: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_216);  mul_216 = None
    add_170: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_217: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_215, add_170);  mul_215 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_56: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_217);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_212: "f32[1576, 768]" = torch.ops.aten.view.default(clone_56, [1576, 768]);  clone_56 = None
    permute_117: "f32[768, 256]" = torch.ops.aten.permute.default(arg222_1, [1, 0]);  arg222_1 = None
    addmm_67: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg223_1, view_212, permute_117);  arg223_1 = view_212 = permute_117 = None
    view_213: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_67, [8, 197, 256]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_57: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_213);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_171: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_167, clone_57);  add_167 = clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    slice_45: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_150, 0, 0, 9223372036854775807)
    slice_46: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_45, 1, 0, 1);  slice_45 = None
    clone_58: "f32[8, 1, 128]" = torch.ops.aten.clone.default(slice_46, memory_format = torch.contiguous_format);  slice_46 = None
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_58, [2], correction = 0, keepdim = True)
    getitem_156: "f32[8, 1, 1]" = var_mean_36[0]
    getitem_157: "f32[8, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_172: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
    rsqrt_36: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_86: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(clone_58, getitem_157);  clone_58 = getitem_157 = None
    mul_218: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_36);  sub_86 = rsqrt_36 = None
    mul_219: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_218, arg224_1);  mul_218 = arg224_1 = None
    add_173: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_219, arg225_1);  mul_219 = arg225_1 = None
    mul_220: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_173, 0.5)
    mul_221: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_173, 0.7071067811865476);  add_173 = None
    erf_20: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
    add_174: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_222: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_220, add_174);  mul_220 = add_174 = None
    view_214: "f32[8, 128]" = torch.ops.aten.view.default(mul_222, [8, 128]);  mul_222 = None
    permute_118: "f32[128, 256]" = torch.ops.aten.permute.default(arg226_1, [1, 0]);  arg226_1 = None
    addmm_68: "f32[8, 256]" = torch.ops.aten.addmm.default(arg227_1, view_214, permute_118);  arg227_1 = view_214 = permute_118 = None
    view_215: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_68, [8, 1, 256]);  addmm_68 = None
    slice_47: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_171, 0, 0, 9223372036854775807)
    slice_48: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_47, 1, 0, 1);  slice_47 = None
    clone_59: "f32[8, 1, 256]" = torch.ops.aten.clone.default(slice_48, memory_format = torch.contiguous_format);  slice_48 = None
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_59, [2], correction = 0, keepdim = True)
    getitem_158: "f32[8, 1, 1]" = var_mean_37[0]
    getitem_159: "f32[8, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_175: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
    rsqrt_37: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_87: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(clone_59, getitem_159);  clone_59 = getitem_159 = None
    mul_223: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_37);  sub_87 = rsqrt_37 = None
    mul_224: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_223, arg228_1);  mul_223 = arg228_1 = None
    add_176: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_224, arg229_1);  mul_224 = arg229_1 = None
    mul_225: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_176, 0.5)
    mul_226: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_176, 0.7071067811865476);  add_176 = None
    erf_21: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_226);  mul_226 = None
    add_177: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_227: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_225, add_177);  mul_225 = add_177 = None
    view_216: "f32[8, 256]" = torch.ops.aten.view.default(mul_227, [8, 256]);  mul_227 = None
    permute_119: "f32[256, 128]" = torch.ops.aten.permute.default(arg230_1, [1, 0]);  arg230_1 = None
    addmm_69: "f32[8, 128]" = torch.ops.aten.addmm.default(arg231_1, view_216, permute_119);  arg231_1 = view_216 = permute_119 = None
    view_217: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_69, [8, 1, 128]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_49: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_171, 0, 0, 9223372036854775807)
    slice_50: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(slice_49, 1, 1, 9223372036854775807);  slice_49 = None
    cat_10: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_215, slice_50], 1);  view_215 = slice_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_51: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(cat_10, 0, 0, 9223372036854775807)
    slice_52: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_51, 1, 0, 1);  slice_51 = None
    var_mean_38 = torch.ops.aten.var_mean.correction(cat_10, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 197, 1]" = var_mean_38[0]
    getitem_161: "f32[8, 197, 1]" = var_mean_38[1];  var_mean_38 = None
    add_178: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_38: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_88: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_10, getitem_161);  cat_10 = getitem_161 = None
    mul_228: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_38);  sub_88 = rsqrt_38 = None
    mul_229: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_228, arg232_1);  mul_228 = arg232_1 = None
    add_179: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_229, arg233_1);  mul_229 = arg233_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_53: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_179, 0, 0, 9223372036854775807)
    slice_54: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_53, 1, 0, 1);  slice_53 = None
    permute_120: "f32[256, 256]" = torch.ops.aten.permute.default(arg234_1, [1, 0]);  arg234_1 = None
    view_218: "f32[8, 256]" = torch.ops.aten.view.default(slice_54, [8, 256]);  slice_54 = None
    mm_4: "f32[8, 256]" = torch.ops.aten.mm.default(view_218, permute_120);  view_218 = permute_120 = None
    view_219: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_4, [8, 1, 256]);  mm_4 = None
    add_180: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(view_219, arg235_1);  view_219 = arg235_1 = None
    view_220: "f32[8, 1, 4, 64]" = torch.ops.aten.view.default(add_180, [8, 1, 4, 64]);  add_180 = None
    permute_121: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_221: "f32[1576, 256]" = torch.ops.aten.view.default(add_179, [1576, 256])
    permute_122: "f32[256, 256]" = torch.ops.aten.permute.default(arg236_1, [1, 0]);  arg236_1 = None
    addmm_70: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg237_1, view_221, permute_122);  arg237_1 = view_221 = permute_122 = None
    view_222: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_70, [8, 197, 256]);  addmm_70 = None
    view_223: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_222, [8, 197, 4, 64]);  view_222 = None
    permute_123: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_224: "f32[1576, 256]" = torch.ops.aten.view.default(add_179, [1576, 256]);  add_179 = None
    permute_124: "f32[256, 256]" = torch.ops.aten.permute.default(arg238_1, [1, 0]);  arg238_1 = None
    addmm_71: "f32[1576, 256]" = torch.ops.aten.addmm.default(arg239_1, view_224, permute_124);  arg239_1 = view_224 = permute_124 = None
    view_225: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_71, [8, 197, 256]);  addmm_71 = None
    view_226: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_225, [8, 197, 4, 64]);  view_225 = None
    permute_125: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_126: "f32[8, 4, 64, 197]" = torch.ops.aten.permute.default(permute_123, [0, 1, 3, 2]);  permute_123 = None
    expand_18: "f32[8, 4, 1, 64]" = torch.ops.aten.expand.default(permute_121, [8, 4, 1, 64]);  permute_121 = None
    view_227: "f32[32, 1, 64]" = torch.ops.aten.view.default(expand_18, [32, 1, 64]);  expand_18 = None
    expand_19: "f32[8, 4, 64, 197]" = torch.ops.aten.expand.default(permute_126, [8, 4, 64, 197]);  permute_126 = None
    clone_60: "f32[8, 4, 64, 197]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_228: "f32[32, 64, 197]" = torch.ops.aten.view.default(clone_60, [32, 64, 197]);  clone_60 = None
    bmm_8: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_227, view_228);  view_227 = view_228 = None
    view_229: "f32[8, 4, 1, 197]" = torch.ops.aten.view.default(bmm_8, [8, 4, 1, 197]);  bmm_8 = None
    mul_230: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(view_229, 0.125);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[8, 4, 1, 1]" = torch.ops.aten.amax.default(mul_230, [-1], True)
    sub_89: "f32[8, 4, 1, 197]" = torch.ops.aten.sub.Tensor(mul_230, amax_4);  mul_230 = amax_4 = None
    exp_4: "f32[8, 4, 1, 197]" = torch.ops.aten.exp.default(sub_89);  sub_89 = None
    sum_5: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 4, 1, 197]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    clone_61: "f32[8, 4, 1, 197]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    expand_20: "f32[8, 4, 1, 197]" = torch.ops.aten.expand.default(clone_61, [8, 4, 1, 197]);  clone_61 = None
    view_230: "f32[32, 1, 197]" = torch.ops.aten.view.default(expand_20, [32, 1, 197]);  expand_20 = None
    expand_21: "f32[8, 4, 197, 64]" = torch.ops.aten.expand.default(permute_125, [8, 4, 197, 64]);  permute_125 = None
    clone_62: "f32[8, 4, 197, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_231: "f32[32, 197, 64]" = torch.ops.aten.view.default(clone_62, [32, 197, 64]);  clone_62 = None
    bmm_9: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_230, view_231);  view_230 = view_231 = None
    view_232: "f32[8, 4, 1, 64]" = torch.ops.aten.view.default(bmm_9, [8, 4, 1, 64]);  bmm_9 = None
    permute_127: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
    view_233: "f32[8, 1, 256]" = torch.ops.aten.view.default(permute_127, [8, 1, 256]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_234: "f32[8, 256]" = torch.ops.aten.view.default(view_233, [8, 256]);  view_233 = None
    permute_128: "f32[256, 256]" = torch.ops.aten.permute.default(arg240_1, [1, 0]);  arg240_1 = None
    addmm_72: "f32[8, 256]" = torch.ops.aten.addmm.default(arg241_1, view_234, permute_128);  arg241_1 = view_234 = permute_128 = None
    view_235: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_72, [8, 1, 256]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    clone_63: "f32[8, 1, 256]" = torch.ops.aten.clone.default(view_235);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_181: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(slice_52, clone_63);  slice_52 = clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    slice_55: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_181, 0, 0, 9223372036854775807);  add_181 = None
    var_mean_39 = torch.ops.aten.var_mean.correction(slice_55, [2], correction = 0, keepdim = True)
    getitem_162: "f32[8, 1, 1]" = var_mean_39[0]
    getitem_163: "f32[8, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_182: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
    rsqrt_39: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_90: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(slice_55, getitem_163);  slice_55 = getitem_163 = None
    mul_231: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_39);  sub_90 = rsqrt_39 = None
    mul_232: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_231, arg242_1);  mul_231 = arg242_1 = None
    add_183: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_232, arg243_1);  mul_232 = arg243_1 = None
    mul_233: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_183, 0.5)
    mul_234: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_183, 0.7071067811865476);  add_183 = None
    erf_22: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_234);  mul_234 = None
    add_184: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_235: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_233, add_184);  mul_233 = add_184 = None
    view_236: "f32[8, 256]" = torch.ops.aten.view.default(mul_235, [8, 256]);  mul_235 = None
    permute_129: "f32[256, 128]" = torch.ops.aten.permute.default(arg244_1, [1, 0]);  arg244_1 = None
    addmm_73: "f32[8, 128]" = torch.ops.aten.addmm.default(arg245_1, view_236, permute_129);  arg245_1 = view_236 = permute_129 = None
    view_237: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_73, [8, 1, 128]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_56: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_150, 0, 0, 9223372036854775807)
    slice_57: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(slice_56, 1, 1, 9223372036854775807);  slice_56 = None
    cat_11: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_237, slice_57], 1);  view_237 = slice_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_58: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_150, 0, 0, 9223372036854775807);  add_150 = None
    slice_59: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(slice_58, 1, 1, 9223372036854775807);  slice_58 = None
    cat_12: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_217, slice_59], 1);  view_217 = slice_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_60: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(cat_12, 0, 0, 9223372036854775807)
    slice_61: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_60, 1, 0, 1);  slice_60 = None
    var_mean_40 = torch.ops.aten.var_mean.correction(cat_12, [2], correction = 0, keepdim = True)
    getitem_164: "f32[8, 401, 1]" = var_mean_40[0]
    getitem_165: "f32[8, 401, 1]" = var_mean_40[1];  var_mean_40 = None
    add_185: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-06);  getitem_164 = None
    rsqrt_40: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_91: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_12, getitem_165);  cat_12 = getitem_165 = None
    mul_236: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_40);  sub_91 = rsqrt_40 = None
    mul_237: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_236, arg246_1);  mul_236 = arg246_1 = None
    add_186: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_237, arg247_1);  mul_237 = arg247_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_62: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_186, 0, 0, 9223372036854775807)
    slice_63: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_62, 1, 0, 1);  slice_62 = None
    permute_130: "f32[128, 128]" = torch.ops.aten.permute.default(arg248_1, [1, 0]);  arg248_1 = None
    view_238: "f32[8, 128]" = torch.ops.aten.view.default(slice_63, [8, 128]);  slice_63 = None
    mm_5: "f32[8, 128]" = torch.ops.aten.mm.default(view_238, permute_130);  view_238 = permute_130 = None
    view_239: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_5, [8, 1, 128]);  mm_5 = None
    add_187: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(view_239, arg249_1);  view_239 = arg249_1 = None
    view_240: "f32[8, 1, 4, 32]" = torch.ops.aten.view.default(add_187, [8, 1, 4, 32]);  add_187 = None
    permute_131: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_241: "f32[3208, 128]" = torch.ops.aten.view.default(add_186, [3208, 128])
    permute_132: "f32[128, 128]" = torch.ops.aten.permute.default(arg250_1, [1, 0]);  arg250_1 = None
    addmm_74: "f32[3208, 128]" = torch.ops.aten.addmm.default(arg251_1, view_241, permute_132);  arg251_1 = view_241 = permute_132 = None
    view_242: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_74, [8, 401, 128]);  addmm_74 = None
    view_243: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_242, [8, 401, 4, 32]);  view_242 = None
    permute_133: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_244: "f32[3208, 128]" = torch.ops.aten.view.default(add_186, [3208, 128]);  add_186 = None
    permute_134: "f32[128, 128]" = torch.ops.aten.permute.default(arg252_1, [1, 0]);  arg252_1 = None
    addmm_75: "f32[3208, 128]" = torch.ops.aten.addmm.default(arg253_1, view_244, permute_134);  arg253_1 = view_244 = permute_134 = None
    view_245: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_75, [8, 401, 128]);  addmm_75 = None
    view_246: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_245, [8, 401, 4, 32]);  view_245 = None
    permute_135: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_136: "f32[8, 4, 32, 401]" = torch.ops.aten.permute.default(permute_133, [0, 1, 3, 2]);  permute_133 = None
    expand_22: "f32[8, 4, 1, 32]" = torch.ops.aten.expand.default(permute_131, [8, 4, 1, 32]);  permute_131 = None
    view_247: "f32[32, 1, 32]" = torch.ops.aten.view.default(expand_22, [32, 1, 32]);  expand_22 = None
    expand_23: "f32[8, 4, 32, 401]" = torch.ops.aten.expand.default(permute_136, [8, 4, 32, 401]);  permute_136 = None
    clone_64: "f32[8, 4, 32, 401]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_248: "f32[32, 32, 401]" = torch.ops.aten.view.default(clone_64, [32, 32, 401]);  clone_64 = None
    bmm_10: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_247, view_248);  view_247 = view_248 = None
    view_249: "f32[8, 4, 1, 401]" = torch.ops.aten.view.default(bmm_10, [8, 4, 1, 401]);  bmm_10 = None
    mul_238: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(view_249, 0.1767766952966369);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 4, 1, 1]" = torch.ops.aten.amax.default(mul_238, [-1], True)
    sub_92: "f32[8, 4, 1, 401]" = torch.ops.aten.sub.Tensor(mul_238, amax_5);  mul_238 = amax_5 = None
    exp_5: "f32[8, 4, 1, 401]" = torch.ops.aten.exp.default(sub_92);  sub_92 = None
    sum_6: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 4, 1, 401]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    clone_65: "f32[8, 4, 1, 401]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    expand_24: "f32[8, 4, 1, 401]" = torch.ops.aten.expand.default(clone_65, [8, 4, 1, 401]);  clone_65 = None
    view_250: "f32[32, 1, 401]" = torch.ops.aten.view.default(expand_24, [32, 1, 401]);  expand_24 = None
    expand_25: "f32[8, 4, 401, 32]" = torch.ops.aten.expand.default(permute_135, [8, 4, 401, 32]);  permute_135 = None
    clone_66: "f32[8, 4, 401, 32]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_251: "f32[32, 401, 32]" = torch.ops.aten.view.default(clone_66, [32, 401, 32]);  clone_66 = None
    bmm_11: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_250, view_251);  view_250 = view_251 = None
    view_252: "f32[8, 4, 1, 32]" = torch.ops.aten.view.default(bmm_11, [8, 4, 1, 32]);  bmm_11 = None
    permute_137: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    view_253: "f32[8, 1, 128]" = torch.ops.aten.view.default(permute_137, [8, 1, 128]);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_254: "f32[8, 128]" = torch.ops.aten.view.default(view_253, [8, 128]);  view_253 = None
    permute_138: "f32[128, 128]" = torch.ops.aten.permute.default(arg254_1, [1, 0]);  arg254_1 = None
    addmm_76: "f32[8, 128]" = torch.ops.aten.addmm.default(arg255_1, view_254, permute_138);  arg255_1 = view_254 = permute_138 = None
    view_255: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_76, [8, 1, 128]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    clone_67: "f32[8, 1, 128]" = torch.ops.aten.clone.default(view_255);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_188: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(slice_61, clone_67);  slice_61 = clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    slice_64: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_188, 0, 0, 9223372036854775807);  add_188 = None
    var_mean_41 = torch.ops.aten.var_mean.correction(slice_64, [2], correction = 0, keepdim = True)
    getitem_166: "f32[8, 1, 1]" = var_mean_41[0]
    getitem_167: "f32[8, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_189: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-06);  getitem_166 = None
    rsqrt_41: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_93: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(slice_64, getitem_167);  slice_64 = getitem_167 = None
    mul_239: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_41);  sub_93 = rsqrt_41 = None
    mul_240: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_239, arg256_1);  mul_239 = arg256_1 = None
    add_190: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_240, arg257_1);  mul_240 = arg257_1 = None
    mul_241: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_190, 0.5)
    mul_242: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_190, 0.7071067811865476);  add_190 = None
    erf_23: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_242);  mul_242 = None
    add_191: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_243: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_241, add_191);  mul_241 = add_191 = None
    view_256: "f32[8, 128]" = torch.ops.aten.view.default(mul_243, [8, 128]);  mul_243 = None
    permute_139: "f32[128, 256]" = torch.ops.aten.permute.default(arg258_1, [1, 0]);  arg258_1 = None
    addmm_77: "f32[8, 256]" = torch.ops.aten.addmm.default(arg259_1, view_256, permute_139);  arg259_1 = view_256 = permute_139 = None
    view_257: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_77, [8, 1, 256]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_65: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_171, 0, 0, 9223372036854775807);  add_171 = None
    slice_66: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(slice_65, 1, 1, 9223372036854775807);  slice_65 = None
    cat_13: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_257, slice_66], 1);  view_257 = slice_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:451, code: xs = [norm(xs[i]) for i, norm in enumerate(self.norm)]
    var_mean_42 = torch.ops.aten.var_mean.correction(cat_11, [2], correction = 0, keepdim = True)
    getitem_168: "f32[8, 401, 1]" = var_mean_42[0]
    getitem_169: "f32[8, 401, 1]" = var_mean_42[1];  var_mean_42 = None
    add_192: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
    rsqrt_42: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_94: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_11, getitem_169);  cat_11 = getitem_169 = None
    mul_244: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_42);  sub_94 = rsqrt_42 = None
    mul_245: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_244, arg260_1);  mul_244 = arg260_1 = None
    add_193: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_245, arg261_1);  mul_245 = arg261_1 = None
    var_mean_43 = torch.ops.aten.var_mean.correction(cat_13, [2], correction = 0, keepdim = True)
    getitem_170: "f32[8, 197, 1]" = var_mean_43[0]
    getitem_171: "f32[8, 197, 1]" = var_mean_43[1];  var_mean_43 = None
    add_194: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-06);  getitem_170 = None
    rsqrt_43: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_95: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_13, getitem_171);  cat_13 = getitem_171 = None
    mul_246: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_43);  sub_95 = rsqrt_43 = None
    mul_247: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_246, arg262_1);  mul_246 = arg262_1 = None
    add_195: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_247, arg263_1);  mul_247 = arg263_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:455, code: xs = [x[:, 1:].mean(dim=1) for x in xs] if self.global_pool == 'avg' else [x[:, 0] for x in xs]
    slice_67: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_193, 0, 0, 9223372036854775807);  add_193 = None
    select: "f32[8, 128]" = torch.ops.aten.select.int(slice_67, 1, 0);  slice_67 = None
    slice_68: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_195, 0, 0, 9223372036854775807);  add_195 = None
    select_1: "f32[8, 256]" = torch.ops.aten.select.int(slice_68, 1, 0);  slice_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:456, code: xs = [self.head_drop(x) for x in xs]
    clone_68: "f32[8, 128]" = torch.ops.aten.clone.default(select);  select = None
    clone_69: "f32[8, 256]" = torch.ops.aten.clone.default(select_1);  select_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:459, code: return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)
    permute_140: "f32[128, 1000]" = torch.ops.aten.permute.default(arg264_1, [1, 0]);  arg264_1 = None
    addmm_78: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg265_1, clone_68, permute_140);  arg265_1 = clone_68 = permute_140 = None
    permute_141: "f32[256, 1000]" = torch.ops.aten.permute.default(arg266_1, [1, 0]);  arg266_1 = None
    addmm_79: "f32[8, 1000]" = torch.ops.aten.addmm.default(arg267_1, clone_69, permute_141);  arg267_1 = clone_69 = permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:459, code: return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)
    cat_14: "f32[16, 1000]" = torch.ops.aten.cat.default([addmm_78, addmm_79]);  addmm_78 = addmm_79 = None
    view_258: "f32[2, 8, 1000]" = torch.ops.aten.view.default(cat_14, [2, 8, 1000]);  cat_14 = None
    mean: "f32[8, 1000]" = torch.ops.aten.mean.dim(view_258, [0]);  view_258 = None
    return (mean,)
    