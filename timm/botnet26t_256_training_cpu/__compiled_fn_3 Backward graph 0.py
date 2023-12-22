from __future__ import annotations



def forward(self, primals_1: "f32[24]", primals_3: "f32[32]", primals_5: "f32[64]", primals_7: "f32[64]", primals_9: "f32[64]", primals_11: "f32[256]", primals_13: "f32[256]", primals_15: "f32[64]", primals_17: "f32[64]", primals_19: "f32[256]", primals_21: "f32[128]", primals_23: "f32[128]", primals_25: "f32[512]", primals_27: "f32[512]", primals_29: "f32[128]", primals_31: "f32[128]", primals_33: "f32[512]", primals_35: "f32[256]", primals_37: "f32[256]", primals_39: "f32[1024]", primals_41: "f32[1024]", primals_43: "f32[256]", primals_47: "f32[256]", primals_49: "f32[1024]", primals_51: "f32[512]", primals_55: "f32[512]", primals_57: "f32[2048]", primals_59: "f32[2048]", primals_61: "f32[512]", primals_65: "f32[512]", primals_67: "f32[2048]", primals_69: "f32[24, 3, 3, 3]", primals_70: "f32[32, 24, 3, 3]", primals_71: "f32[64, 32, 3, 3]", primals_72: "f32[64, 64, 1, 1]", primals_73: "f32[64, 64, 3, 3]", primals_74: "f32[256, 64, 1, 1]", primals_75: "f32[256, 64, 1, 1]", primals_76: "f32[64, 256, 1, 1]", primals_77: "f32[64, 64, 3, 3]", primals_78: "f32[256, 64, 1, 1]", primals_79: "f32[128, 256, 1, 1]", primals_80: "f32[128, 128, 3, 3]", primals_81: "f32[512, 128, 1, 1]", primals_82: "f32[512, 256, 1, 1]", primals_83: "f32[128, 512, 1, 1]", primals_84: "f32[128, 128, 3, 3]", primals_85: "f32[512, 128, 1, 1]", primals_86: "f32[256, 512, 1, 1]", primals_87: "f32[256, 256, 3, 3]", primals_88: "f32[1024, 256, 1, 1]", primals_89: "f32[1024, 512, 1, 1]", primals_90: "f32[256, 1024, 1, 1]", primals_91: "f32[768, 256, 1, 1]", primals_92: "f32[1024, 256, 1, 1]", primals_93: "f32[512, 1024, 1, 1]", primals_94: "f32[1536, 512, 1, 1]", primals_95: "f32[2048, 512, 1, 1]", primals_96: "f32[2048, 1024, 1, 1]", primals_97: "f32[512, 2048, 1, 1]", primals_98: "f32[1536, 512, 1, 1]", primals_99: "f32[2048, 512, 1, 1]", primals_195: "f32[8, 3, 256, 256]", convolution: "f32[8, 24, 128, 128]", squeeze_1: "f32[24]", relu: "f32[8, 24, 128, 128]", convolution_1: "f32[8, 32, 128, 128]", squeeze_4: "f32[32]", relu_1: "f32[8, 32, 128, 128]", convolution_2: "f32[8, 64, 128, 128]", squeeze_7: "f32[64]", relu_2: "f32[8, 64, 128, 128]", getitem_6: "f32[8, 64, 64, 64]", getitem_7: "i64[8, 64, 64, 64]", convolution_3: "f32[8, 64, 64, 64]", squeeze_10: "f32[64]", relu_3: "f32[8, 64, 64, 64]", convolution_4: "f32[8, 64, 64, 64]", squeeze_13: "f32[64]", relu_4: "f32[8, 64, 64, 64]", convolution_5: "f32[8, 256, 64, 64]", squeeze_16: "f32[256]", convolution_6: "f32[8, 256, 64, 64]", squeeze_19: "f32[256]", relu_5: "f32[8, 256, 64, 64]", convolution_7: "f32[8, 64, 64, 64]", squeeze_22: "f32[64]", relu_6: "f32[8, 64, 64, 64]", convolution_8: "f32[8, 64, 64, 64]", squeeze_25: "f32[64]", relu_7: "f32[8, 64, 64, 64]", convolution_9: "f32[8, 256, 64, 64]", squeeze_28: "f32[256]", relu_8: "f32[8, 256, 64, 64]", convolution_10: "f32[8, 128, 64, 64]", squeeze_31: "f32[128]", relu_9: "f32[8, 128, 64, 64]", convolution_11: "f32[8, 128, 32, 32]", squeeze_34: "f32[128]", relu_10: "f32[8, 128, 32, 32]", convolution_12: "f32[8, 512, 32, 32]", squeeze_37: "f32[512]", convolution_13: "f32[8, 512, 32, 32]", squeeze_40: "f32[512]", relu_11: "f32[8, 512, 32, 32]", convolution_14: "f32[8, 128, 32, 32]", squeeze_43: "f32[128]", relu_12: "f32[8, 128, 32, 32]", convolution_15: "f32[8, 128, 32, 32]", squeeze_46: "f32[128]", relu_13: "f32[8, 128, 32, 32]", convolution_16: "f32[8, 512, 32, 32]", squeeze_49: "f32[512]", relu_14: "f32[8, 512, 32, 32]", convolution_17: "f32[8, 256, 32, 32]", squeeze_52: "f32[256]", relu_15: "f32[8, 256, 32, 32]", convolution_18: "f32[8, 256, 16, 16]", squeeze_55: "f32[256]", relu_16: "f32[8, 256, 16, 16]", convolution_19: "f32[8, 1024, 16, 16]", squeeze_58: "f32[1024]", convolution_20: "f32[8, 1024, 16, 16]", squeeze_61: "f32[1024]", relu_17: "f32[8, 1024, 16, 16]", convolution_21: "f32[8, 256, 16, 16]", squeeze_64: "f32[256]", relu_18: "f32[8, 256, 16, 16]", view_7: "f32[8192, 64]", view_13: "f32[8192, 64]", bmm_1: "f32[32, 256, 64]", squeeze_67: "f32[256]", relu_19: "f32[8, 256, 16, 16]", convolution_23: "f32[8, 1024, 16, 16]", squeeze_70: "f32[1024]", relu_20: "f32[8, 1024, 16, 16]", convolution_24: "f32[8, 512, 16, 16]", squeeze_73: "f32[512]", relu_21: "f32[8, 512, 16, 16]", view_31: "f32[8192, 128]", view_37: "f32[8192, 128]", view_47: "f32[8, 512, 16, 16]", avg_pool2d: "f32[8, 512, 8, 8]", squeeze_76: "f32[512]", relu_22: "f32[8, 512, 8, 8]", convolution_26: "f32[8, 2048, 8, 8]", squeeze_79: "f32[2048]", convolution_27: "f32[8, 2048, 8, 8]", squeeze_82: "f32[2048]", relu_23: "f32[8, 2048, 8, 8]", convolution_28: "f32[8, 512, 8, 8]", squeeze_85: "f32[512]", relu_24: "f32[8, 512, 8, 8]", view_55: "f32[2048, 128]", view_61: "f32[2048, 128]", bmm_5: "f32[32, 64, 128]", squeeze_88: "f32[512]", relu_25: "f32[8, 512, 8, 8]", convolution_30: "f32[8, 2048, 8, 8]", squeeze_91: "f32[2048]", clone_21: "f32[8, 2048]", permute_25: "f32[1000, 2048]", le: "b8[8, 2048, 8, 8]", unsqueeze_126: "f32[1, 2048, 1, 1]", unsqueeze_138: "f32[1, 512, 1, 1]", permute_30: "f32[32, 64, 64]", permute_31: "f32[32, 128, 64]", alias_36: "f32[32, 64, 64]", permute_35: "f32[15, 128]", permute_41: "f32[15, 128]", permute_43: "f32[32, 128, 64]", permute_44: "f32[32, 64, 128]", unsqueeze_150: "f32[1, 512, 1, 1]", unsqueeze_162: "f32[1, 2048, 1, 1]", unsqueeze_174: "f32[1, 2048, 1, 1]", unsqueeze_186: "f32[1, 512, 1, 1]", permute_48: "f32[32, 256, 256]", permute_49: "f32[32, 128, 256]", alias_46: "f32[32, 256, 256]", permute_53: "f32[31, 128]", permute_59: "f32[31, 128]", permute_61: "f32[32, 128, 256]", permute_62: "f32[32, 256, 128]", unsqueeze_198: "f32[1, 512, 1, 1]", unsqueeze_210: "f32[1, 1024, 1, 1]", unsqueeze_222: "f32[1, 256, 1, 1]", permute_66: "f32[32, 256, 256]", permute_67: "f32[32, 64, 256]", alias_56: "f32[32, 256, 256]", permute_71: "f32[31, 64]", permute_77: "f32[31, 64]", permute_79: "f32[32, 64, 256]", permute_80: "f32[32, 256, 64]", unsqueeze_234: "f32[1, 256, 1, 1]", unsqueeze_246: "f32[1, 1024, 1, 1]", unsqueeze_258: "f32[1, 1024, 1, 1]", unsqueeze_270: "f32[1, 256, 1, 1]", unsqueeze_282: "f32[1, 256, 1, 1]", unsqueeze_294: "f32[1, 512, 1, 1]", unsqueeze_306: "f32[1, 128, 1, 1]", unsqueeze_318: "f32[1, 128, 1, 1]", unsqueeze_330: "f32[1, 512, 1, 1]", unsqueeze_342: "f32[1, 512, 1, 1]", unsqueeze_354: "f32[1, 128, 1, 1]", unsqueeze_366: "f32[1, 128, 1, 1]", unsqueeze_378: "f32[1, 256, 1, 1]", unsqueeze_390: "f32[1, 64, 1, 1]", unsqueeze_402: "f32[1, 64, 1, 1]", unsqueeze_414: "f32[1, 256, 1, 1]", unsqueeze_426: "f32[1, 256, 1, 1]", unsqueeze_438: "f32[1, 64, 1, 1]", unsqueeze_450: "f32[1, 64, 1, 1]", unsqueeze_462: "f32[1, 64, 1, 1]", unsqueeze_474: "f32[1, 32, 1, 1]", unsqueeze_486: "f32[1, 24, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    permute_7: "f32[32, 64, 256]" = torch.ops.aten.permute.default(bmm_1, [0, 2, 1]);  bmm_1 = None
    clone_6: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_23: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_6, [8, 256, 16, 16]);  clone_6 = None
    permute_23: "f32[32, 128, 64]" = torch.ops.aten.permute.default(bmm_5, [0, 2, 1]);  bmm_5 = None
    clone_20: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_23, memory_format = torch.contiguous_format);  permute_23 = None
    view_71: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_20, [8, 512, 8, 8]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm_6: "f32[8, 2048]" = torch.ops.aten.mm.default(tangents_1, permute_25);  permute_25 = None
    permute_26: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_7: "f32[1000, 2048]" = torch.ops.aten.mm.default(permute_26, clone_21);  permute_26 = clone_21 = None
    permute_27: "f32[2048, 1000]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_4: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_73: "f32[1000]" = torch.ops.aten.view.default(sum_4, [1000]);  sum_4 = None
    permute_28: "f32[1000, 2048]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_74: "f32[8, 2048, 1, 1]" = torch.ops.aten.view.default(mm_6, [8, 2048, 1, 1]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand_18: "f32[8, 2048, 8, 8]" = torch.ops.aten.expand.default(view_74, [8, 2048, 8, 8]);  view_74 = None
    div_3: "f32[8, 2048, 8, 8]" = torch.ops.aten.div.Scalar(expand_18, 64);  expand_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 2048, 8, 8]" = torch.ops.aten.where.self(le, full_default, div_3);  le = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_5: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_34: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_126);  convolution_30 = unsqueeze_126 = None
    mul_220: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(where, sub_34)
    sum_6: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 2, 3]);  mul_220 = None
    mul_221: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
    unsqueeze_127: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_221, 0);  mul_221 = None
    unsqueeze_128: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_127, 2);  unsqueeze_127 = None
    unsqueeze_129: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 3);  unsqueeze_128 = None
    mul_222: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
    mul_223: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_224: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_222, mul_223);  mul_222 = mul_223 = None
    unsqueeze_130: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_224, 0);  mul_224 = None
    unsqueeze_131: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 2);  unsqueeze_130 = None
    unsqueeze_132: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_131, 3);  unsqueeze_131 = None
    mul_225: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_67);  primals_67 = None
    unsqueeze_133: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_225, 0);  mul_225 = None
    unsqueeze_134: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_133, 2);  unsqueeze_133 = None
    unsqueeze_135: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 3);  unsqueeze_134 = None
    mul_226: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_132);  sub_34 = unsqueeze_132 = None
    sub_36: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(where, mul_226);  mul_226 = None
    sub_37: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_36, unsqueeze_129);  sub_36 = unsqueeze_129 = None
    mul_227: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_135);  sub_37 = unsqueeze_135 = None
    mul_228: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_6, squeeze_91);  sum_6 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_227, relu_25, primals_99, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_227 = primals_99 = None
    getitem_73: "f32[8, 512, 8, 8]" = convolution_backward[0]
    getitem_74: "f32[2048, 512, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_34: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_35: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    le_1: "b8[8, 512, 8, 8]" = torch.ops.aten.le.Scalar(alias_35, 0);  alias_35 = None
    where_1: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(le_1, full_default, getitem_73);  le_1 = getitem_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_7: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_38: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(view_71, unsqueeze_138);  view_71 = unsqueeze_138 = None
    mul_229: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_1, sub_38)
    sum_8: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 2, 3]);  mul_229 = None
    mul_230: "f32[512]" = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
    unsqueeze_139: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_230, 0);  mul_230 = None
    unsqueeze_140: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_139, 2);  unsqueeze_139 = None
    unsqueeze_141: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, 3);  unsqueeze_140 = None
    mul_231: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
    mul_232: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_233: "f32[512]" = torch.ops.aten.mul.Tensor(mul_231, mul_232);  mul_231 = mul_232 = None
    unsqueeze_142: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_233, 0);  mul_233 = None
    unsqueeze_143: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 2);  unsqueeze_142 = None
    unsqueeze_144: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_143, 3);  unsqueeze_143 = None
    mul_234: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_65);  primals_65 = None
    unsqueeze_145: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_234, 0);  mul_234 = None
    unsqueeze_146: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_145, 2);  unsqueeze_145 = None
    unsqueeze_147: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, 3);  unsqueeze_146 = None
    mul_235: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_144);  sub_38 = unsqueeze_144 = None
    sub_40: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_1, mul_235);  where_1 = mul_235 = None
    sub_41: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_40, unsqueeze_141);  sub_40 = unsqueeze_141 = None
    mul_236: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_147);  sub_41 = unsqueeze_147 = None
    mul_237: "f32[512]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_88);  sum_8 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_75: "f32[32, 128, 64]" = torch.ops.aten.view.default(mul_236, [32, 128, 64]);  mul_236 = None
    permute_29: "f32[32, 64, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
    bmm_6: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(permute_30, permute_29);  permute_30 = None
    bmm_7: "f32[32, 64, 64]" = torch.ops.aten.bmm.default(permute_29, permute_31);  permute_29 = permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    mul_238: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(bmm_7, alias_36);  bmm_7 = None
    sum_9: "f32[32, 64, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [-1], True)
    mul_239: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(alias_36, sum_9);  alias_36 = sum_9 = None
    sub_42: "f32[32, 64, 64]" = torch.ops.aten.sub.Tensor(mul_238, mul_239);  mul_238 = mul_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_79: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.view.default(sub_42, [32, 8, 8, 8, 8])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_32: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(view_79, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_10: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.sum.dim_IntList(permute_32, [2], True);  permute_32 = None
    view_80: "f32[256, 8, 8]" = torch.ops.aten.view.default(sum_10, [256, 8, 8]);  sum_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_default_2: "f32[256, 8, 15]" = torch.ops.aten.full.default([256, 8, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[256, 8, 15]" = torch.ops.aten.slice_scatter.default(full_default_2, view_80, 2, 7, 9223372036854775807);  view_80 = None
    full_default_3: "f32[256, 9, 15]" = torch.ops.aten.full.default([256, 9, 15], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter, 1, 0, 8);  slice_scatter = None
    slice_scatter_2: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_1, 0, 0, 9223372036854775807);  slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_81: "f32[256, 135]" = torch.ops.aten.view.default(slice_scatter_2, [256, 135]);  slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_12: "f32[256, 128]" = torch.ops.aten.constant_pad_nd.default(view_81, [0, -7]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_82: "f32[256, 8, 16]" = torch.ops.aten.view.default(constant_pad_nd_12, [256, 8, 16]);  constant_pad_nd_12 = None
    constant_pad_nd_13: "f32[256, 8, 15]" = torch.ops.aten.constant_pad_nd.default(view_82, [0, -1]);  view_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_83: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(constant_pad_nd_13, [32, 8, 8, 15]);  constant_pad_nd_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_84: "f32[2048, 15]" = torch.ops.aten.view.default(view_83, [2048, 15]);  view_83 = None
    permute_33: "f32[15, 2048]" = torch.ops.aten.permute.default(view_84, [1, 0])
    mm_8: "f32[15, 128]" = torch.ops.aten.mm.default(permute_33, view_61);  permute_33 = view_61 = None
    permute_34: "f32[128, 15]" = torch.ops.aten.permute.default(mm_8, [1, 0]);  mm_8 = None
    mm_9: "f32[2048, 128]" = torch.ops.aten.mm.default(view_84, permute_35);  view_84 = permute_35 = None
    view_85: "f32[32, 8, 8, 128]" = torch.ops.aten.view.default(mm_9, [32, 8, 8, 128]);  mm_9 = None
    permute_36: "f32[15, 128]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_37: "f32[32, 8, 8, 128]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_38: "f32[32, 8, 8, 8, 8]" = torch.ops.aten.permute.default(view_79, [0, 1, 3, 2, 4]);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_11: "f32[32, 8, 1, 8, 8]" = torch.ops.aten.sum.dim_IntList(permute_38, [2], True);  permute_38 = None
    view_86: "f32[256, 8, 8]" = torch.ops.aten.view.default(sum_11, [256, 8, 8]);  sum_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_scatter_3: "f32[256, 8, 15]" = torch.ops.aten.slice_scatter.default(full_default_2, view_86, 2, 7, 9223372036854775807);  full_default_2 = view_86 = None
    slice_scatter_4: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_3, 1, 0, 8);  slice_scatter_3 = None
    slice_scatter_5: "f32[256, 9, 15]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_4, 0, 0, 9223372036854775807);  full_default_3 = slice_scatter_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_87: "f32[256, 135]" = torch.ops.aten.view.default(slice_scatter_5, [256, 135]);  slice_scatter_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_14: "f32[256, 128]" = torch.ops.aten.constant_pad_nd.default(view_87, [0, -7]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_88: "f32[256, 8, 16]" = torch.ops.aten.view.default(constant_pad_nd_14, [256, 8, 16]);  constant_pad_nd_14 = None
    constant_pad_nd_15: "f32[256, 8, 15]" = torch.ops.aten.constant_pad_nd.default(view_88, [0, -1]);  view_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_89: "f32[32, 8, 8, 15]" = torch.ops.aten.view.default(constant_pad_nd_15, [32, 8, 8, 15]);  constant_pad_nd_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_90: "f32[2048, 15]" = torch.ops.aten.view.default(view_89, [2048, 15]);  view_89 = None
    permute_39: "f32[15, 2048]" = torch.ops.aten.permute.default(view_90, [1, 0])
    mm_10: "f32[15, 128]" = torch.ops.aten.mm.default(permute_39, view_55);  permute_39 = view_55 = None
    permute_40: "f32[128, 15]" = torch.ops.aten.permute.default(mm_10, [1, 0]);  mm_10 = None
    mm_11: "f32[2048, 128]" = torch.ops.aten.mm.default(view_90, permute_41);  view_90 = permute_41 = None
    view_91: "f32[32, 8, 8, 128]" = torch.ops.aten.view.default(mm_11, [32, 8, 8, 128]);  mm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_169: "f32[32, 8, 8, 128]" = torch.ops.aten.add.Tensor(permute_37, view_91);  permute_37 = view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_42: "f32[15, 128]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_22: "f32[32, 8, 8, 128]" = torch.ops.aten.clone.default(add_169, memory_format = torch.contiguous_format);  add_169 = None
    view_92: "f32[32, 64, 128]" = torch.ops.aten.view.default(clone_22, [32, 64, 128]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_240: "f32[32, 64, 64]" = torch.ops.aten.mul.Tensor(sub_42, 0.08838834764831845);  sub_42 = None
    bmm_8: "f32[32, 128, 64]" = torch.ops.aten.bmm.default(permute_43, mul_240);  permute_43 = None
    bmm_9: "f32[32, 64, 128]" = torch.ops.aten.bmm.default(mul_240, permute_44);  mul_240 = permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_170: "f32[32, 64, 128]" = torch.ops.aten.add.Tensor(view_92, bmm_9);  view_92 = bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_45: "f32[32, 128, 64]" = torch.ops.aten.permute.default(bmm_6, [0, 2, 1]);  bmm_6 = None
    clone_23: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_96: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_23, [8, 512, 8, 8]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_97: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(bmm_8, [8, 512, 8, 8]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_46: "f32[32, 128, 64]" = torch.ops.aten.permute.default(add_170, [0, 2, 1]);  add_170 = None
    clone_24: "f32[32, 128, 64]" = torch.ops.aten.clone.default(permute_46, memory_format = torch.contiguous_format);  permute_46 = None
    view_98: "f32[8, 512, 8, 8]" = torch.ops.aten.view.default(clone_24, [8, 512, 8, 8]);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat: "f32[8, 1536, 8, 8]" = torch.ops.aten.cat.default([view_98, view_97, view_96], 1);  view_98 = view_97 = view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(cat, relu_24, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat = primals_98 = None
    getitem_76: "f32[8, 512, 8, 8]" = convolution_backward_1[0]
    getitem_77: "f32[1536, 512, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_38: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_39: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    le_2: "b8[8, 512, 8, 8]" = torch.ops.aten.le.Scalar(alias_39, 0);  alias_39 = None
    where_2: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(le_2, full_default, getitem_76);  le_2 = getitem_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_43: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_150);  convolution_28 = unsqueeze_150 = None
    mul_241: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_2, sub_43)
    sum_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_241, [0, 2, 3]);  mul_241 = None
    mul_242: "f32[512]" = torch.ops.aten.mul.Tensor(sum_12, 0.001953125)
    unsqueeze_151: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_242, 0);  mul_242 = None
    unsqueeze_152: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 2);  unsqueeze_151 = None
    unsqueeze_153: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, 3);  unsqueeze_152 = None
    mul_243: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, 0.001953125)
    mul_244: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_245: "f32[512]" = torch.ops.aten.mul.Tensor(mul_243, mul_244);  mul_243 = mul_244 = None
    unsqueeze_154: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_245, 0);  mul_245 = None
    unsqueeze_155: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, 2);  unsqueeze_154 = None
    unsqueeze_156: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 3);  unsqueeze_155 = None
    mul_246: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_61);  primals_61 = None
    unsqueeze_157: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_246, 0);  mul_246 = None
    unsqueeze_158: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_157, 2);  unsqueeze_157 = None
    unsqueeze_159: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, 3);  unsqueeze_158 = None
    mul_247: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_156);  sub_43 = unsqueeze_156 = None
    sub_45: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_2, mul_247);  where_2 = mul_247 = None
    sub_46: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_45, unsqueeze_153);  sub_45 = unsqueeze_153 = None
    mul_248: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_159);  sub_46 = unsqueeze_159 = None
    mul_249: "f32[512]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_85);  sum_13 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_248, relu_23, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_248 = primals_97 = None
    getitem_79: "f32[8, 2048, 8, 8]" = convolution_backward_2[0]
    getitem_80: "f32[512, 2048, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_171: "f32[8, 2048, 8, 8]" = torch.ops.aten.add.Tensor(where, getitem_79);  where = getitem_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    alias_41: "f32[8, 2048, 8, 8]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_42: "f32[8, 2048, 8, 8]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    le_3: "b8[8, 2048, 8, 8]" = torch.ops.aten.le.Scalar(alias_42, 0);  alias_42 = None
    where_3: "f32[8, 2048, 8, 8]" = torch.ops.aten.where.self(le_3, full_default, add_171);  le_3 = add_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_14: "f32[2048]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_47: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_162);  convolution_27 = unsqueeze_162 = None
    mul_250: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(where_3, sub_47)
    sum_15: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_250, [0, 2, 3]);  mul_250 = None
    mul_251: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_14, 0.001953125)
    unsqueeze_163: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_251, 0);  mul_251 = None
    unsqueeze_164: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 2);  unsqueeze_163 = None
    unsqueeze_165: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, 3);  unsqueeze_164 = None
    mul_252: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
    mul_253: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_254: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_252, mul_253);  mul_252 = mul_253 = None
    unsqueeze_166: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_254, 0);  mul_254 = None
    unsqueeze_167: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, 2);  unsqueeze_166 = None
    unsqueeze_168: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 3);  unsqueeze_167 = None
    mul_255: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_59);  primals_59 = None
    unsqueeze_169: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_255, 0);  mul_255 = None
    unsqueeze_170: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_169, 2);  unsqueeze_169 = None
    unsqueeze_171: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 3);  unsqueeze_170 = None
    mul_256: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_168);  sub_47 = unsqueeze_168 = None
    sub_49: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(where_3, mul_256);  mul_256 = None
    sub_50: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_49, unsqueeze_165);  sub_49 = None
    mul_257: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_171);  sub_50 = unsqueeze_171 = None
    mul_258: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_82);  sum_15 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_257, relu_20, primals_96, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_257 = primals_96 = None
    getitem_82: "f32[8, 1024, 16, 16]" = convolution_backward_3[0]
    getitem_83: "f32[2048, 1024, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_51: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_174);  convolution_26 = unsqueeze_174 = None
    mul_259: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(where_3, sub_51)
    sum_17: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_259, [0, 2, 3]);  mul_259 = None
    mul_261: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
    mul_262: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_263: "f32[2048]" = torch.ops.aten.mul.Tensor(mul_261, mul_262);  mul_261 = mul_262 = None
    unsqueeze_178: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_263, 0);  mul_263 = None
    unsqueeze_179: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 2);  unsqueeze_178 = None
    unsqueeze_180: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 3);  unsqueeze_179 = None
    mul_264: "f32[2048]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_57);  primals_57 = None
    unsqueeze_181: "f32[1, 2048]" = torch.ops.aten.unsqueeze.default(mul_264, 0);  mul_264 = None
    unsqueeze_182: "f32[1, 2048, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_181, 2);  unsqueeze_181 = None
    unsqueeze_183: "f32[1, 2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 3);  unsqueeze_182 = None
    mul_265: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_180);  sub_51 = unsqueeze_180 = None
    sub_53: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(where_3, mul_265);  where_3 = mul_265 = None
    sub_54: "f32[8, 2048, 8, 8]" = torch.ops.aten.sub.Tensor(sub_53, unsqueeze_165);  sub_53 = unsqueeze_165 = None
    mul_266: "f32[8, 2048, 8, 8]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_183);  sub_54 = unsqueeze_183 = None
    mul_267: "f32[2048]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_79);  sum_17 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_266, relu_22, primals_95, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_266 = primals_95 = None
    getitem_85: "f32[8, 512, 8, 8]" = convolution_backward_4[0]
    getitem_86: "f32[2048, 512, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_44: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_45: "f32[8, 512, 8, 8]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    le_4: "b8[8, 512, 8, 8]" = torch.ops.aten.le.Scalar(alias_45, 0);  alias_45 = None
    where_4: "f32[8, 512, 8, 8]" = torch.ops.aten.where.self(le_4, full_default, getitem_85);  le_4 = getitem_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_55: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(avg_pool2d, unsqueeze_186);  avg_pool2d = unsqueeze_186 = None
    mul_268: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(where_4, sub_55)
    sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 2, 3]);  mul_268 = None
    mul_269: "f32[512]" = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
    unsqueeze_187: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_269, 0);  mul_269 = None
    unsqueeze_188: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_187, 2);  unsqueeze_187 = None
    unsqueeze_189: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 3);  unsqueeze_188 = None
    mul_270: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, 0.001953125)
    mul_271: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_272: "f32[512]" = torch.ops.aten.mul.Tensor(mul_270, mul_271);  mul_270 = mul_271 = None
    unsqueeze_190: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_272, 0);  mul_272 = None
    unsqueeze_191: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, 2);  unsqueeze_190 = None
    unsqueeze_192: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 3);  unsqueeze_191 = None
    mul_273: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_55);  primals_55 = None
    unsqueeze_193: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_273, 0);  mul_273 = None
    unsqueeze_194: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_193, 2);  unsqueeze_193 = None
    unsqueeze_195: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 3);  unsqueeze_194 = None
    mul_274: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_192);  sub_55 = unsqueeze_192 = None
    sub_57: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(where_4, mul_274);  where_4 = mul_274 = None
    sub_58: "f32[8, 512, 8, 8]" = torch.ops.aten.sub.Tensor(sub_57, unsqueeze_189);  sub_57 = unsqueeze_189 = None
    mul_275: "f32[8, 512, 8, 8]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_195);  sub_58 = unsqueeze_195 = None
    mul_276: "f32[512]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_76);  sum_19 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:156, code: out = self.pool(out)
    avg_pool2d_backward: "f32[8, 512, 16, 16]" = torch.ops.aten.avg_pool2d_backward.default(mul_275, view_47, [2, 2], [2, 2], [0, 0], False, True, None);  mul_275 = view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_99: "f32[32, 128, 256]" = torch.ops.aten.view.default(avg_pool2d_backward, [32, 128, 256]);  avg_pool2d_backward = None
    permute_47: "f32[32, 256, 128]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    bmm_10: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(permute_48, permute_47);  permute_48 = None
    bmm_11: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(permute_47, permute_49);  permute_47 = permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    mul_277: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(bmm_11, alias_46);  bmm_11 = None
    sum_20: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [-1], True)
    mul_278: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(alias_46, sum_20);  alias_46 = sum_20 = None
    sub_59: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(mul_277, mul_278);  mul_277 = mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_103: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.view.default(sub_59, [32, 16, 16, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_50: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_103, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_21: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_50, [2], True);  permute_50 = None
    view_104: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_21, [512, 16, 16]);  sum_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    full_default_11: "f32[512, 16, 31]" = torch.ops.aten.full.default([512, 16, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_6: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_default_11, view_104, 2, 15, 9223372036854775807);  view_104 = None
    full_default_12: "f32[512, 17, 31]" = torch.ops.aten.full.default([512, 17, 31], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_7: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_scatter_6, 1, 0, 16);  slice_scatter_6 = None
    slice_scatter_8: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_scatter_7, 0, 0, 9223372036854775807);  slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_105: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_8, [512, 527]);  slice_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_16: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_105, [0, -15]);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_106: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_16, [512, 16, 32]);  constant_pad_nd_16 = None
    constant_pad_nd_17: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_106, [0, -1]);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_107: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_17, [32, 16, 16, 31]);  constant_pad_nd_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_108: "f32[8192, 31]" = torch.ops.aten.view.default(view_107, [8192, 31]);  view_107 = None
    permute_51: "f32[31, 8192]" = torch.ops.aten.permute.default(view_108, [1, 0])
    mm_12: "f32[31, 128]" = torch.ops.aten.mm.default(permute_51, view_37);  permute_51 = view_37 = None
    permute_52: "f32[128, 31]" = torch.ops.aten.permute.default(mm_12, [1, 0]);  mm_12 = None
    mm_13: "f32[8192, 128]" = torch.ops.aten.mm.default(view_108, permute_53);  view_108 = permute_53 = None
    view_109: "f32[32, 16, 16, 128]" = torch.ops.aten.view.default(mm_13, [32, 16, 16, 128]);  mm_13 = None
    permute_54: "f32[31, 128]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_55: "f32[32, 16, 16, 128]" = torch.ops.aten.permute.default(view_109, [0, 2, 1, 3]);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_56: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_103, [0, 1, 3, 2, 4]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_22: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_56, [2], True);  permute_56 = None
    view_110: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_22, [512, 16, 16]);  sum_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_scatter_9: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_default_11, view_110, 2, 15, 9223372036854775807);  view_110 = None
    slice_scatter_10: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_scatter_9, 1, 0, 16);  slice_scatter_9 = None
    slice_scatter_11: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_scatter_10, 0, 0, 9223372036854775807);  slice_scatter_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_111: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_11, [512, 527]);  slice_scatter_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_18: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_111, [0, -15]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_112: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_18, [512, 16, 32]);  constant_pad_nd_18 = None
    constant_pad_nd_19: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_112, [0, -1]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_113: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_19, [32, 16, 16, 31]);  constant_pad_nd_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_114: "f32[8192, 31]" = torch.ops.aten.view.default(view_113, [8192, 31]);  view_113 = None
    permute_57: "f32[31, 8192]" = torch.ops.aten.permute.default(view_114, [1, 0])
    mm_14: "f32[31, 128]" = torch.ops.aten.mm.default(permute_57, view_31);  permute_57 = view_31 = None
    permute_58: "f32[128, 31]" = torch.ops.aten.permute.default(mm_14, [1, 0]);  mm_14 = None
    mm_15: "f32[8192, 128]" = torch.ops.aten.mm.default(view_114, permute_59);  view_114 = permute_59 = None
    view_115: "f32[32, 16, 16, 128]" = torch.ops.aten.view.default(mm_15, [32, 16, 16, 128]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_172: "f32[32, 16, 16, 128]" = torch.ops.aten.add.Tensor(permute_55, view_115);  permute_55 = view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_60: "f32[31, 128]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_25: "f32[32, 16, 16, 128]" = torch.ops.aten.clone.default(add_172, memory_format = torch.contiguous_format);  add_172 = None
    view_116: "f32[32, 256, 128]" = torch.ops.aten.view.default(clone_25, [32, 256, 128]);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_279: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_59, 0.08838834764831845);  sub_59 = None
    bmm_12: "f32[32, 128, 256]" = torch.ops.aten.bmm.default(permute_61, mul_279);  permute_61 = None
    bmm_13: "f32[32, 256, 128]" = torch.ops.aten.bmm.default(mul_279, permute_62);  mul_279 = permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_173: "f32[32, 256, 128]" = torch.ops.aten.add.Tensor(view_116, bmm_13);  view_116 = bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_63: "f32[32, 128, 256]" = torch.ops.aten.permute.default(bmm_10, [0, 2, 1]);  bmm_10 = None
    clone_26: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_63, memory_format = torch.contiguous_format);  permute_63 = None
    view_120: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(clone_26, [8, 512, 16, 16]);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_121: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(bmm_12, [8, 512, 16, 16]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_64: "f32[32, 128, 256]" = torch.ops.aten.permute.default(add_173, [0, 2, 1]);  add_173 = None
    clone_27: "f32[32, 128, 256]" = torch.ops.aten.clone.default(permute_64, memory_format = torch.contiguous_format);  permute_64 = None
    view_122: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(clone_27, [8, 512, 16, 16]);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat_1: "f32[8, 1536, 16, 16]" = torch.ops.aten.cat.default([view_122, view_121, view_120], 1);  view_122 = view_121 = view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(cat_1, relu_21, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_1 = primals_94 = None
    getitem_88: "f32[8, 512, 16, 16]" = convolution_backward_5[0]
    getitem_89: "f32[1536, 512, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_48: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_49: "f32[8, 512, 16, 16]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_5: "b8[8, 512, 16, 16]" = torch.ops.aten.le.Scalar(alias_49, 0);  alias_49 = None
    where_5: "f32[8, 512, 16, 16]" = torch.ops.aten.where.self(le_5, full_default, getitem_88);  le_5 = getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_23: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_60: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_198);  convolution_24 = unsqueeze_198 = None
    mul_280: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(where_5, sub_60)
    sum_24: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_280, [0, 2, 3]);  mul_280 = None
    mul_281: "f32[512]" = torch.ops.aten.mul.Tensor(sum_23, 0.00048828125)
    unsqueeze_199: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_281, 0);  mul_281 = None
    unsqueeze_200: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_199, 2);  unsqueeze_199 = None
    unsqueeze_201: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, 3);  unsqueeze_200 = None
    mul_282: "f32[512]" = torch.ops.aten.mul.Tensor(sum_24, 0.00048828125)
    mul_283: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_284: "f32[512]" = torch.ops.aten.mul.Tensor(mul_282, mul_283);  mul_282 = mul_283 = None
    unsqueeze_202: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_284, 0);  mul_284 = None
    unsqueeze_203: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, 2);  unsqueeze_202 = None
    unsqueeze_204: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 3);  unsqueeze_203 = None
    mul_285: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_51);  primals_51 = None
    unsqueeze_205: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_285, 0);  mul_285 = None
    unsqueeze_206: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 2);  unsqueeze_205 = None
    unsqueeze_207: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 3);  unsqueeze_206 = None
    mul_286: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_204);  sub_60 = unsqueeze_204 = None
    sub_62: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(where_5, mul_286);  where_5 = mul_286 = None
    sub_63: "f32[8, 512, 16, 16]" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_201);  sub_62 = unsqueeze_201 = None
    mul_287: "f32[8, 512, 16, 16]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_207);  sub_63 = unsqueeze_207 = None
    mul_288: "f32[512]" = torch.ops.aten.mul.Tensor(sum_24, squeeze_73);  sum_24 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_287, relu_20, primals_93, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_287 = primals_93 = None
    getitem_91: "f32[8, 1024, 16, 16]" = convolution_backward_6[0]
    getitem_92: "f32[512, 1024, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_174: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(getitem_82, getitem_91);  getitem_82 = getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:889, code: return self.act(x)
    alias_51: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_52: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    le_6: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_52, 0);  alias_52 = None
    where_6: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_6, full_default, add_174);  le_6 = add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_25: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_64: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_210);  convolution_23 = unsqueeze_210 = None
    mul_289: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_6, sub_64)
    sum_26: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_289, [0, 2, 3]);  mul_289 = None
    mul_290: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_25, 0.00048828125)
    unsqueeze_211: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_290, 0);  mul_290 = None
    unsqueeze_212: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    unsqueeze_213: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
    mul_291: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_26, 0.00048828125)
    mul_292: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_293: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_291, mul_292);  mul_291 = mul_292 = None
    unsqueeze_214: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_293, 0);  mul_293 = None
    unsqueeze_215: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
    unsqueeze_216: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
    mul_294: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_49);  primals_49 = None
    unsqueeze_217: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_294, 0);  mul_294 = None
    unsqueeze_218: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
    mul_295: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_216);  sub_64 = unsqueeze_216 = None
    sub_66: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_6, mul_295);  mul_295 = None
    sub_67: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_213);  sub_66 = unsqueeze_213 = None
    mul_296: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_219);  sub_67 = unsqueeze_219 = None
    mul_297: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_70);  sum_26 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_296, relu_19, primals_92, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_296 = primals_92 = None
    getitem_94: "f32[8, 256, 16, 16]" = convolution_backward_7[0]
    getitem_95: "f32[1024, 256, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_54: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_55: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_7: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    where_7: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_7, full_default, getitem_94);  le_7 = getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_27: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_68: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(view_23, unsqueeze_222);  view_23 = unsqueeze_222 = None
    mul_298: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_7, sub_68)
    sum_28: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_298, [0, 2, 3]);  mul_298 = None
    mul_299: "f32[256]" = torch.ops.aten.mul.Tensor(sum_27, 0.00048828125)
    unsqueeze_223: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_299, 0);  mul_299 = None
    unsqueeze_224: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    unsqueeze_225: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
    mul_300: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, 0.00048828125)
    mul_301: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_302: "f32[256]" = torch.ops.aten.mul.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    unsqueeze_226: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_302, 0);  mul_302 = None
    unsqueeze_227: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
    unsqueeze_228: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
    mul_303: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_47);  primals_47 = None
    unsqueeze_229: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_303, 0);  mul_303 = None
    unsqueeze_230: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
    mul_304: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_228);  sub_68 = unsqueeze_228 = None
    sub_70: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_7, mul_304);  where_7 = mul_304 = None
    sub_71: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_225);  sub_70 = unsqueeze_225 = None
    mul_305: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_231);  sub_71 = unsqueeze_231 = None
    mul_306: "f32[256]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_67);  sum_28 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:155, code: out = (attn @ v).transpose(-1, -2).reshape(B, self.dim_out_v, H, W)  # B, dim_out, H, W
    view_123: "f32[32, 64, 256]" = torch.ops.aten.view.default(mul_305, [32, 64, 256]);  mul_305 = None
    permute_65: "f32[32, 256, 64]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    bmm_14: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(permute_66, permute_65);  permute_66 = None
    bmm_15: "f32[32, 256, 256]" = torch.ops.aten.bmm.default(permute_65, permute_67);  permute_65 = permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:153, code: attn = attn.softmax(dim=-1)
    mul_307: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(bmm_15, alias_56);  bmm_15 = None
    sum_29: "f32[32, 256, 1]" = torch.ops.aten.sum.dim_IntList(mul_307, [-1], True)
    mul_308: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(alias_56, sum_29);  alias_56 = sum_29 = None
    sub_72: "f32[32, 256, 256]" = torch.ops.aten.sub.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:80, code: rel_logits = rel_logits.reshape(B, HW, HW)
    view_127: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.view.default(sub_72, [32, 16, 16, 16, 16])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_68: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_127, [0, 2, 4, 1, 3])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_30: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_68, [2], True);  permute_68 = None
    view_128: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_30, [512, 16, 16]);  sum_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_scatter_12: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_default_11, view_128, 2, 15, 9223372036854775807);  view_128 = None
    slice_scatter_13: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_scatter_12, 1, 0, 16);  slice_scatter_12 = None
    slice_scatter_14: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_scatter_13, 0, 0, 9223372036854775807);  slice_scatter_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_129: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_14, [512, 527]);  slice_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_20: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_129, [0, -15]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_130: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_20, [512, 16, 32]);  constant_pad_nd_20 = None
    constant_pad_nd_21: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_130, [0, -1]);  view_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_131: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_21, [32, 16, 16, 31]);  constant_pad_nd_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_132: "f32[8192, 31]" = torch.ops.aten.view.default(view_131, [8192, 31]);  view_131 = None
    permute_69: "f32[31, 8192]" = torch.ops.aten.permute.default(view_132, [1, 0])
    mm_16: "f32[31, 64]" = torch.ops.aten.mm.default(permute_69, view_13);  permute_69 = view_13 = None
    permute_70: "f32[64, 31]" = torch.ops.aten.permute.default(mm_16, [1, 0]);  mm_16 = None
    mm_17: "f32[8192, 64]" = torch.ops.aten.mm.default(view_132, permute_71);  view_132 = permute_71 = None
    view_133: "f32[32, 16, 16, 64]" = torch.ops.aten.view.default(mm_17, [32, 16, 16, 64]);  mm_17 = None
    permute_72: "f32[31, 64]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:76, code: q = q.transpose(1, 2)
    permute_73: "f32[32, 16, 16, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:53, code: return x.permute(permute_mask)
    permute_74: "f32[32, 16, 16, 16, 16]" = torch.ops.aten.permute.default(view_127, [0, 1, 3, 2, 4]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:52, code: x = x.reshape(B, H, 1, W, W).expand(-1, -1, H, -1, -1)
    sum_31: "f32[32, 16, 1, 16, 16]" = torch.ops.aten.sum.dim_IntList(permute_74, [2], True);  permute_74 = None
    view_134: "f32[512, 16, 16]" = torch.ops.aten.view.default(sum_31, [512, 16, 16]);  sum_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:49, code: x = x_pad[:, :W, W - 1:]
    slice_scatter_15: "f32[512, 16, 31]" = torch.ops.aten.slice_scatter.default(full_default_11, view_134, 2, 15, 9223372036854775807);  full_default_11 = view_134 = None
    slice_scatter_16: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_scatter_15, 1, 0, 16);  slice_scatter_15 = None
    slice_scatter_17: "f32[512, 17, 31]" = torch.ops.aten.slice_scatter.default(full_default_12, slice_scatter_16, 0, 0, 9223372036854775807);  full_default_12 = slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:48, code: x_pad = x_pad.reshape(-1, W + 1, 2 * W - 1)
    view_135: "f32[512, 527]" = torch.ops.aten.view.default(slice_scatter_17, [512, 527]);  slice_scatter_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:45, code: x_pad = F.pad(x_pad, [0, W - 1])
    constant_pad_nd_22: "f32[512, 512]" = torch.ops.aten.constant_pad_nd.default(view_135, [0, -15]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:44, code: x_pad = F.pad(x, [0, 1]).flatten(1)
    view_136: "f32[512, 16, 32]" = torch.ops.aten.view.default(constant_pad_nd_22, [512, 16, 32]);  constant_pad_nd_22 = None
    constant_pad_nd_23: "f32[512, 16, 31]" = torch.ops.aten.constant_pad_nd.default(view_136, [0, -1]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:41, code: x = x.reshape(-1, W, 2 * W -1)
    view_137: "f32[32, 16, 16, 31]" = torch.ops.aten.view.default(constant_pad_nd_23, [32, 16, 16, 31]);  constant_pad_nd_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    view_138: "f32[8192, 31]" = torch.ops.aten.view.default(view_137, [8192, 31]);  view_137 = None
    permute_75: "f32[31, 8192]" = torch.ops.aten.permute.default(view_138, [1, 0])
    mm_18: "f32[31, 64]" = torch.ops.aten.mm.default(permute_75, view_7);  permute_75 = view_7 = None
    permute_76: "f32[64, 31]" = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
    mm_19: "f32[8192, 64]" = torch.ops.aten.mm.default(view_138, permute_77);  view_138 = permute_77 = None
    view_139: "f32[32, 16, 16, 64]" = torch.ops.aten.view.default(mm_19, [32, 16, 16, 64]);  mm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    add_175: "f32[32, 16, 16, 64]" = torch.ops.aten.add.Tensor(permute_73, view_139);  permute_73 = view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:40, code: x = (q @ rel_k.transpose(-1, -2))
    permute_78: "f32[31, 64]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:72, code: q = q.reshape(B, self.height, self.width, -1)
    clone_28: "f32[32, 16, 16, 64]" = torch.ops.aten.clone.default(add_175, memory_format = torch.contiguous_format);  add_175 = None
    view_140: "f32[32, 256, 64]" = torch.ops.aten.view.default(clone_28, [32, 256, 64]);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    mul_309: "f32[32, 256, 256]" = torch.ops.aten.mul.Tensor(sub_72, 0.125);  sub_72 = None
    bmm_16: "f32[32, 64, 256]" = torch.ops.aten.bmm.default(permute_79, mul_309);  permute_79 = None
    bmm_17: "f32[32, 256, 64]" = torch.ops.aten.bmm.default(mul_309, permute_80);  mul_309 = permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:152, code: attn = (q @ k) * self.scale + self.pos_embed(q)
    add_176: "f32[32, 256, 64]" = torch.ops.aten.add.Tensor(view_140, bmm_17);  view_140 = bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:147, code: v = v.reshape(B * self.num_heads, self.dim_head_v, -1).transpose(-1, -2)
    permute_81: "f32[32, 64, 256]" = torch.ops.aten.permute.default(bmm_14, [0, 2, 1]);  bmm_14 = None
    clone_29: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_81, memory_format = torch.contiguous_format);  permute_81 = None
    view_144: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_29, [8, 256, 16, 16]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:146, code: k = k.reshape(B * self.num_heads, self.dim_head_qk, -1)  # no transpose, for q @ k
    view_145: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(bmm_16, [8, 256, 16, 16]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:145, code: q = q.reshape(B * self.num_heads, self.dim_head_qk, -1).transpose(-1, -2)
    permute_82: "f32[32, 64, 256]" = torch.ops.aten.permute.default(add_176, [0, 2, 1]);  add_176 = None
    clone_30: "f32[32, 64, 256]" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
    view_146: "f32[8, 256, 16, 16]" = torch.ops.aten.view.default(clone_30, [8, 256, 16, 16]);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:144, code: q, k, v = torch.split(x, [self.dim_out_qk, self.dim_out_qk, self.dim_out_v], dim=1)
    cat_2: "f32[8, 768, 16, 16]" = torch.ops.aten.cat.default([view_146, view_145, view_144], 1);  view_146 = view_145 = view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/bottleneck_attn.py:140, code: x = self.qkv(x)  # B, (2 * dim_head_qk + dim_head_v) * num_heads, H, W
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(cat_2, relu_18, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  cat_2 = primals_91 = None
    getitem_97: "f32[8, 256, 16, 16]" = convolution_backward_8[0]
    getitem_98: "f32[768, 256, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_58: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_59: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    le_8: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_59, 0);  alias_59 = None
    where_8: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_8, full_default, getitem_97);  le_8 = getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_32: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_73: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_234);  convolution_21 = unsqueeze_234 = None
    mul_310: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_8, sub_73)
    sum_33: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_310, [0, 2, 3]);  mul_310 = None
    mul_311: "f32[256]" = torch.ops.aten.mul.Tensor(sum_32, 0.00048828125)
    unsqueeze_235: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_311, 0);  mul_311 = None
    unsqueeze_236: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
    mul_312: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, 0.00048828125)
    mul_313: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_314: "f32[256]" = torch.ops.aten.mul.Tensor(mul_312, mul_313);  mul_312 = mul_313 = None
    unsqueeze_238: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_314, 0);  mul_314 = None
    unsqueeze_239: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
    unsqueeze_240: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
    mul_315: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_241: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_315, 0);  mul_315 = None
    unsqueeze_242: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    mul_316: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_240);  sub_73 = unsqueeze_240 = None
    sub_75: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_8, mul_316);  where_8 = mul_316 = None
    sub_76: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_237);  sub_75 = unsqueeze_237 = None
    mul_317: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_243);  sub_76 = unsqueeze_243 = None
    mul_318: "f32[256]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_64);  sum_33 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_317, relu_17, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_317 = primals_90 = None
    getitem_100: "f32[8, 1024, 16, 16]" = convolution_backward_9[0]
    getitem_101: "f32[256, 1024, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_177: "f32[8, 1024, 16, 16]" = torch.ops.aten.add.Tensor(where_6, getitem_100);  where_6 = getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_61: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_62: "f32[8, 1024, 16, 16]" = torch.ops.aten.alias.default(alias_61);  alias_61 = None
    le_9: "b8[8, 1024, 16, 16]" = torch.ops.aten.le.Scalar(alias_62, 0);  alias_62 = None
    where_9: "f32[8, 1024, 16, 16]" = torch.ops.aten.where.self(le_9, full_default, add_177);  le_9 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_34: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_77: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_246);  convolution_20 = unsqueeze_246 = None
    mul_319: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_9, sub_77)
    sum_35: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_319, [0, 2, 3]);  mul_319 = None
    mul_320: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_34, 0.00048828125)
    unsqueeze_247: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_320, 0);  mul_320 = None
    unsqueeze_248: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_321: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, 0.00048828125)
    mul_322: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_323: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_321, mul_322);  mul_321 = mul_322 = None
    unsqueeze_250: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_323, 0);  mul_323 = None
    unsqueeze_251: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_324: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_253: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_324, 0);  mul_324 = None
    unsqueeze_254: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    mul_325: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_252);  sub_77 = unsqueeze_252 = None
    sub_79: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_9, mul_325);  mul_325 = None
    sub_80: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_249);  sub_79 = None
    mul_326: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_255);  sub_80 = unsqueeze_255 = None
    mul_327: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_61);  sum_35 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_326, relu_14, primals_89, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_326 = primals_89 = None
    getitem_103: "f32[8, 512, 32, 32]" = convolution_backward_10[0]
    getitem_104: "f32[1024, 512, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_81: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_258);  convolution_19 = unsqueeze_258 = None
    mul_328: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(where_9, sub_81)
    sum_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_328, [0, 2, 3]);  mul_328 = None
    mul_330: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_37, 0.00048828125)
    mul_331: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_332: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_330, mul_331);  mul_330 = mul_331 = None
    unsqueeze_262: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_332, 0);  mul_332 = None
    unsqueeze_263: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_333: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_265: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_333, 0);  mul_333 = None
    unsqueeze_266: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    mul_334: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_264);  sub_81 = unsqueeze_264 = None
    sub_83: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(where_9, mul_334);  where_9 = mul_334 = None
    sub_84: "f32[8, 1024, 16, 16]" = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_249);  sub_83 = unsqueeze_249 = None
    mul_335: "f32[8, 1024, 16, 16]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_267);  sub_84 = unsqueeze_267 = None
    mul_336: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_58);  sum_37 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_335, relu_16, primals_88, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_335 = primals_88 = None
    getitem_106: "f32[8, 256, 16, 16]" = convolution_backward_11[0]
    getitem_107: "f32[1024, 256, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_64: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_65: "f32[8, 256, 16, 16]" = torch.ops.aten.alias.default(alias_64);  alias_64 = None
    le_10: "b8[8, 256, 16, 16]" = torch.ops.aten.le.Scalar(alias_65, 0);  alias_65 = None
    where_10: "f32[8, 256, 16, 16]" = torch.ops.aten.where.self(le_10, full_default, getitem_106);  le_10 = getitem_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_38: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_85: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_270);  convolution_18 = unsqueeze_270 = None
    mul_337: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(where_10, sub_85)
    sum_39: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 2, 3]);  mul_337 = None
    mul_338: "f32[256]" = torch.ops.aten.mul.Tensor(sum_38, 0.00048828125)
    unsqueeze_271: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_338, 0);  mul_338 = None
    unsqueeze_272: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_339: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, 0.00048828125)
    mul_340: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_341: "f32[256]" = torch.ops.aten.mul.Tensor(mul_339, mul_340);  mul_339 = mul_340 = None
    unsqueeze_274: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_341, 0);  mul_341 = None
    unsqueeze_275: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_342: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_277: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_342, 0);  mul_342 = None
    unsqueeze_278: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    mul_343: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_276);  sub_85 = unsqueeze_276 = None
    sub_87: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(where_10, mul_343);  where_10 = mul_343 = None
    sub_88: "f32[8, 256, 16, 16]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_273);  sub_87 = unsqueeze_273 = None
    mul_344: "f32[8, 256, 16, 16]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_279);  sub_88 = unsqueeze_279 = None
    mul_345: "f32[256]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_55);  sum_39 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_344, relu_15, primals_87, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_344 = primals_87 = None
    getitem_109: "f32[8, 256, 32, 32]" = convolution_backward_12[0]
    getitem_110: "f32[256, 256, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_67: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_68: "f32[8, 256, 32, 32]" = torch.ops.aten.alias.default(alias_67);  alias_67 = None
    le_11: "b8[8, 256, 32, 32]" = torch.ops.aten.le.Scalar(alias_68, 0);  alias_68 = None
    where_11: "f32[8, 256, 32, 32]" = torch.ops.aten.where.self(le_11, full_default, getitem_109);  le_11 = getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_40: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_89: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_282);  convolution_17 = unsqueeze_282 = None
    mul_346: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(where_11, sub_89)
    sum_41: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_346, [0, 2, 3]);  mul_346 = None
    mul_347: "f32[256]" = torch.ops.aten.mul.Tensor(sum_40, 0.0001220703125)
    unsqueeze_283: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_347, 0);  mul_347 = None
    unsqueeze_284: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_348: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, 0.0001220703125)
    mul_349: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_350: "f32[256]" = torch.ops.aten.mul.Tensor(mul_348, mul_349);  mul_348 = mul_349 = None
    unsqueeze_286: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_350, 0);  mul_350 = None
    unsqueeze_287: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_351: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_289: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_351, 0);  mul_351 = None
    unsqueeze_290: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    mul_352: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_288);  sub_89 = unsqueeze_288 = None
    sub_91: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(where_11, mul_352);  where_11 = mul_352 = None
    sub_92: "f32[8, 256, 32, 32]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_285);  sub_91 = unsqueeze_285 = None
    mul_353: "f32[8, 256, 32, 32]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_291);  sub_92 = unsqueeze_291 = None
    mul_354: "f32[256]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_52);  sum_41 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_353, relu_14, primals_86, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_353 = primals_86 = None
    getitem_112: "f32[8, 512, 32, 32]" = convolution_backward_13[0]
    getitem_113: "f32[256, 512, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_178: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(getitem_103, getitem_112);  getitem_103 = getitem_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_70: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_71: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(alias_70);  alias_70 = None
    le_12: "b8[8, 512, 32, 32]" = torch.ops.aten.le.Scalar(alias_71, 0);  alias_71 = None
    where_12: "f32[8, 512, 32, 32]" = torch.ops.aten.where.self(le_12, full_default, add_178);  le_12 = add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_93: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_294);  convolution_16 = unsqueeze_294 = None
    mul_355: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(where_12, sub_93)
    sum_43: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_355, [0, 2, 3]);  mul_355 = None
    mul_356: "f32[512]" = torch.ops.aten.mul.Tensor(sum_42, 0.0001220703125)
    unsqueeze_295: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_356, 0);  mul_356 = None
    unsqueeze_296: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_357: "f32[512]" = torch.ops.aten.mul.Tensor(sum_43, 0.0001220703125)
    mul_358: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_359: "f32[512]" = torch.ops.aten.mul.Tensor(mul_357, mul_358);  mul_357 = mul_358 = None
    unsqueeze_298: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_359, 0);  mul_359 = None
    unsqueeze_299: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_360: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_301: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_360, 0);  mul_360 = None
    unsqueeze_302: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    mul_361: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_300);  sub_93 = unsqueeze_300 = None
    sub_95: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(where_12, mul_361);  mul_361 = None
    sub_96: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_297);  sub_95 = unsqueeze_297 = None
    mul_362: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_303);  sub_96 = unsqueeze_303 = None
    mul_363: "f32[512]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_49);  sum_43 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_362, relu_13, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_362 = primals_85 = None
    getitem_115: "f32[8, 128, 32, 32]" = convolution_backward_14[0]
    getitem_116: "f32[512, 128, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_73: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_74: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_73);  alias_73 = None
    le_13: "b8[8, 128, 32, 32]" = torch.ops.aten.le.Scalar(alias_74, 0);  alias_74 = None
    where_13: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(le_13, full_default, getitem_115);  le_13 = getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_44: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_97: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_306);  convolution_15 = unsqueeze_306 = None
    mul_364: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_13, sub_97)
    sum_45: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3]);  mul_364 = None
    mul_365: "f32[128]" = torch.ops.aten.mul.Tensor(sum_44, 0.0001220703125)
    unsqueeze_307: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_365, 0);  mul_365 = None
    unsqueeze_308: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_366: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, 0.0001220703125)
    mul_367: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_368: "f32[128]" = torch.ops.aten.mul.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
    unsqueeze_310: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
    unsqueeze_311: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_369: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_313: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
    unsqueeze_314: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    mul_370: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_312);  sub_97 = unsqueeze_312 = None
    sub_99: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_13, mul_370);  where_13 = mul_370 = None
    sub_100: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_309);  sub_99 = unsqueeze_309 = None
    mul_371: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_315);  sub_100 = unsqueeze_315 = None
    mul_372: "f32[128]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_46);  sum_45 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_371, relu_12, primals_84, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_371 = primals_84 = None
    getitem_118: "f32[8, 128, 32, 32]" = convolution_backward_15[0]
    getitem_119: "f32[128, 128, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_76: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_77: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_76);  alias_76 = None
    le_14: "b8[8, 128, 32, 32]" = torch.ops.aten.le.Scalar(alias_77, 0);  alias_77 = None
    where_14: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(le_14, full_default, getitem_118);  le_14 = getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_46: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_101: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_318);  convolution_14 = unsqueeze_318 = None
    mul_373: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_14, sub_101)
    sum_47: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 2, 3]);  mul_373 = None
    mul_374: "f32[128]" = torch.ops.aten.mul.Tensor(sum_46, 0.0001220703125)
    unsqueeze_319: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
    unsqueeze_320: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_375: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, 0.0001220703125)
    mul_376: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_377: "f32[128]" = torch.ops.aten.mul.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_322: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_377, 0);  mul_377 = None
    unsqueeze_323: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_378: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_325: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_326: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    mul_379: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_324);  sub_101 = unsqueeze_324 = None
    sub_103: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_14, mul_379);  where_14 = mul_379 = None
    sub_104: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_321);  sub_103 = unsqueeze_321 = None
    mul_380: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_327);  sub_104 = unsqueeze_327 = None
    mul_381: "f32[128]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_43);  sum_47 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_380, relu_11, primals_83, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_380 = primals_83 = None
    getitem_121: "f32[8, 512, 32, 32]" = convolution_backward_16[0]
    getitem_122: "f32[128, 512, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_179: "f32[8, 512, 32, 32]" = torch.ops.aten.add.Tensor(where_12, getitem_121);  where_12 = getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_79: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_80: "f32[8, 512, 32, 32]" = torch.ops.aten.alias.default(alias_79);  alias_79 = None
    le_15: "b8[8, 512, 32, 32]" = torch.ops.aten.le.Scalar(alias_80, 0);  alias_80 = None
    where_15: "f32[8, 512, 32, 32]" = torch.ops.aten.where.self(le_15, full_default, add_179);  le_15 = add_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_48: "f32[512]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_105: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_330);  convolution_13 = unsqueeze_330 = None
    mul_382: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(where_15, sub_105)
    sum_49: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 2, 3]);  mul_382 = None
    mul_383: "f32[512]" = torch.ops.aten.mul.Tensor(sum_48, 0.0001220703125)
    unsqueeze_331: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_383, 0);  mul_383 = None
    unsqueeze_332: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_384: "f32[512]" = torch.ops.aten.mul.Tensor(sum_49, 0.0001220703125)
    mul_385: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_386: "f32[512]" = torch.ops.aten.mul.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
    unsqueeze_334: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_335: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_387: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_337: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
    unsqueeze_338: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    mul_388: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_336);  sub_105 = unsqueeze_336 = None
    sub_107: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(where_15, mul_388);  mul_388 = None
    sub_108: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_333);  sub_107 = None
    mul_389: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_339);  sub_108 = unsqueeze_339 = None
    mul_390: "f32[512]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_40);  sum_49 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_389, relu_8, primals_82, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_389 = primals_82 = None
    getitem_124: "f32[8, 256, 64, 64]" = convolution_backward_17[0]
    getitem_125: "f32[512, 256, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_109: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_342);  convolution_12 = unsqueeze_342 = None
    mul_391: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(where_15, sub_109)
    sum_51: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 2, 3]);  mul_391 = None
    mul_393: "f32[512]" = torch.ops.aten.mul.Tensor(sum_51, 0.0001220703125)
    mul_394: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_395: "f32[512]" = torch.ops.aten.mul.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    unsqueeze_346: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
    unsqueeze_347: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_396: "f32[512]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_349: "f32[1, 512]" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
    unsqueeze_350: "f32[1, 512, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    mul_397: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_348);  sub_109 = unsqueeze_348 = None
    sub_111: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(where_15, mul_397);  where_15 = mul_397 = None
    sub_112: "f32[8, 512, 32, 32]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_333);  sub_111 = unsqueeze_333 = None
    mul_398: "f32[8, 512, 32, 32]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_351);  sub_112 = unsqueeze_351 = None
    mul_399: "f32[512]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_37);  sum_51 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_398, relu_10, primals_81, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_398 = primals_81 = None
    getitem_127: "f32[8, 128, 32, 32]" = convolution_backward_18[0]
    getitem_128: "f32[512, 128, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_82: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_83: "f32[8, 128, 32, 32]" = torch.ops.aten.alias.default(alias_82);  alias_82 = None
    le_16: "b8[8, 128, 32, 32]" = torch.ops.aten.le.Scalar(alias_83, 0);  alias_83 = None
    where_16: "f32[8, 128, 32, 32]" = torch.ops.aten.where.self(le_16, full_default, getitem_127);  le_16 = getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_52: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_113: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_354);  convolution_11 = unsqueeze_354 = None
    mul_400: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(where_16, sub_113)
    sum_53: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
    mul_401: "f32[128]" = torch.ops.aten.mul.Tensor(sum_52, 0.0001220703125)
    unsqueeze_355: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_356: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_402: "f32[128]" = torch.ops.aten.mul.Tensor(sum_53, 0.0001220703125)
    mul_403: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_404: "f32[128]" = torch.ops.aten.mul.Tensor(mul_402, mul_403);  mul_402 = mul_403 = None
    unsqueeze_358: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
    unsqueeze_359: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_405: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_361: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_362: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    mul_406: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_360);  sub_113 = unsqueeze_360 = None
    sub_115: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(where_16, mul_406);  where_16 = mul_406 = None
    sub_116: "f32[8, 128, 32, 32]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_357);  sub_115 = unsqueeze_357 = None
    mul_407: "f32[8, 128, 32, 32]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_363);  sub_116 = unsqueeze_363 = None
    mul_408: "f32[128]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_34);  sum_53 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_407, relu_9, primals_80, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_407 = primals_80 = None
    getitem_130: "f32[8, 128, 64, 64]" = convolution_backward_19[0]
    getitem_131: "f32[128, 128, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_85: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_86: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_85);  alias_85 = None
    le_17: "b8[8, 128, 64, 64]" = torch.ops.aten.le.Scalar(alias_86, 0);  alias_86 = None
    where_17: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(le_17, full_default, getitem_130);  le_17 = getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_117: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_366);  convolution_10 = unsqueeze_366 = None
    mul_409: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_17, sub_117)
    sum_55: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3]);  mul_409 = None
    mul_410: "f32[128]" = torch.ops.aten.mul.Tensor(sum_54, 3.0517578125e-05)
    unsqueeze_367: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
    unsqueeze_368: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_411: "f32[128]" = torch.ops.aten.mul.Tensor(sum_55, 3.0517578125e-05)
    mul_412: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_413: "f32[128]" = torch.ops.aten.mul.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
    unsqueeze_370: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
    unsqueeze_371: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_414: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_373: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
    unsqueeze_374: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    mul_415: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_372);  sub_117 = unsqueeze_372 = None
    sub_119: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_17, mul_415);  where_17 = mul_415 = None
    sub_120: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_369);  sub_119 = unsqueeze_369 = None
    mul_416: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_375);  sub_120 = unsqueeze_375 = None
    mul_417: "f32[128]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_31);  sum_55 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_416, relu_8, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_416 = primals_79 = None
    getitem_133: "f32[8, 256, 64, 64]" = convolution_backward_20[0]
    getitem_134: "f32[128, 256, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_180: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(getitem_124, getitem_133);  getitem_124 = getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_88: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_89: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(alias_88);  alias_88 = None
    le_18: "b8[8, 256, 64, 64]" = torch.ops.aten.le.Scalar(alias_89, 0);  alias_89 = None
    where_18: "f32[8, 256, 64, 64]" = torch.ops.aten.where.self(le_18, full_default, add_180);  le_18 = add_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_121: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_378);  convolution_9 = unsqueeze_378 = None
    mul_418: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(where_18, sub_121)
    sum_57: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
    mul_419: "f32[256]" = torch.ops.aten.mul.Tensor(sum_56, 3.0517578125e-05)
    unsqueeze_379: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
    unsqueeze_380: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_420: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, 3.0517578125e-05)
    mul_421: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_422: "f32[256]" = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    unsqueeze_382: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_383: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_423: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_385: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_386: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_424: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_384);  sub_121 = unsqueeze_384 = None
    sub_123: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(where_18, mul_424);  mul_424 = None
    sub_124: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_381);  sub_123 = unsqueeze_381 = None
    mul_425: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_387);  sub_124 = unsqueeze_387 = None
    mul_426: "f32[256]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_28);  sum_57 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_425, relu_7, primals_78, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_425 = primals_78 = None
    getitem_136: "f32[8, 64, 64, 64]" = convolution_backward_21[0]
    getitem_137: "f32[256, 64, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_91: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_92: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_91);  alias_91 = None
    le_19: "b8[8, 64, 64, 64]" = torch.ops.aten.le.Scalar(alias_92, 0);  alias_92 = None
    where_19: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(le_19, full_default, getitem_136);  le_19 = getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_125: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_390);  convolution_8 = unsqueeze_390 = None
    mul_427: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_19, sub_125)
    sum_59: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None
    mul_428: "f32[64]" = torch.ops.aten.mul.Tensor(sum_58, 3.0517578125e-05)
    unsqueeze_391: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_392: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_429: "f32[64]" = torch.ops.aten.mul.Tensor(sum_59, 3.0517578125e-05)
    mul_430: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_431: "f32[64]" = torch.ops.aten.mul.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
    unsqueeze_394: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_395: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_432: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_397: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
    unsqueeze_398: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    mul_433: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_396);  sub_125 = unsqueeze_396 = None
    sub_127: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_19, mul_433);  where_19 = mul_433 = None
    sub_128: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_393);  sub_127 = unsqueeze_393 = None
    mul_434: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_399);  sub_128 = unsqueeze_399 = None
    mul_435: "f32[64]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_25);  sum_59 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_434, relu_6, primals_77, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_434 = primals_77 = None
    getitem_139: "f32[8, 64, 64, 64]" = convolution_backward_22[0]
    getitem_140: "f32[64, 64, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_94: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_95: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_94);  alias_94 = None
    le_20: "b8[8, 64, 64, 64]" = torch.ops.aten.le.Scalar(alias_95, 0);  alias_95 = None
    where_20: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(le_20, full_default, getitem_139);  le_20 = getitem_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_60: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_129: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_402);  convolution_7 = unsqueeze_402 = None
    mul_436: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_20, sub_129)
    sum_61: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3]);  mul_436 = None
    mul_437: "f32[64]" = torch.ops.aten.mul.Tensor(sum_60, 3.0517578125e-05)
    unsqueeze_403: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_437, 0);  mul_437 = None
    unsqueeze_404: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_438: "f32[64]" = torch.ops.aten.mul.Tensor(sum_61, 3.0517578125e-05)
    mul_439: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_440: "f32[64]" = torch.ops.aten.mul.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_406: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_407: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_441: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_409: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_410: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    mul_442: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_408);  sub_129 = unsqueeze_408 = None
    sub_131: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_20, mul_442);  where_20 = mul_442 = None
    sub_132: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_405);  sub_131 = unsqueeze_405 = None
    mul_443: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_411);  sub_132 = unsqueeze_411 = None
    mul_444: "f32[64]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_22);  sum_61 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_443, relu_5, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_443 = primals_76 = None
    getitem_142: "f32[8, 256, 64, 64]" = convolution_backward_23[0]
    getitem_143: "f32[64, 256, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_181: "f32[8, 256, 64, 64]" = torch.ops.aten.add.Tensor(where_18, getitem_142);  where_18 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_97: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_98: "f32[8, 256, 64, 64]" = torch.ops.aten.alias.default(alias_97);  alias_97 = None
    le_21: "b8[8, 256, 64, 64]" = torch.ops.aten.le.Scalar(alias_98, 0);  alias_98 = None
    where_21: "f32[8, 256, 64, 64]" = torch.ops.aten.where.self(le_21, full_default, add_181);  le_21 = add_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_62: "f32[256]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_133: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_414);  convolution_6 = unsqueeze_414 = None
    mul_445: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(where_21, sub_133)
    sum_63: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
    mul_446: "f32[256]" = torch.ops.aten.mul.Tensor(sum_62, 3.0517578125e-05)
    unsqueeze_415: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
    unsqueeze_416: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_447: "f32[256]" = torch.ops.aten.mul.Tensor(sum_63, 3.0517578125e-05)
    mul_448: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_449: "f32[256]" = torch.ops.aten.mul.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
    unsqueeze_418: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
    unsqueeze_419: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_450: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_421: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_422: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    mul_451: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_420);  sub_133 = unsqueeze_420 = None
    sub_135: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(where_21, mul_451);  mul_451 = None
    sub_136: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_417);  sub_135 = None
    mul_452: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_423);  sub_136 = unsqueeze_423 = None
    mul_453: "f32[256]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_19);  sum_63 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_452, getitem_6, primals_75, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_452 = primals_75 = None
    getitem_145: "f32[8, 64, 64, 64]" = convolution_backward_24[0]
    getitem_146: "f32[256, 64, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_137: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_426);  convolution_5 = unsqueeze_426 = None
    mul_454: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(where_21, sub_137)
    sum_65: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 2, 3]);  mul_454 = None
    mul_456: "f32[256]" = torch.ops.aten.mul.Tensor(sum_65, 3.0517578125e-05)
    mul_457: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_458: "f32[256]" = torch.ops.aten.mul.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    unsqueeze_430: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_431: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_459: "f32[256]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_433: "f32[1, 256]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_434: "f32[1, 256, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    mul_460: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_432);  sub_137 = unsqueeze_432 = None
    sub_139: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(where_21, mul_460);  where_21 = mul_460 = None
    sub_140: "f32[8, 256, 64, 64]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_417);  sub_139 = unsqueeze_417 = None
    mul_461: "f32[8, 256, 64, 64]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_435);  sub_140 = unsqueeze_435 = None
    mul_462: "f32[256]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_16);  sum_65 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_461, relu_4, primals_74, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_461 = primals_74 = None
    getitem_148: "f32[8, 64, 64, 64]" = convolution_backward_25[0]
    getitem_149: "f32[256, 64, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_100: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_101: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_100);  alias_100 = None
    le_22: "b8[8, 64, 64, 64]" = torch.ops.aten.le.Scalar(alias_101, 0);  alias_101 = None
    where_22: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(le_22, full_default, getitem_148);  le_22 = getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_66: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_141: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_438);  convolution_4 = unsqueeze_438 = None
    mul_463: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_22, sub_141)
    sum_67: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
    mul_464: "f32[64]" = torch.ops.aten.mul.Tensor(sum_66, 3.0517578125e-05)
    unsqueeze_439: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_464, 0);  mul_464 = None
    unsqueeze_440: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_465: "f32[64]" = torch.ops.aten.mul.Tensor(sum_67, 3.0517578125e-05)
    mul_466: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_467: "f32[64]" = torch.ops.aten.mul.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
    unsqueeze_442: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
    unsqueeze_443: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_468: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_445: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_446: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    mul_469: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_444);  sub_141 = unsqueeze_444 = None
    sub_143: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_22, mul_469);  where_22 = mul_469 = None
    sub_144: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_441);  sub_143 = unsqueeze_441 = None
    mul_470: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_447);  sub_144 = unsqueeze_447 = None
    mul_471: "f32[64]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_13);  sum_67 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_470, relu_3, primals_73, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_470 = primals_73 = None
    getitem_151: "f32[8, 64, 64, 64]" = convolution_backward_26[0]
    getitem_152: "f32[64, 64, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_103: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_104: "f32[8, 64, 64, 64]" = torch.ops.aten.alias.default(alias_103);  alias_103 = None
    le_23: "b8[8, 64, 64, 64]" = torch.ops.aten.le.Scalar(alias_104, 0);  alias_104 = None
    where_23: "f32[8, 64, 64, 64]" = torch.ops.aten.where.self(le_23, full_default, getitem_151);  le_23 = getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_68: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_145: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_450);  convolution_3 = unsqueeze_450 = None
    mul_472: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(where_23, sub_145)
    sum_69: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 2, 3]);  mul_472 = None
    mul_473: "f32[64]" = torch.ops.aten.mul.Tensor(sum_68, 3.0517578125e-05)
    unsqueeze_451: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_473, 0);  mul_473 = None
    unsqueeze_452: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_474: "f32[64]" = torch.ops.aten.mul.Tensor(sum_69, 3.0517578125e-05)
    mul_475: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_476: "f32[64]" = torch.ops.aten.mul.Tensor(mul_474, mul_475);  mul_474 = mul_475 = None
    unsqueeze_454: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
    unsqueeze_455: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_477: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_457: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_458: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    mul_478: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_456);  sub_145 = unsqueeze_456 = None
    sub_147: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(where_23, mul_478);  where_23 = mul_478 = None
    sub_148: "f32[8, 64, 64, 64]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_453);  sub_147 = unsqueeze_453 = None
    mul_479: "f32[8, 64, 64, 64]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_459);  sub_148 = unsqueeze_459 = None
    mul_480: "f32[64]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_10);  sum_69 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_479, getitem_6, primals_72, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_479 = getitem_6 = primals_72 = None
    getitem_154: "f32[8, 64, 64, 64]" = convolution_backward_27[0]
    getitem_155: "f32[64, 64, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_182: "f32[8, 64, 64, 64]" = torch.ops.aten.add.Tensor(getitem_145, getitem_154);  getitem_145 = getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:1245, code: x = self.stem(x)
    max_pool2d_with_indices_backward: "f32[8, 64, 128, 128]" = torch.ops.aten.max_pool2d_with_indices_backward.default(add_182, relu_2, [3, 3], [2, 2], [1, 1], [1, 1], False, getitem_7);  add_182 = getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_106: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_107: "f32[8, 64, 128, 128]" = torch.ops.aten.alias.default(alias_106);  alias_106 = None
    le_24: "b8[8, 64, 128, 128]" = torch.ops.aten.le.Scalar(alias_107, 0);  alias_107 = None
    where_24: "f32[8, 64, 128, 128]" = torch.ops.aten.where.self(le_24, full_default, max_pool2d_with_indices_backward);  le_24 = max_pool2d_with_indices_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_70: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_149: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_462);  convolution_2 = unsqueeze_462 = None
    mul_481: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(where_24, sub_149)
    sum_71: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 2, 3]);  mul_481 = None
    mul_482: "f32[64]" = torch.ops.aten.mul.Tensor(sum_70, 7.62939453125e-06)
    unsqueeze_463: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
    unsqueeze_464: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_483: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, 7.62939453125e-06)
    mul_484: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_485: "f32[64]" = torch.ops.aten.mul.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    unsqueeze_466: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
    unsqueeze_467: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_486: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_469: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
    unsqueeze_470: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    mul_487: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_468);  sub_149 = unsqueeze_468 = None
    sub_151: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(where_24, mul_487);  where_24 = mul_487 = None
    sub_152: "f32[8, 64, 128, 128]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_465);  sub_151 = unsqueeze_465 = None
    mul_488: "f32[8, 64, 128, 128]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_471);  sub_152 = unsqueeze_471 = None
    mul_489: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_7);  sum_71 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_488, relu_1, primals_71, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_488 = primals_71 = None
    getitem_157: "f32[8, 32, 128, 128]" = convolution_backward_28[0]
    getitem_158: "f32[64, 32, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_109: "f32[8, 32, 128, 128]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_110: "f32[8, 32, 128, 128]" = torch.ops.aten.alias.default(alias_109);  alias_109 = None
    le_25: "b8[8, 32, 128, 128]" = torch.ops.aten.le.Scalar(alias_110, 0);  alias_110 = None
    where_25: "f32[8, 32, 128, 128]" = torch.ops.aten.where.self(le_25, full_default, getitem_157);  le_25 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_153: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_474);  convolution_1 = unsqueeze_474 = None
    mul_490: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(where_25, sub_153)
    sum_73: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2, 3]);  mul_490 = None
    mul_491: "f32[32]" = torch.ops.aten.mul.Tensor(sum_72, 7.62939453125e-06)
    unsqueeze_475: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
    unsqueeze_476: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_492: "f32[32]" = torch.ops.aten.mul.Tensor(sum_73, 7.62939453125e-06)
    mul_493: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_494: "f32[32]" = torch.ops.aten.mul.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    unsqueeze_478: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_479: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_495: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_481: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_482: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    mul_496: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_480);  sub_153 = unsqueeze_480 = None
    sub_155: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(where_25, mul_496);  where_25 = mul_496 = None
    sub_156: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_477);  sub_155 = unsqueeze_477 = None
    mul_497: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_483);  sub_156 = unsqueeze_483 = None
    mul_498: "f32[32]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_4);  sum_73 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_497, relu, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_497 = primals_70 = None
    getitem_160: "f32[8, 24, 128, 128]" = convolution_backward_29[0]
    getitem_161: "f32[32, 24, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_112: "f32[8, 24, 128, 128]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_113: "f32[8, 24, 128, 128]" = torch.ops.aten.alias.default(alias_112);  alias_112 = None
    le_26: "b8[8, 24, 128, 128]" = torch.ops.aten.le.Scalar(alias_113, 0);  alias_113 = None
    where_26: "f32[8, 24, 128, 128]" = torch.ops.aten.where.self(le_26, full_default, getitem_160);  le_26 = full_default = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_157: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_486);  convolution = unsqueeze_486 = None
    mul_499: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(where_26, sub_157)
    sum_75: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 2, 3]);  mul_499 = None
    mul_500: "f32[24]" = torch.ops.aten.mul.Tensor(sum_74, 7.62939453125e-06)
    unsqueeze_487: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
    unsqueeze_488: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_501: "f32[24]" = torch.ops.aten.mul.Tensor(sum_75, 7.62939453125e-06)
    mul_502: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_503: "f32[24]" = torch.ops.aten.mul.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
    unsqueeze_490: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    unsqueeze_491: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_504: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_493: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_494: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    mul_505: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_492);  sub_157 = unsqueeze_492 = None
    sub_159: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(where_26, mul_505);  where_26 = mul_505 = None
    sub_160: "f32[8, 24, 128, 128]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_489);  sub_159 = unsqueeze_489 = None
    mul_506: "f32[8, 24, 128, 128]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_495);  sub_160 = unsqueeze_495 = None
    mul_507: "f32[24]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_1);  sum_75 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_506, primals_195, primals_69, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_506 = primals_195 = primals_69 = None
    getitem_164: "f32[24, 3, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    return [mul_507, sum_74, mul_498, sum_72, mul_489, sum_70, mul_480, sum_68, mul_471, sum_66, mul_462, sum_62, mul_453, sum_62, mul_444, sum_60, mul_435, sum_58, mul_426, sum_56, mul_417, sum_54, mul_408, sum_52, mul_399, sum_48, mul_390, sum_48, mul_381, sum_46, mul_372, sum_44, mul_363, sum_42, mul_354, sum_40, mul_345, sum_38, mul_336, sum_34, mul_327, sum_34, mul_318, sum_32, permute_78, permute_72, mul_306, sum_27, mul_297, sum_25, mul_288, sum_23, permute_60, permute_54, mul_276, sum_18, mul_267, sum_14, mul_258, sum_14, mul_249, sum_12, permute_42, permute_36, mul_237, sum_7, mul_228, sum_5, getitem_164, getitem_161, getitem_158, getitem_155, getitem_152, getitem_149, getitem_146, getitem_143, getitem_140, getitem_137, getitem_134, getitem_131, getitem_128, getitem_125, getitem_122, getitem_119, getitem_116, getitem_113, getitem_110, getitem_107, getitem_104, getitem_101, getitem_98, getitem_95, getitem_92, getitem_89, getitem_86, getitem_83, getitem_80, getitem_77, getitem_74, permute_28, view_73, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    