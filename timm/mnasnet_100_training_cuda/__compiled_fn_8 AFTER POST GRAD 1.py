from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_3: "f32[32]", primals_5: "f32[16]", primals_7: "f32[48]", primals_9: "f32[48]", primals_11: "f32[24]", primals_13: "f32[72]", primals_15: "f32[72]", primals_17: "f32[24]", primals_19: "f32[72]", primals_21: "f32[72]", primals_23: "f32[24]", primals_25: "f32[72]", primals_27: "f32[72]", primals_29: "f32[40]", primals_31: "f32[120]", primals_33: "f32[120]", primals_35: "f32[40]", primals_37: "f32[120]", primals_39: "f32[120]", primals_41: "f32[40]", primals_43: "f32[240]", primals_45: "f32[240]", primals_47: "f32[80]", primals_49: "f32[480]", primals_51: "f32[480]", primals_53: "f32[80]", primals_55: "f32[480]", primals_57: "f32[480]", primals_59: "f32[80]", primals_61: "f32[480]", primals_63: "f32[480]", primals_65: "f32[96]", primals_67: "f32[576]", primals_69: "f32[576]", primals_71: "f32[96]", primals_73: "f32[576]", primals_75: "f32[576]", primals_77: "f32[192]", primals_79: "f32[1152]", primals_81: "f32[1152]", primals_83: "f32[192]", primals_85: "f32[1152]", primals_87: "f32[1152]", primals_89: "f32[192]", primals_91: "f32[1152]", primals_93: "f32[1152]", primals_95: "f32[192]", primals_97: "f32[1152]", primals_99: "f32[1152]", primals_101: "f32[320]", primals_103: "f32[1280]", primals_105: "f32[32, 3, 3, 3]", primals_106: "f32[32, 1, 3, 3]", primals_107: "f32[16, 32, 1, 1]", primals_108: "f32[48, 16, 1, 1]", primals_109: "f32[48, 1, 3, 3]", primals_110: "f32[24, 48, 1, 1]", primals_111: "f32[72, 24, 1, 1]", primals_112: "f32[72, 1, 3, 3]", primals_113: "f32[24, 72, 1, 1]", primals_114: "f32[72, 24, 1, 1]", primals_115: "f32[72, 1, 3, 3]", primals_116: "f32[24, 72, 1, 1]", primals_117: "f32[72, 24, 1, 1]", primals_118: "f32[72, 1, 5, 5]", primals_119: "f32[40, 72, 1, 1]", primals_120: "f32[120, 40, 1, 1]", primals_121: "f32[120, 1, 5, 5]", primals_122: "f32[40, 120, 1, 1]", primals_123: "f32[120, 40, 1, 1]", primals_124: "f32[120, 1, 5, 5]", primals_125: "f32[40, 120, 1, 1]", primals_126: "f32[240, 40, 1, 1]", primals_127: "f32[240, 1, 5, 5]", primals_128: "f32[80, 240, 1, 1]", primals_129: "f32[480, 80, 1, 1]", primals_130: "f32[480, 1, 5, 5]", primals_131: "f32[80, 480, 1, 1]", primals_132: "f32[480, 80, 1, 1]", primals_133: "f32[480, 1, 5, 5]", primals_134: "f32[80, 480, 1, 1]", primals_135: "f32[480, 80, 1, 1]", primals_136: "f32[480, 1, 3, 3]", primals_137: "f32[96, 480, 1, 1]", primals_138: "f32[576, 96, 1, 1]", primals_139: "f32[576, 1, 3, 3]", primals_140: "f32[96, 576, 1, 1]", primals_141: "f32[576, 96, 1, 1]", primals_142: "f32[576, 1, 5, 5]", primals_143: "f32[192, 576, 1, 1]", primals_144: "f32[1152, 192, 1, 1]", primals_145: "f32[1152, 1, 5, 5]", primals_146: "f32[192, 1152, 1, 1]", primals_147: "f32[1152, 192, 1, 1]", primals_148: "f32[1152, 1, 5, 5]", primals_149: "f32[192, 1152, 1, 1]", primals_150: "f32[1152, 192, 1, 1]", primals_151: "f32[1152, 1, 5, 5]", primals_152: "f32[192, 1152, 1, 1]", primals_153: "f32[1152, 192, 1, 1]", primals_154: "f32[1152, 1, 3, 3]", primals_155: "f32[320, 1152, 1, 1]", primals_156: "f32[1280, 320, 1, 1]", primals_315: "f32[8, 3, 224, 224]", convolution: "f32[8, 32, 112, 112]", squeeze_1: "f32[32]", relu: "f32[8, 32, 112, 112]", convolution_1: "f32[8, 32, 112, 112]", squeeze_4: "f32[32]", relu_1: "f32[8, 32, 112, 112]", convolution_2: "f32[8, 16, 112, 112]", squeeze_7: "f32[16]", add_14: "f32[8, 16, 112, 112]", convolution_3: "f32[8, 48, 112, 112]", squeeze_10: "f32[48]", relu_2: "f32[8, 48, 112, 112]", convolution_4: "f32[8, 48, 56, 56]", squeeze_13: "f32[48]", relu_3: "f32[8, 48, 56, 56]", convolution_5: "f32[8, 24, 56, 56]", squeeze_16: "f32[24]", add_29: "f32[8, 24, 56, 56]", convolution_6: "f32[8, 72, 56, 56]", squeeze_19: "f32[72]", relu_4: "f32[8, 72, 56, 56]", convolution_7: "f32[8, 72, 56, 56]", squeeze_22: "f32[72]", relu_5: "f32[8, 72, 56, 56]", convolution_8: "f32[8, 24, 56, 56]", squeeze_25: "f32[24]", add_45: "f32[8, 24, 56, 56]", convolution_9: "f32[8, 72, 56, 56]", squeeze_28: "f32[72]", relu_6: "f32[8, 72, 56, 56]", convolution_10: "f32[8, 72, 56, 56]", squeeze_31: "f32[72]", relu_7: "f32[8, 72, 56, 56]", convolution_11: "f32[8, 24, 56, 56]", squeeze_34: "f32[24]", add_61: "f32[8, 24, 56, 56]", convolution_12: "f32[8, 72, 56, 56]", squeeze_37: "f32[72]", relu_8: "f32[8, 72, 56, 56]", convolution_13: "f32[8, 72, 28, 28]", squeeze_40: "f32[72]", relu_9: "f32[8, 72, 28, 28]", convolution_14: "f32[8, 40, 28, 28]", squeeze_43: "f32[40]", add_76: "f32[8, 40, 28, 28]", convolution_15: "f32[8, 120, 28, 28]", squeeze_46: "f32[120]", relu_10: "f32[8, 120, 28, 28]", convolution_16: "f32[8, 120, 28, 28]", squeeze_49: "f32[120]", relu_11: "f32[8, 120, 28, 28]", convolution_17: "f32[8, 40, 28, 28]", squeeze_52: "f32[40]", add_92: "f32[8, 40, 28, 28]", convolution_18: "f32[8, 120, 28, 28]", squeeze_55: "f32[120]", relu_12: "f32[8, 120, 28, 28]", convolution_19: "f32[8, 120, 28, 28]", squeeze_58: "f32[120]", relu_13: "f32[8, 120, 28, 28]", convolution_20: "f32[8, 40, 28, 28]", squeeze_61: "f32[40]", add_108: "f32[8, 40, 28, 28]", convolution_21: "f32[8, 240, 28, 28]", squeeze_64: "f32[240]", relu_14: "f32[8, 240, 28, 28]", convolution_22: "f32[8, 240, 14, 14]", squeeze_67: "f32[240]", relu_15: "f32[8, 240, 14, 14]", convolution_23: "f32[8, 80, 14, 14]", squeeze_70: "f32[80]", add_123: "f32[8, 80, 14, 14]", convolution_24: "f32[8, 480, 14, 14]", squeeze_73: "f32[480]", relu_16: "f32[8, 480, 14, 14]", convolution_25: "f32[8, 480, 14, 14]", squeeze_76: "f32[480]", relu_17: "f32[8, 480, 14, 14]", convolution_26: "f32[8, 80, 14, 14]", squeeze_79: "f32[80]", add_139: "f32[8, 80, 14, 14]", convolution_27: "f32[8, 480, 14, 14]", squeeze_82: "f32[480]", relu_18: "f32[8, 480, 14, 14]", convolution_28: "f32[8, 480, 14, 14]", squeeze_85: "f32[480]", relu_19: "f32[8, 480, 14, 14]", convolution_29: "f32[8, 80, 14, 14]", squeeze_88: "f32[80]", add_155: "f32[8, 80, 14, 14]", convolution_30: "f32[8, 480, 14, 14]", squeeze_91: "f32[480]", relu_20: "f32[8, 480, 14, 14]", convolution_31: "f32[8, 480, 14, 14]", squeeze_94: "f32[480]", relu_21: "f32[8, 480, 14, 14]", convolution_32: "f32[8, 96, 14, 14]", squeeze_97: "f32[96]", add_170: "f32[8, 96, 14, 14]", convolution_33: "f32[8, 576, 14, 14]", squeeze_100: "f32[576]", relu_22: "f32[8, 576, 14, 14]", convolution_34: "f32[8, 576, 14, 14]", squeeze_103: "f32[576]", relu_23: "f32[8, 576, 14, 14]", convolution_35: "f32[8, 96, 14, 14]", squeeze_106: "f32[96]", add_186: "f32[8, 96, 14, 14]", convolution_36: "f32[8, 576, 14, 14]", squeeze_109: "f32[576]", relu_24: "f32[8, 576, 14, 14]", convolution_37: "f32[8, 576, 7, 7]", squeeze_112: "f32[576]", relu_25: "f32[8, 576, 7, 7]", convolution_38: "f32[8, 192, 7, 7]", squeeze_115: "f32[192]", add_201: "f32[8, 192, 7, 7]", convolution_39: "f32[8, 1152, 7, 7]", squeeze_118: "f32[1152]", relu_26: "f32[8, 1152, 7, 7]", convolution_40: "f32[8, 1152, 7, 7]", squeeze_121: "f32[1152]", relu_27: "f32[8, 1152, 7, 7]", convolution_41: "f32[8, 192, 7, 7]", squeeze_124: "f32[192]", add_217: "f32[8, 192, 7, 7]", convolution_42: "f32[8, 1152, 7, 7]", squeeze_127: "f32[1152]", relu_28: "f32[8, 1152, 7, 7]", convolution_43: "f32[8, 1152, 7, 7]", squeeze_130: "f32[1152]", relu_29: "f32[8, 1152, 7, 7]", convolution_44: "f32[8, 192, 7, 7]", squeeze_133: "f32[192]", add_233: "f32[8, 192, 7, 7]", convolution_45: "f32[8, 1152, 7, 7]", squeeze_136: "f32[1152]", relu_30: "f32[8, 1152, 7, 7]", convolution_46: "f32[8, 1152, 7, 7]", squeeze_139: "f32[1152]", relu_31: "f32[8, 1152, 7, 7]", convolution_47: "f32[8, 192, 7, 7]", squeeze_142: "f32[192]", add_249: "f32[8, 192, 7, 7]", convolution_48: "f32[8, 1152, 7, 7]", squeeze_145: "f32[1152]", relu_32: "f32[8, 1152, 7, 7]", convolution_49: "f32[8, 1152, 7, 7]", squeeze_148: "f32[1152]", relu_33: "f32[8, 1152, 7, 7]", convolution_50: "f32[8, 320, 7, 7]", squeeze_151: "f32[320]", add_264: "f32[8, 320, 7, 7]", convolution_51: "f32[8, 1280, 7, 7]", squeeze_154: "f32[1280]", view: "f32[8, 1280]", permute_1: "f32[1000, 1280]", le: "b8[8, 1280, 7, 7]", unsqueeze_210: "f32[1, 1280, 1, 1]", unsqueeze_222: "f32[1, 320, 1, 1]", unsqueeze_234: "f32[1, 1152, 1, 1]", unsqueeze_246: "f32[1, 1152, 1, 1]", unsqueeze_258: "f32[1, 192, 1, 1]", unsqueeze_270: "f32[1, 1152, 1, 1]", unsqueeze_282: "f32[1, 1152, 1, 1]", unsqueeze_294: "f32[1, 192, 1, 1]", unsqueeze_306: "f32[1, 1152, 1, 1]", unsqueeze_318: "f32[1, 1152, 1, 1]", unsqueeze_330: "f32[1, 192, 1, 1]", unsqueeze_342: "f32[1, 1152, 1, 1]", unsqueeze_354: "f32[1, 1152, 1, 1]", unsqueeze_366: "f32[1, 192, 1, 1]", unsqueeze_378: "f32[1, 576, 1, 1]", unsqueeze_390: "f32[1, 576, 1, 1]", unsqueeze_402: "f32[1, 96, 1, 1]", unsqueeze_414: "f32[1, 576, 1, 1]", unsqueeze_426: "f32[1, 576, 1, 1]", unsqueeze_438: "f32[1, 96, 1, 1]", unsqueeze_450: "f32[1, 480, 1, 1]", unsqueeze_462: "f32[1, 480, 1, 1]", unsqueeze_474: "f32[1, 80, 1, 1]", unsqueeze_486: "f32[1, 480, 1, 1]", unsqueeze_498: "f32[1, 480, 1, 1]", unsqueeze_510: "f32[1, 80, 1, 1]", unsqueeze_522: "f32[1, 480, 1, 1]", unsqueeze_534: "f32[1, 480, 1, 1]", unsqueeze_546: "f32[1, 80, 1, 1]", unsqueeze_558: "f32[1, 240, 1, 1]", unsqueeze_570: "f32[1, 240, 1, 1]", unsqueeze_582: "f32[1, 40, 1, 1]", unsqueeze_594: "f32[1, 120, 1, 1]", unsqueeze_606: "f32[1, 120, 1, 1]", unsqueeze_618: "f32[1, 40, 1, 1]", unsqueeze_630: "f32[1, 120, 1, 1]", unsqueeze_642: "f32[1, 120, 1, 1]", unsqueeze_654: "f32[1, 40, 1, 1]", unsqueeze_666: "f32[1, 72, 1, 1]", unsqueeze_678: "f32[1, 72, 1, 1]", unsqueeze_690: "f32[1, 24, 1, 1]", unsqueeze_702: "f32[1, 72, 1, 1]", unsqueeze_714: "f32[1, 72, 1, 1]", unsqueeze_726: "f32[1, 24, 1, 1]", unsqueeze_738: "f32[1, 72, 1, 1]", unsqueeze_750: "f32[1, 72, 1, 1]", unsqueeze_762: "f32[1, 24, 1, 1]", unsqueeze_774: "f32[1, 48, 1, 1]", unsqueeze_786: "f32[1, 48, 1, 1]", unsqueeze_798: "f32[1, 16, 1, 1]", unsqueeze_810: "f32[1, 32, 1, 1]", unsqueeze_822: "f32[1, 32, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    mm: "f32[8, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1280, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 1280, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1280, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 1280, 7, 7]);  view_2 = None
    div: "f32[8, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 1280, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_52: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_210);  convolution_51 = unsqueeze_210 = None
    mul_364: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_52)
    sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3]);  mul_364 = None
    mul_365: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_211: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_365, 0);  mul_365 = None
    unsqueeze_212: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_211, 2);  unsqueeze_211 = None
    unsqueeze_213: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, 3);  unsqueeze_212 = None
    mul_366: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_367: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_368: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_366, mul_367);  mul_366 = mul_367 = None
    unsqueeze_214: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_368, 0);  mul_368 = None
    unsqueeze_215: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, 2);  unsqueeze_214 = None
    unsqueeze_216: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 3);  unsqueeze_215 = None
    mul_369: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_217: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
    unsqueeze_218: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 2);  unsqueeze_217 = None
    unsqueeze_219: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 3);  unsqueeze_218 = None
    mul_370: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_216);  sub_52 = unsqueeze_216 = None
    sub_54: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_370);  where = mul_370 = None
    sub_55: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_54, unsqueeze_213);  sub_54 = unsqueeze_213 = None
    mul_371: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_219);  sub_55 = unsqueeze_219 = None
    mul_372: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_154);  sum_3 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_371, add_264, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_371 = add_264 = primals_156 = None
    getitem_104: "f32[8, 320, 7, 7]" = convolution_backward[0]
    getitem_105: "f32[1280, 320, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(getitem_104, [0, 2, 3])
    sub_56: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_222);  convolution_50 = unsqueeze_222 = None
    mul_373: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_104, sub_56)
    sum_5: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_373, [0, 2, 3]);  mul_373 = None
    mul_374: "f32[320]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_223: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_374, 0);  mul_374 = None
    unsqueeze_224: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_223, 2);  unsqueeze_223 = None
    unsqueeze_225: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, 3);  unsqueeze_224 = None
    mul_375: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_376: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_377: "f32[320]" = torch.ops.aten.mul.Tensor(mul_375, mul_376);  mul_375 = mul_376 = None
    unsqueeze_226: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_377, 0);  mul_377 = None
    unsqueeze_227: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 2);  unsqueeze_226 = None
    unsqueeze_228: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 3);  unsqueeze_227 = None
    mul_378: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_229: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_230: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_229, 2);  unsqueeze_229 = None
    unsqueeze_231: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 3);  unsqueeze_230 = None
    mul_379: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_228);  sub_56 = unsqueeze_228 = None
    sub_58: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_104, mul_379);  getitem_104 = mul_379 = None
    sub_59: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(sub_58, unsqueeze_225);  sub_58 = unsqueeze_225 = None
    mul_380: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_231);  sub_59 = unsqueeze_231 = None
    mul_381: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_151);  sum_5 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_380, relu_33, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_380 = primals_155 = None
    getitem_107: "f32[8, 1152, 7, 7]" = convolution_backward_1[0]
    getitem_108: "f32[320, 1152, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_1: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
    where_1: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, getitem_107);  le_1 = getitem_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_6: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_60: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_234);  convolution_49 = unsqueeze_234 = None
    mul_382: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_60)
    sum_7: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_382, [0, 2, 3]);  mul_382 = None
    mul_383: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_235: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_383, 0);  mul_383 = None
    unsqueeze_236: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 2);  unsqueeze_235 = None
    unsqueeze_237: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, 3);  unsqueeze_236 = None
    mul_384: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_385: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_386: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
    unsqueeze_238: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_386, 0);  mul_386 = None
    unsqueeze_239: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 2);  unsqueeze_238 = None
    unsqueeze_240: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 3);  unsqueeze_239 = None
    mul_387: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_241: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
    unsqueeze_242: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_241, 2);  unsqueeze_241 = None
    unsqueeze_243: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 3);  unsqueeze_242 = None
    mul_388: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_240);  sub_60 = unsqueeze_240 = None
    sub_62: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_388);  where_1 = mul_388 = None
    sub_63: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_62, unsqueeze_237);  sub_62 = unsqueeze_237 = None
    mul_389: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_243);  sub_63 = unsqueeze_243 = None
    mul_390: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_148);  sum_7 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_389, relu_32, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_389 = primals_154 = None
    getitem_110: "f32[8, 1152, 7, 7]" = convolution_backward_2[0]
    getitem_111: "f32[1152, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_2: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
    where_2: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, getitem_110);  le_2 = getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_8: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_64: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_246);  convolution_48 = unsqueeze_246 = None
    mul_391: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_64)
    sum_9: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_391, [0, 2, 3]);  mul_391 = None
    mul_392: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_247: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_392, 0);  mul_392 = None
    unsqueeze_248: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_393: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_394: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_395: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_393, mul_394);  mul_393 = mul_394 = None
    unsqueeze_250: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_395, 0);  mul_395 = None
    unsqueeze_251: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_396: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_253: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
    unsqueeze_254: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    mul_397: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_252);  sub_64 = unsqueeze_252 = None
    sub_66: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_397);  where_2 = mul_397 = None
    sub_67: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_249);  sub_66 = unsqueeze_249 = None
    mul_398: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_255);  sub_67 = unsqueeze_255 = None
    mul_399: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_145);  sum_9 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_398, add_249, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_398 = add_249 = primals_153 = None
    getitem_113: "f32[8, 192, 7, 7]" = convolution_backward_3[0]
    getitem_114: "f32[1152, 192, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_10: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_113, [0, 2, 3])
    sub_68: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_258);  convolution_47 = unsqueeze_258 = None
    mul_400: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_113, sub_68)
    sum_11: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
    mul_401: "f32[192]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_259: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_260: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
    mul_402: "f32[192]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_403: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_404: "f32[192]" = torch.ops.aten.mul.Tensor(mul_402, mul_403);  mul_402 = mul_403 = None
    unsqueeze_262: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
    unsqueeze_263: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_405: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_265: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_266: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    mul_406: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_264);  sub_68 = unsqueeze_264 = None
    sub_70: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_113, mul_406);  mul_406 = None
    sub_71: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_261);  sub_70 = unsqueeze_261 = None
    mul_407: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_267);  sub_71 = unsqueeze_267 = None
    mul_408: "f32[192]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_142);  sum_11 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_407, relu_31, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_407 = primals_152 = None
    getitem_116: "f32[8, 1152, 7, 7]" = convolution_backward_4[0]
    getitem_117: "f32[192, 1152, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_3: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
    where_3: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, getitem_116);  le_3 = getitem_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_12: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_72: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_270);  convolution_46 = unsqueeze_270 = None
    mul_409: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_72)
    sum_13: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3]);  mul_409 = None
    mul_410: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_271: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
    unsqueeze_272: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_411: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_412: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_413: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_411, mul_412);  mul_411 = mul_412 = None
    unsqueeze_274: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
    unsqueeze_275: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_414: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_277: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
    unsqueeze_278: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    mul_415: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_276);  sub_72 = unsqueeze_276 = None
    sub_74: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_415);  where_3 = mul_415 = None
    sub_75: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_273);  sub_74 = unsqueeze_273 = None
    mul_416: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_279);  sub_75 = unsqueeze_279 = None
    mul_417: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_139);  sum_13 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_416, relu_30, primals_151, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_416 = primals_151 = None
    getitem_119: "f32[8, 1152, 7, 7]" = convolution_backward_5[0]
    getitem_120: "f32[1152, 1, 5, 5]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_4: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
    where_4: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, getitem_119);  le_4 = getitem_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_14: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_76: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_282);  convolution_45 = unsqueeze_282 = None
    mul_418: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_76)
    sum_15: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
    mul_419: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_283: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
    unsqueeze_284: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_420: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_421: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_422: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    unsqueeze_286: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_287: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_423: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_289: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_290: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    mul_424: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_288);  sub_76 = unsqueeze_288 = None
    sub_78: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_424);  where_4 = mul_424 = None
    sub_79: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_285);  sub_78 = unsqueeze_285 = None
    mul_425: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_291);  sub_79 = unsqueeze_291 = None
    mul_426: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_136);  sum_15 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_425, add_233, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_425 = add_233 = primals_150 = None
    getitem_122: "f32[8, 192, 7, 7]" = convolution_backward_6[0]
    getitem_123: "f32[1152, 192, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_270: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(getitem_113, getitem_122);  getitem_113 = getitem_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_16: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_270, [0, 2, 3])
    sub_80: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_294);  convolution_44 = unsqueeze_294 = None
    mul_427: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_270, sub_80)
    sum_17: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None
    mul_428: "f32[192]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_295: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_296: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_429: "f32[192]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_430: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_431: "f32[192]" = torch.ops.aten.mul.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
    unsqueeze_298: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_299: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_432: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_301: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
    unsqueeze_302: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    mul_433: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_300);  sub_80 = unsqueeze_300 = None
    sub_82: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_270, mul_433);  mul_433 = None
    sub_83: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_297);  sub_82 = unsqueeze_297 = None
    mul_434: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_303);  sub_83 = unsqueeze_303 = None
    mul_435: "f32[192]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_133);  sum_17 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_434, relu_29, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_434 = primals_149 = None
    getitem_125: "f32[8, 1152, 7, 7]" = convolution_backward_7[0]
    getitem_126: "f32[192, 1152, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_5: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
    where_5: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, getitem_125);  le_5 = getitem_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_84: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_306);  convolution_43 = unsqueeze_306 = None
    mul_436: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_84)
    sum_19: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3]);  mul_436 = None
    mul_437: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_307: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_437, 0);  mul_437 = None
    unsqueeze_308: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_438: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_439: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_440: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_310: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_311: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_441: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_313: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_314: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    mul_442: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_312);  sub_84 = unsqueeze_312 = None
    sub_86: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_442);  where_5 = mul_442 = None
    sub_87: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_309);  sub_86 = unsqueeze_309 = None
    mul_443: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_315);  sub_87 = unsqueeze_315 = None
    mul_444: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_130);  sum_19 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_443, relu_28, primals_148, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_443 = primals_148 = None
    getitem_128: "f32[8, 1152, 7, 7]" = convolution_backward_8[0]
    getitem_129: "f32[1152, 1, 5, 5]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_6: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
    where_6: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, getitem_128);  le_6 = getitem_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_20: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_88: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_318);  convolution_42 = unsqueeze_318 = None
    mul_445: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_88)
    sum_21: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
    mul_446: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_319: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
    unsqueeze_320: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_447: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_448: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_449: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
    unsqueeze_322: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
    unsqueeze_323: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_450: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_325: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_326: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    mul_451: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_324);  sub_88 = unsqueeze_324 = None
    sub_90: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_451);  where_6 = mul_451 = None
    sub_91: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_321);  sub_90 = unsqueeze_321 = None
    mul_452: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_327);  sub_91 = unsqueeze_327 = None
    mul_453: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_127);  sum_21 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_452, add_217, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_452 = add_217 = primals_147 = None
    getitem_131: "f32[8, 192, 7, 7]" = convolution_backward_9[0]
    getitem_132: "f32[1152, 192, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_271: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_270, getitem_131);  add_270 = getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_22: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_271, [0, 2, 3])
    sub_92: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_330);  convolution_41 = unsqueeze_330 = None
    mul_454: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_271, sub_92)
    sum_23: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 2, 3]);  mul_454 = None
    mul_455: "f32[192]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    unsqueeze_331: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_455, 0);  mul_455 = None
    unsqueeze_332: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_456: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    mul_457: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_458: "f32[192]" = torch.ops.aten.mul.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    unsqueeze_334: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_335: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_459: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_337: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_338: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    mul_460: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_336);  sub_92 = unsqueeze_336 = None
    sub_94: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_271, mul_460);  mul_460 = None
    sub_95: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_333);  sub_94 = unsqueeze_333 = None
    mul_461: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_339);  sub_95 = unsqueeze_339 = None
    mul_462: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_124);  sum_23 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_461, relu_27, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_461 = primals_146 = None
    getitem_134: "f32[8, 1152, 7, 7]" = convolution_backward_10[0]
    getitem_135: "f32[192, 1152, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_7: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
    where_7: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, getitem_134);  le_7 = getitem_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_24: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_96: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_342);  convolution_40 = unsqueeze_342 = None
    mul_463: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_96)
    sum_25: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
    mul_464: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    unsqueeze_343: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_464, 0);  mul_464 = None
    unsqueeze_344: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_465: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    mul_466: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_467: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
    unsqueeze_346: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
    unsqueeze_347: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_468: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_349: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_350: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    mul_469: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_348);  sub_96 = unsqueeze_348 = None
    sub_98: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_469);  where_7 = mul_469 = None
    sub_99: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_345);  sub_98 = unsqueeze_345 = None
    mul_470: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_351);  sub_99 = unsqueeze_351 = None
    mul_471: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_121);  sum_25 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_470, relu_26, primals_145, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_470 = primals_145 = None
    getitem_137: "f32[8, 1152, 7, 7]" = convolution_backward_11[0]
    getitem_138: "f32[1152, 1, 5, 5]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_8: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
    where_8: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_8, full_default, getitem_137);  le_8 = getitem_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_26: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_100: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_354);  convolution_39 = unsqueeze_354 = None
    mul_472: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_100)
    sum_27: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 2, 3]);  mul_472 = None
    mul_473: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    unsqueeze_355: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_473, 0);  mul_473 = None
    unsqueeze_356: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_474: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    mul_475: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_476: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_474, mul_475);  mul_474 = mul_475 = None
    unsqueeze_358: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
    unsqueeze_359: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_477: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_361: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_362: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    mul_478: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_360);  sub_100 = unsqueeze_360 = None
    sub_102: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_478);  where_8 = mul_478 = None
    sub_103: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_357);  sub_102 = unsqueeze_357 = None
    mul_479: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_363);  sub_103 = unsqueeze_363 = None
    mul_480: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_118);  sum_27 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_479, add_201, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_479 = add_201 = primals_144 = None
    getitem_140: "f32[8, 192, 7, 7]" = convolution_backward_12[0]
    getitem_141: "f32[1152, 192, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_272: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_271, getitem_140);  add_271 = getitem_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_28: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_272, [0, 2, 3])
    sub_104: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_366);  convolution_38 = unsqueeze_366 = None
    mul_481: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_272, sub_104)
    sum_29: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 2, 3]);  mul_481 = None
    mul_482: "f32[192]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_367: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_482, 0);  mul_482 = None
    unsqueeze_368: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_483: "f32[192]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_484: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_485: "f32[192]" = torch.ops.aten.mul.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    unsqueeze_370: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
    unsqueeze_371: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_486: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_373: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
    unsqueeze_374: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    mul_487: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_372);  sub_104 = unsqueeze_372 = None
    sub_106: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_272, mul_487);  add_272 = mul_487 = None
    sub_107: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_369);  sub_106 = unsqueeze_369 = None
    mul_488: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_375);  sub_107 = unsqueeze_375 = None
    mul_489: "f32[192]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_115);  sum_29 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_488, relu_25, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_488 = primals_143 = None
    getitem_143: "f32[8, 576, 7, 7]" = convolution_backward_13[0]
    getitem_144: "f32[192, 576, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_9: "b8[8, 576, 7, 7]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    where_9: "f32[8, 576, 7, 7]" = torch.ops.aten.where.self(le_9, full_default, getitem_143);  le_9 = getitem_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_30: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_108: "f32[8, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_378);  convolution_37 = unsqueeze_378 = None
    mul_490: "f32[8, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_108)
    sum_31: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2, 3]);  mul_490 = None
    mul_491: "f32[576]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_379: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_491, 0);  mul_491 = None
    unsqueeze_380: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_492: "f32[576]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_493: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_494: "f32[576]" = torch.ops.aten.mul.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    unsqueeze_382: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_383: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_495: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_385: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_386: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_496: "f32[8, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_384);  sub_108 = unsqueeze_384 = None
    sub_110: "f32[8, 576, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_496);  where_9 = mul_496 = None
    sub_111: "f32[8, 576, 7, 7]" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_381);  sub_110 = unsqueeze_381 = None
    mul_497: "f32[8, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_387);  sub_111 = unsqueeze_387 = None
    mul_498: "f32[576]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_112);  sum_31 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_497, relu_24, primals_142, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_497 = primals_142 = None
    getitem_146: "f32[8, 576, 14, 14]" = convolution_backward_14[0]
    getitem_147: "f32[576, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_10: "b8[8, 576, 14, 14]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
    where_10: "f32[8, 576, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, getitem_146);  le_10 = getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_32: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_112: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_390);  convolution_36 = unsqueeze_390 = None
    mul_499: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_112)
    sum_33: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 2, 3]);  mul_499 = None
    mul_500: "f32[576]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_391: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
    unsqueeze_392: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_501: "f32[576]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_502: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_503: "f32[576]" = torch.ops.aten.mul.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
    unsqueeze_394: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    unsqueeze_395: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_504: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_397: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_398: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    mul_505: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_396);  sub_112 = unsqueeze_396 = None
    sub_114: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_505);  where_10 = mul_505 = None
    sub_115: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_393);  sub_114 = unsqueeze_393 = None
    mul_506: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_399);  sub_115 = unsqueeze_399 = None
    mul_507: "f32[576]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_109);  sum_33 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_506, add_186, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_506 = add_186 = primals_141 = None
    getitem_149: "f32[8, 96, 14, 14]" = convolution_backward_15[0]
    getitem_150: "f32[576, 96, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_34: "f32[96]" = torch.ops.aten.sum.dim_IntList(getitem_149, [0, 2, 3])
    sub_116: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_402);  convolution_35 = unsqueeze_402 = None
    mul_508: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_149, sub_116)
    sum_35: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_508, [0, 2, 3]);  mul_508 = None
    mul_509: "f32[96]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    unsqueeze_403: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_509, 0);  mul_509 = None
    unsqueeze_404: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_510: "f32[96]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    mul_511: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_512: "f32[96]" = torch.ops.aten.mul.Tensor(mul_510, mul_511);  mul_510 = mul_511 = None
    unsqueeze_406: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    unsqueeze_407: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_513: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_409: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
    unsqueeze_410: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    mul_514: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_408);  sub_116 = unsqueeze_408 = None
    sub_118: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_149, mul_514);  mul_514 = None
    sub_119: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_405);  sub_118 = unsqueeze_405 = None
    mul_515: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_411);  sub_119 = unsqueeze_411 = None
    mul_516: "f32[96]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_106);  sum_35 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_515, relu_23, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_515 = primals_140 = None
    getitem_152: "f32[8, 576, 14, 14]" = convolution_backward_16[0]
    getitem_153: "f32[96, 576, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_11: "b8[8, 576, 14, 14]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
    where_11: "f32[8, 576, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, getitem_152);  le_11 = getitem_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_36: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_120: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_414);  convolution_34 = unsqueeze_414 = None
    mul_517: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_120)
    sum_37: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 2, 3]);  mul_517 = None
    mul_518: "f32[576]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    unsqueeze_415: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_518, 0);  mul_518 = None
    unsqueeze_416: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_519: "f32[576]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    mul_520: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_521: "f32[576]" = torch.ops.aten.mul.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
    unsqueeze_418: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_419: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_522: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_421: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
    unsqueeze_422: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    mul_523: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_420);  sub_120 = unsqueeze_420 = None
    sub_122: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_523);  where_11 = mul_523 = None
    sub_123: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_417);  sub_122 = unsqueeze_417 = None
    mul_524: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_423);  sub_123 = unsqueeze_423 = None
    mul_525: "f32[576]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_103);  sum_37 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_524, relu_22, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_524 = primals_139 = None
    getitem_155: "f32[8, 576, 14, 14]" = convolution_backward_17[0]
    getitem_156: "f32[576, 1, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_12: "b8[8, 576, 14, 14]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
    where_12: "f32[8, 576, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, getitem_155);  le_12 = getitem_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_38: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_124: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_426);  convolution_33 = unsqueeze_426 = None
    mul_526: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_124)
    sum_39: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_526, [0, 2, 3]);  mul_526 = None
    mul_527: "f32[576]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    unsqueeze_427: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_527, 0);  mul_527 = None
    unsqueeze_428: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_528: "f32[576]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_529: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_530: "f32[576]" = torch.ops.aten.mul.Tensor(mul_528, mul_529);  mul_528 = mul_529 = None
    unsqueeze_430: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
    unsqueeze_431: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_531: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_433: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
    unsqueeze_434: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    mul_532: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_432);  sub_124 = unsqueeze_432 = None
    sub_126: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_532);  where_12 = mul_532 = None
    sub_127: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_429);  sub_126 = unsqueeze_429 = None
    mul_533: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_435);  sub_127 = unsqueeze_435 = None
    mul_534: "f32[576]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_100);  sum_39 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_533, add_170, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_533 = add_170 = primals_138 = None
    getitem_158: "f32[8, 96, 14, 14]" = convolution_backward_18[0]
    getitem_159: "f32[576, 96, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_273: "f32[8, 96, 14, 14]" = torch.ops.aten.add.Tensor(getitem_149, getitem_158);  getitem_149 = getitem_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_40: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_273, [0, 2, 3])
    sub_128: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_438);  convolution_32 = unsqueeze_438 = None
    mul_535: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_273, sub_128)
    sum_41: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
    mul_536: "f32[96]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    unsqueeze_439: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_440: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_537: "f32[96]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_538: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_539: "f32[96]" = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
    unsqueeze_442: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_443: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_540: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_445: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_446: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    mul_541: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_444);  sub_128 = unsqueeze_444 = None
    sub_130: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(add_273, mul_541);  add_273 = mul_541 = None
    sub_131: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_441);  sub_130 = unsqueeze_441 = None
    mul_542: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_447);  sub_131 = unsqueeze_447 = None
    mul_543: "f32[96]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_97);  sum_41 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_542, relu_21, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_542 = primals_137 = None
    getitem_161: "f32[8, 480, 14, 14]" = convolution_backward_19[0]
    getitem_162: "f32[96, 480, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_13: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    where_13: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, getitem_161);  le_13 = getitem_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_132: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_450);  convolution_31 = unsqueeze_450 = None
    mul_544: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_132)
    sum_43: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 2, 3]);  mul_544 = None
    mul_545: "f32[480]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_451: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_545, 0);  mul_545 = None
    unsqueeze_452: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_546: "f32[480]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_547: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_548: "f32[480]" = torch.ops.aten.mul.Tensor(mul_546, mul_547);  mul_546 = mul_547 = None
    unsqueeze_454: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_455: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_549: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_457: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
    unsqueeze_458: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    mul_550: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_456);  sub_132 = unsqueeze_456 = None
    sub_134: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_550);  where_13 = mul_550 = None
    sub_135: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_453);  sub_134 = unsqueeze_453 = None
    mul_551: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_459);  sub_135 = unsqueeze_459 = None
    mul_552: "f32[480]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_94);  sum_43 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_551, relu_20, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_551 = primals_136 = None
    getitem_164: "f32[8, 480, 14, 14]" = convolution_backward_20[0]
    getitem_165: "f32[480, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_14: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    where_14: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, getitem_164);  le_14 = getitem_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_44: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_136: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_462);  convolution_30 = unsqueeze_462 = None
    mul_553: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_136)
    sum_45: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 2, 3]);  mul_553 = None
    mul_554: "f32[480]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    unsqueeze_463: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
    unsqueeze_464: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_555: "f32[480]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_556: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_557: "f32[480]" = torch.ops.aten.mul.Tensor(mul_555, mul_556);  mul_555 = mul_556 = None
    unsqueeze_466: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
    unsqueeze_467: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_558: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_469: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
    unsqueeze_470: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    mul_559: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_468);  sub_136 = unsqueeze_468 = None
    sub_138: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_559);  where_14 = mul_559 = None
    sub_139: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_465);  sub_138 = unsqueeze_465 = None
    mul_560: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_471);  sub_139 = unsqueeze_471 = None
    mul_561: "f32[480]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_91);  sum_45 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_560, add_155, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_560 = add_155 = primals_135 = None
    getitem_167: "f32[8, 80, 14, 14]" = convolution_backward_21[0]
    getitem_168: "f32[480, 80, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_46: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_167, [0, 2, 3])
    sub_140: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_474);  convolution_29 = unsqueeze_474 = None
    mul_562: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_167, sub_140)
    sum_47: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 2, 3]);  mul_562 = None
    mul_563: "f32[80]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_475: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_563, 0);  mul_563 = None
    unsqueeze_476: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_564: "f32[80]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_565: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_566: "f32[80]" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_478: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    unsqueeze_479: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_567: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_481: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_482: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    mul_568: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_480);  sub_140 = unsqueeze_480 = None
    sub_142: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_167, mul_568);  mul_568 = None
    sub_143: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_477);  sub_142 = unsqueeze_477 = None
    mul_569: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_483);  sub_143 = unsqueeze_483 = None
    mul_570: "f32[80]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_88);  sum_47 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_569, relu_19, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_569 = primals_134 = None
    getitem_170: "f32[8, 480, 14, 14]" = convolution_backward_22[0]
    getitem_171: "f32[80, 480, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_15: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    where_15: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, getitem_170);  le_15 = getitem_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_48: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_144: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_486);  convolution_28 = unsqueeze_486 = None
    mul_571: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_144)
    sum_49: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 2, 3]);  mul_571 = None
    mul_572: "f32[480]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_487: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_572, 0);  mul_572 = None
    unsqueeze_488: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_573: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_574: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_575: "f32[480]" = torch.ops.aten.mul.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
    unsqueeze_490: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
    unsqueeze_491: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_576: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_493: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_494: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    mul_577: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_492);  sub_144 = unsqueeze_492 = None
    sub_146: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_577);  where_15 = mul_577 = None
    sub_147: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_489);  sub_146 = unsqueeze_489 = None
    mul_578: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_495);  sub_147 = unsqueeze_495 = None
    mul_579: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_85);  sum_49 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_578, relu_18, primals_133, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_578 = primals_133 = None
    getitem_173: "f32[8, 480, 14, 14]" = convolution_backward_23[0]
    getitem_174: "f32[480, 1, 5, 5]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_16: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    where_16: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, getitem_173);  le_16 = getitem_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_50: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_148: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_498);  convolution_27 = unsqueeze_498 = None
    mul_580: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_148)
    sum_51: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_580, [0, 2, 3]);  mul_580 = None
    mul_581: "f32[480]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_499: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
    unsqueeze_500: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    unsqueeze_501: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
    mul_582: "f32[480]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_583: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_584: "f32[480]" = torch.ops.aten.mul.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    unsqueeze_502: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
    unsqueeze_503: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    mul_585: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_505: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_506: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    mul_586: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_504);  sub_148 = unsqueeze_504 = None
    sub_150: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_586);  where_16 = mul_586 = None
    sub_151: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_501);  sub_150 = unsqueeze_501 = None
    mul_587: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_507);  sub_151 = unsqueeze_507 = None
    mul_588: "f32[480]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_82);  sum_51 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_587, add_139, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_587 = add_139 = primals_132 = None
    getitem_176: "f32[8, 80, 14, 14]" = convolution_backward_24[0]
    getitem_177: "f32[480, 80, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_274: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_167, getitem_176);  getitem_167 = getitem_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_52: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_274, [0, 2, 3])
    sub_152: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_510);  convolution_26 = unsqueeze_510 = None
    mul_589: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_274, sub_152)
    sum_53: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 2, 3]);  mul_589 = None
    mul_590: "f32[80]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    unsqueeze_511: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_590, 0);  mul_590 = None
    unsqueeze_512: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    unsqueeze_513: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
    mul_591: "f32[80]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    mul_592: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_593: "f32[80]" = torch.ops.aten.mul.Tensor(mul_591, mul_592);  mul_591 = mul_592 = None
    unsqueeze_514: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_593, 0);  mul_593 = None
    unsqueeze_515: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    mul_594: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_517: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
    unsqueeze_518: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    mul_595: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_516);  sub_152 = unsqueeze_516 = None
    sub_154: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_274, mul_595);  mul_595 = None
    sub_155: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_513);  sub_154 = unsqueeze_513 = None
    mul_596: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_519);  sub_155 = unsqueeze_519 = None
    mul_597: "f32[80]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_79);  sum_53 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_596, relu_17, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_596 = primals_131 = None
    getitem_179: "f32[8, 480, 14, 14]" = convolution_backward_25[0]
    getitem_180: "f32[80, 480, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_17: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    where_17: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, getitem_179);  le_17 = getitem_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_156: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_522);  convolution_25 = unsqueeze_522 = None
    mul_598: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_156)
    sum_55: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 2, 3]);  mul_598 = None
    mul_599: "f32[480]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_523: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_599, 0);  mul_599 = None
    unsqueeze_524: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_600: "f32[480]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_601: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_602: "f32[480]" = torch.ops.aten.mul.Tensor(mul_600, mul_601);  mul_600 = mul_601 = None
    unsqueeze_526: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
    unsqueeze_527: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    mul_603: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_529: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
    unsqueeze_530: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    mul_604: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_528);  sub_156 = unsqueeze_528 = None
    sub_158: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_604);  where_17 = mul_604 = None
    sub_159: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_525);  sub_158 = unsqueeze_525 = None
    mul_605: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_531);  sub_159 = unsqueeze_531 = None
    mul_606: "f32[480]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_76);  sum_55 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_605, relu_16, primals_130, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_605 = primals_130 = None
    getitem_182: "f32[8, 480, 14, 14]" = convolution_backward_26[0]
    getitem_183: "f32[480, 1, 5, 5]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_18: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    where_18: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, getitem_182);  le_18 = getitem_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_160: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_534);  convolution_24 = unsqueeze_534 = None
    mul_607: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_160)
    sum_57: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3]);  mul_607 = None
    mul_608: "f32[480]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_535: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    unsqueeze_536: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
    unsqueeze_537: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
    mul_609: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_610: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_611: "f32[480]" = torch.ops.aten.mul.Tensor(mul_609, mul_610);  mul_609 = mul_610 = None
    unsqueeze_538: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
    unsqueeze_539: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    mul_612: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_541: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_542: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    mul_613: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_540);  sub_160 = unsqueeze_540 = None
    sub_162: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_613);  where_18 = mul_613 = None
    sub_163: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_537);  sub_162 = unsqueeze_537 = None
    mul_614: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_543);  sub_163 = unsqueeze_543 = None
    mul_615: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_73);  sum_57 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_614, add_123, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_614 = add_123 = primals_129 = None
    getitem_185: "f32[8, 80, 14, 14]" = convolution_backward_27[0]
    getitem_186: "f32[480, 80, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_275: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_274, getitem_185);  add_274 = getitem_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_275, [0, 2, 3])
    sub_164: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_546);  convolution_23 = unsqueeze_546 = None
    mul_616: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_275, sub_164)
    sum_59: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_616, [0, 2, 3]);  mul_616 = None
    mul_617: "f32[80]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_547: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_617, 0);  mul_617 = None
    unsqueeze_548: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
    unsqueeze_549: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
    mul_618: "f32[80]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_619: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_620: "f32[80]" = torch.ops.aten.mul.Tensor(mul_618, mul_619);  mul_618 = mul_619 = None
    unsqueeze_550: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
    unsqueeze_551: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    mul_621: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_553: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_554: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    mul_622: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_552);  sub_164 = unsqueeze_552 = None
    sub_166: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_275, mul_622);  add_275 = mul_622 = None
    sub_167: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_549);  sub_166 = unsqueeze_549 = None
    mul_623: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_555);  sub_167 = unsqueeze_555 = None
    mul_624: "f32[80]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_70);  sum_59 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_623, relu_15, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_623 = primals_128 = None
    getitem_188: "f32[8, 240, 14, 14]" = convolution_backward_28[0]
    getitem_189: "f32[80, 240, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_19: "b8[8, 240, 14, 14]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_19: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, getitem_188);  le_19 = getitem_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_60: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_168: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_558);  convolution_22 = unsqueeze_558 = None
    mul_625: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_168)
    sum_61: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 2, 3]);  mul_625 = None
    mul_626: "f32[240]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_559: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
    unsqueeze_560: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_627: "f32[240]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_628: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_629: "f32[240]" = torch.ops.aten.mul.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
    unsqueeze_562: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
    unsqueeze_563: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_630: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_565: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    unsqueeze_566: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    mul_631: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_564);  sub_168 = unsqueeze_564 = None
    sub_170: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_631);  where_19 = mul_631 = None
    sub_171: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_561);  sub_170 = unsqueeze_561 = None
    mul_632: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_567);  sub_171 = unsqueeze_567 = None
    mul_633: "f32[240]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_67);  sum_61 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_632, relu_14, primals_127, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_632 = primals_127 = None
    getitem_191: "f32[8, 240, 28, 28]" = convolution_backward_29[0]
    getitem_192: "f32[240, 1, 5, 5]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_20: "b8[8, 240, 28, 28]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_20: "f32[8, 240, 28, 28]" = torch.ops.aten.where.self(le_20, full_default, getitem_191);  le_20 = getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_62: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_172: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_570);  convolution_21 = unsqueeze_570 = None
    mul_634: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(where_20, sub_172)
    sum_63: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 2, 3]);  mul_634 = None
    mul_635: "f32[240]" = torch.ops.aten.mul.Tensor(sum_62, 0.00015943877551020407)
    unsqueeze_571: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_572: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
    unsqueeze_573: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
    mul_636: "f32[240]" = torch.ops.aten.mul.Tensor(sum_63, 0.00015943877551020407)
    mul_637: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_638: "f32[240]" = torch.ops.aten.mul.Tensor(mul_636, mul_637);  mul_636 = mul_637 = None
    unsqueeze_574: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_575: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    mul_639: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_577: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    unsqueeze_578: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    mul_640: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_576);  sub_172 = unsqueeze_576 = None
    sub_174: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(where_20, mul_640);  where_20 = mul_640 = None
    sub_175: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_573);  sub_174 = unsqueeze_573 = None
    mul_641: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_579);  sub_175 = unsqueeze_579 = None
    mul_642: "f32[240]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_64);  sum_63 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_641, add_108, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_641 = add_108 = primals_126 = None
    getitem_194: "f32[8, 40, 28, 28]" = convolution_backward_30[0]
    getitem_195: "f32[240, 40, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_64: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_194, [0, 2, 3])
    sub_176: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_582);  convolution_20 = unsqueeze_582 = None
    mul_643: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_194, sub_176)
    sum_65: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 2, 3]);  mul_643 = None
    mul_644: "f32[40]" = torch.ops.aten.mul.Tensor(sum_64, 0.00015943877551020407)
    unsqueeze_583: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
    unsqueeze_584: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
    unsqueeze_585: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
    mul_645: "f32[40]" = torch.ops.aten.mul.Tensor(sum_65, 0.00015943877551020407)
    mul_646: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_647: "f32[40]" = torch.ops.aten.mul.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
    unsqueeze_586: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
    unsqueeze_587: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    mul_648: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_589: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_590: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    mul_649: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_588);  sub_176 = unsqueeze_588 = None
    sub_178: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_194, mul_649);  mul_649 = None
    sub_179: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_585);  sub_178 = unsqueeze_585 = None
    mul_650: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_591);  sub_179 = unsqueeze_591 = None
    mul_651: "f32[40]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_61);  sum_65 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_650, relu_13, primals_125, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_650 = primals_125 = None
    getitem_197: "f32[8, 120, 28, 28]" = convolution_backward_31[0]
    getitem_198: "f32[40, 120, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_21: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_21: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_21, full_default, getitem_197);  le_21 = getitem_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_66: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_180: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_594);  convolution_19 = unsqueeze_594 = None
    mul_652: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_21, sub_180)
    sum_67: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 2, 3]);  mul_652 = None
    mul_653: "f32[120]" = torch.ops.aten.mul.Tensor(sum_66, 0.00015943877551020407)
    unsqueeze_595: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
    unsqueeze_596: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_654: "f32[120]" = torch.ops.aten.mul.Tensor(sum_67, 0.00015943877551020407)
    mul_655: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_656: "f32[120]" = torch.ops.aten.mul.Tensor(mul_654, mul_655);  mul_654 = mul_655 = None
    unsqueeze_598: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
    unsqueeze_599: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    mul_657: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_601: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
    unsqueeze_602: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    mul_658: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_600);  sub_180 = unsqueeze_600 = None
    sub_182: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_21, mul_658);  where_21 = mul_658 = None
    sub_183: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_597);  sub_182 = unsqueeze_597 = None
    mul_659: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_603);  sub_183 = unsqueeze_603 = None
    mul_660: "f32[120]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_58);  sum_67 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_659, relu_12, primals_124, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_659 = primals_124 = None
    getitem_200: "f32[8, 120, 28, 28]" = convolution_backward_32[0]
    getitem_201: "f32[120, 1, 5, 5]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_22: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_22: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_22, full_default, getitem_200);  le_22 = getitem_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_68: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_184: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_606);  convolution_18 = unsqueeze_606 = None
    mul_661: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_22, sub_184)
    sum_69: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2, 3]);  mul_661 = None
    mul_662: "f32[120]" = torch.ops.aten.mul.Tensor(sum_68, 0.00015943877551020407)
    unsqueeze_607: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
    unsqueeze_608: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
    unsqueeze_609: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
    mul_663: "f32[120]" = torch.ops.aten.mul.Tensor(sum_69, 0.00015943877551020407)
    mul_664: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_665: "f32[120]" = torch.ops.aten.mul.Tensor(mul_663, mul_664);  mul_663 = mul_664 = None
    unsqueeze_610: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
    unsqueeze_611: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    mul_666: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_613: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
    unsqueeze_614: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    mul_667: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_612);  sub_184 = unsqueeze_612 = None
    sub_186: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_22, mul_667);  where_22 = mul_667 = None
    sub_187: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_609);  sub_186 = unsqueeze_609 = None
    mul_668: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_615);  sub_187 = unsqueeze_615 = None
    mul_669: "f32[120]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_55);  sum_69 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_668, add_92, primals_123, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_668 = add_92 = primals_123 = None
    getitem_203: "f32[8, 40, 28, 28]" = convolution_backward_33[0]
    getitem_204: "f32[120, 40, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_276: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_194, getitem_203);  getitem_194 = getitem_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_70: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_276, [0, 2, 3])
    sub_188: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_618);  convolution_17 = unsqueeze_618 = None
    mul_670: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_276, sub_188)
    sum_71: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 2, 3]);  mul_670 = None
    mul_671: "f32[40]" = torch.ops.aten.mul.Tensor(sum_70, 0.00015943877551020407)
    unsqueeze_619: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_671, 0);  mul_671 = None
    unsqueeze_620: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    mul_672: "f32[40]" = torch.ops.aten.mul.Tensor(sum_71, 0.00015943877551020407)
    mul_673: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_674: "f32[40]" = torch.ops.aten.mul.Tensor(mul_672, mul_673);  mul_672 = mul_673 = None
    unsqueeze_622: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
    unsqueeze_623: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_675: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_625: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
    unsqueeze_626: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    mul_676: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_624);  sub_188 = unsqueeze_624 = None
    sub_190: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_276, mul_676);  mul_676 = None
    sub_191: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_621);  sub_190 = unsqueeze_621 = None
    mul_677: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_627);  sub_191 = unsqueeze_627 = None
    mul_678: "f32[40]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_52);  sum_71 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_677, relu_11, primals_122, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_677 = primals_122 = None
    getitem_206: "f32[8, 120, 28, 28]" = convolution_backward_34[0]
    getitem_207: "f32[40, 120, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_23: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    where_23: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_23, full_default, getitem_206);  le_23 = getitem_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_192: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_630);  convolution_16 = unsqueeze_630 = None
    mul_679: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_23, sub_192)
    sum_73: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 2, 3]);  mul_679 = None
    mul_680: "f32[120]" = torch.ops.aten.mul.Tensor(sum_72, 0.00015943877551020407)
    unsqueeze_631: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_680, 0);  mul_680 = None
    unsqueeze_632: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_681: "f32[120]" = torch.ops.aten.mul.Tensor(sum_73, 0.00015943877551020407)
    mul_682: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_683: "f32[120]" = torch.ops.aten.mul.Tensor(mul_681, mul_682);  mul_681 = mul_682 = None
    unsqueeze_634: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
    unsqueeze_635: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    mul_684: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_637: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_638: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    mul_685: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_636);  sub_192 = unsqueeze_636 = None
    sub_194: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_23, mul_685);  where_23 = mul_685 = None
    sub_195: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_633);  sub_194 = unsqueeze_633 = None
    mul_686: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_639);  sub_195 = unsqueeze_639 = None
    mul_687: "f32[120]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_49);  sum_73 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_686, relu_10, primals_121, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_686 = primals_121 = None
    getitem_209: "f32[8, 120, 28, 28]" = convolution_backward_35[0]
    getitem_210: "f32[120, 1, 5, 5]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_24: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_24: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_24, full_default, getitem_209);  le_24 = getitem_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_196: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_642);  convolution_15 = unsqueeze_642 = None
    mul_688: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, sub_196)
    sum_75: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_688, [0, 2, 3]);  mul_688 = None
    mul_689: "f32[120]" = torch.ops.aten.mul.Tensor(sum_74, 0.00015943877551020407)
    unsqueeze_643: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
    unsqueeze_644: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
    unsqueeze_645: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
    mul_690: "f32[120]" = torch.ops.aten.mul.Tensor(sum_75, 0.00015943877551020407)
    mul_691: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_692: "f32[120]" = torch.ops.aten.mul.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
    unsqueeze_646: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_647: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    mul_693: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_649: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
    unsqueeze_650: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    mul_694: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_648);  sub_196 = unsqueeze_648 = None
    sub_198: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_24, mul_694);  where_24 = mul_694 = None
    sub_199: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_645);  sub_198 = unsqueeze_645 = None
    mul_695: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_651);  sub_199 = unsqueeze_651 = None
    mul_696: "f32[120]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_46);  sum_75 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_695, add_76, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_695 = add_76 = primals_120 = None
    getitem_212: "f32[8, 40, 28, 28]" = convolution_backward_36[0]
    getitem_213: "f32[120, 40, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_277: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_276, getitem_212);  add_276 = getitem_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_76: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_277, [0, 2, 3])
    sub_200: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_654);  convolution_14 = unsqueeze_654 = None
    mul_697: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_277, sub_200)
    sum_77: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3]);  mul_697 = None
    mul_698: "f32[40]" = torch.ops.aten.mul.Tensor(sum_76, 0.00015943877551020407)
    unsqueeze_655: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_698, 0);  mul_698 = None
    unsqueeze_656: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
    unsqueeze_657: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
    mul_699: "f32[40]" = torch.ops.aten.mul.Tensor(sum_77, 0.00015943877551020407)
    mul_700: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_701: "f32[40]" = torch.ops.aten.mul.Tensor(mul_699, mul_700);  mul_699 = mul_700 = None
    unsqueeze_658: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
    unsqueeze_659: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    mul_702: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_661: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
    unsqueeze_662: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    mul_703: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_660);  sub_200 = unsqueeze_660 = None
    sub_202: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_277, mul_703);  add_277 = mul_703 = None
    sub_203: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_657);  sub_202 = unsqueeze_657 = None
    mul_704: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_663);  sub_203 = unsqueeze_663 = None
    mul_705: "f32[40]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_43);  sum_77 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_704, relu_9, primals_119, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_704 = primals_119 = None
    getitem_215: "f32[8, 72, 28, 28]" = convolution_backward_37[0]
    getitem_216: "f32[40, 72, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_25: "b8[8, 72, 28, 28]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_25: "f32[8, 72, 28, 28]" = torch.ops.aten.where.self(le_25, full_default, getitem_215);  le_25 = getitem_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_78: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_204: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_666);  convolution_13 = unsqueeze_666 = None
    mul_706: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, sub_204)
    sum_79: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 2, 3]);  mul_706 = None
    mul_707: "f32[72]" = torch.ops.aten.mul.Tensor(sum_78, 0.00015943877551020407)
    unsqueeze_667: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
    unsqueeze_668: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_708: "f32[72]" = torch.ops.aten.mul.Tensor(sum_79, 0.00015943877551020407)
    mul_709: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_710: "f32[72]" = torch.ops.aten.mul.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
    unsqueeze_670: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
    unsqueeze_671: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    mul_711: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_673: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    unsqueeze_674: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    mul_712: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_672);  sub_204 = unsqueeze_672 = None
    sub_206: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_25, mul_712);  where_25 = mul_712 = None
    sub_207: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_669);  sub_206 = unsqueeze_669 = None
    mul_713: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_675);  sub_207 = unsqueeze_675 = None
    mul_714: "f32[72]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_40);  sum_79 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_713, relu_8, primals_118, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_713 = primals_118 = None
    getitem_218: "f32[8, 72, 56, 56]" = convolution_backward_38[0]
    getitem_219: "f32[72, 1, 5, 5]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_26: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_26: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_26, full_default, getitem_218);  le_26 = getitem_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_80: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_208: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_678);  convolution_12 = unsqueeze_678 = None
    mul_715: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_26, sub_208)
    sum_81: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_715, [0, 2, 3]);  mul_715 = None
    mul_716: "f32[72]" = torch.ops.aten.mul.Tensor(sum_80, 3.985969387755102e-05)
    unsqueeze_679: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
    unsqueeze_680: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
    unsqueeze_681: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
    mul_717: "f32[72]" = torch.ops.aten.mul.Tensor(sum_81, 3.985969387755102e-05)
    mul_718: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_719: "f32[72]" = torch.ops.aten.mul.Tensor(mul_717, mul_718);  mul_717 = mul_718 = None
    unsqueeze_682: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
    unsqueeze_683: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    mul_720: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_685: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    unsqueeze_686: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    mul_721: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_684);  sub_208 = unsqueeze_684 = None
    sub_210: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_26, mul_721);  where_26 = mul_721 = None
    sub_211: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_681);  sub_210 = unsqueeze_681 = None
    mul_722: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_687);  sub_211 = unsqueeze_687 = None
    mul_723: "f32[72]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_37);  sum_81 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_722, add_61, primals_117, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_722 = add_61 = primals_117 = None
    getitem_221: "f32[8, 24, 56, 56]" = convolution_backward_39[0]
    getitem_222: "f32[72, 24, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_82: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_221, [0, 2, 3])
    sub_212: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_690);  convolution_11 = unsqueeze_690 = None
    mul_724: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_221, sub_212)
    sum_83: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_724, [0, 2, 3]);  mul_724 = None
    mul_725: "f32[24]" = torch.ops.aten.mul.Tensor(sum_82, 3.985969387755102e-05)
    unsqueeze_691: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
    unsqueeze_692: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
    unsqueeze_693: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
    mul_726: "f32[24]" = torch.ops.aten.mul.Tensor(sum_83, 3.985969387755102e-05)
    mul_727: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_728: "f32[24]" = torch.ops.aten.mul.Tensor(mul_726, mul_727);  mul_726 = mul_727 = None
    unsqueeze_694: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
    unsqueeze_695: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    mul_729: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_697: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
    unsqueeze_698: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    mul_730: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_696);  sub_212 = unsqueeze_696 = None
    sub_214: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_221, mul_730);  mul_730 = None
    sub_215: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_693);  sub_214 = unsqueeze_693 = None
    mul_731: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_699);  sub_215 = unsqueeze_699 = None
    mul_732: "f32[24]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_34);  sum_83 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_731, relu_7, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_731 = primals_116 = None
    getitem_224: "f32[8, 72, 56, 56]" = convolution_backward_40[0]
    getitem_225: "f32[24, 72, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_27: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_27: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_27, full_default, getitem_224);  le_27 = getitem_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_84: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_216: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_702);  convolution_10 = unsqueeze_702 = None
    mul_733: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_27, sub_216)
    sum_85: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
    mul_734: "f32[72]" = torch.ops.aten.mul.Tensor(sum_84, 3.985969387755102e-05)
    unsqueeze_703: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
    unsqueeze_704: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_735: "f32[72]" = torch.ops.aten.mul.Tensor(sum_85, 3.985969387755102e-05)
    mul_736: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_737: "f32[72]" = torch.ops.aten.mul.Tensor(mul_735, mul_736);  mul_735 = mul_736 = None
    unsqueeze_706: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
    unsqueeze_707: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    mul_738: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_709: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_710: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    mul_739: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_708);  sub_216 = unsqueeze_708 = None
    sub_218: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_27, mul_739);  where_27 = mul_739 = None
    sub_219: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_705);  sub_218 = unsqueeze_705 = None
    mul_740: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_711);  sub_219 = unsqueeze_711 = None
    mul_741: "f32[72]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_31);  sum_85 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_740, relu_6, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_740 = primals_115 = None
    getitem_227: "f32[8, 72, 56, 56]" = convolution_backward_41[0]
    getitem_228: "f32[72, 1, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_28: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_28: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_28, full_default, getitem_227);  le_28 = getitem_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_86: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_220: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_714);  convolution_9 = unsqueeze_714 = None
    mul_742: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_28, sub_220)
    sum_87: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_742, [0, 2, 3]);  mul_742 = None
    mul_743: "f32[72]" = torch.ops.aten.mul.Tensor(sum_86, 3.985969387755102e-05)
    unsqueeze_715: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_716: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_744: "f32[72]" = torch.ops.aten.mul.Tensor(sum_87, 3.985969387755102e-05)
    mul_745: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_746: "f32[72]" = torch.ops.aten.mul.Tensor(mul_744, mul_745);  mul_744 = mul_745 = None
    unsqueeze_718: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_719: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    mul_747: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_721: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
    unsqueeze_722: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    mul_748: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_720);  sub_220 = unsqueeze_720 = None
    sub_222: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_28, mul_748);  where_28 = mul_748 = None
    sub_223: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_717);  sub_222 = unsqueeze_717 = None
    mul_749: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_723);  sub_223 = unsqueeze_723 = None
    mul_750: "f32[72]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_28);  sum_87 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_749, add_45, primals_114, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_749 = add_45 = primals_114 = None
    getitem_230: "f32[8, 24, 56, 56]" = convolution_backward_42[0]
    getitem_231: "f32[72, 24, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_278: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_221, getitem_230);  getitem_221 = getitem_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_88: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 2, 3])
    sub_224: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_726);  convolution_8 = unsqueeze_726 = None
    mul_751: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_278, sub_224)
    sum_89: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 2, 3]);  mul_751 = None
    mul_752: "f32[24]" = torch.ops.aten.mul.Tensor(sum_88, 3.985969387755102e-05)
    unsqueeze_727: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
    unsqueeze_728: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_753: "f32[24]" = torch.ops.aten.mul.Tensor(sum_89, 3.985969387755102e-05)
    mul_754: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_755: "f32[24]" = torch.ops.aten.mul.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
    unsqueeze_730: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_731: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    mul_756: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_733: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_734: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    mul_757: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_732);  sub_224 = unsqueeze_732 = None
    sub_226: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_278, mul_757);  mul_757 = None
    sub_227: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_729);  sub_226 = unsqueeze_729 = None
    mul_758: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_735);  sub_227 = unsqueeze_735 = None
    mul_759: "f32[24]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_25);  sum_89 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_758, relu_5, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_758 = primals_113 = None
    getitem_233: "f32[8, 72, 56, 56]" = convolution_backward_43[0]
    getitem_234: "f32[24, 72, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_29: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_29: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_29, full_default, getitem_233);  le_29 = getitem_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_90: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_228: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_738);  convolution_7 = unsqueeze_738 = None
    mul_760: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_29, sub_228)
    sum_91: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_760, [0, 2, 3]);  mul_760 = None
    mul_761: "f32[72]" = torch.ops.aten.mul.Tensor(sum_90, 3.985969387755102e-05)
    unsqueeze_739: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    unsqueeze_740: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
    unsqueeze_741: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
    mul_762: "f32[72]" = torch.ops.aten.mul.Tensor(sum_91, 3.985969387755102e-05)
    mul_763: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_764: "f32[72]" = torch.ops.aten.mul.Tensor(mul_762, mul_763);  mul_762 = mul_763 = None
    unsqueeze_742: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_743: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
    unsqueeze_744: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
    mul_765: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_745: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
    unsqueeze_746: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    mul_766: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_744);  sub_228 = unsqueeze_744 = None
    sub_230: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_29, mul_766);  where_29 = mul_766 = None
    sub_231: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_741);  sub_230 = unsqueeze_741 = None
    mul_767: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_747);  sub_231 = unsqueeze_747 = None
    mul_768: "f32[72]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_22);  sum_91 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_767, relu_4, primals_112, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_767 = primals_112 = None
    getitem_236: "f32[8, 72, 56, 56]" = convolution_backward_44[0]
    getitem_237: "f32[72, 1, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_30: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_30: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_30, full_default, getitem_236);  le_30 = getitem_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_92: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_232: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_750);  convolution_6 = unsqueeze_750 = None
    mul_769: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_30, sub_232)
    sum_93: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3]);  mul_769 = None
    mul_770: "f32[72]" = torch.ops.aten.mul.Tensor(sum_92, 3.985969387755102e-05)
    unsqueeze_751: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
    unsqueeze_752: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
    unsqueeze_753: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
    mul_771: "f32[72]" = torch.ops.aten.mul.Tensor(sum_93, 3.985969387755102e-05)
    mul_772: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_773: "f32[72]" = torch.ops.aten.mul.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
    unsqueeze_754: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_755: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
    unsqueeze_756: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
    mul_774: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_757: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_758: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    mul_775: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_756);  sub_232 = unsqueeze_756 = None
    sub_234: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_30, mul_775);  where_30 = mul_775 = None
    sub_235: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_753);  sub_234 = unsqueeze_753 = None
    mul_776: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_759);  sub_235 = unsqueeze_759 = None
    mul_777: "f32[72]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_19);  sum_93 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_776, add_29, primals_111, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_776 = add_29 = primals_111 = None
    getitem_239: "f32[8, 24, 56, 56]" = convolution_backward_45[0]
    getitem_240: "f32[72, 24, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_279: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_278, getitem_239);  add_278 = getitem_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_94: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_279, [0, 2, 3])
    sub_236: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_762);  convolution_5 = unsqueeze_762 = None
    mul_778: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_279, sub_236)
    sum_95: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 2, 3]);  mul_778 = None
    mul_779: "f32[24]" = torch.ops.aten.mul.Tensor(sum_94, 3.985969387755102e-05)
    unsqueeze_763: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_779, 0);  mul_779 = None
    unsqueeze_764: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
    unsqueeze_765: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
    mul_780: "f32[24]" = torch.ops.aten.mul.Tensor(sum_95, 3.985969387755102e-05)
    mul_781: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_782: "f32[24]" = torch.ops.aten.mul.Tensor(mul_780, mul_781);  mul_780 = mul_781 = None
    unsqueeze_766: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_767: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
    unsqueeze_768: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
    mul_783: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_769: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_770: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    mul_784: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_768);  sub_236 = unsqueeze_768 = None
    sub_238: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_279, mul_784);  add_279 = mul_784 = None
    sub_239: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_765);  sub_238 = unsqueeze_765 = None
    mul_785: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_771);  sub_239 = unsqueeze_771 = None
    mul_786: "f32[24]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_16);  sum_95 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_785, relu_3, primals_110, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_785 = primals_110 = None
    getitem_242: "f32[8, 48, 56, 56]" = convolution_backward_46[0]
    getitem_243: "f32[24, 48, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_31: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_31: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_31, full_default, getitem_242);  le_31 = getitem_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_96: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_240: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_774);  convolution_4 = unsqueeze_774 = None
    mul_787: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_31, sub_240)
    sum_97: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 2, 3]);  mul_787 = None
    mul_788: "f32[48]" = torch.ops.aten.mul.Tensor(sum_96, 3.985969387755102e-05)
    unsqueeze_775: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_788, 0);  mul_788 = None
    unsqueeze_776: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
    unsqueeze_777: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
    mul_789: "f32[48]" = torch.ops.aten.mul.Tensor(sum_97, 3.985969387755102e-05)
    mul_790: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_791: "f32[48]" = torch.ops.aten.mul.Tensor(mul_789, mul_790);  mul_789 = mul_790 = None
    unsqueeze_778: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
    unsqueeze_779: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
    unsqueeze_780: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
    mul_792: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_781: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
    unsqueeze_782: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    mul_793: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_780);  sub_240 = unsqueeze_780 = None
    sub_242: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_31, mul_793);  where_31 = mul_793 = None
    sub_243: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_777);  sub_242 = unsqueeze_777 = None
    mul_794: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_783);  sub_243 = unsqueeze_783 = None
    mul_795: "f32[48]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_13);  sum_97 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_794, relu_2, primals_109, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_794 = primals_109 = None
    getitem_245: "f32[8, 48, 112, 112]" = convolution_backward_47[0]
    getitem_246: "f32[48, 1, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_32: "b8[8, 48, 112, 112]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_32: "f32[8, 48, 112, 112]" = torch.ops.aten.where.self(le_32, full_default, getitem_245);  le_32 = getitem_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_98: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_244: "f32[8, 48, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_786);  convolution_3 = unsqueeze_786 = None
    mul_796: "f32[8, 48, 112, 112]" = torch.ops.aten.mul.Tensor(where_32, sub_244)
    sum_99: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_796, [0, 2, 3]);  mul_796 = None
    mul_797: "f32[48]" = torch.ops.aten.mul.Tensor(sum_98, 9.964923469387754e-06)
    unsqueeze_787: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
    unsqueeze_788: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
    unsqueeze_789: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
    mul_798: "f32[48]" = torch.ops.aten.mul.Tensor(sum_99, 9.964923469387754e-06)
    mul_799: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_800: "f32[48]" = torch.ops.aten.mul.Tensor(mul_798, mul_799);  mul_798 = mul_799 = None
    unsqueeze_790: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_791: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
    unsqueeze_792: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
    mul_801: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_793: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_794: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    mul_802: "f32[8, 48, 112, 112]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_792);  sub_244 = unsqueeze_792 = None
    sub_246: "f32[8, 48, 112, 112]" = torch.ops.aten.sub.Tensor(where_32, mul_802);  where_32 = mul_802 = None
    sub_247: "f32[8, 48, 112, 112]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_789);  sub_246 = unsqueeze_789 = None
    mul_803: "f32[8, 48, 112, 112]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_795);  sub_247 = unsqueeze_795 = None
    mul_804: "f32[48]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_10);  sum_99 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_803, add_14, primals_108, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_803 = add_14 = primals_108 = None
    getitem_248: "f32[8, 16, 112, 112]" = convolution_backward_48[0]
    getitem_249: "f32[48, 16, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_100: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_248, [0, 2, 3])
    sub_248: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_798);  convolution_2 = unsqueeze_798 = None
    mul_805: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_248, sub_248)
    sum_101: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 2, 3]);  mul_805 = None
    mul_806: "f32[16]" = torch.ops.aten.mul.Tensor(sum_100, 9.964923469387754e-06)
    unsqueeze_799: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_806, 0);  mul_806 = None
    unsqueeze_800: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
    unsqueeze_801: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
    mul_807: "f32[16]" = torch.ops.aten.mul.Tensor(sum_101, 9.964923469387754e-06)
    mul_808: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_809: "f32[16]" = torch.ops.aten.mul.Tensor(mul_807, mul_808);  mul_807 = mul_808 = None
    unsqueeze_802: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
    unsqueeze_803: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
    unsqueeze_804: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
    mul_810: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_805: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
    unsqueeze_806: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    mul_811: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_804);  sub_248 = unsqueeze_804 = None
    sub_250: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(getitem_248, mul_811);  getitem_248 = mul_811 = None
    sub_251: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_801);  sub_250 = unsqueeze_801 = None
    mul_812: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_807);  sub_251 = unsqueeze_807 = None
    mul_813: "f32[16]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_7);  sum_101 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_812, relu_1, primals_107, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_812 = primals_107 = None
    getitem_251: "f32[8, 32, 112, 112]" = convolution_backward_49[0]
    getitem_252: "f32[16, 32, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_33: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_33: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_33, full_default, getitem_251);  le_33 = getitem_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_102: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_252: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_810);  convolution_1 = unsqueeze_810 = None
    mul_814: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_33, sub_252)
    sum_103: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3]);  mul_814 = None
    mul_815: "f32[32]" = torch.ops.aten.mul.Tensor(sum_102, 9.964923469387754e-06)
    unsqueeze_811: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
    unsqueeze_812: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
    unsqueeze_813: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
    mul_816: "f32[32]" = torch.ops.aten.mul.Tensor(sum_103, 9.964923469387754e-06)
    mul_817: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_818: "f32[32]" = torch.ops.aten.mul.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
    unsqueeze_814: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_815: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
    unsqueeze_816: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
    mul_819: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_817: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
    unsqueeze_818: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    mul_820: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_816);  sub_252 = unsqueeze_816 = None
    sub_254: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_33, mul_820);  where_33 = mul_820 = None
    sub_255: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_813);  sub_254 = unsqueeze_813 = None
    mul_821: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_819);  sub_255 = unsqueeze_819 = None
    mul_822: "f32[32]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_4);  sum_103 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_821, relu, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_821 = primals_106 = None
    getitem_254: "f32[8, 32, 112, 112]" = convolution_backward_50[0]
    getitem_255: "f32[32, 1, 3, 3]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_34: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_34: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_34, full_default, getitem_254);  le_34 = full_default = getitem_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_104: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_256: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_822);  convolution = unsqueeze_822 = None
    mul_823: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_34, sub_256)
    sum_105: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 2, 3]);  mul_823 = None
    mul_824: "f32[32]" = torch.ops.aten.mul.Tensor(sum_104, 9.964923469387754e-06)
    unsqueeze_823: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_824, 0);  mul_824 = None
    unsqueeze_824: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
    unsqueeze_825: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
    mul_825: "f32[32]" = torch.ops.aten.mul.Tensor(sum_105, 9.964923469387754e-06)
    mul_826: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_827: "f32[32]" = torch.ops.aten.mul.Tensor(mul_825, mul_826);  mul_825 = mul_826 = None
    unsqueeze_826: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
    unsqueeze_827: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
    unsqueeze_828: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
    mul_828: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_829: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
    unsqueeze_830: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    mul_829: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_828);  sub_256 = unsqueeze_828 = None
    sub_258: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_34, mul_829);  where_34 = mul_829 = None
    sub_259: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_825);  sub_258 = unsqueeze_825 = None
    mul_830: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_831);  sub_259 = unsqueeze_831 = None
    mul_831: "f32[32]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_1);  sum_105 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_830, primals_315, primals_105, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_830 = primals_315 = primals_105 = None
    getitem_258: "f32[32, 3, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    return [mul_831, sum_104, mul_822, sum_102, mul_813, sum_100, mul_804, sum_98, mul_795, sum_96, mul_786, sum_94, mul_777, sum_92, mul_768, sum_90, mul_759, sum_88, mul_750, sum_86, mul_741, sum_84, mul_732, sum_82, mul_723, sum_80, mul_714, sum_78, mul_705, sum_76, mul_696, sum_74, mul_687, sum_72, mul_678, sum_70, mul_669, sum_68, mul_660, sum_66, mul_651, sum_64, mul_642, sum_62, mul_633, sum_60, mul_624, sum_58, mul_615, sum_56, mul_606, sum_54, mul_597, sum_52, mul_588, sum_50, mul_579, sum_48, mul_570, sum_46, mul_561, sum_44, mul_552, sum_42, mul_543, sum_40, mul_534, sum_38, mul_525, sum_36, mul_516, sum_34, mul_507, sum_32, mul_498, sum_30, mul_489, sum_28, mul_480, sum_26, mul_471, sum_24, mul_462, sum_22, mul_453, sum_20, mul_444, sum_18, mul_435, sum_16, mul_426, sum_14, mul_417, sum_12, mul_408, sum_10, mul_399, sum_8, mul_390, sum_6, mul_381, sum_4, mul_372, sum_2, getitem_258, getitem_255, getitem_252, getitem_249, getitem_246, getitem_243, getitem_240, getitem_237, getitem_234, getitem_231, getitem_228, getitem_225, getitem_222, getitem_219, getitem_216, getitem_213, getitem_210, getitem_207, getitem_204, getitem_201, getitem_198, getitem_195, getitem_192, getitem_189, getitem_186, getitem_183, getitem_180, getitem_177, getitem_174, getitem_171, getitem_168, getitem_165, getitem_162, getitem_159, getitem_156, getitem_153, getitem_150, getitem_147, getitem_144, getitem_141, getitem_138, getitem_135, getitem_132, getitem_129, getitem_126, getitem_123, getitem_120, getitem_117, getitem_114, getitem_111, getitem_108, getitem_105, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    