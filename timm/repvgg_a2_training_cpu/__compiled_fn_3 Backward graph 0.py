from __future__ import annotations



def forward(self, primals_1: "f32[64]", primals_3: "f32[64]", primals_5: "f32[96]", primals_7: "f32[96]", primals_9: "f32[96]", primals_11: "f32[96]", primals_13: "f32[96]", primals_15: "f32[192]", primals_17: "f32[192]", primals_19: "f32[192]", primals_21: "f32[192]", primals_23: "f32[192]", primals_25: "f32[192]", primals_27: "f32[192]", primals_29: "f32[192]", primals_31: "f32[192]", primals_33: "f32[192]", primals_35: "f32[192]", primals_37: "f32[384]", primals_39: "f32[384]", primals_41: "f32[384]", primals_43: "f32[384]", primals_45: "f32[384]", primals_47: "f32[384]", primals_49: "f32[384]", primals_51: "f32[384]", primals_53: "f32[384]", primals_55: "f32[384]", primals_57: "f32[384]", primals_59: "f32[384]", primals_61: "f32[384]", primals_63: "f32[384]", primals_65: "f32[384]", primals_67: "f32[384]", primals_69: "f32[384]", primals_71: "f32[384]", primals_73: "f32[384]", primals_75: "f32[384]", primals_77: "f32[384]", primals_79: "f32[384]", primals_81: "f32[384]", primals_83: "f32[384]", primals_85: "f32[384]", primals_87: "f32[384]", primals_89: "f32[384]", primals_91: "f32[384]", primals_93: "f32[384]", primals_95: "f32[384]", primals_97: "f32[384]", primals_99: "f32[384]", primals_101: "f32[384]", primals_103: "f32[384]", primals_105: "f32[384]", primals_107: "f32[384]", primals_109: "f32[384]", primals_111: "f32[384]", primals_113: "f32[384]", primals_115: "f32[384]", primals_117: "f32[384]", primals_119: "f32[1408]", primals_121: "f32[1408]", primals_123: "f32[64, 3, 1, 1]", primals_124: "f32[64, 3, 3, 3]", primals_125: "f32[96, 64, 1, 1]", primals_126: "f32[96, 64, 3, 3]", primals_127: "f32[96, 96, 1, 1]", primals_128: "f32[96, 96, 3, 3]", primals_129: "f32[192, 96, 1, 1]", primals_130: "f32[192, 96, 3, 3]", primals_131: "f32[192, 192, 1, 1]", primals_132: "f32[192, 192, 3, 3]", primals_133: "f32[192, 192, 1, 1]", primals_134: "f32[192, 192, 3, 3]", primals_135: "f32[192, 192, 1, 1]", primals_136: "f32[192, 192, 3, 3]", primals_137: "f32[384, 192, 1, 1]", primals_138: "f32[384, 192, 3, 3]", primals_139: "f32[384, 384, 1, 1]", primals_140: "f32[384, 384, 3, 3]", primals_141: "f32[384, 384, 1, 1]", primals_142: "f32[384, 384, 3, 3]", primals_143: "f32[384, 384, 1, 1]", primals_144: "f32[384, 384, 3, 3]", primals_145: "f32[384, 384, 1, 1]", primals_146: "f32[384, 384, 3, 3]", primals_147: "f32[384, 384, 1, 1]", primals_148: "f32[384, 384, 3, 3]", primals_149: "f32[384, 384, 1, 1]", primals_150: "f32[384, 384, 3, 3]", primals_151: "f32[384, 384, 1, 1]", primals_152: "f32[384, 384, 3, 3]", primals_153: "f32[384, 384, 1, 1]", primals_154: "f32[384, 384, 3, 3]", primals_155: "f32[384, 384, 1, 1]", primals_156: "f32[384, 384, 3, 3]", primals_157: "f32[384, 384, 1, 1]", primals_158: "f32[384, 384, 3, 3]", primals_159: "f32[384, 384, 1, 1]", primals_160: "f32[384, 384, 3, 3]", primals_161: "f32[384, 384, 1, 1]", primals_162: "f32[384, 384, 3, 3]", primals_163: "f32[384, 384, 1, 1]", primals_164: "f32[384, 384, 3, 3]", primals_165: "f32[1408, 384, 1, 1]", primals_166: "f32[1408, 384, 3, 3]", primals_352: "f32[8, 3, 224, 224]", convolution: "f32[8, 64, 112, 112]", squeeze_1: "f32[64]", convolution_1: "f32[8, 64, 112, 112]", squeeze_4: "f32[64]", relu: "f32[8, 64, 112, 112]", convolution_2: "f32[8, 96, 56, 56]", squeeze_7: "f32[96]", convolution_3: "f32[8, 96, 56, 56]", squeeze_10: "f32[96]", relu_1: "f32[8, 96, 56, 56]", squeeze_13: "f32[96]", convolution_4: "f32[8, 96, 56, 56]", squeeze_16: "f32[96]", convolution_5: "f32[8, 96, 56, 56]", squeeze_19: "f32[96]", relu_2: "f32[8, 96, 56, 56]", convolution_6: "f32[8, 192, 28, 28]", squeeze_22: "f32[192]", convolution_7: "f32[8, 192, 28, 28]", squeeze_25: "f32[192]", relu_3: "f32[8, 192, 28, 28]", squeeze_28: "f32[192]", convolution_8: "f32[8, 192, 28, 28]", squeeze_31: "f32[192]", convolution_9: "f32[8, 192, 28, 28]", squeeze_34: "f32[192]", relu_4: "f32[8, 192, 28, 28]", squeeze_37: "f32[192]", convolution_10: "f32[8, 192, 28, 28]", squeeze_40: "f32[192]", convolution_11: "f32[8, 192, 28, 28]", squeeze_43: "f32[192]", relu_5: "f32[8, 192, 28, 28]", squeeze_46: "f32[192]", convolution_12: "f32[8, 192, 28, 28]", squeeze_49: "f32[192]", convolution_13: "f32[8, 192, 28, 28]", squeeze_52: "f32[192]", relu_6: "f32[8, 192, 28, 28]", convolution_14: "f32[8, 384, 14, 14]", squeeze_55: "f32[384]", convolution_15: "f32[8, 384, 14, 14]", squeeze_58: "f32[384]", relu_7: "f32[8, 384, 14, 14]", squeeze_61: "f32[384]", convolution_16: "f32[8, 384, 14, 14]", squeeze_64: "f32[384]", convolution_17: "f32[8, 384, 14, 14]", squeeze_67: "f32[384]", relu_8: "f32[8, 384, 14, 14]", squeeze_70: "f32[384]", convolution_18: "f32[8, 384, 14, 14]", squeeze_73: "f32[384]", convolution_19: "f32[8, 384, 14, 14]", squeeze_76: "f32[384]", relu_9: "f32[8, 384, 14, 14]", squeeze_79: "f32[384]", convolution_20: "f32[8, 384, 14, 14]", squeeze_82: "f32[384]", convolution_21: "f32[8, 384, 14, 14]", squeeze_85: "f32[384]", relu_10: "f32[8, 384, 14, 14]", squeeze_88: "f32[384]", convolution_22: "f32[8, 384, 14, 14]", squeeze_91: "f32[384]", convolution_23: "f32[8, 384, 14, 14]", squeeze_94: "f32[384]", relu_11: "f32[8, 384, 14, 14]", squeeze_97: "f32[384]", convolution_24: "f32[8, 384, 14, 14]", squeeze_100: "f32[384]", convolution_25: "f32[8, 384, 14, 14]", squeeze_103: "f32[384]", relu_12: "f32[8, 384, 14, 14]", squeeze_106: "f32[384]", convolution_26: "f32[8, 384, 14, 14]", squeeze_109: "f32[384]", convolution_27: "f32[8, 384, 14, 14]", squeeze_112: "f32[384]", relu_13: "f32[8, 384, 14, 14]", squeeze_115: "f32[384]", convolution_28: "f32[8, 384, 14, 14]", squeeze_118: "f32[384]", convolution_29: "f32[8, 384, 14, 14]", squeeze_121: "f32[384]", relu_14: "f32[8, 384, 14, 14]", squeeze_124: "f32[384]", convolution_30: "f32[8, 384, 14, 14]", squeeze_127: "f32[384]", convolution_31: "f32[8, 384, 14, 14]", squeeze_130: "f32[384]", relu_15: "f32[8, 384, 14, 14]", squeeze_133: "f32[384]", convolution_32: "f32[8, 384, 14, 14]", squeeze_136: "f32[384]", convolution_33: "f32[8, 384, 14, 14]", squeeze_139: "f32[384]", relu_16: "f32[8, 384, 14, 14]", squeeze_142: "f32[384]", convolution_34: "f32[8, 384, 14, 14]", squeeze_145: "f32[384]", convolution_35: "f32[8, 384, 14, 14]", squeeze_148: "f32[384]", relu_17: "f32[8, 384, 14, 14]", squeeze_151: "f32[384]", convolution_36: "f32[8, 384, 14, 14]", squeeze_154: "f32[384]", convolution_37: "f32[8, 384, 14, 14]", squeeze_157: "f32[384]", relu_18: "f32[8, 384, 14, 14]", squeeze_160: "f32[384]", convolution_38: "f32[8, 384, 14, 14]", squeeze_163: "f32[384]", convolution_39: "f32[8, 384, 14, 14]", squeeze_166: "f32[384]", relu_19: "f32[8, 384, 14, 14]", squeeze_169: "f32[384]", convolution_40: "f32[8, 384, 14, 14]", squeeze_172: "f32[384]", convolution_41: "f32[8, 384, 14, 14]", squeeze_175: "f32[384]", relu_20: "f32[8, 384, 14, 14]", convolution_42: "f32[8, 1408, 7, 7]", squeeze_178: "f32[1408]", convolution_43: "f32[8, 1408, 7, 7]", squeeze_181: "f32[1408]", clone: "f32[8, 1408]", permute_1: "f32[1000, 1408]", le: "b8[8, 1408, 7, 7]", unsqueeze_246: "f32[1, 1408, 1, 1]", unsqueeze_258: "f32[1, 1408, 1, 1]", unsqueeze_270: "f32[1, 384, 1, 1]", unsqueeze_282: "f32[1, 384, 1, 1]", unsqueeze_294: "f32[1, 384, 1, 1]", unsqueeze_306: "f32[1, 384, 1, 1]", unsqueeze_318: "f32[1, 384, 1, 1]", unsqueeze_330: "f32[1, 384, 1, 1]", unsqueeze_342: "f32[1, 384, 1, 1]", unsqueeze_354: "f32[1, 384, 1, 1]", unsqueeze_366: "f32[1, 384, 1, 1]", unsqueeze_378: "f32[1, 384, 1, 1]", unsqueeze_390: "f32[1, 384, 1, 1]", unsqueeze_402: "f32[1, 384, 1, 1]", unsqueeze_414: "f32[1, 384, 1, 1]", unsqueeze_426: "f32[1, 384, 1, 1]", unsqueeze_438: "f32[1, 384, 1, 1]", unsqueeze_450: "f32[1, 384, 1, 1]", unsqueeze_462: "f32[1, 384, 1, 1]", unsqueeze_474: "f32[1, 384, 1, 1]", unsqueeze_486: "f32[1, 384, 1, 1]", unsqueeze_498: "f32[1, 384, 1, 1]", unsqueeze_510: "f32[1, 384, 1, 1]", unsqueeze_522: "f32[1, 384, 1, 1]", unsqueeze_534: "f32[1, 384, 1, 1]", unsqueeze_546: "f32[1, 384, 1, 1]", unsqueeze_558: "f32[1, 384, 1, 1]", unsqueeze_570: "f32[1, 384, 1, 1]", unsqueeze_582: "f32[1, 384, 1, 1]", unsqueeze_594: "f32[1, 384, 1, 1]", unsqueeze_606: "f32[1, 384, 1, 1]", unsqueeze_618: "f32[1, 384, 1, 1]", unsqueeze_630: "f32[1, 384, 1, 1]", unsqueeze_642: "f32[1, 384, 1, 1]", unsqueeze_654: "f32[1, 384, 1, 1]", unsqueeze_666: "f32[1, 384, 1, 1]", unsqueeze_678: "f32[1, 384, 1, 1]", unsqueeze_690: "f32[1, 384, 1, 1]", unsqueeze_702: "f32[1, 384, 1, 1]", unsqueeze_714: "f32[1, 384, 1, 1]", unsqueeze_726: "f32[1, 384, 1, 1]", unsqueeze_738: "f32[1, 384, 1, 1]", unsqueeze_750: "f32[1, 384, 1, 1]", unsqueeze_762: "f32[1, 192, 1, 1]", unsqueeze_774: "f32[1, 192, 1, 1]", unsqueeze_786: "f32[1, 192, 1, 1]", unsqueeze_798: "f32[1, 192, 1, 1]", unsqueeze_810: "f32[1, 192, 1, 1]", unsqueeze_822: "f32[1, 192, 1, 1]", unsqueeze_834: "f32[1, 192, 1, 1]", unsqueeze_846: "f32[1, 192, 1, 1]", unsqueeze_858: "f32[1, 192, 1, 1]", unsqueeze_870: "f32[1, 192, 1, 1]", unsqueeze_882: "f32[1, 192, 1, 1]", unsqueeze_894: "f32[1, 96, 1, 1]", unsqueeze_906: "f32[1, 96, 1, 1]", unsqueeze_918: "f32[1, 96, 1, 1]", unsqueeze_930: "f32[1, 96, 1, 1]", unsqueeze_942: "f32[1, 96, 1, 1]", unsqueeze_954: "f32[1, 64, 1, 1]", unsqueeze_966: "f32[1, 64, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm: "f32[8, 1408]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1408]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[1408, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1408]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1408, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1408, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1408, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 1408, 7, 7]);  view_2 = None
    div: "f32[8, 1408, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 1408, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[1408]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_61: "f32[8, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_246);  convolution_43 = unsqueeze_246 = None
    mul_427: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_61)
    sum_3: "f32[1408]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None
    mul_428: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_247: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_248: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 2);  unsqueeze_247 = None
    unsqueeze_249: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, 3);  unsqueeze_248 = None
    mul_429: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_430: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_431: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
    unsqueeze_250: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_251: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 2);  unsqueeze_250 = None
    unsqueeze_252: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 3);  unsqueeze_251 = None
    mul_432: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_121);  primals_121 = None
    unsqueeze_253: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
    unsqueeze_254: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 2);  unsqueeze_253 = None
    unsqueeze_255: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 3);  unsqueeze_254 = None
    mul_433: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_252);  sub_61 = unsqueeze_252 = None
    sub_63: "f32[8, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_433);  mul_433 = None
    sub_64: "f32[8, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(sub_63, unsqueeze_249);  sub_63 = None
    mul_434: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_255);  sub_64 = unsqueeze_255 = None
    mul_435: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_181);  sum_3 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_434, relu_20, primals_166, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_434 = primals_166 = None
    getitem_122: "f32[8, 384, 14, 14]" = convolution_backward[0]
    getitem_123: "f32[1408, 384, 3, 3]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_65: "f32[8, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_258);  convolution_42 = unsqueeze_258 = None
    mul_436: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_65)
    sum_5: "f32[1408]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3]);  mul_436 = None
    mul_438: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_439: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_440: "f32[1408]" = torch.ops.aten.mul.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_262: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_263: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_441: "f32[1408]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_119);  primals_119 = None
    unsqueeze_265: "f32[1, 1408]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_266: "f32[1, 1408, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 1408, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    mul_442: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_264);  sub_65 = unsqueeze_264 = None
    sub_67: "f32[8, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_442);  where = mul_442 = None
    sub_68: "f32[8, 1408, 7, 7]" = torch.ops.aten.sub.Tensor(sub_67, unsqueeze_249);  sub_67 = unsqueeze_249 = None
    mul_443: "f32[8, 1408, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_267);  sub_68 = unsqueeze_267 = None
    mul_444: "f32[1408]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_178);  sum_5 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_443, relu_20, primals_165, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_443 = primals_165 = None
    getitem_125: "f32[8, 384, 14, 14]" = convolution_backward_1[0]
    getitem_126: "f32[1408, 384, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_344: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_122, getitem_125);  getitem_122 = getitem_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_26: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_27: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    le_1: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_27, 0);  alias_27 = None
    where_1: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_1, full_default, add_344);  le_1 = add_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_6: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_69: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_270);  convolution_41 = unsqueeze_270 = None
    mul_445: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_1, sub_69)
    sum_7: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
    mul_446: "f32[384]" = torch.ops.aten.mul.Tensor(sum_6, 0.0006377551020408163)
    unsqueeze_271: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
    unsqueeze_272: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_447: "f32[384]" = torch.ops.aten.mul.Tensor(sum_7, 0.0006377551020408163)
    mul_448: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_449: "f32[384]" = torch.ops.aten.mul.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
    unsqueeze_274: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
    unsqueeze_275: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_450: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_117);  primals_117 = None
    unsqueeze_277: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_278: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    mul_451: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_276);  sub_69 = unsqueeze_276 = None
    sub_71: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_1, mul_451);  mul_451 = None
    sub_72: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_71, unsqueeze_273);  sub_71 = None
    mul_452: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_279);  sub_72 = unsqueeze_279 = None
    mul_453: "f32[384]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_175);  sum_7 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_452, relu_19, primals_164, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_452 = primals_164 = None
    getitem_128: "f32[8, 384, 14, 14]" = convolution_backward_2[0]
    getitem_129: "f32[384, 384, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_73: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_282);  convolution_40 = unsqueeze_282 = None
    mul_454: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_1, sub_73)
    sum_9: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_454, [0, 2, 3]);  mul_454 = None
    mul_456: "f32[384]" = torch.ops.aten.mul.Tensor(sum_9, 0.0006377551020408163)
    mul_457: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_458: "f32[384]" = torch.ops.aten.mul.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
    unsqueeze_286: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_287: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_459: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_115);  primals_115 = None
    unsqueeze_289: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_290: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    mul_460: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_288);  sub_73 = unsqueeze_288 = None
    sub_75: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_1, mul_460);  mul_460 = None
    sub_76: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_273);  sub_75 = None
    mul_461: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_291);  sub_76 = unsqueeze_291 = None
    mul_462: "f32[384]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_172);  sum_9 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_461, relu_19, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_461 = primals_163 = None
    getitem_131: "f32[8, 384, 14, 14]" = convolution_backward_3[0]
    getitem_132: "f32[384, 384, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_345: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_128, getitem_131);  getitem_128 = getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_77: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_19, unsqueeze_294);  unsqueeze_294 = None
    mul_463: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_1, sub_77)
    sum_11: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_463, [0, 2, 3]);  mul_463 = None
    mul_465: "f32[384]" = torch.ops.aten.mul.Tensor(sum_11, 0.0006377551020408163)
    mul_466: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_467: "f32[384]" = torch.ops.aten.mul.Tensor(mul_465, mul_466);  mul_465 = mul_466 = None
    unsqueeze_298: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
    unsqueeze_299: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_468: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_301: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_302: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    mul_469: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_300);  sub_77 = unsqueeze_300 = None
    sub_79: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_1, mul_469);  where_1 = mul_469 = None
    sub_80: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_273);  sub_79 = unsqueeze_273 = None
    mul_470: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_303);  sub_80 = unsqueeze_303 = None
    mul_471: "f32[384]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_169);  sum_11 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_346: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_345, mul_470);  add_345 = mul_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_29: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_30: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    le_2: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_30, 0);  alias_30 = None
    where_2: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_2, full_default, add_346);  le_2 = add_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_12: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_81: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_306);  convolution_39 = unsqueeze_306 = None
    mul_472: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_2, sub_81)
    sum_13: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 2, 3]);  mul_472 = None
    mul_473: "f32[384]" = torch.ops.aten.mul.Tensor(sum_12, 0.0006377551020408163)
    unsqueeze_307: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_473, 0);  mul_473 = None
    unsqueeze_308: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_474: "f32[384]" = torch.ops.aten.mul.Tensor(sum_13, 0.0006377551020408163)
    mul_475: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_476: "f32[384]" = torch.ops.aten.mul.Tensor(mul_474, mul_475);  mul_474 = mul_475 = None
    unsqueeze_310: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
    unsqueeze_311: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_477: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_313: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_314: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    mul_478: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_312);  sub_81 = unsqueeze_312 = None
    sub_83: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_2, mul_478);  mul_478 = None
    sub_84: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_309);  sub_83 = None
    mul_479: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_315);  sub_84 = unsqueeze_315 = None
    mul_480: "f32[384]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_166);  sum_13 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_479, relu_18, primals_162, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_479 = primals_162 = None
    getitem_134: "f32[8, 384, 14, 14]" = convolution_backward_4[0]
    getitem_135: "f32[384, 384, 3, 3]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_85: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_318);  convolution_38 = unsqueeze_318 = None
    mul_481: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_2, sub_85)
    sum_15: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_481, [0, 2, 3]);  mul_481 = None
    mul_483: "f32[384]" = torch.ops.aten.mul.Tensor(sum_15, 0.0006377551020408163)
    mul_484: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_485: "f32[384]" = torch.ops.aten.mul.Tensor(mul_483, mul_484);  mul_483 = mul_484 = None
    unsqueeze_322: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
    unsqueeze_323: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_486: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_325: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
    unsqueeze_326: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    mul_487: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_324);  sub_85 = unsqueeze_324 = None
    sub_87: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_2, mul_487);  mul_487 = None
    sub_88: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_309);  sub_87 = None
    mul_488: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_327);  sub_88 = unsqueeze_327 = None
    mul_489: "f32[384]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_163);  sum_15 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_488, relu_18, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_488 = primals_161 = None
    getitem_137: "f32[8, 384, 14, 14]" = convolution_backward_5[0]
    getitem_138: "f32[384, 384, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_347: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_134, getitem_137);  getitem_134 = getitem_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_89: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_18, unsqueeze_330);  unsqueeze_330 = None
    mul_490: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_2, sub_89)
    sum_17: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 2, 3]);  mul_490 = None
    mul_492: "f32[384]" = torch.ops.aten.mul.Tensor(sum_17, 0.0006377551020408163)
    mul_493: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_494: "f32[384]" = torch.ops.aten.mul.Tensor(mul_492, mul_493);  mul_492 = mul_493 = None
    unsqueeze_334: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_335: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_495: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_337: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_338: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    mul_496: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_336);  sub_89 = unsqueeze_336 = None
    sub_91: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_2, mul_496);  where_2 = mul_496 = None
    sub_92: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_309);  sub_91 = unsqueeze_309 = None
    mul_497: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_339);  sub_92 = unsqueeze_339 = None
    mul_498: "f32[384]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_160);  sum_17 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_348: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_347, mul_497);  add_347 = mul_497 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_32: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_33: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    le_3: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_33, 0);  alias_33 = None
    where_3: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_3, full_default, add_348);  le_3 = add_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_93: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_342);  convolution_37 = unsqueeze_342 = None
    mul_499: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_3, sub_93)
    sum_19: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 2, 3]);  mul_499 = None
    mul_500: "f32[384]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006377551020408163)
    unsqueeze_343: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_500, 0);  mul_500 = None
    unsqueeze_344: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_501: "f32[384]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
    mul_502: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_503: "f32[384]" = torch.ops.aten.mul.Tensor(mul_501, mul_502);  mul_501 = mul_502 = None
    unsqueeze_346: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    unsqueeze_347: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_504: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_349: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_350: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    mul_505: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_348);  sub_93 = unsqueeze_348 = None
    sub_95: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_3, mul_505);  mul_505 = None
    sub_96: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_345);  sub_95 = None
    mul_506: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_351);  sub_96 = unsqueeze_351 = None
    mul_507: "f32[384]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_157);  sum_19 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_506, relu_17, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_506 = primals_160 = None
    getitem_140: "f32[8, 384, 14, 14]" = convolution_backward_6[0]
    getitem_141: "f32[384, 384, 3, 3]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_97: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_354);  convolution_36 = unsqueeze_354 = None
    mul_508: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_3, sub_97)
    sum_21: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_508, [0, 2, 3]);  mul_508 = None
    mul_510: "f32[384]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
    mul_511: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_512: "f32[384]" = torch.ops.aten.mul.Tensor(mul_510, mul_511);  mul_510 = mul_511 = None
    unsqueeze_358: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    unsqueeze_359: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_513: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_361: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
    unsqueeze_362: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    mul_514: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_360);  sub_97 = unsqueeze_360 = None
    sub_99: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_3, mul_514);  mul_514 = None
    sub_100: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_345);  sub_99 = None
    mul_515: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_363);  sub_100 = unsqueeze_363 = None
    mul_516: "f32[384]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_154);  sum_21 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_515, relu_17, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_515 = primals_159 = None
    getitem_143: "f32[8, 384, 14, 14]" = convolution_backward_7[0]
    getitem_144: "f32[384, 384, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_349: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_140, getitem_143);  getitem_140 = getitem_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_101: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_17, unsqueeze_366);  unsqueeze_366 = None
    mul_517: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_3, sub_101)
    sum_23: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_517, [0, 2, 3]);  mul_517 = None
    mul_519: "f32[384]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
    mul_520: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_521: "f32[384]" = torch.ops.aten.mul.Tensor(mul_519, mul_520);  mul_519 = mul_520 = None
    unsqueeze_370: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_371: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_522: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_373: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
    unsqueeze_374: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    mul_523: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_372);  sub_101 = unsqueeze_372 = None
    sub_103: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_3, mul_523);  where_3 = mul_523 = None
    sub_104: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_345);  sub_103 = unsqueeze_345 = None
    mul_524: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_375);  sub_104 = unsqueeze_375 = None
    mul_525: "f32[384]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_151);  sum_23 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_350: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_349, mul_524);  add_349 = mul_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_35: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_36: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    le_4: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_36, 0);  alias_36 = None
    where_4: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_4, full_default, add_350);  le_4 = add_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_24: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_105: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_378);  convolution_35 = unsqueeze_378 = None
    mul_526: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_105)
    sum_25: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_526, [0, 2, 3]);  mul_526 = None
    mul_527: "f32[384]" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
    unsqueeze_379: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_527, 0);  mul_527 = None
    unsqueeze_380: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_528: "f32[384]" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
    mul_529: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_530: "f32[384]" = torch.ops.aten.mul.Tensor(mul_528, mul_529);  mul_528 = mul_529 = None
    unsqueeze_382: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
    unsqueeze_383: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_531: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_385: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
    unsqueeze_386: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_532: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_384);  sub_105 = unsqueeze_384 = None
    sub_107: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_4, mul_532);  mul_532 = None
    sub_108: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_381);  sub_107 = None
    mul_533: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_387);  sub_108 = unsqueeze_387 = None
    mul_534: "f32[384]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_148);  sum_25 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_533, relu_16, primals_158, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_533 = primals_158 = None
    getitem_146: "f32[8, 384, 14, 14]" = convolution_backward_8[0]
    getitem_147: "f32[384, 384, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_109: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_390);  convolution_34 = unsqueeze_390 = None
    mul_535: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_109)
    sum_27: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_535, [0, 2, 3]);  mul_535 = None
    mul_537: "f32[384]" = torch.ops.aten.mul.Tensor(sum_27, 0.0006377551020408163)
    mul_538: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_539: "f32[384]" = torch.ops.aten.mul.Tensor(mul_537, mul_538);  mul_537 = mul_538 = None
    unsqueeze_394: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_395: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_540: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_397: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_398: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    mul_541: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_396);  sub_109 = unsqueeze_396 = None
    sub_111: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_4, mul_541);  mul_541 = None
    sub_112: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_381);  sub_111 = None
    mul_542: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_399);  sub_112 = unsqueeze_399 = None
    mul_543: "f32[384]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_145);  sum_27 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_542, relu_16, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_542 = primals_157 = None
    getitem_149: "f32[8, 384, 14, 14]" = convolution_backward_9[0]
    getitem_150: "f32[384, 384, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_351: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_146, getitem_149);  getitem_146 = getitem_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_113: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_16, unsqueeze_402);  unsqueeze_402 = None
    mul_544: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_113)
    sum_29: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_544, [0, 2, 3]);  mul_544 = None
    mul_546: "f32[384]" = torch.ops.aten.mul.Tensor(sum_29, 0.0006377551020408163)
    mul_547: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_548: "f32[384]" = torch.ops.aten.mul.Tensor(mul_546, mul_547);  mul_546 = mul_547 = None
    unsqueeze_406: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_407: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_549: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_409: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
    unsqueeze_410: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    mul_550: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_408);  sub_113 = unsqueeze_408 = None
    sub_115: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_4, mul_550);  where_4 = mul_550 = None
    sub_116: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_381);  sub_115 = unsqueeze_381 = None
    mul_551: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_411);  sub_116 = unsqueeze_411 = None
    mul_552: "f32[384]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_142);  sum_29 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_352: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_351, mul_551);  add_351 = mul_551 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_38: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_39: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_38);  alias_38 = None
    le_5: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_39, 0);  alias_39 = None
    where_5: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_5, full_default, add_352);  le_5 = add_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_30: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_117: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_414);  convolution_33 = unsqueeze_414 = None
    mul_553: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_5, sub_117)
    sum_31: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 2, 3]);  mul_553 = None
    mul_554: "f32[384]" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
    unsqueeze_415: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_554, 0);  mul_554 = None
    unsqueeze_416: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_555: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
    mul_556: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_557: "f32[384]" = torch.ops.aten.mul.Tensor(mul_555, mul_556);  mul_555 = mul_556 = None
    unsqueeze_418: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
    unsqueeze_419: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_558: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_421: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
    unsqueeze_422: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    mul_559: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_420);  sub_117 = unsqueeze_420 = None
    sub_119: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_5, mul_559);  mul_559 = None
    sub_120: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_417);  sub_119 = None
    mul_560: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_423);  sub_120 = unsqueeze_423 = None
    mul_561: "f32[384]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_139);  sum_31 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_560, relu_15, primals_156, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_560 = primals_156 = None
    getitem_152: "f32[8, 384, 14, 14]" = convolution_backward_10[0]
    getitem_153: "f32[384, 384, 3, 3]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_121: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_426);  convolution_32 = unsqueeze_426 = None
    mul_562: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_5, sub_121)
    sum_33: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_562, [0, 2, 3]);  mul_562 = None
    mul_564: "f32[384]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_565: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_566: "f32[384]" = torch.ops.aten.mul.Tensor(mul_564, mul_565);  mul_564 = mul_565 = None
    unsqueeze_430: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    unsqueeze_431: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_567: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_433: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_434: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    mul_568: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_432);  sub_121 = unsqueeze_432 = None
    sub_123: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_5, mul_568);  mul_568 = None
    sub_124: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_417);  sub_123 = None
    mul_569: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_435);  sub_124 = unsqueeze_435 = None
    mul_570: "f32[384]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_136);  sum_33 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_569, relu_15, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_569 = primals_155 = None
    getitem_155: "f32[8, 384, 14, 14]" = convolution_backward_11[0]
    getitem_156: "f32[384, 384, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_353: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_152, getitem_155);  getitem_152 = getitem_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_125: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_15, unsqueeze_438);  unsqueeze_438 = None
    mul_571: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_5, sub_125)
    sum_35: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_571, [0, 2, 3]);  mul_571 = None
    mul_573: "f32[384]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    mul_574: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_575: "f32[384]" = torch.ops.aten.mul.Tensor(mul_573, mul_574);  mul_573 = mul_574 = None
    unsqueeze_442: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
    unsqueeze_443: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_576: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_445: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_446: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    mul_577: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_444);  sub_125 = unsqueeze_444 = None
    sub_127: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_5, mul_577);  where_5 = mul_577 = None
    sub_128: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_417);  sub_127 = unsqueeze_417 = None
    mul_578: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_447);  sub_128 = unsqueeze_447 = None
    mul_579: "f32[384]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_133);  sum_35 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_354: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_353, mul_578);  add_353 = mul_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_41: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_42: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_41);  alias_41 = None
    le_6: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_42, 0);  alias_42 = None
    where_6: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_6, full_default, add_354);  le_6 = add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_36: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_129: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_450);  convolution_31 = unsqueeze_450 = None
    mul_580: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_129)
    sum_37: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_580, [0, 2, 3]);  mul_580 = None
    mul_581: "f32[384]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    unsqueeze_451: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_581, 0);  mul_581 = None
    unsqueeze_452: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_582: "f32[384]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    mul_583: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_584: "f32[384]" = torch.ops.aten.mul.Tensor(mul_582, mul_583);  mul_582 = mul_583 = None
    unsqueeze_454: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
    unsqueeze_455: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_585: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_457: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_458: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    mul_586: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_456);  sub_129 = unsqueeze_456 = None
    sub_131: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_6, mul_586);  mul_586 = None
    sub_132: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_453);  sub_131 = None
    mul_587: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_459);  sub_132 = unsqueeze_459 = None
    mul_588: "f32[384]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_130);  sum_37 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_587, relu_14, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_587 = primals_154 = None
    getitem_158: "f32[8, 384, 14, 14]" = convolution_backward_12[0]
    getitem_159: "f32[384, 384, 3, 3]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_133: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_462);  convolution_30 = unsqueeze_462 = None
    mul_589: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_133)
    sum_39: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_589, [0, 2, 3]);  mul_589 = None
    mul_591: "f32[384]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_592: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_593: "f32[384]" = torch.ops.aten.mul.Tensor(mul_591, mul_592);  mul_591 = mul_592 = None
    unsqueeze_466: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_593, 0);  mul_593 = None
    unsqueeze_467: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_594: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_469: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
    unsqueeze_470: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    mul_595: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_468);  sub_133 = unsqueeze_468 = None
    sub_135: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_6, mul_595);  mul_595 = None
    sub_136: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_453);  sub_135 = None
    mul_596: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_471);  sub_136 = unsqueeze_471 = None
    mul_597: "f32[384]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_127);  sum_39 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_596, relu_14, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_596 = primals_153 = None
    getitem_161: "f32[8, 384, 14, 14]" = convolution_backward_13[0]
    getitem_162: "f32[384, 384, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_355: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_158, getitem_161);  getitem_158 = getitem_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_137: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_14, unsqueeze_474);  unsqueeze_474 = None
    mul_598: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_137)
    sum_41: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_598, [0, 2, 3]);  mul_598 = None
    mul_600: "f32[384]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_601: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_602: "f32[384]" = torch.ops.aten.mul.Tensor(mul_600, mul_601);  mul_600 = mul_601 = None
    unsqueeze_478: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
    unsqueeze_479: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_603: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_481: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
    unsqueeze_482: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    mul_604: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_480);  sub_137 = unsqueeze_480 = None
    sub_139: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_6, mul_604);  where_6 = mul_604 = None
    sub_140: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_453);  sub_139 = unsqueeze_453 = None
    mul_605: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_483);  sub_140 = unsqueeze_483 = None
    mul_606: "f32[384]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_124);  sum_41 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_356: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_355, mul_605);  add_355 = mul_605 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_44: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_45: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_44);  alias_44 = None
    le_7: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_45, 0);  alias_45 = None
    where_7: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_7, full_default, add_356);  le_7 = add_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_141: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_486);  convolution_29 = unsqueeze_486 = None
    mul_607: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_141)
    sum_43: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_607, [0, 2, 3]);  mul_607 = None
    mul_608: "f32[384]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_487: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_608, 0);  mul_608 = None
    unsqueeze_488: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_609: "f32[384]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_610: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_611: "f32[384]" = torch.ops.aten.mul.Tensor(mul_609, mul_610);  mul_609 = mul_610 = None
    unsqueeze_490: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
    unsqueeze_491: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_612: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_493: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_494: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    mul_613: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_492);  sub_141 = unsqueeze_492 = None
    sub_143: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_7, mul_613);  mul_613 = None
    sub_144: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_489);  sub_143 = None
    mul_614: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_495);  sub_144 = unsqueeze_495 = None
    mul_615: "f32[384]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_121);  sum_43 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_614, relu_13, primals_152, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_614 = primals_152 = None
    getitem_164: "f32[8, 384, 14, 14]" = convolution_backward_14[0]
    getitem_165: "f32[384, 384, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_145: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_498);  convolution_28 = unsqueeze_498 = None
    mul_616: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_145)
    sum_45: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_616, [0, 2, 3]);  mul_616 = None
    mul_618: "f32[384]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_619: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_620: "f32[384]" = torch.ops.aten.mul.Tensor(mul_618, mul_619);  mul_618 = mul_619 = None
    unsqueeze_502: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
    unsqueeze_503: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    mul_621: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_505: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_506: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    mul_622: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_504);  sub_145 = unsqueeze_504 = None
    sub_147: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_7, mul_622);  mul_622 = None
    sub_148: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_489);  sub_147 = None
    mul_623: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_507);  sub_148 = unsqueeze_507 = None
    mul_624: "f32[384]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_118);  sum_45 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_623, relu_13, primals_151, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_623 = primals_151 = None
    getitem_167: "f32[8, 384, 14, 14]" = convolution_backward_15[0]
    getitem_168: "f32[384, 384, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_357: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_164, getitem_167);  getitem_164 = getitem_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_149: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_13, unsqueeze_510);  unsqueeze_510 = None
    mul_625: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_149)
    sum_47: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_625, [0, 2, 3]);  mul_625 = None
    mul_627: "f32[384]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_628: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_629: "f32[384]" = torch.ops.aten.mul.Tensor(mul_627, mul_628);  mul_627 = mul_628 = None
    unsqueeze_514: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
    unsqueeze_515: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    mul_630: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_517: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    unsqueeze_518: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    mul_631: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_516);  sub_149 = unsqueeze_516 = None
    sub_151: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_7, mul_631);  where_7 = mul_631 = None
    sub_152: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_489);  sub_151 = unsqueeze_489 = None
    mul_632: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_519);  sub_152 = unsqueeze_519 = None
    mul_633: "f32[384]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_115);  sum_47 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_358: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_357, mul_632);  add_357 = mul_632 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_47: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_48: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    le_8: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_48, 0);  alias_48 = None
    where_8: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_8, full_default, add_358);  le_8 = add_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_48: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_153: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_522);  convolution_27 = unsqueeze_522 = None
    mul_634: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_153)
    sum_49: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 2, 3]);  mul_634 = None
    mul_635: "f32[384]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_523: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_524: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_636: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_637: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_638: "f32[384]" = torch.ops.aten.mul.Tensor(mul_636, mul_637);  mul_636 = mul_637 = None
    unsqueeze_526: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_527: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    mul_639: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_529: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    unsqueeze_530: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    mul_640: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_528);  sub_153 = unsqueeze_528 = None
    sub_155: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_640);  mul_640 = None
    sub_156: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_525);  sub_155 = None
    mul_641: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_531);  sub_156 = unsqueeze_531 = None
    mul_642: "f32[384]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_112);  sum_49 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_641, relu_12, primals_150, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_641 = primals_150 = None
    getitem_170: "f32[8, 384, 14, 14]" = convolution_backward_16[0]
    getitem_171: "f32[384, 384, 3, 3]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_157: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_534);  convolution_26 = unsqueeze_534 = None
    mul_643: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_157)
    sum_51: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 2, 3]);  mul_643 = None
    mul_645: "f32[384]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_646: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_647: "f32[384]" = torch.ops.aten.mul.Tensor(mul_645, mul_646);  mul_645 = mul_646 = None
    unsqueeze_538: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
    unsqueeze_539: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    mul_648: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_541: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_542: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    mul_649: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_540);  sub_157 = unsqueeze_540 = None
    sub_159: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_649);  mul_649 = None
    sub_160: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_525);  sub_159 = None
    mul_650: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_543);  sub_160 = unsqueeze_543 = None
    mul_651: "f32[384]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_109);  sum_51 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_650, relu_12, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_650 = primals_149 = None
    getitem_173: "f32[8, 384, 14, 14]" = convolution_backward_17[0]
    getitem_174: "f32[384, 384, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_359: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_170, getitem_173);  getitem_170 = getitem_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_161: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_12, unsqueeze_546);  unsqueeze_546 = None
    mul_652: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_161)
    sum_53: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 2, 3]);  mul_652 = None
    mul_654: "f32[384]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    mul_655: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_656: "f32[384]" = torch.ops.aten.mul.Tensor(mul_654, mul_655);  mul_654 = mul_655 = None
    unsqueeze_550: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
    unsqueeze_551: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    mul_657: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_553: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
    unsqueeze_554: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    mul_658: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_552);  sub_161 = unsqueeze_552 = None
    sub_163: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_658);  where_8 = mul_658 = None
    sub_164: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_525);  sub_163 = unsqueeze_525 = None
    mul_659: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_555);  sub_164 = unsqueeze_555 = None
    mul_660: "f32[384]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_106);  sum_53 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_360: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_359, mul_659);  add_359 = mul_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_50: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_51: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    le_9: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_51, 0);  alias_51 = None
    where_9: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_9, full_default, add_360);  le_9 = add_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_165: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_558);  convolution_25 = unsqueeze_558 = None
    mul_661: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_165)
    sum_55: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2, 3]);  mul_661 = None
    mul_662: "f32[384]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_559: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_662, 0);  mul_662 = None
    unsqueeze_560: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_663: "f32[384]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_664: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_665: "f32[384]" = torch.ops.aten.mul.Tensor(mul_663, mul_664);  mul_663 = mul_664 = None
    unsqueeze_562: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
    unsqueeze_563: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_666: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_565: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
    unsqueeze_566: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    mul_667: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_564);  sub_165 = unsqueeze_564 = None
    sub_167: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_9, mul_667);  mul_667 = None
    sub_168: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_561);  sub_167 = None
    mul_668: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_567);  sub_168 = unsqueeze_567 = None
    mul_669: "f32[384]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_103);  sum_55 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_668, relu_11, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_668 = primals_148 = None
    getitem_176: "f32[8, 384, 14, 14]" = convolution_backward_18[0]
    getitem_177: "f32[384, 384, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_169: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_570);  convolution_24 = unsqueeze_570 = None
    mul_670: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_169)
    sum_57: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 2, 3]);  mul_670 = None
    mul_672: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_673: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_674: "f32[384]" = torch.ops.aten.mul.Tensor(mul_672, mul_673);  mul_672 = mul_673 = None
    unsqueeze_574: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
    unsqueeze_575: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    mul_675: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_577: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
    unsqueeze_578: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    mul_676: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_576);  sub_169 = unsqueeze_576 = None
    sub_171: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_9, mul_676);  mul_676 = None
    sub_172: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_561);  sub_171 = None
    mul_677: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_579);  sub_172 = unsqueeze_579 = None
    mul_678: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_100);  sum_57 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_677, relu_11, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_677 = primals_147 = None
    getitem_179: "f32[8, 384, 14, 14]" = convolution_backward_19[0]
    getitem_180: "f32[384, 384, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_361: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_176, getitem_179);  getitem_176 = getitem_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_173: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_11, unsqueeze_582);  unsqueeze_582 = None
    mul_679: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_173)
    sum_59: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_679, [0, 2, 3]);  mul_679 = None
    mul_681: "f32[384]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_682: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_683: "f32[384]" = torch.ops.aten.mul.Tensor(mul_681, mul_682);  mul_681 = mul_682 = None
    unsqueeze_586: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
    unsqueeze_587: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    mul_684: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_589: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_590: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    mul_685: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_588);  sub_173 = unsqueeze_588 = None
    sub_175: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_9, mul_685);  where_9 = mul_685 = None
    sub_176: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_561);  sub_175 = unsqueeze_561 = None
    mul_686: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_591);  sub_176 = unsqueeze_591 = None
    mul_687: "f32[384]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_97);  sum_59 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_362: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_361, mul_686);  add_361 = mul_686 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_53: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_54: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_53);  alias_53 = None
    le_10: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_54, 0);  alias_54 = None
    where_10: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, add_362);  le_10 = add_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_60: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_177: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_594);  convolution_23 = unsqueeze_594 = None
    mul_688: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_177)
    sum_61: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_688, [0, 2, 3]);  mul_688 = None
    mul_689: "f32[384]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_595: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_689, 0);  mul_689 = None
    unsqueeze_596: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_690: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_691: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_692: "f32[384]" = torch.ops.aten.mul.Tensor(mul_690, mul_691);  mul_690 = mul_691 = None
    unsqueeze_598: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_599: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    mul_693: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_601: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
    unsqueeze_602: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    mul_694: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_600);  sub_177 = unsqueeze_600 = None
    sub_179: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_694);  mul_694 = None
    sub_180: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_597);  sub_179 = None
    mul_695: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_603);  sub_180 = unsqueeze_603 = None
    mul_696: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_94);  sum_61 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_695, relu_10, primals_146, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_695 = primals_146 = None
    getitem_182: "f32[8, 384, 14, 14]" = convolution_backward_20[0]
    getitem_183: "f32[384, 384, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_181: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_606);  convolution_22 = unsqueeze_606 = None
    mul_697: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_181)
    sum_63: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_697, [0, 2, 3]);  mul_697 = None
    mul_699: "f32[384]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    mul_700: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_701: "f32[384]" = torch.ops.aten.mul.Tensor(mul_699, mul_700);  mul_699 = mul_700 = None
    unsqueeze_610: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
    unsqueeze_611: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    mul_702: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_613: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
    unsqueeze_614: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    mul_703: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_612);  sub_181 = unsqueeze_612 = None
    sub_183: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_703);  mul_703 = None
    sub_184: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_597);  sub_183 = None
    mul_704: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_615);  sub_184 = unsqueeze_615 = None
    mul_705: "f32[384]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_91);  sum_63 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_704, relu_10, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_704 = primals_145 = None
    getitem_185: "f32[8, 384, 14, 14]" = convolution_backward_21[0]
    getitem_186: "f32[384, 384, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_363: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_182, getitem_185);  getitem_182 = getitem_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_185: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_10, unsqueeze_618);  unsqueeze_618 = None
    mul_706: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_185)
    sum_65: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 2, 3]);  mul_706 = None
    mul_708: "f32[384]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    mul_709: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_710: "f32[384]" = torch.ops.aten.mul.Tensor(mul_708, mul_709);  mul_708 = mul_709 = None
    unsqueeze_622: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
    unsqueeze_623: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_711: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_625: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    unsqueeze_626: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    mul_712: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_624);  sub_185 = unsqueeze_624 = None
    sub_187: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_712);  where_10 = mul_712 = None
    sub_188: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_597);  sub_187 = unsqueeze_597 = None
    mul_713: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_627);  sub_188 = unsqueeze_627 = None
    mul_714: "f32[384]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_88);  sum_65 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_364: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_363, mul_713);  add_363 = mul_713 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_56: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_57: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    le_11: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_57, 0);  alias_57 = None
    where_11: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, add_364);  le_11 = add_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_66: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_189: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_630);  convolution_21 = unsqueeze_630 = None
    mul_715: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_189)
    sum_67: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_715, [0, 2, 3]);  mul_715 = None
    mul_716: "f32[384]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    unsqueeze_631: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
    unsqueeze_632: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_717: "f32[384]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    mul_718: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_719: "f32[384]" = torch.ops.aten.mul.Tensor(mul_717, mul_718);  mul_717 = mul_718 = None
    unsqueeze_634: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
    unsqueeze_635: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    mul_720: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_637: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    unsqueeze_638: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    mul_721: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_636);  sub_189 = unsqueeze_636 = None
    sub_191: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_721);  mul_721 = None
    sub_192: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_633);  sub_191 = None
    mul_722: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_639);  sub_192 = unsqueeze_639 = None
    mul_723: "f32[384]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_85);  sum_67 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_722, relu_9, primals_144, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_722 = primals_144 = None
    getitem_188: "f32[8, 384, 14, 14]" = convolution_backward_22[0]
    getitem_189: "f32[384, 384, 3, 3]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_193: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_642);  convolution_20 = unsqueeze_642 = None
    mul_724: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_193)
    sum_69: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_724, [0, 2, 3]);  mul_724 = None
    mul_726: "f32[384]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    mul_727: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_728: "f32[384]" = torch.ops.aten.mul.Tensor(mul_726, mul_727);  mul_726 = mul_727 = None
    unsqueeze_646: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
    unsqueeze_647: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    mul_729: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_649: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
    unsqueeze_650: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    mul_730: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_648);  sub_193 = unsqueeze_648 = None
    sub_195: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_730);  mul_730 = None
    sub_196: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_633);  sub_195 = None
    mul_731: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_651);  sub_196 = unsqueeze_651 = None
    mul_732: "f32[384]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_82);  sum_69 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_731, relu_9, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_731 = primals_143 = None
    getitem_191: "f32[8, 384, 14, 14]" = convolution_backward_23[0]
    getitem_192: "f32[384, 384, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_365: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_188, getitem_191);  getitem_188 = getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_197: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_9, unsqueeze_654);  unsqueeze_654 = None
    mul_733: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_197)
    sum_71: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_733, [0, 2, 3]);  mul_733 = None
    mul_735: "f32[384]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_736: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_737: "f32[384]" = torch.ops.aten.mul.Tensor(mul_735, mul_736);  mul_735 = mul_736 = None
    unsqueeze_658: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
    unsqueeze_659: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    mul_738: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_661: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_662: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    mul_739: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_660);  sub_197 = unsqueeze_660 = None
    sub_199: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_739);  where_11 = mul_739 = None
    sub_200: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_633);  sub_199 = unsqueeze_633 = None
    mul_740: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_663);  sub_200 = unsqueeze_663 = None
    mul_741: "f32[384]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_79);  sum_71 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_366: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_365, mul_740);  add_365 = mul_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_59: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_60: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    le_12: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_60, 0);  alias_60 = None
    where_12: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, add_366);  le_12 = add_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_201: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_666);  convolution_19 = unsqueeze_666 = None
    mul_742: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_201)
    sum_73: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_742, [0, 2, 3]);  mul_742 = None
    mul_743: "f32[384]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_667: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_668: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_744: "f32[384]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_745: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_746: "f32[384]" = torch.ops.aten.mul.Tensor(mul_744, mul_745);  mul_744 = mul_745 = None
    unsqueeze_670: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_671: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    mul_747: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_673: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
    unsqueeze_674: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    mul_748: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_672);  sub_201 = unsqueeze_672 = None
    sub_203: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_748);  mul_748 = None
    sub_204: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_669);  sub_203 = None
    mul_749: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_675);  sub_204 = unsqueeze_675 = None
    mul_750: "f32[384]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_76);  sum_73 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_749, relu_8, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_749 = primals_142 = None
    getitem_194: "f32[8, 384, 14, 14]" = convolution_backward_24[0]
    getitem_195: "f32[384, 384, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_205: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_678);  convolution_18 = unsqueeze_678 = None
    mul_751: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_205)
    sum_75: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 2, 3]);  mul_751 = None
    mul_753: "f32[384]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_754: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_755: "f32[384]" = torch.ops.aten.mul.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
    unsqueeze_682: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_683: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    mul_756: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_685: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_686: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    mul_757: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_684);  sub_205 = unsqueeze_684 = None
    sub_207: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_757);  mul_757 = None
    sub_208: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_669);  sub_207 = None
    mul_758: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_687);  sub_208 = unsqueeze_687 = None
    mul_759: "f32[384]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_73);  sum_75 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_758, relu_8, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_758 = primals_141 = None
    getitem_197: "f32[8, 384, 14, 14]" = convolution_backward_25[0]
    getitem_198: "f32[384, 384, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_367: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_194, getitem_197);  getitem_194 = getitem_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_209: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_8, unsqueeze_690);  unsqueeze_690 = None
    mul_760: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_209)
    sum_77: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_760, [0, 2, 3]);  mul_760 = None
    mul_762: "f32[384]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_763: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_764: "f32[384]" = torch.ops.aten.mul.Tensor(mul_762, mul_763);  mul_762 = mul_763 = None
    unsqueeze_694: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_695: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    mul_765: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_697: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
    unsqueeze_698: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    mul_766: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_696);  sub_209 = unsqueeze_696 = None
    sub_211: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_766);  where_12 = mul_766 = None
    sub_212: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_669);  sub_211 = unsqueeze_669 = None
    mul_767: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_699);  sub_212 = unsqueeze_699 = None
    mul_768: "f32[384]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_70);  sum_77 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_368: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_367, mul_767);  add_367 = mul_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_62: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_63: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    le_13: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_63, 0);  alias_63 = None
    where_13: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, add_368);  le_13 = add_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_78: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_213: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_702);  convolution_17 = unsqueeze_702 = None
    mul_769: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_213)
    sum_79: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_769, [0, 2, 3]);  mul_769 = None
    mul_770: "f32[384]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    unsqueeze_703: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_770, 0);  mul_770 = None
    unsqueeze_704: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_771: "f32[384]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    mul_772: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_773: "f32[384]" = torch.ops.aten.mul.Tensor(mul_771, mul_772);  mul_771 = mul_772 = None
    unsqueeze_706: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_707: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    mul_774: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_709: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_710: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    mul_775: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_708);  sub_213 = unsqueeze_708 = None
    sub_215: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_775);  mul_775 = None
    sub_216: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_705);  sub_215 = None
    mul_776: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_711);  sub_216 = unsqueeze_711 = None
    mul_777: "f32[384]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_67);  sum_79 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_776, relu_7, primals_140, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_776 = primals_140 = None
    getitem_200: "f32[8, 384, 14, 14]" = convolution_backward_26[0]
    getitem_201: "f32[384, 384, 3, 3]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_217: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_714);  convolution_16 = unsqueeze_714 = None
    mul_778: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_217)
    sum_81: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_778, [0, 2, 3]);  mul_778 = None
    mul_780: "f32[384]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    mul_781: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_782: "f32[384]" = torch.ops.aten.mul.Tensor(mul_780, mul_781);  mul_780 = mul_781 = None
    unsqueeze_718: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_719: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    mul_783: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_721: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_722: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    mul_784: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_720);  sub_217 = unsqueeze_720 = None
    sub_219: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_784);  mul_784 = None
    sub_220: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_705);  sub_219 = None
    mul_785: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_723);  sub_220 = unsqueeze_723 = None
    mul_786: "f32[384]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_64);  sum_81 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_785, relu_7, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_785 = primals_139 = None
    getitem_203: "f32[8, 384, 14, 14]" = convolution_backward_27[0]
    getitem_204: "f32[384, 384, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_369: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(getitem_200, getitem_203);  getitem_200 = getitem_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_221: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(relu_7, unsqueeze_726);  unsqueeze_726 = None
    mul_787: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_221)
    sum_83: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_787, [0, 2, 3]);  mul_787 = None
    mul_789: "f32[384]" = torch.ops.aten.mul.Tensor(sum_83, 0.0006377551020408163)
    mul_790: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_791: "f32[384]" = torch.ops.aten.mul.Tensor(mul_789, mul_790);  mul_789 = mul_790 = None
    unsqueeze_730: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
    unsqueeze_731: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    mul_792: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_733: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
    unsqueeze_734: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    mul_793: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_732);  sub_221 = unsqueeze_732 = None
    sub_223: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_793);  where_13 = mul_793 = None
    sub_224: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_705);  sub_223 = unsqueeze_705 = None
    mul_794: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_735);  sub_224 = unsqueeze_735 = None
    mul_795: "f32[384]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_61);  sum_83 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_370: "f32[8, 384, 14, 14]" = torch.ops.aten.add.Tensor(add_369, mul_794);  add_369 = mul_794 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_65: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_66: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_65);  alias_65 = None
    le_14: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_66, 0);  alias_66 = None
    where_14: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, add_370);  le_14 = add_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_84: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_225: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_738);  convolution_15 = unsqueeze_738 = None
    mul_796: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_225)
    sum_85: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_796, [0, 2, 3]);  mul_796 = None
    mul_797: "f32[384]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    unsqueeze_739: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_797, 0);  mul_797 = None
    unsqueeze_740: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
    unsqueeze_741: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
    mul_798: "f32[384]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    mul_799: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_800: "f32[384]" = torch.ops.aten.mul.Tensor(mul_798, mul_799);  mul_798 = mul_799 = None
    unsqueeze_742: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_743: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
    unsqueeze_744: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
    mul_801: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_745: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_746: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    mul_802: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_744);  sub_225 = unsqueeze_744 = None
    sub_227: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_802);  mul_802 = None
    sub_228: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_741);  sub_227 = None
    mul_803: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_747);  sub_228 = unsqueeze_747 = None
    mul_804: "f32[384]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_58);  sum_85 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_803, relu_6, primals_138, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_803 = primals_138 = None
    getitem_206: "f32[8, 192, 28, 28]" = convolution_backward_28[0]
    getitem_207: "f32[384, 192, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_229: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_750);  convolution_14 = unsqueeze_750 = None
    mul_805: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_229)
    sum_87: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_805, [0, 2, 3]);  mul_805 = None
    mul_807: "f32[384]" = torch.ops.aten.mul.Tensor(sum_87, 0.0006377551020408163)
    mul_808: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_809: "f32[384]" = torch.ops.aten.mul.Tensor(mul_807, mul_808);  mul_807 = mul_808 = None
    unsqueeze_754: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
    unsqueeze_755: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
    unsqueeze_756: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
    mul_810: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_757: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
    unsqueeze_758: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    mul_811: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_756);  sub_229 = unsqueeze_756 = None
    sub_231: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_811);  where_14 = mul_811 = None
    sub_232: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_741);  sub_231 = unsqueeze_741 = None
    mul_812: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_759);  sub_232 = unsqueeze_759 = None
    mul_813: "f32[384]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_55);  sum_87 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_812, relu_6, primals_137, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_812 = primals_137 = None
    getitem_209: "f32[8, 192, 28, 28]" = convolution_backward_29[0]
    getitem_210: "f32[384, 192, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_371: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(getitem_206, getitem_209);  getitem_206 = getitem_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_68: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_69: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_68);  alias_68 = None
    le_15: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_69, 0);  alias_69 = None
    where_15: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_15, full_default, add_371);  le_15 = add_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_88: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_233: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_762);  convolution_13 = unsqueeze_762 = None
    mul_814: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_15, sub_233)
    sum_89: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_814, [0, 2, 3]);  mul_814 = None
    mul_815: "f32[192]" = torch.ops.aten.mul.Tensor(sum_88, 0.00015943877551020407)
    unsqueeze_763: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_815, 0);  mul_815 = None
    unsqueeze_764: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
    unsqueeze_765: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
    mul_816: "f32[192]" = torch.ops.aten.mul.Tensor(sum_89, 0.00015943877551020407)
    mul_817: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_818: "f32[192]" = torch.ops.aten.mul.Tensor(mul_816, mul_817);  mul_816 = mul_817 = None
    unsqueeze_766: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_767: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
    unsqueeze_768: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
    mul_819: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_769: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
    unsqueeze_770: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    mul_820: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_768);  sub_233 = unsqueeze_768 = None
    sub_235: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_15, mul_820);  mul_820 = None
    sub_236: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_765);  sub_235 = None
    mul_821: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_771);  sub_236 = unsqueeze_771 = None
    mul_822: "f32[192]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_52);  sum_89 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_821, relu_5, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_821 = primals_136 = None
    getitem_212: "f32[8, 192, 28, 28]" = convolution_backward_30[0]
    getitem_213: "f32[192, 192, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_237: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_774);  convolution_12 = unsqueeze_774 = None
    mul_823: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_15, sub_237)
    sum_91: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_823, [0, 2, 3]);  mul_823 = None
    mul_825: "f32[192]" = torch.ops.aten.mul.Tensor(sum_91, 0.00015943877551020407)
    mul_826: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_827: "f32[192]" = torch.ops.aten.mul.Tensor(mul_825, mul_826);  mul_825 = mul_826 = None
    unsqueeze_778: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
    unsqueeze_779: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
    unsqueeze_780: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
    mul_828: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_781: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
    unsqueeze_782: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    mul_829: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_780);  sub_237 = unsqueeze_780 = None
    sub_239: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_15, mul_829);  mul_829 = None
    sub_240: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_765);  sub_239 = None
    mul_830: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_783);  sub_240 = unsqueeze_783 = None
    mul_831: "f32[192]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_49);  sum_91 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_830, relu_5, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_830 = primals_135 = None
    getitem_215: "f32[8, 192, 28, 28]" = convolution_backward_31[0]
    getitem_216: "f32[192, 192, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_372: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(getitem_212, getitem_215);  getitem_212 = getitem_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_241: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(relu_5, unsqueeze_786);  unsqueeze_786 = None
    mul_832: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_15, sub_241)
    sum_93: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_832, [0, 2, 3]);  mul_832 = None
    mul_834: "f32[192]" = torch.ops.aten.mul.Tensor(sum_93, 0.00015943877551020407)
    mul_835: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_836: "f32[192]" = torch.ops.aten.mul.Tensor(mul_834, mul_835);  mul_834 = mul_835 = None
    unsqueeze_790: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_836, 0);  mul_836 = None
    unsqueeze_791: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
    unsqueeze_792: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
    mul_837: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_793: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
    unsqueeze_794: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    mul_838: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_792);  sub_241 = unsqueeze_792 = None
    sub_243: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_15, mul_838);  where_15 = mul_838 = None
    sub_244: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_765);  sub_243 = unsqueeze_765 = None
    mul_839: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_795);  sub_244 = unsqueeze_795 = None
    mul_840: "f32[192]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_46);  sum_93 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_373: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_372, mul_839);  add_372 = mul_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_71: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_72: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_71);  alias_71 = None
    le_16: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_72, 0);  alias_72 = None
    where_16: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_16, full_default, add_373);  le_16 = add_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_94: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_245: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_798);  convolution_11 = unsqueeze_798 = None
    mul_841: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_16, sub_245)
    sum_95: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_841, [0, 2, 3]);  mul_841 = None
    mul_842: "f32[192]" = torch.ops.aten.mul.Tensor(sum_94, 0.00015943877551020407)
    unsqueeze_799: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_842, 0);  mul_842 = None
    unsqueeze_800: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
    unsqueeze_801: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
    mul_843: "f32[192]" = torch.ops.aten.mul.Tensor(sum_95, 0.00015943877551020407)
    mul_844: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_845: "f32[192]" = torch.ops.aten.mul.Tensor(mul_843, mul_844);  mul_843 = mul_844 = None
    unsqueeze_802: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
    unsqueeze_803: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
    unsqueeze_804: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
    mul_846: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_805: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
    unsqueeze_806: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    mul_847: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_804);  sub_245 = unsqueeze_804 = None
    sub_247: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_16, mul_847);  mul_847 = None
    sub_248: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_801);  sub_247 = None
    mul_848: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_807);  sub_248 = unsqueeze_807 = None
    mul_849: "f32[192]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_43);  sum_95 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_848, relu_4, primals_134, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_848 = primals_134 = None
    getitem_218: "f32[8, 192, 28, 28]" = convolution_backward_32[0]
    getitem_219: "f32[192, 192, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_249: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_810);  convolution_10 = unsqueeze_810 = None
    mul_850: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_16, sub_249)
    sum_97: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_850, [0, 2, 3]);  mul_850 = None
    mul_852: "f32[192]" = torch.ops.aten.mul.Tensor(sum_97, 0.00015943877551020407)
    mul_853: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_854: "f32[192]" = torch.ops.aten.mul.Tensor(mul_852, mul_853);  mul_852 = mul_853 = None
    unsqueeze_814: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_854, 0);  mul_854 = None
    unsqueeze_815: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
    unsqueeze_816: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
    mul_855: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_817: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
    unsqueeze_818: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    mul_856: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_816);  sub_249 = unsqueeze_816 = None
    sub_251: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_16, mul_856);  mul_856 = None
    sub_252: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_801);  sub_251 = None
    mul_857: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_819);  sub_252 = unsqueeze_819 = None
    mul_858: "f32[192]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_40);  sum_97 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_857, relu_4, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_857 = primals_133 = None
    getitem_221: "f32[8, 192, 28, 28]" = convolution_backward_33[0]
    getitem_222: "f32[192, 192, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_374: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(getitem_218, getitem_221);  getitem_218 = getitem_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_253: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(relu_4, unsqueeze_822);  unsqueeze_822 = None
    mul_859: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_16, sub_253)
    sum_99: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_859, [0, 2, 3]);  mul_859 = None
    mul_861: "f32[192]" = torch.ops.aten.mul.Tensor(sum_99, 0.00015943877551020407)
    mul_862: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_863: "f32[192]" = torch.ops.aten.mul.Tensor(mul_861, mul_862);  mul_861 = mul_862 = None
    unsqueeze_826: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
    unsqueeze_827: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
    unsqueeze_828: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
    mul_864: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_829: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
    unsqueeze_830: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    mul_865: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_828);  sub_253 = unsqueeze_828 = None
    sub_255: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_16, mul_865);  where_16 = mul_865 = None
    sub_256: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_801);  sub_255 = unsqueeze_801 = None
    mul_866: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_831);  sub_256 = unsqueeze_831 = None
    mul_867: "f32[192]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_37);  sum_99 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_375: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_374, mul_866);  add_374 = mul_866 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_74: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_75: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    le_17: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_75, 0);  alias_75 = None
    where_17: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_17, full_default, add_375);  le_17 = add_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_100: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_257: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_834);  convolution_9 = unsqueeze_834 = None
    mul_868: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_17, sub_257)
    sum_101: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_868, [0, 2, 3]);  mul_868 = None
    mul_869: "f32[192]" = torch.ops.aten.mul.Tensor(sum_100, 0.00015943877551020407)
    unsqueeze_835: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_869, 0);  mul_869 = None
    unsqueeze_836: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
    unsqueeze_837: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
    mul_870: "f32[192]" = torch.ops.aten.mul.Tensor(sum_101, 0.00015943877551020407)
    mul_871: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_872: "f32[192]" = torch.ops.aten.mul.Tensor(mul_870, mul_871);  mul_870 = mul_871 = None
    unsqueeze_838: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    unsqueeze_839: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
    unsqueeze_840: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
    mul_873: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_841: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_873, 0);  mul_873 = None
    unsqueeze_842: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    mul_874: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_840);  sub_257 = unsqueeze_840 = None
    sub_259: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_17, mul_874);  mul_874 = None
    sub_260: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_837);  sub_259 = None
    mul_875: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_843);  sub_260 = unsqueeze_843 = None
    mul_876: "f32[192]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_34);  sum_101 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_875, relu_3, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_875 = primals_132 = None
    getitem_224: "f32[8, 192, 28, 28]" = convolution_backward_34[0]
    getitem_225: "f32[192, 192, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_261: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_846);  convolution_8 = unsqueeze_846 = None
    mul_877: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_17, sub_261)
    sum_103: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_877, [0, 2, 3]);  mul_877 = None
    mul_879: "f32[192]" = torch.ops.aten.mul.Tensor(sum_103, 0.00015943877551020407)
    mul_880: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_881: "f32[192]" = torch.ops.aten.mul.Tensor(mul_879, mul_880);  mul_879 = mul_880 = None
    unsqueeze_850: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_851: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
    unsqueeze_852: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
    mul_882: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_853: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
    unsqueeze_854: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    mul_883: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_852);  sub_261 = unsqueeze_852 = None
    sub_263: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_17, mul_883);  mul_883 = None
    sub_264: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_837);  sub_263 = None
    mul_884: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_855);  sub_264 = unsqueeze_855 = None
    mul_885: "f32[192]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_31);  sum_103 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_884, relu_3, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_884 = primals_131 = None
    getitem_227: "f32[8, 192, 28, 28]" = convolution_backward_35[0]
    getitem_228: "f32[192, 192, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_376: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(getitem_224, getitem_227);  getitem_224 = getitem_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_265: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(relu_3, unsqueeze_858);  unsqueeze_858 = None
    mul_886: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_17, sub_265)
    sum_105: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_886, [0, 2, 3]);  mul_886 = None
    mul_888: "f32[192]" = torch.ops.aten.mul.Tensor(sum_105, 0.00015943877551020407)
    mul_889: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_890: "f32[192]" = torch.ops.aten.mul.Tensor(mul_888, mul_889);  mul_888 = mul_889 = None
    unsqueeze_862: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
    unsqueeze_863: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
    unsqueeze_864: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
    mul_891: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_865: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
    unsqueeze_866: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    mul_892: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_864);  sub_265 = unsqueeze_864 = None
    sub_267: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_17, mul_892);  where_17 = mul_892 = None
    sub_268: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_837);  sub_267 = unsqueeze_837 = None
    mul_893: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_867);  sub_268 = unsqueeze_867 = None
    mul_894: "f32[192]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_28);  sum_105 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_377: "f32[8, 192, 28, 28]" = torch.ops.aten.add.Tensor(add_376, mul_893);  add_376 = mul_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_77: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_78: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_77);  alias_77 = None
    le_18: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_78, 0);  alias_78 = None
    where_18: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_18, full_default, add_377);  le_18 = add_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_106: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_269: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_870);  convolution_7 = unsqueeze_870 = None
    mul_895: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_18, sub_269)
    sum_107: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_895, [0, 2, 3]);  mul_895 = None
    mul_896: "f32[192]" = torch.ops.aten.mul.Tensor(sum_106, 0.00015943877551020407)
    unsqueeze_871: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_896, 0);  mul_896 = None
    unsqueeze_872: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
    unsqueeze_873: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
    mul_897: "f32[192]" = torch.ops.aten.mul.Tensor(sum_107, 0.00015943877551020407)
    mul_898: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_899: "f32[192]" = torch.ops.aten.mul.Tensor(mul_897, mul_898);  mul_897 = mul_898 = None
    unsqueeze_874: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
    unsqueeze_875: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
    unsqueeze_876: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
    mul_900: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_877: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
    unsqueeze_878: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    mul_901: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_876);  sub_269 = unsqueeze_876 = None
    sub_271: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_18, mul_901);  mul_901 = None
    sub_272: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_873);  sub_271 = None
    mul_902: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_879);  sub_272 = unsqueeze_879 = None
    mul_903: "f32[192]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_25);  sum_107 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_902, relu_2, primals_130, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_902 = primals_130 = None
    getitem_230: "f32[8, 96, 56, 56]" = convolution_backward_36[0]
    getitem_231: "f32[192, 96, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_273: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_882);  convolution_6 = unsqueeze_882 = None
    mul_904: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_18, sub_273)
    sum_109: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_904, [0, 2, 3]);  mul_904 = None
    mul_906: "f32[192]" = torch.ops.aten.mul.Tensor(sum_109, 0.00015943877551020407)
    mul_907: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_908: "f32[192]" = torch.ops.aten.mul.Tensor(mul_906, mul_907);  mul_906 = mul_907 = None
    unsqueeze_886: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
    unsqueeze_887: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
    unsqueeze_888: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
    mul_909: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_889: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
    unsqueeze_890: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    mul_910: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_888);  sub_273 = unsqueeze_888 = None
    sub_275: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_18, mul_910);  where_18 = mul_910 = None
    sub_276: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_873);  sub_275 = unsqueeze_873 = None
    mul_911: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_891);  sub_276 = unsqueeze_891 = None
    mul_912: "f32[192]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_22);  sum_109 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_911, relu_2, primals_129, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_911 = primals_129 = None
    getitem_233: "f32[8, 96, 56, 56]" = convolution_backward_37[0]
    getitem_234: "f32[192, 96, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_378: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(getitem_230, getitem_233);  getitem_230 = getitem_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_80: "f32[8, 96, 56, 56]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_81: "f32[8, 96, 56, 56]" = torch.ops.aten.alias.default(alias_80);  alias_80 = None
    le_19: "b8[8, 96, 56, 56]" = torch.ops.aten.le.Scalar(alias_81, 0);  alias_81 = None
    where_19: "f32[8, 96, 56, 56]" = torch.ops.aten.where.self(le_19, full_default, add_378);  le_19 = add_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_110: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_277: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_894);  convolution_5 = unsqueeze_894 = None
    mul_913: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_19, sub_277)
    sum_111: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_913, [0, 2, 3]);  mul_913 = None
    mul_914: "f32[96]" = torch.ops.aten.mul.Tensor(sum_110, 3.985969387755102e-05)
    unsqueeze_895: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_914, 0);  mul_914 = None
    unsqueeze_896: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
    unsqueeze_897: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
    mul_915: "f32[96]" = torch.ops.aten.mul.Tensor(sum_111, 3.985969387755102e-05)
    mul_916: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_917: "f32[96]" = torch.ops.aten.mul.Tensor(mul_915, mul_916);  mul_915 = mul_916 = None
    unsqueeze_898: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
    unsqueeze_899: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
    unsqueeze_900: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
    mul_918: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_901: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_902: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    mul_919: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_900);  sub_277 = unsqueeze_900 = None
    sub_279: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(where_19, mul_919);  mul_919 = None
    sub_280: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_897);  sub_279 = None
    mul_920: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_903);  sub_280 = unsqueeze_903 = None
    mul_921: "f32[96]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_19);  sum_111 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_920, relu_1, primals_128, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_920 = primals_128 = None
    getitem_236: "f32[8, 96, 56, 56]" = convolution_backward_38[0]
    getitem_237: "f32[96, 96, 3, 3]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_281: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_906);  convolution_4 = unsqueeze_906 = None
    mul_922: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_19, sub_281)
    sum_113: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_922, [0, 2, 3]);  mul_922 = None
    mul_924: "f32[96]" = torch.ops.aten.mul.Tensor(sum_113, 3.985969387755102e-05)
    mul_925: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_926: "f32[96]" = torch.ops.aten.mul.Tensor(mul_924, mul_925);  mul_924 = mul_925 = None
    unsqueeze_910: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_926, 0);  mul_926 = None
    unsqueeze_911: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
    unsqueeze_912: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
    mul_927: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_913: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
    unsqueeze_914: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    mul_928: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_912);  sub_281 = unsqueeze_912 = None
    sub_283: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(where_19, mul_928);  mul_928 = None
    sub_284: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_897);  sub_283 = None
    mul_929: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_915);  sub_284 = unsqueeze_915 = None
    mul_930: "f32[96]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_16);  sum_113 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_929, relu_1, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_929 = primals_127 = None
    getitem_239: "f32[8, 96, 56, 56]" = convolution_backward_39[0]
    getitem_240: "f32[96, 96, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_379: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(getitem_236, getitem_239);  getitem_236 = getitem_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_285: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(relu_1, unsqueeze_918);  unsqueeze_918 = None
    mul_931: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_19, sub_285)
    sum_115: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_931, [0, 2, 3]);  mul_931 = None
    mul_933: "f32[96]" = torch.ops.aten.mul.Tensor(sum_115, 3.985969387755102e-05)
    mul_934: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_935: "f32[96]" = torch.ops.aten.mul.Tensor(mul_933, mul_934);  mul_933 = mul_934 = None
    unsqueeze_922: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
    unsqueeze_923: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
    unsqueeze_924: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
    mul_936: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_925: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
    unsqueeze_926: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    mul_937: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_924);  sub_285 = unsqueeze_924 = None
    sub_287: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(where_19, mul_937);  where_19 = mul_937 = None
    sub_288: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_897);  sub_287 = unsqueeze_897 = None
    mul_938: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_927);  sub_288 = unsqueeze_927 = None
    mul_939: "f32[96]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_13);  sum_115 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    add_380: "f32[8, 96, 56, 56]" = torch.ops.aten.add.Tensor(add_379, mul_938);  add_379 = mul_938 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_83: "f32[8, 96, 56, 56]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_84: "f32[8, 96, 56, 56]" = torch.ops.aten.alias.default(alias_83);  alias_83 = None
    le_20: "b8[8, 96, 56, 56]" = torch.ops.aten.le.Scalar(alias_84, 0);  alias_84 = None
    where_20: "f32[8, 96, 56, 56]" = torch.ops.aten.where.self(le_20, full_default, add_380);  le_20 = add_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_116: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_289: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_930);  convolution_3 = unsqueeze_930 = None
    mul_940: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_20, sub_289)
    sum_117: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_940, [0, 2, 3]);  mul_940 = None
    mul_941: "f32[96]" = torch.ops.aten.mul.Tensor(sum_116, 3.985969387755102e-05)
    unsqueeze_931: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_941, 0);  mul_941 = None
    unsqueeze_932: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 2);  unsqueeze_931 = None
    unsqueeze_933: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 3);  unsqueeze_932 = None
    mul_942: "f32[96]" = torch.ops.aten.mul.Tensor(sum_117, 3.985969387755102e-05)
    mul_943: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_944: "f32[96]" = torch.ops.aten.mul.Tensor(mul_942, mul_943);  mul_942 = mul_943 = None
    unsqueeze_934: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
    unsqueeze_935: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 2);  unsqueeze_934 = None
    unsqueeze_936: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 3);  unsqueeze_935 = None
    mul_945: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_937: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_945, 0);  mul_945 = None
    unsqueeze_938: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
    unsqueeze_939: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
    mul_946: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_936);  sub_289 = unsqueeze_936 = None
    sub_291: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(where_20, mul_946);  mul_946 = None
    sub_292: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_933);  sub_291 = None
    mul_947: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_939);  sub_292 = unsqueeze_939 = None
    mul_948: "f32[96]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_10);  sum_117 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_947, relu, primals_126, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_947 = primals_126 = None
    getitem_242: "f32[8, 64, 112, 112]" = convolution_backward_40[0]
    getitem_243: "f32[96, 64, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_293: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_942);  convolution_2 = unsqueeze_942 = None
    mul_949: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_20, sub_293)
    sum_119: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_949, [0, 2, 3]);  mul_949 = None
    mul_951: "f32[96]" = torch.ops.aten.mul.Tensor(sum_119, 3.985969387755102e-05)
    mul_952: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_953: "f32[96]" = torch.ops.aten.mul.Tensor(mul_951, mul_952);  mul_951 = mul_952 = None
    unsqueeze_946: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
    unsqueeze_947: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 2);  unsqueeze_946 = None
    unsqueeze_948: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 3);  unsqueeze_947 = None
    mul_954: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_949: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
    unsqueeze_950: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
    unsqueeze_951: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
    mul_955: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_948);  sub_293 = unsqueeze_948 = None
    sub_295: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(where_20, mul_955);  where_20 = mul_955 = None
    sub_296: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_933);  sub_295 = unsqueeze_933 = None
    mul_956: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_951);  sub_296 = unsqueeze_951 = None
    mul_957: "f32[96]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_7);  sum_119 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_956, relu, primals_125, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_956 = primals_125 = None
    getitem_245: "f32[8, 64, 112, 112]" = convolution_backward_41[0]
    getitem_246: "f32[96, 64, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_381: "f32[8, 64, 112, 112]" = torch.ops.aten.add.Tensor(getitem_242, getitem_245);  getitem_242 = getitem_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:541, code: return self.act(x)
    alias_86: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_87: "f32[8, 64, 112, 112]" = torch.ops.aten.alias.default(alias_86);  alias_86 = None
    le_21: "b8[8, 64, 112, 112]" = torch.ops.aten.le.Scalar(alias_87, 0);  alias_87 = None
    where_21: "f32[8, 64, 112, 112]" = torch.ops.aten.where.self(le_21, full_default, add_381);  le_21 = full_default = add_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_120: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_297: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_954);  convolution_1 = unsqueeze_954 = None
    mul_958: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_21, sub_297)
    sum_121: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_958, [0, 2, 3]);  mul_958 = None
    mul_959: "f32[64]" = torch.ops.aten.mul.Tensor(sum_120, 9.964923469387754e-06)
    unsqueeze_955: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_959, 0);  mul_959 = None
    unsqueeze_956: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 2);  unsqueeze_955 = None
    unsqueeze_957: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 3);  unsqueeze_956 = None
    mul_960: "f32[64]" = torch.ops.aten.mul.Tensor(sum_121, 9.964923469387754e-06)
    mul_961: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_962: "f32[64]" = torch.ops.aten.mul.Tensor(mul_960, mul_961);  mul_960 = mul_961 = None
    unsqueeze_958: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
    unsqueeze_959: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 2);  unsqueeze_958 = None
    unsqueeze_960: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 3);  unsqueeze_959 = None
    mul_963: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_961: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
    unsqueeze_962: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
    unsqueeze_963: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
    mul_964: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_960);  sub_297 = unsqueeze_960 = None
    sub_299: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_21, mul_964);  mul_964 = None
    sub_300: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_299, unsqueeze_957);  sub_299 = None
    mul_965: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_963);  sub_300 = unsqueeze_963 = None
    mul_966: "f32[64]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_4);  sum_121 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_965, primals_352, primals_124, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_965 = primals_124 = None
    getitem_249: "f32[64, 3, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_301: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_966);  convolution = unsqueeze_966 = None
    mul_967: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_21, sub_301)
    sum_123: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_967, [0, 2, 3]);  mul_967 = None
    mul_969: "f32[64]" = torch.ops.aten.mul.Tensor(sum_123, 9.964923469387754e-06)
    mul_970: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_971: "f32[64]" = torch.ops.aten.mul.Tensor(mul_969, mul_970);  mul_969 = mul_970 = None
    unsqueeze_970: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
    unsqueeze_971: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 2);  unsqueeze_970 = None
    unsqueeze_972: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 3);  unsqueeze_971 = None
    mul_972: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_973: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
    unsqueeze_974: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
    unsqueeze_975: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
    mul_973: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_972);  sub_301 = unsqueeze_972 = None
    sub_303: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(where_21, mul_973);  where_21 = mul_973 = None
    sub_304: "f32[8, 64, 112, 112]" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_957);  sub_303 = unsqueeze_957 = None
    mul_974: "f32[8, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_975);  sub_304 = unsqueeze_975 = None
    mul_975: "f32[64]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_1);  sum_123 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_974, primals_352, primals_123, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_974 = primals_352 = primals_123 = None
    getitem_252: "f32[64, 3, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    return [mul_975, sum_120, mul_966, sum_120, mul_957, sum_116, mul_948, sum_116, mul_939, sum_110, mul_930, sum_110, mul_921, sum_110, mul_912, sum_106, mul_903, sum_106, mul_894, sum_100, mul_885, sum_100, mul_876, sum_100, mul_867, sum_94, mul_858, sum_94, mul_849, sum_94, mul_840, sum_88, mul_831, sum_88, mul_822, sum_88, mul_813, sum_84, mul_804, sum_84, mul_795, sum_78, mul_786, sum_78, mul_777, sum_78, mul_768, sum_72, mul_759, sum_72, mul_750, sum_72, mul_741, sum_66, mul_732, sum_66, mul_723, sum_66, mul_714, sum_60, mul_705, sum_60, mul_696, sum_60, mul_687, sum_54, mul_678, sum_54, mul_669, sum_54, mul_660, sum_48, mul_651, sum_48, mul_642, sum_48, mul_633, sum_42, mul_624, sum_42, mul_615, sum_42, mul_606, sum_36, mul_597, sum_36, mul_588, sum_36, mul_579, sum_30, mul_570, sum_30, mul_561, sum_30, mul_552, sum_24, mul_543, sum_24, mul_534, sum_24, mul_525, sum_18, mul_516, sum_18, mul_507, sum_18, mul_498, sum_12, mul_489, sum_12, mul_480, sum_12, mul_471, sum_6, mul_462, sum_6, mul_453, sum_6, mul_444, sum_2, mul_435, sum_2, getitem_252, getitem_249, getitem_246, getitem_243, getitem_240, getitem_237, getitem_234, getitem_231, getitem_228, getitem_225, getitem_222, getitem_219, getitem_216, getitem_213, getitem_210, getitem_207, getitem_204, getitem_201, getitem_198, getitem_195, getitem_192, getitem_189, getitem_186, getitem_183, getitem_180, getitem_177, getitem_174, getitem_171, getitem_168, getitem_165, getitem_162, getitem_159, getitem_156, getitem_153, getitem_150, getitem_147, getitem_144, getitem_141, getitem_138, getitem_135, getitem_132, getitem_129, getitem_126, getitem_123, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    