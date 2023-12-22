from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_3: "f32[128]", primals_5: "f32[128]", primals_7: "f32[128]", primals_9: "f32[192]", primals_11: "f32[192]", primals_13: "f32[192]", primals_15: "f32[192]", primals_17: "f32[192]", primals_19: "f32[160]", primals_21: "f32[160]", primals_23: "f32[640]", primals_25: "f32[640]", primals_27: "f32[160]", primals_29: "f32[160]", primals_31: "f32[640]", primals_33: "f32[160]", primals_35: "f32[160]", primals_37: "f32[640]", primals_39: "f32[160]", primals_41: "f32[160]", primals_43: "f32[640]", primals_45: "f32[160]", primals_47: "f32[160]", primals_49: "f32[640]", primals_51: "f32[160]", primals_53: "f32[160]", primals_55: "f32[640]", primals_57: "f32[1920]", primals_59: "f32[1920]", primals_61: "f32[640]", primals_63: "f32[640]", primals_65: "f32[1920]", primals_67: "f32[1920]", primals_69: "f32[640]", primals_71: "f32[1920]", primals_73: "f32[1920]", primals_75: "f32[640]", primals_77: "f32[1920]", primals_79: "f32[1920]", primals_81: "f32[640]", primals_83: "f32[1920]", primals_85: "f32[1920]", primals_87: "f32[640]", primals_89: "f32[1920]", primals_91: "f32[1920]", primals_93: "f32[640]", primals_95: "f32[1920]", primals_97: "f32[1920]", primals_99: "f32[640]", primals_101: "f32[1920]", primals_103: "f32[1920]", primals_105: "f32[640]", primals_107: "f32[1920]", primals_109: "f32[1920]", primals_111: "f32[640]", primals_113: "f32[2560]", primals_115: "f32[32, 3, 3, 3]", primals_116: "f32[128, 32, 3, 3]", primals_117: "f32[128, 128, 3, 3]", primals_118: "f32[128, 32, 1, 1]", primals_119: "f32[192, 128, 3, 3]", primals_120: "f32[192, 192, 3, 3]", primals_121: "f32[192, 128, 1, 1]", primals_122: "f32[192, 192, 3, 3]", primals_123: "f32[192, 192, 3, 3]", primals_124: "f32[160, 192, 1, 1]", primals_125: "f32[160, 160, 3, 3]", primals_126: "f32[640, 160, 1, 1]", primals_127: "f32[640, 192, 1, 1]", primals_128: "f32[160, 640, 1, 1]", primals_129: "f32[160, 160, 3, 3]", primals_130: "f32[640, 160, 1, 1]", primals_131: "f32[160, 640, 1, 1]", primals_132: "f32[160, 160, 3, 3]", primals_133: "f32[640, 160, 1, 1]", primals_134: "f32[160, 640, 1, 1]", primals_135: "f32[160, 160, 3, 3]", primals_136: "f32[640, 160, 1, 1]", primals_137: "f32[160, 640, 1, 1]", primals_138: "f32[160, 160, 3, 3]", primals_139: "f32[640, 160, 1, 1]", primals_140: "f32[160, 640, 1, 1]", primals_141: "f32[160, 160, 3, 3]", primals_142: "f32[640, 160, 1, 1]", primals_143: "f32[1920, 640, 1, 1]", primals_144: "f32[1920, 1, 3, 3]", primals_145: "f32[640, 1920, 1, 1]", primals_146: "f32[640, 640, 1, 1]", primals_147: "f32[1920, 640, 1, 1]", primals_148: "f32[1920, 1, 3, 3]", primals_149: "f32[640, 1920, 1, 1]", primals_150: "f32[1920, 640, 1, 1]", primals_151: "f32[1920, 1, 3, 3]", primals_152: "f32[640, 1920, 1, 1]", primals_153: "f32[1920, 640, 1, 1]", primals_154: "f32[1920, 1, 3, 3]", primals_155: "f32[640, 1920, 1, 1]", primals_156: "f32[1920, 640, 1, 1]", primals_157: "f32[1920, 1, 3, 3]", primals_158: "f32[640, 1920, 1, 1]", primals_159: "f32[1920, 640, 1, 1]", primals_160: "f32[1920, 1, 3, 3]", primals_161: "f32[640, 1920, 1, 1]", primals_162: "f32[1920, 640, 1, 1]", primals_163: "f32[1920, 1, 3, 3]", primals_164: "f32[640, 1920, 1, 1]", primals_165: "f32[1920, 640, 1, 1]", primals_166: "f32[1920, 1, 3, 3]", primals_167: "f32[640, 1920, 1, 1]", primals_168: "f32[1920, 640, 1, 1]", primals_169: "f32[1920, 1, 3, 3]", primals_170: "f32[640, 1920, 1, 1]", primals_171: "f32[2560, 640, 1, 1]", primals_345: "f32[8, 3, 256, 256]", convolution: "f32[8, 32, 128, 128]", squeeze_1: "f32[32]", relu: "f32[8, 32, 128, 128]", convolution_1: "f32[8, 128, 64, 64]", squeeze_4: "f32[128]", relu_1: "f32[8, 128, 64, 64]", convolution_2: "f32[8, 128, 64, 64]", squeeze_7: "f32[128]", convolution_3: "f32[8, 128, 64, 64]", squeeze_10: "f32[128]", relu_2: "f32[8, 128, 64, 64]", convolution_4: "f32[8, 192, 32, 32]", squeeze_13: "f32[192]", relu_3: "f32[8, 192, 32, 32]", convolution_5: "f32[8, 192, 32, 32]", squeeze_16: "f32[192]", convolution_6: "f32[8, 192, 32, 32]", squeeze_19: "f32[192]", relu_4: "f32[8, 192, 32, 32]", convolution_7: "f32[8, 192, 32, 32]", squeeze_22: "f32[192]", relu_5: "f32[8, 192, 32, 32]", convolution_8: "f32[8, 192, 32, 32]", squeeze_25: "f32[192]", relu_6: "f32[8, 192, 32, 32]", convolution_9: "f32[8, 160, 32, 32]", squeeze_28: "f32[160]", relu_7: "f32[8, 160, 32, 32]", convolution_10: "f32[8, 160, 16, 16]", squeeze_31: "f32[160]", relu_8: "f32[8, 160, 16, 16]", convolution_11: "f32[8, 640, 16, 16]", squeeze_34: "f32[640]", convolution_12: "f32[8, 640, 16, 16]", squeeze_37: "f32[640]", relu_9: "f32[8, 640, 16, 16]", convolution_13: "f32[8, 160, 16, 16]", squeeze_40: "f32[160]", relu_10: "f32[8, 160, 16, 16]", convolution_14: "f32[8, 160, 16, 16]", squeeze_43: "f32[160]", relu_11: "f32[8, 160, 16, 16]", convolution_15: "f32[8, 640, 16, 16]", squeeze_46: "f32[640]", relu_12: "f32[8, 640, 16, 16]", convolution_16: "f32[8, 160, 16, 16]", squeeze_49: "f32[160]", relu_13: "f32[8, 160, 16, 16]", convolution_17: "f32[8, 160, 16, 16]", squeeze_52: "f32[160]", relu_14: "f32[8, 160, 16, 16]", convolution_18: "f32[8, 640, 16, 16]", squeeze_55: "f32[640]", relu_15: "f32[8, 640, 16, 16]", convolution_19: "f32[8, 160, 16, 16]", squeeze_58: "f32[160]", relu_16: "f32[8, 160, 16, 16]", convolution_20: "f32[8, 160, 16, 16]", squeeze_61: "f32[160]", relu_17: "f32[8, 160, 16, 16]", convolution_21: "f32[8, 640, 16, 16]", squeeze_64: "f32[640]", relu_18: "f32[8, 640, 16, 16]", convolution_22: "f32[8, 160, 16, 16]", squeeze_67: "f32[160]", relu_19: "f32[8, 160, 16, 16]", convolution_23: "f32[8, 160, 16, 16]", squeeze_70: "f32[160]", relu_20: "f32[8, 160, 16, 16]", convolution_24: "f32[8, 640, 16, 16]", squeeze_73: "f32[640]", relu_21: "f32[8, 640, 16, 16]", convolution_25: "f32[8, 160, 16, 16]", squeeze_76: "f32[160]", relu_22: "f32[8, 160, 16, 16]", convolution_26: "f32[8, 160, 16, 16]", squeeze_79: "f32[160]", relu_23: "f32[8, 160, 16, 16]", convolution_27: "f32[8, 640, 16, 16]", squeeze_82: "f32[640]", relu_24: "f32[8, 640, 16, 16]", convolution_28: "f32[8, 1920, 16, 16]", squeeze_85: "f32[1920]", relu_25: "f32[8, 1920, 16, 16]", convolution_29: "f32[8, 1920, 8, 8]", squeeze_88: "f32[1920]", relu_26: "f32[8, 1920, 8, 8]", convolution_30: "f32[8, 640, 8, 8]", squeeze_91: "f32[640]", convolution_31: "f32[8, 640, 8, 8]", squeeze_94: "f32[640]", relu_27: "f32[8, 640, 8, 8]", convolution_32: "f32[8, 1920, 8, 8]", squeeze_97: "f32[1920]", relu_28: "f32[8, 1920, 8, 8]", convolution_33: "f32[8, 1920, 8, 8]", squeeze_100: "f32[1920]", relu_29: "f32[8, 1920, 8, 8]", convolution_34: "f32[8, 640, 8, 8]", squeeze_103: "f32[640]", relu_30: "f32[8, 640, 8, 8]", convolution_35: "f32[8, 1920, 8, 8]", squeeze_106: "f32[1920]", relu_31: "f32[8, 1920, 8, 8]", convolution_36: "f32[8, 1920, 8, 8]", squeeze_109: "f32[1920]", relu_32: "f32[8, 1920, 8, 8]", convolution_37: "f32[8, 640, 8, 8]", squeeze_112: "f32[640]", relu_33: "f32[8, 640, 8, 8]", convolution_38: "f32[8, 1920, 8, 8]", squeeze_115: "f32[1920]", relu_34: "f32[8, 1920, 8, 8]", convolution_39: "f32[8, 1920, 8, 8]", squeeze_118: "f32[1920]", relu_35: "f32[8, 1920, 8, 8]", convolution_40: "f32[8, 640, 8, 8]", squeeze_121: "f32[640]", relu_36: "f32[8, 640, 8, 8]", convolution_41: "f32[8, 1920, 8, 8]", squeeze_124: "f32[1920]", relu_37: "f32[8, 1920, 8, 8]", convolution_42: "f32[8, 1920, 8, 8]", squeeze_127: "f32[1920]", relu_38: "f32[8, 1920, 8, 8]", convolution_43: "f32[8, 640, 8, 8]", squeeze_130: "f32[640]", relu_39: "f32[8, 640, 8, 8]", convolution_44: "f32[8, 1920, 8, 8]", squeeze_133: "f32[1920]", relu_40: "f32[8, 1920, 8, 8]", convolution_45: "f32[8, 1920, 8, 8]", squeeze_136: "f32[1920]", relu_41: "f32[8, 1920, 8, 8]", convolution_46: "f32[8, 640, 8, 8]", squeeze_139: "f32[640]", relu_42: "f32[8, 640, 8, 8]", convolution_47: "f32[8, 1920, 8, 8]", squeeze_142: "f32[1920]", relu_43: "f32[8, 1920, 8, 8]", convolution_48: "f32[8, 1920, 8, 8]", squeeze_145: "f32[1920]", relu_44: "f32[8, 1920, 8, 8]", convolution_49: "f32[8, 640, 8, 8]", squeeze_148: "f32[640]", relu_45: "f32[8, 640, 8, 8]", convolution_50: "f32[8, 1920, 8, 8]", squeeze_151: "f32[1920]", relu_46: "f32[8, 1920, 8, 8]", convolution_51: "f32[8, 1920, 8, 8]", squeeze_154: "f32[1920]", relu_47: "f32[8, 1920, 8, 8]", convolution_52: "f32[8, 640, 8, 8]", squeeze_157: "f32[640]", relu_48: "f32[8, 640, 8, 8]", convolution_53: "f32[8, 1920, 8, 8]", squeeze_160: "f32[1920]", relu_49: "f32[8, 1920, 8, 8]", convolution_54: "f32[8, 1920, 8, 8]", squeeze_163: "f32[1920]", relu_50: "f32[8, 1920, 8, 8]", convolution_55: "f32[8, 640, 8, 8]", squeeze_166: "f32[640]", relu_51: "f32[8, 640, 8, 8]", convolution_56: "f32[8, 2560, 8, 8]", squeeze_169: "f32[2560]", clone: "f32[8, 2560]", permute_1: "f32[1000, 2560]", le: "b8[8, 2560, 8, 8]", unsqueeze_230: "f32[1, 2560, 1, 1]", unsqueeze_242: "f32[1, 640, 1, 1]", unsqueeze_254: "f32[1, 1920, 1, 1]", unsqueeze_266: "f32[1, 1920, 1, 1]", unsqueeze_278: "f32[1, 640, 1, 1]", unsqueeze_290: "f32[1, 1920, 1, 1]", unsqueeze_302: "f32[1, 1920, 1, 1]", unsqueeze_314: "f32[1, 640, 1, 1]", unsqueeze_326: "f32[1, 1920, 1, 1]", unsqueeze_338: "f32[1, 1920, 1, 1]", unsqueeze_350: "f32[1, 640, 1, 1]", unsqueeze_362: "f32[1, 1920, 1, 1]", unsqueeze_374: "f32[1, 1920, 1, 1]", unsqueeze_386: "f32[1, 640, 1, 1]", unsqueeze_398: "f32[1, 1920, 1, 1]", unsqueeze_410: "f32[1, 1920, 1, 1]", unsqueeze_422: "f32[1, 640, 1, 1]", unsqueeze_434: "f32[1, 1920, 1, 1]", unsqueeze_446: "f32[1, 1920, 1, 1]", unsqueeze_458: "f32[1, 640, 1, 1]", unsqueeze_470: "f32[1, 1920, 1, 1]", unsqueeze_482: "f32[1, 1920, 1, 1]", unsqueeze_494: "f32[1, 640, 1, 1]", unsqueeze_506: "f32[1, 1920, 1, 1]", unsqueeze_518: "f32[1, 1920, 1, 1]", unsqueeze_530: "f32[1, 640, 1, 1]", unsqueeze_542: "f32[1, 640, 1, 1]", unsqueeze_554: "f32[1, 1920, 1, 1]", unsqueeze_566: "f32[1, 1920, 1, 1]", unsqueeze_578: "f32[1, 640, 1, 1]", unsqueeze_590: "f32[1, 160, 1, 1]", unsqueeze_602: "f32[1, 160, 1, 1]", unsqueeze_614: "f32[1, 640, 1, 1]", unsqueeze_626: "f32[1, 160, 1, 1]", unsqueeze_638: "f32[1, 160, 1, 1]", unsqueeze_650: "f32[1, 640, 1, 1]", unsqueeze_662: "f32[1, 160, 1, 1]", unsqueeze_674: "f32[1, 160, 1, 1]", unsqueeze_686: "f32[1, 640, 1, 1]", unsqueeze_698: "f32[1, 160, 1, 1]", unsqueeze_710: "f32[1, 160, 1, 1]", unsqueeze_722: "f32[1, 640, 1, 1]", unsqueeze_734: "f32[1, 160, 1, 1]", unsqueeze_746: "f32[1, 160, 1, 1]", unsqueeze_758: "f32[1, 640, 1, 1]", unsqueeze_770: "f32[1, 640, 1, 1]", unsqueeze_782: "f32[1, 160, 1, 1]", unsqueeze_794: "f32[1, 160, 1, 1]", unsqueeze_806: "f32[1, 192, 1, 1]", unsqueeze_818: "f32[1, 192, 1, 1]", unsqueeze_830: "f32[1, 192, 1, 1]", unsqueeze_842: "f32[1, 192, 1, 1]", unsqueeze_854: "f32[1, 192, 1, 1]", unsqueeze_866: "f32[1, 128, 1, 1]", unsqueeze_878: "f32[1, 128, 1, 1]", unsqueeze_890: "f32[1, 128, 1, 1]", unsqueeze_902: "f32[1, 32, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm: "f32[8, 2560]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 2560]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[2560, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 2560]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 2560, 1, 1]" = torch.ops.aten.view.default(mm, [8, 2560, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 2560, 8, 8]" = torch.ops.aten.expand.default(view_2, [8, 2560, 8, 8]);  view_2 = None
    div: "f32[8, 2560, 8, 8]" = torch.ops.aten.div.Scalar(expand, 64);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 2560, 8, 8]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[2560]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_57: "f32[8, 2560, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_230);  convolution_56 = unsqueeze_230 = None
    mul_399: "f32[8, 2560, 8, 8]" = torch.ops.aten.mul.Tensor(where, sub_57)
    sum_3: "f32[2560]" = torch.ops.aten.sum.dim_IntList(mul_399, [0, 2, 3]);  mul_399 = None
    mul_400: "f32[2560]" = torch.ops.aten.mul.Tensor(sum_2, 0.001953125)
    unsqueeze_231: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(mul_400, 0);  mul_400 = None
    unsqueeze_232: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 2);  unsqueeze_231 = None
    unsqueeze_233: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, 3);  unsqueeze_232 = None
    mul_401: "f32[2560]" = torch.ops.aten.mul.Tensor(sum_3, 0.001953125)
    mul_402: "f32[2560]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_403: "f32[2560]" = torch.ops.aten.mul.Tensor(mul_401, mul_402);  mul_401 = mul_402 = None
    unsqueeze_234: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(mul_403, 0);  mul_403 = None
    unsqueeze_235: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 2);  unsqueeze_234 = None
    unsqueeze_236: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_235, 3);  unsqueeze_235 = None
    mul_404: "f32[2560]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_237: "f32[1, 2560]" = torch.ops.aten.unsqueeze.default(mul_404, 0);  mul_404 = None
    unsqueeze_238: "f32[1, 2560, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_237, 2);  unsqueeze_237 = None
    unsqueeze_239: "f32[1, 2560, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 3);  unsqueeze_238 = None
    mul_405: "f32[8, 2560, 8, 8]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_236);  sub_57 = unsqueeze_236 = None
    sub_59: "f32[8, 2560, 8, 8]" = torch.ops.aten.sub.Tensor(where, mul_405);  where = mul_405 = None
    sub_60: "f32[8, 2560, 8, 8]" = torch.ops.aten.sub.Tensor(sub_59, unsqueeze_233);  sub_59 = unsqueeze_233 = None
    mul_406: "f32[8, 2560, 8, 8]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_239);  sub_60 = unsqueeze_239 = None
    mul_407: "f32[2560]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_169);  sum_3 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_406, relu_51, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_406 = primals_171 = None
    getitem_114: "f32[8, 640, 8, 8]" = convolution_backward[0]
    getitem_115: "f32[2560, 640, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_57: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(relu_51);  relu_51 = None
    alias_58: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    le_1: "b8[8, 640, 8, 8]" = torch.ops.aten.le.Scalar(alias_58, 0);  alias_58 = None
    where_1: "f32[8, 640, 8, 8]" = torch.ops.aten.where.self(le_1, full_default, getitem_114);  le_1 = getitem_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_61: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_242);  convolution_55 = unsqueeze_242 = None
    mul_408: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(where_1, sub_61)
    sum_5: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 2, 3]);  mul_408 = None
    mul_409: "f32[640]" = torch.ops.aten.mul.Tensor(sum_4, 0.001953125)
    unsqueeze_243: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
    unsqueeze_244: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 2);  unsqueeze_243 = None
    unsqueeze_245: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, 3);  unsqueeze_244 = None
    mul_410: "f32[640]" = torch.ops.aten.mul.Tensor(sum_5, 0.001953125)
    mul_411: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_412: "f32[640]" = torch.ops.aten.mul.Tensor(mul_410, mul_411);  mul_410 = mul_411 = None
    unsqueeze_246: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_412, 0);  mul_412 = None
    unsqueeze_247: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 2);  unsqueeze_246 = None
    unsqueeze_248: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_247, 3);  unsqueeze_247 = None
    mul_413: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_249: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_413, 0);  mul_413 = None
    unsqueeze_250: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_249, 2);  unsqueeze_249 = None
    unsqueeze_251: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, 3);  unsqueeze_250 = None
    mul_414: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_248);  sub_61 = unsqueeze_248 = None
    sub_63: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(where_1, mul_414);  mul_414 = None
    sub_64: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(sub_63, unsqueeze_245);  sub_63 = unsqueeze_245 = None
    mul_415: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_251);  sub_64 = unsqueeze_251 = None
    mul_416: "f32[640]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_166);  sum_5 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_415, relu_50, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_415 = primals_170 = None
    getitem_117: "f32[8, 1920, 8, 8]" = convolution_backward_1[0]
    getitem_118: "f32[640, 1920, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_60: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_50);  relu_50 = None
    alias_61: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_2: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_61, 0);  alias_61 = None
    where_2: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_2, full_default, getitem_117);  le_2 = getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_6: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_65: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_254);  convolution_54 = unsqueeze_254 = None
    mul_417: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_2, sub_65)
    sum_7: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 2, 3]);  mul_417 = None
    mul_418: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_6, 0.001953125)
    unsqueeze_255: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
    unsqueeze_256: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 2);  unsqueeze_255 = None
    unsqueeze_257: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, 3);  unsqueeze_256 = None
    mul_419: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_7, 0.001953125)
    mul_420: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_421: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_419, mul_420);  mul_419 = mul_420 = None
    unsqueeze_258: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_421, 0);  mul_421 = None
    unsqueeze_259: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 2);  unsqueeze_258 = None
    unsqueeze_260: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 3);  unsqueeze_259 = None
    mul_422: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_261: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_422, 0);  mul_422 = None
    unsqueeze_262: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_261, 2);  unsqueeze_261 = None
    unsqueeze_263: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 3);  unsqueeze_262 = None
    mul_423: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_260);  sub_65 = unsqueeze_260 = None
    sub_67: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_2, mul_423);  where_2 = mul_423 = None
    sub_68: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_67, unsqueeze_257);  sub_67 = unsqueeze_257 = None
    mul_424: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_263);  sub_68 = unsqueeze_263 = None
    mul_425: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_163);  sum_7 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_424, relu_49, primals_169, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_424 = primals_169 = None
    getitem_120: "f32[8, 1920, 8, 8]" = convolution_backward_2[0]
    getitem_121: "f32[1920, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_63: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_49);  relu_49 = None
    alias_64: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    le_3: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_64, 0);  alias_64 = None
    where_3: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_3, full_default, getitem_120);  le_3 = getitem_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_8: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_69: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_266);  convolution_53 = unsqueeze_266 = None
    mul_426: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_3, sub_69)
    sum_9: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 2, 3]);  mul_426 = None
    mul_427: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_8, 0.001953125)
    unsqueeze_267: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_427, 0);  mul_427 = None
    unsqueeze_268: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 2);  unsqueeze_267 = None
    unsqueeze_269: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, 3);  unsqueeze_268 = None
    mul_428: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_9, 0.001953125)
    mul_429: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_430: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_428, mul_429);  mul_428 = mul_429 = None
    unsqueeze_270: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_430, 0);  mul_430 = None
    unsqueeze_271: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 2);  unsqueeze_270 = None
    unsqueeze_272: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 3);  unsqueeze_271 = None
    mul_431: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_273: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_274: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_273, 2);  unsqueeze_273 = None
    unsqueeze_275: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 3);  unsqueeze_274 = None
    mul_432: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_272);  sub_69 = unsqueeze_272 = None
    sub_71: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_3, mul_432);  where_3 = mul_432 = None
    sub_72: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_71, unsqueeze_269);  sub_71 = unsqueeze_269 = None
    mul_433: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_275);  sub_72 = unsqueeze_275 = None
    mul_434: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_160);  sum_9 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_433, relu_48, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_433 = primals_168 = None
    getitem_123: "f32[8, 640, 8, 8]" = convolution_backward_3[0]
    getitem_124: "f32[1920, 640, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_303: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(where_1, getitem_123);  where_1 = getitem_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_66: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(relu_48);  relu_48 = None
    alias_67: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_4: "b8[8, 640, 8, 8]" = torch.ops.aten.le.Scalar(alias_67, 0);  alias_67 = None
    where_4: "f32[8, 640, 8, 8]" = torch.ops.aten.where.self(le_4, full_default, add_303);  le_4 = add_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_10: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_73: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_278);  convolution_52 = unsqueeze_278 = None
    mul_435: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(where_4, sub_73)
    sum_11: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_435, [0, 2, 3]);  mul_435 = None
    mul_436: "f32[640]" = torch.ops.aten.mul.Tensor(sum_10, 0.001953125)
    unsqueeze_279: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_436, 0);  mul_436 = None
    unsqueeze_280: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 2);  unsqueeze_279 = None
    unsqueeze_281: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 3);  unsqueeze_280 = None
    mul_437: "f32[640]" = torch.ops.aten.mul.Tensor(sum_11, 0.001953125)
    mul_438: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_439: "f32[640]" = torch.ops.aten.mul.Tensor(mul_437, mul_438);  mul_437 = mul_438 = None
    unsqueeze_282: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_439, 0);  mul_439 = None
    unsqueeze_283: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 2);  unsqueeze_282 = None
    unsqueeze_284: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 3);  unsqueeze_283 = None
    mul_440: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_285: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_286: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_285, 2);  unsqueeze_285 = None
    unsqueeze_287: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 3);  unsqueeze_286 = None
    mul_441: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_284);  sub_73 = unsqueeze_284 = None
    sub_75: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(where_4, mul_441);  mul_441 = None
    sub_76: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_281);  sub_75 = unsqueeze_281 = None
    mul_442: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_287);  sub_76 = unsqueeze_287 = None
    mul_443: "f32[640]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_157);  sum_11 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_442, relu_47, primals_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_442 = primals_167 = None
    getitem_126: "f32[8, 1920, 8, 8]" = convolution_backward_4[0]
    getitem_127: "f32[640, 1920, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_69: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_47);  relu_47 = None
    alias_70: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    le_5: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_70, 0);  alias_70 = None
    where_5: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_5, full_default, getitem_126);  le_5 = getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_12: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_77: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_290);  convolution_51 = unsqueeze_290 = None
    mul_444: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_5, sub_77)
    sum_13: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 2, 3]);  mul_444 = None
    mul_445: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_12, 0.001953125)
    unsqueeze_291: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_445, 0);  mul_445 = None
    unsqueeze_292: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 2);  unsqueeze_291 = None
    unsqueeze_293: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 3);  unsqueeze_292 = None
    mul_446: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_13, 0.001953125)
    mul_447: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_448: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_446, mul_447);  mul_446 = mul_447 = None
    unsqueeze_294: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_448, 0);  mul_448 = None
    unsqueeze_295: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 2);  unsqueeze_294 = None
    unsqueeze_296: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 3);  unsqueeze_295 = None
    mul_449: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_297: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
    unsqueeze_298: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_297, 2);  unsqueeze_297 = None
    unsqueeze_299: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 3);  unsqueeze_298 = None
    mul_450: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_296);  sub_77 = unsqueeze_296 = None
    sub_79: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_5, mul_450);  where_5 = mul_450 = None
    sub_80: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_293);  sub_79 = unsqueeze_293 = None
    mul_451: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_299);  sub_80 = unsqueeze_299 = None
    mul_452: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_154);  sum_13 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_451, relu_46, primals_166, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_451 = primals_166 = None
    getitem_129: "f32[8, 1920, 8, 8]" = convolution_backward_5[0]
    getitem_130: "f32[1920, 1, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_72: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_46);  relu_46 = None
    alias_73: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    le_6: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_73, 0);  alias_73 = None
    where_6: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_6, full_default, getitem_129);  le_6 = getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_14: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_81: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_302);  convolution_50 = unsqueeze_302 = None
    mul_453: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_6, sub_81)
    sum_15: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_453, [0, 2, 3]);  mul_453 = None
    mul_454: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_14, 0.001953125)
    unsqueeze_303: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_454, 0);  mul_454 = None
    unsqueeze_304: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 2);  unsqueeze_303 = None
    unsqueeze_305: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, 3);  unsqueeze_304 = None
    mul_455: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_15, 0.001953125)
    mul_456: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_457: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_455, mul_456);  mul_455 = mul_456 = None
    unsqueeze_306: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_457, 0);  mul_457 = None
    unsqueeze_307: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 2);  unsqueeze_306 = None
    unsqueeze_308: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 3);  unsqueeze_307 = None
    mul_458: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_309: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_310: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 2);  unsqueeze_309 = None
    unsqueeze_311: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 3);  unsqueeze_310 = None
    mul_459: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_308);  sub_81 = unsqueeze_308 = None
    sub_83: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_6, mul_459);  where_6 = mul_459 = None
    sub_84: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_305);  sub_83 = unsqueeze_305 = None
    mul_460: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_311);  sub_84 = unsqueeze_311 = None
    mul_461: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_151);  sum_15 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_460, relu_45, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_460 = primals_165 = None
    getitem_132: "f32[8, 640, 8, 8]" = convolution_backward_6[0]
    getitem_133: "f32[1920, 640, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_304: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(where_4, getitem_132);  where_4 = getitem_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_75: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(relu_45);  relu_45 = None
    alias_76: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(alias_75);  alias_75 = None
    le_7: "b8[8, 640, 8, 8]" = torch.ops.aten.le.Scalar(alias_76, 0);  alias_76 = None
    where_7: "f32[8, 640, 8, 8]" = torch.ops.aten.where.self(le_7, full_default, add_304);  le_7 = add_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_16: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_85: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_314);  convolution_49 = unsqueeze_314 = None
    mul_462: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(where_7, sub_85)
    sum_17: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_462, [0, 2, 3]);  mul_462 = None
    mul_463: "f32[640]" = torch.ops.aten.mul.Tensor(sum_16, 0.001953125)
    unsqueeze_315: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
    unsqueeze_316: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 2);  unsqueeze_315 = None
    unsqueeze_317: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, 3);  unsqueeze_316 = None
    mul_464: "f32[640]" = torch.ops.aten.mul.Tensor(sum_17, 0.001953125)
    mul_465: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_466: "f32[640]" = torch.ops.aten.mul.Tensor(mul_464, mul_465);  mul_464 = mul_465 = None
    unsqueeze_318: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_466, 0);  mul_466 = None
    unsqueeze_319: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 2);  unsqueeze_318 = None
    unsqueeze_320: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 3);  unsqueeze_319 = None
    mul_467: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_321: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
    unsqueeze_322: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_321, 2);  unsqueeze_321 = None
    unsqueeze_323: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 3);  unsqueeze_322 = None
    mul_468: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_320);  sub_85 = unsqueeze_320 = None
    sub_87: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(where_7, mul_468);  mul_468 = None
    sub_88: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_317);  sub_87 = unsqueeze_317 = None
    mul_469: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_323);  sub_88 = unsqueeze_323 = None
    mul_470: "f32[640]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_148);  sum_17 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_469, relu_44, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_469 = primals_164 = None
    getitem_135: "f32[8, 1920, 8, 8]" = convolution_backward_7[0]
    getitem_136: "f32[640, 1920, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_78: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_44);  relu_44 = None
    alias_79: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_78);  alias_78 = None
    le_8: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_79, 0);  alias_79 = None
    where_8: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_8, full_default, getitem_135);  le_8 = getitem_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_89: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_326);  convolution_48 = unsqueeze_326 = None
    mul_471: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_8, sub_89)
    sum_19: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_471, [0, 2, 3]);  mul_471 = None
    mul_472: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_18, 0.001953125)
    unsqueeze_327: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
    unsqueeze_328: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 2);  unsqueeze_327 = None
    unsqueeze_329: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 3);  unsqueeze_328 = None
    mul_473: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_19, 0.001953125)
    mul_474: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_475: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_473, mul_474);  mul_473 = mul_474 = None
    unsqueeze_330: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_475, 0);  mul_475 = None
    unsqueeze_331: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 2);  unsqueeze_330 = None
    unsqueeze_332: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 3);  unsqueeze_331 = None
    mul_476: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_333: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
    unsqueeze_334: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_333, 2);  unsqueeze_333 = None
    unsqueeze_335: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 3);  unsqueeze_334 = None
    mul_477: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_332);  sub_89 = unsqueeze_332 = None
    sub_91: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_8, mul_477);  where_8 = mul_477 = None
    sub_92: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_329);  sub_91 = unsqueeze_329 = None
    mul_478: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_335);  sub_92 = unsqueeze_335 = None
    mul_479: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_145);  sum_19 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_478, relu_43, primals_163, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_478 = primals_163 = None
    getitem_138: "f32[8, 1920, 8, 8]" = convolution_backward_8[0]
    getitem_139: "f32[1920, 1, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_81: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_43);  relu_43 = None
    alias_82: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    le_9: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_82, 0);  alias_82 = None
    where_9: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_9, full_default, getitem_138);  le_9 = getitem_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_20: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_93: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_338);  convolution_47 = unsqueeze_338 = None
    mul_480: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_9, sub_93)
    sum_21: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 2, 3]);  mul_480 = None
    mul_481: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_20, 0.001953125)
    unsqueeze_339: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
    unsqueeze_340: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 2);  unsqueeze_339 = None
    unsqueeze_341: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, 3);  unsqueeze_340 = None
    mul_482: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_21, 0.001953125)
    mul_483: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_484: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_482, mul_483);  mul_482 = mul_483 = None
    unsqueeze_342: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_484, 0);  mul_484 = None
    unsqueeze_343: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 2);  unsqueeze_342 = None
    unsqueeze_344: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 3);  unsqueeze_343 = None
    mul_485: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_345: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
    unsqueeze_346: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_345, 2);  unsqueeze_345 = None
    unsqueeze_347: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 3);  unsqueeze_346 = None
    mul_486: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_344);  sub_93 = unsqueeze_344 = None
    sub_95: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_9, mul_486);  where_9 = mul_486 = None
    sub_96: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_341);  sub_95 = unsqueeze_341 = None
    mul_487: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_347);  sub_96 = unsqueeze_347 = None
    mul_488: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_142);  sum_21 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_487, relu_42, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_487 = primals_162 = None
    getitem_141: "f32[8, 640, 8, 8]" = convolution_backward_9[0]
    getitem_142: "f32[1920, 640, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_305: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(where_7, getitem_141);  where_7 = getitem_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_84: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_85: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    le_10: "b8[8, 640, 8, 8]" = torch.ops.aten.le.Scalar(alias_85, 0);  alias_85 = None
    where_10: "f32[8, 640, 8, 8]" = torch.ops.aten.where.self(le_10, full_default, add_305);  le_10 = add_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_22: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_97: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_350);  convolution_46 = unsqueeze_350 = None
    mul_489: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(where_10, sub_97)
    sum_23: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_489, [0, 2, 3]);  mul_489 = None
    mul_490: "f32[640]" = torch.ops.aten.mul.Tensor(sum_22, 0.001953125)
    unsqueeze_351: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
    unsqueeze_352: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 2);  unsqueeze_351 = None
    unsqueeze_353: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, 3);  unsqueeze_352 = None
    mul_491: "f32[640]" = torch.ops.aten.mul.Tensor(sum_23, 0.001953125)
    mul_492: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_493: "f32[640]" = torch.ops.aten.mul.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
    unsqueeze_354: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_493, 0);  mul_493 = None
    unsqueeze_355: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 2);  unsqueeze_354 = None
    unsqueeze_356: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 3);  unsqueeze_355 = None
    mul_494: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_357: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_358: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_357, 2);  unsqueeze_357 = None
    unsqueeze_359: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 3);  unsqueeze_358 = None
    mul_495: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_356);  sub_97 = unsqueeze_356 = None
    sub_99: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(where_10, mul_495);  mul_495 = None
    sub_100: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_353);  sub_99 = unsqueeze_353 = None
    mul_496: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_359);  sub_100 = unsqueeze_359 = None
    mul_497: "f32[640]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_139);  sum_23 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_496, relu_41, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_496 = primals_161 = None
    getitem_144: "f32[8, 1920, 8, 8]" = convolution_backward_10[0]
    getitem_145: "f32[640, 1920, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_87: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_88: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_87);  alias_87 = None
    le_11: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_88, 0);  alias_88 = None
    where_11: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_11, full_default, getitem_144);  le_11 = getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_24: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_101: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_362);  convolution_45 = unsqueeze_362 = None
    mul_498: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_11, sub_101)
    sum_25: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_498, [0, 2, 3]);  mul_498 = None
    mul_499: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_24, 0.001953125)
    unsqueeze_363: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
    unsqueeze_364: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 2);  unsqueeze_363 = None
    unsqueeze_365: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, 3);  unsqueeze_364 = None
    mul_500: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_25, 0.001953125)
    mul_501: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_502: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    unsqueeze_366: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
    unsqueeze_367: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 2);  unsqueeze_366 = None
    unsqueeze_368: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 3);  unsqueeze_367 = None
    mul_503: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_369: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    unsqueeze_370: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 2);  unsqueeze_369 = None
    unsqueeze_371: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 3);  unsqueeze_370 = None
    mul_504: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_368);  sub_101 = unsqueeze_368 = None
    sub_103: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_11, mul_504);  where_11 = mul_504 = None
    sub_104: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_365);  sub_103 = unsqueeze_365 = None
    mul_505: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_371);  sub_104 = unsqueeze_371 = None
    mul_506: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_136);  sum_25 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_505, relu_40, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_505 = primals_160 = None
    getitem_147: "f32[8, 1920, 8, 8]" = convolution_backward_11[0]
    getitem_148: "f32[1920, 1, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_90: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
    alias_91: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_90);  alias_90 = None
    le_12: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_91, 0);  alias_91 = None
    where_12: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_12, full_default, getitem_147);  le_12 = getitem_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_26: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_105: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_374);  convolution_44 = unsqueeze_374 = None
    mul_507: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_12, sub_105)
    sum_27: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
    mul_508: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_26, 0.001953125)
    unsqueeze_375: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_376: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 2);  unsqueeze_375 = None
    unsqueeze_377: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 3);  unsqueeze_376 = None
    mul_509: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_27, 0.001953125)
    mul_510: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_511: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
    unsqueeze_378: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
    unsqueeze_379: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 2);  unsqueeze_378 = None
    unsqueeze_380: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 3);  unsqueeze_379 = None
    mul_512: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_381: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    unsqueeze_382: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 2);  unsqueeze_381 = None
    unsqueeze_383: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 3);  unsqueeze_382 = None
    mul_513: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_380);  sub_105 = unsqueeze_380 = None
    sub_107: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_12, mul_513);  where_12 = mul_513 = None
    sub_108: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_377);  sub_107 = unsqueeze_377 = None
    mul_514: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_383);  sub_108 = unsqueeze_383 = None
    mul_515: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_133);  sum_27 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_514, relu_39, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_514 = primals_159 = None
    getitem_150: "f32[8, 640, 8, 8]" = convolution_backward_12[0]
    getitem_151: "f32[1920, 640, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_306: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(where_10, getitem_150);  where_10 = getitem_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_93: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_94: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(alias_93);  alias_93 = None
    le_13: "b8[8, 640, 8, 8]" = torch.ops.aten.le.Scalar(alias_94, 0);  alias_94 = None
    where_13: "f32[8, 640, 8, 8]" = torch.ops.aten.where.self(le_13, full_default, add_306);  le_13 = add_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_28: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_109: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_386);  convolution_43 = unsqueeze_386 = None
    mul_516: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(where_13, sub_109)
    sum_29: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 2, 3]);  mul_516 = None
    mul_517: "f32[640]" = torch.ops.aten.mul.Tensor(sum_28, 0.001953125)
    unsqueeze_387: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
    unsqueeze_388: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 2);  unsqueeze_387 = None
    unsqueeze_389: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 3);  unsqueeze_388 = None
    mul_518: "f32[640]" = torch.ops.aten.mul.Tensor(sum_29, 0.001953125)
    mul_519: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_520: "f32[640]" = torch.ops.aten.mul.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
    unsqueeze_390: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_391: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 2);  unsqueeze_390 = None
    unsqueeze_392: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 3);  unsqueeze_391 = None
    mul_521: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_393: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_394: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 2);  unsqueeze_393 = None
    unsqueeze_395: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 3);  unsqueeze_394 = None
    mul_522: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_392);  sub_109 = unsqueeze_392 = None
    sub_111: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(where_13, mul_522);  mul_522 = None
    sub_112: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_389);  sub_111 = unsqueeze_389 = None
    mul_523: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_395);  sub_112 = unsqueeze_395 = None
    mul_524: "f32[640]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_130);  sum_29 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_523, relu_38, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_523 = primals_158 = None
    getitem_153: "f32[8, 1920, 8, 8]" = convolution_backward_13[0]
    getitem_154: "f32[640, 1920, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_96: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_97: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_96);  alias_96 = None
    le_14: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_97, 0);  alias_97 = None
    where_14: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_14, full_default, getitem_153);  le_14 = getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_30: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_113: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_398);  convolution_42 = unsqueeze_398 = None
    mul_525: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_14, sub_113)
    sum_31: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_525, [0, 2, 3]);  mul_525 = None
    mul_526: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_30, 0.001953125)
    unsqueeze_399: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
    unsqueeze_400: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 2);  unsqueeze_399 = None
    unsqueeze_401: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, 3);  unsqueeze_400 = None
    mul_527: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_31, 0.001953125)
    mul_528: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_529: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_527, mul_528);  mul_527 = mul_528 = None
    unsqueeze_402: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_529, 0);  mul_529 = None
    unsqueeze_403: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 2);  unsqueeze_402 = None
    unsqueeze_404: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 3);  unsqueeze_403 = None
    mul_530: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_405: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
    unsqueeze_406: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 2);  unsqueeze_405 = None
    unsqueeze_407: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 3);  unsqueeze_406 = None
    mul_531: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_404);  sub_113 = unsqueeze_404 = None
    sub_115: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_14, mul_531);  where_14 = mul_531 = None
    sub_116: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_401);  sub_115 = unsqueeze_401 = None
    mul_532: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_407);  sub_116 = unsqueeze_407 = None
    mul_533: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_127);  sum_31 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_532, relu_37, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_532 = primals_157 = None
    getitem_156: "f32[8, 1920, 8, 8]" = convolution_backward_14[0]
    getitem_157: "f32[1920, 1, 3, 3]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_99: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_100: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    le_15: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_100, 0);  alias_100 = None
    where_15: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_15, full_default, getitem_156);  le_15 = getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_32: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_117: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_410);  convolution_41 = unsqueeze_410 = None
    mul_534: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_15, sub_117)
    sum_33: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_534, [0, 2, 3]);  mul_534 = None
    mul_535: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_32, 0.001953125)
    unsqueeze_411: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_535, 0);  mul_535 = None
    unsqueeze_412: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 2);  unsqueeze_411 = None
    unsqueeze_413: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, 3);  unsqueeze_412 = None
    mul_536: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_33, 0.001953125)
    mul_537: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_538: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_536, mul_537);  mul_536 = mul_537 = None
    unsqueeze_414: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_538, 0);  mul_538 = None
    unsqueeze_415: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 2);  unsqueeze_414 = None
    unsqueeze_416: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 3);  unsqueeze_415 = None
    mul_539: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_417: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_418: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 2);  unsqueeze_417 = None
    unsqueeze_419: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 3);  unsqueeze_418 = None
    mul_540: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_416);  sub_117 = unsqueeze_416 = None
    sub_119: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_15, mul_540);  where_15 = mul_540 = None
    sub_120: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_413);  sub_119 = unsqueeze_413 = None
    mul_541: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_419);  sub_120 = unsqueeze_419 = None
    mul_542: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_124);  sum_33 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_541, relu_36, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_541 = primals_156 = None
    getitem_159: "f32[8, 640, 8, 8]" = convolution_backward_15[0]
    getitem_160: "f32[1920, 640, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_307: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(where_13, getitem_159);  where_13 = getitem_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_102: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_103: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    le_16: "b8[8, 640, 8, 8]" = torch.ops.aten.le.Scalar(alias_103, 0);  alias_103 = None
    where_16: "f32[8, 640, 8, 8]" = torch.ops.aten.where.self(le_16, full_default, add_307);  le_16 = add_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_34: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_121: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_422);  convolution_40 = unsqueeze_422 = None
    mul_543: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(where_16, sub_121)
    sum_35: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_543, [0, 2, 3]);  mul_543 = None
    mul_544: "f32[640]" = torch.ops.aten.mul.Tensor(sum_34, 0.001953125)
    unsqueeze_423: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
    unsqueeze_424: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 2);  unsqueeze_423 = None
    unsqueeze_425: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, 3);  unsqueeze_424 = None
    mul_545: "f32[640]" = torch.ops.aten.mul.Tensor(sum_35, 0.001953125)
    mul_546: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_547: "f32[640]" = torch.ops.aten.mul.Tensor(mul_545, mul_546);  mul_545 = mul_546 = None
    unsqueeze_426: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_547, 0);  mul_547 = None
    unsqueeze_427: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 2);  unsqueeze_426 = None
    unsqueeze_428: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 3);  unsqueeze_427 = None
    mul_548: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_429: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_430: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 2);  unsqueeze_429 = None
    unsqueeze_431: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 3);  unsqueeze_430 = None
    mul_549: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_428);  sub_121 = unsqueeze_428 = None
    sub_123: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(where_16, mul_549);  mul_549 = None
    sub_124: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_425);  sub_123 = unsqueeze_425 = None
    mul_550: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_431);  sub_124 = unsqueeze_431 = None
    mul_551: "f32[640]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_121);  sum_35 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_550, relu_35, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_550 = primals_155 = None
    getitem_162: "f32[8, 1920, 8, 8]" = convolution_backward_16[0]
    getitem_163: "f32[640, 1920, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_105: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_106: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_105);  alias_105 = None
    le_17: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_106, 0);  alias_106 = None
    where_17: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_17, full_default, getitem_162);  le_17 = getitem_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_36: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_125: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_434);  convolution_39 = unsqueeze_434 = None
    mul_552: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_17, sub_125)
    sum_37: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_552, [0, 2, 3]);  mul_552 = None
    mul_553: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_36, 0.001953125)
    unsqueeze_435: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    unsqueeze_436: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 2);  unsqueeze_435 = None
    unsqueeze_437: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, 3);  unsqueeze_436 = None
    mul_554: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_37, 0.001953125)
    mul_555: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_556: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
    unsqueeze_438: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_556, 0);  mul_556 = None
    unsqueeze_439: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 2);  unsqueeze_438 = None
    unsqueeze_440: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 3);  unsqueeze_439 = None
    mul_557: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_441: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
    unsqueeze_442: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 2);  unsqueeze_441 = None
    unsqueeze_443: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 3);  unsqueeze_442 = None
    mul_558: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_440);  sub_125 = unsqueeze_440 = None
    sub_127: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_17, mul_558);  where_17 = mul_558 = None
    sub_128: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_437);  sub_127 = unsqueeze_437 = None
    mul_559: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_443);  sub_128 = unsqueeze_443 = None
    mul_560: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_118);  sum_37 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_559, relu_34, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_559 = primals_154 = None
    getitem_165: "f32[8, 1920, 8, 8]" = convolution_backward_17[0]
    getitem_166: "f32[1920, 1, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_108: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_109: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_108);  alias_108 = None
    le_18: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_109, 0);  alias_109 = None
    where_18: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_18, full_default, getitem_165);  le_18 = getitem_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_38: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_129: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_446);  convolution_38 = unsqueeze_446 = None
    mul_561: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_18, sub_129)
    sum_39: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_561, [0, 2, 3]);  mul_561 = None
    mul_562: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_38, 0.001953125)
    unsqueeze_447: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
    unsqueeze_448: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 2);  unsqueeze_447 = None
    unsqueeze_449: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, 3);  unsqueeze_448 = None
    mul_563: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_39, 0.001953125)
    mul_564: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_565: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_563, mul_564);  mul_563 = mul_564 = None
    unsqueeze_450: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
    unsqueeze_451: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 2);  unsqueeze_450 = None
    unsqueeze_452: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 3);  unsqueeze_451 = None
    mul_566: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_453: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    unsqueeze_454: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 2);  unsqueeze_453 = None
    unsqueeze_455: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 3);  unsqueeze_454 = None
    mul_567: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_452);  sub_129 = unsqueeze_452 = None
    sub_131: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_18, mul_567);  where_18 = mul_567 = None
    sub_132: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_449);  sub_131 = unsqueeze_449 = None
    mul_568: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_455);  sub_132 = unsqueeze_455 = None
    mul_569: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_115);  sum_39 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_568, relu_33, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_568 = primals_153 = None
    getitem_168: "f32[8, 640, 8, 8]" = convolution_backward_18[0]
    getitem_169: "f32[1920, 640, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_308: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(where_16, getitem_168);  where_16 = getitem_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_111: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_112: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(alias_111);  alias_111 = None
    le_19: "b8[8, 640, 8, 8]" = torch.ops.aten.le.Scalar(alias_112, 0);  alias_112 = None
    where_19: "f32[8, 640, 8, 8]" = torch.ops.aten.where.self(le_19, full_default, add_308);  le_19 = add_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_40: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_133: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_458);  convolution_37 = unsqueeze_458 = None
    mul_570: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(where_19, sub_133)
    sum_41: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_570, [0, 2, 3]);  mul_570 = None
    mul_571: "f32[640]" = torch.ops.aten.mul.Tensor(sum_40, 0.001953125)
    unsqueeze_459: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_571, 0);  mul_571 = None
    unsqueeze_460: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 2);  unsqueeze_459 = None
    unsqueeze_461: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, 3);  unsqueeze_460 = None
    mul_572: "f32[640]" = torch.ops.aten.mul.Tensor(sum_41, 0.001953125)
    mul_573: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_574: "f32[640]" = torch.ops.aten.mul.Tensor(mul_572, mul_573);  mul_572 = mul_573 = None
    unsqueeze_462: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
    unsqueeze_463: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 2);  unsqueeze_462 = None
    unsqueeze_464: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 3);  unsqueeze_463 = None
    mul_575: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_465: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
    unsqueeze_466: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 2);  unsqueeze_465 = None
    unsqueeze_467: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 3);  unsqueeze_466 = None
    mul_576: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_464);  sub_133 = unsqueeze_464 = None
    sub_135: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(where_19, mul_576);  mul_576 = None
    sub_136: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_461);  sub_135 = unsqueeze_461 = None
    mul_577: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_467);  sub_136 = unsqueeze_467 = None
    mul_578: "f32[640]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_112);  sum_41 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_577, relu_32, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_577 = primals_152 = None
    getitem_171: "f32[8, 1920, 8, 8]" = convolution_backward_19[0]
    getitem_172: "f32[640, 1920, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_114: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_115: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_114);  alias_114 = None
    le_20: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_115, 0);  alias_115 = None
    where_20: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_20, full_default, getitem_171);  le_20 = getitem_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_137: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_470);  convolution_36 = unsqueeze_470 = None
    mul_579: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_20, sub_137)
    sum_43: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_579, [0, 2, 3]);  mul_579 = None
    mul_580: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_42, 0.001953125)
    unsqueeze_471: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
    unsqueeze_472: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 2);  unsqueeze_471 = None
    unsqueeze_473: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, 3);  unsqueeze_472 = None
    mul_581: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_43, 0.001953125)
    mul_582: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_583: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_581, mul_582);  mul_581 = mul_582 = None
    unsqueeze_474: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
    unsqueeze_475: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 2);  unsqueeze_474 = None
    unsqueeze_476: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 3);  unsqueeze_475 = None
    mul_584: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_477: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
    unsqueeze_478: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 2);  unsqueeze_477 = None
    unsqueeze_479: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 3);  unsqueeze_478 = None
    mul_585: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_476);  sub_137 = unsqueeze_476 = None
    sub_139: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_20, mul_585);  where_20 = mul_585 = None
    sub_140: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_473);  sub_139 = unsqueeze_473 = None
    mul_586: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_479);  sub_140 = unsqueeze_479 = None
    mul_587: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_109);  sum_43 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_586, relu_31, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_586 = primals_151 = None
    getitem_174: "f32[8, 1920, 8, 8]" = convolution_backward_20[0]
    getitem_175: "f32[1920, 1, 3, 3]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_117: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_118: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_117);  alias_117 = None
    le_21: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_118, 0);  alias_118 = None
    where_21: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_21, full_default, getitem_174);  le_21 = getitem_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_44: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_141: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_482);  convolution_35 = unsqueeze_482 = None
    mul_588: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_21, sub_141)
    sum_45: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_588, [0, 2, 3]);  mul_588 = None
    mul_589: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_44, 0.001953125)
    unsqueeze_483: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
    unsqueeze_484: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 2);  unsqueeze_483 = None
    unsqueeze_485: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 3);  unsqueeze_484 = None
    mul_590: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_45, 0.001953125)
    mul_591: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_592: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_590, mul_591);  mul_590 = mul_591 = None
    unsqueeze_486: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
    unsqueeze_487: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 2);  unsqueeze_486 = None
    unsqueeze_488: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 3);  unsqueeze_487 = None
    mul_593: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_489: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_593, 0);  mul_593 = None
    unsqueeze_490: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 2);  unsqueeze_489 = None
    unsqueeze_491: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 3);  unsqueeze_490 = None
    mul_594: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_488);  sub_141 = unsqueeze_488 = None
    sub_143: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_21, mul_594);  where_21 = mul_594 = None
    sub_144: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_485);  sub_143 = unsqueeze_485 = None
    mul_595: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_491);  sub_144 = unsqueeze_491 = None
    mul_596: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_106);  sum_45 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_595, relu_30, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_595 = primals_150 = None
    getitem_177: "f32[8, 640, 8, 8]" = convolution_backward_21[0]
    getitem_178: "f32[1920, 640, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_309: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(where_19, getitem_177);  where_19 = getitem_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_120: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_121: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(alias_120);  alias_120 = None
    le_22: "b8[8, 640, 8, 8]" = torch.ops.aten.le.Scalar(alias_121, 0);  alias_121 = None
    where_22: "f32[8, 640, 8, 8]" = torch.ops.aten.where.self(le_22, full_default, add_309);  le_22 = add_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_46: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_145: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_494);  convolution_34 = unsqueeze_494 = None
    mul_597: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(where_22, sub_145)
    sum_47: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_597, [0, 2, 3]);  mul_597 = None
    mul_598: "f32[640]" = torch.ops.aten.mul.Tensor(sum_46, 0.001953125)
    unsqueeze_495: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
    unsqueeze_496: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 2);  unsqueeze_495 = None
    unsqueeze_497: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, 3);  unsqueeze_496 = None
    mul_599: "f32[640]" = torch.ops.aten.mul.Tensor(sum_47, 0.001953125)
    mul_600: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_601: "f32[640]" = torch.ops.aten.mul.Tensor(mul_599, mul_600);  mul_599 = mul_600 = None
    unsqueeze_498: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
    unsqueeze_499: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 2);  unsqueeze_498 = None
    unsqueeze_500: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 3);  unsqueeze_499 = None
    mul_602: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_501: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
    unsqueeze_502: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 2);  unsqueeze_501 = None
    unsqueeze_503: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 3);  unsqueeze_502 = None
    mul_603: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_500);  sub_145 = unsqueeze_500 = None
    sub_147: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(where_22, mul_603);  mul_603 = None
    sub_148: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_497);  sub_147 = unsqueeze_497 = None
    mul_604: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_503);  sub_148 = unsqueeze_503 = None
    mul_605: "f32[640]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_103);  sum_47 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_604, relu_29, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_604 = primals_149 = None
    getitem_180: "f32[8, 1920, 8, 8]" = convolution_backward_22[0]
    getitem_181: "f32[640, 1920, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_123: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_124: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_123);  alias_123 = None
    le_23: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_124, 0);  alias_124 = None
    where_23: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_23, full_default, getitem_180);  le_23 = getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_48: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_149: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_506);  convolution_33 = unsqueeze_506 = None
    mul_606: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_23, sub_149)
    sum_49: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_606, [0, 2, 3]);  mul_606 = None
    mul_607: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_48, 0.001953125)
    unsqueeze_507: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
    unsqueeze_508: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 2);  unsqueeze_507 = None
    unsqueeze_509: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, 3);  unsqueeze_508 = None
    mul_608: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_49, 0.001953125)
    mul_609: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_610: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_608, mul_609);  mul_608 = mul_609 = None
    unsqueeze_510: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
    unsqueeze_511: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 2);  unsqueeze_510 = None
    unsqueeze_512: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 3);  unsqueeze_511 = None
    mul_611: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_513: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
    unsqueeze_514: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 2);  unsqueeze_513 = None
    unsqueeze_515: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 3);  unsqueeze_514 = None
    mul_612: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_512);  sub_149 = unsqueeze_512 = None
    sub_151: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_23, mul_612);  where_23 = mul_612 = None
    sub_152: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_509);  sub_151 = unsqueeze_509 = None
    mul_613: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_515);  sub_152 = unsqueeze_515 = None
    mul_614: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_100);  sum_49 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_613, relu_28, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_613 = primals_148 = None
    getitem_183: "f32[8, 1920, 8, 8]" = convolution_backward_23[0]
    getitem_184: "f32[1920, 1, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_126: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_127: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_126);  alias_126 = None
    le_24: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_127, 0);  alias_127 = None
    where_24: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_24, full_default, getitem_183);  le_24 = getitem_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_50: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_153: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_518);  convolution_32 = unsqueeze_518 = None
    mul_615: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_24, sub_153)
    sum_51: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_615, [0, 2, 3]);  mul_615 = None
    mul_616: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_50, 0.001953125)
    unsqueeze_519: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
    unsqueeze_520: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 2);  unsqueeze_519 = None
    unsqueeze_521: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, 3);  unsqueeze_520 = None
    mul_617: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_51, 0.001953125)
    mul_618: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_619: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_617, mul_618);  mul_617 = mul_618 = None
    unsqueeze_522: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_619, 0);  mul_619 = None
    unsqueeze_523: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 2);  unsqueeze_522 = None
    unsqueeze_524: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 3);  unsqueeze_523 = None
    mul_620: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_525: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
    unsqueeze_526: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 2);  unsqueeze_525 = None
    unsqueeze_527: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 3);  unsqueeze_526 = None
    mul_621: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_524);  sub_153 = unsqueeze_524 = None
    sub_155: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_24, mul_621);  where_24 = mul_621 = None
    sub_156: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_521);  sub_155 = unsqueeze_521 = None
    mul_622: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_527);  sub_156 = unsqueeze_527 = None
    mul_623: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_97);  sum_51 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_622, relu_27, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_622 = primals_147 = None
    getitem_186: "f32[8, 640, 8, 8]" = convolution_backward_24[0]
    getitem_187: "f32[1920, 640, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_310: "f32[8, 640, 8, 8]" = torch.ops.aten.add.Tensor(where_22, getitem_186);  where_22 = getitem_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_129: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_130: "f32[8, 640, 8, 8]" = torch.ops.aten.alias.default(alias_129);  alias_129 = None
    le_25: "b8[8, 640, 8, 8]" = torch.ops.aten.le.Scalar(alias_130, 0);  alias_130 = None
    where_25: "f32[8, 640, 8, 8]" = torch.ops.aten.where.self(le_25, full_default, add_310);  le_25 = add_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_52: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_157: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_530);  convolution_31 = unsqueeze_530 = None
    mul_624: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(where_25, sub_157)
    sum_53: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_624, [0, 2, 3]);  mul_624 = None
    mul_625: "f32[640]" = torch.ops.aten.mul.Tensor(sum_52, 0.001953125)
    unsqueeze_531: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
    unsqueeze_532: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 2);  unsqueeze_531 = None
    unsqueeze_533: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, 3);  unsqueeze_532 = None
    mul_626: "f32[640]" = torch.ops.aten.mul.Tensor(sum_53, 0.001953125)
    mul_627: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_628: "f32[640]" = torch.ops.aten.mul.Tensor(mul_626, mul_627);  mul_626 = mul_627 = None
    unsqueeze_534: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_628, 0);  mul_628 = None
    unsqueeze_535: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 2);  unsqueeze_534 = None
    unsqueeze_536: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 3);  unsqueeze_535 = None
    mul_629: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_537: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
    unsqueeze_538: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 2);  unsqueeze_537 = None
    unsqueeze_539: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 3);  unsqueeze_538 = None
    mul_630: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_536);  sub_157 = unsqueeze_536 = None
    sub_159: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(where_25, mul_630);  mul_630 = None
    sub_160: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_533);  sub_159 = None
    mul_631: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_539);  sub_160 = unsqueeze_539 = None
    mul_632: "f32[640]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_94);  sum_53 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_631, relu_24, primals_146, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_631 = primals_146 = None
    getitem_189: "f32[8, 640, 16, 16]" = convolution_backward_25[0]
    getitem_190: "f32[640, 640, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_161: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_542);  convolution_30 = unsqueeze_542 = None
    mul_633: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(where_25, sub_161)
    sum_55: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_633, [0, 2, 3]);  mul_633 = None
    mul_635: "f32[640]" = torch.ops.aten.mul.Tensor(sum_55, 0.001953125)
    mul_636: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_637: "f32[640]" = torch.ops.aten.mul.Tensor(mul_635, mul_636);  mul_635 = mul_636 = None
    unsqueeze_546: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_637, 0);  mul_637 = None
    unsqueeze_547: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 2);  unsqueeze_546 = None
    unsqueeze_548: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 3);  unsqueeze_547 = None
    mul_638: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_549: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_550: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 2);  unsqueeze_549 = None
    unsqueeze_551: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 3);  unsqueeze_550 = None
    mul_639: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_548);  sub_161 = unsqueeze_548 = None
    sub_163: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(where_25, mul_639);  where_25 = mul_639 = None
    sub_164: "f32[8, 640, 8, 8]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_533);  sub_163 = unsqueeze_533 = None
    mul_640: "f32[8, 640, 8, 8]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_551);  sub_164 = unsqueeze_551 = None
    mul_641: "f32[640]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_91);  sum_55 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_640, relu_26, primals_145, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_640 = primals_145 = None
    getitem_192: "f32[8, 1920, 8, 8]" = convolution_backward_26[0]
    getitem_193: "f32[640, 1920, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_132: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_133: "f32[8, 1920, 8, 8]" = torch.ops.aten.alias.default(alias_132);  alias_132 = None
    le_26: "b8[8, 1920, 8, 8]" = torch.ops.aten.le.Scalar(alias_133, 0);  alias_133 = None
    where_26: "f32[8, 1920, 8, 8]" = torch.ops.aten.where.self(le_26, full_default, getitem_192);  le_26 = getitem_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_165: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_554);  convolution_29 = unsqueeze_554 = None
    mul_642: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(where_26, sub_165)
    sum_57: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_642, [0, 2, 3]);  mul_642 = None
    mul_643: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_56, 0.001953125)
    unsqueeze_555: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
    unsqueeze_556: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 2);  unsqueeze_555 = None
    unsqueeze_557: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, 3);  unsqueeze_556 = None
    mul_644: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_57, 0.001953125)
    mul_645: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_646: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_644, mul_645);  mul_644 = mul_645 = None
    unsqueeze_558: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_646, 0);  mul_646 = None
    unsqueeze_559: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 2);  unsqueeze_558 = None
    unsqueeze_560: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 3);  unsqueeze_559 = None
    mul_647: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_561: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
    unsqueeze_562: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 2);  unsqueeze_561 = None
    unsqueeze_563: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 3);  unsqueeze_562 = None
    mul_648: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_560);  sub_165 = unsqueeze_560 = None
    sub_167: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(where_26, mul_648);  where_26 = mul_648 = None
    sub_168: "f32[8, 1920, 8, 8]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_557);  sub_167 = unsqueeze_557 = None
    mul_649: "f32[8, 1920, 8, 8]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_563);  sub_168 = unsqueeze_563 = None
    mul_650: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_88);  sum_57 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_649, relu_25, primals_144, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1920, [True, True, False]);  mul_649 = primals_144 = None
    getitem_195: "f32[8, 1920, 16, 16]" = convolution_backward_27[0]
    getitem_196: "f32[1920, 1, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_135: "f32[8, 1920, 16, 16]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_136: "f32[8, 1920, 16, 16]" = torch.ops.aten.alias.default(alias_135);  alias_135 = None
    le_27: "b8[8, 1920, 16, 16]" = torch.ops.aten.le.Scalar(alias_136, 0);  alias_136 = None
    where_27: "f32[8, 1920, 16, 16]" = torch.ops.aten.where.self(le_27, full_default, getitem_195);  le_27 = getitem_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[1920]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_169: "f32[8, 1920, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_566);  convolution_28 = unsqueeze_566 = None
    mul_651: "f32[8, 1920, 16, 16]" = torch.ops.aten.mul.Tensor(where_27, sub_169)
    sum_59: "f32[1920]" = torch.ops.aten.sum.dim_IntList(mul_651, [0, 2, 3]);  mul_651 = None
    mul_652: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_58, 0.00048828125)
    unsqueeze_567: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
    unsqueeze_568: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 2);  unsqueeze_567 = None
    unsqueeze_569: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, 3);  unsqueeze_568 = None
    mul_653: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_59, 0.00048828125)
    mul_654: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_655: "f32[1920]" = torch.ops.aten.mul.Tensor(mul_653, mul_654);  mul_653 = mul_654 = None
    unsqueeze_570: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_655, 0);  mul_655 = None
    unsqueeze_571: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 2);  unsqueeze_570 = None
    unsqueeze_572: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 3);  unsqueeze_571 = None
    mul_656: "f32[1920]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_573: "f32[1, 1920]" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
    unsqueeze_574: "f32[1, 1920, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 2);  unsqueeze_573 = None
    unsqueeze_575: "f32[1, 1920, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 3);  unsqueeze_574 = None
    mul_657: "f32[8, 1920, 16, 16]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_572);  sub_169 = unsqueeze_572 = None
    sub_171: "f32[8, 1920, 16, 16]" = torch.ops.aten.sub.Tensor(where_27, mul_657);  where_27 = mul_657 = None
    sub_172: "f32[8, 1920, 16, 16]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_569);  sub_171 = unsqueeze_569 = None
    mul_658: "f32[8, 1920, 16, 16]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_575);  sub_172 = unsqueeze_575 = None
    mul_659: "f32[1920]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_85);  sum_59 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_658, relu_24, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_658 = primals_143 = None
    getitem_198: "f32[8, 640, 16, 16]" = convolution_backward_28[0]
    getitem_199: "f32[1920, 640, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_311: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(getitem_189, getitem_198);  getitem_189 = getitem_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_138: "f32[8, 640, 16, 16]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_139: "f32[8, 640, 16, 16]" = torch.ops.aten.alias.default(alias_138);  alias_138 = None
    le_28: "b8[8, 640, 16, 16]" = torch.ops.aten.le.Scalar(alias_139, 0);  alias_139 = None
    where_28: "f32[8, 640, 16, 16]" = torch.ops.aten.where.self(le_28, full_default, add_311);  le_28 = add_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_60: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_173: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_578);  convolution_27 = unsqueeze_578 = None
    mul_660: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(where_28, sub_173)
    sum_61: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_660, [0, 2, 3]);  mul_660 = None
    mul_661: "f32[640]" = torch.ops.aten.mul.Tensor(sum_60, 0.00048828125)
    unsqueeze_579: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_661, 0);  mul_661 = None
    unsqueeze_580: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 2);  unsqueeze_579 = None
    unsqueeze_581: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 3);  unsqueeze_580 = None
    mul_662: "f32[640]" = torch.ops.aten.mul.Tensor(sum_61, 0.00048828125)
    mul_663: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_664: "f32[640]" = torch.ops.aten.mul.Tensor(mul_662, mul_663);  mul_662 = mul_663 = None
    unsqueeze_582: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_664, 0);  mul_664 = None
    unsqueeze_583: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 2);  unsqueeze_582 = None
    unsqueeze_584: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 3);  unsqueeze_583 = None
    mul_665: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_585: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
    unsqueeze_586: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 2);  unsqueeze_585 = None
    unsqueeze_587: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 3);  unsqueeze_586 = None
    mul_666: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_584);  sub_173 = unsqueeze_584 = None
    sub_175: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(where_28, mul_666);  mul_666 = None
    sub_176: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_581);  sub_175 = unsqueeze_581 = None
    mul_667: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_587);  sub_176 = unsqueeze_587 = None
    mul_668: "f32[640]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_82);  sum_61 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_667, relu_23, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_667 = primals_142 = None
    getitem_201: "f32[8, 160, 16, 16]" = convolution_backward_29[0]
    getitem_202: "f32[640, 160, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_141: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_142: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(alias_141);  alias_141 = None
    le_29: "b8[8, 160, 16, 16]" = torch.ops.aten.le.Scalar(alias_142, 0);  alias_142 = None
    where_29: "f32[8, 160, 16, 16]" = torch.ops.aten.where.self(le_29, full_default, getitem_201);  le_29 = getitem_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_62: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_177: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_590);  convolution_26 = unsqueeze_590 = None
    mul_669: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(where_29, sub_177)
    sum_63: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_669, [0, 2, 3]);  mul_669 = None
    mul_670: "f32[160]" = torch.ops.aten.mul.Tensor(sum_62, 0.00048828125)
    unsqueeze_591: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_670, 0);  mul_670 = None
    unsqueeze_592: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 2);  unsqueeze_591 = None
    unsqueeze_593: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 3);  unsqueeze_592 = None
    mul_671: "f32[160]" = torch.ops.aten.mul.Tensor(sum_63, 0.00048828125)
    mul_672: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_673: "f32[160]" = torch.ops.aten.mul.Tensor(mul_671, mul_672);  mul_671 = mul_672 = None
    unsqueeze_594: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_673, 0);  mul_673 = None
    unsqueeze_595: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 2);  unsqueeze_594 = None
    unsqueeze_596: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 3);  unsqueeze_595 = None
    mul_674: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_597: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
    unsqueeze_598: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 2);  unsqueeze_597 = None
    unsqueeze_599: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 3);  unsqueeze_598 = None
    mul_675: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_596);  sub_177 = unsqueeze_596 = None
    sub_179: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(where_29, mul_675);  where_29 = mul_675 = None
    sub_180: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_593);  sub_179 = unsqueeze_593 = None
    mul_676: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_599);  sub_180 = unsqueeze_599 = None
    mul_677: "f32[160]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_79);  sum_63 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_676, relu_22, primals_141, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_676 = primals_141 = None
    getitem_204: "f32[8, 160, 16, 16]" = convolution_backward_30[0]
    getitem_205: "f32[160, 160, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_144: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_145: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(alias_144);  alias_144 = None
    le_30: "b8[8, 160, 16, 16]" = torch.ops.aten.le.Scalar(alias_145, 0);  alias_145 = None
    where_30: "f32[8, 160, 16, 16]" = torch.ops.aten.where.self(le_30, full_default, getitem_204);  le_30 = getitem_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_64: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_181: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_602);  convolution_25 = unsqueeze_602 = None
    mul_678: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(where_30, sub_181)
    sum_65: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_678, [0, 2, 3]);  mul_678 = None
    mul_679: "f32[160]" = torch.ops.aten.mul.Tensor(sum_64, 0.00048828125)
    unsqueeze_603: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_679, 0);  mul_679 = None
    unsqueeze_604: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 2);  unsqueeze_603 = None
    unsqueeze_605: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, 3);  unsqueeze_604 = None
    mul_680: "f32[160]" = torch.ops.aten.mul.Tensor(sum_65, 0.00048828125)
    mul_681: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_682: "f32[160]" = torch.ops.aten.mul.Tensor(mul_680, mul_681);  mul_680 = mul_681 = None
    unsqueeze_606: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_682, 0);  mul_682 = None
    unsqueeze_607: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 2);  unsqueeze_606 = None
    unsqueeze_608: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 3);  unsqueeze_607 = None
    mul_683: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_609: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
    unsqueeze_610: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 2);  unsqueeze_609 = None
    unsqueeze_611: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 3);  unsqueeze_610 = None
    mul_684: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_608);  sub_181 = unsqueeze_608 = None
    sub_183: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(where_30, mul_684);  where_30 = mul_684 = None
    sub_184: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_605);  sub_183 = unsqueeze_605 = None
    mul_685: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_611);  sub_184 = unsqueeze_611 = None
    mul_686: "f32[160]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_76);  sum_65 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_685, relu_21, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_685 = primals_140 = None
    getitem_207: "f32[8, 640, 16, 16]" = convolution_backward_31[0]
    getitem_208: "f32[160, 640, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_312: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(where_28, getitem_207);  where_28 = getitem_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_147: "f32[8, 640, 16, 16]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_148: "f32[8, 640, 16, 16]" = torch.ops.aten.alias.default(alias_147);  alias_147 = None
    le_31: "b8[8, 640, 16, 16]" = torch.ops.aten.le.Scalar(alias_148, 0);  alias_148 = None
    where_31: "f32[8, 640, 16, 16]" = torch.ops.aten.where.self(le_31, full_default, add_312);  le_31 = add_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_66: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_185: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_614);  convolution_24 = unsqueeze_614 = None
    mul_687: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(where_31, sub_185)
    sum_67: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_687, [0, 2, 3]);  mul_687 = None
    mul_688: "f32[640]" = torch.ops.aten.mul.Tensor(sum_66, 0.00048828125)
    unsqueeze_615: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_688, 0);  mul_688 = None
    unsqueeze_616: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 2);  unsqueeze_615 = None
    unsqueeze_617: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, 3);  unsqueeze_616 = None
    mul_689: "f32[640]" = torch.ops.aten.mul.Tensor(sum_67, 0.00048828125)
    mul_690: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_691: "f32[640]" = torch.ops.aten.mul.Tensor(mul_689, mul_690);  mul_689 = mul_690 = None
    unsqueeze_618: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_691, 0);  mul_691 = None
    unsqueeze_619: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 2);  unsqueeze_618 = None
    unsqueeze_620: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 3);  unsqueeze_619 = None
    mul_692: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_621: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_622: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 2);  unsqueeze_621 = None
    unsqueeze_623: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 3);  unsqueeze_622 = None
    mul_693: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_620);  sub_185 = unsqueeze_620 = None
    sub_187: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(where_31, mul_693);  mul_693 = None
    sub_188: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_617);  sub_187 = unsqueeze_617 = None
    mul_694: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_623);  sub_188 = unsqueeze_623 = None
    mul_695: "f32[640]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_73);  sum_67 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_694, relu_20, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_694 = primals_139 = None
    getitem_210: "f32[8, 160, 16, 16]" = convolution_backward_32[0]
    getitem_211: "f32[640, 160, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_150: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_151: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(alias_150);  alias_150 = None
    le_32: "b8[8, 160, 16, 16]" = torch.ops.aten.le.Scalar(alias_151, 0);  alias_151 = None
    where_32: "f32[8, 160, 16, 16]" = torch.ops.aten.where.self(le_32, full_default, getitem_210);  le_32 = getitem_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_68: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_189: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_626);  convolution_23 = unsqueeze_626 = None
    mul_696: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(where_32, sub_189)
    sum_69: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_696, [0, 2, 3]);  mul_696 = None
    mul_697: "f32[160]" = torch.ops.aten.mul.Tensor(sum_68, 0.00048828125)
    unsqueeze_627: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_697, 0);  mul_697 = None
    unsqueeze_628: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 2);  unsqueeze_627 = None
    unsqueeze_629: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, 3);  unsqueeze_628 = None
    mul_698: "f32[160]" = torch.ops.aten.mul.Tensor(sum_69, 0.00048828125)
    mul_699: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_700: "f32[160]" = torch.ops.aten.mul.Tensor(mul_698, mul_699);  mul_698 = mul_699 = None
    unsqueeze_630: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_700, 0);  mul_700 = None
    unsqueeze_631: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 2);  unsqueeze_630 = None
    unsqueeze_632: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 3);  unsqueeze_631 = None
    mul_701: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_633: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
    unsqueeze_634: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 2);  unsqueeze_633 = None
    unsqueeze_635: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 3);  unsqueeze_634 = None
    mul_702: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_632);  sub_189 = unsqueeze_632 = None
    sub_191: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(where_32, mul_702);  where_32 = mul_702 = None
    sub_192: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_629);  sub_191 = unsqueeze_629 = None
    mul_703: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_635);  sub_192 = unsqueeze_635 = None
    mul_704: "f32[160]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_70);  sum_69 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_703, relu_19, primals_138, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_703 = primals_138 = None
    getitem_213: "f32[8, 160, 16, 16]" = convolution_backward_33[0]
    getitem_214: "f32[160, 160, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_153: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_154: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(alias_153);  alias_153 = None
    le_33: "b8[8, 160, 16, 16]" = torch.ops.aten.le.Scalar(alias_154, 0);  alias_154 = None
    where_33: "f32[8, 160, 16, 16]" = torch.ops.aten.where.self(le_33, full_default, getitem_213);  le_33 = getitem_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_70: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_193: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_638);  convolution_22 = unsqueeze_638 = None
    mul_705: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(where_33, sub_193)
    sum_71: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_705, [0, 2, 3]);  mul_705 = None
    mul_706: "f32[160]" = torch.ops.aten.mul.Tensor(sum_70, 0.00048828125)
    unsqueeze_639: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_706, 0);  mul_706 = None
    unsqueeze_640: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 2);  unsqueeze_639 = None
    unsqueeze_641: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, 3);  unsqueeze_640 = None
    mul_707: "f32[160]" = torch.ops.aten.mul.Tensor(sum_71, 0.00048828125)
    mul_708: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_709: "f32[160]" = torch.ops.aten.mul.Tensor(mul_707, mul_708);  mul_707 = mul_708 = None
    unsqueeze_642: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_709, 0);  mul_709 = None
    unsqueeze_643: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 2);  unsqueeze_642 = None
    unsqueeze_644: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 3);  unsqueeze_643 = None
    mul_710: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_645: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
    unsqueeze_646: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 2);  unsqueeze_645 = None
    unsqueeze_647: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 3);  unsqueeze_646 = None
    mul_711: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_644);  sub_193 = unsqueeze_644 = None
    sub_195: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(where_33, mul_711);  where_33 = mul_711 = None
    sub_196: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_641);  sub_195 = unsqueeze_641 = None
    mul_712: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_647);  sub_196 = unsqueeze_647 = None
    mul_713: "f32[160]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_67);  sum_71 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_712, relu_18, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_712 = primals_137 = None
    getitem_216: "f32[8, 640, 16, 16]" = convolution_backward_34[0]
    getitem_217: "f32[160, 640, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_313: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(where_31, getitem_216);  where_31 = getitem_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_156: "f32[8, 640, 16, 16]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_157: "f32[8, 640, 16, 16]" = torch.ops.aten.alias.default(alias_156);  alias_156 = None
    le_34: "b8[8, 640, 16, 16]" = torch.ops.aten.le.Scalar(alias_157, 0);  alias_157 = None
    where_34: "f32[8, 640, 16, 16]" = torch.ops.aten.where.self(le_34, full_default, add_313);  le_34 = add_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_197: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_650);  convolution_21 = unsqueeze_650 = None
    mul_714: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(where_34, sub_197)
    sum_73: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_714, [0, 2, 3]);  mul_714 = None
    mul_715: "f32[640]" = torch.ops.aten.mul.Tensor(sum_72, 0.00048828125)
    unsqueeze_651: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_715, 0);  mul_715 = None
    unsqueeze_652: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 2);  unsqueeze_651 = None
    unsqueeze_653: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, 3);  unsqueeze_652 = None
    mul_716: "f32[640]" = torch.ops.aten.mul.Tensor(sum_73, 0.00048828125)
    mul_717: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_718: "f32[640]" = torch.ops.aten.mul.Tensor(mul_716, mul_717);  mul_716 = mul_717 = None
    unsqueeze_654: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_718, 0);  mul_718 = None
    unsqueeze_655: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 2);  unsqueeze_654 = None
    unsqueeze_656: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 3);  unsqueeze_655 = None
    mul_719: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_657: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
    unsqueeze_658: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 2);  unsqueeze_657 = None
    unsqueeze_659: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 3);  unsqueeze_658 = None
    mul_720: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_656);  sub_197 = unsqueeze_656 = None
    sub_199: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(where_34, mul_720);  mul_720 = None
    sub_200: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_653);  sub_199 = unsqueeze_653 = None
    mul_721: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_659);  sub_200 = unsqueeze_659 = None
    mul_722: "f32[640]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_64);  sum_73 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_721, relu_17, primals_136, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_721 = primals_136 = None
    getitem_219: "f32[8, 160, 16, 16]" = convolution_backward_35[0]
    getitem_220: "f32[640, 160, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_159: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_160: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(alias_159);  alias_159 = None
    le_35: "b8[8, 160, 16, 16]" = torch.ops.aten.le.Scalar(alias_160, 0);  alias_160 = None
    where_35: "f32[8, 160, 16, 16]" = torch.ops.aten.where.self(le_35, full_default, getitem_219);  le_35 = getitem_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_201: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_662);  convolution_20 = unsqueeze_662 = None
    mul_723: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(where_35, sub_201)
    sum_75: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_723, [0, 2, 3]);  mul_723 = None
    mul_724: "f32[160]" = torch.ops.aten.mul.Tensor(sum_74, 0.00048828125)
    unsqueeze_663: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_724, 0);  mul_724 = None
    unsqueeze_664: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 2);  unsqueeze_663 = None
    unsqueeze_665: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, 3);  unsqueeze_664 = None
    mul_725: "f32[160]" = torch.ops.aten.mul.Tensor(sum_75, 0.00048828125)
    mul_726: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_727: "f32[160]" = torch.ops.aten.mul.Tensor(mul_725, mul_726);  mul_725 = mul_726 = None
    unsqueeze_666: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_727, 0);  mul_727 = None
    unsqueeze_667: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 2);  unsqueeze_666 = None
    unsqueeze_668: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 3);  unsqueeze_667 = None
    mul_728: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_669: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
    unsqueeze_670: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 2);  unsqueeze_669 = None
    unsqueeze_671: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 3);  unsqueeze_670 = None
    mul_729: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_668);  sub_201 = unsqueeze_668 = None
    sub_203: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(where_35, mul_729);  where_35 = mul_729 = None
    sub_204: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_665);  sub_203 = unsqueeze_665 = None
    mul_730: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_671);  sub_204 = unsqueeze_671 = None
    mul_731: "f32[160]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_61);  sum_75 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_730, relu_16, primals_135, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_730 = primals_135 = None
    getitem_222: "f32[8, 160, 16, 16]" = convolution_backward_36[0]
    getitem_223: "f32[160, 160, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_162: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_163: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(alias_162);  alias_162 = None
    le_36: "b8[8, 160, 16, 16]" = torch.ops.aten.le.Scalar(alias_163, 0);  alias_163 = None
    where_36: "f32[8, 160, 16, 16]" = torch.ops.aten.where.self(le_36, full_default, getitem_222);  le_36 = getitem_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_76: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_205: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_674);  convolution_19 = unsqueeze_674 = None
    mul_732: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(where_36, sub_205)
    sum_77: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 2, 3]);  mul_732 = None
    mul_733: "f32[160]" = torch.ops.aten.mul.Tensor(sum_76, 0.00048828125)
    unsqueeze_675: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
    unsqueeze_676: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 2);  unsqueeze_675 = None
    unsqueeze_677: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, 3);  unsqueeze_676 = None
    mul_734: "f32[160]" = torch.ops.aten.mul.Tensor(sum_77, 0.00048828125)
    mul_735: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_736: "f32[160]" = torch.ops.aten.mul.Tensor(mul_734, mul_735);  mul_734 = mul_735 = None
    unsqueeze_678: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_736, 0);  mul_736 = None
    unsqueeze_679: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 2);  unsqueeze_678 = None
    unsqueeze_680: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 3);  unsqueeze_679 = None
    mul_737: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_681: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
    unsqueeze_682: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 2);  unsqueeze_681 = None
    unsqueeze_683: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 3);  unsqueeze_682 = None
    mul_738: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_680);  sub_205 = unsqueeze_680 = None
    sub_207: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(where_36, mul_738);  where_36 = mul_738 = None
    sub_208: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_677);  sub_207 = unsqueeze_677 = None
    mul_739: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_683);  sub_208 = unsqueeze_683 = None
    mul_740: "f32[160]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_58);  sum_77 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_739, relu_15, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_739 = primals_134 = None
    getitem_225: "f32[8, 640, 16, 16]" = convolution_backward_37[0]
    getitem_226: "f32[160, 640, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_314: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(where_34, getitem_225);  where_34 = getitem_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_165: "f32[8, 640, 16, 16]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_166: "f32[8, 640, 16, 16]" = torch.ops.aten.alias.default(alias_165);  alias_165 = None
    le_37: "b8[8, 640, 16, 16]" = torch.ops.aten.le.Scalar(alias_166, 0);  alias_166 = None
    where_37: "f32[8, 640, 16, 16]" = torch.ops.aten.where.self(le_37, full_default, add_314);  le_37 = add_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_78: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_209: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_686);  convolution_18 = unsqueeze_686 = None
    mul_741: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(where_37, sub_209)
    sum_79: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_741, [0, 2, 3]);  mul_741 = None
    mul_742: "f32[640]" = torch.ops.aten.mul.Tensor(sum_78, 0.00048828125)
    unsqueeze_687: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_742, 0);  mul_742 = None
    unsqueeze_688: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 2);  unsqueeze_687 = None
    unsqueeze_689: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, 3);  unsqueeze_688 = None
    mul_743: "f32[640]" = torch.ops.aten.mul.Tensor(sum_79, 0.00048828125)
    mul_744: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_745: "f32[640]" = torch.ops.aten.mul.Tensor(mul_743, mul_744);  mul_743 = mul_744 = None
    unsqueeze_690: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_745, 0);  mul_745 = None
    unsqueeze_691: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 2);  unsqueeze_690 = None
    unsqueeze_692: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 3);  unsqueeze_691 = None
    mul_746: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_693: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_694: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 2);  unsqueeze_693 = None
    unsqueeze_695: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 3);  unsqueeze_694 = None
    mul_747: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_692);  sub_209 = unsqueeze_692 = None
    sub_211: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(where_37, mul_747);  mul_747 = None
    sub_212: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_689);  sub_211 = unsqueeze_689 = None
    mul_748: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_695);  sub_212 = unsqueeze_695 = None
    mul_749: "f32[640]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_55);  sum_79 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_748, relu_14, primals_133, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_748 = primals_133 = None
    getitem_228: "f32[8, 160, 16, 16]" = convolution_backward_38[0]
    getitem_229: "f32[640, 160, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_168: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_169: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(alias_168);  alias_168 = None
    le_38: "b8[8, 160, 16, 16]" = torch.ops.aten.le.Scalar(alias_169, 0);  alias_169 = None
    where_38: "f32[8, 160, 16, 16]" = torch.ops.aten.where.self(le_38, full_default, getitem_228);  le_38 = getitem_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_80: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_213: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_698);  convolution_17 = unsqueeze_698 = None
    mul_750: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(where_38, sub_213)
    sum_81: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_750, [0, 2, 3]);  mul_750 = None
    mul_751: "f32[160]" = torch.ops.aten.mul.Tensor(sum_80, 0.00048828125)
    unsqueeze_699: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_751, 0);  mul_751 = None
    unsqueeze_700: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 2);  unsqueeze_699 = None
    unsqueeze_701: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, 3);  unsqueeze_700 = None
    mul_752: "f32[160]" = torch.ops.aten.mul.Tensor(sum_81, 0.00048828125)
    mul_753: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_754: "f32[160]" = torch.ops.aten.mul.Tensor(mul_752, mul_753);  mul_752 = mul_753 = None
    unsqueeze_702: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_754, 0);  mul_754 = None
    unsqueeze_703: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 2);  unsqueeze_702 = None
    unsqueeze_704: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 3);  unsqueeze_703 = None
    mul_755: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_705: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_706: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 2);  unsqueeze_705 = None
    unsqueeze_707: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 3);  unsqueeze_706 = None
    mul_756: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_704);  sub_213 = unsqueeze_704 = None
    sub_215: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(where_38, mul_756);  where_38 = mul_756 = None
    sub_216: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_701);  sub_215 = unsqueeze_701 = None
    mul_757: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_707);  sub_216 = unsqueeze_707 = None
    mul_758: "f32[160]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_52);  sum_81 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_757, relu_13, primals_132, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_757 = primals_132 = None
    getitem_231: "f32[8, 160, 16, 16]" = convolution_backward_39[0]
    getitem_232: "f32[160, 160, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_171: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_172: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(alias_171);  alias_171 = None
    le_39: "b8[8, 160, 16, 16]" = torch.ops.aten.le.Scalar(alias_172, 0);  alias_172 = None
    where_39: "f32[8, 160, 16, 16]" = torch.ops.aten.where.self(le_39, full_default, getitem_231);  le_39 = getitem_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_82: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_217: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_710);  convolution_16 = unsqueeze_710 = None
    mul_759: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(where_39, sub_217)
    sum_83: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_759, [0, 2, 3]);  mul_759 = None
    mul_760: "f32[160]" = torch.ops.aten.mul.Tensor(sum_82, 0.00048828125)
    unsqueeze_711: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_760, 0);  mul_760 = None
    unsqueeze_712: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 2);  unsqueeze_711 = None
    unsqueeze_713: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, 3);  unsqueeze_712 = None
    mul_761: "f32[160]" = torch.ops.aten.mul.Tensor(sum_83, 0.00048828125)
    mul_762: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_763: "f32[160]" = torch.ops.aten.mul.Tensor(mul_761, mul_762);  mul_761 = mul_762 = None
    unsqueeze_714: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_763, 0);  mul_763 = None
    unsqueeze_715: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 2);  unsqueeze_714 = None
    unsqueeze_716: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 3);  unsqueeze_715 = None
    mul_764: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_717: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_718: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 2);  unsqueeze_717 = None
    unsqueeze_719: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 3);  unsqueeze_718 = None
    mul_765: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_716);  sub_217 = unsqueeze_716 = None
    sub_219: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(where_39, mul_765);  where_39 = mul_765 = None
    sub_220: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_713);  sub_219 = unsqueeze_713 = None
    mul_766: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_719);  sub_220 = unsqueeze_719 = None
    mul_767: "f32[160]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_49);  sum_83 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_766, relu_12, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_766 = primals_131 = None
    getitem_234: "f32[8, 640, 16, 16]" = convolution_backward_40[0]
    getitem_235: "f32[160, 640, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_315: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(where_37, getitem_234);  where_37 = getitem_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_174: "f32[8, 640, 16, 16]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_175: "f32[8, 640, 16, 16]" = torch.ops.aten.alias.default(alias_174);  alias_174 = None
    le_40: "b8[8, 640, 16, 16]" = torch.ops.aten.le.Scalar(alias_175, 0);  alias_175 = None
    where_40: "f32[8, 640, 16, 16]" = torch.ops.aten.where.self(le_40, full_default, add_315);  le_40 = add_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_84: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_221: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_722);  convolution_15 = unsqueeze_722 = None
    mul_768: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(where_40, sub_221)
    sum_85: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_768, [0, 2, 3]);  mul_768 = None
    mul_769: "f32[640]" = torch.ops.aten.mul.Tensor(sum_84, 0.00048828125)
    unsqueeze_723: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_769, 0);  mul_769 = None
    unsqueeze_724: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 2);  unsqueeze_723 = None
    unsqueeze_725: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, 3);  unsqueeze_724 = None
    mul_770: "f32[640]" = torch.ops.aten.mul.Tensor(sum_85, 0.00048828125)
    mul_771: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_772: "f32[640]" = torch.ops.aten.mul.Tensor(mul_770, mul_771);  mul_770 = mul_771 = None
    unsqueeze_726: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_772, 0);  mul_772 = None
    unsqueeze_727: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 2);  unsqueeze_726 = None
    unsqueeze_728: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 3);  unsqueeze_727 = None
    mul_773: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_729: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_730: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 2);  unsqueeze_729 = None
    unsqueeze_731: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 3);  unsqueeze_730 = None
    mul_774: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_728);  sub_221 = unsqueeze_728 = None
    sub_223: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(where_40, mul_774);  mul_774 = None
    sub_224: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_725);  sub_223 = unsqueeze_725 = None
    mul_775: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_731);  sub_224 = unsqueeze_731 = None
    mul_776: "f32[640]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_46);  sum_85 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_775, relu_11, primals_130, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_775 = primals_130 = None
    getitem_237: "f32[8, 160, 16, 16]" = convolution_backward_41[0]
    getitem_238: "f32[640, 160, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_177: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_178: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(alias_177);  alias_177 = None
    le_41: "b8[8, 160, 16, 16]" = torch.ops.aten.le.Scalar(alias_178, 0);  alias_178 = None
    where_41: "f32[8, 160, 16, 16]" = torch.ops.aten.where.self(le_41, full_default, getitem_237);  le_41 = getitem_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_86: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_225: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_734);  convolution_14 = unsqueeze_734 = None
    mul_777: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(where_41, sub_225)
    sum_87: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_777, [0, 2, 3]);  mul_777 = None
    mul_778: "f32[160]" = torch.ops.aten.mul.Tensor(sum_86, 0.00048828125)
    unsqueeze_735: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_778, 0);  mul_778 = None
    unsqueeze_736: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 2);  unsqueeze_735 = None
    unsqueeze_737: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, 3);  unsqueeze_736 = None
    mul_779: "f32[160]" = torch.ops.aten.mul.Tensor(sum_87, 0.00048828125)
    mul_780: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_781: "f32[160]" = torch.ops.aten.mul.Tensor(mul_779, mul_780);  mul_779 = mul_780 = None
    unsqueeze_738: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_781, 0);  mul_781 = None
    unsqueeze_739: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 2);  unsqueeze_738 = None
    unsqueeze_740: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 3);  unsqueeze_739 = None
    mul_782: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_741: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_742: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 2);  unsqueeze_741 = None
    unsqueeze_743: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 3);  unsqueeze_742 = None
    mul_783: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_740);  sub_225 = unsqueeze_740 = None
    sub_227: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(where_41, mul_783);  where_41 = mul_783 = None
    sub_228: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_737);  sub_227 = unsqueeze_737 = None
    mul_784: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_743);  sub_228 = unsqueeze_743 = None
    mul_785: "f32[160]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_43);  sum_87 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_784, relu_10, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_784 = primals_129 = None
    getitem_240: "f32[8, 160, 16, 16]" = convolution_backward_42[0]
    getitem_241: "f32[160, 160, 3, 3]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_180: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_181: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(alias_180);  alias_180 = None
    le_42: "b8[8, 160, 16, 16]" = torch.ops.aten.le.Scalar(alias_181, 0);  alias_181 = None
    where_42: "f32[8, 160, 16, 16]" = torch.ops.aten.where.self(le_42, full_default, getitem_240);  le_42 = getitem_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_88: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_229: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_746);  convolution_13 = unsqueeze_746 = None
    mul_786: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(where_42, sub_229)
    sum_89: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_786, [0, 2, 3]);  mul_786 = None
    mul_787: "f32[160]" = torch.ops.aten.mul.Tensor(sum_88, 0.00048828125)
    unsqueeze_747: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_787, 0);  mul_787 = None
    unsqueeze_748: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 2);  unsqueeze_747 = None
    unsqueeze_749: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, 3);  unsqueeze_748 = None
    mul_788: "f32[160]" = torch.ops.aten.mul.Tensor(sum_89, 0.00048828125)
    mul_789: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_790: "f32[160]" = torch.ops.aten.mul.Tensor(mul_788, mul_789);  mul_788 = mul_789 = None
    unsqueeze_750: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_790, 0);  mul_790 = None
    unsqueeze_751: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 2);  unsqueeze_750 = None
    unsqueeze_752: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 3);  unsqueeze_751 = None
    mul_791: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_753: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
    unsqueeze_754: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 2);  unsqueeze_753 = None
    unsqueeze_755: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 3);  unsqueeze_754 = None
    mul_792: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_752);  sub_229 = unsqueeze_752 = None
    sub_231: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(where_42, mul_792);  where_42 = mul_792 = None
    sub_232: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_749);  sub_231 = unsqueeze_749 = None
    mul_793: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_755);  sub_232 = unsqueeze_755 = None
    mul_794: "f32[160]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_40);  sum_89 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_793, relu_9, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_793 = primals_128 = None
    getitem_243: "f32[8, 640, 16, 16]" = convolution_backward_43[0]
    getitem_244: "f32[160, 640, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_316: "f32[8, 640, 16, 16]" = torch.ops.aten.add.Tensor(where_40, getitem_243);  where_40 = getitem_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:337, code: return self.act(x)
    alias_183: "f32[8, 640, 16, 16]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_184: "f32[8, 640, 16, 16]" = torch.ops.aten.alias.default(alias_183);  alias_183 = None
    le_43: "b8[8, 640, 16, 16]" = torch.ops.aten.le.Scalar(alias_184, 0);  alias_184 = None
    where_43: "f32[8, 640, 16, 16]" = torch.ops.aten.where.self(le_43, full_default, add_316);  le_43 = add_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_90: "f32[640]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_233: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_758);  convolution_12 = unsqueeze_758 = None
    mul_795: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(where_43, sub_233)
    sum_91: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_795, [0, 2, 3]);  mul_795 = None
    mul_796: "f32[640]" = torch.ops.aten.mul.Tensor(sum_90, 0.00048828125)
    unsqueeze_759: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_796, 0);  mul_796 = None
    unsqueeze_760: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 2);  unsqueeze_759 = None
    unsqueeze_761: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, 3);  unsqueeze_760 = None
    mul_797: "f32[640]" = torch.ops.aten.mul.Tensor(sum_91, 0.00048828125)
    mul_798: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_799: "f32[640]" = torch.ops.aten.mul.Tensor(mul_797, mul_798);  mul_797 = mul_798 = None
    unsqueeze_762: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_799, 0);  mul_799 = None
    unsqueeze_763: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 2);  unsqueeze_762 = None
    unsqueeze_764: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 3);  unsqueeze_763 = None
    mul_800: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_765: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_766: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 2);  unsqueeze_765 = None
    unsqueeze_767: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 3);  unsqueeze_766 = None
    mul_801: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_764);  sub_233 = unsqueeze_764 = None
    sub_235: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(where_43, mul_801);  mul_801 = None
    sub_236: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_761);  sub_235 = None
    mul_802: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_767);  sub_236 = unsqueeze_767 = None
    mul_803: "f32[640]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_37);  sum_91 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_802, relu_6, primals_127, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_802 = primals_127 = None
    getitem_246: "f32[8, 192, 32, 32]" = convolution_backward_44[0]
    getitem_247: "f32[640, 192, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_237: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_770);  convolution_11 = unsqueeze_770 = None
    mul_804: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(where_43, sub_237)
    sum_93: "f32[640]" = torch.ops.aten.sum.dim_IntList(mul_804, [0, 2, 3]);  mul_804 = None
    mul_806: "f32[640]" = torch.ops.aten.mul.Tensor(sum_93, 0.00048828125)
    mul_807: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_808: "f32[640]" = torch.ops.aten.mul.Tensor(mul_806, mul_807);  mul_806 = mul_807 = None
    unsqueeze_774: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_808, 0);  mul_808 = None
    unsqueeze_775: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 2);  unsqueeze_774 = None
    unsqueeze_776: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 3);  unsqueeze_775 = None
    mul_809: "f32[640]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_777: "f32[1, 640]" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
    unsqueeze_778: "f32[1, 640, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 2);  unsqueeze_777 = None
    unsqueeze_779: "f32[1, 640, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 3);  unsqueeze_778 = None
    mul_810: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_776);  sub_237 = unsqueeze_776 = None
    sub_239: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(where_43, mul_810);  where_43 = mul_810 = None
    sub_240: "f32[8, 640, 16, 16]" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_761);  sub_239 = unsqueeze_761 = None
    mul_811: "f32[8, 640, 16, 16]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_779);  sub_240 = unsqueeze_779 = None
    mul_812: "f32[640]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_34);  sum_93 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_811, relu_8, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_811 = primals_126 = None
    getitem_249: "f32[8, 160, 16, 16]" = convolution_backward_45[0]
    getitem_250: "f32[640, 160, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_186: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_187: "f32[8, 160, 16, 16]" = torch.ops.aten.alias.default(alias_186);  alias_186 = None
    le_44: "b8[8, 160, 16, 16]" = torch.ops.aten.le.Scalar(alias_187, 0);  alias_187 = None
    where_44: "f32[8, 160, 16, 16]" = torch.ops.aten.where.self(le_44, full_default, getitem_249);  le_44 = getitem_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_94: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_241: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_782);  convolution_10 = unsqueeze_782 = None
    mul_813: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(where_44, sub_241)
    sum_95: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_813, [0, 2, 3]);  mul_813 = None
    mul_814: "f32[160]" = torch.ops.aten.mul.Tensor(sum_94, 0.00048828125)
    unsqueeze_783: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_814, 0);  mul_814 = None
    unsqueeze_784: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 2);  unsqueeze_783 = None
    unsqueeze_785: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, 3);  unsqueeze_784 = None
    mul_815: "f32[160]" = torch.ops.aten.mul.Tensor(sum_95, 0.00048828125)
    mul_816: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_817: "f32[160]" = torch.ops.aten.mul.Tensor(mul_815, mul_816);  mul_815 = mul_816 = None
    unsqueeze_786: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_817, 0);  mul_817 = None
    unsqueeze_787: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 2);  unsqueeze_786 = None
    unsqueeze_788: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 3);  unsqueeze_787 = None
    mul_818: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_789: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_790: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 2);  unsqueeze_789 = None
    unsqueeze_791: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 3);  unsqueeze_790 = None
    mul_819: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_788);  sub_241 = unsqueeze_788 = None
    sub_243: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(where_44, mul_819);  where_44 = mul_819 = None
    sub_244: "f32[8, 160, 16, 16]" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_785);  sub_243 = unsqueeze_785 = None
    mul_820: "f32[8, 160, 16, 16]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_791);  sub_244 = unsqueeze_791 = None
    mul_821: "f32[160]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_31);  sum_95 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_820, relu_7, primals_125, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_820 = primals_125 = None
    getitem_252: "f32[8, 160, 32, 32]" = convolution_backward_46[0]
    getitem_253: "f32[160, 160, 3, 3]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_189: "f32[8, 160, 32, 32]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_190: "f32[8, 160, 32, 32]" = torch.ops.aten.alias.default(alias_189);  alias_189 = None
    le_45: "b8[8, 160, 32, 32]" = torch.ops.aten.le.Scalar(alias_190, 0);  alias_190 = None
    where_45: "f32[8, 160, 32, 32]" = torch.ops.aten.where.self(le_45, full_default, getitem_252);  le_45 = getitem_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_96: "f32[160]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_245: "f32[8, 160, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_794);  convolution_9 = unsqueeze_794 = None
    mul_822: "f32[8, 160, 32, 32]" = torch.ops.aten.mul.Tensor(where_45, sub_245)
    sum_97: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_822, [0, 2, 3]);  mul_822 = None
    mul_823: "f32[160]" = torch.ops.aten.mul.Tensor(sum_96, 0.0001220703125)
    unsqueeze_795: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_823, 0);  mul_823 = None
    unsqueeze_796: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 2);  unsqueeze_795 = None
    unsqueeze_797: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, 3);  unsqueeze_796 = None
    mul_824: "f32[160]" = torch.ops.aten.mul.Tensor(sum_97, 0.0001220703125)
    mul_825: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_826: "f32[160]" = torch.ops.aten.mul.Tensor(mul_824, mul_825);  mul_824 = mul_825 = None
    unsqueeze_798: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_826, 0);  mul_826 = None
    unsqueeze_799: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 2);  unsqueeze_798 = None
    unsqueeze_800: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 3);  unsqueeze_799 = None
    mul_827: "f32[160]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_801: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
    unsqueeze_802: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 2);  unsqueeze_801 = None
    unsqueeze_803: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 3);  unsqueeze_802 = None
    mul_828: "f32[8, 160, 32, 32]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_800);  sub_245 = unsqueeze_800 = None
    sub_247: "f32[8, 160, 32, 32]" = torch.ops.aten.sub.Tensor(where_45, mul_828);  where_45 = mul_828 = None
    sub_248: "f32[8, 160, 32, 32]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_797);  sub_247 = unsqueeze_797 = None
    mul_829: "f32[8, 160, 32, 32]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_803);  sub_248 = unsqueeze_803 = None
    mul_830: "f32[160]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_28);  sum_97 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_829, relu_6, primals_124, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_829 = primals_124 = None
    getitem_255: "f32[8, 192, 32, 32]" = convolution_backward_47[0]
    getitem_256: "f32[160, 192, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_317: "f32[8, 192, 32, 32]" = torch.ops.aten.add.Tensor(getitem_246, getitem_255);  getitem_246 = getitem_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:268, code: return self.act(x)
    alias_192: "f32[8, 192, 32, 32]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_193: "f32[8, 192, 32, 32]" = torch.ops.aten.alias.default(alias_192);  alias_192 = None
    le_46: "b8[8, 192, 32, 32]" = torch.ops.aten.le.Scalar(alias_193, 0);  alias_193 = None
    where_46: "f32[8, 192, 32, 32]" = torch.ops.aten.where.self(le_46, full_default, add_317);  le_46 = add_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_98: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_249: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_806);  convolution_8 = unsqueeze_806 = None
    mul_831: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(where_46, sub_249)
    sum_99: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_831, [0, 2, 3]);  mul_831 = None
    mul_832: "f32[192]" = torch.ops.aten.mul.Tensor(sum_98, 0.0001220703125)
    unsqueeze_807: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_832, 0);  mul_832 = None
    unsqueeze_808: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 2);  unsqueeze_807 = None
    unsqueeze_809: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, 3);  unsqueeze_808 = None
    mul_833: "f32[192]" = torch.ops.aten.mul.Tensor(sum_99, 0.0001220703125)
    mul_834: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_835: "f32[192]" = torch.ops.aten.mul.Tensor(mul_833, mul_834);  mul_833 = mul_834 = None
    unsqueeze_810: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_835, 0);  mul_835 = None
    unsqueeze_811: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 2);  unsqueeze_810 = None
    unsqueeze_812: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 3);  unsqueeze_811 = None
    mul_836: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_813: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_836, 0);  mul_836 = None
    unsqueeze_814: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 2);  unsqueeze_813 = None
    unsqueeze_815: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 3);  unsqueeze_814 = None
    mul_837: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_812);  sub_249 = unsqueeze_812 = None
    sub_251: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(where_46, mul_837);  mul_837 = None
    sub_252: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_809);  sub_251 = unsqueeze_809 = None
    mul_838: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_815);  sub_252 = unsqueeze_815 = None
    mul_839: "f32[192]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_25);  sum_99 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_838, relu_5, primals_123, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_838 = primals_123 = None
    getitem_258: "f32[8, 192, 32, 32]" = convolution_backward_48[0]
    getitem_259: "f32[192, 192, 3, 3]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_195: "f32[8, 192, 32, 32]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_196: "f32[8, 192, 32, 32]" = torch.ops.aten.alias.default(alias_195);  alias_195 = None
    le_47: "b8[8, 192, 32, 32]" = torch.ops.aten.le.Scalar(alias_196, 0);  alias_196 = None
    where_47: "f32[8, 192, 32, 32]" = torch.ops.aten.where.self(le_47, full_default, getitem_258);  le_47 = getitem_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_100: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_253: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_818);  convolution_7 = unsqueeze_818 = None
    mul_840: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(where_47, sub_253)
    sum_101: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_840, [0, 2, 3]);  mul_840 = None
    mul_841: "f32[192]" = torch.ops.aten.mul.Tensor(sum_100, 0.0001220703125)
    unsqueeze_819: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_841, 0);  mul_841 = None
    unsqueeze_820: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 2);  unsqueeze_819 = None
    unsqueeze_821: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, 3);  unsqueeze_820 = None
    mul_842: "f32[192]" = torch.ops.aten.mul.Tensor(sum_101, 0.0001220703125)
    mul_843: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_844: "f32[192]" = torch.ops.aten.mul.Tensor(mul_842, mul_843);  mul_842 = mul_843 = None
    unsqueeze_822: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_844, 0);  mul_844 = None
    unsqueeze_823: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 2);  unsqueeze_822 = None
    unsqueeze_824: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 3);  unsqueeze_823 = None
    mul_845: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_825: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
    unsqueeze_826: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 2);  unsqueeze_825 = None
    unsqueeze_827: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 3);  unsqueeze_826 = None
    mul_846: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_824);  sub_253 = unsqueeze_824 = None
    sub_255: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(where_47, mul_846);  where_47 = mul_846 = None
    sub_256: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_821);  sub_255 = unsqueeze_821 = None
    mul_847: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_827);  sub_256 = unsqueeze_827 = None
    mul_848: "f32[192]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_22);  sum_101 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_847, relu_4, primals_122, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_847 = primals_122 = None
    getitem_261: "f32[8, 192, 32, 32]" = convolution_backward_49[0]
    getitem_262: "f32[192, 192, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_318: "f32[8, 192, 32, 32]" = torch.ops.aten.add.Tensor(where_46, getitem_261);  where_46 = getitem_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:268, code: return self.act(x)
    alias_198: "f32[8, 192, 32, 32]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_199: "f32[8, 192, 32, 32]" = torch.ops.aten.alias.default(alias_198);  alias_198 = None
    le_48: "b8[8, 192, 32, 32]" = torch.ops.aten.le.Scalar(alias_199, 0);  alias_199 = None
    where_48: "f32[8, 192, 32, 32]" = torch.ops.aten.where.self(le_48, full_default, add_318);  le_48 = add_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_102: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_257: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_830);  convolution_6 = unsqueeze_830 = None
    mul_849: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(where_48, sub_257)
    sum_103: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_849, [0, 2, 3]);  mul_849 = None
    mul_850: "f32[192]" = torch.ops.aten.mul.Tensor(sum_102, 0.0001220703125)
    unsqueeze_831: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_850, 0);  mul_850 = None
    unsqueeze_832: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 2);  unsqueeze_831 = None
    unsqueeze_833: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, 3);  unsqueeze_832 = None
    mul_851: "f32[192]" = torch.ops.aten.mul.Tensor(sum_103, 0.0001220703125)
    mul_852: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_853: "f32[192]" = torch.ops.aten.mul.Tensor(mul_851, mul_852);  mul_851 = mul_852 = None
    unsqueeze_834: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_853, 0);  mul_853 = None
    unsqueeze_835: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 2);  unsqueeze_834 = None
    unsqueeze_836: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 3);  unsqueeze_835 = None
    mul_854: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_837: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_854, 0);  mul_854 = None
    unsqueeze_838: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 2);  unsqueeze_837 = None
    unsqueeze_839: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 3);  unsqueeze_838 = None
    mul_855: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_836);  sub_257 = unsqueeze_836 = None
    sub_259: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(where_48, mul_855);  mul_855 = None
    sub_260: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_833);  sub_259 = None
    mul_856: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_839);  sub_260 = unsqueeze_839 = None
    mul_857: "f32[192]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_19);  sum_103 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_856, relu_2, primals_121, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_856 = primals_121 = None
    getitem_264: "f32[8, 128, 64, 64]" = convolution_backward_50[0]
    getitem_265: "f32[192, 128, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_261: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_842);  convolution_5 = unsqueeze_842 = None
    mul_858: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(where_48, sub_261)
    sum_105: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_858, [0, 2, 3]);  mul_858 = None
    mul_860: "f32[192]" = torch.ops.aten.mul.Tensor(sum_105, 0.0001220703125)
    mul_861: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_862: "f32[192]" = torch.ops.aten.mul.Tensor(mul_860, mul_861);  mul_860 = mul_861 = None
    unsqueeze_846: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_862, 0);  mul_862 = None
    unsqueeze_847: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 2);  unsqueeze_846 = None
    unsqueeze_848: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 3);  unsqueeze_847 = None
    mul_863: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_849: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
    unsqueeze_850: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 2);  unsqueeze_849 = None
    unsqueeze_851: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 3);  unsqueeze_850 = None
    mul_864: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_848);  sub_261 = unsqueeze_848 = None
    sub_263: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(where_48, mul_864);  where_48 = mul_864 = None
    sub_264: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_833);  sub_263 = unsqueeze_833 = None
    mul_865: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_851);  sub_264 = unsqueeze_851 = None
    mul_866: "f32[192]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_16);  sum_105 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_865, relu_3, primals_120, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_865 = primals_120 = None
    getitem_267: "f32[8, 192, 32, 32]" = convolution_backward_51[0]
    getitem_268: "f32[192, 192, 3, 3]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_201: "f32[8, 192, 32, 32]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_202: "f32[8, 192, 32, 32]" = torch.ops.aten.alias.default(alias_201);  alias_201 = None
    le_49: "b8[8, 192, 32, 32]" = torch.ops.aten.le.Scalar(alias_202, 0);  alias_202 = None
    where_49: "f32[8, 192, 32, 32]" = torch.ops.aten.where.self(le_49, full_default, getitem_267);  le_49 = getitem_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_106: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_265: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_854);  convolution_4 = unsqueeze_854 = None
    mul_867: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(where_49, sub_265)
    sum_107: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_867, [0, 2, 3]);  mul_867 = None
    mul_868: "f32[192]" = torch.ops.aten.mul.Tensor(sum_106, 0.0001220703125)
    unsqueeze_855: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_868, 0);  mul_868 = None
    unsqueeze_856: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 2);  unsqueeze_855 = None
    unsqueeze_857: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, 3);  unsqueeze_856 = None
    mul_869: "f32[192]" = torch.ops.aten.mul.Tensor(sum_107, 0.0001220703125)
    mul_870: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_871: "f32[192]" = torch.ops.aten.mul.Tensor(mul_869, mul_870);  mul_869 = mul_870 = None
    unsqueeze_858: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_871, 0);  mul_871 = None
    unsqueeze_859: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 2);  unsqueeze_858 = None
    unsqueeze_860: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 3);  unsqueeze_859 = None
    mul_872: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_861: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    unsqueeze_862: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 2);  unsqueeze_861 = None
    unsqueeze_863: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 3);  unsqueeze_862 = None
    mul_873: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_860);  sub_265 = unsqueeze_860 = None
    sub_267: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(where_49, mul_873);  where_49 = mul_873 = None
    sub_268: "f32[8, 192, 32, 32]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_857);  sub_267 = unsqueeze_857 = None
    mul_874: "f32[8, 192, 32, 32]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_863);  sub_268 = unsqueeze_863 = None
    mul_875: "f32[192]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_13);  sum_107 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_874, relu_2, primals_119, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_874 = primals_119 = None
    getitem_270: "f32[8, 128, 64, 64]" = convolution_backward_52[0]
    getitem_271: "f32[192, 128, 3, 3]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_319: "f32[8, 128, 64, 64]" = torch.ops.aten.add.Tensor(getitem_264, getitem_270);  getitem_264 = getitem_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/byobnet.py:268, code: return self.act(x)
    alias_204: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_205: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_204);  alias_204 = None
    le_50: "b8[8, 128, 64, 64]" = torch.ops.aten.le.Scalar(alias_205, 0);  alias_205 = None
    where_50: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(le_50, full_default, add_319);  le_50 = add_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_108: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_269: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_866);  convolution_3 = unsqueeze_866 = None
    mul_876: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_50, sub_269)
    sum_109: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_876, [0, 2, 3]);  mul_876 = None
    mul_877: "f32[128]" = torch.ops.aten.mul.Tensor(sum_108, 3.0517578125e-05)
    unsqueeze_867: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_877, 0);  mul_877 = None
    unsqueeze_868: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 2);  unsqueeze_867 = None
    unsqueeze_869: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, 3);  unsqueeze_868 = None
    mul_878: "f32[128]" = torch.ops.aten.mul.Tensor(sum_109, 3.0517578125e-05)
    mul_879: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_880: "f32[128]" = torch.ops.aten.mul.Tensor(mul_878, mul_879);  mul_878 = mul_879 = None
    unsqueeze_870: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_880, 0);  mul_880 = None
    unsqueeze_871: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 2);  unsqueeze_870 = None
    unsqueeze_872: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 3);  unsqueeze_871 = None
    mul_881: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_873: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_874: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 2);  unsqueeze_873 = None
    unsqueeze_875: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 3);  unsqueeze_874 = None
    mul_882: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_872);  sub_269 = unsqueeze_872 = None
    sub_271: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_50, mul_882);  mul_882 = None
    sub_272: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_869);  sub_271 = None
    mul_883: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_875);  sub_272 = unsqueeze_875 = None
    mul_884: "f32[128]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_10);  sum_109 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_883, relu, primals_118, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_883 = primals_118 = None
    getitem_273: "f32[8, 32, 128, 128]" = convolution_backward_53[0]
    getitem_274: "f32[128, 32, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_273: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_878);  convolution_2 = unsqueeze_878 = None
    mul_885: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_50, sub_273)
    sum_111: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_885, [0, 2, 3]);  mul_885 = None
    mul_887: "f32[128]" = torch.ops.aten.mul.Tensor(sum_111, 3.0517578125e-05)
    mul_888: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_889: "f32[128]" = torch.ops.aten.mul.Tensor(mul_887, mul_888);  mul_887 = mul_888 = None
    unsqueeze_882: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_889, 0);  mul_889 = None
    unsqueeze_883: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 2);  unsqueeze_882 = None
    unsqueeze_884: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 3);  unsqueeze_883 = None
    mul_890: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_885: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
    unsqueeze_886: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 2);  unsqueeze_885 = None
    unsqueeze_887: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 3);  unsqueeze_886 = None
    mul_891: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_884);  sub_273 = unsqueeze_884 = None
    sub_275: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_50, mul_891);  where_50 = mul_891 = None
    sub_276: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_869);  sub_275 = unsqueeze_869 = None
    mul_892: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_887);  sub_276 = unsqueeze_887 = None
    mul_893: "f32[128]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_7);  sum_111 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_892, relu_1, primals_117, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_892 = primals_117 = None
    getitem_276: "f32[8, 128, 64, 64]" = convolution_backward_54[0]
    getitem_277: "f32[128, 128, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_207: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_208: "f32[8, 128, 64, 64]" = torch.ops.aten.alias.default(alias_207);  alias_207 = None
    le_51: "b8[8, 128, 64, 64]" = torch.ops.aten.le.Scalar(alias_208, 0);  alias_208 = None
    where_51: "f32[8, 128, 64, 64]" = torch.ops.aten.where.self(le_51, full_default, getitem_276);  le_51 = getitem_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_112: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_277: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_890);  convolution_1 = unsqueeze_890 = None
    mul_894: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(where_51, sub_277)
    sum_113: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_894, [0, 2, 3]);  mul_894 = None
    mul_895: "f32[128]" = torch.ops.aten.mul.Tensor(sum_112, 3.0517578125e-05)
    unsqueeze_891: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_895, 0);  mul_895 = None
    unsqueeze_892: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 2);  unsqueeze_891 = None
    unsqueeze_893: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, 3);  unsqueeze_892 = None
    mul_896: "f32[128]" = torch.ops.aten.mul.Tensor(sum_113, 3.0517578125e-05)
    mul_897: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_898: "f32[128]" = torch.ops.aten.mul.Tensor(mul_896, mul_897);  mul_896 = mul_897 = None
    unsqueeze_894: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_898, 0);  mul_898 = None
    unsqueeze_895: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 2);  unsqueeze_894 = None
    unsqueeze_896: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 3);  unsqueeze_895 = None
    mul_899: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_897: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
    unsqueeze_898: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 2);  unsqueeze_897 = None
    unsqueeze_899: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 3);  unsqueeze_898 = None
    mul_900: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_896);  sub_277 = unsqueeze_896 = None
    sub_279: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(where_51, mul_900);  where_51 = mul_900 = None
    sub_280: "f32[8, 128, 64, 64]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_893);  sub_279 = unsqueeze_893 = None
    mul_901: "f32[8, 128, 64, 64]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_899);  sub_280 = unsqueeze_899 = None
    mul_902: "f32[128]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_4);  sum_113 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_901, relu, primals_116, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_901 = primals_116 = None
    getitem_279: "f32[8, 32, 128, 128]" = convolution_backward_55[0]
    getitem_280: "f32[128, 32, 3, 3]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_320: "f32[8, 32, 128, 128]" = torch.ops.aten.add.Tensor(getitem_273, getitem_279);  getitem_273 = getitem_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_210: "f32[8, 32, 128, 128]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_211: "f32[8, 32, 128, 128]" = torch.ops.aten.alias.default(alias_210);  alias_210 = None
    le_52: "b8[8, 32, 128, 128]" = torch.ops.aten.le.Scalar(alias_211, 0);  alias_211 = None
    where_52: "f32[8, 32, 128, 128]" = torch.ops.aten.where.self(le_52, full_default, add_320);  le_52 = full_default = add_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_114: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_281: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_902);  convolution = unsqueeze_902 = None
    mul_903: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(where_52, sub_281)
    sum_115: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_903, [0, 2, 3]);  mul_903 = None
    mul_904: "f32[32]" = torch.ops.aten.mul.Tensor(sum_114, 7.62939453125e-06)
    unsqueeze_903: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_904, 0);  mul_904 = None
    unsqueeze_904: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 2);  unsqueeze_903 = None
    unsqueeze_905: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, 3);  unsqueeze_904 = None
    mul_905: "f32[32]" = torch.ops.aten.mul.Tensor(sum_115, 7.62939453125e-06)
    mul_906: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_907: "f32[32]" = torch.ops.aten.mul.Tensor(mul_905, mul_906);  mul_905 = mul_906 = None
    unsqueeze_906: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_907, 0);  mul_907 = None
    unsqueeze_907: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 2);  unsqueeze_906 = None
    unsqueeze_908: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 3);  unsqueeze_907 = None
    mul_908: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_909: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
    unsqueeze_910: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 2);  unsqueeze_909 = None
    unsqueeze_911: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 3);  unsqueeze_910 = None
    mul_909: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_908);  sub_281 = unsqueeze_908 = None
    sub_283: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(where_52, mul_909);  where_52 = mul_909 = None
    sub_284: "f32[8, 32, 128, 128]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_905);  sub_283 = unsqueeze_905 = None
    mul_910: "f32[8, 32, 128, 128]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_911);  sub_284 = unsqueeze_911 = None
    mul_911: "f32[32]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_1);  sum_115 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_910, primals_345, primals_115, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_910 = primals_345 = primals_115 = None
    getitem_283: "f32[32, 3, 3, 3]" = convolution_backward_56[1];  convolution_backward_56 = None
    return [mul_911, sum_114, mul_902, sum_112, mul_893, sum_108, mul_884, sum_108, mul_875, sum_106, mul_866, sum_102, mul_857, sum_102, mul_848, sum_100, mul_839, sum_98, mul_830, sum_96, mul_821, sum_94, mul_812, sum_90, mul_803, sum_90, mul_794, sum_88, mul_785, sum_86, mul_776, sum_84, mul_767, sum_82, mul_758, sum_80, mul_749, sum_78, mul_740, sum_76, mul_731, sum_74, mul_722, sum_72, mul_713, sum_70, mul_704, sum_68, mul_695, sum_66, mul_686, sum_64, mul_677, sum_62, mul_668, sum_60, mul_659, sum_58, mul_650, sum_56, mul_641, sum_52, mul_632, sum_52, mul_623, sum_50, mul_614, sum_48, mul_605, sum_46, mul_596, sum_44, mul_587, sum_42, mul_578, sum_40, mul_569, sum_38, mul_560, sum_36, mul_551, sum_34, mul_542, sum_32, mul_533, sum_30, mul_524, sum_28, mul_515, sum_26, mul_506, sum_24, mul_497, sum_22, mul_488, sum_20, mul_479, sum_18, mul_470, sum_16, mul_461, sum_14, mul_452, sum_12, mul_443, sum_10, mul_434, sum_8, mul_425, sum_6, mul_416, sum_4, mul_407, sum_2, getitem_283, getitem_280, getitem_277, getitem_274, getitem_271, getitem_268, getitem_265, getitem_262, getitem_259, getitem_256, getitem_253, getitem_250, getitem_247, getitem_244, getitem_241, getitem_238, getitem_235, getitem_232, getitem_229, getitem_226, getitem_223, getitem_220, getitem_217, getitem_214, getitem_211, getitem_208, getitem_205, getitem_202, getitem_199, getitem_196, getitem_193, getitem_190, getitem_187, getitem_184, getitem_181, getitem_178, getitem_175, getitem_172, getitem_169, getitem_166, getitem_163, getitem_160, getitem_157, getitem_154, getitem_151, getitem_148, getitem_145, getitem_142, getitem_139, getitem_136, getitem_133, getitem_130, getitem_127, getitem_124, getitem_121, getitem_118, getitem_115, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    