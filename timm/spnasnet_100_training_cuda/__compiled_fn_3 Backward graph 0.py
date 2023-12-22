from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_3: "f32[32]", primals_5: "f32[16]", primals_7: "f32[48]", primals_9: "f32[48]", primals_11: "f32[24]", primals_13: "f32[72]", primals_15: "f32[72]", primals_17: "f32[24]", primals_19: "f32[72]", primals_21: "f32[72]", primals_23: "f32[24]", primals_25: "f32[144]", primals_27: "f32[144]", primals_29: "f32[40]", primals_31: "f32[120]", primals_33: "f32[120]", primals_35: "f32[40]", primals_37: "f32[120]", primals_39: "f32[120]", primals_41: "f32[40]", primals_43: "f32[120]", primals_45: "f32[120]", primals_47: "f32[40]", primals_49: "f32[240]", primals_51: "f32[240]", primals_53: "f32[80]", primals_55: "f32[240]", primals_57: "f32[240]", primals_59: "f32[80]", primals_61: "f32[240]", primals_63: "f32[240]", primals_65: "f32[80]", primals_67: "f32[240]", primals_69: "f32[240]", primals_71: "f32[80]", primals_73: "f32[480]", primals_75: "f32[480]", primals_77: "f32[96]", primals_79: "f32[288]", primals_81: "f32[288]", primals_83: "f32[96]", primals_85: "f32[288]", primals_87: "f32[288]", primals_89: "f32[96]", primals_91: "f32[288]", primals_93: "f32[288]", primals_95: "f32[96]", primals_97: "f32[576]", primals_99: "f32[576]", primals_101: "f32[192]", primals_103: "f32[1152]", primals_105: "f32[1152]", primals_107: "f32[192]", primals_109: "f32[1152]", primals_111: "f32[1152]", primals_113: "f32[192]", primals_115: "f32[1152]", primals_117: "f32[1152]", primals_119: "f32[192]", primals_121: "f32[1152]", primals_123: "f32[1152]", primals_125: "f32[320]", primals_127: "f32[1280]", primals_129: "f32[32, 3, 3, 3]", primals_130: "f32[32, 1, 3, 3]", primals_131: "f32[16, 32, 1, 1]", primals_132: "f32[48, 16, 1, 1]", primals_133: "f32[48, 1, 3, 3]", primals_134: "f32[24, 48, 1, 1]", primals_135: "f32[72, 24, 1, 1]", primals_136: "f32[72, 1, 3, 3]", primals_137: "f32[24, 72, 1, 1]", primals_138: "f32[72, 24, 1, 1]", primals_139: "f32[72, 1, 3, 3]", primals_140: "f32[24, 72, 1, 1]", primals_141: "f32[144, 24, 1, 1]", primals_142: "f32[144, 1, 5, 5]", primals_143: "f32[40, 144, 1, 1]", primals_144: "f32[120, 40, 1, 1]", primals_145: "f32[120, 1, 3, 3]", primals_146: "f32[40, 120, 1, 1]", primals_147: "f32[120, 40, 1, 1]", primals_148: "f32[120, 1, 3, 3]", primals_149: "f32[40, 120, 1, 1]", primals_150: "f32[120, 40, 1, 1]", primals_151: "f32[120, 1, 3, 3]", primals_152: "f32[40, 120, 1, 1]", primals_153: "f32[240, 40, 1, 1]", primals_154: "f32[240, 1, 5, 5]", primals_155: "f32[80, 240, 1, 1]", primals_156: "f32[240, 80, 1, 1]", primals_157: "f32[240, 1, 3, 3]", primals_158: "f32[80, 240, 1, 1]", primals_159: "f32[240, 80, 1, 1]", primals_160: "f32[240, 1, 3, 3]", primals_161: "f32[80, 240, 1, 1]", primals_162: "f32[240, 80, 1, 1]", primals_163: "f32[240, 1, 3, 3]", primals_164: "f32[80, 240, 1, 1]", primals_165: "f32[480, 80, 1, 1]", primals_166: "f32[480, 1, 5, 5]", primals_167: "f32[96, 480, 1, 1]", primals_168: "f32[288, 96, 1, 1]", primals_169: "f32[288, 1, 5, 5]", primals_170: "f32[96, 288, 1, 1]", primals_171: "f32[288, 96, 1, 1]", primals_172: "f32[288, 1, 5, 5]", primals_173: "f32[96, 288, 1, 1]", primals_174: "f32[288, 96, 1, 1]", primals_175: "f32[288, 1, 5, 5]", primals_176: "f32[96, 288, 1, 1]", primals_177: "f32[576, 96, 1, 1]", primals_178: "f32[576, 1, 5, 5]", primals_179: "f32[192, 576, 1, 1]", primals_180: "f32[1152, 192, 1, 1]", primals_181: "f32[1152, 1, 5, 5]", primals_182: "f32[192, 1152, 1, 1]", primals_183: "f32[1152, 192, 1, 1]", primals_184: "f32[1152, 1, 5, 5]", primals_185: "f32[192, 1152, 1, 1]", primals_186: "f32[1152, 192, 1, 1]", primals_187: "f32[1152, 1, 5, 5]", primals_188: "f32[192, 1152, 1, 1]", primals_189: "f32[1152, 192, 1, 1]", primals_190: "f32[1152, 1, 3, 3]", primals_191: "f32[320, 1152, 1, 1]", primals_192: "f32[1280, 320, 1, 1]", primals_387: "f32[8, 3, 224, 224]", convolution: "f32[8, 32, 112, 112]", squeeze_1: "f32[32]", relu: "f32[8, 32, 112, 112]", convolution_1: "f32[8, 32, 112, 112]", squeeze_4: "f32[32]", relu_1: "f32[8, 32, 112, 112]", convolution_2: "f32[8, 16, 112, 112]", squeeze_7: "f32[16]", add_14: "f32[8, 16, 112, 112]", convolution_3: "f32[8, 48, 112, 112]", squeeze_10: "f32[48]", relu_2: "f32[8, 48, 112, 112]", convolution_4: "f32[8, 48, 56, 56]", squeeze_13: "f32[48]", relu_3: "f32[8, 48, 56, 56]", convolution_5: "f32[8, 24, 56, 56]", squeeze_16: "f32[24]", add_29: "f32[8, 24, 56, 56]", convolution_6: "f32[8, 72, 56, 56]", squeeze_19: "f32[72]", relu_4: "f32[8, 72, 56, 56]", convolution_7: "f32[8, 72, 56, 56]", squeeze_22: "f32[72]", relu_5: "f32[8, 72, 56, 56]", convolution_8: "f32[8, 24, 56, 56]", squeeze_25: "f32[24]", add_45: "f32[8, 24, 56, 56]", convolution_9: "f32[8, 72, 56, 56]", squeeze_28: "f32[72]", relu_6: "f32[8, 72, 56, 56]", convolution_10: "f32[8, 72, 56, 56]", squeeze_31: "f32[72]", relu_7: "f32[8, 72, 56, 56]", convolution_11: "f32[8, 24, 56, 56]", squeeze_34: "f32[24]", add_61: "f32[8, 24, 56, 56]", convolution_12: "f32[8, 144, 56, 56]", squeeze_37: "f32[144]", relu_8: "f32[8, 144, 56, 56]", convolution_13: "f32[8, 144, 28, 28]", squeeze_40: "f32[144]", relu_9: "f32[8, 144, 28, 28]", convolution_14: "f32[8, 40, 28, 28]", squeeze_43: "f32[40]", add_76: "f32[8, 40, 28, 28]", convolution_15: "f32[8, 120, 28, 28]", squeeze_46: "f32[120]", relu_10: "f32[8, 120, 28, 28]", convolution_16: "f32[8, 120, 28, 28]", squeeze_49: "f32[120]", relu_11: "f32[8, 120, 28, 28]", convolution_17: "f32[8, 40, 28, 28]", squeeze_52: "f32[40]", add_92: "f32[8, 40, 28, 28]", convolution_18: "f32[8, 120, 28, 28]", squeeze_55: "f32[120]", relu_12: "f32[8, 120, 28, 28]", convolution_19: "f32[8, 120, 28, 28]", squeeze_58: "f32[120]", relu_13: "f32[8, 120, 28, 28]", convolution_20: "f32[8, 40, 28, 28]", squeeze_61: "f32[40]", add_108: "f32[8, 40, 28, 28]", convolution_21: "f32[8, 120, 28, 28]", squeeze_64: "f32[120]", relu_14: "f32[8, 120, 28, 28]", convolution_22: "f32[8, 120, 28, 28]", squeeze_67: "f32[120]", relu_15: "f32[8, 120, 28, 28]", convolution_23: "f32[8, 40, 28, 28]", squeeze_70: "f32[40]", add_124: "f32[8, 40, 28, 28]", convolution_24: "f32[8, 240, 28, 28]", squeeze_73: "f32[240]", relu_16: "f32[8, 240, 28, 28]", convolution_25: "f32[8, 240, 14, 14]", squeeze_76: "f32[240]", relu_17: "f32[8, 240, 14, 14]", convolution_26: "f32[8, 80, 14, 14]", squeeze_79: "f32[80]", add_139: "f32[8, 80, 14, 14]", convolution_27: "f32[8, 240, 14, 14]", squeeze_82: "f32[240]", relu_18: "f32[8, 240, 14, 14]", convolution_28: "f32[8, 240, 14, 14]", squeeze_85: "f32[240]", relu_19: "f32[8, 240, 14, 14]", convolution_29: "f32[8, 80, 14, 14]", squeeze_88: "f32[80]", add_155: "f32[8, 80, 14, 14]", convolution_30: "f32[8, 240, 14, 14]", squeeze_91: "f32[240]", relu_20: "f32[8, 240, 14, 14]", convolution_31: "f32[8, 240, 14, 14]", squeeze_94: "f32[240]", relu_21: "f32[8, 240, 14, 14]", convolution_32: "f32[8, 80, 14, 14]", squeeze_97: "f32[80]", add_171: "f32[8, 80, 14, 14]", convolution_33: "f32[8, 240, 14, 14]", squeeze_100: "f32[240]", relu_22: "f32[8, 240, 14, 14]", convolution_34: "f32[8, 240, 14, 14]", squeeze_103: "f32[240]", relu_23: "f32[8, 240, 14, 14]", convolution_35: "f32[8, 80, 14, 14]", squeeze_106: "f32[80]", add_187: "f32[8, 80, 14, 14]", convolution_36: "f32[8, 480, 14, 14]", squeeze_109: "f32[480]", relu_24: "f32[8, 480, 14, 14]", convolution_37: "f32[8, 480, 14, 14]", squeeze_112: "f32[480]", relu_25: "f32[8, 480, 14, 14]", convolution_38: "f32[8, 96, 14, 14]", squeeze_115: "f32[96]", add_202: "f32[8, 96, 14, 14]", convolution_39: "f32[8, 288, 14, 14]", squeeze_118: "f32[288]", relu_26: "f32[8, 288, 14, 14]", convolution_40: "f32[8, 288, 14, 14]", squeeze_121: "f32[288]", relu_27: "f32[8, 288, 14, 14]", convolution_41: "f32[8, 96, 14, 14]", squeeze_124: "f32[96]", add_218: "f32[8, 96, 14, 14]", convolution_42: "f32[8, 288, 14, 14]", squeeze_127: "f32[288]", relu_28: "f32[8, 288, 14, 14]", convolution_43: "f32[8, 288, 14, 14]", squeeze_130: "f32[288]", relu_29: "f32[8, 288, 14, 14]", convolution_44: "f32[8, 96, 14, 14]", squeeze_133: "f32[96]", add_234: "f32[8, 96, 14, 14]", convolution_45: "f32[8, 288, 14, 14]", squeeze_136: "f32[288]", relu_30: "f32[8, 288, 14, 14]", convolution_46: "f32[8, 288, 14, 14]", squeeze_139: "f32[288]", relu_31: "f32[8, 288, 14, 14]", convolution_47: "f32[8, 96, 14, 14]", squeeze_142: "f32[96]", add_250: "f32[8, 96, 14, 14]", convolution_48: "f32[8, 576, 14, 14]", squeeze_145: "f32[576]", relu_32: "f32[8, 576, 14, 14]", convolution_49: "f32[8, 576, 7, 7]", squeeze_148: "f32[576]", relu_33: "f32[8, 576, 7, 7]", convolution_50: "f32[8, 192, 7, 7]", squeeze_151: "f32[192]", add_265: "f32[8, 192, 7, 7]", convolution_51: "f32[8, 1152, 7, 7]", squeeze_154: "f32[1152]", relu_34: "f32[8, 1152, 7, 7]", convolution_52: "f32[8, 1152, 7, 7]", squeeze_157: "f32[1152]", relu_35: "f32[8, 1152, 7, 7]", convolution_53: "f32[8, 192, 7, 7]", squeeze_160: "f32[192]", add_281: "f32[8, 192, 7, 7]", convolution_54: "f32[8, 1152, 7, 7]", squeeze_163: "f32[1152]", relu_36: "f32[8, 1152, 7, 7]", convolution_55: "f32[8, 1152, 7, 7]", squeeze_166: "f32[1152]", relu_37: "f32[8, 1152, 7, 7]", convolution_56: "f32[8, 192, 7, 7]", squeeze_169: "f32[192]", add_297: "f32[8, 192, 7, 7]", convolution_57: "f32[8, 1152, 7, 7]", squeeze_172: "f32[1152]", relu_38: "f32[8, 1152, 7, 7]", convolution_58: "f32[8, 1152, 7, 7]", squeeze_175: "f32[1152]", relu_39: "f32[8, 1152, 7, 7]", convolution_59: "f32[8, 192, 7, 7]", squeeze_178: "f32[192]", add_313: "f32[8, 192, 7, 7]", convolution_60: "f32[8, 1152, 7, 7]", squeeze_181: "f32[1152]", relu_40: "f32[8, 1152, 7, 7]", convolution_61: "f32[8, 1152, 7, 7]", squeeze_184: "f32[1152]", relu_41: "f32[8, 1152, 7, 7]", convolution_62: "f32[8, 320, 7, 7]", squeeze_187: "f32[320]", add_328: "f32[8, 320, 7, 7]", convolution_63: "f32[8, 1280, 7, 7]", squeeze_190: "f32[1280]", view: "f32[8, 1280]", permute_1: "f32[1000, 1280]", le: "b8[8, 1280, 7, 7]", unsqueeze_258: "f32[1, 1280, 1, 1]", unsqueeze_270: "f32[1, 320, 1, 1]", unsqueeze_282: "f32[1, 1152, 1, 1]", unsqueeze_294: "f32[1, 1152, 1, 1]", unsqueeze_306: "f32[1, 192, 1, 1]", unsqueeze_318: "f32[1, 1152, 1, 1]", unsqueeze_330: "f32[1, 1152, 1, 1]", unsqueeze_342: "f32[1, 192, 1, 1]", unsqueeze_354: "f32[1, 1152, 1, 1]", unsqueeze_366: "f32[1, 1152, 1, 1]", unsqueeze_378: "f32[1, 192, 1, 1]", unsqueeze_390: "f32[1, 1152, 1, 1]", unsqueeze_402: "f32[1, 1152, 1, 1]", unsqueeze_414: "f32[1, 192, 1, 1]", unsqueeze_426: "f32[1, 576, 1, 1]", unsqueeze_438: "f32[1, 576, 1, 1]", unsqueeze_450: "f32[1, 96, 1, 1]", unsqueeze_462: "f32[1, 288, 1, 1]", unsqueeze_474: "f32[1, 288, 1, 1]", unsqueeze_486: "f32[1, 96, 1, 1]", unsqueeze_498: "f32[1, 288, 1, 1]", unsqueeze_510: "f32[1, 288, 1, 1]", unsqueeze_522: "f32[1, 96, 1, 1]", unsqueeze_534: "f32[1, 288, 1, 1]", unsqueeze_546: "f32[1, 288, 1, 1]", unsqueeze_558: "f32[1, 96, 1, 1]", unsqueeze_570: "f32[1, 480, 1, 1]", unsqueeze_582: "f32[1, 480, 1, 1]", unsqueeze_594: "f32[1, 80, 1, 1]", unsqueeze_606: "f32[1, 240, 1, 1]", unsqueeze_618: "f32[1, 240, 1, 1]", unsqueeze_630: "f32[1, 80, 1, 1]", unsqueeze_642: "f32[1, 240, 1, 1]", unsqueeze_654: "f32[1, 240, 1, 1]", unsqueeze_666: "f32[1, 80, 1, 1]", unsqueeze_678: "f32[1, 240, 1, 1]", unsqueeze_690: "f32[1, 240, 1, 1]", unsqueeze_702: "f32[1, 80, 1, 1]", unsqueeze_714: "f32[1, 240, 1, 1]", unsqueeze_726: "f32[1, 240, 1, 1]", unsqueeze_738: "f32[1, 40, 1, 1]", unsqueeze_750: "f32[1, 120, 1, 1]", unsqueeze_762: "f32[1, 120, 1, 1]", unsqueeze_774: "f32[1, 40, 1, 1]", unsqueeze_786: "f32[1, 120, 1, 1]", unsqueeze_798: "f32[1, 120, 1, 1]", unsqueeze_810: "f32[1, 40, 1, 1]", unsqueeze_822: "f32[1, 120, 1, 1]", unsqueeze_834: "f32[1, 120, 1, 1]", unsqueeze_846: "f32[1, 40, 1, 1]", unsqueeze_858: "f32[1, 144, 1, 1]", unsqueeze_870: "f32[1, 144, 1, 1]", unsqueeze_882: "f32[1, 24, 1, 1]", unsqueeze_894: "f32[1, 72, 1, 1]", unsqueeze_906: "f32[1, 72, 1, 1]", unsqueeze_918: "f32[1, 24, 1, 1]", unsqueeze_930: "f32[1, 72, 1, 1]", unsqueeze_942: "f32[1, 72, 1, 1]", unsqueeze_954: "f32[1, 24, 1, 1]", unsqueeze_966: "f32[1, 48, 1, 1]", unsqueeze_978: "f32[1, 48, 1, 1]", unsqueeze_990: "f32[1, 16, 1, 1]", unsqueeze_1002: "f32[1, 32, 1, 1]", unsqueeze_1014: "f32[1, 32, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    mm: "f32[8, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1280, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1280, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1280, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 1280, 7, 7]);  view_2 = None
    div: "f32[8, 1280, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 1280, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_64: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_258);  convolution_63 = unsqueeze_258 = None
    mul_448: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_64)
    sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 2, 3]);  mul_448 = None
    mul_449: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_259: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
    unsqueeze_260: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 2);  unsqueeze_259 = None
    unsqueeze_261: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, 3);  unsqueeze_260 = None
    mul_450: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_451: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_452: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_450, mul_451);  mul_450 = mul_451 = None
    unsqueeze_262: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_452, 0);  mul_452 = None
    unsqueeze_263: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, 2);  unsqueeze_262 = None
    unsqueeze_264: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 3);  unsqueeze_263 = None
    mul_453: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_127);  primals_127 = None
    unsqueeze_265: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_453, 0);  mul_453 = None
    unsqueeze_266: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_265, 2);  unsqueeze_265 = None
    unsqueeze_267: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 3);  unsqueeze_266 = None
    mul_454: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_264);  sub_64 = unsqueeze_264 = None
    sub_66: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_454);  where = mul_454 = None
    sub_67: "f32[8, 1280, 7, 7]" = torch.ops.aten.sub.Tensor(sub_66, unsqueeze_261);  sub_66 = unsqueeze_261 = None
    mul_455: "f32[8, 1280, 7, 7]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_267);  sub_67 = unsqueeze_267 = None
    mul_456: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_190);  sum_3 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_455, add_328, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_455 = add_328 = primals_192 = None
    getitem_128: "f32[8, 320, 7, 7]" = convolution_backward[0]
    getitem_129: "f32[1280, 320, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[320]" = torch.ops.aten.sum.dim_IntList(getitem_128, [0, 2, 3])
    sub_68: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_270);  convolution_62 = unsqueeze_270 = None
    mul_457: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_128, sub_68)
    sum_5: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2, 3]);  mul_457 = None
    mul_458: "f32[320]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_271: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_272: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_271, 2);  unsqueeze_271 = None
    unsqueeze_273: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, 3);  unsqueeze_272 = None
    mul_459: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_460: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_461: "f32[320]" = torch.ops.aten.mul.Tensor(mul_459, mul_460);  mul_459 = mul_460 = None
    unsqueeze_274: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_461, 0);  mul_461 = None
    unsqueeze_275: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, 2);  unsqueeze_274 = None
    unsqueeze_276: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 3);  unsqueeze_275 = None
    mul_462: "f32[320]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_125);  primals_125 = None
    unsqueeze_277: "f32[1, 320]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_278: "f32[1, 320, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_277, 2);  unsqueeze_277 = None
    unsqueeze_279: "f32[1, 320, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 3);  unsqueeze_278 = None
    mul_463: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_276);  sub_68 = unsqueeze_276 = None
    sub_70: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_128, mul_463);  getitem_128 = mul_463 = None
    sub_71: "f32[8, 320, 7, 7]" = torch.ops.aten.sub.Tensor(sub_70, unsqueeze_273);  sub_70 = unsqueeze_273 = None
    mul_464: "f32[8, 320, 7, 7]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_279);  sub_71 = unsqueeze_279 = None
    mul_465: "f32[320]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_187);  sum_5 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_464, relu_41, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_464 = primals_191 = None
    getitem_131: "f32[8, 1152, 7, 7]" = convolution_backward_1[0]
    getitem_132: "f32[320, 1152, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_47: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_48: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_47);  alias_47 = None
    le_1: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_48, 0);  alias_48 = None
    where_1: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, getitem_131);  le_1 = getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_6: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_72: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_282);  convolution_61 = unsqueeze_282 = None
    mul_466: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_72)
    sum_7: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_466, [0, 2, 3]);  mul_466 = None
    mul_467: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_283: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_467, 0);  mul_467 = None
    unsqueeze_284: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_283, 2);  unsqueeze_283 = None
    unsqueeze_285: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 3);  unsqueeze_284 = None
    mul_468: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_469: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_470: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_468, mul_469);  mul_468 = mul_469 = None
    unsqueeze_286: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_470, 0);  mul_470 = None
    unsqueeze_287: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, 2);  unsqueeze_286 = None
    unsqueeze_288: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 3);  unsqueeze_287 = None
    mul_471: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_123);  primals_123 = None
    unsqueeze_289: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_290: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_289, 2);  unsqueeze_289 = None
    unsqueeze_291: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 3);  unsqueeze_290 = None
    mul_472: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_288);  sub_72 = unsqueeze_288 = None
    sub_74: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_472);  where_1 = mul_472 = None
    sub_75: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_74, unsqueeze_285);  sub_74 = unsqueeze_285 = None
    mul_473: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_291);  sub_75 = unsqueeze_291 = None
    mul_474: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_184);  sum_7 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_473, relu_40, primals_190, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_473 = primals_190 = None
    getitem_134: "f32[8, 1152, 7, 7]" = convolution_backward_2[0]
    getitem_135: "f32[1152, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_50: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
    alias_51: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_50);  alias_50 = None
    le_2: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_51, 0);  alias_51 = None
    where_2: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, getitem_134);  le_2 = getitem_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_8: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_76: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_294);  convolution_60 = unsqueeze_294 = None
    mul_475: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_76)
    sum_9: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 2, 3]);  mul_475 = None
    mul_476: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_295: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_476, 0);  mul_476 = None
    unsqueeze_296: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_295, 2);  unsqueeze_295 = None
    unsqueeze_297: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, 3);  unsqueeze_296 = None
    mul_477: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_478: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_479: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
    unsqueeze_298: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_479, 0);  mul_479 = None
    unsqueeze_299: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, 2);  unsqueeze_298 = None
    unsqueeze_300: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 3);  unsqueeze_299 = None
    mul_480: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_121);  primals_121 = None
    unsqueeze_301: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_302: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 2);  unsqueeze_301 = None
    unsqueeze_303: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 3);  unsqueeze_302 = None
    mul_481: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_300);  sub_76 = unsqueeze_300 = None
    sub_78: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_481);  where_2 = mul_481 = None
    sub_79: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_78, unsqueeze_297);  sub_78 = unsqueeze_297 = None
    mul_482: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_303);  sub_79 = unsqueeze_303 = None
    mul_483: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_181);  sum_9 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_482, add_313, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_482 = add_313 = primals_189 = None
    getitem_137: "f32[8, 192, 7, 7]" = convolution_backward_3[0]
    getitem_138: "f32[1152, 192, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_10: "f32[192]" = torch.ops.aten.sum.dim_IntList(getitem_137, [0, 2, 3])
    sub_80: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_306);  convolution_59 = unsqueeze_306 = None
    mul_484: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_137, sub_80)
    sum_11: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_484, [0, 2, 3]);  mul_484 = None
    mul_485: "f32[192]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_307: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
    unsqueeze_308: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_307, 2);  unsqueeze_307 = None
    unsqueeze_309: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, 3);  unsqueeze_308 = None
    mul_486: "f32[192]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_487: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_488: "f32[192]" = torch.ops.aten.mul.Tensor(mul_486, mul_487);  mul_486 = mul_487 = None
    unsqueeze_310: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_488, 0);  mul_488 = None
    unsqueeze_311: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, 2);  unsqueeze_310 = None
    unsqueeze_312: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 3);  unsqueeze_311 = None
    mul_489: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_119);  primals_119 = None
    unsqueeze_313: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_489, 0);  mul_489 = None
    unsqueeze_314: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 2);  unsqueeze_313 = None
    unsqueeze_315: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 3);  unsqueeze_314 = None
    mul_490: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_312);  sub_80 = unsqueeze_312 = None
    sub_82: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_137, mul_490);  mul_490 = None
    sub_83: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_82, unsqueeze_309);  sub_82 = unsqueeze_309 = None
    mul_491: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_315);  sub_83 = unsqueeze_315 = None
    mul_492: "f32[192]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_178);  sum_11 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_491, relu_39, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_491 = primals_188 = None
    getitem_140: "f32[8, 1152, 7, 7]" = convolution_backward_4[0]
    getitem_141: "f32[192, 1152, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_53: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_54: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_53);  alias_53 = None
    le_3: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_54, 0);  alias_54 = None
    where_3: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, getitem_140);  le_3 = getitem_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_12: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_84: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_318);  convolution_58 = unsqueeze_318 = None
    mul_493: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_84)
    sum_13: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_493, [0, 2, 3]);  mul_493 = None
    mul_494: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_319: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_320: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_319, 2);  unsqueeze_319 = None
    unsqueeze_321: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, 3);  unsqueeze_320 = None
    mul_495: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_496: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_497: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_495, mul_496);  mul_495 = mul_496 = None
    unsqueeze_322: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_497, 0);  mul_497 = None
    unsqueeze_323: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, 2);  unsqueeze_322 = None
    unsqueeze_324: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 3);  unsqueeze_323 = None
    mul_498: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_117);  primals_117 = None
    unsqueeze_325: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_498, 0);  mul_498 = None
    unsqueeze_326: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_325, 2);  unsqueeze_325 = None
    unsqueeze_327: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 3);  unsqueeze_326 = None
    mul_499: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_324);  sub_84 = unsqueeze_324 = None
    sub_86: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_499);  where_3 = mul_499 = None
    sub_87: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_86, unsqueeze_321);  sub_86 = unsqueeze_321 = None
    mul_500: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_327);  sub_87 = unsqueeze_327 = None
    mul_501: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_175);  sum_13 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_500, relu_38, primals_187, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_500 = primals_187 = None
    getitem_143: "f32[8, 1152, 7, 7]" = convolution_backward_5[0]
    getitem_144: "f32[1152, 1, 5, 5]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_56: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_57: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_56);  alias_56 = None
    le_4: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_57, 0);  alias_57 = None
    where_4: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, getitem_143);  le_4 = getitem_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_14: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_88: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_330);  convolution_57 = unsqueeze_330 = None
    mul_502: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_88)
    sum_15: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_502, [0, 2, 3]);  mul_502 = None
    mul_503: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_331: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    unsqueeze_332: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_331, 2);  unsqueeze_331 = None
    unsqueeze_333: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, 3);  unsqueeze_332 = None
    mul_504: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_505: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_506: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_504, mul_505);  mul_504 = mul_505 = None
    unsqueeze_334: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_506, 0);  mul_506 = None
    unsqueeze_335: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 2);  unsqueeze_334 = None
    unsqueeze_336: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 3);  unsqueeze_335 = None
    mul_507: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_115);  primals_115 = None
    unsqueeze_337: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
    unsqueeze_338: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_337, 2);  unsqueeze_337 = None
    unsqueeze_339: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 3);  unsqueeze_338 = None
    mul_508: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_336);  sub_88 = unsqueeze_336 = None
    sub_90: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_508);  where_4 = mul_508 = None
    sub_91: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_333);  sub_90 = unsqueeze_333 = None
    mul_509: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_339);  sub_91 = unsqueeze_339 = None
    mul_510: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_172);  sum_15 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_509, add_297, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_509 = add_297 = primals_186 = None
    getitem_146: "f32[8, 192, 7, 7]" = convolution_backward_6[0]
    getitem_147: "f32[1152, 192, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_334: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(getitem_137, getitem_146);  getitem_137 = getitem_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_16: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_334, [0, 2, 3])
    sub_92: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_342);  convolution_56 = unsqueeze_342 = None
    mul_511: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_334, sub_92)
    sum_17: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 2, 3]);  mul_511 = None
    mul_512: "f32[192]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_343: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    unsqueeze_344: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_343, 2);  unsqueeze_343 = None
    unsqueeze_345: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, 3);  unsqueeze_344 = None
    mul_513: "f32[192]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_514: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_515: "f32[192]" = torch.ops.aten.mul.Tensor(mul_513, mul_514);  mul_513 = mul_514 = None
    unsqueeze_346: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_515, 0);  mul_515 = None
    unsqueeze_347: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, 2);  unsqueeze_346 = None
    unsqueeze_348: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 3);  unsqueeze_347 = None
    mul_516: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_349: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
    unsqueeze_350: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_349, 2);  unsqueeze_349 = None
    unsqueeze_351: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 3);  unsqueeze_350 = None
    mul_517: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_348);  sub_92 = unsqueeze_348 = None
    sub_94: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_334, mul_517);  mul_517 = None
    sub_95: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_345);  sub_94 = unsqueeze_345 = None
    mul_518: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_351);  sub_95 = unsqueeze_351 = None
    mul_519: "f32[192]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_169);  sum_17 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_518, relu_37, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_518 = primals_185 = None
    getitem_149: "f32[8, 1152, 7, 7]" = convolution_backward_7[0]
    getitem_150: "f32[192, 1152, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_59: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_60: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_59);  alias_59 = None
    le_5: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_60, 0);  alias_60 = None
    where_5: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, getitem_149);  le_5 = getitem_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_96: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_354);  convolution_55 = unsqueeze_354 = None
    mul_520: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_96)
    sum_19: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_520, [0, 2, 3]);  mul_520 = None
    mul_521: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_355: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_356: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 2);  unsqueeze_355 = None
    unsqueeze_357: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, 3);  unsqueeze_356 = None
    mul_522: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_523: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_524: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_522, mul_523);  mul_522 = mul_523 = None
    unsqueeze_358: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_524, 0);  mul_524 = None
    unsqueeze_359: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, 2);  unsqueeze_358 = None
    unsqueeze_360: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 3);  unsqueeze_359 = None
    mul_525: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_361: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_525, 0);  mul_525 = None
    unsqueeze_362: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_361, 2);  unsqueeze_361 = None
    unsqueeze_363: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 3);  unsqueeze_362 = None
    mul_526: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_360);  sub_96 = unsqueeze_360 = None
    sub_98: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_526);  where_5 = mul_526 = None
    sub_99: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_357);  sub_98 = unsqueeze_357 = None
    mul_527: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_363);  sub_99 = unsqueeze_363 = None
    mul_528: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_166);  sum_19 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_527, relu_36, primals_184, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_527 = primals_184 = None
    getitem_152: "f32[8, 1152, 7, 7]" = convolution_backward_8[0]
    getitem_153: "f32[1152, 1, 5, 5]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_62: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_63: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_62);  alias_62 = None
    le_6: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_63, 0);  alias_63 = None
    where_6: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, getitem_152);  le_6 = getitem_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_20: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_100: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_366);  convolution_54 = unsqueeze_366 = None
    mul_529: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_100)
    sum_21: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_529, [0, 2, 3]);  mul_529 = None
    mul_530: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_367: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
    unsqueeze_368: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 2);  unsqueeze_367 = None
    unsqueeze_369: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 3);  unsqueeze_368 = None
    mul_531: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_532: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_533: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_531, mul_532);  mul_531 = mul_532 = None
    unsqueeze_370: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
    unsqueeze_371: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, 2);  unsqueeze_370 = None
    unsqueeze_372: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 3);  unsqueeze_371 = None
    mul_534: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_373: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
    unsqueeze_374: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_373, 2);  unsqueeze_373 = None
    unsqueeze_375: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 3);  unsqueeze_374 = None
    mul_535: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_372);  sub_100 = unsqueeze_372 = None
    sub_102: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_535);  where_6 = mul_535 = None
    sub_103: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_102, unsqueeze_369);  sub_102 = unsqueeze_369 = None
    mul_536: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_375);  sub_103 = unsqueeze_375 = None
    mul_537: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_163);  sum_21 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_536, add_281, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_536 = add_281 = primals_183 = None
    getitem_155: "f32[8, 192, 7, 7]" = convolution_backward_9[0]
    getitem_156: "f32[1152, 192, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_335: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_334, getitem_155);  add_334 = getitem_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_22: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_335, [0, 2, 3])
    sub_104: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_378);  convolution_53 = unsqueeze_378 = None
    mul_538: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_335, sub_104)
    sum_23: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_538, [0, 2, 3]);  mul_538 = None
    mul_539: "f32[192]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    unsqueeze_379: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_380: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_379, 2);  unsqueeze_379 = None
    unsqueeze_381: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 3);  unsqueeze_380 = None
    mul_540: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    mul_541: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_542: "f32[192]" = torch.ops.aten.mul.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    unsqueeze_382: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
    unsqueeze_383: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, 2);  unsqueeze_382 = None
    unsqueeze_384: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 3);  unsqueeze_383 = None
    mul_543: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_385: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_543, 0);  mul_543 = None
    unsqueeze_386: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_385, 2);  unsqueeze_385 = None
    unsqueeze_387: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 3);  unsqueeze_386 = None
    mul_544: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_384);  sub_104 = unsqueeze_384 = None
    sub_106: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_335, mul_544);  mul_544 = None
    sub_107: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_106, unsqueeze_381);  sub_106 = unsqueeze_381 = None
    mul_545: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_387);  sub_107 = unsqueeze_387 = None
    mul_546: "f32[192]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_160);  sum_23 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_545, relu_35, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_545 = primals_182 = None
    getitem_158: "f32[8, 1152, 7, 7]" = convolution_backward_10[0]
    getitem_159: "f32[192, 1152, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_65: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_66: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_65);  alias_65 = None
    le_7: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_66, 0);  alias_66 = None
    where_7: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, getitem_158);  le_7 = getitem_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_24: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_108: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_390);  convolution_52 = unsqueeze_390 = None
    mul_547: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_108)
    sum_25: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3]);  mul_547 = None
    mul_548: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    unsqueeze_391: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_392: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_391, 2);  unsqueeze_391 = None
    unsqueeze_393: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 3);  unsqueeze_392 = None
    mul_549: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    mul_550: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_551: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
    unsqueeze_394: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
    unsqueeze_395: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, 2);  unsqueeze_394 = None
    unsqueeze_396: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 3);  unsqueeze_395 = None
    mul_552: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_397: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_398: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_397, 2);  unsqueeze_397 = None
    unsqueeze_399: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 3);  unsqueeze_398 = None
    mul_553: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_396);  sub_108 = unsqueeze_396 = None
    sub_110: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_553);  where_7 = mul_553 = None
    sub_111: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_110, unsqueeze_393);  sub_110 = unsqueeze_393 = None
    mul_554: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_399);  sub_111 = unsqueeze_399 = None
    mul_555: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_157);  sum_25 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_554, relu_34, primals_181, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1152, [True, True, False]);  mul_554 = primals_181 = None
    getitem_161: "f32[8, 1152, 7, 7]" = convolution_backward_11[0]
    getitem_162: "f32[1152, 1, 5, 5]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_68: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_69: "f32[8, 1152, 7, 7]" = torch.ops.aten.alias.default(alias_68);  alias_68 = None
    le_8: "b8[8, 1152, 7, 7]" = torch.ops.aten.le.Scalar(alias_69, 0);  alias_69 = None
    where_8: "f32[8, 1152, 7, 7]" = torch.ops.aten.where.self(le_8, full_default, getitem_161);  le_8 = getitem_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_26: "f32[1152]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_112: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_402);  convolution_51 = unsqueeze_402 = None
    mul_556: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_112)
    sum_27: "f32[1152]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 2, 3]);  mul_556 = None
    mul_557: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    unsqueeze_403: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_557, 0);  mul_557 = None
    unsqueeze_404: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 2);  unsqueeze_403 = None
    unsqueeze_405: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 3);  unsqueeze_404 = None
    mul_558: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    mul_559: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_560: "f32[1152]" = torch.ops.aten.mul.Tensor(mul_558, mul_559);  mul_558 = mul_559 = None
    unsqueeze_406: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
    unsqueeze_407: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, 2);  unsqueeze_406 = None
    unsqueeze_408: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 3);  unsqueeze_407 = None
    mul_561: "f32[1152]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_409: "f32[1, 1152]" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
    unsqueeze_410: "f32[1, 1152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 2);  unsqueeze_409 = None
    unsqueeze_411: "f32[1, 1152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 3);  unsqueeze_410 = None
    mul_562: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_408);  sub_112 = unsqueeze_408 = None
    sub_114: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_562);  where_8 = mul_562 = None
    sub_115: "f32[8, 1152, 7, 7]" = torch.ops.aten.sub.Tensor(sub_114, unsqueeze_405);  sub_114 = unsqueeze_405 = None
    mul_563: "f32[8, 1152, 7, 7]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_411);  sub_115 = unsqueeze_411 = None
    mul_564: "f32[1152]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_154);  sum_27 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_563, add_265, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_563 = add_265 = primals_180 = None
    getitem_164: "f32[8, 192, 7, 7]" = convolution_backward_12[0]
    getitem_165: "f32[1152, 192, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_336: "f32[8, 192, 7, 7]" = torch.ops.aten.add.Tensor(add_335, getitem_164);  add_335 = getitem_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_28: "f32[192]" = torch.ops.aten.sum.dim_IntList(add_336, [0, 2, 3])
    sub_116: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_414);  convolution_50 = unsqueeze_414 = None
    mul_565: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(add_336, sub_116)
    sum_29: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_565, [0, 2, 3]);  mul_565 = None
    mul_566: "f32[192]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_415: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_566, 0);  mul_566 = None
    unsqueeze_416: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_415, 2);  unsqueeze_415 = None
    unsqueeze_417: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 3);  unsqueeze_416 = None
    mul_567: "f32[192]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_568: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_569: "f32[192]" = torch.ops.aten.mul.Tensor(mul_567, mul_568);  mul_567 = mul_568 = None
    unsqueeze_418: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_569, 0);  mul_569 = None
    unsqueeze_419: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, 2);  unsqueeze_418 = None
    unsqueeze_420: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 3);  unsqueeze_419 = None
    mul_570: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_421: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_570, 0);  mul_570 = None
    unsqueeze_422: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_421, 2);  unsqueeze_421 = None
    unsqueeze_423: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 3);  unsqueeze_422 = None
    mul_571: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_420);  sub_116 = unsqueeze_420 = None
    sub_118: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(add_336, mul_571);  add_336 = mul_571 = None
    sub_119: "f32[8, 192, 7, 7]" = torch.ops.aten.sub.Tensor(sub_118, unsqueeze_417);  sub_118 = unsqueeze_417 = None
    mul_572: "f32[8, 192, 7, 7]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_423);  sub_119 = unsqueeze_423 = None
    mul_573: "f32[192]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_151);  sum_29 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_572, relu_33, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_572 = primals_179 = None
    getitem_167: "f32[8, 576, 7, 7]" = convolution_backward_13[0]
    getitem_168: "f32[192, 576, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_71: "f32[8, 576, 7, 7]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_72: "f32[8, 576, 7, 7]" = torch.ops.aten.alias.default(alias_71);  alias_71 = None
    le_9: "b8[8, 576, 7, 7]" = torch.ops.aten.le.Scalar(alias_72, 0);  alias_72 = None
    where_9: "f32[8, 576, 7, 7]" = torch.ops.aten.where.self(le_9, full_default, getitem_167);  le_9 = getitem_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_30: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_120: "f32[8, 576, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_426);  convolution_49 = unsqueeze_426 = None
    mul_574: "f32[8, 576, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_120)
    sum_31: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_574, [0, 2, 3]);  mul_574 = None
    mul_575: "f32[576]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_427: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_575, 0);  mul_575 = None
    unsqueeze_428: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_427, 2);  unsqueeze_427 = None
    unsqueeze_429: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 3);  unsqueeze_428 = None
    mul_576: "f32[576]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_577: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_578: "f32[576]" = torch.ops.aten.mul.Tensor(mul_576, mul_577);  mul_576 = mul_577 = None
    unsqueeze_430: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_578, 0);  mul_578 = None
    unsqueeze_431: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 2);  unsqueeze_430 = None
    unsqueeze_432: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 3);  unsqueeze_431 = None
    mul_579: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_433: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_434: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_433, 2);  unsqueeze_433 = None
    unsqueeze_435: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 3);  unsqueeze_434 = None
    mul_580: "f32[8, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_432);  sub_120 = unsqueeze_432 = None
    sub_122: "f32[8, 576, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_580);  where_9 = mul_580 = None
    sub_123: "f32[8, 576, 7, 7]" = torch.ops.aten.sub.Tensor(sub_122, unsqueeze_429);  sub_122 = unsqueeze_429 = None
    mul_581: "f32[8, 576, 7, 7]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_435);  sub_123 = unsqueeze_435 = None
    mul_582: "f32[576]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_148);  sum_31 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_581, relu_32, primals_178, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 576, [True, True, False]);  mul_581 = primals_178 = None
    getitem_170: "f32[8, 576, 14, 14]" = convolution_backward_14[0]
    getitem_171: "f32[576, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_74: "f32[8, 576, 14, 14]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_75: "f32[8, 576, 14, 14]" = torch.ops.aten.alias.default(alias_74);  alias_74 = None
    le_10: "b8[8, 576, 14, 14]" = torch.ops.aten.le.Scalar(alias_75, 0);  alias_75 = None
    where_10: "f32[8, 576, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, getitem_170);  le_10 = getitem_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_32: "f32[576]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_124: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_438);  convolution_48 = unsqueeze_438 = None
    mul_583: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_124)
    sum_33: "f32[576]" = torch.ops.aten.sum.dim_IntList(mul_583, [0, 2, 3]);  mul_583 = None
    mul_584: "f32[576]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_439: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_584, 0);  mul_584 = None
    unsqueeze_440: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_439, 2);  unsqueeze_439 = None
    unsqueeze_441: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 3);  unsqueeze_440 = None
    mul_585: "f32[576]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_586: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_587: "f32[576]" = torch.ops.aten.mul.Tensor(mul_585, mul_586);  mul_585 = mul_586 = None
    unsqueeze_442: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_587, 0);  mul_587 = None
    unsqueeze_443: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 2);  unsqueeze_442 = None
    unsqueeze_444: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 3);  unsqueeze_443 = None
    mul_588: "f32[576]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_445: "f32[1, 576]" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
    unsqueeze_446: "f32[1, 576, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_445, 2);  unsqueeze_445 = None
    unsqueeze_447: "f32[1, 576, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 3);  unsqueeze_446 = None
    mul_589: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_444);  sub_124 = unsqueeze_444 = None
    sub_126: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_589);  where_10 = mul_589 = None
    sub_127: "f32[8, 576, 14, 14]" = torch.ops.aten.sub.Tensor(sub_126, unsqueeze_441);  sub_126 = unsqueeze_441 = None
    mul_590: "f32[8, 576, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_447);  sub_127 = unsqueeze_447 = None
    mul_591: "f32[576]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_145);  sum_33 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_590, add_250, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_590 = add_250 = primals_177 = None
    getitem_173: "f32[8, 96, 14, 14]" = convolution_backward_15[0]
    getitem_174: "f32[576, 96, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_34: "f32[96]" = torch.ops.aten.sum.dim_IntList(getitem_173, [0, 2, 3])
    sub_128: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_450);  convolution_47 = unsqueeze_450 = None
    mul_592: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_173, sub_128)
    sum_35: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_592, [0, 2, 3]);  mul_592 = None
    mul_593: "f32[96]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    unsqueeze_451: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_593, 0);  mul_593 = None
    unsqueeze_452: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 2);  unsqueeze_451 = None
    unsqueeze_453: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 3);  unsqueeze_452 = None
    mul_594: "f32[96]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    mul_595: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_596: "f32[96]" = torch.ops.aten.mul.Tensor(mul_594, mul_595);  mul_594 = mul_595 = None
    unsqueeze_454: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_596, 0);  mul_596 = None
    unsqueeze_455: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, 2);  unsqueeze_454 = None
    unsqueeze_456: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 3);  unsqueeze_455 = None
    mul_597: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_457: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
    unsqueeze_458: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_457, 2);  unsqueeze_457 = None
    unsqueeze_459: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 3);  unsqueeze_458 = None
    mul_598: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_456);  sub_128 = unsqueeze_456 = None
    sub_130: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_173, mul_598);  mul_598 = None
    sub_131: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(sub_130, unsqueeze_453);  sub_130 = unsqueeze_453 = None
    mul_599: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_459);  sub_131 = unsqueeze_459 = None
    mul_600: "f32[96]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_142);  sum_35 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_599, relu_31, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_599 = primals_176 = None
    getitem_176: "f32[8, 288, 14, 14]" = convolution_backward_16[0]
    getitem_177: "f32[96, 288, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_77: "f32[8, 288, 14, 14]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_78: "f32[8, 288, 14, 14]" = torch.ops.aten.alias.default(alias_77);  alias_77 = None
    le_11: "b8[8, 288, 14, 14]" = torch.ops.aten.le.Scalar(alias_78, 0);  alias_78 = None
    where_11: "f32[8, 288, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, getitem_176);  le_11 = getitem_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_36: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_132: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_462);  convolution_46 = unsqueeze_462 = None
    mul_601: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_132)
    sum_37: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_601, [0, 2, 3]);  mul_601 = None
    mul_602: "f32[288]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    unsqueeze_463: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_602, 0);  mul_602 = None
    unsqueeze_464: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 2);  unsqueeze_463 = None
    unsqueeze_465: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 3);  unsqueeze_464 = None
    mul_603: "f32[288]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    mul_604: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_605: "f32[288]" = torch.ops.aten.mul.Tensor(mul_603, mul_604);  mul_603 = mul_604 = None
    unsqueeze_466: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_605, 0);  mul_605 = None
    unsqueeze_467: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, 2);  unsqueeze_466 = None
    unsqueeze_468: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 3);  unsqueeze_467 = None
    mul_606: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_469: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_606, 0);  mul_606 = None
    unsqueeze_470: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_469, 2);  unsqueeze_469 = None
    unsqueeze_471: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 3);  unsqueeze_470 = None
    mul_607: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_468);  sub_132 = unsqueeze_468 = None
    sub_134: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_607);  where_11 = mul_607 = None
    sub_135: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(sub_134, unsqueeze_465);  sub_134 = unsqueeze_465 = None
    mul_608: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_471);  sub_135 = unsqueeze_471 = None
    mul_609: "f32[288]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_139);  sum_37 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_608, relu_30, primals_175, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 288, [True, True, False]);  mul_608 = primals_175 = None
    getitem_179: "f32[8, 288, 14, 14]" = convolution_backward_17[0]
    getitem_180: "f32[288, 1, 5, 5]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_80: "f32[8, 288, 14, 14]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_81: "f32[8, 288, 14, 14]" = torch.ops.aten.alias.default(alias_80);  alias_80 = None
    le_12: "b8[8, 288, 14, 14]" = torch.ops.aten.le.Scalar(alias_81, 0);  alias_81 = None
    where_12: "f32[8, 288, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, getitem_179);  le_12 = getitem_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_38: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_136: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_474);  convolution_45 = unsqueeze_474 = None
    mul_610: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_136)
    sum_39: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_610, [0, 2, 3]);  mul_610 = None
    mul_611: "f32[288]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    unsqueeze_475: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_611, 0);  mul_611 = None
    unsqueeze_476: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_475, 2);  unsqueeze_475 = None
    unsqueeze_477: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 3);  unsqueeze_476 = None
    mul_612: "f32[288]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_613: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_614: "f32[288]" = torch.ops.aten.mul.Tensor(mul_612, mul_613);  mul_612 = mul_613 = None
    unsqueeze_478: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_614, 0);  mul_614 = None
    unsqueeze_479: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 2);  unsqueeze_478 = None
    unsqueeze_480: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 3);  unsqueeze_479 = None
    mul_615: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_481: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_615, 0);  mul_615 = None
    unsqueeze_482: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_481, 2);  unsqueeze_481 = None
    unsqueeze_483: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 3);  unsqueeze_482 = None
    mul_616: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_480);  sub_136 = unsqueeze_480 = None
    sub_138: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_616);  where_12 = mul_616 = None
    sub_139: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(sub_138, unsqueeze_477);  sub_138 = unsqueeze_477 = None
    mul_617: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_483);  sub_139 = unsqueeze_483 = None
    mul_618: "f32[288]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_136);  sum_39 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_617, add_234, primals_174, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_617 = add_234 = primals_174 = None
    getitem_182: "f32[8, 96, 14, 14]" = convolution_backward_18[0]
    getitem_183: "f32[288, 96, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_337: "f32[8, 96, 14, 14]" = torch.ops.aten.add.Tensor(getitem_173, getitem_182);  getitem_173 = getitem_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_40: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_337, [0, 2, 3])
    sub_140: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_486);  convolution_44 = unsqueeze_486 = None
    mul_619: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_337, sub_140)
    sum_41: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_619, [0, 2, 3]);  mul_619 = None
    mul_620: "f32[96]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    unsqueeze_487: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_620, 0);  mul_620 = None
    unsqueeze_488: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_487, 2);  unsqueeze_487 = None
    unsqueeze_489: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 3);  unsqueeze_488 = None
    mul_621: "f32[96]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_622: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_623: "f32[96]" = torch.ops.aten.mul.Tensor(mul_621, mul_622);  mul_621 = mul_622 = None
    unsqueeze_490: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
    unsqueeze_491: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, 2);  unsqueeze_490 = None
    unsqueeze_492: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 3);  unsqueeze_491 = None
    mul_624: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_493: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
    unsqueeze_494: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_493, 2);  unsqueeze_493 = None
    unsqueeze_495: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 3);  unsqueeze_494 = None
    mul_625: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_492);  sub_140 = unsqueeze_492 = None
    sub_142: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(add_337, mul_625);  mul_625 = None
    sub_143: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(sub_142, unsqueeze_489);  sub_142 = unsqueeze_489 = None
    mul_626: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_495);  sub_143 = unsqueeze_495 = None
    mul_627: "f32[96]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_133);  sum_41 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_626, relu_29, primals_173, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_626 = primals_173 = None
    getitem_185: "f32[8, 288, 14, 14]" = convolution_backward_19[0]
    getitem_186: "f32[96, 288, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_83: "f32[8, 288, 14, 14]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_84: "f32[8, 288, 14, 14]" = torch.ops.aten.alias.default(alias_83);  alias_83 = None
    le_13: "b8[8, 288, 14, 14]" = torch.ops.aten.le.Scalar(alias_84, 0);  alias_84 = None
    where_13: "f32[8, 288, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, getitem_185);  le_13 = getitem_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_144: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_498);  convolution_43 = unsqueeze_498 = None
    mul_628: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_144)
    sum_43: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_628, [0, 2, 3]);  mul_628 = None
    mul_629: "f32[288]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_499: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_629, 0);  mul_629 = None
    unsqueeze_500: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_499, 2);  unsqueeze_499 = None
    unsqueeze_501: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 3);  unsqueeze_500 = None
    mul_630: "f32[288]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_631: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_632: "f32[288]" = torch.ops.aten.mul.Tensor(mul_630, mul_631);  mul_630 = mul_631 = None
    unsqueeze_502: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_503: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, 2);  unsqueeze_502 = None
    unsqueeze_504: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 3);  unsqueeze_503 = None
    mul_633: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_505: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
    unsqueeze_506: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 2);  unsqueeze_505 = None
    unsqueeze_507: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 3);  unsqueeze_506 = None
    mul_634: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_504);  sub_144 = unsqueeze_504 = None
    sub_146: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_634);  where_13 = mul_634 = None
    sub_147: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_501);  sub_146 = unsqueeze_501 = None
    mul_635: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_507);  sub_147 = unsqueeze_507 = None
    mul_636: "f32[288]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_130);  sum_43 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_635, relu_28, primals_172, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 288, [True, True, False]);  mul_635 = primals_172 = None
    getitem_188: "f32[8, 288, 14, 14]" = convolution_backward_20[0]
    getitem_189: "f32[288, 1, 5, 5]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_86: "f32[8, 288, 14, 14]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_87: "f32[8, 288, 14, 14]" = torch.ops.aten.alias.default(alias_86);  alias_86 = None
    le_14: "b8[8, 288, 14, 14]" = torch.ops.aten.le.Scalar(alias_87, 0);  alias_87 = None
    where_14: "f32[8, 288, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, getitem_188);  le_14 = getitem_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_44: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_148: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_510);  convolution_42 = unsqueeze_510 = None
    mul_637: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_148)
    sum_45: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_637, [0, 2, 3]);  mul_637 = None
    mul_638: "f32[288]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    unsqueeze_511: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_638, 0);  mul_638 = None
    unsqueeze_512: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_511, 2);  unsqueeze_511 = None
    unsqueeze_513: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 3);  unsqueeze_512 = None
    mul_639: "f32[288]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_640: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_641: "f32[288]" = torch.ops.aten.mul.Tensor(mul_639, mul_640);  mul_639 = mul_640 = None
    unsqueeze_514: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_641, 0);  mul_641 = None
    unsqueeze_515: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, 2);  unsqueeze_514 = None
    unsqueeze_516: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 3);  unsqueeze_515 = None
    mul_642: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_517: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
    unsqueeze_518: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 2);  unsqueeze_517 = None
    unsqueeze_519: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 3);  unsqueeze_518 = None
    mul_643: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_516);  sub_148 = unsqueeze_516 = None
    sub_150: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_643);  where_14 = mul_643 = None
    sub_151: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_513);  sub_150 = unsqueeze_513 = None
    mul_644: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_519);  sub_151 = unsqueeze_519 = None
    mul_645: "f32[288]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_127);  sum_45 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_644, add_218, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_644 = add_218 = primals_171 = None
    getitem_191: "f32[8, 96, 14, 14]" = convolution_backward_21[0]
    getitem_192: "f32[288, 96, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_338: "f32[8, 96, 14, 14]" = torch.ops.aten.add.Tensor(add_337, getitem_191);  add_337 = getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_46: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_338, [0, 2, 3])
    sub_152: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_522);  convolution_41 = unsqueeze_522 = None
    mul_646: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_338, sub_152)
    sum_47: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_646, [0, 2, 3]);  mul_646 = None
    mul_647: "f32[96]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_523: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_647, 0);  mul_647 = None
    unsqueeze_524: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_523, 2);  unsqueeze_523 = None
    unsqueeze_525: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 3);  unsqueeze_524 = None
    mul_648: "f32[96]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_649: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_650: "f32[96]" = torch.ops.aten.mul.Tensor(mul_648, mul_649);  mul_648 = mul_649 = None
    unsqueeze_526: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_650, 0);  mul_650 = None
    unsqueeze_527: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 2);  unsqueeze_526 = None
    unsqueeze_528: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 3);  unsqueeze_527 = None
    mul_651: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_529: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
    unsqueeze_530: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_529, 2);  unsqueeze_529 = None
    unsqueeze_531: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 3);  unsqueeze_530 = None
    mul_652: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_528);  sub_152 = unsqueeze_528 = None
    sub_154: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(add_338, mul_652);  mul_652 = None
    sub_155: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_525);  sub_154 = unsqueeze_525 = None
    mul_653: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_531);  sub_155 = unsqueeze_531 = None
    mul_654: "f32[96]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_124);  sum_47 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_653, relu_27, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_653 = primals_170 = None
    getitem_194: "f32[8, 288, 14, 14]" = convolution_backward_22[0]
    getitem_195: "f32[96, 288, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_89: "f32[8, 288, 14, 14]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_90: "f32[8, 288, 14, 14]" = torch.ops.aten.alias.default(alias_89);  alias_89 = None
    le_15: "b8[8, 288, 14, 14]" = torch.ops.aten.le.Scalar(alias_90, 0);  alias_90 = None
    where_15: "f32[8, 288, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, getitem_194);  le_15 = getitem_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_48: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_156: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_534);  convolution_40 = unsqueeze_534 = None
    mul_655: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_156)
    sum_49: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_655, [0, 2, 3]);  mul_655 = None
    mul_656: "f32[288]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_535: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_656, 0);  mul_656 = None
    unsqueeze_536: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_535, 2);  unsqueeze_535 = None
    unsqueeze_537: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 3);  unsqueeze_536 = None
    mul_657: "f32[288]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_658: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_659: "f32[288]" = torch.ops.aten.mul.Tensor(mul_657, mul_658);  mul_657 = mul_658 = None
    unsqueeze_538: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_659, 0);  mul_659 = None
    unsqueeze_539: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 2);  unsqueeze_538 = None
    unsqueeze_540: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 3);  unsqueeze_539 = None
    mul_660: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_541: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_660, 0);  mul_660 = None
    unsqueeze_542: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_541, 2);  unsqueeze_541 = None
    unsqueeze_543: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 3);  unsqueeze_542 = None
    mul_661: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_540);  sub_156 = unsqueeze_540 = None
    sub_158: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_661);  where_15 = mul_661 = None
    sub_159: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(sub_158, unsqueeze_537);  sub_158 = unsqueeze_537 = None
    mul_662: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_159, unsqueeze_543);  sub_159 = unsqueeze_543 = None
    mul_663: "f32[288]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_121);  sum_49 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_662, relu_26, primals_169, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 288, [True, True, False]);  mul_662 = primals_169 = None
    getitem_197: "f32[8, 288, 14, 14]" = convolution_backward_23[0]
    getitem_198: "f32[288, 1, 5, 5]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_92: "f32[8, 288, 14, 14]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_93: "f32[8, 288, 14, 14]" = torch.ops.aten.alias.default(alias_92);  alias_92 = None
    le_16: "b8[8, 288, 14, 14]" = torch.ops.aten.le.Scalar(alias_93, 0);  alias_93 = None
    where_16: "f32[8, 288, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, getitem_197);  le_16 = getitem_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_50: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_160: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_546);  convolution_39 = unsqueeze_546 = None
    mul_664: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_160)
    sum_51: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_664, [0, 2, 3]);  mul_664 = None
    mul_665: "f32[288]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_547: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_665, 0);  mul_665 = None
    unsqueeze_548: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_547, 2);  unsqueeze_547 = None
    unsqueeze_549: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 3);  unsqueeze_548 = None
    mul_666: "f32[288]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_667: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_668: "f32[288]" = torch.ops.aten.mul.Tensor(mul_666, mul_667);  mul_666 = mul_667 = None
    unsqueeze_550: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_668, 0);  mul_668 = None
    unsqueeze_551: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, 2);  unsqueeze_550 = None
    unsqueeze_552: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 3);  unsqueeze_551 = None
    mul_669: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_553: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_669, 0);  mul_669 = None
    unsqueeze_554: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 2);  unsqueeze_553 = None
    unsqueeze_555: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 3);  unsqueeze_554 = None
    mul_670: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_552);  sub_160 = unsqueeze_552 = None
    sub_162: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_670);  where_16 = mul_670 = None
    sub_163: "f32[8, 288, 14, 14]" = torch.ops.aten.sub.Tensor(sub_162, unsqueeze_549);  sub_162 = unsqueeze_549 = None
    mul_671: "f32[8, 288, 14, 14]" = torch.ops.aten.mul.Tensor(sub_163, unsqueeze_555);  sub_163 = unsqueeze_555 = None
    mul_672: "f32[288]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_118);  sum_51 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_671, add_202, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_671 = add_202 = primals_168 = None
    getitem_200: "f32[8, 96, 14, 14]" = convolution_backward_24[0]
    getitem_201: "f32[288, 96, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_339: "f32[8, 96, 14, 14]" = torch.ops.aten.add.Tensor(add_338, getitem_200);  add_338 = getitem_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_52: "f32[96]" = torch.ops.aten.sum.dim_IntList(add_339, [0, 2, 3])
    sub_164: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_558);  convolution_38 = unsqueeze_558 = None
    mul_673: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(add_339, sub_164)
    sum_53: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_673, [0, 2, 3]);  mul_673 = None
    mul_674: "f32[96]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    unsqueeze_559: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_674, 0);  mul_674 = None
    unsqueeze_560: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 2);  unsqueeze_559 = None
    unsqueeze_561: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 3);  unsqueeze_560 = None
    mul_675: "f32[96]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    mul_676: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_677: "f32[96]" = torch.ops.aten.mul.Tensor(mul_675, mul_676);  mul_675 = mul_676 = None
    unsqueeze_562: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_677, 0);  mul_677 = None
    unsqueeze_563: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, 2);  unsqueeze_562 = None
    unsqueeze_564: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 3);  unsqueeze_563 = None
    mul_678: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_565: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_678, 0);  mul_678 = None
    unsqueeze_566: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_565, 2);  unsqueeze_565 = None
    unsqueeze_567: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 3);  unsqueeze_566 = None
    mul_679: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_564);  sub_164 = unsqueeze_564 = None
    sub_166: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(add_339, mul_679);  add_339 = mul_679 = None
    sub_167: "f32[8, 96, 14, 14]" = torch.ops.aten.sub.Tensor(sub_166, unsqueeze_561);  sub_166 = unsqueeze_561 = None
    mul_680: "f32[8, 96, 14, 14]" = torch.ops.aten.mul.Tensor(sub_167, unsqueeze_567);  sub_167 = unsqueeze_567 = None
    mul_681: "f32[96]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_115);  sum_53 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_680, relu_25, primals_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_680 = primals_167 = None
    getitem_203: "f32[8, 480, 14, 14]" = convolution_backward_25[0]
    getitem_204: "f32[96, 480, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_95: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_96: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(alias_95);  alias_95 = None
    le_17: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_96, 0);  alias_96 = None
    where_17: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, getitem_203);  le_17 = getitem_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_168: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_570);  convolution_37 = unsqueeze_570 = None
    mul_682: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_168)
    sum_55: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_682, [0, 2, 3]);  mul_682 = None
    mul_683: "f32[480]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_571: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_683, 0);  mul_683 = None
    unsqueeze_572: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_571, 2);  unsqueeze_571 = None
    unsqueeze_573: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 3);  unsqueeze_572 = None
    mul_684: "f32[480]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_685: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_686: "f32[480]" = torch.ops.aten.mul.Tensor(mul_684, mul_685);  mul_684 = mul_685 = None
    unsqueeze_574: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_686, 0);  mul_686 = None
    unsqueeze_575: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, 2);  unsqueeze_574 = None
    unsqueeze_576: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 3);  unsqueeze_575 = None
    mul_687: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_577: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_687, 0);  mul_687 = None
    unsqueeze_578: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_577, 2);  unsqueeze_577 = None
    unsqueeze_579: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 3);  unsqueeze_578 = None
    mul_688: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_576);  sub_168 = unsqueeze_576 = None
    sub_170: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_688);  where_17 = mul_688 = None
    sub_171: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_170, unsqueeze_573);  sub_170 = unsqueeze_573 = None
    mul_689: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_171, unsqueeze_579);  sub_171 = unsqueeze_579 = None
    mul_690: "f32[480]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_112);  sum_55 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_689, relu_24, primals_166, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_689 = primals_166 = None
    getitem_206: "f32[8, 480, 14, 14]" = convolution_backward_26[0]
    getitem_207: "f32[480, 1, 5, 5]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_98: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_99: "f32[8, 480, 14, 14]" = torch.ops.aten.alias.default(alias_98);  alias_98 = None
    le_18: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(alias_99, 0);  alias_99 = None
    where_18: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, getitem_206);  le_18 = getitem_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_172: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_582);  convolution_36 = unsqueeze_582 = None
    mul_691: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_172)
    sum_57: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_691, [0, 2, 3]);  mul_691 = None
    mul_692: "f32[480]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_583: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_692, 0);  mul_692 = None
    unsqueeze_584: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_583, 2);  unsqueeze_583 = None
    unsqueeze_585: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 3);  unsqueeze_584 = None
    mul_693: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_694: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_695: "f32[480]" = torch.ops.aten.mul.Tensor(mul_693, mul_694);  mul_693 = mul_694 = None
    unsqueeze_586: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_695, 0);  mul_695 = None
    unsqueeze_587: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, 2);  unsqueeze_586 = None
    unsqueeze_588: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 3);  unsqueeze_587 = None
    mul_696: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_589: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_696, 0);  mul_696 = None
    unsqueeze_590: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_589, 2);  unsqueeze_589 = None
    unsqueeze_591: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 3);  unsqueeze_590 = None
    mul_697: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_588);  sub_172 = unsqueeze_588 = None
    sub_174: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_697);  where_18 = mul_697 = None
    sub_175: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_174, unsqueeze_585);  sub_174 = unsqueeze_585 = None
    mul_698: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_175, unsqueeze_591);  sub_175 = unsqueeze_591 = None
    mul_699: "f32[480]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_109);  sum_57 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_698, add_187, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_698 = add_187 = primals_165 = None
    getitem_209: "f32[8, 80, 14, 14]" = convolution_backward_27[0]
    getitem_210: "f32[480, 80, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_209, [0, 2, 3])
    sub_176: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_594);  convolution_35 = unsqueeze_594 = None
    mul_700: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_209, sub_176)
    sum_59: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_700, [0, 2, 3]);  mul_700 = None
    mul_701: "f32[80]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_595: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_701, 0);  mul_701 = None
    unsqueeze_596: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_595, 2);  unsqueeze_595 = None
    unsqueeze_597: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 3);  unsqueeze_596 = None
    mul_702: "f32[80]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_703: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_704: "f32[80]" = torch.ops.aten.mul.Tensor(mul_702, mul_703);  mul_702 = mul_703 = None
    unsqueeze_598: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_704, 0);  mul_704 = None
    unsqueeze_599: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, 2);  unsqueeze_598 = None
    unsqueeze_600: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 3);  unsqueeze_599 = None
    mul_705: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_601: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_705, 0);  mul_705 = None
    unsqueeze_602: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 2);  unsqueeze_601 = None
    unsqueeze_603: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 3);  unsqueeze_602 = None
    mul_706: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_600);  sub_176 = unsqueeze_600 = None
    sub_178: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_209, mul_706);  mul_706 = None
    sub_179: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_178, unsqueeze_597);  sub_178 = unsqueeze_597 = None
    mul_707: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_179, unsqueeze_603);  sub_179 = unsqueeze_603 = None
    mul_708: "f32[80]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_106);  sum_59 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_707, relu_23, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_707 = primals_164 = None
    getitem_212: "f32[8, 240, 14, 14]" = convolution_backward_28[0]
    getitem_213: "f32[80, 240, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_101: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_102: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(alias_101);  alias_101 = None
    le_19: "b8[8, 240, 14, 14]" = torch.ops.aten.le.Scalar(alias_102, 0);  alias_102 = None
    where_19: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, getitem_212);  le_19 = getitem_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_60: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_180: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_606);  convolution_34 = unsqueeze_606 = None
    mul_709: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_180)
    sum_61: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_709, [0, 2, 3]);  mul_709 = None
    mul_710: "f32[240]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_607: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_710, 0);  mul_710 = None
    unsqueeze_608: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_607, 2);  unsqueeze_607 = None
    unsqueeze_609: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 3);  unsqueeze_608 = None
    mul_711: "f32[240]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_712: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_713: "f32[240]" = torch.ops.aten.mul.Tensor(mul_711, mul_712);  mul_711 = mul_712 = None
    unsqueeze_610: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_713, 0);  mul_713 = None
    unsqueeze_611: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, 2);  unsqueeze_610 = None
    unsqueeze_612: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 3);  unsqueeze_611 = None
    mul_714: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_613: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_714, 0);  mul_714 = None
    unsqueeze_614: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_613, 2);  unsqueeze_613 = None
    unsqueeze_615: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 3);  unsqueeze_614 = None
    mul_715: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_612);  sub_180 = unsqueeze_612 = None
    sub_182: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_715);  where_19 = mul_715 = None
    sub_183: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_182, unsqueeze_609);  sub_182 = unsqueeze_609 = None
    mul_716: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_183, unsqueeze_615);  sub_183 = unsqueeze_615 = None
    mul_717: "f32[240]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_103);  sum_61 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_716, relu_22, primals_163, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_716 = primals_163 = None
    getitem_215: "f32[8, 240, 14, 14]" = convolution_backward_29[0]
    getitem_216: "f32[240, 1, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_104: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_105: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(alias_104);  alias_104 = None
    le_20: "b8[8, 240, 14, 14]" = torch.ops.aten.le.Scalar(alias_105, 0);  alias_105 = None
    where_20: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, getitem_215);  le_20 = getitem_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_62: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_184: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_618);  convolution_33 = unsqueeze_618 = None
    mul_718: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_184)
    sum_63: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_718, [0, 2, 3]);  mul_718 = None
    mul_719: "f32[240]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    unsqueeze_619: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_719, 0);  mul_719 = None
    unsqueeze_620: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_619, 2);  unsqueeze_619 = None
    unsqueeze_621: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 3);  unsqueeze_620 = None
    mul_720: "f32[240]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    mul_721: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_722: "f32[240]" = torch.ops.aten.mul.Tensor(mul_720, mul_721);  mul_720 = mul_721 = None
    unsqueeze_622: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_722, 0);  mul_722 = None
    unsqueeze_623: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, 2);  unsqueeze_622 = None
    unsqueeze_624: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 3);  unsqueeze_623 = None
    mul_723: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_625: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_723, 0);  mul_723 = None
    unsqueeze_626: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_625, 2);  unsqueeze_625 = None
    unsqueeze_627: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 3);  unsqueeze_626 = None
    mul_724: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_624);  sub_184 = unsqueeze_624 = None
    sub_186: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_724);  where_20 = mul_724 = None
    sub_187: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_186, unsqueeze_621);  sub_186 = unsqueeze_621 = None
    mul_725: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_627);  sub_187 = unsqueeze_627 = None
    mul_726: "f32[240]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_100);  sum_63 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_725, add_171, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_725 = add_171 = primals_162 = None
    getitem_218: "f32[8, 80, 14, 14]" = convolution_backward_30[0]
    getitem_219: "f32[240, 80, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_340: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_209, getitem_218);  getitem_209 = getitem_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_64: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_340, [0, 2, 3])
    sub_188: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_630);  convolution_32 = unsqueeze_630 = None
    mul_727: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_340, sub_188)
    sum_65: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_727, [0, 2, 3]);  mul_727 = None
    mul_728: "f32[80]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    unsqueeze_631: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_728, 0);  mul_728 = None
    unsqueeze_632: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_631, 2);  unsqueeze_631 = None
    unsqueeze_633: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 3);  unsqueeze_632 = None
    mul_729: "f32[80]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    mul_730: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_731: "f32[80]" = torch.ops.aten.mul.Tensor(mul_729, mul_730);  mul_729 = mul_730 = None
    unsqueeze_634: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_731, 0);  mul_731 = None
    unsqueeze_635: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, 2);  unsqueeze_634 = None
    unsqueeze_636: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 3);  unsqueeze_635 = None
    mul_732: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_637: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_732, 0);  mul_732 = None
    unsqueeze_638: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_637, 2);  unsqueeze_637 = None
    unsqueeze_639: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 3);  unsqueeze_638 = None
    mul_733: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_636);  sub_188 = unsqueeze_636 = None
    sub_190: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_340, mul_733);  mul_733 = None
    sub_191: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_190, unsqueeze_633);  sub_190 = unsqueeze_633 = None
    mul_734: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_639);  sub_191 = unsqueeze_639 = None
    mul_735: "f32[80]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_97);  sum_65 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_734, relu_21, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_734 = primals_161 = None
    getitem_221: "f32[8, 240, 14, 14]" = convolution_backward_31[0]
    getitem_222: "f32[80, 240, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_107: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_108: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(alias_107);  alias_107 = None
    le_21: "b8[8, 240, 14, 14]" = torch.ops.aten.le.Scalar(alias_108, 0);  alias_108 = None
    where_21: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(le_21, full_default, getitem_221);  le_21 = getitem_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_66: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_192: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_642);  convolution_31 = unsqueeze_642 = None
    mul_736: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_192)
    sum_67: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 2, 3]);  mul_736 = None
    mul_737: "f32[240]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    unsqueeze_643: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_737, 0);  mul_737 = None
    unsqueeze_644: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_643, 2);  unsqueeze_643 = None
    unsqueeze_645: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 3);  unsqueeze_644 = None
    mul_738: "f32[240]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    mul_739: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_740: "f32[240]" = torch.ops.aten.mul.Tensor(mul_738, mul_739);  mul_738 = mul_739 = None
    unsqueeze_646: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_740, 0);  mul_740 = None
    unsqueeze_647: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, 2);  unsqueeze_646 = None
    unsqueeze_648: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 3);  unsqueeze_647 = None
    mul_741: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_649: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_741, 0);  mul_741 = None
    unsqueeze_650: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_649, 2);  unsqueeze_649 = None
    unsqueeze_651: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 3);  unsqueeze_650 = None
    mul_742: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_648);  sub_192 = unsqueeze_648 = None
    sub_194: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_742);  where_21 = mul_742 = None
    sub_195: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_194, unsqueeze_645);  sub_194 = unsqueeze_645 = None
    mul_743: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_651);  sub_195 = unsqueeze_651 = None
    mul_744: "f32[240]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_94);  sum_67 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_743, relu_20, primals_160, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_743 = primals_160 = None
    getitem_224: "f32[8, 240, 14, 14]" = convolution_backward_32[0]
    getitem_225: "f32[240, 1, 3, 3]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_110: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_111: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(alias_110);  alias_110 = None
    le_22: "b8[8, 240, 14, 14]" = torch.ops.aten.le.Scalar(alias_111, 0);  alias_111 = None
    where_22: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, getitem_224);  le_22 = getitem_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_68: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_196: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_654);  convolution_30 = unsqueeze_654 = None
    mul_745: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_196)
    sum_69: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_745, [0, 2, 3]);  mul_745 = None
    mul_746: "f32[240]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    unsqueeze_655: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_656: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_655, 2);  unsqueeze_655 = None
    unsqueeze_657: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 3);  unsqueeze_656 = None
    mul_747: "f32[240]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    mul_748: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_749: "f32[240]" = torch.ops.aten.mul.Tensor(mul_747, mul_748);  mul_747 = mul_748 = None
    unsqueeze_658: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_749, 0);  mul_749 = None
    unsqueeze_659: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, 2);  unsqueeze_658 = None
    unsqueeze_660: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 3);  unsqueeze_659 = None
    mul_750: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_661: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_750, 0);  mul_750 = None
    unsqueeze_662: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_661, 2);  unsqueeze_661 = None
    unsqueeze_663: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 3);  unsqueeze_662 = None
    mul_751: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_660);  sub_196 = unsqueeze_660 = None
    sub_198: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_751);  where_22 = mul_751 = None
    sub_199: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_198, unsqueeze_657);  sub_198 = unsqueeze_657 = None
    mul_752: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_663);  sub_199 = unsqueeze_663 = None
    mul_753: "f32[240]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_91);  sum_69 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_752, add_155, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_752 = add_155 = primals_159 = None
    getitem_227: "f32[8, 80, 14, 14]" = convolution_backward_33[0]
    getitem_228: "f32[240, 80, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_341: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_340, getitem_227);  add_340 = getitem_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_70: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_341, [0, 2, 3])
    sub_200: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_666);  convolution_29 = unsqueeze_666 = None
    mul_754: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_341, sub_200)
    sum_71: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_754, [0, 2, 3]);  mul_754 = None
    mul_755: "f32[80]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_667: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_668: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_667, 2);  unsqueeze_667 = None
    unsqueeze_669: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 3);  unsqueeze_668 = None
    mul_756: "f32[80]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_757: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_758: "f32[80]" = torch.ops.aten.mul.Tensor(mul_756, mul_757);  mul_756 = mul_757 = None
    unsqueeze_670: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_758, 0);  mul_758 = None
    unsqueeze_671: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, 2);  unsqueeze_670 = None
    unsqueeze_672: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 3);  unsqueeze_671 = None
    mul_759: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_673: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_759, 0);  mul_759 = None
    unsqueeze_674: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_673, 2);  unsqueeze_673 = None
    unsqueeze_675: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 3);  unsqueeze_674 = None
    mul_760: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_672);  sub_200 = unsqueeze_672 = None
    sub_202: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_341, mul_760);  mul_760 = None
    sub_203: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_202, unsqueeze_669);  sub_202 = unsqueeze_669 = None
    mul_761: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_203, unsqueeze_675);  sub_203 = unsqueeze_675 = None
    mul_762: "f32[80]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_88);  sum_71 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_761, relu_19, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_761 = primals_158 = None
    getitem_230: "f32[8, 240, 14, 14]" = convolution_backward_34[0]
    getitem_231: "f32[80, 240, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_113: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_114: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(alias_113);  alias_113 = None
    le_23: "b8[8, 240, 14, 14]" = torch.ops.aten.le.Scalar(alias_114, 0);  alias_114 = None
    where_23: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, getitem_230);  le_23 = getitem_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_204: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_678);  convolution_28 = unsqueeze_678 = None
    mul_763: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_204)
    sum_73: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_763, [0, 2, 3]);  mul_763 = None
    mul_764: "f32[240]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_679: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_680: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_679, 2);  unsqueeze_679 = None
    unsqueeze_681: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 3);  unsqueeze_680 = None
    mul_765: "f32[240]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_766: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_767: "f32[240]" = torch.ops.aten.mul.Tensor(mul_765, mul_766);  mul_765 = mul_766 = None
    unsqueeze_682: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_767, 0);  mul_767 = None
    unsqueeze_683: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, 2);  unsqueeze_682 = None
    unsqueeze_684: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 3);  unsqueeze_683 = None
    mul_768: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_685: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_768, 0);  mul_768 = None
    unsqueeze_686: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_685, 2);  unsqueeze_685 = None
    unsqueeze_687: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 3);  unsqueeze_686 = None
    mul_769: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_684);  sub_204 = unsqueeze_684 = None
    sub_206: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_769);  where_23 = mul_769 = None
    sub_207: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_681);  sub_206 = unsqueeze_681 = None
    mul_770: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_687);  sub_207 = unsqueeze_687 = None
    mul_771: "f32[240]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_85);  sum_73 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_770, relu_18, primals_157, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_770 = primals_157 = None
    getitem_233: "f32[8, 240, 14, 14]" = convolution_backward_35[0]
    getitem_234: "f32[240, 1, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_116: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_117: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(alias_116);  alias_116 = None
    le_24: "b8[8, 240, 14, 14]" = torch.ops.aten.le.Scalar(alias_117, 0);  alias_117 = None
    where_24: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(le_24, full_default, getitem_233);  le_24 = getitem_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_208: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_690);  convolution_27 = unsqueeze_690 = None
    mul_772: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_208)
    sum_75: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_772, [0, 2, 3]);  mul_772 = None
    mul_773: "f32[240]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_691: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_773, 0);  mul_773 = None
    unsqueeze_692: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_691, 2);  unsqueeze_691 = None
    unsqueeze_693: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 3);  unsqueeze_692 = None
    mul_774: "f32[240]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_775: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_776: "f32[240]" = torch.ops.aten.mul.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
    unsqueeze_694: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_776, 0);  mul_776 = None
    unsqueeze_695: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, 2);  unsqueeze_694 = None
    unsqueeze_696: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 3);  unsqueeze_695 = None
    mul_777: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_697: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_777, 0);  mul_777 = None
    unsqueeze_698: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_697, 2);  unsqueeze_697 = None
    unsqueeze_699: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 3);  unsqueeze_698 = None
    mul_778: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_696);  sub_208 = unsqueeze_696 = None
    sub_210: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_778);  where_24 = mul_778 = None
    sub_211: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_693);  sub_210 = unsqueeze_693 = None
    mul_779: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_699);  sub_211 = unsqueeze_699 = None
    mul_780: "f32[240]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_82);  sum_75 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_779, add_139, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_779 = add_139 = primals_156 = None
    getitem_236: "f32[8, 80, 14, 14]" = convolution_backward_36[0]
    getitem_237: "f32[240, 80, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_342: "f32[8, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_341, getitem_236);  add_341 = getitem_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_76: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_342, [0, 2, 3])
    sub_212: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_702);  convolution_26 = unsqueeze_702 = None
    mul_781: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_342, sub_212)
    sum_77: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_781, [0, 2, 3]);  mul_781 = None
    mul_782: "f32[80]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    unsqueeze_703: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_782, 0);  mul_782 = None
    unsqueeze_704: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_703, 2);  unsqueeze_703 = None
    unsqueeze_705: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 3);  unsqueeze_704 = None
    mul_783: "f32[80]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_784: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_785: "f32[80]" = torch.ops.aten.mul.Tensor(mul_783, mul_784);  mul_783 = mul_784 = None
    unsqueeze_706: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_785, 0);  mul_785 = None
    unsqueeze_707: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, 2);  unsqueeze_706 = None
    unsqueeze_708: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 3);  unsqueeze_707 = None
    mul_786: "f32[80]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_709: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_786, 0);  mul_786 = None
    unsqueeze_710: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_709, 2);  unsqueeze_709 = None
    unsqueeze_711: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 3);  unsqueeze_710 = None
    mul_787: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_708);  sub_212 = unsqueeze_708 = None
    sub_214: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(add_342, mul_787);  add_342 = mul_787 = None
    sub_215: "f32[8, 80, 14, 14]" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_705);  sub_214 = unsqueeze_705 = None
    mul_788: "f32[8, 80, 14, 14]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_711);  sub_215 = unsqueeze_711 = None
    mul_789: "f32[80]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_79);  sum_77 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_788, relu_17, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_788 = primals_155 = None
    getitem_239: "f32[8, 240, 14, 14]" = convolution_backward_37[0]
    getitem_240: "f32[80, 240, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_119: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_120: "f32[8, 240, 14, 14]" = torch.ops.aten.alias.default(alias_119);  alias_119 = None
    le_25: "b8[8, 240, 14, 14]" = torch.ops.aten.le.Scalar(alias_120, 0);  alias_120 = None
    where_25: "f32[8, 240, 14, 14]" = torch.ops.aten.where.self(le_25, full_default, getitem_239);  le_25 = getitem_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_78: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_216: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_714);  convolution_25 = unsqueeze_714 = None
    mul_790: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_216)
    sum_79: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_790, [0, 2, 3]);  mul_790 = None
    mul_791: "f32[240]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    unsqueeze_715: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_791, 0);  mul_791 = None
    unsqueeze_716: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_715, 2);  unsqueeze_715 = None
    unsqueeze_717: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 3);  unsqueeze_716 = None
    mul_792: "f32[240]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    mul_793: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_794: "f32[240]" = torch.ops.aten.mul.Tensor(mul_792, mul_793);  mul_792 = mul_793 = None
    unsqueeze_718: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_794, 0);  mul_794 = None
    unsqueeze_719: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, 2);  unsqueeze_718 = None
    unsqueeze_720: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 3);  unsqueeze_719 = None
    mul_795: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_721: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_795, 0);  mul_795 = None
    unsqueeze_722: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_721, 2);  unsqueeze_721 = None
    unsqueeze_723: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 3);  unsqueeze_722 = None
    mul_796: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_720);  sub_216 = unsqueeze_720 = None
    sub_218: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_796);  where_25 = mul_796 = None
    sub_219: "f32[8, 240, 14, 14]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_717);  sub_218 = unsqueeze_717 = None
    mul_797: "f32[8, 240, 14, 14]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_723);  sub_219 = unsqueeze_723 = None
    mul_798: "f32[240]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_76);  sum_79 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_797, relu_16, primals_154, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_797 = primals_154 = None
    getitem_242: "f32[8, 240, 28, 28]" = convolution_backward_38[0]
    getitem_243: "f32[240, 1, 5, 5]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_122: "f32[8, 240, 28, 28]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_123: "f32[8, 240, 28, 28]" = torch.ops.aten.alias.default(alias_122);  alias_122 = None
    le_26: "b8[8, 240, 28, 28]" = torch.ops.aten.le.Scalar(alias_123, 0);  alias_123 = None
    where_26: "f32[8, 240, 28, 28]" = torch.ops.aten.where.self(le_26, full_default, getitem_242);  le_26 = getitem_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_80: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_220: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_726);  convolution_24 = unsqueeze_726 = None
    mul_799: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, sub_220)
    sum_81: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_799, [0, 2, 3]);  mul_799 = None
    mul_800: "f32[240]" = torch.ops.aten.mul.Tensor(sum_80, 0.00015943877551020407)
    unsqueeze_727: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_800, 0);  mul_800 = None
    unsqueeze_728: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_727, 2);  unsqueeze_727 = None
    unsqueeze_729: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 3);  unsqueeze_728 = None
    mul_801: "f32[240]" = torch.ops.aten.mul.Tensor(sum_81, 0.00015943877551020407)
    mul_802: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_803: "f32[240]" = torch.ops.aten.mul.Tensor(mul_801, mul_802);  mul_801 = mul_802 = None
    unsqueeze_730: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_803, 0);  mul_803 = None
    unsqueeze_731: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, 2);  unsqueeze_730 = None
    unsqueeze_732: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 3);  unsqueeze_731 = None
    mul_804: "f32[240]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_733: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_804, 0);  mul_804 = None
    unsqueeze_734: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_733, 2);  unsqueeze_733 = None
    unsqueeze_735: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 3);  unsqueeze_734 = None
    mul_805: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_732);  sub_220 = unsqueeze_732 = None
    sub_222: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(where_26, mul_805);  where_26 = mul_805 = None
    sub_223: "f32[8, 240, 28, 28]" = torch.ops.aten.sub.Tensor(sub_222, unsqueeze_729);  sub_222 = unsqueeze_729 = None
    mul_806: "f32[8, 240, 28, 28]" = torch.ops.aten.mul.Tensor(sub_223, unsqueeze_735);  sub_223 = unsqueeze_735 = None
    mul_807: "f32[240]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_73);  sum_81 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_806, add_124, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_806 = add_124 = primals_153 = None
    getitem_245: "f32[8, 40, 28, 28]" = convolution_backward_39[0]
    getitem_246: "f32[240, 40, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_82: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_245, [0, 2, 3])
    sub_224: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_738);  convolution_23 = unsqueeze_738 = None
    mul_808: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_245, sub_224)
    sum_83: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_808, [0, 2, 3]);  mul_808 = None
    mul_809: "f32[40]" = torch.ops.aten.mul.Tensor(sum_82, 0.00015943877551020407)
    unsqueeze_739: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_809, 0);  mul_809 = None
    unsqueeze_740: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_739, 2);  unsqueeze_739 = None
    unsqueeze_741: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 3);  unsqueeze_740 = None
    mul_810: "f32[40]" = torch.ops.aten.mul.Tensor(sum_83, 0.00015943877551020407)
    mul_811: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_812: "f32[40]" = torch.ops.aten.mul.Tensor(mul_810, mul_811);  mul_810 = mul_811 = None
    unsqueeze_742: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_812, 0);  mul_812 = None
    unsqueeze_743: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, 2);  unsqueeze_742 = None
    unsqueeze_744: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 3);  unsqueeze_743 = None
    mul_813: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_745: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_813, 0);  mul_813 = None
    unsqueeze_746: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_745, 2);  unsqueeze_745 = None
    unsqueeze_747: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 3);  unsqueeze_746 = None
    mul_814: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_744);  sub_224 = unsqueeze_744 = None
    sub_226: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_245, mul_814);  mul_814 = None
    sub_227: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_226, unsqueeze_741);  sub_226 = unsqueeze_741 = None
    mul_815: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_227, unsqueeze_747);  sub_227 = unsqueeze_747 = None
    mul_816: "f32[40]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_70);  sum_83 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_815, relu_15, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_815 = primals_152 = None
    getitem_248: "f32[8, 120, 28, 28]" = convolution_backward_40[0]
    getitem_249: "f32[40, 120, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_125: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_126: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_125);  alias_125 = None
    le_27: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_126, 0);  alias_126 = None
    where_27: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_27, full_default, getitem_248);  le_27 = getitem_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_84: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_228: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_750);  convolution_22 = unsqueeze_750 = None
    mul_817: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, sub_228)
    sum_85: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_817, [0, 2, 3]);  mul_817 = None
    mul_818: "f32[120]" = torch.ops.aten.mul.Tensor(sum_84, 0.00015943877551020407)
    unsqueeze_751: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_818, 0);  mul_818 = None
    unsqueeze_752: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_751, 2);  unsqueeze_751 = None
    unsqueeze_753: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 3);  unsqueeze_752 = None
    mul_819: "f32[120]" = torch.ops.aten.mul.Tensor(sum_85, 0.00015943877551020407)
    mul_820: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_821: "f32[120]" = torch.ops.aten.mul.Tensor(mul_819, mul_820);  mul_819 = mul_820 = None
    unsqueeze_754: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_821, 0);  mul_821 = None
    unsqueeze_755: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, 2);  unsqueeze_754 = None
    unsqueeze_756: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 3);  unsqueeze_755 = None
    mul_822: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_757: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_822, 0);  mul_822 = None
    unsqueeze_758: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_757, 2);  unsqueeze_757 = None
    unsqueeze_759: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 3);  unsqueeze_758 = None
    mul_823: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_756);  sub_228 = unsqueeze_756 = None
    sub_230: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_27, mul_823);  where_27 = mul_823 = None
    sub_231: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_230, unsqueeze_753);  sub_230 = unsqueeze_753 = None
    mul_824: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_231, unsqueeze_759);  sub_231 = unsqueeze_759 = None
    mul_825: "f32[120]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_67);  sum_85 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_824, relu_14, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_824 = primals_151 = None
    getitem_251: "f32[8, 120, 28, 28]" = convolution_backward_41[0]
    getitem_252: "f32[120, 1, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_128: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_129: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_128);  alias_128 = None
    le_28: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_129, 0);  alias_129 = None
    where_28: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_28, full_default, getitem_251);  le_28 = getitem_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_86: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_232: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_762);  convolution_21 = unsqueeze_762 = None
    mul_826: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, sub_232)
    sum_87: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_826, [0, 2, 3]);  mul_826 = None
    mul_827: "f32[120]" = torch.ops.aten.mul.Tensor(sum_86, 0.00015943877551020407)
    unsqueeze_763: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_827, 0);  mul_827 = None
    unsqueeze_764: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_763, 2);  unsqueeze_763 = None
    unsqueeze_765: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 3);  unsqueeze_764 = None
    mul_828: "f32[120]" = torch.ops.aten.mul.Tensor(sum_87, 0.00015943877551020407)
    mul_829: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_830: "f32[120]" = torch.ops.aten.mul.Tensor(mul_828, mul_829);  mul_828 = mul_829 = None
    unsqueeze_766: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_830, 0);  mul_830 = None
    unsqueeze_767: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, 2);  unsqueeze_766 = None
    unsqueeze_768: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 3);  unsqueeze_767 = None
    mul_831: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_769: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_831, 0);  mul_831 = None
    unsqueeze_770: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_769, 2);  unsqueeze_769 = None
    unsqueeze_771: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 3);  unsqueeze_770 = None
    mul_832: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_768);  sub_232 = unsqueeze_768 = None
    sub_234: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_28, mul_832);  where_28 = mul_832 = None
    sub_235: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_234, unsqueeze_765);  sub_234 = unsqueeze_765 = None
    mul_833: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_235, unsqueeze_771);  sub_235 = unsqueeze_771 = None
    mul_834: "f32[120]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_64);  sum_87 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_833, add_108, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_833 = add_108 = primals_150 = None
    getitem_254: "f32[8, 40, 28, 28]" = convolution_backward_42[0]
    getitem_255: "f32[120, 40, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_343: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_245, getitem_254);  getitem_245 = getitem_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_88: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_343, [0, 2, 3])
    sub_236: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_774);  convolution_20 = unsqueeze_774 = None
    mul_835: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_343, sub_236)
    sum_89: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_835, [0, 2, 3]);  mul_835 = None
    mul_836: "f32[40]" = torch.ops.aten.mul.Tensor(sum_88, 0.00015943877551020407)
    unsqueeze_775: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_836, 0);  mul_836 = None
    unsqueeze_776: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_775, 2);  unsqueeze_775 = None
    unsqueeze_777: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 3);  unsqueeze_776 = None
    mul_837: "f32[40]" = torch.ops.aten.mul.Tensor(sum_89, 0.00015943877551020407)
    mul_838: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_839: "f32[40]" = torch.ops.aten.mul.Tensor(mul_837, mul_838);  mul_837 = mul_838 = None
    unsqueeze_778: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_839, 0);  mul_839 = None
    unsqueeze_779: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, 2);  unsqueeze_778 = None
    unsqueeze_780: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 3);  unsqueeze_779 = None
    mul_840: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_781: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_840, 0);  mul_840 = None
    unsqueeze_782: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_781, 2);  unsqueeze_781 = None
    unsqueeze_783: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 3);  unsqueeze_782 = None
    mul_841: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_780);  sub_236 = unsqueeze_780 = None
    sub_238: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_343, mul_841);  mul_841 = None
    sub_239: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_238, unsqueeze_777);  sub_238 = unsqueeze_777 = None
    mul_842: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_239, unsqueeze_783);  sub_239 = unsqueeze_783 = None
    mul_843: "f32[40]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_61);  sum_89 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_842, relu_13, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_842 = primals_149 = None
    getitem_257: "f32[8, 120, 28, 28]" = convolution_backward_43[0]
    getitem_258: "f32[40, 120, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_131: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_132: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_131);  alias_131 = None
    le_29: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_132, 0);  alias_132 = None
    where_29: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_29, full_default, getitem_257);  le_29 = getitem_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_90: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_240: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_786);  convolution_19 = unsqueeze_786 = None
    mul_844: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, sub_240)
    sum_91: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_844, [0, 2, 3]);  mul_844 = None
    mul_845: "f32[120]" = torch.ops.aten.mul.Tensor(sum_90, 0.00015943877551020407)
    unsqueeze_787: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_845, 0);  mul_845 = None
    unsqueeze_788: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_787, 2);  unsqueeze_787 = None
    unsqueeze_789: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 3);  unsqueeze_788 = None
    mul_846: "f32[120]" = torch.ops.aten.mul.Tensor(sum_91, 0.00015943877551020407)
    mul_847: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_848: "f32[120]" = torch.ops.aten.mul.Tensor(mul_846, mul_847);  mul_846 = mul_847 = None
    unsqueeze_790: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_848, 0);  mul_848 = None
    unsqueeze_791: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, 2);  unsqueeze_790 = None
    unsqueeze_792: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 3);  unsqueeze_791 = None
    mul_849: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_793: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_849, 0);  mul_849 = None
    unsqueeze_794: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_793, 2);  unsqueeze_793 = None
    unsqueeze_795: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 3);  unsqueeze_794 = None
    mul_850: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_792);  sub_240 = unsqueeze_792 = None
    sub_242: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_29, mul_850);  where_29 = mul_850 = None
    sub_243: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_242, unsqueeze_789);  sub_242 = unsqueeze_789 = None
    mul_851: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_243, unsqueeze_795);  sub_243 = unsqueeze_795 = None
    mul_852: "f32[120]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_58);  sum_91 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_851, relu_12, primals_148, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_851 = primals_148 = None
    getitem_260: "f32[8, 120, 28, 28]" = convolution_backward_44[0]
    getitem_261: "f32[120, 1, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_134: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_135: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_134);  alias_134 = None
    le_30: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_135, 0);  alias_135 = None
    where_30: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_30, full_default, getitem_260);  le_30 = getitem_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_92: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_244: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_798);  convolution_18 = unsqueeze_798 = None
    mul_853: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_30, sub_244)
    sum_93: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_853, [0, 2, 3]);  mul_853 = None
    mul_854: "f32[120]" = torch.ops.aten.mul.Tensor(sum_92, 0.00015943877551020407)
    unsqueeze_799: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_854, 0);  mul_854 = None
    unsqueeze_800: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_799, 2);  unsqueeze_799 = None
    unsqueeze_801: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 3);  unsqueeze_800 = None
    mul_855: "f32[120]" = torch.ops.aten.mul.Tensor(sum_93, 0.00015943877551020407)
    mul_856: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_857: "f32[120]" = torch.ops.aten.mul.Tensor(mul_855, mul_856);  mul_855 = mul_856 = None
    unsqueeze_802: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_857, 0);  mul_857 = None
    unsqueeze_803: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, 2);  unsqueeze_802 = None
    unsqueeze_804: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 3);  unsqueeze_803 = None
    mul_858: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_805: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_858, 0);  mul_858 = None
    unsqueeze_806: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_805, 2);  unsqueeze_805 = None
    unsqueeze_807: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 3);  unsqueeze_806 = None
    mul_859: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_804);  sub_244 = unsqueeze_804 = None
    sub_246: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_30, mul_859);  where_30 = mul_859 = None
    sub_247: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_246, unsqueeze_801);  sub_246 = unsqueeze_801 = None
    mul_860: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_247, unsqueeze_807);  sub_247 = unsqueeze_807 = None
    mul_861: "f32[120]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_55);  sum_93 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_860, add_92, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_860 = add_92 = primals_147 = None
    getitem_263: "f32[8, 40, 28, 28]" = convolution_backward_45[0]
    getitem_264: "f32[120, 40, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_344: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_343, getitem_263);  add_343 = getitem_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_94: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_344, [0, 2, 3])
    sub_248: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_810);  convolution_17 = unsqueeze_810 = None
    mul_862: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_344, sub_248)
    sum_95: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_862, [0, 2, 3]);  mul_862 = None
    mul_863: "f32[40]" = torch.ops.aten.mul.Tensor(sum_94, 0.00015943877551020407)
    unsqueeze_811: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_863, 0);  mul_863 = None
    unsqueeze_812: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_811, 2);  unsqueeze_811 = None
    unsqueeze_813: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 3);  unsqueeze_812 = None
    mul_864: "f32[40]" = torch.ops.aten.mul.Tensor(sum_95, 0.00015943877551020407)
    mul_865: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_866: "f32[40]" = torch.ops.aten.mul.Tensor(mul_864, mul_865);  mul_864 = mul_865 = None
    unsqueeze_814: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_866, 0);  mul_866 = None
    unsqueeze_815: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, 2);  unsqueeze_814 = None
    unsqueeze_816: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 3);  unsqueeze_815 = None
    mul_867: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_817: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_867, 0);  mul_867 = None
    unsqueeze_818: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_817, 2);  unsqueeze_817 = None
    unsqueeze_819: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 3);  unsqueeze_818 = None
    mul_868: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_816);  sub_248 = unsqueeze_816 = None
    sub_250: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_344, mul_868);  mul_868 = None
    sub_251: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_250, unsqueeze_813);  sub_250 = unsqueeze_813 = None
    mul_869: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_251, unsqueeze_819);  sub_251 = unsqueeze_819 = None
    mul_870: "f32[40]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_52);  sum_95 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_869, relu_11, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_869 = primals_146 = None
    getitem_266: "f32[8, 120, 28, 28]" = convolution_backward_46[0]
    getitem_267: "f32[40, 120, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_137: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_138: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_137);  alias_137 = None
    le_31: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_138, 0);  alias_138 = None
    where_31: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_31, full_default, getitem_266);  le_31 = getitem_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_96: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_252: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_822);  convolution_16 = unsqueeze_822 = None
    mul_871: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_31, sub_252)
    sum_97: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_871, [0, 2, 3]);  mul_871 = None
    mul_872: "f32[120]" = torch.ops.aten.mul.Tensor(sum_96, 0.00015943877551020407)
    unsqueeze_823: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_872, 0);  mul_872 = None
    unsqueeze_824: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_823, 2);  unsqueeze_823 = None
    unsqueeze_825: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 3);  unsqueeze_824 = None
    mul_873: "f32[120]" = torch.ops.aten.mul.Tensor(sum_97, 0.00015943877551020407)
    mul_874: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_875: "f32[120]" = torch.ops.aten.mul.Tensor(mul_873, mul_874);  mul_873 = mul_874 = None
    unsqueeze_826: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_875, 0);  mul_875 = None
    unsqueeze_827: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, 2);  unsqueeze_826 = None
    unsqueeze_828: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 3);  unsqueeze_827 = None
    mul_876: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_829: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_876, 0);  mul_876 = None
    unsqueeze_830: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_829, 2);  unsqueeze_829 = None
    unsqueeze_831: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 3);  unsqueeze_830 = None
    mul_877: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_828);  sub_252 = unsqueeze_828 = None
    sub_254: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_31, mul_877);  where_31 = mul_877 = None
    sub_255: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_254, unsqueeze_825);  sub_254 = unsqueeze_825 = None
    mul_878: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_255, unsqueeze_831);  sub_255 = unsqueeze_831 = None
    mul_879: "f32[120]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_49);  sum_97 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_878, relu_10, primals_145, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_878 = primals_145 = None
    getitem_269: "f32[8, 120, 28, 28]" = convolution_backward_47[0]
    getitem_270: "f32[120, 1, 3, 3]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_140: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_141: "f32[8, 120, 28, 28]" = torch.ops.aten.alias.default(alias_140);  alias_140 = None
    le_32: "b8[8, 120, 28, 28]" = torch.ops.aten.le.Scalar(alias_141, 0);  alias_141 = None
    where_32: "f32[8, 120, 28, 28]" = torch.ops.aten.where.self(le_32, full_default, getitem_269);  le_32 = getitem_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_98: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_256: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_834);  convolution_15 = unsqueeze_834 = None
    mul_880: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_32, sub_256)
    sum_99: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_880, [0, 2, 3]);  mul_880 = None
    mul_881: "f32[120]" = torch.ops.aten.mul.Tensor(sum_98, 0.00015943877551020407)
    unsqueeze_835: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_881, 0);  mul_881 = None
    unsqueeze_836: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_835, 2);  unsqueeze_835 = None
    unsqueeze_837: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 3);  unsqueeze_836 = None
    mul_882: "f32[120]" = torch.ops.aten.mul.Tensor(sum_99, 0.00015943877551020407)
    mul_883: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_884: "f32[120]" = torch.ops.aten.mul.Tensor(mul_882, mul_883);  mul_882 = mul_883 = None
    unsqueeze_838: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_884, 0);  mul_884 = None
    unsqueeze_839: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, 2);  unsqueeze_838 = None
    unsqueeze_840: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 3);  unsqueeze_839 = None
    mul_885: "f32[120]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_841: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_885, 0);  mul_885 = None
    unsqueeze_842: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_841, 2);  unsqueeze_841 = None
    unsqueeze_843: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 3);  unsqueeze_842 = None
    mul_886: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_840);  sub_256 = unsqueeze_840 = None
    sub_258: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(where_32, mul_886);  where_32 = mul_886 = None
    sub_259: "f32[8, 120, 28, 28]" = torch.ops.aten.sub.Tensor(sub_258, unsqueeze_837);  sub_258 = unsqueeze_837 = None
    mul_887: "f32[8, 120, 28, 28]" = torch.ops.aten.mul.Tensor(sub_259, unsqueeze_843);  sub_259 = unsqueeze_843 = None
    mul_888: "f32[120]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_46);  sum_99 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_887, add_76, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_887 = add_76 = primals_144 = None
    getitem_272: "f32[8, 40, 28, 28]" = convolution_backward_48[0]
    getitem_273: "f32[120, 40, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_345: "f32[8, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_344, getitem_272);  add_344 = getitem_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_100: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_345, [0, 2, 3])
    sub_260: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_846);  convolution_14 = unsqueeze_846 = None
    mul_889: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_345, sub_260)
    sum_101: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_889, [0, 2, 3]);  mul_889 = None
    mul_890: "f32[40]" = torch.ops.aten.mul.Tensor(sum_100, 0.00015943877551020407)
    unsqueeze_847: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_890, 0);  mul_890 = None
    unsqueeze_848: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_847, 2);  unsqueeze_847 = None
    unsqueeze_849: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 3);  unsqueeze_848 = None
    mul_891: "f32[40]" = torch.ops.aten.mul.Tensor(sum_101, 0.00015943877551020407)
    mul_892: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_893: "f32[40]" = torch.ops.aten.mul.Tensor(mul_891, mul_892);  mul_891 = mul_892 = None
    unsqueeze_850: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_893, 0);  mul_893 = None
    unsqueeze_851: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, 2);  unsqueeze_850 = None
    unsqueeze_852: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 3);  unsqueeze_851 = None
    mul_894: "f32[40]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_853: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_894, 0);  mul_894 = None
    unsqueeze_854: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_853, 2);  unsqueeze_853 = None
    unsqueeze_855: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 3);  unsqueeze_854 = None
    mul_895: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_852);  sub_260 = unsqueeze_852 = None
    sub_262: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(add_345, mul_895);  add_345 = mul_895 = None
    sub_263: "f32[8, 40, 28, 28]" = torch.ops.aten.sub.Tensor(sub_262, unsqueeze_849);  sub_262 = unsqueeze_849 = None
    mul_896: "f32[8, 40, 28, 28]" = torch.ops.aten.mul.Tensor(sub_263, unsqueeze_855);  sub_263 = unsqueeze_855 = None
    mul_897: "f32[40]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_43);  sum_101 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_896, relu_9, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_896 = primals_143 = None
    getitem_275: "f32[8, 144, 28, 28]" = convolution_backward_49[0]
    getitem_276: "f32[40, 144, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_143: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_144: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(alias_143);  alias_143 = None
    le_33: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(alias_144, 0);  alias_144 = None
    where_33: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_33, full_default, getitem_275);  le_33 = getitem_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_102: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_264: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_858);  convolution_13 = unsqueeze_858 = None
    mul_898: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_33, sub_264)
    sum_103: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_898, [0, 2, 3]);  mul_898 = None
    mul_899: "f32[144]" = torch.ops.aten.mul.Tensor(sum_102, 0.00015943877551020407)
    unsqueeze_859: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_899, 0);  mul_899 = None
    unsqueeze_860: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_859, 2);  unsqueeze_859 = None
    unsqueeze_861: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 3);  unsqueeze_860 = None
    mul_900: "f32[144]" = torch.ops.aten.mul.Tensor(sum_103, 0.00015943877551020407)
    mul_901: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_902: "f32[144]" = torch.ops.aten.mul.Tensor(mul_900, mul_901);  mul_900 = mul_901 = None
    unsqueeze_862: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_902, 0);  mul_902 = None
    unsqueeze_863: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, 2);  unsqueeze_862 = None
    unsqueeze_864: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 3);  unsqueeze_863 = None
    mul_903: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_865: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_903, 0);  mul_903 = None
    unsqueeze_866: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_865, 2);  unsqueeze_865 = None
    unsqueeze_867: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 3);  unsqueeze_866 = None
    mul_904: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_864);  sub_264 = unsqueeze_864 = None
    sub_266: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_33, mul_904);  where_33 = mul_904 = None
    sub_267: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_266, unsqueeze_861);  sub_266 = unsqueeze_861 = None
    mul_905: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_267, unsqueeze_867);  sub_267 = unsqueeze_867 = None
    mul_906: "f32[144]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_40);  sum_103 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_905, relu_8, primals_142, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_905 = primals_142 = None
    getitem_278: "f32[8, 144, 56, 56]" = convolution_backward_50[0]
    getitem_279: "f32[144, 1, 5, 5]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_146: "f32[8, 144, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_147: "f32[8, 144, 56, 56]" = torch.ops.aten.alias.default(alias_146);  alias_146 = None
    le_34: "b8[8, 144, 56, 56]" = torch.ops.aten.le.Scalar(alias_147, 0);  alias_147 = None
    where_34: "f32[8, 144, 56, 56]" = torch.ops.aten.where.self(le_34, full_default, getitem_278);  le_34 = getitem_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_104: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_268: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_870);  convolution_12 = unsqueeze_870 = None
    mul_907: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(where_34, sub_268)
    sum_105: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_907, [0, 2, 3]);  mul_907 = None
    mul_908: "f32[144]" = torch.ops.aten.mul.Tensor(sum_104, 3.985969387755102e-05)
    unsqueeze_871: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_908, 0);  mul_908 = None
    unsqueeze_872: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_871, 2);  unsqueeze_871 = None
    unsqueeze_873: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 3);  unsqueeze_872 = None
    mul_909: "f32[144]" = torch.ops.aten.mul.Tensor(sum_105, 3.985969387755102e-05)
    mul_910: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_911: "f32[144]" = torch.ops.aten.mul.Tensor(mul_909, mul_910);  mul_909 = mul_910 = None
    unsqueeze_874: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_911, 0);  mul_911 = None
    unsqueeze_875: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, 2);  unsqueeze_874 = None
    unsqueeze_876: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 3);  unsqueeze_875 = None
    mul_912: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_877: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_912, 0);  mul_912 = None
    unsqueeze_878: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_877, 2);  unsqueeze_877 = None
    unsqueeze_879: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 3);  unsqueeze_878 = None
    mul_913: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_876);  sub_268 = unsqueeze_876 = None
    sub_270: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(where_34, mul_913);  where_34 = mul_913 = None
    sub_271: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(sub_270, unsqueeze_873);  sub_270 = unsqueeze_873 = None
    mul_914: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_271, unsqueeze_879);  sub_271 = unsqueeze_879 = None
    mul_915: "f32[144]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_37);  sum_105 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_914, add_61, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_914 = add_61 = primals_141 = None
    getitem_281: "f32[8, 24, 56, 56]" = convolution_backward_51[0]
    getitem_282: "f32[144, 24, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_106: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_281, [0, 2, 3])
    sub_272: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_882);  convolution_11 = unsqueeze_882 = None
    mul_916: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_281, sub_272)
    sum_107: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_916, [0, 2, 3]);  mul_916 = None
    mul_917: "f32[24]" = torch.ops.aten.mul.Tensor(sum_106, 3.985969387755102e-05)
    unsqueeze_883: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_917, 0);  mul_917 = None
    unsqueeze_884: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_883, 2);  unsqueeze_883 = None
    unsqueeze_885: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 3);  unsqueeze_884 = None
    mul_918: "f32[24]" = torch.ops.aten.mul.Tensor(sum_107, 3.985969387755102e-05)
    mul_919: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_920: "f32[24]" = torch.ops.aten.mul.Tensor(mul_918, mul_919);  mul_918 = mul_919 = None
    unsqueeze_886: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_920, 0);  mul_920 = None
    unsqueeze_887: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, 2);  unsqueeze_886 = None
    unsqueeze_888: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 3);  unsqueeze_887 = None
    mul_921: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_889: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_921, 0);  mul_921 = None
    unsqueeze_890: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_889, 2);  unsqueeze_889 = None
    unsqueeze_891: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 3);  unsqueeze_890 = None
    mul_922: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_888);  sub_272 = unsqueeze_888 = None
    sub_274: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_281, mul_922);  mul_922 = None
    sub_275: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_274, unsqueeze_885);  sub_274 = unsqueeze_885 = None
    mul_923: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_275, unsqueeze_891);  sub_275 = unsqueeze_891 = None
    mul_924: "f32[24]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_34);  sum_107 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_923, relu_7, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_923 = primals_140 = None
    getitem_284: "f32[8, 72, 56, 56]" = convolution_backward_52[0]
    getitem_285: "f32[24, 72, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_149: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_150: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(alias_149);  alias_149 = None
    le_35: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_150, 0);  alias_150 = None
    where_35: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_35, full_default, getitem_284);  le_35 = getitem_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_108: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_276: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_894);  convolution_10 = unsqueeze_894 = None
    mul_925: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_35, sub_276)
    sum_109: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_925, [0, 2, 3]);  mul_925 = None
    mul_926: "f32[72]" = torch.ops.aten.mul.Tensor(sum_108, 3.985969387755102e-05)
    unsqueeze_895: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_926, 0);  mul_926 = None
    unsqueeze_896: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_895, 2);  unsqueeze_895 = None
    unsqueeze_897: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 3);  unsqueeze_896 = None
    mul_927: "f32[72]" = torch.ops.aten.mul.Tensor(sum_109, 3.985969387755102e-05)
    mul_928: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_929: "f32[72]" = torch.ops.aten.mul.Tensor(mul_927, mul_928);  mul_927 = mul_928 = None
    unsqueeze_898: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_929, 0);  mul_929 = None
    unsqueeze_899: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, 2);  unsqueeze_898 = None
    unsqueeze_900: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 3);  unsqueeze_899 = None
    mul_930: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_901: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_930, 0);  mul_930 = None
    unsqueeze_902: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_901, 2);  unsqueeze_901 = None
    unsqueeze_903: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 3);  unsqueeze_902 = None
    mul_931: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_900);  sub_276 = unsqueeze_900 = None
    sub_278: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_35, mul_931);  where_35 = mul_931 = None
    sub_279: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_278, unsqueeze_897);  sub_278 = unsqueeze_897 = None
    mul_932: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_279, unsqueeze_903);  sub_279 = unsqueeze_903 = None
    mul_933: "f32[72]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_31);  sum_109 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_932, relu_6, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_932 = primals_139 = None
    getitem_287: "f32[8, 72, 56, 56]" = convolution_backward_53[0]
    getitem_288: "f32[72, 1, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_152: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_153: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(alias_152);  alias_152 = None
    le_36: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_153, 0);  alias_153 = None
    where_36: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_36, full_default, getitem_287);  le_36 = getitem_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_110: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_280: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_906);  convolution_9 = unsqueeze_906 = None
    mul_934: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_36, sub_280)
    sum_111: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_934, [0, 2, 3]);  mul_934 = None
    mul_935: "f32[72]" = torch.ops.aten.mul.Tensor(sum_110, 3.985969387755102e-05)
    unsqueeze_907: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_935, 0);  mul_935 = None
    unsqueeze_908: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_907, 2);  unsqueeze_907 = None
    unsqueeze_909: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 3);  unsqueeze_908 = None
    mul_936: "f32[72]" = torch.ops.aten.mul.Tensor(sum_111, 3.985969387755102e-05)
    mul_937: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_938: "f32[72]" = torch.ops.aten.mul.Tensor(mul_936, mul_937);  mul_936 = mul_937 = None
    unsqueeze_910: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_938, 0);  mul_938 = None
    unsqueeze_911: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, 2);  unsqueeze_910 = None
    unsqueeze_912: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 3);  unsqueeze_911 = None
    mul_939: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_913: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_939, 0);  mul_939 = None
    unsqueeze_914: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_913, 2);  unsqueeze_913 = None
    unsqueeze_915: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 3);  unsqueeze_914 = None
    mul_940: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_912);  sub_280 = unsqueeze_912 = None
    sub_282: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_36, mul_940);  where_36 = mul_940 = None
    sub_283: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_282, unsqueeze_909);  sub_282 = unsqueeze_909 = None
    mul_941: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_283, unsqueeze_915);  sub_283 = unsqueeze_915 = None
    mul_942: "f32[72]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_28);  sum_111 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_941, add_45, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_941 = add_45 = primals_138 = None
    getitem_290: "f32[8, 24, 56, 56]" = convolution_backward_54[0]
    getitem_291: "f32[72, 24, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_346: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_281, getitem_290);  getitem_281 = getitem_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_112: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_346, [0, 2, 3])
    sub_284: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_918);  convolution_8 = unsqueeze_918 = None
    mul_943: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_346, sub_284)
    sum_113: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_943, [0, 2, 3]);  mul_943 = None
    mul_944: "f32[24]" = torch.ops.aten.mul.Tensor(sum_112, 3.985969387755102e-05)
    unsqueeze_919: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_944, 0);  mul_944 = None
    unsqueeze_920: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_919, 2);  unsqueeze_919 = None
    unsqueeze_921: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, 3);  unsqueeze_920 = None
    mul_945: "f32[24]" = torch.ops.aten.mul.Tensor(sum_113, 3.985969387755102e-05)
    mul_946: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_947: "f32[24]" = torch.ops.aten.mul.Tensor(mul_945, mul_946);  mul_945 = mul_946 = None
    unsqueeze_922: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_947, 0);  mul_947 = None
    unsqueeze_923: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, 2);  unsqueeze_922 = None
    unsqueeze_924: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 3);  unsqueeze_923 = None
    mul_948: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_925: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_948, 0);  mul_948 = None
    unsqueeze_926: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_925, 2);  unsqueeze_925 = None
    unsqueeze_927: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 3);  unsqueeze_926 = None
    mul_949: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_924);  sub_284 = unsqueeze_924 = None
    sub_286: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_346, mul_949);  mul_949 = None
    sub_287: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_286, unsqueeze_921);  sub_286 = unsqueeze_921 = None
    mul_950: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_287, unsqueeze_927);  sub_287 = unsqueeze_927 = None
    mul_951: "f32[24]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_25);  sum_113 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_950, relu_5, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_950 = primals_137 = None
    getitem_293: "f32[8, 72, 56, 56]" = convolution_backward_55[0]
    getitem_294: "f32[24, 72, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_155: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_156: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(alias_155);  alias_155 = None
    le_37: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_156, 0);  alias_156 = None
    where_37: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_37, full_default, getitem_293);  le_37 = getitem_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_114: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_288: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_930);  convolution_7 = unsqueeze_930 = None
    mul_952: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_37, sub_288)
    sum_115: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_952, [0, 2, 3]);  mul_952 = None
    mul_953: "f32[72]" = torch.ops.aten.mul.Tensor(sum_114, 3.985969387755102e-05)
    unsqueeze_931: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_953, 0);  mul_953 = None
    unsqueeze_932: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_931, 2);  unsqueeze_931 = None
    unsqueeze_933: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, 3);  unsqueeze_932 = None
    mul_954: "f32[72]" = torch.ops.aten.mul.Tensor(sum_115, 3.985969387755102e-05)
    mul_955: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_956: "f32[72]" = torch.ops.aten.mul.Tensor(mul_954, mul_955);  mul_954 = mul_955 = None
    unsqueeze_934: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_956, 0);  mul_956 = None
    unsqueeze_935: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, 2);  unsqueeze_934 = None
    unsqueeze_936: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 3);  unsqueeze_935 = None
    mul_957: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_937: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_957, 0);  mul_957 = None
    unsqueeze_938: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_937, 2);  unsqueeze_937 = None
    unsqueeze_939: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 3);  unsqueeze_938 = None
    mul_958: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_936);  sub_288 = unsqueeze_936 = None
    sub_290: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_37, mul_958);  where_37 = mul_958 = None
    sub_291: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_290, unsqueeze_933);  sub_290 = unsqueeze_933 = None
    mul_959: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_291, unsqueeze_939);  sub_291 = unsqueeze_939 = None
    mul_960: "f32[72]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_22);  sum_115 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_959, relu_4, primals_136, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_959 = primals_136 = None
    getitem_296: "f32[8, 72, 56, 56]" = convolution_backward_56[0]
    getitem_297: "f32[72, 1, 3, 3]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_158: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_159: "f32[8, 72, 56, 56]" = torch.ops.aten.alias.default(alias_158);  alias_158 = None
    le_38: "b8[8, 72, 56, 56]" = torch.ops.aten.le.Scalar(alias_159, 0);  alias_159 = None
    where_38: "f32[8, 72, 56, 56]" = torch.ops.aten.where.self(le_38, full_default, getitem_296);  le_38 = getitem_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_116: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_292: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_942);  convolution_6 = unsqueeze_942 = None
    mul_961: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_38, sub_292)
    sum_117: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_961, [0, 2, 3]);  mul_961 = None
    mul_962: "f32[72]" = torch.ops.aten.mul.Tensor(sum_116, 3.985969387755102e-05)
    unsqueeze_943: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_962, 0);  mul_962 = None
    unsqueeze_944: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_943, 2);  unsqueeze_943 = None
    unsqueeze_945: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, 3);  unsqueeze_944 = None
    mul_963: "f32[72]" = torch.ops.aten.mul.Tensor(sum_117, 3.985969387755102e-05)
    mul_964: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_965: "f32[72]" = torch.ops.aten.mul.Tensor(mul_963, mul_964);  mul_963 = mul_964 = None
    unsqueeze_946: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_965, 0);  mul_965 = None
    unsqueeze_947: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, 2);  unsqueeze_946 = None
    unsqueeze_948: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 3);  unsqueeze_947 = None
    mul_966: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_949: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_966, 0);  mul_966 = None
    unsqueeze_950: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_949, 2);  unsqueeze_949 = None
    unsqueeze_951: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 3);  unsqueeze_950 = None
    mul_967: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_948);  sub_292 = unsqueeze_948 = None
    sub_294: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(where_38, mul_967);  where_38 = mul_967 = None
    sub_295: "f32[8, 72, 56, 56]" = torch.ops.aten.sub.Tensor(sub_294, unsqueeze_945);  sub_294 = unsqueeze_945 = None
    mul_968: "f32[8, 72, 56, 56]" = torch.ops.aten.mul.Tensor(sub_295, unsqueeze_951);  sub_295 = unsqueeze_951 = None
    mul_969: "f32[72]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_19);  sum_117 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_968, add_29, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_968 = add_29 = primals_135 = None
    getitem_299: "f32[8, 24, 56, 56]" = convolution_backward_57[0]
    getitem_300: "f32[72, 24, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_347: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_346, getitem_299);  add_346 = getitem_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_118: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_347, [0, 2, 3])
    sub_296: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_954);  convolution_5 = unsqueeze_954 = None
    mul_970: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_347, sub_296)
    sum_119: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_970, [0, 2, 3]);  mul_970 = None
    mul_971: "f32[24]" = torch.ops.aten.mul.Tensor(sum_118, 3.985969387755102e-05)
    unsqueeze_955: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_971, 0);  mul_971 = None
    unsqueeze_956: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_955, 2);  unsqueeze_955 = None
    unsqueeze_957: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, 3);  unsqueeze_956 = None
    mul_972: "f32[24]" = torch.ops.aten.mul.Tensor(sum_119, 3.985969387755102e-05)
    mul_973: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_974: "f32[24]" = torch.ops.aten.mul.Tensor(mul_972, mul_973);  mul_972 = mul_973 = None
    unsqueeze_958: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_974, 0);  mul_974 = None
    unsqueeze_959: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, 2);  unsqueeze_958 = None
    unsqueeze_960: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 3);  unsqueeze_959 = None
    mul_975: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_961: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_975, 0);  mul_975 = None
    unsqueeze_962: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_961, 2);  unsqueeze_961 = None
    unsqueeze_963: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 3);  unsqueeze_962 = None
    mul_976: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_960);  sub_296 = unsqueeze_960 = None
    sub_298: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_347, mul_976);  add_347 = mul_976 = None
    sub_299: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_298, unsqueeze_957);  sub_298 = unsqueeze_957 = None
    mul_977: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_299, unsqueeze_963);  sub_299 = unsqueeze_963 = None
    mul_978: "f32[24]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_16);  sum_119 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_977, relu_3, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_977 = primals_134 = None
    getitem_302: "f32[8, 48, 56, 56]" = convolution_backward_58[0]
    getitem_303: "f32[24, 48, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_161: "f32[8, 48, 56, 56]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_162: "f32[8, 48, 56, 56]" = torch.ops.aten.alias.default(alias_161);  alias_161 = None
    le_39: "b8[8, 48, 56, 56]" = torch.ops.aten.le.Scalar(alias_162, 0);  alias_162 = None
    where_39: "f32[8, 48, 56, 56]" = torch.ops.aten.where.self(le_39, full_default, getitem_302);  le_39 = getitem_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_120: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_300: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_966);  convolution_4 = unsqueeze_966 = None
    mul_979: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(where_39, sub_300)
    sum_121: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_979, [0, 2, 3]);  mul_979 = None
    mul_980: "f32[48]" = torch.ops.aten.mul.Tensor(sum_120, 3.985969387755102e-05)
    unsqueeze_967: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_980, 0);  mul_980 = None
    unsqueeze_968: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_967, 2);  unsqueeze_967 = None
    unsqueeze_969: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, 3);  unsqueeze_968 = None
    mul_981: "f32[48]" = torch.ops.aten.mul.Tensor(sum_121, 3.985969387755102e-05)
    mul_982: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_983: "f32[48]" = torch.ops.aten.mul.Tensor(mul_981, mul_982);  mul_981 = mul_982 = None
    unsqueeze_970: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_983, 0);  mul_983 = None
    unsqueeze_971: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, 2);  unsqueeze_970 = None
    unsqueeze_972: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 3);  unsqueeze_971 = None
    mul_984: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_973: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_984, 0);  mul_984 = None
    unsqueeze_974: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_973, 2);  unsqueeze_973 = None
    unsqueeze_975: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 3);  unsqueeze_974 = None
    mul_985: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_972);  sub_300 = unsqueeze_972 = None
    sub_302: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(where_39, mul_985);  where_39 = mul_985 = None
    sub_303: "f32[8, 48, 56, 56]" = torch.ops.aten.sub.Tensor(sub_302, unsqueeze_969);  sub_302 = unsqueeze_969 = None
    mul_986: "f32[8, 48, 56, 56]" = torch.ops.aten.mul.Tensor(sub_303, unsqueeze_975);  sub_303 = unsqueeze_975 = None
    mul_987: "f32[48]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_13);  sum_121 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_986, relu_2, primals_133, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 48, [True, True, False]);  mul_986 = primals_133 = None
    getitem_305: "f32[8, 48, 112, 112]" = convolution_backward_59[0]
    getitem_306: "f32[48, 1, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_164: "f32[8, 48, 112, 112]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_165: "f32[8, 48, 112, 112]" = torch.ops.aten.alias.default(alias_164);  alias_164 = None
    le_40: "b8[8, 48, 112, 112]" = torch.ops.aten.le.Scalar(alias_165, 0);  alias_165 = None
    where_40: "f32[8, 48, 112, 112]" = torch.ops.aten.where.self(le_40, full_default, getitem_305);  le_40 = getitem_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_122: "f32[48]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_304: "f32[8, 48, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_978);  convolution_3 = unsqueeze_978 = None
    mul_988: "f32[8, 48, 112, 112]" = torch.ops.aten.mul.Tensor(where_40, sub_304)
    sum_123: "f32[48]" = torch.ops.aten.sum.dim_IntList(mul_988, [0, 2, 3]);  mul_988 = None
    mul_989: "f32[48]" = torch.ops.aten.mul.Tensor(sum_122, 9.964923469387754e-06)
    unsqueeze_979: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_989, 0);  mul_989 = None
    unsqueeze_980: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_979, 2);  unsqueeze_979 = None
    unsqueeze_981: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, 3);  unsqueeze_980 = None
    mul_990: "f32[48]" = torch.ops.aten.mul.Tensor(sum_123, 9.964923469387754e-06)
    mul_991: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_992: "f32[48]" = torch.ops.aten.mul.Tensor(mul_990, mul_991);  mul_990 = mul_991 = None
    unsqueeze_982: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_992, 0);  mul_992 = None
    unsqueeze_983: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, 2);  unsqueeze_982 = None
    unsqueeze_984: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 3);  unsqueeze_983 = None
    mul_993: "f32[48]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_985: "f32[1, 48]" = torch.ops.aten.unsqueeze.default(mul_993, 0);  mul_993 = None
    unsqueeze_986: "f32[1, 48, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_985, 2);  unsqueeze_985 = None
    unsqueeze_987: "f32[1, 48, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 3);  unsqueeze_986 = None
    mul_994: "f32[8, 48, 112, 112]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_984);  sub_304 = unsqueeze_984 = None
    sub_306: "f32[8, 48, 112, 112]" = torch.ops.aten.sub.Tensor(where_40, mul_994);  where_40 = mul_994 = None
    sub_307: "f32[8, 48, 112, 112]" = torch.ops.aten.sub.Tensor(sub_306, unsqueeze_981);  sub_306 = unsqueeze_981 = None
    mul_995: "f32[8, 48, 112, 112]" = torch.ops.aten.mul.Tensor(sub_307, unsqueeze_987);  sub_307 = unsqueeze_987 = None
    mul_996: "f32[48]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_10);  sum_123 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_995, add_14, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_995 = add_14 = primals_132 = None
    getitem_308: "f32[8, 16, 112, 112]" = convolution_backward_60[0]
    getitem_309: "f32[48, 16, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_124: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_308, [0, 2, 3])
    sub_308: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_990);  convolution_2 = unsqueeze_990 = None
    mul_997: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_308, sub_308)
    sum_125: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_997, [0, 2, 3]);  mul_997 = None
    mul_998: "f32[16]" = torch.ops.aten.mul.Tensor(sum_124, 9.964923469387754e-06)
    unsqueeze_991: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_998, 0);  mul_998 = None
    unsqueeze_992: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_991, 2);  unsqueeze_991 = None
    unsqueeze_993: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, 3);  unsqueeze_992 = None
    mul_999: "f32[16]" = torch.ops.aten.mul.Tensor(sum_125, 9.964923469387754e-06)
    mul_1000: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1001: "f32[16]" = torch.ops.aten.mul.Tensor(mul_999, mul_1000);  mul_999 = mul_1000 = None
    unsqueeze_994: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1001, 0);  mul_1001 = None
    unsqueeze_995: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, 2);  unsqueeze_994 = None
    unsqueeze_996: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 3);  unsqueeze_995 = None
    mul_1002: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_997: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1002, 0);  mul_1002 = None
    unsqueeze_998: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_997, 2);  unsqueeze_997 = None
    unsqueeze_999: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 3);  unsqueeze_998 = None
    mul_1003: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_996);  sub_308 = unsqueeze_996 = None
    sub_310: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(getitem_308, mul_1003);  getitem_308 = mul_1003 = None
    sub_311: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_310, unsqueeze_993);  sub_310 = unsqueeze_993 = None
    mul_1004: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_311, unsqueeze_999);  sub_311 = unsqueeze_999 = None
    mul_1005: "f32[16]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_7);  sum_125 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1004, relu_1, primals_131, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1004 = primals_131 = None
    getitem_311: "f32[8, 32, 112, 112]" = convolution_backward_61[0]
    getitem_312: "f32[16, 32, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_167: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_168: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(alias_167);  alias_167 = None
    le_41: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_168, 0);  alias_168 = None
    where_41: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_41, full_default, getitem_311);  le_41 = getitem_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_126: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_312: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1002);  convolution_1 = unsqueeze_1002 = None
    mul_1006: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_41, sub_312)
    sum_127: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1006, [0, 2, 3]);  mul_1006 = None
    mul_1007: "f32[32]" = torch.ops.aten.mul.Tensor(sum_126, 9.964923469387754e-06)
    unsqueeze_1003: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1007, 0);  mul_1007 = None
    unsqueeze_1004: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1003, 2);  unsqueeze_1003 = None
    unsqueeze_1005: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, 3);  unsqueeze_1004 = None
    mul_1008: "f32[32]" = torch.ops.aten.mul.Tensor(sum_127, 9.964923469387754e-06)
    mul_1009: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1010: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1008, mul_1009);  mul_1008 = mul_1009 = None
    unsqueeze_1006: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1010, 0);  mul_1010 = None
    unsqueeze_1007: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, 2);  unsqueeze_1006 = None
    unsqueeze_1008: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 3);  unsqueeze_1007 = None
    mul_1011: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_1009: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1011, 0);  mul_1011 = None
    unsqueeze_1010: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1009, 2);  unsqueeze_1009 = None
    unsqueeze_1011: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 3);  unsqueeze_1010 = None
    mul_1012: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1008);  sub_312 = unsqueeze_1008 = None
    sub_314: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_41, mul_1012);  where_41 = mul_1012 = None
    sub_315: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_314, unsqueeze_1005);  sub_314 = unsqueeze_1005 = None
    mul_1013: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_315, unsqueeze_1011);  sub_315 = unsqueeze_1011 = None
    mul_1014: "f32[32]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_4);  sum_127 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1013, relu, primals_130, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, False]);  mul_1013 = primals_130 = None
    getitem_314: "f32[8, 32, 112, 112]" = convolution_backward_62[0]
    getitem_315: "f32[32, 1, 3, 3]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_170: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_171: "f32[8, 32, 112, 112]" = torch.ops.aten.alias.default(alias_170);  alias_170 = None
    le_42: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(alias_171, 0);  alias_171 = None
    where_42: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_42, full_default, getitem_314);  le_42 = full_default = getitem_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_128: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_316: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1014);  convolution = unsqueeze_1014 = None
    mul_1015: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_42, sub_316)
    sum_129: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_1015, [0, 2, 3]);  mul_1015 = None
    mul_1016: "f32[32]" = torch.ops.aten.mul.Tensor(sum_128, 9.964923469387754e-06)
    unsqueeze_1015: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1016, 0);  mul_1016 = None
    unsqueeze_1016: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1015, 2);  unsqueeze_1015 = None
    unsqueeze_1017: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, 3);  unsqueeze_1016 = None
    mul_1017: "f32[32]" = torch.ops.aten.mul.Tensor(sum_129, 9.964923469387754e-06)
    mul_1018: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1019: "f32[32]" = torch.ops.aten.mul.Tensor(mul_1017, mul_1018);  mul_1017 = mul_1018 = None
    unsqueeze_1018: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1019, 0);  mul_1019 = None
    unsqueeze_1019: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, 2);  unsqueeze_1018 = None
    unsqueeze_1020: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 3);  unsqueeze_1019 = None
    mul_1020: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_1021: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_1020, 0);  mul_1020 = None
    unsqueeze_1022: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1021, 2);  unsqueeze_1021 = None
    unsqueeze_1023: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 3);  unsqueeze_1022 = None
    mul_1021: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1020);  sub_316 = unsqueeze_1020 = None
    sub_318: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_42, mul_1021);  where_42 = mul_1021 = None
    sub_319: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_318, unsqueeze_1017);  sub_318 = unsqueeze_1017 = None
    mul_1022: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_319, unsqueeze_1023);  sub_319 = unsqueeze_1023 = None
    mul_1023: "f32[32]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_1);  sum_129 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1022, primals_387, primals_129, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1022 = primals_387 = primals_129 = None
    getitem_318: "f32[32, 3, 3, 3]" = convolution_backward_63[1];  convolution_backward_63 = None
    return [mul_1023, sum_128, mul_1014, sum_126, mul_1005, sum_124, mul_996, sum_122, mul_987, sum_120, mul_978, sum_118, mul_969, sum_116, mul_960, sum_114, mul_951, sum_112, mul_942, sum_110, mul_933, sum_108, mul_924, sum_106, mul_915, sum_104, mul_906, sum_102, mul_897, sum_100, mul_888, sum_98, mul_879, sum_96, mul_870, sum_94, mul_861, sum_92, mul_852, sum_90, mul_843, sum_88, mul_834, sum_86, mul_825, sum_84, mul_816, sum_82, mul_807, sum_80, mul_798, sum_78, mul_789, sum_76, mul_780, sum_74, mul_771, sum_72, mul_762, sum_70, mul_753, sum_68, mul_744, sum_66, mul_735, sum_64, mul_726, sum_62, mul_717, sum_60, mul_708, sum_58, mul_699, sum_56, mul_690, sum_54, mul_681, sum_52, mul_672, sum_50, mul_663, sum_48, mul_654, sum_46, mul_645, sum_44, mul_636, sum_42, mul_627, sum_40, mul_618, sum_38, mul_609, sum_36, mul_600, sum_34, mul_591, sum_32, mul_582, sum_30, mul_573, sum_28, mul_564, sum_26, mul_555, sum_24, mul_546, sum_22, mul_537, sum_20, mul_528, sum_18, mul_519, sum_16, mul_510, sum_14, mul_501, sum_12, mul_492, sum_10, mul_483, sum_8, mul_474, sum_6, mul_465, sum_4, mul_456, sum_2, getitem_318, getitem_315, getitem_312, getitem_309, getitem_306, getitem_303, getitem_300, getitem_297, getitem_294, getitem_291, getitem_288, getitem_285, getitem_282, getitem_279, getitem_276, getitem_273, getitem_270, getitem_267, getitem_264, getitem_261, getitem_258, getitem_255, getitem_252, getitem_249, getitem_246, getitem_243, getitem_240, getitem_237, getitem_234, getitem_231, getitem_228, getitem_225, getitem_222, getitem_219, getitem_216, getitem_213, getitem_210, getitem_207, getitem_204, getitem_201, getitem_198, getitem_195, getitem_192, getitem_189, getitem_186, getitem_183, getitem_180, getitem_177, getitem_174, getitem_171, getitem_168, getitem_165, getitem_162, getitem_159, getitem_156, getitem_153, getitem_150, getitem_147, getitem_144, getitem_141, getitem_138, getitem_135, getitem_132, getitem_129, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    