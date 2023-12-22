from __future__ import annotations



def forward(self, primals_1: "f32[16]", primals_3: "f32[16]", primals_5: "f32[16]", primals_7: "f32[16]", primals_9: "f32[96]", primals_11: "f32[96]", primals_13: "f32[24]", primals_15: "f32[24]", primals_17: "f32[24]", primals_19: "f32[24]", primals_21: "f32[24]", primals_23: "f32[24]", primals_25: "f32[24]", primals_27: "f32[144]", primals_29: "f32[144]", primals_31: "f32[32]", primals_33: "f32[96]", primals_35: "f32[96]", primals_37: "f32[32]", primals_39: "f32[192]", primals_41: "f32[192]", primals_43: "f32[32]", primals_45: "f32[192]", primals_47: "f32[192]", primals_49: "f32[32]", primals_51: "f32[192]", primals_53: "f32[192]", primals_55: "f32[64]", primals_57: "f32[192]", primals_59: "f32[192]", primals_61: "f32[64]", primals_63: "f32[384]", primals_65: "f32[384]", primals_67: "f32[64]", primals_69: "f32[384]", primals_71: "f32[384]", primals_73: "f32[64]", primals_75: "f32[384]", primals_77: "f32[384]", primals_79: "f32[112]", primals_81: "f32[672]", primals_83: "f32[672]", primals_85: "f32[112]", primals_87: "f32[672]", primals_89: "f32[672]", primals_91: "f32[112]", primals_93: "f32[336]", primals_95: "f32[336]", primals_97: "f32[112]", primals_99: "f32[672]", primals_101: "f32[672]", primals_103: "f32[184]", primals_105: "f32[1104]", primals_107: "f32[1104]", primals_109: "f32[184]", primals_111: "f32[1104]", primals_113: "f32[1104]", primals_115: "f32[184]", primals_117: "f32[1104]", primals_119: "f32[1104]", primals_121: "f32[184]", primals_123: "f32[1104]", primals_125: "f32[1104]", primals_127: "f32[352]", primals_129: "f32[1984]", primals_131: "f32[16, 3, 3, 3]", primals_132: "f32[16, 16, 1, 1]", primals_133: "f32[16, 1, 3, 3]", primals_134: "f32[16, 16, 1, 1]", primals_135: "f32[96, 16, 1, 1]", primals_136: "f32[96, 1, 3, 3]", primals_137: "f32[24, 96, 1, 1]", primals_138: "f32[24, 24, 1, 1]", primals_139: "f32[24, 1, 3, 3]", primals_140: "f32[24, 24, 1, 1]", primals_141: "f32[24, 24, 1, 1]", primals_142: "f32[24, 1, 3, 3]", primals_143: "f32[24, 24, 1, 1]", primals_144: "f32[144, 24, 1, 1]", primals_145: "f32[144, 1, 5, 5]", primals_146: "f32[32, 144, 1, 1]", primals_147: "f32[96, 32, 1, 1]", primals_148: "f32[96, 1, 5, 5]", primals_149: "f32[32, 96, 1, 1]", primals_150: "f32[192, 32, 1, 1]", primals_151: "f32[192, 1, 5, 5]", primals_152: "f32[32, 192, 1, 1]", primals_153: "f32[192, 32, 1, 1]", primals_154: "f32[192, 1, 3, 3]", primals_155: "f32[32, 192, 1, 1]", primals_156: "f32[192, 32, 1, 1]", primals_157: "f32[192, 1, 5, 5]", primals_158: "f32[64, 192, 1, 1]", primals_159: "f32[192, 64, 1, 1]", primals_160: "f32[192, 1, 5, 5]", primals_161: "f32[64, 192, 1, 1]", primals_162: "f32[384, 64, 1, 1]", primals_163: "f32[384, 1, 5, 5]", primals_164: "f32[64, 384, 1, 1]", primals_165: "f32[384, 64, 1, 1]", primals_166: "f32[384, 1, 5, 5]", primals_167: "f32[64, 384, 1, 1]", primals_168: "f32[384, 64, 1, 1]", primals_169: "f32[384, 1, 5, 5]", primals_170: "f32[112, 384, 1, 1]", primals_171: "f32[672, 112, 1, 1]", primals_172: "f32[672, 1, 5, 5]", primals_173: "f32[112, 672, 1, 1]", primals_174: "f32[672, 112, 1, 1]", primals_175: "f32[672, 1, 5, 5]", primals_176: "f32[112, 672, 1, 1]", primals_177: "f32[336, 112, 1, 1]", primals_178: "f32[336, 1, 5, 5]", primals_179: "f32[112, 336, 1, 1]", primals_180: "f32[672, 112, 1, 1]", primals_181: "f32[672, 1, 5, 5]", primals_182: "f32[184, 672, 1, 1]", primals_183: "f32[1104, 184, 1, 1]", primals_184: "f32[1104, 1, 5, 5]", primals_185: "f32[184, 1104, 1, 1]", primals_186: "f32[1104, 184, 1, 1]", primals_187: "f32[1104, 1, 5, 5]", primals_188: "f32[184, 1104, 1, 1]", primals_189: "f32[1104, 184, 1, 1]", primals_190: "f32[1104, 1, 5, 5]", primals_191: "f32[184, 1104, 1, 1]", primals_192: "f32[1104, 184, 1, 1]", primals_193: "f32[1104, 1, 3, 3]", primals_194: "f32[352, 1104, 1, 1]", primals_195: "f32[1984, 352, 1, 1]", primals_393: "f32[8, 3, 224, 224]", convolution: "f32[8, 16, 112, 112]", squeeze_1: "f32[16]", relu: "f32[8, 16, 112, 112]", convolution_1: "f32[8, 16, 112, 112]", squeeze_4: "f32[16]", relu_1: "f32[8, 16, 112, 112]", convolution_2: "f32[8, 16, 112, 112]", squeeze_7: "f32[16]", relu_2: "f32[8, 16, 112, 112]", convolution_3: "f32[8, 16, 112, 112]", squeeze_10: "f32[16]", add_20: "f32[8, 16, 112, 112]", convolution_4: "f32[8, 96, 112, 112]", squeeze_13: "f32[96]", relu_3: "f32[8, 96, 112, 112]", convolution_5: "f32[8, 96, 56, 56]", squeeze_16: "f32[96]", relu_4: "f32[8, 96, 56, 56]", convolution_6: "f32[8, 24, 56, 56]", squeeze_19: "f32[24]", add_35: "f32[8, 24, 56, 56]", convolution_7: "f32[8, 24, 56, 56]", squeeze_22: "f32[24]", relu_5: "f32[8, 24, 56, 56]", convolution_8: "f32[8, 24, 56, 56]", squeeze_25: "f32[24]", relu_6: "f32[8, 24, 56, 56]", convolution_9: "f32[8, 24, 56, 56]", squeeze_28: "f32[24]", add_51: "f32[8, 24, 56, 56]", convolution_10: "f32[8, 24, 56, 56]", squeeze_31: "f32[24]", relu_7: "f32[8, 24, 56, 56]", convolution_11: "f32[8, 24, 56, 56]", squeeze_34: "f32[24]", relu_8: "f32[8, 24, 56, 56]", convolution_12: "f32[8, 24, 56, 56]", squeeze_37: "f32[24]", add_67: "f32[8, 24, 56, 56]", convolution_13: "f32[8, 144, 56, 56]", squeeze_40: "f32[144]", relu_9: "f32[8, 144, 56, 56]", convolution_14: "f32[8, 144, 28, 28]", squeeze_43: "f32[144]", relu_10: "f32[8, 144, 28, 28]", convolution_15: "f32[8, 32, 28, 28]", squeeze_46: "f32[32]", add_82: "f32[8, 32, 28, 28]", convolution_16: "f32[8, 96, 28, 28]", squeeze_49: "f32[96]", relu_11: "f32[8, 96, 28, 28]", convolution_17: "f32[8, 96, 28, 28]", squeeze_52: "f32[96]", relu_12: "f32[8, 96, 28, 28]", convolution_18: "f32[8, 32, 28, 28]", squeeze_55: "f32[32]", add_98: "f32[8, 32, 28, 28]", convolution_19: "f32[8, 192, 28, 28]", squeeze_58: "f32[192]", relu_13: "f32[8, 192, 28, 28]", convolution_20: "f32[8, 192, 28, 28]", squeeze_61: "f32[192]", relu_14: "f32[8, 192, 28, 28]", convolution_21: "f32[8, 32, 28, 28]", squeeze_64: "f32[32]", add_114: "f32[8, 32, 28, 28]", convolution_22: "f32[8, 192, 28, 28]", squeeze_67: "f32[192]", relu_15: "f32[8, 192, 28, 28]", convolution_23: "f32[8, 192, 28, 28]", squeeze_70: "f32[192]", relu_16: "f32[8, 192, 28, 28]", convolution_24: "f32[8, 32, 28, 28]", squeeze_73: "f32[32]", add_130: "f32[8, 32, 28, 28]", convolution_25: "f32[8, 192, 28, 28]", squeeze_76: "f32[192]", relu_17: "f32[8, 192, 28, 28]", convolution_26: "f32[8, 192, 14, 14]", squeeze_79: "f32[192]", relu_18: "f32[8, 192, 14, 14]", convolution_27: "f32[8, 64, 14, 14]", squeeze_82: "f32[64]", add_145: "f32[8, 64, 14, 14]", convolution_28: "f32[8, 192, 14, 14]", squeeze_85: "f32[192]", relu_19: "f32[8, 192, 14, 14]", convolution_29: "f32[8, 192, 14, 14]", squeeze_88: "f32[192]", relu_20: "f32[8, 192, 14, 14]", convolution_30: "f32[8, 64, 14, 14]", squeeze_91: "f32[64]", add_161: "f32[8, 64, 14, 14]", convolution_31: "f32[8, 384, 14, 14]", squeeze_94: "f32[384]", relu_21: "f32[8, 384, 14, 14]", convolution_32: "f32[8, 384, 14, 14]", squeeze_97: "f32[384]", relu_22: "f32[8, 384, 14, 14]", convolution_33: "f32[8, 64, 14, 14]", squeeze_100: "f32[64]", add_177: "f32[8, 64, 14, 14]", convolution_34: "f32[8, 384, 14, 14]", squeeze_103: "f32[384]", relu_23: "f32[8, 384, 14, 14]", convolution_35: "f32[8, 384, 14, 14]", squeeze_106: "f32[384]", relu_24: "f32[8, 384, 14, 14]", convolution_36: "f32[8, 64, 14, 14]", squeeze_109: "f32[64]", add_193: "f32[8, 64, 14, 14]", convolution_37: "f32[8, 384, 14, 14]", squeeze_112: "f32[384]", relu_25: "f32[8, 384, 14, 14]", convolution_38: "f32[8, 384, 14, 14]", squeeze_115: "f32[384]", relu_26: "f32[8, 384, 14, 14]", convolution_39: "f32[8, 112, 14, 14]", squeeze_118: "f32[112]", add_208: "f32[8, 112, 14, 14]", convolution_40: "f32[8, 672, 14, 14]", squeeze_121: "f32[672]", relu_27: "f32[8, 672, 14, 14]", convolution_41: "f32[8, 672, 14, 14]", squeeze_124: "f32[672]", relu_28: "f32[8, 672, 14, 14]", convolution_42: "f32[8, 112, 14, 14]", squeeze_127: "f32[112]", add_224: "f32[8, 112, 14, 14]", convolution_43: "f32[8, 672, 14, 14]", squeeze_130: "f32[672]", relu_29: "f32[8, 672, 14, 14]", convolution_44: "f32[8, 672, 14, 14]", squeeze_133: "f32[672]", relu_30: "f32[8, 672, 14, 14]", convolution_45: "f32[8, 112, 14, 14]", squeeze_136: "f32[112]", add_240: "f32[8, 112, 14, 14]", convolution_46: "f32[8, 336, 14, 14]", squeeze_139: "f32[336]", relu_31: "f32[8, 336, 14, 14]", convolution_47: "f32[8, 336, 14, 14]", squeeze_142: "f32[336]", relu_32: "f32[8, 336, 14, 14]", convolution_48: "f32[8, 112, 14, 14]", squeeze_145: "f32[112]", add_256: "f32[8, 112, 14, 14]", convolution_49: "f32[8, 672, 14, 14]", squeeze_148: "f32[672]", relu_33: "f32[8, 672, 14, 14]", convolution_50: "f32[8, 672, 7, 7]", squeeze_151: "f32[672]", relu_34: "f32[8, 672, 7, 7]", convolution_51: "f32[8, 184, 7, 7]", squeeze_154: "f32[184]", add_271: "f32[8, 184, 7, 7]", convolution_52: "f32[8, 1104, 7, 7]", squeeze_157: "f32[1104]", relu_35: "f32[8, 1104, 7, 7]", convolution_53: "f32[8, 1104, 7, 7]", squeeze_160: "f32[1104]", relu_36: "f32[8, 1104, 7, 7]", convolution_54: "f32[8, 184, 7, 7]", squeeze_163: "f32[184]", add_287: "f32[8, 184, 7, 7]", convolution_55: "f32[8, 1104, 7, 7]", squeeze_166: "f32[1104]", relu_37: "f32[8, 1104, 7, 7]", convolution_56: "f32[8, 1104, 7, 7]", squeeze_169: "f32[1104]", relu_38: "f32[8, 1104, 7, 7]", convolution_57: "f32[8, 184, 7, 7]", squeeze_172: "f32[184]", add_303: "f32[8, 184, 7, 7]", convolution_58: "f32[8, 1104, 7, 7]", squeeze_175: "f32[1104]", relu_39: "f32[8, 1104, 7, 7]", convolution_59: "f32[8, 1104, 7, 7]", squeeze_178: "f32[1104]", relu_40: "f32[8, 1104, 7, 7]", convolution_60: "f32[8, 184, 7, 7]", squeeze_181: "f32[184]", add_319: "f32[8, 184, 7, 7]", convolution_61: "f32[8, 1104, 7, 7]", squeeze_184: "f32[1104]", relu_41: "f32[8, 1104, 7, 7]", convolution_62: "f32[8, 1104, 7, 7]", squeeze_187: "f32[1104]", relu_42: "f32[8, 1104, 7, 7]", convolution_63: "f32[8, 352, 7, 7]", squeeze_190: "f32[352]", add_334: "f32[8, 352, 7, 7]", convolution_64: "f32[8, 1984, 7, 7]", squeeze_193: "f32[1984]", view: "f32[8, 1984]", permute_1: "f32[1000, 1984]", le: "b8[8, 1984, 7, 7]", unsqueeze_262: "f32[1, 1984, 1, 1]", unsqueeze_274: "f32[1, 352, 1, 1]", unsqueeze_286: "f32[1, 1104, 1, 1]", unsqueeze_298: "f32[1, 1104, 1, 1]", unsqueeze_310: "f32[1, 184, 1, 1]", unsqueeze_322: "f32[1, 1104, 1, 1]", unsqueeze_334: "f32[1, 1104, 1, 1]", unsqueeze_346: "f32[1, 184, 1, 1]", unsqueeze_358: "f32[1, 1104, 1, 1]", unsqueeze_370: "f32[1, 1104, 1, 1]", unsqueeze_382: "f32[1, 184, 1, 1]", unsqueeze_394: "f32[1, 1104, 1, 1]", unsqueeze_406: "f32[1, 1104, 1, 1]", unsqueeze_418: "f32[1, 184, 1, 1]", unsqueeze_430: "f32[1, 672, 1, 1]", unsqueeze_442: "f32[1, 672, 1, 1]", unsqueeze_454: "f32[1, 112, 1, 1]", unsqueeze_466: "f32[1, 336, 1, 1]", unsqueeze_478: "f32[1, 336, 1, 1]", unsqueeze_490: "f32[1, 112, 1, 1]", unsqueeze_502: "f32[1, 672, 1, 1]", unsqueeze_514: "f32[1, 672, 1, 1]", unsqueeze_526: "f32[1, 112, 1, 1]", unsqueeze_538: "f32[1, 672, 1, 1]", unsqueeze_550: "f32[1, 672, 1, 1]", unsqueeze_562: "f32[1, 112, 1, 1]", unsqueeze_574: "f32[1, 384, 1, 1]", unsqueeze_586: "f32[1, 384, 1, 1]", unsqueeze_598: "f32[1, 64, 1, 1]", unsqueeze_610: "f32[1, 384, 1, 1]", unsqueeze_622: "f32[1, 384, 1, 1]", unsqueeze_634: "f32[1, 64, 1, 1]", unsqueeze_646: "f32[1, 384, 1, 1]", unsqueeze_658: "f32[1, 384, 1, 1]", unsqueeze_670: "f32[1, 64, 1, 1]", unsqueeze_682: "f32[1, 192, 1, 1]", unsqueeze_694: "f32[1, 192, 1, 1]", unsqueeze_706: "f32[1, 64, 1, 1]", unsqueeze_718: "f32[1, 192, 1, 1]", unsqueeze_730: "f32[1, 192, 1, 1]", unsqueeze_742: "f32[1, 32, 1, 1]", unsqueeze_754: "f32[1, 192, 1, 1]", unsqueeze_766: "f32[1, 192, 1, 1]", unsqueeze_778: "f32[1, 32, 1, 1]", unsqueeze_790: "f32[1, 192, 1, 1]", unsqueeze_802: "f32[1, 192, 1, 1]", unsqueeze_814: "f32[1, 32, 1, 1]", unsqueeze_826: "f32[1, 96, 1, 1]", unsqueeze_838: "f32[1, 96, 1, 1]", unsqueeze_850: "f32[1, 32, 1, 1]", unsqueeze_862: "f32[1, 144, 1, 1]", unsqueeze_874: "f32[1, 144, 1, 1]", unsqueeze_886: "f32[1, 24, 1, 1]", unsqueeze_898: "f32[1, 24, 1, 1]", unsqueeze_910: "f32[1, 24, 1, 1]", unsqueeze_922: "f32[1, 24, 1, 1]", unsqueeze_934: "f32[1, 24, 1, 1]", unsqueeze_946: "f32[1, 24, 1, 1]", unsqueeze_958: "f32[1, 24, 1, 1]", unsqueeze_970: "f32[1, 96, 1, 1]", unsqueeze_982: "f32[1, 96, 1, 1]", unsqueeze_994: "f32[1, 16, 1, 1]", unsqueeze_1006: "f32[1, 16, 1, 1]", unsqueeze_1018: "f32[1, 16, 1, 1]", unsqueeze_1030: "f32[1, 16, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    mm: "f32[8, 1984]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1984]" = torch.ops.aten.mm.default(permute_2, view);  permute_2 = view = None
    permute_3: "f32[1984, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1984]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1984, 1, 1]" = torch.ops.aten.view.default(mm, [8, 1984, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1984, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 1984, 7, 7]);  view_2 = None
    div: "f32[8, 1984, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 1984, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[1984]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_65: "f32[8, 1984, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_262);  convolution_64 = unsqueeze_262 = None
    mul_455: "f32[8, 1984, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_65)
    sum_3: "f32[1984]" = torch.ops.aten.sum.dim_IntList(mul_455, [0, 2, 3]);  mul_455 = None
    mul_456: "f32[1984]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_263: "f32[1, 1984]" = torch.ops.aten.unsqueeze.default(mul_456, 0);  mul_456 = None
    unsqueeze_264: "f32[1, 1984, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_457: "f32[1984]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_458: "f32[1984]" = torch.ops.aten.mul.Tensor(squeeze_193, squeeze_193)
    mul_459: "f32[1984]" = torch.ops.aten.mul.Tensor(mul_457, mul_458);  mul_457 = mul_458 = None
    unsqueeze_266: "f32[1, 1984]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_267: "f32[1, 1984, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_460: "f32[1984]" = torch.ops.aten.mul.Tensor(squeeze_193, primals_129);  primals_129 = None
    unsqueeze_269: "f32[1, 1984]" = torch.ops.aten.unsqueeze.default(mul_460, 0);  mul_460 = None
    unsqueeze_270: "f32[1, 1984, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 1984, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    mul_461: "f32[8, 1984, 7, 7]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_268);  sub_65 = unsqueeze_268 = None
    sub_67: "f32[8, 1984, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_461);  where = mul_461 = None
    sub_68: "f32[8, 1984, 7, 7]" = torch.ops.aten.sub.Tensor(sub_67, unsqueeze_265);  sub_67 = unsqueeze_265 = None
    mul_462: "f32[8, 1984, 7, 7]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_271);  sub_68 = unsqueeze_271 = None
    mul_463: "f32[1984]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_193);  sum_3 = squeeze_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_462, add_334, primals_195, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_462 = add_334 = primals_195 = None
    getitem_130: "f32[8, 352, 7, 7]" = convolution_backward[0]
    getitem_131: "f32[1984, 352, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_4: "f32[352]" = torch.ops.aten.sum.dim_IntList(getitem_130, [0, 2, 3])
    sub_69: "f32[8, 352, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_274);  convolution_63 = unsqueeze_274 = None
    mul_464: "f32[8, 352, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_130, sub_69)
    sum_5: "f32[352]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3]);  mul_464 = None
    mul_465: "f32[352]" = torch.ops.aten.mul.Tensor(sum_4, 0.002551020408163265)
    unsqueeze_275: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(mul_465, 0);  mul_465 = None
    unsqueeze_276: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_466: "f32[352]" = torch.ops.aten.mul.Tensor(sum_5, 0.002551020408163265)
    mul_467: "f32[352]" = torch.ops.aten.mul.Tensor(squeeze_190, squeeze_190)
    mul_468: "f32[352]" = torch.ops.aten.mul.Tensor(mul_466, mul_467);  mul_466 = mul_467 = None
    unsqueeze_278: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_279: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_469: "f32[352]" = torch.ops.aten.mul.Tensor(squeeze_190, primals_127);  primals_127 = None
    unsqueeze_281: "f32[1, 352]" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
    unsqueeze_282: "f32[1, 352, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 352, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    mul_470: "f32[8, 352, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_280);  sub_69 = unsqueeze_280 = None
    sub_71: "f32[8, 352, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_130, mul_470);  getitem_130 = mul_470 = None
    sub_72: "f32[8, 352, 7, 7]" = torch.ops.aten.sub.Tensor(sub_71, unsqueeze_277);  sub_71 = unsqueeze_277 = None
    mul_471: "f32[8, 352, 7, 7]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_283);  sub_72 = unsqueeze_283 = None
    mul_472: "f32[352]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_190);  sum_5 = squeeze_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_471, relu_42, primals_194, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_471 = primals_194 = None
    getitem_133: "f32[8, 1104, 7, 7]" = convolution_backward_1[0]
    getitem_134: "f32[352, 1104, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_48: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_42);  relu_42 = None
    alias_49: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    le_1: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_49, 0);  alias_49 = None
    where_1: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_1, full_default, getitem_133);  le_1 = getitem_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_6: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_73: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_286);  convolution_62 = unsqueeze_286 = None
    mul_473: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_1, sub_73)
    sum_7: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_473, [0, 2, 3]);  mul_473 = None
    mul_474: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_287: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_474, 0);  mul_474 = None
    unsqueeze_288: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_475: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_476: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_187, squeeze_187)
    mul_477: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_475, mul_476);  mul_475 = mul_476 = None
    unsqueeze_290: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_291: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_478: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_187, primals_125);  primals_125 = None
    unsqueeze_293: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_478, 0);  mul_478 = None
    unsqueeze_294: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    mul_479: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_292);  sub_73 = unsqueeze_292 = None
    sub_75: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_1, mul_479);  where_1 = mul_479 = None
    sub_76: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_289);  sub_75 = unsqueeze_289 = None
    mul_480: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_295);  sub_76 = unsqueeze_295 = None
    mul_481: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_187);  sum_7 = squeeze_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_480, relu_41, primals_193, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1104, [True, True, False]);  mul_480 = primals_193 = None
    getitem_136: "f32[8, 1104, 7, 7]" = convolution_backward_2[0]
    getitem_137: "f32[1104, 1, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_51: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_41);  relu_41 = None
    alias_52: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_51);  alias_51 = None
    le_2: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_52, 0);  alias_52 = None
    where_2: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, getitem_136);  le_2 = getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_8: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_77: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_298);  convolution_61 = unsqueeze_298 = None
    mul_482: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_77)
    sum_9: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_482, [0, 2, 3]);  mul_482 = None
    mul_483: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_299: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_483, 0);  mul_483 = None
    unsqueeze_300: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_484: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_485: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_184, squeeze_184)
    mul_486: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_484, mul_485);  mul_484 = mul_485 = None
    unsqueeze_302: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
    unsqueeze_303: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_487: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_184, primals_123);  primals_123 = None
    unsqueeze_305: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_487, 0);  mul_487 = None
    unsqueeze_306: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    mul_488: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_304);  sub_77 = unsqueeze_304 = None
    sub_79: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_488);  where_2 = mul_488 = None
    sub_80: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_301);  sub_79 = unsqueeze_301 = None
    mul_489: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_307);  sub_80 = unsqueeze_307 = None
    mul_490: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_184);  sum_9 = squeeze_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_489, add_319, primals_192, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_489 = add_319 = primals_192 = None
    getitem_139: "f32[8, 184, 7, 7]" = convolution_backward_3[0]
    getitem_140: "f32[1104, 184, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_10: "f32[184]" = torch.ops.aten.sum.dim_IntList(getitem_139, [0, 2, 3])
    sub_81: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_310);  convolution_60 = unsqueeze_310 = None
    mul_491: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_139, sub_81)
    sum_11: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
    mul_492: "f32[184]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    unsqueeze_311: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_492, 0);  mul_492 = None
    unsqueeze_312: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_493: "f32[184]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    mul_494: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_181, squeeze_181)
    mul_495: "f32[184]" = torch.ops.aten.mul.Tensor(mul_493, mul_494);  mul_493 = mul_494 = None
    unsqueeze_314: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_315: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_496: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_181, primals_121);  primals_121 = None
    unsqueeze_317: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_318: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    mul_497: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_316);  sub_81 = unsqueeze_316 = None
    sub_83: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(getitem_139, mul_497);  mul_497 = None
    sub_84: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_313);  sub_83 = unsqueeze_313 = None
    mul_498: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_319);  sub_84 = unsqueeze_319 = None
    mul_499: "f32[184]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_181);  sum_11 = squeeze_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_498, relu_40, primals_191, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_498 = primals_191 = None
    getitem_142: "f32[8, 1104, 7, 7]" = convolution_backward_4[0]
    getitem_143: "f32[184, 1104, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_54: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_40);  relu_40 = None
    alias_55: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    le_3: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_55, 0);  alias_55 = None
    where_3: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, getitem_142);  le_3 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_12: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_85: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_322);  convolution_59 = unsqueeze_322 = None
    mul_500: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_85)
    sum_13: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 2, 3]);  mul_500 = None
    mul_501: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    unsqueeze_323: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_501, 0);  mul_501 = None
    unsqueeze_324: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_502: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_13, 0.002551020408163265)
    mul_503: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_178, squeeze_178)
    mul_504: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_502, mul_503);  mul_502 = mul_503 = None
    unsqueeze_326: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_327: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_505: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_178, primals_119);  primals_119 = None
    unsqueeze_329: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_505, 0);  mul_505 = None
    unsqueeze_330: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    mul_506: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_328);  sub_85 = unsqueeze_328 = None
    sub_87: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_506);  where_3 = mul_506 = None
    sub_88: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_325);  sub_87 = unsqueeze_325 = None
    mul_507: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_331);  sub_88 = unsqueeze_331 = None
    mul_508: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_178);  sum_13 = squeeze_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_507, relu_39, primals_190, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1104, [True, True, False]);  mul_507 = primals_190 = None
    getitem_145: "f32[8, 1104, 7, 7]" = convolution_backward_5[0]
    getitem_146: "f32[1104, 1, 5, 5]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_57: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_39);  relu_39 = None
    alias_58: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_57);  alias_57 = None
    le_4: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_58, 0);  alias_58 = None
    where_4: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, getitem_145);  le_4 = getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_14: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_89: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_334);  convolution_58 = unsqueeze_334 = None
    mul_509: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_89)
    sum_15: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_509, [0, 2, 3]);  mul_509 = None
    mul_510: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_14, 0.002551020408163265)
    unsqueeze_335: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_510, 0);  mul_510 = None
    unsqueeze_336: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_511: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_15, 0.002551020408163265)
    mul_512: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_175, squeeze_175)
    mul_513: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_511, mul_512);  mul_511 = mul_512 = None
    unsqueeze_338: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
    unsqueeze_339: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_514: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_175, primals_117);  primals_117 = None
    unsqueeze_341: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_514, 0);  mul_514 = None
    unsqueeze_342: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    mul_515: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_340);  sub_89 = unsqueeze_340 = None
    sub_91: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_515);  where_4 = mul_515 = None
    sub_92: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_337);  sub_91 = unsqueeze_337 = None
    mul_516: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_343);  sub_92 = unsqueeze_343 = None
    mul_517: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_175);  sum_15 = squeeze_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_516, add_303, primals_189, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_516 = add_303 = primals_189 = None
    getitem_148: "f32[8, 184, 7, 7]" = convolution_backward_6[0]
    getitem_149: "f32[1104, 184, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_340: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(getitem_139, getitem_148);  getitem_139 = getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_16: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_340, [0, 2, 3])
    sub_93: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_346);  convolution_57 = unsqueeze_346 = None
    mul_518: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_340, sub_93)
    sum_17: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 2, 3]);  mul_518 = None
    mul_519: "f32[184]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_347: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_519, 0);  mul_519 = None
    unsqueeze_348: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_520: "f32[184]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_521: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_172, squeeze_172)
    mul_522: "f32[184]" = torch.ops.aten.mul.Tensor(mul_520, mul_521);  mul_520 = mul_521 = None
    unsqueeze_350: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
    unsqueeze_351: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_523: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_172, primals_115);  primals_115 = None
    unsqueeze_353: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_523, 0);  mul_523 = None
    unsqueeze_354: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    mul_524: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_352);  sub_93 = unsqueeze_352 = None
    sub_95: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_340, mul_524);  mul_524 = None
    sub_96: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_349);  sub_95 = unsqueeze_349 = None
    mul_525: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_355);  sub_96 = unsqueeze_355 = None
    mul_526: "f32[184]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_172);  sum_17 = squeeze_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_525, relu_38, primals_188, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_525 = primals_188 = None
    getitem_151: "f32[8, 1104, 7, 7]" = convolution_backward_7[0]
    getitem_152: "f32[184, 1104, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_60: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_38);  relu_38 = None
    alias_61: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    le_5: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_61, 0);  alias_61 = None
    where_5: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_5, full_default, getitem_151);  le_5 = getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_97: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_358);  convolution_56 = unsqueeze_358 = None
    mul_527: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_5, sub_97)
    sum_19: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_527, [0, 2, 3]);  mul_527 = None
    mul_528: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_359: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
    unsqueeze_360: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_529: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_530: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_169, squeeze_169)
    mul_531: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_529, mul_530);  mul_529 = mul_530 = None
    unsqueeze_362: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
    unsqueeze_363: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_532: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_169, primals_113);  primals_113 = None
    unsqueeze_365: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_532, 0);  mul_532 = None
    unsqueeze_366: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    mul_533: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_364);  sub_97 = unsqueeze_364 = None
    sub_99: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_5, mul_533);  where_5 = mul_533 = None
    sub_100: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_361);  sub_99 = unsqueeze_361 = None
    mul_534: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_367);  sub_100 = unsqueeze_367 = None
    mul_535: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_169);  sum_19 = squeeze_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_534, relu_37, primals_187, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1104, [True, True, False]);  mul_534 = primals_187 = None
    getitem_154: "f32[8, 1104, 7, 7]" = convolution_backward_8[0]
    getitem_155: "f32[1104, 1, 5, 5]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_63: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_37);  relu_37 = None
    alias_64: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_63);  alias_63 = None
    le_6: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_64, 0);  alias_64 = None
    where_6: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, getitem_154);  le_6 = getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_20: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_101: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_370);  convolution_55 = unsqueeze_370 = None
    mul_536: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_101)
    sum_21: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 2, 3]);  mul_536 = None
    mul_537: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_371: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_537, 0);  mul_537 = None
    unsqueeze_372: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_538: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_539: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_166, squeeze_166)
    mul_540: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_538, mul_539);  mul_538 = mul_539 = None
    unsqueeze_374: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_375: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_541: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_166, primals_111);  primals_111 = None
    unsqueeze_377: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_541, 0);  mul_541 = None
    unsqueeze_378: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    mul_542: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_376);  sub_101 = unsqueeze_376 = None
    sub_103: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_542);  where_6 = mul_542 = None
    sub_104: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_373);  sub_103 = unsqueeze_373 = None
    mul_543: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_379);  sub_104 = unsqueeze_379 = None
    mul_544: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_166);  sum_21 = squeeze_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_543, add_287, primals_186, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_543 = add_287 = primals_186 = None
    getitem_157: "f32[8, 184, 7, 7]" = convolution_backward_9[0]
    getitem_158: "f32[1104, 184, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_341: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_340, getitem_157);  add_340 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_22: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_341, [0, 2, 3])
    sub_105: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_382);  convolution_54 = unsqueeze_382 = None
    mul_545: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_341, sub_105)
    sum_23: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_545, [0, 2, 3]);  mul_545 = None
    mul_546: "f32[184]" = torch.ops.aten.mul.Tensor(sum_22, 0.002551020408163265)
    unsqueeze_383: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_546, 0);  mul_546 = None
    unsqueeze_384: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_547: "f32[184]" = torch.ops.aten.mul.Tensor(sum_23, 0.002551020408163265)
    mul_548: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_163, squeeze_163)
    mul_549: "f32[184]" = torch.ops.aten.mul.Tensor(mul_547, mul_548);  mul_547 = mul_548 = None
    unsqueeze_386: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
    unsqueeze_387: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_550: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_163, primals_109);  primals_109 = None
    unsqueeze_389: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_550, 0);  mul_550 = None
    unsqueeze_390: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    mul_551: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_388);  sub_105 = unsqueeze_388 = None
    sub_107: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_341, mul_551);  mul_551 = None
    sub_108: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_385);  sub_107 = unsqueeze_385 = None
    mul_552: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_391);  sub_108 = unsqueeze_391 = None
    mul_553: "f32[184]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_163);  sum_23 = squeeze_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_552, relu_36, primals_185, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_552 = primals_185 = None
    getitem_160: "f32[8, 1104, 7, 7]" = convolution_backward_10[0]
    getitem_161: "f32[184, 1104, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_66: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_36);  relu_36 = None
    alias_67: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    le_7: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_67, 0);  alias_67 = None
    where_7: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, getitem_160);  le_7 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_24: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_109: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_394);  convolution_53 = unsqueeze_394 = None
    mul_554: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_109)
    sum_25: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_554, [0, 2, 3]);  mul_554 = None
    mul_555: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_24, 0.002551020408163265)
    unsqueeze_395: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_555, 0);  mul_555 = None
    unsqueeze_396: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_556: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    mul_557: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_160, squeeze_160)
    mul_558: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_556, mul_557);  mul_556 = mul_557 = None
    unsqueeze_398: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
    unsqueeze_399: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_559: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_160, primals_107);  primals_107 = None
    unsqueeze_401: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_559, 0);  mul_559 = None
    unsqueeze_402: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    mul_560: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_400);  sub_109 = unsqueeze_400 = None
    sub_111: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_560);  where_7 = mul_560 = None
    sub_112: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_397);  sub_111 = unsqueeze_397 = None
    mul_561: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_403);  sub_112 = unsqueeze_403 = None
    mul_562: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_160);  sum_25 = squeeze_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_561, relu_35, primals_184, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 1104, [True, True, False]);  mul_561 = primals_184 = None
    getitem_163: "f32[8, 1104, 7, 7]" = convolution_backward_11[0]
    getitem_164: "f32[1104, 1, 5, 5]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_69: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(relu_35);  relu_35 = None
    alias_70: "f32[8, 1104, 7, 7]" = torch.ops.aten.alias.default(alias_69);  alias_69 = None
    le_8: "b8[8, 1104, 7, 7]" = torch.ops.aten.le.Scalar(alias_70, 0);  alias_70 = None
    where_8: "f32[8, 1104, 7, 7]" = torch.ops.aten.where.self(le_8, full_default, getitem_163);  le_8 = getitem_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_26: "f32[1104]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_113: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_406);  convolution_52 = unsqueeze_406 = None
    mul_563: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_113)
    sum_27: "f32[1104]" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3]);  mul_563 = None
    mul_564: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    unsqueeze_407: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
    unsqueeze_408: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    unsqueeze_409: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
    mul_565: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    mul_566: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_157, squeeze_157)
    mul_567: "f32[1104]" = torch.ops.aten.mul.Tensor(mul_565, mul_566);  mul_565 = mul_566 = None
    unsqueeze_410: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_411: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_568: "f32[1104]" = torch.ops.aten.mul.Tensor(squeeze_157, primals_105);  primals_105 = None
    unsqueeze_413: "f32[1, 1104]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_414: "f32[1, 1104, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 1104, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    mul_569: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_412);  sub_113 = unsqueeze_412 = None
    sub_115: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_569);  where_8 = mul_569 = None
    sub_116: "f32[8, 1104, 7, 7]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_409);  sub_115 = unsqueeze_409 = None
    mul_570: "f32[8, 1104, 7, 7]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_415);  sub_116 = unsqueeze_415 = None
    mul_571: "f32[1104]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_157);  sum_27 = squeeze_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_570, add_271, primals_183, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_570 = add_271 = primals_183 = None
    getitem_166: "f32[8, 184, 7, 7]" = convolution_backward_12[0]
    getitem_167: "f32[1104, 184, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_342: "f32[8, 184, 7, 7]" = torch.ops.aten.add.Tensor(add_341, getitem_166);  add_341 = getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_28: "f32[184]" = torch.ops.aten.sum.dim_IntList(add_342, [0, 2, 3])
    sub_117: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_418);  convolution_51 = unsqueeze_418 = None
    mul_572: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(add_342, sub_117)
    sum_29: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_572, [0, 2, 3]);  mul_572 = None
    mul_573: "f32[184]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    unsqueeze_419: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
    unsqueeze_420: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_574: "f32[184]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    mul_575: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_154, squeeze_154)
    mul_576: "f32[184]" = torch.ops.aten.mul.Tensor(mul_574, mul_575);  mul_574 = mul_575 = None
    unsqueeze_422: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_423: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_577: "f32[184]" = torch.ops.aten.mul.Tensor(squeeze_154, primals_103);  primals_103 = None
    unsqueeze_425: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_577, 0);  mul_577 = None
    unsqueeze_426: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_578: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_424);  sub_117 = unsqueeze_424 = None
    sub_119: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(add_342, mul_578);  add_342 = mul_578 = None
    sub_120: "f32[8, 184, 7, 7]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_421);  sub_119 = unsqueeze_421 = None
    mul_579: "f32[8, 184, 7, 7]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_427);  sub_120 = unsqueeze_427 = None
    mul_580: "f32[184]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_154);  sum_29 = squeeze_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_579, relu_34, primals_182, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_579 = primals_182 = None
    getitem_169: "f32[8, 672, 7, 7]" = convolution_backward_13[0]
    getitem_170: "f32[184, 672, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_72: "f32[8, 672, 7, 7]" = torch.ops.aten.alias.default(relu_34);  relu_34 = None
    alias_73: "f32[8, 672, 7, 7]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    le_9: "b8[8, 672, 7, 7]" = torch.ops.aten.le.Scalar(alias_73, 0);  alias_73 = None
    where_9: "f32[8, 672, 7, 7]" = torch.ops.aten.where.self(le_9, full_default, getitem_169);  le_9 = getitem_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_30: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_121: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_430);  convolution_50 = unsqueeze_430 = None
    mul_581: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_121)
    sum_31: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 2, 3]);  mul_581 = None
    mul_582: "f32[672]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    unsqueeze_431: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
    unsqueeze_432: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_583: "f32[672]" = torch.ops.aten.mul.Tensor(sum_31, 0.002551020408163265)
    mul_584: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_151, squeeze_151)
    mul_585: "f32[672]" = torch.ops.aten.mul.Tensor(mul_583, mul_584);  mul_583 = mul_584 = None
    unsqueeze_434: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_435: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_586: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_151, primals_101);  primals_101 = None
    unsqueeze_437: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_586, 0);  mul_586 = None
    unsqueeze_438: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_587: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_436);  sub_121 = unsqueeze_436 = None
    sub_123: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(where_9, mul_587);  where_9 = mul_587 = None
    sub_124: "f32[8, 672, 7, 7]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_433);  sub_123 = unsqueeze_433 = None
    mul_588: "f32[8, 672, 7, 7]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_439);  sub_124 = unsqueeze_439 = None
    mul_589: "f32[672]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_151);  sum_31 = squeeze_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_588, relu_33, primals_181, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_588 = primals_181 = None
    getitem_172: "f32[8, 672, 14, 14]" = convolution_backward_14[0]
    getitem_173: "f32[672, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_75: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(relu_33);  relu_33 = None
    alias_76: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(alias_75);  alias_75 = None
    le_10: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(alias_76, 0);  alias_76 = None
    where_10: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, getitem_172);  le_10 = getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_32: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_125: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_442);  convolution_49 = unsqueeze_442 = None
    mul_590: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_125)
    sum_33: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_590, [0, 2, 3]);  mul_590 = None
    mul_591: "f32[672]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_443: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_591, 0);  mul_591 = None
    unsqueeze_444: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_592: "f32[672]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_593: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_148, squeeze_148)
    mul_594: "f32[672]" = torch.ops.aten.mul.Tensor(mul_592, mul_593);  mul_592 = mul_593 = None
    unsqueeze_446: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
    unsqueeze_447: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_595: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_148, primals_99);  primals_99 = None
    unsqueeze_449: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
    unsqueeze_450: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_596: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_448);  sub_125 = unsqueeze_448 = None
    sub_127: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_596);  where_10 = mul_596 = None
    sub_128: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_445);  sub_127 = unsqueeze_445 = None
    mul_597: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_451);  sub_128 = unsqueeze_451 = None
    mul_598: "f32[672]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_148);  sum_33 = squeeze_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_597, add_256, primals_180, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_597 = add_256 = primals_180 = None
    getitem_175: "f32[8, 112, 14, 14]" = convolution_backward_15[0]
    getitem_176: "f32[672, 112, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_34: "f32[112]" = torch.ops.aten.sum.dim_IntList(getitem_175, [0, 2, 3])
    sub_129: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_454);  convolution_48 = unsqueeze_454 = None
    mul_599: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_175, sub_129)
    sum_35: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_599, [0, 2, 3]);  mul_599 = None
    mul_600: "f32[112]" = torch.ops.aten.mul.Tensor(sum_34, 0.0006377551020408163)
    unsqueeze_455: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_600, 0);  mul_600 = None
    unsqueeze_456: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_601: "f32[112]" = torch.ops.aten.mul.Tensor(sum_35, 0.0006377551020408163)
    mul_602: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_145, squeeze_145)
    mul_603: "f32[112]" = torch.ops.aten.mul.Tensor(mul_601, mul_602);  mul_601 = mul_602 = None
    unsqueeze_458: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
    unsqueeze_459: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_604: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_145, primals_97);  primals_97 = None
    unsqueeze_461: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    unsqueeze_462: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_605: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_460);  sub_129 = unsqueeze_460 = None
    sub_131: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_175, mul_605);  mul_605 = None
    sub_132: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_457);  sub_131 = unsqueeze_457 = None
    mul_606: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_463);  sub_132 = unsqueeze_463 = None
    mul_607: "f32[112]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_145);  sum_35 = squeeze_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_606, relu_32, primals_179, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_606 = primals_179 = None
    getitem_178: "f32[8, 336, 14, 14]" = convolution_backward_16[0]
    getitem_179: "f32[112, 336, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_78: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(relu_32);  relu_32 = None
    alias_79: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(alias_78);  alias_78 = None
    le_11: "b8[8, 336, 14, 14]" = torch.ops.aten.le.Scalar(alias_79, 0);  alias_79 = None
    where_11: "f32[8, 336, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, getitem_178);  le_11 = getitem_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_36: "f32[336]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_133: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_466);  convolution_47 = unsqueeze_466 = None
    mul_608: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_133)
    sum_37: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 2, 3]);  mul_608 = None
    mul_609: "f32[336]" = torch.ops.aten.mul.Tensor(sum_36, 0.0006377551020408163)
    unsqueeze_467: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_609, 0);  mul_609 = None
    unsqueeze_468: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_610: "f32[336]" = torch.ops.aten.mul.Tensor(sum_37, 0.0006377551020408163)
    mul_611: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_142, squeeze_142)
    mul_612: "f32[336]" = torch.ops.aten.mul.Tensor(mul_610, mul_611);  mul_610 = mul_611 = None
    unsqueeze_470: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_471: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_613: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_142, primals_95);  primals_95 = None
    unsqueeze_473: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    unsqueeze_474: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_614: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_472);  sub_133 = unsqueeze_472 = None
    sub_135: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_614);  where_11 = mul_614 = None
    sub_136: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_469);  sub_135 = unsqueeze_469 = None
    mul_615: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_475);  sub_136 = unsqueeze_475 = None
    mul_616: "f32[336]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_142);  sum_37 = squeeze_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_615, relu_31, primals_178, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 336, [True, True, False]);  mul_615 = primals_178 = None
    getitem_181: "f32[8, 336, 14, 14]" = convolution_backward_17[0]
    getitem_182: "f32[336, 1, 5, 5]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_81: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(relu_31);  relu_31 = None
    alias_82: "f32[8, 336, 14, 14]" = torch.ops.aten.alias.default(alias_81);  alias_81 = None
    le_12: "b8[8, 336, 14, 14]" = torch.ops.aten.le.Scalar(alias_82, 0);  alias_82 = None
    where_12: "f32[8, 336, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, getitem_181);  le_12 = getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_38: "f32[336]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_137: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_478);  convolution_46 = unsqueeze_478 = None
    mul_617: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_137)
    sum_39: "f32[336]" = torch.ops.aten.sum.dim_IntList(mul_617, [0, 2, 3]);  mul_617 = None
    mul_618: "f32[336]" = torch.ops.aten.mul.Tensor(sum_38, 0.0006377551020408163)
    unsqueeze_479: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_618, 0);  mul_618 = None
    unsqueeze_480: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_619: "f32[336]" = torch.ops.aten.mul.Tensor(sum_39, 0.0006377551020408163)
    mul_620: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_139, squeeze_139)
    mul_621: "f32[336]" = torch.ops.aten.mul.Tensor(mul_619, mul_620);  mul_619 = mul_620 = None
    unsqueeze_482: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_483: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_622: "f32[336]" = torch.ops.aten.mul.Tensor(squeeze_139, primals_93);  primals_93 = None
    unsqueeze_485: "f32[1, 336]" = torch.ops.aten.unsqueeze.default(mul_622, 0);  mul_622 = None
    unsqueeze_486: "f32[1, 336, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 336, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_623: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_484);  sub_137 = unsqueeze_484 = None
    sub_139: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_623);  where_12 = mul_623 = None
    sub_140: "f32[8, 336, 14, 14]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_481);  sub_139 = unsqueeze_481 = None
    mul_624: "f32[8, 336, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_487);  sub_140 = unsqueeze_487 = None
    mul_625: "f32[336]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_139);  sum_39 = squeeze_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_624, add_240, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_624 = add_240 = primals_177 = None
    getitem_184: "f32[8, 112, 14, 14]" = convolution_backward_18[0]
    getitem_185: "f32[336, 112, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_343: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(getitem_175, getitem_184);  getitem_175 = getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_40: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_343, [0, 2, 3])
    sub_141: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_490);  convolution_45 = unsqueeze_490 = None
    mul_626: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_343, sub_141)
    sum_41: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_626, [0, 2, 3]);  mul_626 = None
    mul_627: "f32[112]" = torch.ops.aten.mul.Tensor(sum_40, 0.0006377551020408163)
    unsqueeze_491: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
    unsqueeze_492: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_628: "f32[112]" = torch.ops.aten.mul.Tensor(sum_41, 0.0006377551020408163)
    mul_629: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_136, squeeze_136)
    mul_630: "f32[112]" = torch.ops.aten.mul.Tensor(mul_628, mul_629);  mul_628 = mul_629 = None
    unsqueeze_494: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    unsqueeze_495: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_631: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_136, primals_91);  primals_91 = None
    unsqueeze_497: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_631, 0);  mul_631 = None
    unsqueeze_498: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_632: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_496);  sub_141 = unsqueeze_496 = None
    sub_143: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(add_343, mul_632);  mul_632 = None
    sub_144: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_493);  sub_143 = unsqueeze_493 = None
    mul_633: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_499);  sub_144 = unsqueeze_499 = None
    mul_634: "f32[112]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_136);  sum_41 = squeeze_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_633, relu_30, primals_176, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_633 = primals_176 = None
    getitem_187: "f32[8, 672, 14, 14]" = convolution_backward_19[0]
    getitem_188: "f32[112, 672, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_84: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(relu_30);  relu_30 = None
    alias_85: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(alias_84);  alias_84 = None
    le_13: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(alias_85, 0);  alias_85 = None
    where_13: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, getitem_187);  le_13 = getitem_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_42: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_145: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_502);  convolution_44 = unsqueeze_502 = None
    mul_635: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_145)
    sum_43: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_635, [0, 2, 3]);  mul_635 = None
    mul_636: "f32[672]" = torch.ops.aten.mul.Tensor(sum_42, 0.0006377551020408163)
    unsqueeze_503: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
    unsqueeze_504: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_637: "f32[672]" = torch.ops.aten.mul.Tensor(sum_43, 0.0006377551020408163)
    mul_638: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_133, squeeze_133)
    mul_639: "f32[672]" = torch.ops.aten.mul.Tensor(mul_637, mul_638);  mul_637 = mul_638 = None
    unsqueeze_506: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    unsqueeze_507: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_640: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_133, primals_89);  primals_89 = None
    unsqueeze_509: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_640, 0);  mul_640 = None
    unsqueeze_510: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_641: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_508);  sub_145 = unsqueeze_508 = None
    sub_147: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_641);  where_13 = mul_641 = None
    sub_148: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_505);  sub_147 = unsqueeze_505 = None
    mul_642: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_511);  sub_148 = unsqueeze_511 = None
    mul_643: "f32[672]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_133);  sum_43 = squeeze_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_642, relu_29, primals_175, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_642 = primals_175 = None
    getitem_190: "f32[8, 672, 14, 14]" = convolution_backward_20[0]
    getitem_191: "f32[672, 1, 5, 5]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_87: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(relu_29);  relu_29 = None
    alias_88: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(alias_87);  alias_87 = None
    le_14: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(alias_88, 0);  alias_88 = None
    where_14: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, getitem_190);  le_14 = getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_44: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_149: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_514);  convolution_43 = unsqueeze_514 = None
    mul_644: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_149)
    sum_45: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_644, [0, 2, 3]);  mul_644 = None
    mul_645: "f32[672]" = torch.ops.aten.mul.Tensor(sum_44, 0.0006377551020408163)
    unsqueeze_515: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
    unsqueeze_516: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_646: "f32[672]" = torch.ops.aten.mul.Tensor(sum_45, 0.0006377551020408163)
    mul_647: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_648: "f32[672]" = torch.ops.aten.mul.Tensor(mul_646, mul_647);  mul_646 = mul_647 = None
    unsqueeze_518: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_519: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_649: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_521: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_649, 0);  mul_649 = None
    unsqueeze_522: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_650: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_520);  sub_149 = unsqueeze_520 = None
    sub_151: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_650);  where_14 = mul_650 = None
    sub_152: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_517);  sub_151 = unsqueeze_517 = None
    mul_651: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_523);  sub_152 = unsqueeze_523 = None
    mul_652: "f32[672]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_130);  sum_45 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_651, add_224, primals_174, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_651 = add_224 = primals_174 = None
    getitem_193: "f32[8, 112, 14, 14]" = convolution_backward_21[0]
    getitem_194: "f32[672, 112, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_344: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_343, getitem_193);  add_343 = getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_46: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_344, [0, 2, 3])
    sub_153: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_526);  convolution_42 = unsqueeze_526 = None
    mul_653: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_344, sub_153)
    sum_47: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_653, [0, 2, 3]);  mul_653 = None
    mul_654: "f32[112]" = torch.ops.aten.mul.Tensor(sum_46, 0.0006377551020408163)
    unsqueeze_527: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
    unsqueeze_528: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_655: "f32[112]" = torch.ops.aten.mul.Tensor(sum_47, 0.0006377551020408163)
    mul_656: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_657: "f32[112]" = torch.ops.aten.mul.Tensor(mul_655, mul_656);  mul_655 = mul_656 = None
    unsqueeze_530: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_657, 0);  mul_657 = None
    unsqueeze_531: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_658: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_533: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_658, 0);  mul_658 = None
    unsqueeze_534: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_659: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_532);  sub_153 = unsqueeze_532 = None
    sub_155: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(add_344, mul_659);  mul_659 = None
    sub_156: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_529);  sub_155 = unsqueeze_529 = None
    mul_660: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_535);  sub_156 = unsqueeze_535 = None
    mul_661: "f32[112]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_127);  sum_47 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_660, relu_28, primals_173, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_660 = primals_173 = None
    getitem_196: "f32[8, 672, 14, 14]" = convolution_backward_22[0]
    getitem_197: "f32[112, 672, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_90: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(relu_28);  relu_28 = None
    alias_91: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(alias_90);  alias_90 = None
    le_15: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(alias_91, 0);  alias_91 = None
    where_15: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, getitem_196);  le_15 = getitem_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_48: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_157: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_538);  convolution_41 = unsqueeze_538 = None
    mul_662: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_157)
    sum_49: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 2, 3]);  mul_662 = None
    mul_663: "f32[672]" = torch.ops.aten.mul.Tensor(sum_48, 0.0006377551020408163)
    unsqueeze_539: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
    unsqueeze_540: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
    unsqueeze_541: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
    mul_664: "f32[672]" = torch.ops.aten.mul.Tensor(sum_49, 0.0006377551020408163)
    mul_665: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_666: "f32[672]" = torch.ops.aten.mul.Tensor(mul_664, mul_665);  mul_664 = mul_665 = None
    unsqueeze_542: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
    unsqueeze_543: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_667: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_545: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
    unsqueeze_546: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_668: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_544);  sub_157 = unsqueeze_544 = None
    sub_159: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_668);  where_15 = mul_668 = None
    sub_160: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_541);  sub_159 = unsqueeze_541 = None
    mul_669: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_547);  sub_160 = unsqueeze_547 = None
    mul_670: "f32[672]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_124);  sum_49 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_669, relu_27, primals_172, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_669 = primals_172 = None
    getitem_199: "f32[8, 672, 14, 14]" = convolution_backward_23[0]
    getitem_200: "f32[672, 1, 5, 5]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_93: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(relu_27);  relu_27 = None
    alias_94: "f32[8, 672, 14, 14]" = torch.ops.aten.alias.default(alias_93);  alias_93 = None
    le_16: "b8[8, 672, 14, 14]" = torch.ops.aten.le.Scalar(alias_94, 0);  alias_94 = None
    where_16: "f32[8, 672, 14, 14]" = torch.ops.aten.where.self(le_16, full_default, getitem_199);  le_16 = getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_50: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_161: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_550);  convolution_40 = unsqueeze_550 = None
    mul_671: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_16, sub_161)
    sum_51: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 2, 3]);  mul_671 = None
    mul_672: "f32[672]" = torch.ops.aten.mul.Tensor(sum_50, 0.0006377551020408163)
    unsqueeze_551: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_552: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_673: "f32[672]" = torch.ops.aten.mul.Tensor(sum_51, 0.0006377551020408163)
    mul_674: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_675: "f32[672]" = torch.ops.aten.mul.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
    unsqueeze_554: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
    unsqueeze_555: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_676: "f32[672]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_557: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
    unsqueeze_558: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_677: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_556);  sub_161 = unsqueeze_556 = None
    sub_163: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(where_16, mul_677);  where_16 = mul_677 = None
    sub_164: "f32[8, 672, 14, 14]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_553);  sub_163 = unsqueeze_553 = None
    mul_678: "f32[8, 672, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_559);  sub_164 = unsqueeze_559 = None
    mul_679: "f32[672]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_121);  sum_51 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_678, add_208, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_678 = add_208 = primals_171 = None
    getitem_202: "f32[8, 112, 14, 14]" = convolution_backward_24[0]
    getitem_203: "f32[672, 112, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_345: "f32[8, 112, 14, 14]" = torch.ops.aten.add.Tensor(add_344, getitem_202);  add_344 = getitem_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_52: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_345, [0, 2, 3])
    sub_165: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_562);  convolution_39 = unsqueeze_562 = None
    mul_680: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_345, sub_165)
    sum_53: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2, 3]);  mul_680 = None
    mul_681: "f32[112]" = torch.ops.aten.mul.Tensor(sum_52, 0.0006377551020408163)
    unsqueeze_563: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_564: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
    unsqueeze_565: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
    mul_682: "f32[112]" = torch.ops.aten.mul.Tensor(sum_53, 0.0006377551020408163)
    mul_683: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_684: "f32[112]" = torch.ops.aten.mul.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
    unsqueeze_566: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_567: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_685: "f32[112]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_569: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_570: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_686: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_568);  sub_165 = unsqueeze_568 = None
    sub_167: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(add_345, mul_686);  add_345 = mul_686 = None
    sub_168: "f32[8, 112, 14, 14]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_565);  sub_167 = unsqueeze_565 = None
    mul_687: "f32[8, 112, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_571);  sub_168 = unsqueeze_571 = None
    mul_688: "f32[112]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_118);  sum_53 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_687, relu_26, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_687 = primals_170 = None
    getitem_205: "f32[8, 384, 14, 14]" = convolution_backward_25[0]
    getitem_206: "f32[112, 384, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_96: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_26);  relu_26 = None
    alias_97: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_96);  alias_96 = None
    le_17: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_97, 0);  alias_97 = None
    where_17: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_17, full_default, getitem_205);  le_17 = getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_169: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_574);  convolution_38 = unsqueeze_574 = None
    mul_689: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_17, sub_169)
    sum_55: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3]);  mul_689 = None
    mul_690: "f32[384]" = torch.ops.aten.mul.Tensor(sum_54, 0.0006377551020408163)
    unsqueeze_575: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_690, 0);  mul_690 = None
    unsqueeze_576: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_691: "f32[384]" = torch.ops.aten.mul.Tensor(sum_55, 0.0006377551020408163)
    mul_692: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_693: "f32[384]" = torch.ops.aten.mul.Tensor(mul_691, mul_692);  mul_691 = mul_692 = None
    unsqueeze_578: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
    unsqueeze_579: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_694: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_581: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_582: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_695: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_580);  sub_169 = unsqueeze_580 = None
    sub_171: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_17, mul_695);  where_17 = mul_695 = None
    sub_172: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_577);  sub_171 = unsqueeze_577 = None
    mul_696: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_583);  sub_172 = unsqueeze_583 = None
    mul_697: "f32[384]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_115);  sum_55 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_696, relu_25, primals_169, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_696 = primals_169 = None
    getitem_208: "f32[8, 384, 14, 14]" = convolution_backward_26[0]
    getitem_209: "f32[384, 1, 5, 5]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_99: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_25);  relu_25 = None
    alias_100: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_99);  alias_99 = None
    le_18: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_100, 0);  alias_100 = None
    where_18: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_18, full_default, getitem_208);  le_18 = getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_173: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_586);  convolution_37 = unsqueeze_586 = None
    mul_698: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_18, sub_173)
    sum_57: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_698, [0, 2, 3]);  mul_698 = None
    mul_699: "f32[384]" = torch.ops.aten.mul.Tensor(sum_56, 0.0006377551020408163)
    unsqueeze_587: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_699, 0);  mul_699 = None
    unsqueeze_588: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_700: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, 0.0006377551020408163)
    mul_701: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_702: "f32[384]" = torch.ops.aten.mul.Tensor(mul_700, mul_701);  mul_700 = mul_701 = None
    unsqueeze_590: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_702, 0);  mul_702 = None
    unsqueeze_591: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_703: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_593: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_703, 0);  mul_703 = None
    unsqueeze_594: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_704: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_592);  sub_173 = unsqueeze_592 = None
    sub_175: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_18, mul_704);  where_18 = mul_704 = None
    sub_176: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_589);  sub_175 = unsqueeze_589 = None
    mul_705: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_595);  sub_176 = unsqueeze_595 = None
    mul_706: "f32[384]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_112);  sum_57 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_705, add_193, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_705 = add_193 = primals_168 = None
    getitem_211: "f32[8, 64, 14, 14]" = convolution_backward_27[0]
    getitem_212: "f32[384, 64, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_58: "f32[64]" = torch.ops.aten.sum.dim_IntList(getitem_211, [0, 2, 3])
    sub_177: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_598);  convolution_36 = unsqueeze_598 = None
    mul_707: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_211, sub_177)
    sum_59: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_707, [0, 2, 3]);  mul_707 = None
    mul_708: "f32[64]" = torch.ops.aten.mul.Tensor(sum_58, 0.0006377551020408163)
    unsqueeze_599: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_708, 0);  mul_708 = None
    unsqueeze_600: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_709: "f32[64]" = torch.ops.aten.mul.Tensor(sum_59, 0.0006377551020408163)
    mul_710: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_711: "f32[64]" = torch.ops.aten.mul.Tensor(mul_709, mul_710);  mul_709 = mul_710 = None
    unsqueeze_602: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_711, 0);  mul_711 = None
    unsqueeze_603: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_712: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_605: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    unsqueeze_606: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_713: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_604);  sub_177 = unsqueeze_604 = None
    sub_179: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(getitem_211, mul_713);  mul_713 = None
    sub_180: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_601);  sub_179 = unsqueeze_601 = None
    mul_714: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_607);  sub_180 = unsqueeze_607 = None
    mul_715: "f32[64]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_109);  sum_59 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_714, relu_24, primals_167, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_714 = primals_167 = None
    getitem_214: "f32[8, 384, 14, 14]" = convolution_backward_28[0]
    getitem_215: "f32[64, 384, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_102: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_24);  relu_24 = None
    alias_103: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_102);  alias_102 = None
    le_19: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_103, 0);  alias_103 = None
    where_19: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_19, full_default, getitem_214);  le_19 = getitem_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_60: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_181: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_610);  convolution_35 = unsqueeze_610 = None
    mul_716: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_19, sub_181)
    sum_61: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_716, [0, 2, 3]);  mul_716 = None
    mul_717: "f32[384]" = torch.ops.aten.mul.Tensor(sum_60, 0.0006377551020408163)
    unsqueeze_611: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_717, 0);  mul_717 = None
    unsqueeze_612: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
    unsqueeze_613: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
    mul_718: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, 0.0006377551020408163)
    mul_719: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_720: "f32[384]" = torch.ops.aten.mul.Tensor(mul_718, mul_719);  mul_718 = mul_719 = None
    unsqueeze_614: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_720, 0);  mul_720 = None
    unsqueeze_615: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
    unsqueeze_616: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
    mul_721: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_617: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    unsqueeze_618: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_722: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_616);  sub_181 = unsqueeze_616 = None
    sub_183: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_19, mul_722);  where_19 = mul_722 = None
    sub_184: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_613);  sub_183 = unsqueeze_613 = None
    mul_723: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_619);  sub_184 = unsqueeze_619 = None
    mul_724: "f32[384]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_106);  sum_61 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_723, relu_23, primals_166, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_723 = primals_166 = None
    getitem_217: "f32[8, 384, 14, 14]" = convolution_backward_29[0]
    getitem_218: "f32[384, 1, 5, 5]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_105: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_23);  relu_23 = None
    alias_106: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_105);  alias_105 = None
    le_20: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_106, 0);  alias_106 = None
    where_20: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_20, full_default, getitem_217);  le_20 = getitem_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_62: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_185: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_622);  convolution_34 = unsqueeze_622 = None
    mul_725: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_20, sub_185)
    sum_63: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_725, [0, 2, 3]);  mul_725 = None
    mul_726: "f32[384]" = torch.ops.aten.mul.Tensor(sum_62, 0.0006377551020408163)
    unsqueeze_623: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_726, 0);  mul_726 = None
    unsqueeze_624: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
    unsqueeze_625: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
    mul_727: "f32[384]" = torch.ops.aten.mul.Tensor(sum_63, 0.0006377551020408163)
    mul_728: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_729: "f32[384]" = torch.ops.aten.mul.Tensor(mul_727, mul_728);  mul_727 = mul_728 = None
    unsqueeze_626: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_729, 0);  mul_729 = None
    unsqueeze_627: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
    unsqueeze_628: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
    mul_730: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_629: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_730, 0);  mul_730 = None
    unsqueeze_630: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_731: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_628);  sub_185 = unsqueeze_628 = None
    sub_187: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_20, mul_731);  where_20 = mul_731 = None
    sub_188: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_625);  sub_187 = unsqueeze_625 = None
    mul_732: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_631);  sub_188 = unsqueeze_631 = None
    mul_733: "f32[384]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_103);  sum_63 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_732, add_177, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_732 = add_177 = primals_165 = None
    getitem_220: "f32[8, 64, 14, 14]" = convolution_backward_30[0]
    getitem_221: "f32[384, 64, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_346: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(getitem_211, getitem_220);  getitem_211 = getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_64: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_346, [0, 2, 3])
    sub_189: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_634);  convolution_33 = unsqueeze_634 = None
    mul_734: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_346, sub_189)
    sum_65: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_734, [0, 2, 3]);  mul_734 = None
    mul_735: "f32[64]" = torch.ops.aten.mul.Tensor(sum_64, 0.0006377551020408163)
    unsqueeze_635: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_735, 0);  mul_735 = None
    unsqueeze_636: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_736: "f32[64]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    mul_737: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_738: "f32[64]" = torch.ops.aten.mul.Tensor(mul_736, mul_737);  mul_736 = mul_737 = None
    unsqueeze_638: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_738, 0);  mul_738 = None
    unsqueeze_639: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_739: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_641: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_739, 0);  mul_739 = None
    unsqueeze_642: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_740: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_640);  sub_189 = unsqueeze_640 = None
    sub_191: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(add_346, mul_740);  mul_740 = None
    sub_192: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_637);  sub_191 = unsqueeze_637 = None
    mul_741: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_643);  sub_192 = unsqueeze_643 = None
    mul_742: "f32[64]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_100);  sum_65 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_741, relu_22, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_741 = primals_164 = None
    getitem_223: "f32[8, 384, 14, 14]" = convolution_backward_31[0]
    getitem_224: "f32[64, 384, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_108: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_22);  relu_22 = None
    alias_109: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_108);  alias_108 = None
    le_21: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_109, 0);  alias_109 = None
    where_21: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_21, full_default, getitem_223);  le_21 = getitem_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_66: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_193: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_646);  convolution_32 = unsqueeze_646 = None
    mul_743: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_193)
    sum_67: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_743, [0, 2, 3]);  mul_743 = None
    mul_744: "f32[384]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    unsqueeze_647: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_744, 0);  mul_744 = None
    unsqueeze_648: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_745: "f32[384]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    mul_746: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_747: "f32[384]" = torch.ops.aten.mul.Tensor(mul_745, mul_746);  mul_745 = mul_746 = None
    unsqueeze_650: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
    unsqueeze_651: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_748: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_653: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_748, 0);  mul_748 = None
    unsqueeze_654: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_749: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_652);  sub_193 = unsqueeze_652 = None
    sub_195: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_21, mul_749);  where_21 = mul_749 = None
    sub_196: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_649);  sub_195 = unsqueeze_649 = None
    mul_750: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_655);  sub_196 = unsqueeze_655 = None
    mul_751: "f32[384]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_97);  sum_67 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_750, relu_21, primals_163, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 384, [True, True, False]);  mul_750 = primals_163 = None
    getitem_226: "f32[8, 384, 14, 14]" = convolution_backward_32[0]
    getitem_227: "f32[384, 1, 5, 5]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_111: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(relu_21);  relu_21 = None
    alias_112: "f32[8, 384, 14, 14]" = torch.ops.aten.alias.default(alias_111);  alias_111 = None
    le_22: "b8[8, 384, 14, 14]" = torch.ops.aten.le.Scalar(alias_112, 0);  alias_112 = None
    where_22: "f32[8, 384, 14, 14]" = torch.ops.aten.where.self(le_22, full_default, getitem_226);  le_22 = getitem_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_68: "f32[384]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_197: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_658);  convolution_31 = unsqueeze_658 = None
    mul_752: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(where_22, sub_197)
    sum_69: "f32[384]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 2, 3]);  mul_752 = None
    mul_753: "f32[384]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    unsqueeze_659: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_753, 0);  mul_753 = None
    unsqueeze_660: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_659, 2);  unsqueeze_659 = None
    unsqueeze_661: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, 3);  unsqueeze_660 = None
    mul_754: "f32[384]" = torch.ops.aten.mul.Tensor(sum_69, 0.0006377551020408163)
    mul_755: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_756: "f32[384]" = torch.ops.aten.mul.Tensor(mul_754, mul_755);  mul_754 = mul_755 = None
    unsqueeze_662: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_663: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
    unsqueeze_664: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
    mul_757: "f32[384]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_665: "f32[1, 384]" = torch.ops.aten.unsqueeze.default(mul_757, 0);  mul_757 = None
    unsqueeze_666: "f32[1, 384, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 384, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_758: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_664);  sub_197 = unsqueeze_664 = None
    sub_199: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(where_22, mul_758);  where_22 = mul_758 = None
    sub_200: "f32[8, 384, 14, 14]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_661);  sub_199 = unsqueeze_661 = None
    mul_759: "f32[8, 384, 14, 14]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_667);  sub_200 = unsqueeze_667 = None
    mul_760: "f32[384]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_94);  sum_69 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_759, add_161, primals_162, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_759 = add_161 = primals_162 = None
    getitem_229: "f32[8, 64, 14, 14]" = convolution_backward_33[0]
    getitem_230: "f32[384, 64, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_347: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_346, getitem_229);  add_346 = getitem_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_70: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_347, [0, 2, 3])
    sub_201: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_670);  convolution_30 = unsqueeze_670 = None
    mul_761: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_347, sub_201)
    sum_71: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_761, [0, 2, 3]);  mul_761 = None
    mul_762: "f32[64]" = torch.ops.aten.mul.Tensor(sum_70, 0.0006377551020408163)
    unsqueeze_671: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_762, 0);  mul_762 = None
    unsqueeze_672: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
    unsqueeze_673: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
    mul_763: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, 0.0006377551020408163)
    mul_764: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_765: "f32[64]" = torch.ops.aten.mul.Tensor(mul_763, mul_764);  mul_763 = mul_764 = None
    unsqueeze_674: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
    unsqueeze_675: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
    unsqueeze_676: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
    mul_766: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_677: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_766, 0);  mul_766 = None
    unsqueeze_678: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_767: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_676);  sub_201 = unsqueeze_676 = None
    sub_203: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(add_347, mul_767);  mul_767 = None
    sub_204: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_673);  sub_203 = unsqueeze_673 = None
    mul_768: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_679);  sub_204 = unsqueeze_679 = None
    mul_769: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_91);  sum_71 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_768, relu_20, primals_161, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_768 = primals_161 = None
    getitem_232: "f32[8, 192, 14, 14]" = convolution_backward_34[0]
    getitem_233: "f32[64, 192, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_114: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(relu_20);  relu_20 = None
    alias_115: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(alias_114);  alias_114 = None
    le_23: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_115, 0);  alias_115 = None
    where_23: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_23, full_default, getitem_232);  le_23 = getitem_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_205: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_682);  convolution_29 = unsqueeze_682 = None
    mul_770: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_23, sub_205)
    sum_73: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_770, [0, 2, 3]);  mul_770 = None
    mul_771: "f32[192]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_683: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_771, 0);  mul_771 = None
    unsqueeze_684: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
    unsqueeze_685: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
    mul_772: "f32[192]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_773: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_774: "f32[192]" = torch.ops.aten.mul.Tensor(mul_772, mul_773);  mul_772 = mul_773 = None
    unsqueeze_686: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_774, 0);  mul_774 = None
    unsqueeze_687: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
    unsqueeze_688: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
    mul_775: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_689: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_775, 0);  mul_775 = None
    unsqueeze_690: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_776: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_205, unsqueeze_688);  sub_205 = unsqueeze_688 = None
    sub_207: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_23, mul_776);  where_23 = mul_776 = None
    sub_208: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_207, unsqueeze_685);  sub_207 = unsqueeze_685 = None
    mul_777: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_691);  sub_208 = unsqueeze_691 = None
    mul_778: "f32[192]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_88);  sum_73 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_777, relu_19, primals_160, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 192, [True, True, False]);  mul_777 = primals_160 = None
    getitem_235: "f32[8, 192, 14, 14]" = convolution_backward_35[0]
    getitem_236: "f32[192, 1, 5, 5]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_117: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(relu_19);  relu_19 = None
    alias_118: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(alias_117);  alias_117 = None
    le_24: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_118, 0);  alias_118 = None
    where_24: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_24, full_default, getitem_235);  le_24 = getitem_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_209: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_694);  convolution_28 = unsqueeze_694 = None
    mul_779: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_24, sub_209)
    sum_75: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_779, [0, 2, 3]);  mul_779 = None
    mul_780: "f32[192]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_695: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_780, 0);  mul_780 = None
    unsqueeze_696: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
    unsqueeze_697: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
    mul_781: "f32[192]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_782: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_783: "f32[192]" = torch.ops.aten.mul.Tensor(mul_781, mul_782);  mul_781 = mul_782 = None
    unsqueeze_698: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_783, 0);  mul_783 = None
    unsqueeze_699: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
    unsqueeze_700: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
    mul_784: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_701: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_784, 0);  mul_784 = None
    unsqueeze_702: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_785: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_209, unsqueeze_700);  sub_209 = unsqueeze_700 = None
    sub_211: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_24, mul_785);  where_24 = mul_785 = None
    sub_212: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_211, unsqueeze_697);  sub_211 = unsqueeze_697 = None
    mul_786: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_703);  sub_212 = unsqueeze_703 = None
    mul_787: "f32[192]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_85);  sum_75 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_786, add_145, primals_159, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_786 = add_145 = primals_159 = None
    getitem_238: "f32[8, 64, 14, 14]" = convolution_backward_36[0]
    getitem_239: "f32[192, 64, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_348: "f32[8, 64, 14, 14]" = torch.ops.aten.add.Tensor(add_347, getitem_238);  add_347 = getitem_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_76: "f32[64]" = torch.ops.aten.sum.dim_IntList(add_348, [0, 2, 3])
    sub_213: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_706);  convolution_27 = unsqueeze_706 = None
    mul_788: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(add_348, sub_213)
    sum_77: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_788, [0, 2, 3]);  mul_788 = None
    mul_789: "f32[64]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    unsqueeze_707: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_789, 0);  mul_789 = None
    unsqueeze_708: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_707, 2);  unsqueeze_707 = None
    unsqueeze_709: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, 3);  unsqueeze_708 = None
    mul_790: "f32[64]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_791: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_792: "f32[64]" = torch.ops.aten.mul.Tensor(mul_790, mul_791);  mul_790 = mul_791 = None
    unsqueeze_710: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_792, 0);  mul_792 = None
    unsqueeze_711: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, 2);  unsqueeze_710 = None
    unsqueeze_712: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_711, 3);  unsqueeze_711 = None
    mul_793: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_713: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_793, 0);  mul_793 = None
    unsqueeze_714: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    mul_794: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_213, unsqueeze_712);  sub_213 = unsqueeze_712 = None
    sub_215: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(add_348, mul_794);  add_348 = mul_794 = None
    sub_216: "f32[8, 64, 14, 14]" = torch.ops.aten.sub.Tensor(sub_215, unsqueeze_709);  sub_215 = unsqueeze_709 = None
    mul_795: "f32[8, 64, 14, 14]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_715);  sub_216 = unsqueeze_715 = None
    mul_796: "f32[64]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_82);  sum_77 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_795, relu_18, primals_158, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_795 = primals_158 = None
    getitem_241: "f32[8, 192, 14, 14]" = convolution_backward_37[0]
    getitem_242: "f32[64, 192, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_120: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(relu_18);  relu_18 = None
    alias_121: "f32[8, 192, 14, 14]" = torch.ops.aten.alias.default(alias_120);  alias_120 = None
    le_25: "b8[8, 192, 14, 14]" = torch.ops.aten.le.Scalar(alias_121, 0);  alias_121 = None
    where_25: "f32[8, 192, 14, 14]" = torch.ops.aten.where.self(le_25, full_default, getitem_241);  le_25 = getitem_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_78: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_217: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_718);  convolution_26 = unsqueeze_718 = None
    mul_797: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_217)
    sum_79: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_797, [0, 2, 3]);  mul_797 = None
    mul_798: "f32[192]" = torch.ops.aten.mul.Tensor(sum_78, 0.0006377551020408163)
    unsqueeze_719: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_798, 0);  mul_798 = None
    unsqueeze_720: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_719, 2);  unsqueeze_719 = None
    unsqueeze_721: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, 3);  unsqueeze_720 = None
    mul_799: "f32[192]" = torch.ops.aten.mul.Tensor(sum_79, 0.0006377551020408163)
    mul_800: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_801: "f32[192]" = torch.ops.aten.mul.Tensor(mul_799, mul_800);  mul_799 = mul_800 = None
    unsqueeze_722: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_801, 0);  mul_801 = None
    unsqueeze_723: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, 2);  unsqueeze_722 = None
    unsqueeze_724: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_723, 3);  unsqueeze_723 = None
    mul_802: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_725: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_802, 0);  mul_802 = None
    unsqueeze_726: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    mul_803: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_217, unsqueeze_724);  sub_217 = unsqueeze_724 = None
    sub_219: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(where_25, mul_803);  where_25 = mul_803 = None
    sub_220: "f32[8, 192, 14, 14]" = torch.ops.aten.sub.Tensor(sub_219, unsqueeze_721);  sub_219 = unsqueeze_721 = None
    mul_804: "f32[8, 192, 14, 14]" = torch.ops.aten.mul.Tensor(sub_220, unsqueeze_727);  sub_220 = unsqueeze_727 = None
    mul_805: "f32[192]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_79);  sum_79 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_804, relu_17, primals_157, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 192, [True, True, False]);  mul_804 = primals_157 = None
    getitem_244: "f32[8, 192, 28, 28]" = convolution_backward_38[0]
    getitem_245: "f32[192, 1, 5, 5]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_123: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_17);  relu_17 = None
    alias_124: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_123);  alias_123 = None
    le_26: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_124, 0);  alias_124 = None
    where_26: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_26, full_default, getitem_244);  le_26 = getitem_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_80: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_221: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_730);  convolution_25 = unsqueeze_730 = None
    mul_806: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, sub_221)
    sum_81: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_806, [0, 2, 3]);  mul_806 = None
    mul_807: "f32[192]" = torch.ops.aten.mul.Tensor(sum_80, 0.00015943877551020407)
    unsqueeze_731: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_807, 0);  mul_807 = None
    unsqueeze_732: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_731, 2);  unsqueeze_731 = None
    unsqueeze_733: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, 3);  unsqueeze_732 = None
    mul_808: "f32[192]" = torch.ops.aten.mul.Tensor(sum_81, 0.00015943877551020407)
    mul_809: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_810: "f32[192]" = torch.ops.aten.mul.Tensor(mul_808, mul_809);  mul_808 = mul_809 = None
    unsqueeze_734: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_810, 0);  mul_810 = None
    unsqueeze_735: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, 2);  unsqueeze_734 = None
    unsqueeze_736: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_735, 3);  unsqueeze_735 = None
    mul_811: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_737: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_811, 0);  mul_811 = None
    unsqueeze_738: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_812: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_736);  sub_221 = unsqueeze_736 = None
    sub_223: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_26, mul_812);  where_26 = mul_812 = None
    sub_224: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_733);  sub_223 = unsqueeze_733 = None
    mul_813: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_739);  sub_224 = unsqueeze_739 = None
    mul_814: "f32[192]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_76);  sum_81 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_813, add_130, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_813 = add_130 = primals_156 = None
    getitem_247: "f32[8, 32, 28, 28]" = convolution_backward_39[0]
    getitem_248: "f32[192, 32, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_82: "f32[32]" = torch.ops.aten.sum.dim_IntList(getitem_247, [0, 2, 3])
    sub_225: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_742);  convolution_24 = unsqueeze_742 = None
    mul_815: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_247, sub_225)
    sum_83: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_815, [0, 2, 3]);  mul_815 = None
    mul_816: "f32[32]" = torch.ops.aten.mul.Tensor(sum_82, 0.00015943877551020407)
    unsqueeze_743: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_816, 0);  mul_816 = None
    unsqueeze_744: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_743, 2);  unsqueeze_743 = None
    unsqueeze_745: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, 3);  unsqueeze_744 = None
    mul_817: "f32[32]" = torch.ops.aten.mul.Tensor(sum_83, 0.00015943877551020407)
    mul_818: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_819: "f32[32]" = torch.ops.aten.mul.Tensor(mul_817, mul_818);  mul_817 = mul_818 = None
    unsqueeze_746: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_819, 0);  mul_819 = None
    unsqueeze_747: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, 2);  unsqueeze_746 = None
    unsqueeze_748: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_747, 3);  unsqueeze_747 = None
    mul_820: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_749: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_820, 0);  mul_820 = None
    unsqueeze_750: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    mul_821: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_748);  sub_225 = unsqueeze_748 = None
    sub_227: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(getitem_247, mul_821);  mul_821 = None
    sub_228: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_745);  sub_227 = unsqueeze_745 = None
    mul_822: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_751);  sub_228 = unsqueeze_751 = None
    mul_823: "f32[32]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_73);  sum_83 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_822, relu_16, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_822 = primals_155 = None
    getitem_250: "f32[8, 192, 28, 28]" = convolution_backward_40[0]
    getitem_251: "f32[32, 192, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_126: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_16);  relu_16 = None
    alias_127: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_126);  alias_126 = None
    le_27: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_127, 0);  alias_127 = None
    where_27: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_27, full_default, getitem_250);  le_27 = getitem_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_84: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_229: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_754);  convolution_23 = unsqueeze_754 = None
    mul_824: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, sub_229)
    sum_85: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_824, [0, 2, 3]);  mul_824 = None
    mul_825: "f32[192]" = torch.ops.aten.mul.Tensor(sum_84, 0.00015943877551020407)
    unsqueeze_755: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_825, 0);  mul_825 = None
    unsqueeze_756: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_755, 2);  unsqueeze_755 = None
    unsqueeze_757: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, 3);  unsqueeze_756 = None
    mul_826: "f32[192]" = torch.ops.aten.mul.Tensor(sum_85, 0.00015943877551020407)
    mul_827: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_828: "f32[192]" = torch.ops.aten.mul.Tensor(mul_826, mul_827);  mul_826 = mul_827 = None
    unsqueeze_758: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_828, 0);  mul_828 = None
    unsqueeze_759: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, 2);  unsqueeze_758 = None
    unsqueeze_760: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_759, 3);  unsqueeze_759 = None
    mul_829: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_761: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_829, 0);  mul_829 = None
    unsqueeze_762: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    mul_830: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_760);  sub_229 = unsqueeze_760 = None
    sub_231: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_27, mul_830);  where_27 = mul_830 = None
    sub_232: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_757);  sub_231 = unsqueeze_757 = None
    mul_831: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_763);  sub_232 = unsqueeze_763 = None
    mul_832: "f32[192]" = torch.ops.aten.mul.Tensor(sum_85, squeeze_70);  sum_85 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_831, relu_15, primals_154, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 192, [True, True, False]);  mul_831 = primals_154 = None
    getitem_253: "f32[8, 192, 28, 28]" = convolution_backward_41[0]
    getitem_254: "f32[192, 1, 3, 3]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_129: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_15);  relu_15 = None
    alias_130: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_129);  alias_129 = None
    le_28: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_130, 0);  alias_130 = None
    where_28: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_28, full_default, getitem_253);  le_28 = getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_86: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_233: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_766);  convolution_22 = unsqueeze_766 = None
    mul_833: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_28, sub_233)
    sum_87: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_833, [0, 2, 3]);  mul_833 = None
    mul_834: "f32[192]" = torch.ops.aten.mul.Tensor(sum_86, 0.00015943877551020407)
    unsqueeze_767: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_834, 0);  mul_834 = None
    unsqueeze_768: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_767, 2);  unsqueeze_767 = None
    unsqueeze_769: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, 3);  unsqueeze_768 = None
    mul_835: "f32[192]" = torch.ops.aten.mul.Tensor(sum_87, 0.00015943877551020407)
    mul_836: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_837: "f32[192]" = torch.ops.aten.mul.Tensor(mul_835, mul_836);  mul_835 = mul_836 = None
    unsqueeze_770: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_837, 0);  mul_837 = None
    unsqueeze_771: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, 2);  unsqueeze_770 = None
    unsqueeze_772: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_771, 3);  unsqueeze_771 = None
    mul_838: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_773: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_838, 0);  mul_838 = None
    unsqueeze_774: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    mul_839: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_233, unsqueeze_772);  sub_233 = unsqueeze_772 = None
    sub_235: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_28, mul_839);  where_28 = mul_839 = None
    sub_236: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_235, unsqueeze_769);  sub_235 = unsqueeze_769 = None
    mul_840: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_236, unsqueeze_775);  sub_236 = unsqueeze_775 = None
    mul_841: "f32[192]" = torch.ops.aten.mul.Tensor(sum_87, squeeze_67);  sum_87 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_840, add_114, primals_153, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_840 = add_114 = primals_153 = None
    getitem_256: "f32[8, 32, 28, 28]" = convolution_backward_42[0]
    getitem_257: "f32[192, 32, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_349: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(getitem_247, getitem_256);  getitem_247 = getitem_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_88: "f32[32]" = torch.ops.aten.sum.dim_IntList(add_349, [0, 2, 3])
    sub_237: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_778);  convolution_21 = unsqueeze_778 = None
    mul_842: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_349, sub_237)
    sum_89: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_842, [0, 2, 3]);  mul_842 = None
    mul_843: "f32[32]" = torch.ops.aten.mul.Tensor(sum_88, 0.00015943877551020407)
    unsqueeze_779: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_843, 0);  mul_843 = None
    unsqueeze_780: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_779, 2);  unsqueeze_779 = None
    unsqueeze_781: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, 3);  unsqueeze_780 = None
    mul_844: "f32[32]" = torch.ops.aten.mul.Tensor(sum_89, 0.00015943877551020407)
    mul_845: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_846: "f32[32]" = torch.ops.aten.mul.Tensor(mul_844, mul_845);  mul_844 = mul_845 = None
    unsqueeze_782: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_846, 0);  mul_846 = None
    unsqueeze_783: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, 2);  unsqueeze_782 = None
    unsqueeze_784: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_783, 3);  unsqueeze_783 = None
    mul_847: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_785: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_847, 0);  mul_847 = None
    unsqueeze_786: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    mul_848: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_237, unsqueeze_784);  sub_237 = unsqueeze_784 = None
    sub_239: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(add_349, mul_848);  mul_848 = None
    sub_240: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(sub_239, unsqueeze_781);  sub_239 = unsqueeze_781 = None
    mul_849: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_240, unsqueeze_787);  sub_240 = unsqueeze_787 = None
    mul_850: "f32[32]" = torch.ops.aten.mul.Tensor(sum_89, squeeze_64);  sum_89 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_849, relu_14, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_849 = primals_152 = None
    getitem_259: "f32[8, 192, 28, 28]" = convolution_backward_43[0]
    getitem_260: "f32[32, 192, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_132: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_14);  relu_14 = None
    alias_133: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_132);  alias_132 = None
    le_29: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_133, 0);  alias_133 = None
    where_29: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_29, full_default, getitem_259);  le_29 = getitem_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_90: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_241: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_790);  convolution_20 = unsqueeze_790 = None
    mul_851: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_29, sub_241)
    sum_91: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_851, [0, 2, 3]);  mul_851 = None
    mul_852: "f32[192]" = torch.ops.aten.mul.Tensor(sum_90, 0.00015943877551020407)
    unsqueeze_791: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_852, 0);  mul_852 = None
    unsqueeze_792: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_791, 2);  unsqueeze_791 = None
    unsqueeze_793: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, 3);  unsqueeze_792 = None
    mul_853: "f32[192]" = torch.ops.aten.mul.Tensor(sum_91, 0.00015943877551020407)
    mul_854: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_855: "f32[192]" = torch.ops.aten.mul.Tensor(mul_853, mul_854);  mul_853 = mul_854 = None
    unsqueeze_794: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_855, 0);  mul_855 = None
    unsqueeze_795: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, 2);  unsqueeze_794 = None
    unsqueeze_796: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_795, 3);  unsqueeze_795 = None
    mul_856: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_797: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_856, 0);  mul_856 = None
    unsqueeze_798: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_857: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_241, unsqueeze_796);  sub_241 = unsqueeze_796 = None
    sub_243: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_29, mul_857);  where_29 = mul_857 = None
    sub_244: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_243, unsqueeze_793);  sub_243 = unsqueeze_793 = None
    mul_858: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_244, unsqueeze_799);  sub_244 = unsqueeze_799 = None
    mul_859: "f32[192]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_61);  sum_91 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_858, relu_13, primals_151, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 192, [True, True, False]);  mul_858 = primals_151 = None
    getitem_262: "f32[8, 192, 28, 28]" = convolution_backward_44[0]
    getitem_263: "f32[192, 1, 5, 5]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_135: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(relu_13);  relu_13 = None
    alias_136: "f32[8, 192, 28, 28]" = torch.ops.aten.alias.default(alias_135);  alias_135 = None
    le_30: "b8[8, 192, 28, 28]" = torch.ops.aten.le.Scalar(alias_136, 0);  alias_136 = None
    where_30: "f32[8, 192, 28, 28]" = torch.ops.aten.where.self(le_30, full_default, getitem_262);  le_30 = getitem_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_92: "f32[192]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_245: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_802);  convolution_19 = unsqueeze_802 = None
    mul_860: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(where_30, sub_245)
    sum_93: "f32[192]" = torch.ops.aten.sum.dim_IntList(mul_860, [0, 2, 3]);  mul_860 = None
    mul_861: "f32[192]" = torch.ops.aten.mul.Tensor(sum_92, 0.00015943877551020407)
    unsqueeze_803: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_861, 0);  mul_861 = None
    unsqueeze_804: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_803, 2);  unsqueeze_803 = None
    unsqueeze_805: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, 3);  unsqueeze_804 = None
    mul_862: "f32[192]" = torch.ops.aten.mul.Tensor(sum_93, 0.00015943877551020407)
    mul_863: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_864: "f32[192]" = torch.ops.aten.mul.Tensor(mul_862, mul_863);  mul_862 = mul_863 = None
    unsqueeze_806: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_864, 0);  mul_864 = None
    unsqueeze_807: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, 2);  unsqueeze_806 = None
    unsqueeze_808: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_807, 3);  unsqueeze_807 = None
    mul_865: "f32[192]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_809: "f32[1, 192]" = torch.ops.aten.unsqueeze.default(mul_865, 0);  mul_865 = None
    unsqueeze_810: "f32[1, 192, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 192, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_866: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_245, unsqueeze_808);  sub_245 = unsqueeze_808 = None
    sub_247: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(where_30, mul_866);  where_30 = mul_866 = None
    sub_248: "f32[8, 192, 28, 28]" = torch.ops.aten.sub.Tensor(sub_247, unsqueeze_805);  sub_247 = unsqueeze_805 = None
    mul_867: "f32[8, 192, 28, 28]" = torch.ops.aten.mul.Tensor(sub_248, unsqueeze_811);  sub_248 = unsqueeze_811 = None
    mul_868: "f32[192]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_58);  sum_93 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_867, add_98, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_867 = add_98 = primals_150 = None
    getitem_265: "f32[8, 32, 28, 28]" = convolution_backward_45[0]
    getitem_266: "f32[192, 32, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_350: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_349, getitem_265);  add_349 = getitem_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_94: "f32[32]" = torch.ops.aten.sum.dim_IntList(add_350, [0, 2, 3])
    sub_249: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_814);  convolution_18 = unsqueeze_814 = None
    mul_869: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_350, sub_249)
    sum_95: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_869, [0, 2, 3]);  mul_869 = None
    mul_870: "f32[32]" = torch.ops.aten.mul.Tensor(sum_94, 0.00015943877551020407)
    unsqueeze_815: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_870, 0);  mul_870 = None
    unsqueeze_816: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_815, 2);  unsqueeze_815 = None
    unsqueeze_817: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, 3);  unsqueeze_816 = None
    mul_871: "f32[32]" = torch.ops.aten.mul.Tensor(sum_95, 0.00015943877551020407)
    mul_872: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_873: "f32[32]" = torch.ops.aten.mul.Tensor(mul_871, mul_872);  mul_871 = mul_872 = None
    unsqueeze_818: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_873, 0);  mul_873 = None
    unsqueeze_819: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, 2);  unsqueeze_818 = None
    unsqueeze_820: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_819, 3);  unsqueeze_819 = None
    mul_874: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_821: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_874, 0);  mul_874 = None
    unsqueeze_822: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_875: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_249, unsqueeze_820);  sub_249 = unsqueeze_820 = None
    sub_251: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(add_350, mul_875);  mul_875 = None
    sub_252: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(sub_251, unsqueeze_817);  sub_251 = unsqueeze_817 = None
    mul_876: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_252, unsqueeze_823);  sub_252 = unsqueeze_823 = None
    mul_877: "f32[32]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_55);  sum_95 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_876, relu_12, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_876 = primals_149 = None
    getitem_268: "f32[8, 96, 28, 28]" = convolution_backward_46[0]
    getitem_269: "f32[32, 96, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_138: "f32[8, 96, 28, 28]" = torch.ops.aten.alias.default(relu_12);  relu_12 = None
    alias_139: "f32[8, 96, 28, 28]" = torch.ops.aten.alias.default(alias_138);  alias_138 = None
    le_31: "b8[8, 96, 28, 28]" = torch.ops.aten.le.Scalar(alias_139, 0);  alias_139 = None
    where_31: "f32[8, 96, 28, 28]" = torch.ops.aten.where.self(le_31, full_default, getitem_268);  le_31 = getitem_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_96: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_253: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_826);  convolution_17 = unsqueeze_826 = None
    mul_878: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(where_31, sub_253)
    sum_97: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_878, [0, 2, 3]);  mul_878 = None
    mul_879: "f32[96]" = torch.ops.aten.mul.Tensor(sum_96, 0.00015943877551020407)
    unsqueeze_827: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_879, 0);  mul_879 = None
    unsqueeze_828: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_827, 2);  unsqueeze_827 = None
    unsqueeze_829: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, 3);  unsqueeze_828 = None
    mul_880: "f32[96]" = torch.ops.aten.mul.Tensor(sum_97, 0.00015943877551020407)
    mul_881: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_882: "f32[96]" = torch.ops.aten.mul.Tensor(mul_880, mul_881);  mul_880 = mul_881 = None
    unsqueeze_830: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_882, 0);  mul_882 = None
    unsqueeze_831: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, 2);  unsqueeze_830 = None
    unsqueeze_832: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_831, 3);  unsqueeze_831 = None
    mul_883: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_833: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_883, 0);  mul_883 = None
    unsqueeze_834: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    mul_884: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_253, unsqueeze_832);  sub_253 = unsqueeze_832 = None
    sub_255: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(where_31, mul_884);  where_31 = mul_884 = None
    sub_256: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(sub_255, unsqueeze_829);  sub_255 = unsqueeze_829 = None
    mul_885: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_256, unsqueeze_835);  sub_256 = unsqueeze_835 = None
    mul_886: "f32[96]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_52);  sum_97 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_885, relu_11, primals_148, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 96, [True, True, False]);  mul_885 = primals_148 = None
    getitem_271: "f32[8, 96, 28, 28]" = convolution_backward_47[0]
    getitem_272: "f32[96, 1, 5, 5]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_141: "f32[8, 96, 28, 28]" = torch.ops.aten.alias.default(relu_11);  relu_11 = None
    alias_142: "f32[8, 96, 28, 28]" = torch.ops.aten.alias.default(alias_141);  alias_141 = None
    le_32: "b8[8, 96, 28, 28]" = torch.ops.aten.le.Scalar(alias_142, 0);  alias_142 = None
    where_32: "f32[8, 96, 28, 28]" = torch.ops.aten.where.self(le_32, full_default, getitem_271);  le_32 = getitem_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_98: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_257: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_838);  convolution_16 = unsqueeze_838 = None
    mul_887: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(where_32, sub_257)
    sum_99: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_887, [0, 2, 3]);  mul_887 = None
    mul_888: "f32[96]" = torch.ops.aten.mul.Tensor(sum_98, 0.00015943877551020407)
    unsqueeze_839: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_888, 0);  mul_888 = None
    unsqueeze_840: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_839, 2);  unsqueeze_839 = None
    unsqueeze_841: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, 3);  unsqueeze_840 = None
    mul_889: "f32[96]" = torch.ops.aten.mul.Tensor(sum_99, 0.00015943877551020407)
    mul_890: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_891: "f32[96]" = torch.ops.aten.mul.Tensor(mul_889, mul_890);  mul_889 = mul_890 = None
    unsqueeze_842: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_891, 0);  mul_891 = None
    unsqueeze_843: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, 2);  unsqueeze_842 = None
    unsqueeze_844: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_843, 3);  unsqueeze_843 = None
    mul_892: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_845: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_892, 0);  mul_892 = None
    unsqueeze_846: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    mul_893: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_257, unsqueeze_844);  sub_257 = unsqueeze_844 = None
    sub_259: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(where_32, mul_893);  where_32 = mul_893 = None
    sub_260: "f32[8, 96, 28, 28]" = torch.ops.aten.sub.Tensor(sub_259, unsqueeze_841);  sub_259 = unsqueeze_841 = None
    mul_894: "f32[8, 96, 28, 28]" = torch.ops.aten.mul.Tensor(sub_260, unsqueeze_847);  sub_260 = unsqueeze_847 = None
    mul_895: "f32[96]" = torch.ops.aten.mul.Tensor(sum_99, squeeze_49);  sum_99 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_894, add_82, primals_147, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_894 = add_82 = primals_147 = None
    getitem_274: "f32[8, 32, 28, 28]" = convolution_backward_48[0]
    getitem_275: "f32[96, 32, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_351: "f32[8, 32, 28, 28]" = torch.ops.aten.add.Tensor(add_350, getitem_274);  add_350 = getitem_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_100: "f32[32]" = torch.ops.aten.sum.dim_IntList(add_351, [0, 2, 3])
    sub_261: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_850);  convolution_15 = unsqueeze_850 = None
    mul_896: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(add_351, sub_261)
    sum_101: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_896, [0, 2, 3]);  mul_896 = None
    mul_897: "f32[32]" = torch.ops.aten.mul.Tensor(sum_100, 0.00015943877551020407)
    unsqueeze_851: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_897, 0);  mul_897 = None
    unsqueeze_852: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_851, 2);  unsqueeze_851 = None
    unsqueeze_853: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, 3);  unsqueeze_852 = None
    mul_898: "f32[32]" = torch.ops.aten.mul.Tensor(sum_101, 0.00015943877551020407)
    mul_899: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_900: "f32[32]" = torch.ops.aten.mul.Tensor(mul_898, mul_899);  mul_898 = mul_899 = None
    unsqueeze_854: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_900, 0);  mul_900 = None
    unsqueeze_855: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, 2);  unsqueeze_854 = None
    unsqueeze_856: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_855, 3);  unsqueeze_855 = None
    mul_901: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_857: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_901, 0);  mul_901 = None
    unsqueeze_858: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    mul_902: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_261, unsqueeze_856);  sub_261 = unsqueeze_856 = None
    sub_263: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(add_351, mul_902);  add_351 = mul_902 = None
    sub_264: "f32[8, 32, 28, 28]" = torch.ops.aten.sub.Tensor(sub_263, unsqueeze_853);  sub_263 = unsqueeze_853 = None
    mul_903: "f32[8, 32, 28, 28]" = torch.ops.aten.mul.Tensor(sub_264, unsqueeze_859);  sub_264 = unsqueeze_859 = None
    mul_904: "f32[32]" = torch.ops.aten.mul.Tensor(sum_101, squeeze_46);  sum_101 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_903, relu_10, primals_146, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_903 = primals_146 = None
    getitem_277: "f32[8, 144, 28, 28]" = convolution_backward_49[0]
    getitem_278: "f32[32, 144, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_144: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(relu_10);  relu_10 = None
    alias_145: "f32[8, 144, 28, 28]" = torch.ops.aten.alias.default(alias_144);  alias_144 = None
    le_33: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(alias_145, 0);  alias_145 = None
    where_33: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_33, full_default, getitem_277);  le_33 = getitem_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_102: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_265: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_862);  convolution_14 = unsqueeze_862 = None
    mul_905: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_33, sub_265)
    sum_103: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_905, [0, 2, 3]);  mul_905 = None
    mul_906: "f32[144]" = torch.ops.aten.mul.Tensor(sum_102, 0.00015943877551020407)
    unsqueeze_863: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_906, 0);  mul_906 = None
    unsqueeze_864: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_863, 2);  unsqueeze_863 = None
    unsqueeze_865: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, 3);  unsqueeze_864 = None
    mul_907: "f32[144]" = torch.ops.aten.mul.Tensor(sum_103, 0.00015943877551020407)
    mul_908: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_909: "f32[144]" = torch.ops.aten.mul.Tensor(mul_907, mul_908);  mul_907 = mul_908 = None
    unsqueeze_866: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_909, 0);  mul_909 = None
    unsqueeze_867: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, 2);  unsqueeze_866 = None
    unsqueeze_868: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_867, 3);  unsqueeze_867 = None
    mul_910: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_869: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_910, 0);  mul_910 = None
    unsqueeze_870: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    mul_911: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_265, unsqueeze_868);  sub_265 = unsqueeze_868 = None
    sub_267: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_33, mul_911);  where_33 = mul_911 = None
    sub_268: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_267, unsqueeze_865);  sub_267 = unsqueeze_865 = None
    mul_912: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_268, unsqueeze_871);  sub_268 = unsqueeze_871 = None
    mul_913: "f32[144]" = torch.ops.aten.mul.Tensor(sum_103, squeeze_43);  sum_103 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_912, relu_9, primals_145, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 144, [True, True, False]);  mul_912 = primals_145 = None
    getitem_280: "f32[8, 144, 56, 56]" = convolution_backward_50[0]
    getitem_281: "f32[144, 1, 5, 5]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_147: "f32[8, 144, 56, 56]" = torch.ops.aten.alias.default(relu_9);  relu_9 = None
    alias_148: "f32[8, 144, 56, 56]" = torch.ops.aten.alias.default(alias_147);  alias_147 = None
    le_34: "b8[8, 144, 56, 56]" = torch.ops.aten.le.Scalar(alias_148, 0);  alias_148 = None
    where_34: "f32[8, 144, 56, 56]" = torch.ops.aten.where.self(le_34, full_default, getitem_280);  le_34 = getitem_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_104: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_269: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_874);  convolution_13 = unsqueeze_874 = None
    mul_914: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(where_34, sub_269)
    sum_105: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_914, [0, 2, 3]);  mul_914 = None
    mul_915: "f32[144]" = torch.ops.aten.mul.Tensor(sum_104, 3.985969387755102e-05)
    unsqueeze_875: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_915, 0);  mul_915 = None
    unsqueeze_876: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_875, 2);  unsqueeze_875 = None
    unsqueeze_877: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, 3);  unsqueeze_876 = None
    mul_916: "f32[144]" = torch.ops.aten.mul.Tensor(sum_105, 3.985969387755102e-05)
    mul_917: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_918: "f32[144]" = torch.ops.aten.mul.Tensor(mul_916, mul_917);  mul_916 = mul_917 = None
    unsqueeze_878: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_918, 0);  mul_918 = None
    unsqueeze_879: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, 2);  unsqueeze_878 = None
    unsqueeze_880: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_879, 3);  unsqueeze_879 = None
    mul_919: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_881: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_919, 0);  mul_919 = None
    unsqueeze_882: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    mul_920: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_269, unsqueeze_880);  sub_269 = unsqueeze_880 = None
    sub_271: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(where_34, mul_920);  where_34 = mul_920 = None
    sub_272: "f32[8, 144, 56, 56]" = torch.ops.aten.sub.Tensor(sub_271, unsqueeze_877);  sub_271 = unsqueeze_877 = None
    mul_921: "f32[8, 144, 56, 56]" = torch.ops.aten.mul.Tensor(sub_272, unsqueeze_883);  sub_272 = unsqueeze_883 = None
    mul_922: "f32[144]" = torch.ops.aten.mul.Tensor(sum_105, squeeze_40);  sum_105 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_921, add_67, primals_144, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_921 = add_67 = primals_144 = None
    getitem_283: "f32[8, 24, 56, 56]" = convolution_backward_51[0]
    getitem_284: "f32[144, 24, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_106: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_283, [0, 2, 3])
    sub_273: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_886);  convolution_12 = unsqueeze_886 = None
    mul_923: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_283, sub_273)
    sum_107: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_923, [0, 2, 3]);  mul_923 = None
    mul_924: "f32[24]" = torch.ops.aten.mul.Tensor(sum_106, 3.985969387755102e-05)
    unsqueeze_887: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_924, 0);  mul_924 = None
    unsqueeze_888: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_887, 2);  unsqueeze_887 = None
    unsqueeze_889: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, 3);  unsqueeze_888 = None
    mul_925: "f32[24]" = torch.ops.aten.mul.Tensor(sum_107, 3.985969387755102e-05)
    mul_926: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_927: "f32[24]" = torch.ops.aten.mul.Tensor(mul_925, mul_926);  mul_925 = mul_926 = None
    unsqueeze_890: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_927, 0);  mul_927 = None
    unsqueeze_891: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, 2);  unsqueeze_890 = None
    unsqueeze_892: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_891, 3);  unsqueeze_891 = None
    mul_928: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_893: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_928, 0);  mul_928 = None
    unsqueeze_894: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    mul_929: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_273, unsqueeze_892);  sub_273 = unsqueeze_892 = None
    sub_275: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(getitem_283, mul_929);  mul_929 = None
    sub_276: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_275, unsqueeze_889);  sub_275 = unsqueeze_889 = None
    mul_930: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_276, unsqueeze_895);  sub_276 = unsqueeze_895 = None
    mul_931: "f32[24]" = torch.ops.aten.mul.Tensor(sum_107, squeeze_37);  sum_107 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_930, relu_8, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_930 = primals_143 = None
    getitem_286: "f32[8, 24, 56, 56]" = convolution_backward_52[0]
    getitem_287: "f32[24, 24, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_150: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(relu_8);  relu_8 = None
    alias_151: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(alias_150);  alias_150 = None
    le_35: "b8[8, 24, 56, 56]" = torch.ops.aten.le.Scalar(alias_151, 0);  alias_151 = None
    where_35: "f32[8, 24, 56, 56]" = torch.ops.aten.where.self(le_35, full_default, getitem_286);  le_35 = getitem_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_108: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_277: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_898);  convolution_11 = unsqueeze_898 = None
    mul_932: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_35, sub_277)
    sum_109: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_932, [0, 2, 3]);  mul_932 = None
    mul_933: "f32[24]" = torch.ops.aten.mul.Tensor(sum_108, 3.985969387755102e-05)
    unsqueeze_899: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_933, 0);  mul_933 = None
    unsqueeze_900: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_899, 2);  unsqueeze_899 = None
    unsqueeze_901: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, 3);  unsqueeze_900 = None
    mul_934: "f32[24]" = torch.ops.aten.mul.Tensor(sum_109, 3.985969387755102e-05)
    mul_935: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_936: "f32[24]" = torch.ops.aten.mul.Tensor(mul_934, mul_935);  mul_934 = mul_935 = None
    unsqueeze_902: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_936, 0);  mul_936 = None
    unsqueeze_903: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, 2);  unsqueeze_902 = None
    unsqueeze_904: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_903, 3);  unsqueeze_903 = None
    mul_937: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_905: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_937, 0);  mul_937 = None
    unsqueeze_906: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    mul_938: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_277, unsqueeze_904);  sub_277 = unsqueeze_904 = None
    sub_279: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_35, mul_938);  where_35 = mul_938 = None
    sub_280: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_279, unsqueeze_901);  sub_279 = unsqueeze_901 = None
    mul_939: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_280, unsqueeze_907);  sub_280 = unsqueeze_907 = None
    mul_940: "f32[24]" = torch.ops.aten.mul.Tensor(sum_109, squeeze_34);  sum_109 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_939, relu_7, primals_142, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 24, [True, True, False]);  mul_939 = primals_142 = None
    getitem_289: "f32[8, 24, 56, 56]" = convolution_backward_53[0]
    getitem_290: "f32[24, 1, 3, 3]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_153: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(relu_7);  relu_7 = None
    alias_154: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(alias_153);  alias_153 = None
    le_36: "b8[8, 24, 56, 56]" = torch.ops.aten.le.Scalar(alias_154, 0);  alias_154 = None
    where_36: "f32[8, 24, 56, 56]" = torch.ops.aten.where.self(le_36, full_default, getitem_289);  le_36 = getitem_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_110: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_281: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_910);  convolution_10 = unsqueeze_910 = None
    mul_941: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_36, sub_281)
    sum_111: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_941, [0, 2, 3]);  mul_941 = None
    mul_942: "f32[24]" = torch.ops.aten.mul.Tensor(sum_110, 3.985969387755102e-05)
    unsqueeze_911: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_942, 0);  mul_942 = None
    unsqueeze_912: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_911, 2);  unsqueeze_911 = None
    unsqueeze_913: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, 3);  unsqueeze_912 = None
    mul_943: "f32[24]" = torch.ops.aten.mul.Tensor(sum_111, 3.985969387755102e-05)
    mul_944: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_945: "f32[24]" = torch.ops.aten.mul.Tensor(mul_943, mul_944);  mul_943 = mul_944 = None
    unsqueeze_914: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_945, 0);  mul_945 = None
    unsqueeze_915: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, 2);  unsqueeze_914 = None
    unsqueeze_916: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_915, 3);  unsqueeze_915 = None
    mul_946: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_917: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_946, 0);  mul_946 = None
    unsqueeze_918: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    mul_947: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_281, unsqueeze_916);  sub_281 = unsqueeze_916 = None
    sub_283: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_36, mul_947);  where_36 = mul_947 = None
    sub_284: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_283, unsqueeze_913);  sub_283 = unsqueeze_913 = None
    mul_948: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_284, unsqueeze_919);  sub_284 = unsqueeze_919 = None
    mul_949: "f32[24]" = torch.ops.aten.mul.Tensor(sum_111, squeeze_31);  sum_111 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_948, add_51, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_948 = add_51 = primals_141 = None
    getitem_292: "f32[8, 24, 56, 56]" = convolution_backward_54[0]
    getitem_293: "f32[24, 24, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_352: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_283, getitem_292);  getitem_283 = getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_112: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_352, [0, 2, 3])
    sub_285: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_922);  convolution_9 = unsqueeze_922 = None
    mul_950: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_352, sub_285)
    sum_113: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_950, [0, 2, 3]);  mul_950 = None
    mul_951: "f32[24]" = torch.ops.aten.mul.Tensor(sum_112, 3.985969387755102e-05)
    unsqueeze_923: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_951, 0);  mul_951 = None
    unsqueeze_924: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_923, 2);  unsqueeze_923 = None
    unsqueeze_925: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, 3);  unsqueeze_924 = None
    mul_952: "f32[24]" = torch.ops.aten.mul.Tensor(sum_113, 3.985969387755102e-05)
    mul_953: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_954: "f32[24]" = torch.ops.aten.mul.Tensor(mul_952, mul_953);  mul_952 = mul_953 = None
    unsqueeze_926: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_954, 0);  mul_954 = None
    unsqueeze_927: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, 2);  unsqueeze_926 = None
    unsqueeze_928: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_927, 3);  unsqueeze_927 = None
    mul_955: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_929: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_955, 0);  mul_955 = None
    unsqueeze_930: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_929, 2);  unsqueeze_929 = None
    unsqueeze_931: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, 3);  unsqueeze_930 = None
    mul_956: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_285, unsqueeze_928);  sub_285 = unsqueeze_928 = None
    sub_287: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_352, mul_956);  mul_956 = None
    sub_288: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_287, unsqueeze_925);  sub_287 = unsqueeze_925 = None
    mul_957: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_288, unsqueeze_931);  sub_288 = unsqueeze_931 = None
    mul_958: "f32[24]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_28);  sum_113 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_957, relu_6, primals_140, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_957 = primals_140 = None
    getitem_295: "f32[8, 24, 56, 56]" = convolution_backward_55[0]
    getitem_296: "f32[24, 24, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_156: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(relu_6);  relu_6 = None
    alias_157: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(alias_156);  alias_156 = None
    le_37: "b8[8, 24, 56, 56]" = torch.ops.aten.le.Scalar(alias_157, 0);  alias_157 = None
    where_37: "f32[8, 24, 56, 56]" = torch.ops.aten.where.self(le_37, full_default, getitem_295);  le_37 = getitem_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_114: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_289: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_934);  convolution_8 = unsqueeze_934 = None
    mul_959: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_37, sub_289)
    sum_115: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_959, [0, 2, 3]);  mul_959 = None
    mul_960: "f32[24]" = torch.ops.aten.mul.Tensor(sum_114, 3.985969387755102e-05)
    unsqueeze_935: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_960, 0);  mul_960 = None
    unsqueeze_936: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_935, 2);  unsqueeze_935 = None
    unsqueeze_937: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, 3);  unsqueeze_936 = None
    mul_961: "f32[24]" = torch.ops.aten.mul.Tensor(sum_115, 3.985969387755102e-05)
    mul_962: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_963: "f32[24]" = torch.ops.aten.mul.Tensor(mul_961, mul_962);  mul_961 = mul_962 = None
    unsqueeze_938: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_963, 0);  mul_963 = None
    unsqueeze_939: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, 2);  unsqueeze_938 = None
    unsqueeze_940: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_939, 3);  unsqueeze_939 = None
    mul_964: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_941: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_964, 0);  mul_964 = None
    unsqueeze_942: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_941, 2);  unsqueeze_941 = None
    unsqueeze_943: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, 3);  unsqueeze_942 = None
    mul_965: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_289, unsqueeze_940);  sub_289 = unsqueeze_940 = None
    sub_291: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_37, mul_965);  where_37 = mul_965 = None
    sub_292: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_291, unsqueeze_937);  sub_291 = unsqueeze_937 = None
    mul_966: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_292, unsqueeze_943);  sub_292 = unsqueeze_943 = None
    mul_967: "f32[24]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_25);  sum_115 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_966, relu_5, primals_139, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 24, [True, True, False]);  mul_966 = primals_139 = None
    getitem_298: "f32[8, 24, 56, 56]" = convolution_backward_56[0]
    getitem_299: "f32[24, 1, 3, 3]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_159: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(relu_5);  relu_5 = None
    alias_160: "f32[8, 24, 56, 56]" = torch.ops.aten.alias.default(alias_159);  alias_159 = None
    le_38: "b8[8, 24, 56, 56]" = torch.ops.aten.le.Scalar(alias_160, 0);  alias_160 = None
    where_38: "f32[8, 24, 56, 56]" = torch.ops.aten.where.self(le_38, full_default, getitem_298);  le_38 = getitem_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_116: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_293: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_946);  convolution_7 = unsqueeze_946 = None
    mul_968: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_38, sub_293)
    sum_117: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_968, [0, 2, 3]);  mul_968 = None
    mul_969: "f32[24]" = torch.ops.aten.mul.Tensor(sum_116, 3.985969387755102e-05)
    unsqueeze_947: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_969, 0);  mul_969 = None
    unsqueeze_948: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_947, 2);  unsqueeze_947 = None
    unsqueeze_949: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, 3);  unsqueeze_948 = None
    mul_970: "f32[24]" = torch.ops.aten.mul.Tensor(sum_117, 3.985969387755102e-05)
    mul_971: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_972: "f32[24]" = torch.ops.aten.mul.Tensor(mul_970, mul_971);  mul_970 = mul_971 = None
    unsqueeze_950: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_972, 0);  mul_972 = None
    unsqueeze_951: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, 2);  unsqueeze_950 = None
    unsqueeze_952: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_951, 3);  unsqueeze_951 = None
    mul_973: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_953: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_973, 0);  mul_973 = None
    unsqueeze_954: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_953, 2);  unsqueeze_953 = None
    unsqueeze_955: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, 3);  unsqueeze_954 = None
    mul_974: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_293, unsqueeze_952);  sub_293 = unsqueeze_952 = None
    sub_295: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_38, mul_974);  where_38 = mul_974 = None
    sub_296: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_295, unsqueeze_949);  sub_295 = unsqueeze_949 = None
    mul_975: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_296, unsqueeze_955);  sub_296 = unsqueeze_955 = None
    mul_976: "f32[24]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_22);  sum_117 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_975, add_35, primals_138, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_975 = add_35 = primals_138 = None
    getitem_301: "f32[8, 24, 56, 56]" = convolution_backward_57[0]
    getitem_302: "f32[24, 24, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_353: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(add_352, getitem_301);  add_352 = getitem_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_118: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_353, [0, 2, 3])
    sub_297: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_958);  convolution_6 = unsqueeze_958 = None
    mul_977: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_353, sub_297)
    sum_119: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_977, [0, 2, 3]);  mul_977 = None
    mul_978: "f32[24]" = torch.ops.aten.mul.Tensor(sum_118, 3.985969387755102e-05)
    unsqueeze_959: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_978, 0);  mul_978 = None
    unsqueeze_960: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_959, 2);  unsqueeze_959 = None
    unsqueeze_961: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, 3);  unsqueeze_960 = None
    mul_979: "f32[24]" = torch.ops.aten.mul.Tensor(sum_119, 3.985969387755102e-05)
    mul_980: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_981: "f32[24]" = torch.ops.aten.mul.Tensor(mul_979, mul_980);  mul_979 = mul_980 = None
    unsqueeze_962: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_981, 0);  mul_981 = None
    unsqueeze_963: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, 2);  unsqueeze_962 = None
    unsqueeze_964: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_963, 3);  unsqueeze_963 = None
    mul_982: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_965: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_982, 0);  mul_982 = None
    unsqueeze_966: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_965, 2);  unsqueeze_965 = None
    unsqueeze_967: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, 3);  unsqueeze_966 = None
    mul_983: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_297, unsqueeze_964);  sub_297 = unsqueeze_964 = None
    sub_299: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(add_353, mul_983);  add_353 = mul_983 = None
    sub_300: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_299, unsqueeze_961);  sub_299 = unsqueeze_961 = None
    mul_984: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_300, unsqueeze_967);  sub_300 = unsqueeze_967 = None
    mul_985: "f32[24]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_19);  sum_119 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_984, relu_4, primals_137, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_984 = primals_137 = None
    getitem_304: "f32[8, 96, 56, 56]" = convolution_backward_58[0]
    getitem_305: "f32[24, 96, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_162: "f32[8, 96, 56, 56]" = torch.ops.aten.alias.default(relu_4);  relu_4 = None
    alias_163: "f32[8, 96, 56, 56]" = torch.ops.aten.alias.default(alias_162);  alias_162 = None
    le_39: "b8[8, 96, 56, 56]" = torch.ops.aten.le.Scalar(alias_163, 0);  alias_163 = None
    where_39: "f32[8, 96, 56, 56]" = torch.ops.aten.where.self(le_39, full_default, getitem_304);  le_39 = getitem_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_120: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_301: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_970);  convolution_5 = unsqueeze_970 = None
    mul_986: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(where_39, sub_301)
    sum_121: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_986, [0, 2, 3]);  mul_986 = None
    mul_987: "f32[96]" = torch.ops.aten.mul.Tensor(sum_120, 3.985969387755102e-05)
    unsqueeze_971: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_987, 0);  mul_987 = None
    unsqueeze_972: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_971, 2);  unsqueeze_971 = None
    unsqueeze_973: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, 3);  unsqueeze_972 = None
    mul_988: "f32[96]" = torch.ops.aten.mul.Tensor(sum_121, 3.985969387755102e-05)
    mul_989: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_990: "f32[96]" = torch.ops.aten.mul.Tensor(mul_988, mul_989);  mul_988 = mul_989 = None
    unsqueeze_974: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_990, 0);  mul_990 = None
    unsqueeze_975: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, 2);  unsqueeze_974 = None
    unsqueeze_976: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_975, 3);  unsqueeze_975 = None
    mul_991: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_977: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_991, 0);  mul_991 = None
    unsqueeze_978: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_977, 2);  unsqueeze_977 = None
    unsqueeze_979: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, 3);  unsqueeze_978 = None
    mul_992: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_301, unsqueeze_976);  sub_301 = unsqueeze_976 = None
    sub_303: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(where_39, mul_992);  where_39 = mul_992 = None
    sub_304: "f32[8, 96, 56, 56]" = torch.ops.aten.sub.Tensor(sub_303, unsqueeze_973);  sub_303 = unsqueeze_973 = None
    mul_993: "f32[8, 96, 56, 56]" = torch.ops.aten.mul.Tensor(sub_304, unsqueeze_979);  sub_304 = unsqueeze_979 = None
    mul_994: "f32[96]" = torch.ops.aten.mul.Tensor(sum_121, squeeze_16);  sum_121 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_993, relu_3, primals_136, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 96, [True, True, False]);  mul_993 = primals_136 = None
    getitem_307: "f32[8, 96, 112, 112]" = convolution_backward_59[0]
    getitem_308: "f32[96, 1, 3, 3]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_165: "f32[8, 96, 112, 112]" = torch.ops.aten.alias.default(relu_3);  relu_3 = None
    alias_166: "f32[8, 96, 112, 112]" = torch.ops.aten.alias.default(alias_165);  alias_165 = None
    le_40: "b8[8, 96, 112, 112]" = torch.ops.aten.le.Scalar(alias_166, 0);  alias_166 = None
    where_40: "f32[8, 96, 112, 112]" = torch.ops.aten.where.self(le_40, full_default, getitem_307);  le_40 = getitem_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_122: "f32[96]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_305: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_982);  convolution_4 = unsqueeze_982 = None
    mul_995: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(where_40, sub_305)
    sum_123: "f32[96]" = torch.ops.aten.sum.dim_IntList(mul_995, [0, 2, 3]);  mul_995 = None
    mul_996: "f32[96]" = torch.ops.aten.mul.Tensor(sum_122, 9.964923469387754e-06)
    unsqueeze_983: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_996, 0);  mul_996 = None
    unsqueeze_984: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_983, 2);  unsqueeze_983 = None
    unsqueeze_985: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, 3);  unsqueeze_984 = None
    mul_997: "f32[96]" = torch.ops.aten.mul.Tensor(sum_123, 9.964923469387754e-06)
    mul_998: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_999: "f32[96]" = torch.ops.aten.mul.Tensor(mul_997, mul_998);  mul_997 = mul_998 = None
    unsqueeze_986: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_999, 0);  mul_999 = None
    unsqueeze_987: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, 2);  unsqueeze_986 = None
    unsqueeze_988: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_987, 3);  unsqueeze_987 = None
    mul_1000: "f32[96]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_989: "f32[1, 96]" = torch.ops.aten.unsqueeze.default(mul_1000, 0);  mul_1000 = None
    unsqueeze_990: "f32[1, 96, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_989, 2);  unsqueeze_989 = None
    unsqueeze_991: "f32[1, 96, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, 3);  unsqueeze_990 = None
    mul_1001: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_305, unsqueeze_988);  sub_305 = unsqueeze_988 = None
    sub_307: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(where_40, mul_1001);  where_40 = mul_1001 = None
    sub_308: "f32[8, 96, 112, 112]" = torch.ops.aten.sub.Tensor(sub_307, unsqueeze_985);  sub_307 = unsqueeze_985 = None
    mul_1002: "f32[8, 96, 112, 112]" = torch.ops.aten.mul.Tensor(sub_308, unsqueeze_991);  sub_308 = unsqueeze_991 = None
    mul_1003: "f32[96]" = torch.ops.aten.mul.Tensor(sum_123, squeeze_13);  sum_123 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_1002, add_20, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1002 = add_20 = primals_135 = None
    getitem_310: "f32[8, 16, 112, 112]" = convolution_backward_60[0]
    getitem_311: "f32[96, 16, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_124: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_310, [0, 2, 3])
    sub_309: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_994);  convolution_3 = unsqueeze_994 = None
    mul_1004: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_310, sub_309)
    sum_125: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1004, [0, 2, 3]);  mul_1004 = None
    mul_1005: "f32[16]" = torch.ops.aten.mul.Tensor(sum_124, 9.964923469387754e-06)
    unsqueeze_995: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1005, 0);  mul_1005 = None
    unsqueeze_996: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_995, 2);  unsqueeze_995 = None
    unsqueeze_997: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, 3);  unsqueeze_996 = None
    mul_1006: "f32[16]" = torch.ops.aten.mul.Tensor(sum_125, 9.964923469387754e-06)
    mul_1007: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_1008: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1006, mul_1007);  mul_1006 = mul_1007 = None
    unsqueeze_998: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1008, 0);  mul_1008 = None
    unsqueeze_999: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, 2);  unsqueeze_998 = None
    unsqueeze_1000: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_999, 3);  unsqueeze_999 = None
    mul_1009: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_1001: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1009, 0);  mul_1009 = None
    unsqueeze_1002: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1001, 2);  unsqueeze_1001 = None
    unsqueeze_1003: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, 3);  unsqueeze_1002 = None
    mul_1010: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_309, unsqueeze_1000);  sub_309 = unsqueeze_1000 = None
    sub_311: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(getitem_310, mul_1010);  mul_1010 = None
    sub_312: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_311, unsqueeze_997);  sub_311 = unsqueeze_997 = None
    mul_1011: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_312, unsqueeze_1003);  sub_312 = unsqueeze_1003 = None
    mul_1012: "f32[16]" = torch.ops.aten.mul.Tensor(sum_125, squeeze_10);  sum_125 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_1011, relu_2, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1011 = primals_134 = None
    getitem_313: "f32[8, 16, 112, 112]" = convolution_backward_61[0]
    getitem_314: "f32[16, 16, 1, 1]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_168: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(relu_2);  relu_2 = None
    alias_169: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(alias_168);  alias_168 = None
    le_41: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(alias_169, 0);  alias_169 = None
    where_41: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_41, full_default, getitem_313);  le_41 = getitem_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_126: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_313: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_1006);  convolution_2 = unsqueeze_1006 = None
    mul_1013: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_41, sub_313)
    sum_127: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1013, [0, 2, 3]);  mul_1013 = None
    mul_1014: "f32[16]" = torch.ops.aten.mul.Tensor(sum_126, 9.964923469387754e-06)
    unsqueeze_1007: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1014, 0);  mul_1014 = None
    unsqueeze_1008: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1007, 2);  unsqueeze_1007 = None
    unsqueeze_1009: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, 3);  unsqueeze_1008 = None
    mul_1015: "f32[16]" = torch.ops.aten.mul.Tensor(sum_127, 9.964923469387754e-06)
    mul_1016: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_1017: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1015, mul_1016);  mul_1015 = mul_1016 = None
    unsqueeze_1010: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1017, 0);  mul_1017 = None
    unsqueeze_1011: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, 2);  unsqueeze_1010 = None
    unsqueeze_1012: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1011, 3);  unsqueeze_1011 = None
    mul_1018: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_1013: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1018, 0);  mul_1018 = None
    unsqueeze_1014: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1013, 2);  unsqueeze_1013 = None
    unsqueeze_1015: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, 3);  unsqueeze_1014 = None
    mul_1019: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_313, unsqueeze_1012);  sub_313 = unsqueeze_1012 = None
    sub_315: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_41, mul_1019);  where_41 = mul_1019 = None
    sub_316: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_315, unsqueeze_1009);  sub_315 = unsqueeze_1009 = None
    mul_1020: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_316, unsqueeze_1015);  sub_316 = unsqueeze_1015 = None
    mul_1021: "f32[16]" = torch.ops.aten.mul.Tensor(sum_127, squeeze_7);  sum_127 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_1020, relu_1, primals_133, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_1020 = primals_133 = None
    getitem_316: "f32[8, 16, 112, 112]" = convolution_backward_62[0]
    getitem_317: "f32[16, 1, 3, 3]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_171: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(relu_1);  relu_1 = None
    alias_172: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(alias_171);  alias_171 = None
    le_42: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(alias_172, 0);  alias_172 = None
    where_42: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_42, full_default, getitem_316);  le_42 = getitem_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_128: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_317: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_1018);  convolution_1 = unsqueeze_1018 = None
    mul_1022: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_42, sub_317)
    sum_129: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1022, [0, 2, 3]);  mul_1022 = None
    mul_1023: "f32[16]" = torch.ops.aten.mul.Tensor(sum_128, 9.964923469387754e-06)
    unsqueeze_1019: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1023, 0);  mul_1023 = None
    unsqueeze_1020: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1019, 2);  unsqueeze_1019 = None
    unsqueeze_1021: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, 3);  unsqueeze_1020 = None
    mul_1024: "f32[16]" = torch.ops.aten.mul.Tensor(sum_129, 9.964923469387754e-06)
    mul_1025: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_1026: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1024, mul_1025);  mul_1024 = mul_1025 = None
    unsqueeze_1022: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1026, 0);  mul_1026 = None
    unsqueeze_1023: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, 2);  unsqueeze_1022 = None
    unsqueeze_1024: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1023, 3);  unsqueeze_1023 = None
    mul_1027: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_1025: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1027, 0);  mul_1027 = None
    unsqueeze_1026: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1025, 2);  unsqueeze_1025 = None
    unsqueeze_1027: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, 3);  unsqueeze_1026 = None
    mul_1028: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_317, unsqueeze_1024);  sub_317 = unsqueeze_1024 = None
    sub_319: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_42, mul_1028);  where_42 = mul_1028 = None
    sub_320: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_319, unsqueeze_1021);  sub_319 = unsqueeze_1021 = None
    mul_1029: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_320, unsqueeze_1027);  sub_320 = unsqueeze_1027 = None
    mul_1030: "f32[16]" = torch.ops.aten.mul.Tensor(sum_129, squeeze_4);  sum_129 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_1029, relu, primals_132, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_1029 = primals_132 = None
    getitem_319: "f32[8, 16, 112, 112]" = convolution_backward_63[0]
    getitem_320: "f32[16, 16, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    add_354: "f32[8, 16, 112, 112]" = torch.ops.aten.add.Tensor(getitem_310, getitem_319);  getitem_310 = getitem_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    alias_174: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(relu);  relu = None
    alias_175: "f32[8, 16, 112, 112]" = torch.ops.aten.alias.default(alias_174);  alias_174 = None
    le_43: "b8[8, 16, 112, 112]" = torch.ops.aten.le.Scalar(alias_175, 0);  alias_175 = None
    where_43: "f32[8, 16, 112, 112]" = torch.ops.aten.where.self(le_43, full_default, add_354);  le_43 = full_default = add_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_130: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_321: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1030);  convolution = unsqueeze_1030 = None
    mul_1031: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_43, sub_321)
    sum_131: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_1031, [0, 2, 3]);  mul_1031 = None
    mul_1032: "f32[16]" = torch.ops.aten.mul.Tensor(sum_130, 9.964923469387754e-06)
    unsqueeze_1031: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1032, 0);  mul_1032 = None
    unsqueeze_1032: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1031, 2);  unsqueeze_1031 = None
    unsqueeze_1033: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, 3);  unsqueeze_1032 = None
    mul_1033: "f32[16]" = torch.ops.aten.mul.Tensor(sum_131, 9.964923469387754e-06)
    mul_1034: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_1035: "f32[16]" = torch.ops.aten.mul.Tensor(mul_1033, mul_1034);  mul_1033 = mul_1034 = None
    unsqueeze_1034: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1035, 0);  mul_1035 = None
    unsqueeze_1035: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, 2);  unsqueeze_1034 = None
    unsqueeze_1036: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1035, 3);  unsqueeze_1035 = None
    mul_1036: "f32[16]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_1037: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_1036, 0);  mul_1036 = None
    unsqueeze_1038: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1037, 2);  unsqueeze_1037 = None
    unsqueeze_1039: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, 3);  unsqueeze_1038 = None
    mul_1037: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_321, unsqueeze_1036);  sub_321 = unsqueeze_1036 = None
    sub_323: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(where_43, mul_1037);  where_43 = mul_1037 = None
    sub_324: "f32[8, 16, 112, 112]" = torch.ops.aten.sub.Tensor(sub_323, unsqueeze_1033);  sub_323 = unsqueeze_1033 = None
    mul_1038: "f32[8, 16, 112, 112]" = torch.ops.aten.mul.Tensor(sub_324, unsqueeze_1039);  sub_324 = unsqueeze_1039 = None
    mul_1039: "f32[16]" = torch.ops.aten.mul.Tensor(sum_131, squeeze_1);  sum_131 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_1038, primals_393, primals_131, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_1038 = primals_393 = primals_131 = None
    getitem_323: "f32[16, 3, 3, 3]" = convolution_backward_64[1];  convolution_backward_64 = None
    return [mul_1039, sum_130, mul_1030, sum_128, mul_1021, sum_126, mul_1012, sum_124, mul_1003, sum_122, mul_994, sum_120, mul_985, sum_118, mul_976, sum_116, mul_967, sum_114, mul_958, sum_112, mul_949, sum_110, mul_940, sum_108, mul_931, sum_106, mul_922, sum_104, mul_913, sum_102, mul_904, sum_100, mul_895, sum_98, mul_886, sum_96, mul_877, sum_94, mul_868, sum_92, mul_859, sum_90, mul_850, sum_88, mul_841, sum_86, mul_832, sum_84, mul_823, sum_82, mul_814, sum_80, mul_805, sum_78, mul_796, sum_76, mul_787, sum_74, mul_778, sum_72, mul_769, sum_70, mul_760, sum_68, mul_751, sum_66, mul_742, sum_64, mul_733, sum_62, mul_724, sum_60, mul_715, sum_58, mul_706, sum_56, mul_697, sum_54, mul_688, sum_52, mul_679, sum_50, mul_670, sum_48, mul_661, sum_46, mul_652, sum_44, mul_643, sum_42, mul_634, sum_40, mul_625, sum_38, mul_616, sum_36, mul_607, sum_34, mul_598, sum_32, mul_589, sum_30, mul_580, sum_28, mul_571, sum_26, mul_562, sum_24, mul_553, sum_22, mul_544, sum_20, mul_535, sum_18, mul_526, sum_16, mul_517, sum_14, mul_508, sum_12, mul_499, sum_10, mul_490, sum_8, mul_481, sum_6, mul_472, sum_4, mul_463, sum_2, getitem_323, getitem_320, getitem_317, getitem_314, getitem_311, getitem_308, getitem_305, getitem_302, getitem_299, getitem_296, getitem_293, getitem_290, getitem_287, getitem_284, getitem_281, getitem_278, getitem_275, getitem_272, getitem_269, getitem_266, getitem_263, getitem_260, getitem_257, getitem_254, getitem_251, getitem_248, getitem_245, getitem_242, getitem_239, getitem_236, getitem_233, getitem_230, getitem_227, getitem_224, getitem_221, getitem_218, getitem_215, getitem_212, getitem_209, getitem_206, getitem_203, getitem_200, getitem_197, getitem_194, getitem_191, getitem_188, getitem_185, getitem_182, getitem_179, getitem_176, getitem_173, getitem_170, getitem_167, getitem_164, getitem_161, getitem_158, getitem_155, getitem_152, getitem_149, getitem_146, getitem_143, getitem_140, getitem_137, getitem_134, getitem_131, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    