from __future__ import annotations



def forward(self, primals_1: "f32[32]", primals_3: "f32[24]", primals_5: "f32[24]", primals_7: "f32[24]", primals_9: "f32[24]", primals_11: "f32[56]", primals_13: "f32[56]", primals_15: "f32[56]", primals_17: "f32[56]", primals_19: "f32[152]", primals_21: "f32[152]", primals_23: "f32[152]", primals_25: "f32[152]", primals_27: "f32[152]", primals_29: "f32[152]", primals_31: "f32[152]", primals_33: "f32[152]", primals_35: "f32[152]", primals_37: "f32[152]", primals_39: "f32[152]", primals_41: "f32[152]", primals_43: "f32[152]", primals_45: "f32[368]", primals_47: "f32[368]", primals_49: "f32[368]", primals_51: "f32[368]", primals_53: "f32[368]", primals_55: "f32[368]", primals_57: "f32[368]", primals_59: "f32[368]", primals_61: "f32[368]", primals_63: "f32[368]", primals_65: "f32[368]", primals_67: "f32[368]", primals_69: "f32[368]", primals_71: "f32[368]", primals_73: "f32[368]", primals_75: "f32[368]", primals_77: "f32[368]", primals_79: "f32[368]", primals_81: "f32[368]", primals_83: "f32[368]", primals_85: "f32[368]", primals_87: "f32[368]", primals_89: "f32[32, 3, 3, 3]", primals_90: "f32[24, 32, 1, 1]", primals_91: "f32[24, 8, 3, 3]", primals_92: "f32[8, 24, 1, 1]", primals_94: "f32[24, 8, 1, 1]", primals_96: "f32[24, 24, 1, 1]", primals_97: "f32[24, 32, 1, 1]", primals_98: "f32[56, 24, 1, 1]", primals_99: "f32[56, 8, 3, 3]", primals_100: "f32[6, 56, 1, 1]", primals_102: "f32[56, 6, 1, 1]", primals_104: "f32[56, 56, 1, 1]", primals_105: "f32[56, 24, 1, 1]", primals_106: "f32[152, 56, 1, 1]", primals_107: "f32[152, 8, 3, 3]", primals_108: "f32[14, 152, 1, 1]", primals_110: "f32[152, 14, 1, 1]", primals_112: "f32[152, 152, 1, 1]", primals_113: "f32[152, 56, 1, 1]", primals_114: "f32[152, 152, 1, 1]", primals_115: "f32[152, 8, 3, 3]", primals_116: "f32[38, 152, 1, 1]", primals_118: "f32[152, 38, 1, 1]", primals_120: "f32[152, 152, 1, 1]", primals_121: "f32[152, 152, 1, 1]", primals_122: "f32[152, 8, 3, 3]", primals_123: "f32[38, 152, 1, 1]", primals_125: "f32[152, 38, 1, 1]", primals_127: "f32[152, 152, 1, 1]", primals_128: "f32[152, 152, 1, 1]", primals_129: "f32[152, 8, 3, 3]", primals_130: "f32[38, 152, 1, 1]", primals_132: "f32[152, 38, 1, 1]", primals_134: "f32[152, 152, 1, 1]", primals_135: "f32[368, 152, 1, 1]", primals_136: "f32[368, 8, 3, 3]", primals_137: "f32[38, 368, 1, 1]", primals_139: "f32[368, 38, 1, 1]", primals_141: "f32[368, 368, 1, 1]", primals_142: "f32[368, 152, 1, 1]", primals_143: "f32[368, 368, 1, 1]", primals_144: "f32[368, 8, 3, 3]", primals_145: "f32[92, 368, 1, 1]", primals_147: "f32[368, 92, 1, 1]", primals_149: "f32[368, 368, 1, 1]", primals_150: "f32[368, 368, 1, 1]", primals_151: "f32[368, 8, 3, 3]", primals_152: "f32[92, 368, 1, 1]", primals_154: "f32[368, 92, 1, 1]", primals_156: "f32[368, 368, 1, 1]", primals_157: "f32[368, 368, 1, 1]", primals_158: "f32[368, 8, 3, 3]", primals_159: "f32[92, 368, 1, 1]", primals_161: "f32[368, 92, 1, 1]", primals_163: "f32[368, 368, 1, 1]", primals_164: "f32[368, 368, 1, 1]", primals_165: "f32[368, 8, 3, 3]", primals_166: "f32[92, 368, 1, 1]", primals_168: "f32[368, 92, 1, 1]", primals_170: "f32[368, 368, 1, 1]", primals_171: "f32[368, 368, 1, 1]", primals_172: "f32[368, 8, 3, 3]", primals_173: "f32[92, 368, 1, 1]", primals_175: "f32[368, 92, 1, 1]", primals_177: "f32[368, 368, 1, 1]", primals_178: "f32[368, 368, 1, 1]", primals_179: "f32[368, 8, 3, 3]", primals_180: "f32[92, 368, 1, 1]", primals_182: "f32[368, 92, 1, 1]", primals_184: "f32[368, 368, 1, 1]", primals_319: "f32[8, 3, 224, 224]", convolution: "f32[8, 32, 112, 112]", squeeze_1: "f32[32]", relu: "f32[8, 32, 112, 112]", convolution_1: "f32[8, 24, 112, 112]", squeeze_4: "f32[24]", relu_1: "f32[8, 24, 112, 112]", convolution_2: "f32[8, 24, 56, 56]", squeeze_7: "f32[24]", relu_2: "f32[8, 24, 56, 56]", mean: "f32[8, 24, 1, 1]", relu_3: "f32[8, 8, 1, 1]", convolution_4: "f32[8, 24, 1, 1]", mul_21: "f32[8, 24, 56, 56]", convolution_5: "f32[8, 24, 56, 56]", squeeze_10: "f32[24]", convolution_6: "f32[8, 24, 56, 56]", squeeze_13: "f32[24]", relu_4: "f32[8, 24, 56, 56]", convolution_7: "f32[8, 56, 56, 56]", squeeze_16: "f32[56]", relu_5: "f32[8, 56, 56, 56]", convolution_8: "f32[8, 56, 28, 28]", squeeze_19: "f32[56]", relu_6: "f32[8, 56, 28, 28]", mean_1: "f32[8, 56, 1, 1]", relu_7: "f32[8, 6, 1, 1]", convolution_10: "f32[8, 56, 1, 1]", mul_50: "f32[8, 56, 28, 28]", convolution_11: "f32[8, 56, 28, 28]", squeeze_22: "f32[56]", convolution_12: "f32[8, 56, 28, 28]", squeeze_25: "f32[56]", relu_8: "f32[8, 56, 28, 28]", convolution_13: "f32[8, 152, 28, 28]", squeeze_28: "f32[152]", relu_9: "f32[8, 152, 28, 28]", convolution_14: "f32[8, 152, 14, 14]", squeeze_31: "f32[152]", relu_10: "f32[8, 152, 14, 14]", mean_2: "f32[8, 152, 1, 1]", relu_11: "f32[8, 14, 1, 1]", convolution_16: "f32[8, 152, 1, 1]", mul_79: "f32[8, 152, 14, 14]", convolution_17: "f32[8, 152, 14, 14]", squeeze_34: "f32[152]", convolution_18: "f32[8, 152, 14, 14]", squeeze_37: "f32[152]", relu_12: "f32[8, 152, 14, 14]", convolution_19: "f32[8, 152, 14, 14]", squeeze_40: "f32[152]", relu_13: "f32[8, 152, 14, 14]", convolution_20: "f32[8, 152, 14, 14]", squeeze_43: "f32[152]", relu_14: "f32[8, 152, 14, 14]", mean_3: "f32[8, 152, 1, 1]", relu_15: "f32[8, 38, 1, 1]", convolution_22: "f32[8, 152, 1, 1]", mul_108: "f32[8, 152, 14, 14]", convolution_23: "f32[8, 152, 14, 14]", squeeze_46: "f32[152]", relu_16: "f32[8, 152, 14, 14]", convolution_24: "f32[8, 152, 14, 14]", squeeze_49: "f32[152]", relu_17: "f32[8, 152, 14, 14]", convolution_25: "f32[8, 152, 14, 14]", squeeze_52: "f32[152]", relu_18: "f32[8, 152, 14, 14]", mean_4: "f32[8, 152, 1, 1]", relu_19: "f32[8, 38, 1, 1]", convolution_27: "f32[8, 152, 1, 1]", mul_130: "f32[8, 152, 14, 14]", convolution_28: "f32[8, 152, 14, 14]", squeeze_55: "f32[152]", relu_20: "f32[8, 152, 14, 14]", convolution_29: "f32[8, 152, 14, 14]", squeeze_58: "f32[152]", relu_21: "f32[8, 152, 14, 14]", convolution_30: "f32[8, 152, 14, 14]", squeeze_61: "f32[152]", relu_22: "f32[8, 152, 14, 14]", mean_5: "f32[8, 152, 1, 1]", relu_23: "f32[8, 38, 1, 1]", convolution_32: "f32[8, 152, 1, 1]", mul_152: "f32[8, 152, 14, 14]", convolution_33: "f32[8, 152, 14, 14]", squeeze_64: "f32[152]", relu_24: "f32[8, 152, 14, 14]", convolution_34: "f32[8, 368, 14, 14]", squeeze_67: "f32[368]", relu_25: "f32[8, 368, 14, 14]", convolution_35: "f32[8, 368, 7, 7]", squeeze_70: "f32[368]", relu_26: "f32[8, 368, 7, 7]", mean_6: "f32[8, 368, 1, 1]", relu_27: "f32[8, 38, 1, 1]", convolution_37: "f32[8, 368, 1, 1]", mul_174: "f32[8, 368, 7, 7]", convolution_38: "f32[8, 368, 7, 7]", squeeze_73: "f32[368]", convolution_39: "f32[8, 368, 7, 7]", squeeze_76: "f32[368]", relu_28: "f32[8, 368, 7, 7]", convolution_40: "f32[8, 368, 7, 7]", squeeze_79: "f32[368]", relu_29: "f32[8, 368, 7, 7]", convolution_41: "f32[8, 368, 7, 7]", squeeze_82: "f32[368]", relu_30: "f32[8, 368, 7, 7]", mean_7: "f32[8, 368, 1, 1]", relu_31: "f32[8, 92, 1, 1]", convolution_43: "f32[8, 368, 1, 1]", mul_203: "f32[8, 368, 7, 7]", convolution_44: "f32[8, 368, 7, 7]", squeeze_85: "f32[368]", relu_32: "f32[8, 368, 7, 7]", convolution_45: "f32[8, 368, 7, 7]", squeeze_88: "f32[368]", relu_33: "f32[8, 368, 7, 7]", convolution_46: "f32[8, 368, 7, 7]", squeeze_91: "f32[368]", relu_34: "f32[8, 368, 7, 7]", mean_8: "f32[8, 368, 1, 1]", relu_35: "f32[8, 92, 1, 1]", convolution_48: "f32[8, 368, 1, 1]", mul_225: "f32[8, 368, 7, 7]", convolution_49: "f32[8, 368, 7, 7]", squeeze_94: "f32[368]", relu_36: "f32[8, 368, 7, 7]", convolution_50: "f32[8, 368, 7, 7]", squeeze_97: "f32[368]", relu_37: "f32[8, 368, 7, 7]", convolution_51: "f32[8, 368, 7, 7]", squeeze_100: "f32[368]", relu_38: "f32[8, 368, 7, 7]", mean_9: "f32[8, 368, 1, 1]", relu_39: "f32[8, 92, 1, 1]", convolution_53: "f32[8, 368, 1, 1]", mul_247: "f32[8, 368, 7, 7]", convolution_54: "f32[8, 368, 7, 7]", squeeze_103: "f32[368]", relu_40: "f32[8, 368, 7, 7]", convolution_55: "f32[8, 368, 7, 7]", squeeze_106: "f32[368]", relu_41: "f32[8, 368, 7, 7]", convolution_56: "f32[8, 368, 7, 7]", squeeze_109: "f32[368]", relu_42: "f32[8, 368, 7, 7]", mean_10: "f32[8, 368, 1, 1]", relu_43: "f32[8, 92, 1, 1]", convolution_58: "f32[8, 368, 1, 1]", mul_269: "f32[8, 368, 7, 7]", convolution_59: "f32[8, 368, 7, 7]", squeeze_112: "f32[368]", relu_44: "f32[8, 368, 7, 7]", convolution_60: "f32[8, 368, 7, 7]", squeeze_115: "f32[368]", relu_45: "f32[8, 368, 7, 7]", convolution_61: "f32[8, 368, 7, 7]", squeeze_118: "f32[368]", relu_46: "f32[8, 368, 7, 7]", mean_11: "f32[8, 368, 1, 1]", relu_47: "f32[8, 92, 1, 1]", convolution_63: "f32[8, 368, 1, 1]", mul_291: "f32[8, 368, 7, 7]", convolution_64: "f32[8, 368, 7, 7]", squeeze_121: "f32[368]", relu_48: "f32[8, 368, 7, 7]", convolution_65: "f32[8, 368, 7, 7]", squeeze_124: "f32[368]", relu_49: "f32[8, 368, 7, 7]", convolution_66: "f32[8, 368, 7, 7]", squeeze_127: "f32[368]", relu_50: "f32[8, 368, 7, 7]", mean_12: "f32[8, 368, 1, 1]", relu_51: "f32[8, 92, 1, 1]", convolution_68: "f32[8, 368, 1, 1]", mul_313: "f32[8, 368, 7, 7]", convolution_69: "f32[8, 368, 7, 7]", squeeze_130: "f32[368]", clone: "f32[8, 368]", permute_1: "f32[1000, 368]", le: "b8[8, 368, 7, 7]", unsqueeze_178: "f32[1, 368, 1, 1]", unsqueeze_190: "f32[1, 368, 1, 1]", unsqueeze_202: "f32[1, 368, 1, 1]", unsqueeze_214: "f32[1, 368, 1, 1]", unsqueeze_226: "f32[1, 368, 1, 1]", unsqueeze_238: "f32[1, 368, 1, 1]", unsqueeze_250: "f32[1, 368, 1, 1]", unsqueeze_262: "f32[1, 368, 1, 1]", unsqueeze_274: "f32[1, 368, 1, 1]", unsqueeze_286: "f32[1, 368, 1, 1]", unsqueeze_298: "f32[1, 368, 1, 1]", unsqueeze_310: "f32[1, 368, 1, 1]", unsqueeze_322: "f32[1, 368, 1, 1]", unsqueeze_334: "f32[1, 368, 1, 1]", unsqueeze_346: "f32[1, 368, 1, 1]", unsqueeze_358: "f32[1, 368, 1, 1]", unsqueeze_370: "f32[1, 368, 1, 1]", unsqueeze_382: "f32[1, 368, 1, 1]", unsqueeze_394: "f32[1, 368, 1, 1]", unsqueeze_406: "f32[1, 368, 1, 1]", unsqueeze_418: "f32[1, 368, 1, 1]", unsqueeze_430: "f32[1, 368, 1, 1]", unsqueeze_442: "f32[1, 152, 1, 1]", unsqueeze_454: "f32[1, 152, 1, 1]", unsqueeze_466: "f32[1, 152, 1, 1]", unsqueeze_478: "f32[1, 152, 1, 1]", unsqueeze_490: "f32[1, 152, 1, 1]", unsqueeze_502: "f32[1, 152, 1, 1]", unsqueeze_514: "f32[1, 152, 1, 1]", unsqueeze_526: "f32[1, 152, 1, 1]", unsqueeze_538: "f32[1, 152, 1, 1]", unsqueeze_550: "f32[1, 152, 1, 1]", unsqueeze_562: "f32[1, 152, 1, 1]", unsqueeze_574: "f32[1, 152, 1, 1]", unsqueeze_586: "f32[1, 152, 1, 1]", unsqueeze_598: "f32[1, 56, 1, 1]", unsqueeze_610: "f32[1, 56, 1, 1]", unsqueeze_622: "f32[1, 56, 1, 1]", unsqueeze_634: "f32[1, 56, 1, 1]", unsqueeze_646: "f32[1, 24, 1, 1]", unsqueeze_658: "f32[1, 24, 1, 1]", unsqueeze_670: "f32[1, 24, 1, 1]", unsqueeze_682: "f32[1, 24, 1, 1]", unsqueeze_694: "f32[1, 32, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid: "f32[8, 24, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_4);  convolution_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_1: "f32[8, 56, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_10);  convolution_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_2: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_16);  convolution_16 = None
    sigmoid_3: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_22);  convolution_22 = None
    sigmoid_4: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_27);  convolution_27 = None
    sigmoid_5: "f32[8, 152, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_32);  convolution_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sigmoid_6: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_37);  convolution_37 = None
    sigmoid_7: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_43);  convolution_43 = None
    sigmoid_8: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_48);  convolution_48 = None
    sigmoid_9: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_53);  convolution_53 = None
    sigmoid_10: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_58);  convolution_58 = None
    sigmoid_11: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_63);  convolution_63 = None
    sigmoid_12: "f32[8, 368, 1, 1]" = torch.ops.aten.sigmoid.default(convolution_68);  convolution_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/classifier.py:134, code: x = self.fc(x)
    mm: "f32[8, 368]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 368]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[368, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 368]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 368, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 368, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(view_2, [8, 368, 7, 7]);  view_2 = None
    div: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_2: "f32[368]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_44: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_178);  convolution_69 = unsqueeze_178 = None
    mul_321: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where, sub_44)
    sum_3: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_321, [0, 2, 3]);  mul_321 = None
    mul_322: "f32[368]" = torch.ops.aten.mul.Tensor(sum_2, 0.002551020408163265)
    unsqueeze_179: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_322, 0);  mul_322 = None
    unsqueeze_180: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    unsqueeze_181: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
    mul_323: "f32[368]" = torch.ops.aten.mul.Tensor(sum_3, 0.002551020408163265)
    mul_324: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_130, squeeze_130)
    mul_325: "f32[368]" = torch.ops.aten.mul.Tensor(mul_323, mul_324);  mul_323 = mul_324 = None
    unsqueeze_182: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_325, 0);  mul_325 = None
    unsqueeze_183: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
    unsqueeze_184: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
    mul_326: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_130, primals_87);  primals_87 = None
    unsqueeze_185: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_326, 0);  mul_326 = None
    unsqueeze_186: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    unsqueeze_187: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
    mul_327: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_184);  sub_44 = unsqueeze_184 = None
    sub_46: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where, mul_327);  mul_327 = None
    sub_47: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_46, unsqueeze_181);  sub_46 = unsqueeze_181 = None
    mul_328: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_187);  sub_47 = unsqueeze_187 = None
    mul_329: "f32[368]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_130);  sum_3 = squeeze_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_328, mul_313, primals_184, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_328 = mul_313 = primals_184 = None
    getitem_88: "f32[8, 368, 7, 7]" = convolution_backward[0]
    getitem_89: "f32[368, 368, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_330: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_88, relu_50)
    mul_331: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_88, sigmoid_12);  getitem_88 = None
    sum_4: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_330, [2, 3], True);  mul_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_48: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_12)
    mul_332: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_12, sub_48);  sigmoid_12 = sub_48 = None
    mul_333: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_4, mul_332);  sum_4 = mul_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_5: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_333, [0, 2, 3])
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_333, relu_51, primals_182, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_333 = primals_182 = None
    getitem_91: "f32[8, 92, 1, 1]" = convolution_backward_1[0]
    getitem_92: "f32[368, 92, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_1: "b8[8, 92, 1, 1]" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
    where_1: "f32[8, 92, 1, 1]" = torch.ops.aten.where.self(le_1, full_default, getitem_91);  le_1 = getitem_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_6: "f32[92]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_1, mean_12, primals_180, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_1 = mean_12 = primals_180 = None
    getitem_94: "f32[8, 368, 1, 1]" = convolution_backward_2[0]
    getitem_95: "f32[92, 368, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_1: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_94, [8, 368, 7, 7]);  getitem_94 = None
    div_1: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_233: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_331, div_1);  mul_331 = div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_2: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
    where_2: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, add_233);  le_2 = add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_7: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_49: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_190);  convolution_66 = unsqueeze_190 = None
    mul_334: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_49)
    sum_8: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 2, 3]);  mul_334 = None
    mul_335: "f32[368]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    unsqueeze_191: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_335, 0);  mul_335 = None
    unsqueeze_192: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    unsqueeze_193: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
    mul_336: "f32[368]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    mul_337: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_127, squeeze_127)
    mul_338: "f32[368]" = torch.ops.aten.mul.Tensor(mul_336, mul_337);  mul_336 = mul_337 = None
    unsqueeze_194: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_338, 0);  mul_338 = None
    unsqueeze_195: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
    unsqueeze_196: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
    mul_339: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_127, primals_85);  primals_85 = None
    unsqueeze_197: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_339, 0);  mul_339 = None
    unsqueeze_198: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    unsqueeze_199: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
    mul_340: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_196);  sub_49 = unsqueeze_196 = None
    sub_51: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_340);  where_2 = mul_340 = None
    sub_52: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_51, unsqueeze_193);  sub_51 = unsqueeze_193 = None
    mul_341: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_199);  sub_52 = unsqueeze_199 = None
    mul_342: "f32[368]" = torch.ops.aten.mul.Tensor(sum_8, squeeze_127);  sum_8 = squeeze_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_341, relu_49, primals_179, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_341 = primals_179 = None
    getitem_97: "f32[8, 368, 7, 7]" = convolution_backward_3[0]
    getitem_98: "f32[368, 8, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_3: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
    where_3: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, getitem_97);  le_3 = getitem_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_9: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_53: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_202);  convolution_65 = unsqueeze_202 = None
    mul_343: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_53)
    sum_10: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 2, 3]);  mul_343 = None
    mul_344: "f32[368]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    unsqueeze_203: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_344, 0);  mul_344 = None
    unsqueeze_204: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    unsqueeze_205: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
    mul_345: "f32[368]" = torch.ops.aten.mul.Tensor(sum_10, 0.002551020408163265)
    mul_346: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_124, squeeze_124)
    mul_347: "f32[368]" = torch.ops.aten.mul.Tensor(mul_345, mul_346);  mul_345 = mul_346 = None
    unsqueeze_206: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_347, 0);  mul_347 = None
    unsqueeze_207: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
    unsqueeze_208: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
    mul_348: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_124, primals_83);  primals_83 = None
    unsqueeze_209: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_348, 0);  mul_348 = None
    unsqueeze_210: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    unsqueeze_211: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
    mul_349: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_208);  sub_53 = unsqueeze_208 = None
    sub_55: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_349);  where_3 = mul_349 = None
    sub_56: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_55, unsqueeze_205);  sub_55 = unsqueeze_205 = None
    mul_350: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_211);  sub_56 = unsqueeze_211 = None
    mul_351: "f32[368]" = torch.ops.aten.mul.Tensor(sum_10, squeeze_124);  sum_10 = squeeze_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_350, relu_48, primals_178, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_350 = primals_178 = None
    getitem_100: "f32[8, 368, 7, 7]" = convolution_backward_4[0]
    getitem_101: "f32[368, 368, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_234: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(where, getitem_100);  where = getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_4: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
    where_4: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_4, full_default, add_234);  le_4 = add_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_11: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_57: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_214);  convolution_64 = unsqueeze_214 = None
    mul_352: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_4, sub_57)
    sum_12: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_352, [0, 2, 3]);  mul_352 = None
    mul_353: "f32[368]" = torch.ops.aten.mul.Tensor(sum_11, 0.002551020408163265)
    unsqueeze_215: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
    unsqueeze_216: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    unsqueeze_217: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
    mul_354: "f32[368]" = torch.ops.aten.mul.Tensor(sum_12, 0.002551020408163265)
    mul_355: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_356: "f32[368]" = torch.ops.aten.mul.Tensor(mul_354, mul_355);  mul_354 = mul_355 = None
    unsqueeze_218: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_356, 0);  mul_356 = None
    unsqueeze_219: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
    unsqueeze_220: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
    mul_357: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_81);  primals_81 = None
    unsqueeze_221: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_357, 0);  mul_357 = None
    unsqueeze_222: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    unsqueeze_223: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
    mul_358: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_220);  sub_57 = unsqueeze_220 = None
    sub_59: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_4, mul_358);  mul_358 = None
    sub_60: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_59, unsqueeze_217);  sub_59 = unsqueeze_217 = None
    mul_359: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_223);  sub_60 = unsqueeze_223 = None
    mul_360: "f32[368]" = torch.ops.aten.mul.Tensor(sum_12, squeeze_121);  sum_12 = squeeze_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_359, mul_291, primals_177, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_359 = mul_291 = primals_177 = None
    getitem_103: "f32[8, 368, 7, 7]" = convolution_backward_5[0]
    getitem_104: "f32[368, 368, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_361: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_103, relu_46)
    mul_362: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_103, sigmoid_11);  getitem_103 = None
    sum_13: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_361, [2, 3], True);  mul_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_61: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_11)
    mul_363: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_11, sub_61);  sigmoid_11 = sub_61 = None
    mul_364: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_13, mul_363);  sum_13 = mul_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_14: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_364, [0, 2, 3])
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_364, relu_47, primals_175, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_364 = primals_175 = None
    getitem_106: "f32[8, 92, 1, 1]" = convolution_backward_6[0]
    getitem_107: "f32[368, 92, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_5: "b8[8, 92, 1, 1]" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
    where_5: "f32[8, 92, 1, 1]" = torch.ops.aten.where.self(le_5, full_default, getitem_106);  le_5 = getitem_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_15: "f32[92]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_5, mean_11, primals_173, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_11 = primals_173 = None
    getitem_109: "f32[8, 368, 1, 1]" = convolution_backward_7[0]
    getitem_110: "f32[92, 368, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_2: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_109, [8, 368, 7, 7]);  getitem_109 = None
    div_2: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_235: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_362, div_2);  mul_362 = div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_6: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
    where_6: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_6, full_default, add_235);  le_6 = add_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_16: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_62: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_226);  convolution_61 = unsqueeze_226 = None
    mul_365: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_6, sub_62)
    sum_17: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_365, [0, 2, 3]);  mul_365 = None
    mul_366: "f32[368]" = torch.ops.aten.mul.Tensor(sum_16, 0.002551020408163265)
    unsqueeze_227: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_366, 0);  mul_366 = None
    unsqueeze_228: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    unsqueeze_229: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
    mul_367: "f32[368]" = torch.ops.aten.mul.Tensor(sum_17, 0.002551020408163265)
    mul_368: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_369: "f32[368]" = torch.ops.aten.mul.Tensor(mul_367, mul_368);  mul_367 = mul_368 = None
    unsqueeze_230: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
    unsqueeze_231: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
    unsqueeze_232: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
    mul_370: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_79);  primals_79 = None
    unsqueeze_233: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_370, 0);  mul_370 = None
    unsqueeze_234: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    unsqueeze_235: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
    mul_371: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_232);  sub_62 = unsqueeze_232 = None
    sub_64: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_6, mul_371);  where_6 = mul_371 = None
    sub_65: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_64, unsqueeze_229);  sub_64 = unsqueeze_229 = None
    mul_372: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_235);  sub_65 = unsqueeze_235 = None
    mul_373: "f32[368]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_118);  sum_17 = squeeze_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_372, relu_45, primals_172, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_372 = primals_172 = None
    getitem_112: "f32[8, 368, 7, 7]" = convolution_backward_8[0]
    getitem_113: "f32[368, 8, 3, 3]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_7: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
    where_7: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_7, full_default, getitem_112);  le_7 = getitem_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_18: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_66: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_238);  convolution_60 = unsqueeze_238 = None
    mul_374: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_66)
    sum_19: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 2, 3]);  mul_374 = None
    mul_375: "f32[368]" = torch.ops.aten.mul.Tensor(sum_18, 0.002551020408163265)
    unsqueeze_239: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_375, 0);  mul_375 = None
    unsqueeze_240: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    unsqueeze_241: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
    mul_376: "f32[368]" = torch.ops.aten.mul.Tensor(sum_19, 0.002551020408163265)
    mul_377: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_378: "f32[368]" = torch.ops.aten.mul.Tensor(mul_376, mul_377);  mul_376 = mul_377 = None
    unsqueeze_242: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_243: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
    unsqueeze_244: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
    mul_379: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_77);  primals_77 = None
    unsqueeze_245: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
    unsqueeze_246: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    unsqueeze_247: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
    mul_380: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_244);  sub_66 = unsqueeze_244 = None
    sub_68: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_7, mul_380);  where_7 = mul_380 = None
    sub_69: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_68, unsqueeze_241);  sub_68 = unsqueeze_241 = None
    mul_381: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_247);  sub_69 = unsqueeze_247 = None
    mul_382: "f32[368]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_115);  sum_19 = squeeze_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_381, relu_44, primals_171, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_381 = primals_171 = None
    getitem_115: "f32[8, 368, 7, 7]" = convolution_backward_9[0]
    getitem_116: "f32[368, 368, 1, 1]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_236: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(where_4, getitem_115);  where_4 = getitem_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_8: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
    where_8: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_8, full_default, add_236);  le_8 = add_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_20: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_70: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_250);  convolution_59 = unsqueeze_250 = None
    mul_383: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_8, sub_70)
    sum_21: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 2, 3]);  mul_383 = None
    mul_384: "f32[368]" = torch.ops.aten.mul.Tensor(sum_20, 0.002551020408163265)
    unsqueeze_251: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_384, 0);  mul_384 = None
    unsqueeze_252: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    unsqueeze_253: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
    mul_385: "f32[368]" = torch.ops.aten.mul.Tensor(sum_21, 0.002551020408163265)
    mul_386: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_387: "f32[368]" = torch.ops.aten.mul.Tensor(mul_385, mul_386);  mul_385 = mul_386 = None
    unsqueeze_254: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
    unsqueeze_255: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
    unsqueeze_256: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
    mul_388: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_75);  primals_75 = None
    unsqueeze_257: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_388, 0);  mul_388 = None
    unsqueeze_258: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    mul_389: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_256);  sub_70 = unsqueeze_256 = None
    sub_72: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_8, mul_389);  mul_389 = None
    sub_73: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_72, unsqueeze_253);  sub_72 = unsqueeze_253 = None
    mul_390: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_259);  sub_73 = unsqueeze_259 = None
    mul_391: "f32[368]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_112);  sum_21 = squeeze_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_390, mul_269, primals_170, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_390 = mul_269 = primals_170 = None
    getitem_118: "f32[8, 368, 7, 7]" = convolution_backward_10[0]
    getitem_119: "f32[368, 368, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_392: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_118, relu_42)
    mul_393: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_118, sigmoid_10);  getitem_118 = None
    sum_22: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_392, [2, 3], True);  mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_74: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_10)
    mul_394: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_10, sub_74);  sigmoid_10 = sub_74 = None
    mul_395: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_22, mul_394);  sum_22 = mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_23: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 2, 3])
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_395, relu_43, primals_168, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_395 = primals_168 = None
    getitem_121: "f32[8, 92, 1, 1]" = convolution_backward_11[0]
    getitem_122: "f32[368, 92, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_9: "b8[8, 92, 1, 1]" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
    where_9: "f32[8, 92, 1, 1]" = torch.ops.aten.where.self(le_9, full_default, getitem_121);  le_9 = getitem_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_24: "f32[92]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_9, mean_10, primals_166, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_9 = mean_10 = primals_166 = None
    getitem_124: "f32[8, 368, 1, 1]" = convolution_backward_12[0]
    getitem_125: "f32[92, 368, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_3: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_124, [8, 368, 7, 7]);  getitem_124 = None
    div_3: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_237: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_393, div_3);  mul_393 = div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_10: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
    where_10: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_10, full_default, add_237);  le_10 = add_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_25: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_75: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_262);  convolution_56 = unsqueeze_262 = None
    mul_396: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_10, sub_75)
    sum_26: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_396, [0, 2, 3]);  mul_396 = None
    mul_397: "f32[368]" = torch.ops.aten.mul.Tensor(sum_25, 0.002551020408163265)
    unsqueeze_263: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
    unsqueeze_264: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_398: "f32[368]" = torch.ops.aten.mul.Tensor(sum_26, 0.002551020408163265)
    mul_399: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_400: "f32[368]" = torch.ops.aten.mul.Tensor(mul_398, mul_399);  mul_398 = mul_399 = None
    unsqueeze_266: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_400, 0);  mul_400 = None
    unsqueeze_267: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_401: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_73);  primals_73 = None
    unsqueeze_269: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_401, 0);  mul_401 = None
    unsqueeze_270: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    mul_402: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_268);  sub_75 = unsqueeze_268 = None
    sub_77: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_10, mul_402);  where_10 = mul_402 = None
    sub_78: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_77, unsqueeze_265);  sub_77 = unsqueeze_265 = None
    mul_403: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_271);  sub_78 = unsqueeze_271 = None
    mul_404: "f32[368]" = torch.ops.aten.mul.Tensor(sum_26, squeeze_109);  sum_26 = squeeze_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_403, relu_41, primals_165, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_403 = primals_165 = None
    getitem_127: "f32[8, 368, 7, 7]" = convolution_backward_13[0]
    getitem_128: "f32[368, 8, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_11: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
    where_11: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_11, full_default, getitem_127);  le_11 = getitem_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_27: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_79: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_274);  convolution_55 = unsqueeze_274 = None
    mul_405: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_11, sub_79)
    sum_28: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_405, [0, 2, 3]);  mul_405 = None
    mul_406: "f32[368]" = torch.ops.aten.mul.Tensor(sum_27, 0.002551020408163265)
    unsqueeze_275: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_406, 0);  mul_406 = None
    unsqueeze_276: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_407: "f32[368]" = torch.ops.aten.mul.Tensor(sum_28, 0.002551020408163265)
    mul_408: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_409: "f32[368]" = torch.ops.aten.mul.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
    unsqueeze_278: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
    unsqueeze_279: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_410: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_71);  primals_71 = None
    unsqueeze_281: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_410, 0);  mul_410 = None
    unsqueeze_282: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    mul_411: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_280);  sub_79 = unsqueeze_280 = None
    sub_81: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_11, mul_411);  where_11 = mul_411 = None
    sub_82: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_81, unsqueeze_277);  sub_81 = unsqueeze_277 = None
    mul_412: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_283);  sub_82 = unsqueeze_283 = None
    mul_413: "f32[368]" = torch.ops.aten.mul.Tensor(sum_28, squeeze_106);  sum_28 = squeeze_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_412, relu_40, primals_164, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_412 = primals_164 = None
    getitem_130: "f32[8, 368, 7, 7]" = convolution_backward_14[0]
    getitem_131: "f32[368, 368, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_238: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(where_8, getitem_130);  where_8 = getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_12: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
    where_12: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_12, full_default, add_238);  le_12 = add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_29: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_83: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_286);  convolution_54 = unsqueeze_286 = None
    mul_414: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_12, sub_83)
    sum_30: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_414, [0, 2, 3]);  mul_414 = None
    mul_415: "f32[368]" = torch.ops.aten.mul.Tensor(sum_29, 0.002551020408163265)
    unsqueeze_287: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_415, 0);  mul_415 = None
    unsqueeze_288: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_416: "f32[368]" = torch.ops.aten.mul.Tensor(sum_30, 0.002551020408163265)
    mul_417: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_418: "f32[368]" = torch.ops.aten.mul.Tensor(mul_416, mul_417);  mul_416 = mul_417 = None
    unsqueeze_290: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
    unsqueeze_291: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_419: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_69);  primals_69 = None
    unsqueeze_293: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_419, 0);  mul_419 = None
    unsqueeze_294: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    mul_420: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_292);  sub_83 = unsqueeze_292 = None
    sub_85: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_12, mul_420);  mul_420 = None
    sub_86: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_85, unsqueeze_289);  sub_85 = unsqueeze_289 = None
    mul_421: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_295);  sub_86 = unsqueeze_295 = None
    mul_422: "f32[368]" = torch.ops.aten.mul.Tensor(sum_30, squeeze_103);  sum_30 = squeeze_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_421, mul_247, primals_163, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_421 = mul_247 = primals_163 = None
    getitem_133: "f32[8, 368, 7, 7]" = convolution_backward_15[0]
    getitem_134: "f32[368, 368, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_423: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_133, relu_38)
    mul_424: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_133, sigmoid_9);  getitem_133 = None
    sum_31: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_423, [2, 3], True);  mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_87: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_9)
    mul_425: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_9, sub_87);  sigmoid_9 = sub_87 = None
    mul_426: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_31, mul_425);  sum_31 = mul_425 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_32: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 2, 3])
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_426, relu_39, primals_161, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_426 = primals_161 = None
    getitem_136: "f32[8, 92, 1, 1]" = convolution_backward_16[0]
    getitem_137: "f32[368, 92, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_13: "b8[8, 92, 1, 1]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
    where_13: "f32[8, 92, 1, 1]" = torch.ops.aten.where.self(le_13, full_default, getitem_136);  le_13 = getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_33: "f32[92]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(where_13, mean_9, primals_159, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_13 = mean_9 = primals_159 = None
    getitem_139: "f32[8, 368, 1, 1]" = convolution_backward_17[0]
    getitem_140: "f32[92, 368, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_4: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_139, [8, 368, 7, 7]);  getitem_139 = None
    div_4: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_4, 49);  expand_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_239: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_424, div_4);  mul_424 = div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_14: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
    where_14: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_14, full_default, add_239);  le_14 = add_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_34: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_88: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_298);  convolution_51 = unsqueeze_298 = None
    mul_427: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_14, sub_88)
    sum_35: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_427, [0, 2, 3]);  mul_427 = None
    mul_428: "f32[368]" = torch.ops.aten.mul.Tensor(sum_34, 0.002551020408163265)
    unsqueeze_299: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_428, 0);  mul_428 = None
    unsqueeze_300: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_429: "f32[368]" = torch.ops.aten.mul.Tensor(sum_35, 0.002551020408163265)
    mul_430: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_431: "f32[368]" = torch.ops.aten.mul.Tensor(mul_429, mul_430);  mul_429 = mul_430 = None
    unsqueeze_302: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_303: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_432: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_67);  primals_67 = None
    unsqueeze_305: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
    unsqueeze_306: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    mul_433: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_304);  sub_88 = unsqueeze_304 = None
    sub_90: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_14, mul_433);  where_14 = mul_433 = None
    sub_91: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_90, unsqueeze_301);  sub_90 = unsqueeze_301 = None
    mul_434: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_307);  sub_91 = unsqueeze_307 = None
    mul_435: "f32[368]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_100);  sum_35 = squeeze_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_434, relu_37, primals_158, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_434 = primals_158 = None
    getitem_142: "f32[8, 368, 7, 7]" = convolution_backward_18[0]
    getitem_143: "f32[368, 8, 3, 3]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_15: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
    where_15: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_15, full_default, getitem_142);  le_15 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_36: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_92: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_310);  convolution_50 = unsqueeze_310 = None
    mul_436: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, sub_92)
    sum_37: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_436, [0, 2, 3]);  mul_436 = None
    mul_437: "f32[368]" = torch.ops.aten.mul.Tensor(sum_36, 0.002551020408163265)
    unsqueeze_311: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_437, 0);  mul_437 = None
    unsqueeze_312: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_438: "f32[368]" = torch.ops.aten.mul.Tensor(sum_37, 0.002551020408163265)
    mul_439: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_440: "f32[368]" = torch.ops.aten.mul.Tensor(mul_438, mul_439);  mul_438 = mul_439 = None
    unsqueeze_314: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_440, 0);  mul_440 = None
    unsqueeze_315: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_441: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_65);  primals_65 = None
    unsqueeze_317: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_318: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    mul_442: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_316);  sub_92 = unsqueeze_316 = None
    sub_94: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_15, mul_442);  where_15 = mul_442 = None
    sub_95: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_94, unsqueeze_313);  sub_94 = unsqueeze_313 = None
    mul_443: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_319);  sub_95 = unsqueeze_319 = None
    mul_444: "f32[368]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_97);  sum_37 = squeeze_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_443, relu_36, primals_157, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_443 = primals_157 = None
    getitem_145: "f32[8, 368, 7, 7]" = convolution_backward_19[0]
    getitem_146: "f32[368, 368, 1, 1]" = convolution_backward_19[1];  convolution_backward_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_240: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(where_12, getitem_145);  where_12 = getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_16: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
    where_16: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_16, full_default, add_240);  le_16 = add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_38: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_96: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_322);  convolution_49 = unsqueeze_322 = None
    mul_445: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_16, sub_96)
    sum_39: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
    mul_446: "f32[368]" = torch.ops.aten.mul.Tensor(sum_38, 0.002551020408163265)
    unsqueeze_323: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_446, 0);  mul_446 = None
    unsqueeze_324: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_447: "f32[368]" = torch.ops.aten.mul.Tensor(sum_39, 0.002551020408163265)
    mul_448: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_449: "f32[368]" = torch.ops.aten.mul.Tensor(mul_447, mul_448);  mul_447 = mul_448 = None
    unsqueeze_326: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_449, 0);  mul_449 = None
    unsqueeze_327: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_450: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_63);  primals_63 = None
    unsqueeze_329: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_330: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    mul_451: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_328);  sub_96 = unsqueeze_328 = None
    sub_98: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_16, mul_451);  mul_451 = None
    sub_99: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_98, unsqueeze_325);  sub_98 = unsqueeze_325 = None
    mul_452: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_331);  sub_99 = unsqueeze_331 = None
    mul_453: "f32[368]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_94);  sum_39 = squeeze_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_452, mul_225, primals_156, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_452 = mul_225 = primals_156 = None
    getitem_148: "f32[8, 368, 7, 7]" = convolution_backward_20[0]
    getitem_149: "f32[368, 368, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_454: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_148, relu_34)
    mul_455: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_148, sigmoid_8);  getitem_148 = None
    sum_40: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [2, 3], True);  mul_454 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_100: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_8)
    mul_456: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_8, sub_100);  sigmoid_8 = sub_100 = None
    mul_457: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_40, mul_456);  sum_40 = mul_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_41: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_457, [0, 2, 3])
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_457, relu_35, primals_154, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_457 = primals_154 = None
    getitem_151: "f32[8, 92, 1, 1]" = convolution_backward_21[0]
    getitem_152: "f32[368, 92, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_17: "b8[8, 92, 1, 1]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
    where_17: "f32[8, 92, 1, 1]" = torch.ops.aten.where.self(le_17, full_default, getitem_151);  le_17 = getitem_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_42: "f32[92]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(where_17, mean_8, primals_152, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_17 = mean_8 = primals_152 = None
    getitem_154: "f32[8, 368, 1, 1]" = convolution_backward_22[0]
    getitem_155: "f32[92, 368, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_5: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_154, [8, 368, 7, 7]);  getitem_154 = None
    div_5: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_5, 49);  expand_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_241: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_455, div_5);  mul_455 = div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_18: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
    where_18: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_18, full_default, add_241);  le_18 = add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_43: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_101: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_334);  convolution_46 = unsqueeze_334 = None
    mul_458: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_18, sub_101)
    sum_44: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 2, 3]);  mul_458 = None
    mul_459: "f32[368]" = torch.ops.aten.mul.Tensor(sum_43, 0.002551020408163265)
    unsqueeze_335: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_336: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_460: "f32[368]" = torch.ops.aten.mul.Tensor(sum_44, 0.002551020408163265)
    mul_461: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_462: "f32[368]" = torch.ops.aten.mul.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_338: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_339: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_463: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_61);  primals_61 = None
    unsqueeze_341: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
    unsqueeze_342: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    mul_464: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_340);  sub_101 = unsqueeze_340 = None
    sub_103: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_18, mul_464);  where_18 = mul_464 = None
    sub_104: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_337);  sub_103 = unsqueeze_337 = None
    mul_465: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_343);  sub_104 = unsqueeze_343 = None
    mul_466: "f32[368]" = torch.ops.aten.mul.Tensor(sum_44, squeeze_91);  sum_44 = squeeze_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_465, relu_33, primals_151, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_465 = primals_151 = None
    getitem_157: "f32[8, 368, 7, 7]" = convolution_backward_23[0]
    getitem_158: "f32[368, 8, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_19: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
    where_19: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_19, full_default, getitem_157);  le_19 = getitem_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_45: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_105: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_346);  convolution_45 = unsqueeze_346 = None
    mul_467: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_19, sub_105)
    sum_46: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 2, 3]);  mul_467 = None
    mul_468: "f32[368]" = torch.ops.aten.mul.Tensor(sum_45, 0.002551020408163265)
    unsqueeze_347: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_348: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_469: "f32[368]" = torch.ops.aten.mul.Tensor(sum_46, 0.002551020408163265)
    mul_470: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_471: "f32[368]" = torch.ops.aten.mul.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
    unsqueeze_350: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_351: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_472: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_59);  primals_59 = None
    unsqueeze_353: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
    unsqueeze_354: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    mul_473: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_352);  sub_105 = unsqueeze_352 = None
    sub_107: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_19, mul_473);  where_19 = mul_473 = None
    sub_108: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_349);  sub_107 = unsqueeze_349 = None
    mul_474: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_355);  sub_108 = unsqueeze_355 = None
    mul_475: "f32[368]" = torch.ops.aten.mul.Tensor(sum_46, squeeze_88);  sum_46 = squeeze_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_474, relu_32, primals_150, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = primals_150 = None
    getitem_160: "f32[8, 368, 7, 7]" = convolution_backward_24[0]
    getitem_161: "f32[368, 368, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_242: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(where_16, getitem_160);  where_16 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_20: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
    where_20: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_20, full_default, add_242);  le_20 = add_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_47: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_109: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_358);  convolution_44 = unsqueeze_358 = None
    mul_476: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_20, sub_109)
    sum_48: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 2, 3]);  mul_476 = None
    mul_477: "f32[368]" = torch.ops.aten.mul.Tensor(sum_47, 0.002551020408163265)
    unsqueeze_359: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_360: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_478: "f32[368]" = torch.ops.aten.mul.Tensor(sum_48, 0.002551020408163265)
    mul_479: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_480: "f32[368]" = torch.ops.aten.mul.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
    unsqueeze_362: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_363: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_481: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_57);  primals_57 = None
    unsqueeze_365: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
    unsqueeze_366: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    mul_482: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_364);  sub_109 = unsqueeze_364 = None
    sub_111: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_20, mul_482);  mul_482 = None
    sub_112: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_361);  sub_111 = unsqueeze_361 = None
    mul_483: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_367);  sub_112 = unsqueeze_367 = None
    mul_484: "f32[368]" = torch.ops.aten.mul.Tensor(sum_48, squeeze_85);  sum_48 = squeeze_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_483, mul_203, primals_149, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_483 = mul_203 = primals_149 = None
    getitem_163: "f32[8, 368, 7, 7]" = convolution_backward_25[0]
    getitem_164: "f32[368, 368, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_485: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_163, relu_30)
    mul_486: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_163, sigmoid_7);  getitem_163 = None
    sum_49: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2, 3], True);  mul_485 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_113: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_7)
    mul_487: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_7, sub_113);  sigmoid_7 = sub_113 = None
    mul_488: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_49, mul_487);  sum_49 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_50: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_488, [0, 2, 3])
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_488, relu_31, primals_147, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_488 = primals_147 = None
    getitem_166: "f32[8, 92, 1, 1]" = convolution_backward_26[0]
    getitem_167: "f32[368, 92, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_21: "b8[8, 92, 1, 1]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
    where_21: "f32[8, 92, 1, 1]" = torch.ops.aten.where.self(le_21, full_default, getitem_166);  le_21 = getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_51: "f32[92]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(where_21, mean_7, primals_145, [92], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_21 = mean_7 = primals_145 = None
    getitem_169: "f32[8, 368, 1, 1]" = convolution_backward_27[0]
    getitem_170: "f32[92, 368, 1, 1]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_6: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_169, [8, 368, 7, 7]);  getitem_169 = None
    div_6: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_6, 49);  expand_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_243: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_486, div_6);  mul_486 = div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_22: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
    where_22: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_22, full_default, add_243);  le_22 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_52: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_114: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_370);  convolution_41 = unsqueeze_370 = None
    mul_489: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_22, sub_114)
    sum_53: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_489, [0, 2, 3]);  mul_489 = None
    mul_490: "f32[368]" = torch.ops.aten.mul.Tensor(sum_52, 0.002551020408163265)
    unsqueeze_371: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
    unsqueeze_372: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_491: "f32[368]" = torch.ops.aten.mul.Tensor(sum_53, 0.002551020408163265)
    mul_492: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_493: "f32[368]" = torch.ops.aten.mul.Tensor(mul_491, mul_492);  mul_491 = mul_492 = None
    unsqueeze_374: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_493, 0);  mul_493 = None
    unsqueeze_375: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_494: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_55);  primals_55 = None
    unsqueeze_377: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_494, 0);  mul_494 = None
    unsqueeze_378: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    mul_495: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_376);  sub_114 = unsqueeze_376 = None
    sub_116: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_22, mul_495);  where_22 = mul_495 = None
    sub_117: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_116, unsqueeze_373);  sub_116 = unsqueeze_373 = None
    mul_496: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_379);  sub_117 = unsqueeze_379 = None
    mul_497: "f32[368]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_82);  sum_53 = squeeze_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_496, relu_29, primals_144, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_496 = primals_144 = None
    getitem_172: "f32[8, 368, 7, 7]" = convolution_backward_28[0]
    getitem_173: "f32[368, 8, 3, 3]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_23: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
    where_23: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_23, full_default, getitem_172);  le_23 = getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_54: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_118: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_382);  convolution_40 = unsqueeze_382 = None
    mul_498: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_23, sub_118)
    sum_55: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_498, [0, 2, 3]);  mul_498 = None
    mul_499: "f32[368]" = torch.ops.aten.mul.Tensor(sum_54, 0.002551020408163265)
    unsqueeze_383: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
    unsqueeze_384: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_500: "f32[368]" = torch.ops.aten.mul.Tensor(sum_55, 0.002551020408163265)
    mul_501: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_502: "f32[368]" = torch.ops.aten.mul.Tensor(mul_500, mul_501);  mul_500 = mul_501 = None
    unsqueeze_386: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_502, 0);  mul_502 = None
    unsqueeze_387: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_503: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_53);  primals_53 = None
    unsqueeze_389: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_503, 0);  mul_503 = None
    unsqueeze_390: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    mul_504: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_388);  sub_118 = unsqueeze_388 = None
    sub_120: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_23, mul_504);  where_23 = mul_504 = None
    sub_121: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_120, unsqueeze_385);  sub_120 = unsqueeze_385 = None
    mul_505: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_391);  sub_121 = unsqueeze_391 = None
    mul_506: "f32[368]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_79);  sum_55 = squeeze_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_505, relu_28, primals_143, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_505 = primals_143 = None
    getitem_175: "f32[8, 368, 7, 7]" = convolution_backward_29[0]
    getitem_176: "f32[368, 368, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_244: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(where_20, getitem_175);  where_20 = getitem_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_24: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
    where_24: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_24, full_default, add_244);  le_24 = add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_56: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_122: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_394);  convolution_39 = unsqueeze_394 = None
    mul_507: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_24, sub_122)
    sum_57: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
    mul_508: "f32[368]" = torch.ops.aten.mul.Tensor(sum_56, 0.002551020408163265)
    unsqueeze_395: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_396: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_509: "f32[368]" = torch.ops.aten.mul.Tensor(sum_57, 0.002551020408163265)
    mul_510: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_511: "f32[368]" = torch.ops.aten.mul.Tensor(mul_509, mul_510);  mul_509 = mul_510 = None
    unsqueeze_398: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_511, 0);  mul_511 = None
    unsqueeze_399: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_512: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_51);  primals_51 = None
    unsqueeze_401: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    unsqueeze_402: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    mul_513: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_400);  sub_122 = unsqueeze_400 = None
    sub_124: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_24, mul_513);  mul_513 = None
    sub_125: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_124, unsqueeze_397);  sub_124 = None
    mul_514: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_403);  sub_125 = unsqueeze_403 = None
    mul_515: "f32[368]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_76);  sum_57 = squeeze_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_514, relu_24, primals_142, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_514 = primals_142 = None
    getitem_178: "f32[8, 152, 14, 14]" = convolution_backward_30[0]
    getitem_179: "f32[368, 152, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_126: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_406);  convolution_38 = unsqueeze_406 = None
    mul_516: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_24, sub_126)
    sum_59: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_516, [0, 2, 3]);  mul_516 = None
    mul_518: "f32[368]" = torch.ops.aten.mul.Tensor(sum_59, 0.002551020408163265)
    mul_519: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_520: "f32[368]" = torch.ops.aten.mul.Tensor(mul_518, mul_519);  mul_518 = mul_519 = None
    unsqueeze_410: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_411: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_521: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_49);  primals_49 = None
    unsqueeze_413: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_521, 0);  mul_521 = None
    unsqueeze_414: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    mul_522: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_412);  sub_126 = unsqueeze_412 = None
    sub_128: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_24, mul_522);  where_24 = mul_522 = None
    sub_129: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_128, unsqueeze_397);  sub_128 = unsqueeze_397 = None
    mul_523: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_415);  sub_129 = unsqueeze_415 = None
    mul_524: "f32[368]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_73);  sum_59 = squeeze_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_523, mul_174, primals_141, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_523 = mul_174 = primals_141 = None
    getitem_181: "f32[8, 368, 7, 7]" = convolution_backward_31[0]
    getitem_182: "f32[368, 368, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_525: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_181, relu_26)
    mul_526: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_181, sigmoid_6);  getitem_181 = None
    sum_60: "f32[8, 368, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_525, [2, 3], True);  mul_525 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_130: "f32[8, 368, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_6)
    mul_527: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_6, sub_130);  sigmoid_6 = sub_130 = None
    mul_528: "f32[8, 368, 1, 1]" = torch.ops.aten.mul.Tensor(sum_60, mul_527);  sum_60 = mul_527 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_61: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 2, 3])
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_528, relu_27, primals_139, [368], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_528 = primals_139 = None
    getitem_184: "f32[8, 38, 1, 1]" = convolution_backward_32[0]
    getitem_185: "f32[368, 38, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_25: "b8[8, 38, 1, 1]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
    where_25: "f32[8, 38, 1, 1]" = torch.ops.aten.where.self(le_25, full_default, getitem_184);  le_25 = getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_62: "f32[38]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(where_25, mean_6, primals_137, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_25 = mean_6 = primals_137 = None
    getitem_187: "f32[8, 368, 1, 1]" = convolution_backward_33[0]
    getitem_188: "f32[38, 368, 1, 1]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_7: "f32[8, 368, 7, 7]" = torch.ops.aten.expand.default(getitem_187, [8, 368, 7, 7]);  getitem_187 = None
    div_7: "f32[8, 368, 7, 7]" = torch.ops.aten.div.Scalar(expand_7, 49);  expand_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_245: "f32[8, 368, 7, 7]" = torch.ops.aten.add.Tensor(mul_526, div_7);  mul_526 = div_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_26: "b8[8, 368, 7, 7]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
    where_26: "f32[8, 368, 7, 7]" = torch.ops.aten.where.self(le_26, full_default, add_245);  le_26 = add_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_63: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_131: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_418);  convolution_35 = unsqueeze_418 = None
    mul_529: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(where_26, sub_131)
    sum_64: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_529, [0, 2, 3]);  mul_529 = None
    mul_530: "f32[368]" = torch.ops.aten.mul.Tensor(sum_63, 0.002551020408163265)
    unsqueeze_419: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_530, 0);  mul_530 = None
    unsqueeze_420: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_531: "f32[368]" = torch.ops.aten.mul.Tensor(sum_64, 0.002551020408163265)
    mul_532: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_533: "f32[368]" = torch.ops.aten.mul.Tensor(mul_531, mul_532);  mul_531 = mul_532 = None
    unsqueeze_422: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_533, 0);  mul_533 = None
    unsqueeze_423: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_534: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_47);  primals_47 = None
    unsqueeze_425: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
    unsqueeze_426: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_535: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_424);  sub_131 = unsqueeze_424 = None
    sub_133: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(where_26, mul_535);  where_26 = mul_535 = None
    sub_134: "f32[8, 368, 7, 7]" = torch.ops.aten.sub.Tensor(sub_133, unsqueeze_421);  sub_133 = unsqueeze_421 = None
    mul_536: "f32[8, 368, 7, 7]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_427);  sub_134 = unsqueeze_427 = None
    mul_537: "f32[368]" = torch.ops.aten.mul.Tensor(sum_64, squeeze_70);  sum_64 = squeeze_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_536, relu_25, primals_136, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 46, [True, True, False]);  mul_536 = primals_136 = None
    getitem_190: "f32[8, 368, 14, 14]" = convolution_backward_34[0]
    getitem_191: "f32[368, 8, 3, 3]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_27: "b8[8, 368, 14, 14]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    where_27: "f32[8, 368, 14, 14]" = torch.ops.aten.where.self(le_27, full_default, getitem_190);  le_27 = getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_65: "f32[368]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_135: "f32[8, 368, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_430);  convolution_34 = unsqueeze_430 = None
    mul_538: "f32[8, 368, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_135)
    sum_66: "f32[368]" = torch.ops.aten.sum.dim_IntList(mul_538, [0, 2, 3]);  mul_538 = None
    mul_539: "f32[368]" = torch.ops.aten.mul.Tensor(sum_65, 0.0006377551020408163)
    unsqueeze_431: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_539, 0);  mul_539 = None
    unsqueeze_432: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_540: "f32[368]" = torch.ops.aten.mul.Tensor(sum_66, 0.0006377551020408163)
    mul_541: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_542: "f32[368]" = torch.ops.aten.mul.Tensor(mul_540, mul_541);  mul_540 = mul_541 = None
    unsqueeze_434: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_542, 0);  mul_542 = None
    unsqueeze_435: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_543: "f32[368]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_45);  primals_45 = None
    unsqueeze_437: "f32[1, 368]" = torch.ops.aten.unsqueeze.default(mul_543, 0);  mul_543 = None
    unsqueeze_438: "f32[1, 368, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 368, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_544: "f32[8, 368, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_436);  sub_135 = unsqueeze_436 = None
    sub_137: "f32[8, 368, 14, 14]" = torch.ops.aten.sub.Tensor(where_27, mul_544);  where_27 = mul_544 = None
    sub_138: "f32[8, 368, 14, 14]" = torch.ops.aten.sub.Tensor(sub_137, unsqueeze_433);  sub_137 = unsqueeze_433 = None
    mul_545: "f32[8, 368, 14, 14]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_439);  sub_138 = unsqueeze_439 = None
    mul_546: "f32[368]" = torch.ops.aten.mul.Tensor(sum_66, squeeze_67);  sum_66 = squeeze_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_545, relu_24, primals_135, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_545 = primals_135 = None
    getitem_193: "f32[8, 152, 14, 14]" = convolution_backward_35[0]
    getitem_194: "f32[368, 152, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_246: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(getitem_178, getitem_193);  getitem_178 = getitem_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_28: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
    where_28: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_28, full_default, add_246);  le_28 = add_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_67: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_139: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_442);  convolution_33 = unsqueeze_442 = None
    mul_547: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_28, sub_139)
    sum_68: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3]);  mul_547 = None
    mul_548: "f32[152]" = torch.ops.aten.mul.Tensor(sum_67, 0.0006377551020408163)
    unsqueeze_443: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_548, 0);  mul_548 = None
    unsqueeze_444: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_549: "f32[152]" = torch.ops.aten.mul.Tensor(sum_68, 0.0006377551020408163)
    mul_550: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_551: "f32[152]" = torch.ops.aten.mul.Tensor(mul_549, mul_550);  mul_549 = mul_550 = None
    unsqueeze_446: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_551, 0);  mul_551 = None
    unsqueeze_447: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_552: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_43);  primals_43 = None
    unsqueeze_449: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_450: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_553: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_448);  sub_139 = unsqueeze_448 = None
    sub_141: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_28, mul_553);  mul_553 = None
    sub_142: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_141, unsqueeze_445);  sub_141 = unsqueeze_445 = None
    mul_554: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_451);  sub_142 = unsqueeze_451 = None
    mul_555: "f32[152]" = torch.ops.aten.mul.Tensor(sum_68, squeeze_64);  sum_68 = squeeze_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_554, mul_152, primals_134, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_554 = mul_152 = primals_134 = None
    getitem_196: "f32[8, 152, 14, 14]" = convolution_backward_36[0]
    getitem_197: "f32[152, 152, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_556: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_196, relu_22)
    mul_557: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_196, sigmoid_5);  getitem_196 = None
    sum_69: "f32[8, 152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_556, [2, 3], True);  mul_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_143: "f32[8, 152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_5)
    mul_558: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_5, sub_143);  sigmoid_5 = sub_143 = None
    mul_559: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_69, mul_558);  sum_69 = mul_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_70: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 2, 3])
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_559, relu_23, primals_132, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_559 = primals_132 = None
    getitem_199: "f32[8, 38, 1, 1]" = convolution_backward_37[0]
    getitem_200: "f32[152, 38, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_29: "b8[8, 38, 1, 1]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
    where_29: "f32[8, 38, 1, 1]" = torch.ops.aten.where.self(le_29, full_default, getitem_199);  le_29 = getitem_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_71: "f32[38]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(where_29, mean_5, primals_130, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_29 = mean_5 = primals_130 = None
    getitem_202: "f32[8, 152, 1, 1]" = convolution_backward_38[0]
    getitem_203: "f32[38, 152, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_8: "f32[8, 152, 14, 14]" = torch.ops.aten.expand.default(getitem_202, [8, 152, 14, 14]);  getitem_202 = None
    div_8: "f32[8, 152, 14, 14]" = torch.ops.aten.div.Scalar(expand_8, 196);  expand_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_247: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_557, div_8);  mul_557 = div_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_30: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
    where_30: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_30, full_default, add_247);  le_30 = add_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_72: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_144: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_454);  convolution_30 = unsqueeze_454 = None
    mul_560: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_30, sub_144)
    sum_73: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_560, [0, 2, 3]);  mul_560 = None
    mul_561: "f32[152]" = torch.ops.aten.mul.Tensor(sum_72, 0.0006377551020408163)
    unsqueeze_455: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
    unsqueeze_456: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_562: "f32[152]" = torch.ops.aten.mul.Tensor(sum_73, 0.0006377551020408163)
    mul_563: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_564: "f32[152]" = torch.ops.aten.mul.Tensor(mul_562, mul_563);  mul_562 = mul_563 = None
    unsqueeze_458: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_564, 0);  mul_564 = None
    unsqueeze_459: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_565: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_41);  primals_41 = None
    unsqueeze_461: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_565, 0);  mul_565 = None
    unsqueeze_462: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_566: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_460);  sub_144 = unsqueeze_460 = None
    sub_146: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_30, mul_566);  where_30 = mul_566 = None
    sub_147: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_146, unsqueeze_457);  sub_146 = unsqueeze_457 = None
    mul_567: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_463);  sub_147 = unsqueeze_463 = None
    mul_568: "f32[152]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_61);  sum_73 = squeeze_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_567, relu_21, primals_129, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False]);  mul_567 = primals_129 = None
    getitem_205: "f32[8, 152, 14, 14]" = convolution_backward_39[0]
    getitem_206: "f32[152, 8, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_31: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    where_31: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_31, full_default, getitem_205);  le_31 = getitem_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_74: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_148: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_466);  convolution_29 = unsqueeze_466 = None
    mul_569: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_148)
    sum_75: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_569, [0, 2, 3]);  mul_569 = None
    mul_570: "f32[152]" = torch.ops.aten.mul.Tensor(sum_74, 0.0006377551020408163)
    unsqueeze_467: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_570, 0);  mul_570 = None
    unsqueeze_468: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_571: "f32[152]" = torch.ops.aten.mul.Tensor(sum_75, 0.0006377551020408163)
    mul_572: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_573: "f32[152]" = torch.ops.aten.mul.Tensor(mul_571, mul_572);  mul_571 = mul_572 = None
    unsqueeze_470: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_573, 0);  mul_573 = None
    unsqueeze_471: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_574: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_39);  primals_39 = None
    unsqueeze_473: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_574, 0);  mul_574 = None
    unsqueeze_474: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_575: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_472);  sub_148 = unsqueeze_472 = None
    sub_150: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_31, mul_575);  where_31 = mul_575 = None
    sub_151: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_150, unsqueeze_469);  sub_150 = unsqueeze_469 = None
    mul_576: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_475);  sub_151 = unsqueeze_475 = None
    mul_577: "f32[152]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_58);  sum_75 = squeeze_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_576, relu_20, primals_128, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_576 = primals_128 = None
    getitem_208: "f32[8, 152, 14, 14]" = convolution_backward_40[0]
    getitem_209: "f32[152, 152, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_248: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(where_28, getitem_208);  where_28 = getitem_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_32: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    where_32: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_32, full_default, add_248);  le_32 = add_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_76: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_152: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_478);  convolution_28 = unsqueeze_478 = None
    mul_578: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_32, sub_152)
    sum_77: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 2, 3]);  mul_578 = None
    mul_579: "f32[152]" = torch.ops.aten.mul.Tensor(sum_76, 0.0006377551020408163)
    unsqueeze_479: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_480: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_580: "f32[152]" = torch.ops.aten.mul.Tensor(sum_77, 0.0006377551020408163)
    mul_581: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_582: "f32[152]" = torch.ops.aten.mul.Tensor(mul_580, mul_581);  mul_580 = mul_581 = None
    unsqueeze_482: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_582, 0);  mul_582 = None
    unsqueeze_483: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_583: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_37);  primals_37 = None
    unsqueeze_485: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_583, 0);  mul_583 = None
    unsqueeze_486: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_584: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_484);  sub_152 = unsqueeze_484 = None
    sub_154: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_32, mul_584);  mul_584 = None
    sub_155: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_154, unsqueeze_481);  sub_154 = unsqueeze_481 = None
    mul_585: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_155, unsqueeze_487);  sub_155 = unsqueeze_487 = None
    mul_586: "f32[152]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_55);  sum_77 = squeeze_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_585, mul_130, primals_127, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_585 = mul_130 = primals_127 = None
    getitem_211: "f32[8, 152, 14, 14]" = convolution_backward_41[0]
    getitem_212: "f32[152, 152, 1, 1]" = convolution_backward_41[1];  convolution_backward_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_587: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_211, relu_18)
    mul_588: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_211, sigmoid_4);  getitem_211 = None
    sum_78: "f32[8, 152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_587, [2, 3], True);  mul_587 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_156: "f32[8, 152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_4)
    mul_589: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_4, sub_156);  sigmoid_4 = sub_156 = None
    mul_590: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_78, mul_589);  sum_78 = mul_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_79: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_590, [0, 2, 3])
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_590, relu_19, primals_125, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_590 = primals_125 = None
    getitem_214: "f32[8, 38, 1, 1]" = convolution_backward_42[0]
    getitem_215: "f32[152, 38, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_33: "b8[8, 38, 1, 1]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    where_33: "f32[8, 38, 1, 1]" = torch.ops.aten.where.self(le_33, full_default, getitem_214);  le_33 = getitem_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_80: "f32[38]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(where_33, mean_4, primals_123, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_33 = mean_4 = primals_123 = None
    getitem_217: "f32[8, 152, 1, 1]" = convolution_backward_43[0]
    getitem_218: "f32[38, 152, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_9: "f32[8, 152, 14, 14]" = torch.ops.aten.expand.default(getitem_217, [8, 152, 14, 14]);  getitem_217 = None
    div_9: "f32[8, 152, 14, 14]" = torch.ops.aten.div.Scalar(expand_9, 196);  expand_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_249: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_588, div_9);  mul_588 = div_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_34: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    where_34: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_34, full_default, add_249);  le_34 = add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_81: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_157: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_490);  convolution_25 = unsqueeze_490 = None
    mul_591: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_34, sub_157)
    sum_82: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_591, [0, 2, 3]);  mul_591 = None
    mul_592: "f32[152]" = torch.ops.aten.mul.Tensor(sum_81, 0.0006377551020408163)
    unsqueeze_491: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_592, 0);  mul_592 = None
    unsqueeze_492: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_593: "f32[152]" = torch.ops.aten.mul.Tensor(sum_82, 0.0006377551020408163)
    mul_594: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_595: "f32[152]" = torch.ops.aten.mul.Tensor(mul_593, mul_594);  mul_593 = mul_594 = None
    unsqueeze_494: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_595, 0);  mul_595 = None
    unsqueeze_495: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_596: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_35);  primals_35 = None
    unsqueeze_497: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_596, 0);  mul_596 = None
    unsqueeze_498: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_597: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_496);  sub_157 = unsqueeze_496 = None
    sub_159: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_34, mul_597);  where_34 = mul_597 = None
    sub_160: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_493);  sub_159 = unsqueeze_493 = None
    mul_598: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_499);  sub_160 = unsqueeze_499 = None
    mul_599: "f32[152]" = torch.ops.aten.mul.Tensor(sum_82, squeeze_52);  sum_82 = squeeze_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(mul_598, relu_17, primals_122, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False]);  mul_598 = primals_122 = None
    getitem_220: "f32[8, 152, 14, 14]" = convolution_backward_44[0]
    getitem_221: "f32[152, 8, 3, 3]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_35: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    where_35: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_35, full_default, getitem_220);  le_35 = getitem_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_83: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_161: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_502);  convolution_24 = unsqueeze_502 = None
    mul_600: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_161)
    sum_84: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_600, [0, 2, 3]);  mul_600 = None
    mul_601: "f32[152]" = torch.ops.aten.mul.Tensor(sum_83, 0.0006377551020408163)
    unsqueeze_503: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_601, 0);  mul_601 = None
    unsqueeze_504: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_602: "f32[152]" = torch.ops.aten.mul.Tensor(sum_84, 0.0006377551020408163)
    mul_603: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_604: "f32[152]" = torch.ops.aten.mul.Tensor(mul_602, mul_603);  mul_602 = mul_603 = None
    unsqueeze_506: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_604, 0);  mul_604 = None
    unsqueeze_507: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_605: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_33);  primals_33 = None
    unsqueeze_509: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_605, 0);  mul_605 = None
    unsqueeze_510: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_606: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_508);  sub_161 = unsqueeze_508 = None
    sub_163: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_35, mul_606);  where_35 = mul_606 = None
    sub_164: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_505);  sub_163 = unsqueeze_505 = None
    mul_607: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_511);  sub_164 = unsqueeze_511 = None
    mul_608: "f32[152]" = torch.ops.aten.mul.Tensor(sum_84, squeeze_49);  sum_84 = squeeze_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(mul_607, relu_16, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_607 = primals_121 = None
    getitem_223: "f32[8, 152, 14, 14]" = convolution_backward_45[0]
    getitem_224: "f32[152, 152, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_250: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(where_32, getitem_223);  where_32 = getitem_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_36: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    where_36: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_36, full_default, add_250);  le_36 = add_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_85: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_165: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_514);  convolution_23 = unsqueeze_514 = None
    mul_609: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_36, sub_165)
    sum_86: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_609, [0, 2, 3]);  mul_609 = None
    mul_610: "f32[152]" = torch.ops.aten.mul.Tensor(sum_85, 0.0006377551020408163)
    unsqueeze_515: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_610, 0);  mul_610 = None
    unsqueeze_516: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_611: "f32[152]" = torch.ops.aten.mul.Tensor(sum_86, 0.0006377551020408163)
    mul_612: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_613: "f32[152]" = torch.ops.aten.mul.Tensor(mul_611, mul_612);  mul_611 = mul_612 = None
    unsqueeze_518: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_613, 0);  mul_613 = None
    unsqueeze_519: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_614: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_31);  primals_31 = None
    unsqueeze_521: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_614, 0);  mul_614 = None
    unsqueeze_522: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_615: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_520);  sub_165 = unsqueeze_520 = None
    sub_167: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_36, mul_615);  mul_615 = None
    sub_168: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_517);  sub_167 = unsqueeze_517 = None
    mul_616: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_523);  sub_168 = unsqueeze_523 = None
    mul_617: "f32[152]" = torch.ops.aten.mul.Tensor(sum_86, squeeze_46);  sum_86 = squeeze_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_616, mul_108, primals_120, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_616 = mul_108 = primals_120 = None
    getitem_226: "f32[8, 152, 14, 14]" = convolution_backward_46[0]
    getitem_227: "f32[152, 152, 1, 1]" = convolution_backward_46[1];  convolution_backward_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_618: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_226, relu_14)
    mul_619: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_226, sigmoid_3);  getitem_226 = None
    sum_87: "f32[8, 152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_618, [2, 3], True);  mul_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_169: "f32[8, 152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_3)
    mul_620: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_3, sub_169);  sigmoid_3 = sub_169 = None
    mul_621: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_87, mul_620);  sum_87 = mul_620 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_88: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_621, [0, 2, 3])
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_621, relu_15, primals_118, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_621 = primals_118 = None
    getitem_229: "f32[8, 38, 1, 1]" = convolution_backward_47[0]
    getitem_230: "f32[152, 38, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_37: "b8[8, 38, 1, 1]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_37: "f32[8, 38, 1, 1]" = torch.ops.aten.where.self(le_37, full_default, getitem_229);  le_37 = getitem_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_89: "f32[38]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(where_37, mean_3, primals_116, [38], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_37 = mean_3 = primals_116 = None
    getitem_232: "f32[8, 152, 1, 1]" = convolution_backward_48[0]
    getitem_233: "f32[38, 152, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_10: "f32[8, 152, 14, 14]" = torch.ops.aten.expand.default(getitem_232, [8, 152, 14, 14]);  getitem_232 = None
    div_10: "f32[8, 152, 14, 14]" = torch.ops.aten.div.Scalar(expand_10, 196);  expand_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_251: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_619, div_10);  mul_619 = div_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_38: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_38: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_38, full_default, add_251);  le_38 = add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_90: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_170: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_526);  convolution_20 = unsqueeze_526 = None
    mul_622: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_38, sub_170)
    sum_91: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_622, [0, 2, 3]);  mul_622 = None
    mul_623: "f32[152]" = torch.ops.aten.mul.Tensor(sum_90, 0.0006377551020408163)
    unsqueeze_527: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_623, 0);  mul_623 = None
    unsqueeze_528: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_624: "f32[152]" = torch.ops.aten.mul.Tensor(sum_91, 0.0006377551020408163)
    mul_625: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_626: "f32[152]" = torch.ops.aten.mul.Tensor(mul_624, mul_625);  mul_624 = mul_625 = None
    unsqueeze_530: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_626, 0);  mul_626 = None
    unsqueeze_531: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_627: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_29);  primals_29 = None
    unsqueeze_533: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_627, 0);  mul_627 = None
    unsqueeze_534: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_628: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_170, unsqueeze_532);  sub_170 = unsqueeze_532 = None
    sub_172: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_38, mul_628);  where_38 = mul_628 = None
    sub_173: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_172, unsqueeze_529);  sub_172 = unsqueeze_529 = None
    mul_629: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_535);  sub_173 = unsqueeze_535 = None
    mul_630: "f32[152]" = torch.ops.aten.mul.Tensor(sum_91, squeeze_43);  sum_91 = squeeze_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(mul_629, relu_13, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False]);  mul_629 = primals_115 = None
    getitem_235: "f32[8, 152, 14, 14]" = convolution_backward_49[0]
    getitem_236: "f32[152, 8, 3, 3]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_39: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_39: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_39, full_default, getitem_235);  le_39 = getitem_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_92: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_174: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_538);  convolution_19 = unsqueeze_538 = None
    mul_631: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_174)
    sum_93: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_631, [0, 2, 3]);  mul_631 = None
    mul_632: "f32[152]" = torch.ops.aten.mul.Tensor(sum_92, 0.0006377551020408163)
    unsqueeze_539: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_632, 0);  mul_632 = None
    unsqueeze_540: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
    unsqueeze_541: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
    mul_633: "f32[152]" = torch.ops.aten.mul.Tensor(sum_93, 0.0006377551020408163)
    mul_634: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_635: "f32[152]" = torch.ops.aten.mul.Tensor(mul_633, mul_634);  mul_633 = mul_634 = None
    unsqueeze_542: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_635, 0);  mul_635 = None
    unsqueeze_543: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_636: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_27);  primals_27 = None
    unsqueeze_545: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_636, 0);  mul_636 = None
    unsqueeze_546: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_637: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_174, unsqueeze_544);  sub_174 = unsqueeze_544 = None
    sub_176: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_39, mul_637);  where_39 = mul_637 = None
    sub_177: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_176, unsqueeze_541);  sub_176 = unsqueeze_541 = None
    mul_638: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_547);  sub_177 = unsqueeze_547 = None
    mul_639: "f32[152]" = torch.ops.aten.mul.Tensor(sum_93, squeeze_40);  sum_93 = squeeze_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(mul_638, relu_12, primals_114, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_638 = primals_114 = None
    getitem_238: "f32[8, 152, 14, 14]" = convolution_backward_50[0]
    getitem_239: "f32[152, 152, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_252: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(where_36, getitem_238);  where_36 = getitem_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_40: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_40: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_40, full_default, add_252);  le_40 = add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_94: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_178: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_550);  convolution_18 = unsqueeze_550 = None
    mul_640: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_178)
    sum_95: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 2, 3]);  mul_640 = None
    mul_641: "f32[152]" = torch.ops.aten.mul.Tensor(sum_94, 0.0006377551020408163)
    unsqueeze_551: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_641, 0);  mul_641 = None
    unsqueeze_552: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_642: "f32[152]" = torch.ops.aten.mul.Tensor(sum_95, 0.0006377551020408163)
    mul_643: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_644: "f32[152]" = torch.ops.aten.mul.Tensor(mul_642, mul_643);  mul_642 = mul_643 = None
    unsqueeze_554: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_644, 0);  mul_644 = None
    unsqueeze_555: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_645: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_25);  primals_25 = None
    unsqueeze_557: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_645, 0);  mul_645 = None
    unsqueeze_558: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_646: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_178, unsqueeze_556);  sub_178 = unsqueeze_556 = None
    sub_180: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_646);  mul_646 = None
    sub_181: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_180, unsqueeze_553);  sub_180 = None
    mul_647: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_559);  sub_181 = unsqueeze_559 = None
    mul_648: "f32[152]" = torch.ops.aten.mul.Tensor(sum_95, squeeze_37);  sum_95 = squeeze_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_647, relu_8, primals_113, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_647 = primals_113 = None
    getitem_241: "f32[8, 56, 28, 28]" = convolution_backward_51[0]
    getitem_242: "f32[152, 56, 1, 1]" = convolution_backward_51[1];  convolution_backward_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_182: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_562);  convolution_17 = unsqueeze_562 = None
    mul_649: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_40, sub_182)
    sum_97: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_649, [0, 2, 3]);  mul_649 = None
    mul_651: "f32[152]" = torch.ops.aten.mul.Tensor(sum_97, 0.0006377551020408163)
    mul_652: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_653: "f32[152]" = torch.ops.aten.mul.Tensor(mul_651, mul_652);  mul_651 = mul_652 = None
    unsqueeze_566: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_653, 0);  mul_653 = None
    unsqueeze_567: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_654: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_23);  primals_23 = None
    unsqueeze_569: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_654, 0);  mul_654 = None
    unsqueeze_570: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_655: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_182, unsqueeze_568);  sub_182 = unsqueeze_568 = None
    sub_184: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_40, mul_655);  where_40 = mul_655 = None
    sub_185: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_184, unsqueeze_553);  sub_184 = unsqueeze_553 = None
    mul_656: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_571);  sub_185 = unsqueeze_571 = None
    mul_657: "f32[152]" = torch.ops.aten.mul.Tensor(sum_97, squeeze_34);  sum_97 = squeeze_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_656, mul_79, primals_112, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_656 = mul_79 = primals_112 = None
    getitem_244: "f32[8, 152, 14, 14]" = convolution_backward_52[0]
    getitem_245: "f32[152, 152, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_658: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_244, relu_10)
    mul_659: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_244, sigmoid_2);  getitem_244 = None
    sum_98: "f32[8, 152, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_658, [2, 3], True);  mul_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_186: "f32[8, 152, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_2)
    mul_660: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_2, sub_186);  sigmoid_2 = sub_186 = None
    mul_661: "f32[8, 152, 1, 1]" = torch.ops.aten.mul.Tensor(sum_98, mul_660);  sum_98 = mul_660 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_99: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_661, [0, 2, 3])
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_661, relu_11, primals_110, [152], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_661 = primals_110 = None
    getitem_247: "f32[8, 14, 1, 1]" = convolution_backward_53[0]
    getitem_248: "f32[152, 14, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_41: "b8[8, 14, 1, 1]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    where_41: "f32[8, 14, 1, 1]" = torch.ops.aten.where.self(le_41, full_default, getitem_247);  le_41 = getitem_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_100: "f32[14]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(where_41, mean_2, primals_108, [14], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_41 = mean_2 = primals_108 = None
    getitem_250: "f32[8, 152, 1, 1]" = convolution_backward_54[0]
    getitem_251: "f32[14, 152, 1, 1]" = convolution_backward_54[1];  convolution_backward_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_11: "f32[8, 152, 14, 14]" = torch.ops.aten.expand.default(getitem_250, [8, 152, 14, 14]);  getitem_250 = None
    div_11: "f32[8, 152, 14, 14]" = torch.ops.aten.div.Scalar(expand_11, 196);  expand_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_253: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(mul_659, div_11);  mul_659 = div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_42: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_42: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_42, full_default, add_253);  le_42 = add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_101: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_42, [0, 2, 3])
    sub_187: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_574);  convolution_14 = unsqueeze_574 = None
    mul_662: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_42, sub_187)
    sum_102: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_662, [0, 2, 3]);  mul_662 = None
    mul_663: "f32[152]" = torch.ops.aten.mul.Tensor(sum_101, 0.0006377551020408163)
    unsqueeze_575: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_663, 0);  mul_663 = None
    unsqueeze_576: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_664: "f32[152]" = torch.ops.aten.mul.Tensor(sum_102, 0.0006377551020408163)
    mul_665: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_666: "f32[152]" = torch.ops.aten.mul.Tensor(mul_664, mul_665);  mul_664 = mul_665 = None
    unsqueeze_578: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_666, 0);  mul_666 = None
    unsqueeze_579: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_667: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_21);  primals_21 = None
    unsqueeze_581: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_667, 0);  mul_667 = None
    unsqueeze_582: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_668: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_187, unsqueeze_580);  sub_187 = unsqueeze_580 = None
    sub_189: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_42, mul_668);  where_42 = mul_668 = None
    sub_190: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_189, unsqueeze_577);  sub_189 = unsqueeze_577 = None
    mul_669: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_190, unsqueeze_583);  sub_190 = unsqueeze_583 = None
    mul_670: "f32[152]" = torch.ops.aten.mul.Tensor(sum_102, squeeze_31);  sum_102 = squeeze_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_669, relu_9, primals_107, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 19, [True, True, False]);  mul_669 = primals_107 = None
    getitem_253: "f32[8, 152, 28, 28]" = convolution_backward_55[0]
    getitem_254: "f32[152, 8, 3, 3]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_43: "b8[8, 152, 28, 28]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_43: "f32[8, 152, 28, 28]" = torch.ops.aten.where.self(le_43, full_default, getitem_253);  le_43 = getitem_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_103: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_191: "f32[8, 152, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_586);  convolution_13 = unsqueeze_586 = None
    mul_671: "f32[8, 152, 28, 28]" = torch.ops.aten.mul.Tensor(where_43, sub_191)
    sum_104: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 2, 3]);  mul_671 = None
    mul_672: "f32[152]" = torch.ops.aten.mul.Tensor(sum_103, 0.00015943877551020407)
    unsqueeze_587: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_672, 0);  mul_672 = None
    unsqueeze_588: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_673: "f32[152]" = torch.ops.aten.mul.Tensor(sum_104, 0.00015943877551020407)
    mul_674: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_675: "f32[152]" = torch.ops.aten.mul.Tensor(mul_673, mul_674);  mul_673 = mul_674 = None
    unsqueeze_590: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_675, 0);  mul_675 = None
    unsqueeze_591: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_676: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_19);  primals_19 = None
    unsqueeze_593: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_676, 0);  mul_676 = None
    unsqueeze_594: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_677: "f32[8, 152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_191, unsqueeze_592);  sub_191 = unsqueeze_592 = None
    sub_193: "f32[8, 152, 28, 28]" = torch.ops.aten.sub.Tensor(where_43, mul_677);  where_43 = mul_677 = None
    sub_194: "f32[8, 152, 28, 28]" = torch.ops.aten.sub.Tensor(sub_193, unsqueeze_589);  sub_193 = unsqueeze_589 = None
    mul_678: "f32[8, 152, 28, 28]" = torch.ops.aten.mul.Tensor(sub_194, unsqueeze_595);  sub_194 = unsqueeze_595 = None
    mul_679: "f32[152]" = torch.ops.aten.mul.Tensor(sum_104, squeeze_28);  sum_104 = squeeze_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_678, relu_8, primals_106, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_678 = primals_106 = None
    getitem_256: "f32[8, 56, 28, 28]" = convolution_backward_56[0]
    getitem_257: "f32[152, 56, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_254: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(getitem_241, getitem_256);  getitem_241 = getitem_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_44: "b8[8, 56, 28, 28]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_44: "f32[8, 56, 28, 28]" = torch.ops.aten.where.self(le_44, full_default, add_254);  le_44 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_105: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_44, [0, 2, 3])
    sub_195: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_598);  convolution_12 = unsqueeze_598 = None
    mul_680: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(where_44, sub_195)
    sum_106: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 2, 3]);  mul_680 = None
    mul_681: "f32[56]" = torch.ops.aten.mul.Tensor(sum_105, 0.00015943877551020407)
    unsqueeze_599: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_681, 0);  mul_681 = None
    unsqueeze_600: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_682: "f32[56]" = torch.ops.aten.mul.Tensor(sum_106, 0.00015943877551020407)
    mul_683: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_684: "f32[56]" = torch.ops.aten.mul.Tensor(mul_682, mul_683);  mul_682 = mul_683 = None
    unsqueeze_602: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_684, 0);  mul_684 = None
    unsqueeze_603: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_685: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_17);  primals_17 = None
    unsqueeze_605: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_685, 0);  mul_685 = None
    unsqueeze_606: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_686: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_195, unsqueeze_604);  sub_195 = unsqueeze_604 = None
    sub_197: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(where_44, mul_686);  mul_686 = None
    sub_198: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_197, unsqueeze_601);  sub_197 = None
    mul_687: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_198, unsqueeze_607);  sub_198 = unsqueeze_607 = None
    mul_688: "f32[56]" = torch.ops.aten.mul.Tensor(sum_106, squeeze_25);  sum_106 = squeeze_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_687, relu_4, primals_105, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_687 = primals_105 = None
    getitem_259: "f32[8, 24, 56, 56]" = convolution_backward_57[0]
    getitem_260: "f32[56, 24, 1, 1]" = convolution_backward_57[1];  convolution_backward_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_199: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_610);  convolution_11 = unsqueeze_610 = None
    mul_689: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(where_44, sub_199)
    sum_108: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_689, [0, 2, 3]);  mul_689 = None
    mul_691: "f32[56]" = torch.ops.aten.mul.Tensor(sum_108, 0.00015943877551020407)
    mul_692: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_693: "f32[56]" = torch.ops.aten.mul.Tensor(mul_691, mul_692);  mul_691 = mul_692 = None
    unsqueeze_614: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_693, 0);  mul_693 = None
    unsqueeze_615: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
    unsqueeze_616: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
    mul_694: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_15);  primals_15 = None
    unsqueeze_617: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_694, 0);  mul_694 = None
    unsqueeze_618: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_695: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_199, unsqueeze_616);  sub_199 = unsqueeze_616 = None
    sub_201: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(where_44, mul_695);  where_44 = mul_695 = None
    sub_202: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_201, unsqueeze_601);  sub_201 = unsqueeze_601 = None
    mul_696: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_202, unsqueeze_619);  sub_202 = unsqueeze_619 = None
    mul_697: "f32[56]" = torch.ops.aten.mul.Tensor(sum_108, squeeze_22);  sum_108 = squeeze_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_696, mul_50, primals_104, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_696 = mul_50 = primals_104 = None
    getitem_262: "f32[8, 56, 28, 28]" = convolution_backward_58[0]
    getitem_263: "f32[56, 56, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_698: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_262, relu_6)
    mul_699: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_262, sigmoid_1);  getitem_262 = None
    sum_109: "f32[8, 56, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_698, [2, 3], True);  mul_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_203: "f32[8, 56, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid_1)
    mul_700: "f32[8, 56, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid_1, sub_203);  sigmoid_1 = sub_203 = None
    mul_701: "f32[8, 56, 1, 1]" = torch.ops.aten.mul.Tensor(sum_109, mul_700);  sum_109 = mul_700 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_110: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_701, [0, 2, 3])
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_701, relu_7, primals_102, [56], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_701 = primals_102 = None
    getitem_265: "f32[8, 6, 1, 1]" = convolution_backward_59[0]
    getitem_266: "f32[56, 6, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_45: "b8[8, 6, 1, 1]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_45: "f32[8, 6, 1, 1]" = torch.ops.aten.where.self(le_45, full_default, getitem_265);  le_45 = getitem_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_111: "f32[6]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(where_45, mean_1, primals_100, [6], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_45 = mean_1 = primals_100 = None
    getitem_268: "f32[8, 56, 1, 1]" = convolution_backward_60[0]
    getitem_269: "f32[6, 56, 1, 1]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_12: "f32[8, 56, 28, 28]" = torch.ops.aten.expand.default(getitem_268, [8, 56, 28, 28]);  getitem_268 = None
    div_12: "f32[8, 56, 28, 28]" = torch.ops.aten.div.Scalar(expand_12, 784);  expand_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_255: "f32[8, 56, 28, 28]" = torch.ops.aten.add.Tensor(mul_699, div_12);  mul_699 = div_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_46: "b8[8, 56, 28, 28]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_46: "f32[8, 56, 28, 28]" = torch.ops.aten.where.self(le_46, full_default, add_255);  le_46 = add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_112: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_46, [0, 2, 3])
    sub_204: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_622);  convolution_8 = unsqueeze_622 = None
    mul_702: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(where_46, sub_204)
    sum_113: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_702, [0, 2, 3]);  mul_702 = None
    mul_703: "f32[56]" = torch.ops.aten.mul.Tensor(sum_112, 0.00015943877551020407)
    unsqueeze_623: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_703, 0);  mul_703 = None
    unsqueeze_624: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
    unsqueeze_625: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
    mul_704: "f32[56]" = torch.ops.aten.mul.Tensor(sum_113, 0.00015943877551020407)
    mul_705: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_706: "f32[56]" = torch.ops.aten.mul.Tensor(mul_704, mul_705);  mul_704 = mul_705 = None
    unsqueeze_626: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_706, 0);  mul_706 = None
    unsqueeze_627: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
    unsqueeze_628: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
    mul_707: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_13);  primals_13 = None
    unsqueeze_629: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_707, 0);  mul_707 = None
    unsqueeze_630: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_708: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_628);  sub_204 = unsqueeze_628 = None
    sub_206: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(where_46, mul_708);  where_46 = mul_708 = None
    sub_207: "f32[8, 56, 28, 28]" = torch.ops.aten.sub.Tensor(sub_206, unsqueeze_625);  sub_206 = unsqueeze_625 = None
    mul_709: "f32[8, 56, 28, 28]" = torch.ops.aten.mul.Tensor(sub_207, unsqueeze_631);  sub_207 = unsqueeze_631 = None
    mul_710: "f32[56]" = torch.ops.aten.mul.Tensor(sum_113, squeeze_19);  sum_113 = squeeze_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_709, relu_5, primals_99, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 7, [True, True, False]);  mul_709 = primals_99 = None
    getitem_271: "f32[8, 56, 56, 56]" = convolution_backward_61[0]
    getitem_272: "f32[56, 8, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_47: "b8[8, 56, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_47: "f32[8, 56, 56, 56]" = torch.ops.aten.where.self(le_47, full_default, getitem_271);  le_47 = getitem_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_114: "f32[56]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_208: "f32[8, 56, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_634);  convolution_7 = unsqueeze_634 = None
    mul_711: "f32[8, 56, 56, 56]" = torch.ops.aten.mul.Tensor(where_47, sub_208)
    sum_115: "f32[56]" = torch.ops.aten.sum.dim_IntList(mul_711, [0, 2, 3]);  mul_711 = None
    mul_712: "f32[56]" = torch.ops.aten.mul.Tensor(sum_114, 3.985969387755102e-05)
    unsqueeze_635: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_712, 0);  mul_712 = None
    unsqueeze_636: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_713: "f32[56]" = torch.ops.aten.mul.Tensor(sum_115, 3.985969387755102e-05)
    mul_714: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_715: "f32[56]" = torch.ops.aten.mul.Tensor(mul_713, mul_714);  mul_713 = mul_714 = None
    unsqueeze_638: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_715, 0);  mul_715 = None
    unsqueeze_639: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_716: "f32[56]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_11);  primals_11 = None
    unsqueeze_641: "f32[1, 56]" = torch.ops.aten.unsqueeze.default(mul_716, 0);  mul_716 = None
    unsqueeze_642: "f32[1, 56, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 56, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_717: "f32[8, 56, 56, 56]" = torch.ops.aten.mul.Tensor(sub_208, unsqueeze_640);  sub_208 = unsqueeze_640 = None
    sub_210: "f32[8, 56, 56, 56]" = torch.ops.aten.sub.Tensor(where_47, mul_717);  where_47 = mul_717 = None
    sub_211: "f32[8, 56, 56, 56]" = torch.ops.aten.sub.Tensor(sub_210, unsqueeze_637);  sub_210 = unsqueeze_637 = None
    mul_718: "f32[8, 56, 56, 56]" = torch.ops.aten.mul.Tensor(sub_211, unsqueeze_643);  sub_211 = unsqueeze_643 = None
    mul_719: "f32[56]" = torch.ops.aten.mul.Tensor(sum_115, squeeze_16);  sum_115 = squeeze_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_62 = torch.ops.aten.convolution_backward.default(mul_718, relu_4, primals_98, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_718 = primals_98 = None
    getitem_274: "f32[8, 24, 56, 56]" = convolution_backward_62[0]
    getitem_275: "f32[56, 24, 1, 1]" = convolution_backward_62[1];  convolution_backward_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_256: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_259, getitem_274);  getitem_259 = getitem_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/regnet.py:245, code: x = self.act3(x)
    le_48: "b8[8, 24, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_48: "f32[8, 24, 56, 56]" = torch.ops.aten.where.self(le_48, full_default, add_256);  le_48 = add_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_116: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_48, [0, 2, 3])
    sub_212: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_646);  convolution_6 = unsqueeze_646 = None
    mul_720: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_48, sub_212)
    sum_117: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_720, [0, 2, 3]);  mul_720 = None
    mul_721: "f32[24]" = torch.ops.aten.mul.Tensor(sum_116, 3.985969387755102e-05)
    unsqueeze_647: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_721, 0);  mul_721 = None
    unsqueeze_648: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_722: "f32[24]" = torch.ops.aten.mul.Tensor(sum_117, 3.985969387755102e-05)
    mul_723: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_724: "f32[24]" = torch.ops.aten.mul.Tensor(mul_722, mul_723);  mul_722 = mul_723 = None
    unsqueeze_650: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_724, 0);  mul_724 = None
    unsqueeze_651: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_725: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_9);  primals_9 = None
    unsqueeze_653: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_725, 0);  mul_725 = None
    unsqueeze_654: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_726: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_212, unsqueeze_652);  sub_212 = unsqueeze_652 = None
    sub_214: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_48, mul_726);  mul_726 = None
    sub_215: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_214, unsqueeze_649);  sub_214 = None
    mul_727: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_215, unsqueeze_655);  sub_215 = unsqueeze_655 = None
    mul_728: "f32[24]" = torch.ops.aten.mul.Tensor(sum_117, squeeze_13);  sum_117 = squeeze_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_63 = torch.ops.aten.convolution_backward.default(mul_727, relu, primals_97, [0], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_727 = primals_97 = None
    getitem_277: "f32[8, 32, 112, 112]" = convolution_backward_63[0]
    getitem_278: "f32[24, 32, 1, 1]" = convolution_backward_63[1];  convolution_backward_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sub_216: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_658);  convolution_5 = unsqueeze_658 = None
    mul_729: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_48, sub_216)
    sum_119: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_729, [0, 2, 3]);  mul_729 = None
    mul_731: "f32[24]" = torch.ops.aten.mul.Tensor(sum_119, 3.985969387755102e-05)
    mul_732: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_733: "f32[24]" = torch.ops.aten.mul.Tensor(mul_731, mul_732);  mul_731 = mul_732 = None
    unsqueeze_662: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_733, 0);  mul_733 = None
    unsqueeze_663: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, 2);  unsqueeze_662 = None
    unsqueeze_664: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_663, 3);  unsqueeze_663 = None
    mul_734: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_7);  primals_7 = None
    unsqueeze_665: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_734, 0);  mul_734 = None
    unsqueeze_666: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_735: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_216, unsqueeze_664);  sub_216 = unsqueeze_664 = None
    sub_218: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_48, mul_735);  where_48 = mul_735 = None
    sub_219: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_218, unsqueeze_649);  sub_218 = unsqueeze_649 = None
    mul_736: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_219, unsqueeze_667);  sub_219 = unsqueeze_667 = None
    mul_737: "f32[24]" = torch.ops.aten.mul.Tensor(sum_119, squeeze_10);  sum_119 = squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_64 = torch.ops.aten.convolution_backward.default(mul_736, mul_21, primals_96, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_736 = mul_21 = primals_96 = None
    getitem_280: "f32[8, 24, 56, 56]" = convolution_backward_64[0]
    getitem_281: "f32[24, 24, 1, 1]" = convolution_backward_64[1];  convolution_backward_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:49, code: return x * self.gate(x_se)
    mul_738: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_280, relu_2)
    mul_739: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_280, sigmoid);  getitem_280 = None
    sum_120: "f32[8, 24, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_738, [2, 3], True);  mul_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/activations.py:57, code: return x.sigmoid_() if self.inplace else x.sigmoid()
    sub_220: "f32[8, 24, 1, 1]" = torch.ops.aten.sub.Tensor(1, sigmoid)
    mul_740: "f32[8, 24, 1, 1]" = torch.ops.aten.mul.Tensor(sigmoid, sub_220);  sigmoid = sub_220 = None
    mul_741: "f32[8, 24, 1, 1]" = torch.ops.aten.mul.Tensor(sum_120, mul_740);  sum_120 = mul_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:48, code: x_se = self.fc2(x_se)
    sum_121: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_741, [0, 2, 3])
    convolution_backward_65 = torch.ops.aten.convolution_backward.default(mul_741, relu_3, primals_94, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_741 = primals_94 = None
    getitem_283: "f32[8, 8, 1, 1]" = convolution_backward_65[0]
    getitem_284: "f32[24, 8, 1, 1]" = convolution_backward_65[1];  convolution_backward_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:47, code: x_se = self.act(self.bn(x_se))
    le_49: "b8[8, 8, 1, 1]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_49: "f32[8, 8, 1, 1]" = torch.ops.aten.where.self(le_49, full_default, getitem_283);  le_49 = getitem_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:46, code: x_se = self.fc1(x_se)
    sum_122: "f32[8]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    convolution_backward_66 = torch.ops.aten.convolution_backward.default(where_49, mean, primals_92, [8], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_49 = mean = primals_92 = None
    getitem_286: "f32[8, 24, 1, 1]" = convolution_backward_66[0]
    getitem_287: "f32[8, 24, 1, 1]" = convolution_backward_66[1];  convolution_backward_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    expand_13: "f32[8, 24, 56, 56]" = torch.ops.aten.expand.default(getitem_286, [8, 24, 56, 56]);  getitem_286 = None
    div_13: "f32[8, 24, 56, 56]" = torch.ops.aten.div.Scalar(expand_13, 3136);  expand_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/squeeze_excite.py:42, code: x_se = x.mean((2, 3), keepdim=True)
    add_257: "f32[8, 24, 56, 56]" = torch.ops.aten.add.Tensor(mul_739, div_13);  mul_739 = div_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_50: "b8[8, 24, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_50: "f32[8, 24, 56, 56]" = torch.ops.aten.where.self(le_50, full_default, add_257);  le_50 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_123: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    sub_221: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_670);  convolution_2 = unsqueeze_670 = None
    mul_742: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(where_50, sub_221)
    sum_124: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_742, [0, 2, 3]);  mul_742 = None
    mul_743: "f32[24]" = torch.ops.aten.mul.Tensor(sum_123, 3.985969387755102e-05)
    unsqueeze_671: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_743, 0);  mul_743 = None
    unsqueeze_672: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_671, 2);  unsqueeze_671 = None
    unsqueeze_673: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, 3);  unsqueeze_672 = None
    mul_744: "f32[24]" = torch.ops.aten.mul.Tensor(sum_124, 3.985969387755102e-05)
    mul_745: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_746: "f32[24]" = torch.ops.aten.mul.Tensor(mul_744, mul_745);  mul_744 = mul_745 = None
    unsqueeze_674: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_746, 0);  mul_746 = None
    unsqueeze_675: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, 2);  unsqueeze_674 = None
    unsqueeze_676: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_675, 3);  unsqueeze_675 = None
    mul_747: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_5);  primals_5 = None
    unsqueeze_677: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_747, 0);  mul_747 = None
    unsqueeze_678: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_748: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_221, unsqueeze_676);  sub_221 = unsqueeze_676 = None
    sub_223: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(where_50, mul_748);  where_50 = mul_748 = None
    sub_224: "f32[8, 24, 56, 56]" = torch.ops.aten.sub.Tensor(sub_223, unsqueeze_673);  sub_223 = unsqueeze_673 = None
    mul_749: "f32[8, 24, 56, 56]" = torch.ops.aten.mul.Tensor(sub_224, unsqueeze_679);  sub_224 = unsqueeze_679 = None
    mul_750: "f32[24]" = torch.ops.aten.mul.Tensor(sum_124, squeeze_7);  sum_124 = squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_67 = torch.ops.aten.convolution_backward.default(mul_749, relu_1, primals_91, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 3, [True, True, False]);  mul_749 = primals_91 = None
    getitem_289: "f32[8, 24, 112, 112]" = convolution_backward_67[0]
    getitem_290: "f32[24, 8, 3, 3]" = convolution_backward_67[1];  convolution_backward_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_51: "b8[8, 24, 112, 112]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_51: "f32[8, 24, 112, 112]" = torch.ops.aten.where.self(le_51, full_default, getitem_289);  le_51 = getitem_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_125: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    sub_225: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_682);  convolution_1 = unsqueeze_682 = None
    mul_751: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(where_51, sub_225)
    sum_126: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_751, [0, 2, 3]);  mul_751 = None
    mul_752: "f32[24]" = torch.ops.aten.mul.Tensor(sum_125, 9.964923469387754e-06)
    unsqueeze_683: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_752, 0);  mul_752 = None
    unsqueeze_684: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_683, 2);  unsqueeze_683 = None
    unsqueeze_685: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, 3);  unsqueeze_684 = None
    mul_753: "f32[24]" = torch.ops.aten.mul.Tensor(sum_126, 9.964923469387754e-06)
    mul_754: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_755: "f32[24]" = torch.ops.aten.mul.Tensor(mul_753, mul_754);  mul_753 = mul_754 = None
    unsqueeze_686: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_755, 0);  mul_755 = None
    unsqueeze_687: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, 2);  unsqueeze_686 = None
    unsqueeze_688: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_687, 3);  unsqueeze_687 = None
    mul_756: "f32[24]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_3);  primals_3 = None
    unsqueeze_689: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_756, 0);  mul_756 = None
    unsqueeze_690: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_757: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_225, unsqueeze_688);  sub_225 = unsqueeze_688 = None
    sub_227: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(where_51, mul_757);  where_51 = mul_757 = None
    sub_228: "f32[8, 24, 112, 112]" = torch.ops.aten.sub.Tensor(sub_227, unsqueeze_685);  sub_227 = unsqueeze_685 = None
    mul_758: "f32[8, 24, 112, 112]" = torch.ops.aten.mul.Tensor(sub_228, unsqueeze_691);  sub_228 = unsqueeze_691 = None
    mul_759: "f32[24]" = torch.ops.aten.mul.Tensor(sum_126, squeeze_4);  sum_126 = squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_68 = torch.ops.aten.convolution_backward.default(mul_758, relu, primals_90, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_758 = primals_90 = None
    getitem_292: "f32[8, 32, 112, 112]" = convolution_backward_68[0]
    getitem_293: "f32[24, 32, 1, 1]" = convolution_backward_68[1];  convolution_backward_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    add_258: "f32[8, 32, 112, 112]" = torch.ops.aten.add.Tensor(getitem_277, getitem_292);  getitem_277 = getitem_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    le_52: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_52: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_52, full_default, add_258);  le_52 = full_default = add_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    sum_127: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_229: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_694);  convolution = unsqueeze_694 = None
    mul_760: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_52, sub_229)
    sum_128: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_760, [0, 2, 3]);  mul_760 = None
    mul_761: "f32[32]" = torch.ops.aten.mul.Tensor(sum_127, 9.964923469387754e-06)
    unsqueeze_695: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_761, 0);  mul_761 = None
    unsqueeze_696: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_695, 2);  unsqueeze_695 = None
    unsqueeze_697: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, 3);  unsqueeze_696 = None
    mul_762: "f32[32]" = torch.ops.aten.mul.Tensor(sum_128, 9.964923469387754e-06)
    mul_763: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_764: "f32[32]" = torch.ops.aten.mul.Tensor(mul_762, mul_763);  mul_762 = mul_763 = None
    unsqueeze_698: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_764, 0);  mul_764 = None
    unsqueeze_699: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, 2);  unsqueeze_698 = None
    unsqueeze_700: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_699, 3);  unsqueeze_699 = None
    mul_765: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_1);  primals_1 = None
    unsqueeze_701: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_765, 0);  mul_765 = None
    unsqueeze_702: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_766: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_229, unsqueeze_700);  sub_229 = unsqueeze_700 = None
    sub_231: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_52, mul_766);  where_52 = mul_766 = None
    sub_232: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_231, unsqueeze_697);  sub_231 = unsqueeze_697 = None
    mul_767: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_232, unsqueeze_703);  sub_232 = unsqueeze_703 = None
    mul_768: "f32[32]" = torch.ops.aten.mul.Tensor(sum_128, squeeze_1);  sum_128 = squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/conv_bn_act.py:59, code: x = self.conv(x)
    convolution_backward_69 = torch.ops.aten.convolution_backward.default(mul_767, primals_319, primals_89, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_767 = primals_319 = primals_89 = None
    getitem_296: "f32[32, 3, 3, 3]" = convolution_backward_69[1];  convolution_backward_69 = None
    return [mul_768, sum_127, mul_759, sum_125, mul_750, sum_123, mul_737, sum_116, mul_728, sum_116, mul_719, sum_114, mul_710, sum_112, mul_697, sum_105, mul_688, sum_105, mul_679, sum_103, mul_670, sum_101, mul_657, sum_94, mul_648, sum_94, mul_639, sum_92, mul_630, sum_90, mul_617, sum_85, mul_608, sum_83, mul_599, sum_81, mul_586, sum_76, mul_577, sum_74, mul_568, sum_72, mul_555, sum_67, mul_546, sum_65, mul_537, sum_63, mul_524, sum_56, mul_515, sum_56, mul_506, sum_54, mul_497, sum_52, mul_484, sum_47, mul_475, sum_45, mul_466, sum_43, mul_453, sum_38, mul_444, sum_36, mul_435, sum_34, mul_422, sum_29, mul_413, sum_27, mul_404, sum_25, mul_391, sum_20, mul_382, sum_18, mul_373, sum_16, mul_360, sum_11, mul_351, sum_9, mul_342, sum_7, mul_329, sum_2, getitem_296, getitem_293, getitem_290, getitem_287, sum_122, getitem_284, sum_121, getitem_281, getitem_278, getitem_275, getitem_272, getitem_269, sum_111, getitem_266, sum_110, getitem_263, getitem_260, getitem_257, getitem_254, getitem_251, sum_100, getitem_248, sum_99, getitem_245, getitem_242, getitem_239, getitem_236, getitem_233, sum_89, getitem_230, sum_88, getitem_227, getitem_224, getitem_221, getitem_218, sum_80, getitem_215, sum_79, getitem_212, getitem_209, getitem_206, getitem_203, sum_71, getitem_200, sum_70, getitem_197, getitem_194, getitem_191, getitem_188, sum_62, getitem_185, sum_61, getitem_182, getitem_179, getitem_176, getitem_173, getitem_170, sum_51, getitem_167, sum_50, getitem_164, getitem_161, getitem_158, getitem_155, sum_42, getitem_152, sum_41, getitem_149, getitem_146, getitem_143, getitem_140, sum_33, getitem_137, sum_32, getitem_134, getitem_131, getitem_128, getitem_125, sum_24, getitem_122, sum_23, getitem_119, getitem_116, getitem_113, getitem_110, sum_15, getitem_107, sum_14, getitem_104, getitem_101, getitem_98, getitem_95, sum_6, getitem_92, sum_5, getitem_89, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    