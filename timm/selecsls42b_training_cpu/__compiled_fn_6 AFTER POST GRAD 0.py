from __future__ import annotations



def forward(self, primals_1: "f32[32, 3, 3, 3]", primals_2: "f32[32]", primals_4: "f32[64, 32, 3, 3]", primals_5: "f32[64]", primals_7: "f32[64, 64, 1, 1]", primals_8: "f32[64]", primals_10: "f32[32, 64, 3, 3]", primals_11: "f32[32]", primals_13: "f32[64, 32, 1, 1]", primals_14: "f32[64]", primals_16: "f32[32, 64, 3, 3]", primals_17: "f32[32]", primals_19: "f32[64, 128, 1, 1]", primals_20: "f32[64]", primals_22: "f32[64, 64, 3, 3]", primals_23: "f32[64]", primals_25: "f32[64, 64, 1, 1]", primals_26: "f32[64]", primals_28: "f32[32, 64, 3, 3]", primals_29: "f32[32]", primals_31: "f32[64, 32, 1, 1]", primals_32: "f32[64]", primals_34: "f32[32, 64, 3, 3]", primals_35: "f32[32]", primals_37: "f32[128, 192, 1, 1]", primals_38: "f32[128]", primals_40: "f32[144, 128, 3, 3]", primals_41: "f32[144]", primals_43: "f32[144, 144, 1, 1]", primals_44: "f32[144]", primals_46: "f32[72, 144, 3, 3]", primals_47: "f32[72]", primals_49: "f32[144, 72, 1, 1]", primals_50: "f32[144]", primals_52: "f32[72, 144, 3, 3]", primals_53: "f32[72]", primals_55: "f32[144, 288, 1, 1]", primals_56: "f32[144]", primals_58: "f32[144, 144, 3, 3]", primals_59: "f32[144]", primals_61: "f32[144, 144, 1, 1]", primals_62: "f32[144]", primals_64: "f32[72, 144, 3, 3]", primals_65: "f32[72]", primals_67: "f32[144, 72, 1, 1]", primals_68: "f32[144]", primals_70: "f32[72, 144, 3, 3]", primals_71: "f32[72]", primals_73: "f32[288, 432, 1, 1]", primals_74: "f32[288]", primals_76: "f32[304, 288, 3, 3]", primals_77: "f32[304]", primals_79: "f32[304, 304, 1, 1]", primals_80: "f32[304]", primals_82: "f32[152, 304, 3, 3]", primals_83: "f32[152]", primals_85: "f32[304, 152, 1, 1]", primals_86: "f32[304]", primals_88: "f32[152, 304, 3, 3]", primals_89: "f32[152]", primals_91: "f32[304, 608, 1, 1]", primals_92: "f32[304]", primals_94: "f32[304, 304, 3, 3]", primals_95: "f32[304]", primals_97: "f32[304, 304, 1, 1]", primals_98: "f32[304]", primals_100: "f32[152, 304, 3, 3]", primals_101: "f32[152]", primals_103: "f32[304, 152, 1, 1]", primals_104: "f32[304]", primals_106: "f32[152, 304, 3, 3]", primals_107: "f32[152]", primals_109: "f32[480, 912, 1, 1]", primals_110: "f32[480]", primals_112: "f32[960, 480, 3, 3]", primals_113: "f32[960]", primals_115: "f32[1024, 960, 3, 3]", primals_116: "f32[1024]", primals_118: "f32[1280, 1024, 3, 3]", primals_119: "f32[1280]", primals_121: "f32[1024, 1280, 1, 1]", primals_122: "f32[1024]", primals_249: "f32[8, 3, 224, 224]", convolution: "f32[8, 32, 112, 112]", squeeze_1: "f32[32]", relu: "f32[8, 32, 112, 112]", convolution_1: "f32[8, 64, 56, 56]", squeeze_4: "f32[64]", relu_1: "f32[8, 64, 56, 56]", convolution_2: "f32[8, 64, 56, 56]", squeeze_7: "f32[64]", relu_2: "f32[8, 64, 56, 56]", convolution_3: "f32[8, 32, 56, 56]", squeeze_10: "f32[32]", relu_3: "f32[8, 32, 56, 56]", convolution_4: "f32[8, 64, 56, 56]", squeeze_13: "f32[64]", relu_4: "f32[8, 64, 56, 56]", convolution_5: "f32[8, 32, 56, 56]", squeeze_16: "f32[32]", cat: "f32[8, 128, 56, 56]", convolution_6: "f32[8, 64, 56, 56]", squeeze_19: "f32[64]", relu_6: "f32[8, 64, 56, 56]", convolution_7: "f32[8, 64, 56, 56]", squeeze_22: "f32[64]", relu_7: "f32[8, 64, 56, 56]", convolution_8: "f32[8, 64, 56, 56]", squeeze_25: "f32[64]", relu_8: "f32[8, 64, 56, 56]", convolution_9: "f32[8, 32, 56, 56]", squeeze_28: "f32[32]", relu_9: "f32[8, 32, 56, 56]", convolution_10: "f32[8, 64, 56, 56]", squeeze_31: "f32[64]", relu_10: "f32[8, 64, 56, 56]", convolution_11: "f32[8, 32, 56, 56]", squeeze_34: "f32[32]", cat_1: "f32[8, 192, 56, 56]", convolution_12: "f32[8, 128, 56, 56]", squeeze_37: "f32[128]", relu_12: "f32[8, 128, 56, 56]", convolution_13: "f32[8, 144, 28, 28]", squeeze_40: "f32[144]", relu_13: "f32[8, 144, 28, 28]", convolution_14: "f32[8, 144, 28, 28]", squeeze_43: "f32[144]", relu_14: "f32[8, 144, 28, 28]", convolution_15: "f32[8, 72, 28, 28]", squeeze_46: "f32[72]", relu_15: "f32[8, 72, 28, 28]", convolution_16: "f32[8, 144, 28, 28]", squeeze_49: "f32[144]", relu_16: "f32[8, 144, 28, 28]", convolution_17: "f32[8, 72, 28, 28]", squeeze_52: "f32[72]", cat_2: "f32[8, 288, 28, 28]", convolution_18: "f32[8, 144, 28, 28]", squeeze_55: "f32[144]", relu_18: "f32[8, 144, 28, 28]", convolution_19: "f32[8, 144, 28, 28]", squeeze_58: "f32[144]", relu_19: "f32[8, 144, 28, 28]", convolution_20: "f32[8, 144, 28, 28]", squeeze_61: "f32[144]", relu_20: "f32[8, 144, 28, 28]", convolution_21: "f32[8, 72, 28, 28]", squeeze_64: "f32[72]", relu_21: "f32[8, 72, 28, 28]", convolution_22: "f32[8, 144, 28, 28]", squeeze_67: "f32[144]", relu_22: "f32[8, 144, 28, 28]", convolution_23: "f32[8, 72, 28, 28]", squeeze_70: "f32[72]", cat_3: "f32[8, 432, 28, 28]", convolution_24: "f32[8, 288, 28, 28]", squeeze_73: "f32[288]", relu_24: "f32[8, 288, 28, 28]", convolution_25: "f32[8, 304, 14, 14]", squeeze_76: "f32[304]", relu_25: "f32[8, 304, 14, 14]", convolution_26: "f32[8, 304, 14, 14]", squeeze_79: "f32[304]", relu_26: "f32[8, 304, 14, 14]", convolution_27: "f32[8, 152, 14, 14]", squeeze_82: "f32[152]", relu_27: "f32[8, 152, 14, 14]", convolution_28: "f32[8, 304, 14, 14]", squeeze_85: "f32[304]", relu_28: "f32[8, 304, 14, 14]", convolution_29: "f32[8, 152, 14, 14]", squeeze_88: "f32[152]", cat_4: "f32[8, 608, 14, 14]", convolution_30: "f32[8, 304, 14, 14]", squeeze_91: "f32[304]", relu_30: "f32[8, 304, 14, 14]", convolution_31: "f32[8, 304, 14, 14]", squeeze_94: "f32[304]", relu_31: "f32[8, 304, 14, 14]", convolution_32: "f32[8, 304, 14, 14]", squeeze_97: "f32[304]", relu_32: "f32[8, 304, 14, 14]", convolution_33: "f32[8, 152, 14, 14]", squeeze_100: "f32[152]", relu_33: "f32[8, 152, 14, 14]", convolution_34: "f32[8, 304, 14, 14]", squeeze_103: "f32[304]", relu_34: "f32[8, 304, 14, 14]", convolution_35: "f32[8, 152, 14, 14]", squeeze_106: "f32[152]", cat_5: "f32[8, 912, 14, 14]", convolution_36: "f32[8, 480, 14, 14]", squeeze_109: "f32[480]", relu_36: "f32[8, 480, 14, 14]", convolution_37: "f32[8, 960, 7, 7]", squeeze_112: "f32[960]", relu_37: "f32[8, 960, 7, 7]", convolution_38: "f32[8, 1024, 7, 7]", squeeze_115: "f32[1024]", relu_38: "f32[8, 1024, 7, 7]", convolution_39: "f32[8, 1280, 4, 4]", squeeze_118: "f32[1280]", relu_39: "f32[8, 1280, 4, 4]", convolution_40: "f32[8, 1024, 4, 4]", squeeze_121: "f32[1024]", clone: "f32[8, 1024]", permute_1: "f32[1000, 1024]", le: "b8[8, 1024, 4, 4]", unsqueeze_166: "f32[1, 1024, 1, 1]", unsqueeze_178: "f32[1, 1280, 1, 1]", unsqueeze_190: "f32[1, 1024, 1, 1]", unsqueeze_202: "f32[1, 960, 1, 1]", unsqueeze_214: "f32[1, 480, 1, 1]", le_5: "b8[8, 152, 14, 14]", unsqueeze_226: "f32[1, 152, 1, 1]", unsqueeze_238: "f32[1, 304, 1, 1]", unsqueeze_250: "f32[1, 152, 1, 1]", unsqueeze_262: "f32[1, 304, 1, 1]", unsqueeze_274: "f32[1, 304, 1, 1]", unsqueeze_286: "f32[1, 304, 1, 1]", le_11: "b8[8, 152, 14, 14]", unsqueeze_298: "f32[1, 152, 1, 1]", unsqueeze_310: "f32[1, 304, 1, 1]", unsqueeze_322: "f32[1, 152, 1, 1]", unsqueeze_334: "f32[1, 304, 1, 1]", unsqueeze_346: "f32[1, 304, 1, 1]", unsqueeze_358: "f32[1, 288, 1, 1]", le_17: "b8[8, 72, 28, 28]", unsqueeze_370: "f32[1, 72, 1, 1]", unsqueeze_382: "f32[1, 144, 1, 1]", unsqueeze_394: "f32[1, 72, 1, 1]", unsqueeze_406: "f32[1, 144, 1, 1]", unsqueeze_418: "f32[1, 144, 1, 1]", unsqueeze_430: "f32[1, 144, 1, 1]", le_23: "b8[8, 72, 28, 28]", unsqueeze_442: "f32[1, 72, 1, 1]", unsqueeze_454: "f32[1, 144, 1, 1]", unsqueeze_466: "f32[1, 72, 1, 1]", unsqueeze_478: "f32[1, 144, 1, 1]", unsqueeze_490: "f32[1, 144, 1, 1]", unsqueeze_502: "f32[1, 128, 1, 1]", le_29: "b8[8, 32, 56, 56]", unsqueeze_514: "f32[1, 32, 1, 1]", unsqueeze_526: "f32[1, 64, 1, 1]", unsqueeze_538: "f32[1, 32, 1, 1]", unsqueeze_550: "f32[1, 64, 1, 1]", unsqueeze_562: "f32[1, 64, 1, 1]", unsqueeze_574: "f32[1, 64, 1, 1]", le_35: "b8[8, 32, 56, 56]", unsqueeze_586: "f32[1, 32, 1, 1]", unsqueeze_598: "f32[1, 64, 1, 1]", unsqueeze_610: "f32[1, 32, 1, 1]", unsqueeze_622: "f32[1, 64, 1, 1]", unsqueeze_634: "f32[1, 64, 1, 1]", unsqueeze_646: "f32[1, 32, 1, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:177, code: return x if pre_logits else self.fc(x)
    mm: "f32[8, 1024]" = torch.ops.aten.mm.default(tangents_1, permute_1);  permute_1 = None
    permute_2: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1024]" = torch.ops.aten.mm.default(permute_2, clone);  permute_2 = clone = None
    permute_3: "f32[1024, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_4: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    view_2: "f32[8, 1024, 1, 1]" = torch.ops.aten.reshape.default(mm, [8, 1024, 1, 1]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    expand: "f32[8, 1024, 4, 4]" = torch.ops.aten.expand.default(view_2, [8, 1024, 4, 4]);  view_2 = None
    div: "f32[8, 1024, 4, 4]" = torch.ops.aten.div.Scalar(expand, 16);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:171, code: x = self.head(self.from_seq(x))
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where: "f32[8, 1024, 4, 4]" = torch.ops.aten.where.self(le, full_default, div);  le = div = None
    sum_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where, [0, 2, 3])
    sub_41: "f32[8, 1024, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_166);  convolution_40 = unsqueeze_166 = None
    mul_287: "f32[8, 1024, 4, 4]" = torch.ops.aten.mul.Tensor(where, sub_41)
    sum_3: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 2, 3]);  mul_287 = None
    mul_288: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_2, 0.0078125)
    unsqueeze_167: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_288, 0);  mul_288 = None
    unsqueeze_168: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 2);  unsqueeze_167 = None
    unsqueeze_169: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, 3);  unsqueeze_168 = None
    mul_289: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, 0.0078125)
    mul_290: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_121, squeeze_121)
    mul_291: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_289, mul_290);  mul_289 = mul_290 = None
    unsqueeze_170: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_291, 0);  mul_291 = None
    unsqueeze_171: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, 2);  unsqueeze_170 = None
    unsqueeze_172: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_171, 3);  unsqueeze_171 = None
    mul_292: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_121, primals_122);  primals_122 = None
    unsqueeze_173: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_292, 0);  mul_292 = None
    unsqueeze_174: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_173, 2);  unsqueeze_173 = None
    unsqueeze_175: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, 3);  unsqueeze_174 = None
    mul_293: "f32[8, 1024, 4, 4]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_172);  sub_41 = unsqueeze_172 = None
    sub_43: "f32[8, 1024, 4, 4]" = torch.ops.aten.sub.Tensor(where, mul_293);  where = mul_293 = None
    sub_44: "f32[8, 1024, 4, 4]" = torch.ops.aten.sub.Tensor(sub_43, unsqueeze_169);  sub_43 = unsqueeze_169 = None
    mul_294: "f32[8, 1024, 4, 4]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_175);  sub_44 = unsqueeze_175 = None
    mul_295: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_3, squeeze_121);  sum_3 = squeeze_121 = None
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_294, relu_39, primals_121, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_294 = primals_121 = None
    getitem_82: "f32[8, 1280, 4, 4]" = convolution_backward[0]
    getitem_83: "f32[1024, 1280, 1, 1]" = convolution_backward[1];  convolution_backward = None
    le_1: "b8[8, 1280, 4, 4]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
    where_1: "f32[8, 1280, 4, 4]" = torch.ops.aten.where.self(le_1, full_default, getitem_82);  le_1 = getitem_82 = None
    sum_4: "f32[1280]" = torch.ops.aten.sum.dim_IntList(where_1, [0, 2, 3])
    sub_45: "f32[8, 1280, 4, 4]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_178);  convolution_39 = unsqueeze_178 = None
    mul_296: "f32[8, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(where_1, sub_45)
    sum_5: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_296, [0, 2, 3]);  mul_296 = None
    mul_297: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_4, 0.0078125)
    unsqueeze_179: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_297, 0);  mul_297 = None
    unsqueeze_180: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_179, 2);  unsqueeze_179 = None
    unsqueeze_181: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 3);  unsqueeze_180 = None
    mul_298: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_5, 0.0078125)
    mul_299: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_118, squeeze_118)
    mul_300: "f32[1280]" = torch.ops.aten.mul.Tensor(mul_298, mul_299);  mul_298 = mul_299 = None
    unsqueeze_182: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_300, 0);  mul_300 = None
    unsqueeze_183: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, 2);  unsqueeze_182 = None
    unsqueeze_184: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_183, 3);  unsqueeze_183 = None
    mul_301: "f32[1280]" = torch.ops.aten.mul.Tensor(squeeze_118, primals_119);  primals_119 = None
    unsqueeze_185: "f32[1, 1280]" = torch.ops.aten.unsqueeze.default(mul_301, 0);  mul_301 = None
    unsqueeze_186: "f32[1, 1280, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_185, 2);  unsqueeze_185 = None
    unsqueeze_187: "f32[1, 1280, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, 3);  unsqueeze_186 = None
    mul_302: "f32[8, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_184);  sub_45 = unsqueeze_184 = None
    sub_47: "f32[8, 1280, 4, 4]" = torch.ops.aten.sub.Tensor(where_1, mul_302);  where_1 = mul_302 = None
    sub_48: "f32[8, 1280, 4, 4]" = torch.ops.aten.sub.Tensor(sub_47, unsqueeze_181);  sub_47 = unsqueeze_181 = None
    mul_303: "f32[8, 1280, 4, 4]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_187);  sub_48 = unsqueeze_187 = None
    mul_304: "f32[1280]" = torch.ops.aten.mul.Tensor(sum_5, squeeze_118);  sum_5 = squeeze_118 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_303, relu_38, primals_118, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_303 = primals_118 = None
    getitem_85: "f32[8, 1024, 7, 7]" = convolution_backward_1[0]
    getitem_86: "f32[1280, 1024, 3, 3]" = convolution_backward_1[1];  convolution_backward_1 = None
    le_2: "b8[8, 1024, 7, 7]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
    where_2: "f32[8, 1024, 7, 7]" = torch.ops.aten.where.self(le_2, full_default, getitem_85);  le_2 = getitem_85 = None
    sum_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(where_2, [0, 2, 3])
    sub_49: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_190);  convolution_38 = unsqueeze_190 = None
    mul_305: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(where_2, sub_49)
    sum_7: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 2, 3]);  mul_305 = None
    mul_306: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_6, 0.002551020408163265)
    unsqueeze_191: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_306, 0);  mul_306 = None
    unsqueeze_192: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_191, 2);  unsqueeze_191 = None
    unsqueeze_193: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 3);  unsqueeze_192 = None
    mul_307: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_7, 0.002551020408163265)
    mul_308: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_115, squeeze_115)
    mul_309: "f32[1024]" = torch.ops.aten.mul.Tensor(mul_307, mul_308);  mul_307 = mul_308 = None
    unsqueeze_194: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_309, 0);  mul_309 = None
    unsqueeze_195: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, 2);  unsqueeze_194 = None
    unsqueeze_196: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_195, 3);  unsqueeze_195 = None
    mul_310: "f32[1024]" = torch.ops.aten.mul.Tensor(squeeze_115, primals_116);  primals_116 = None
    unsqueeze_197: "f32[1, 1024]" = torch.ops.aten.unsqueeze.default(mul_310, 0);  mul_310 = None
    unsqueeze_198: "f32[1, 1024, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_197, 2);  unsqueeze_197 = None
    unsqueeze_199: "f32[1, 1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, 3);  unsqueeze_198 = None
    mul_311: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_196);  sub_49 = unsqueeze_196 = None
    sub_51: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(where_2, mul_311);  where_2 = mul_311 = None
    sub_52: "f32[8, 1024, 7, 7]" = torch.ops.aten.sub.Tensor(sub_51, unsqueeze_193);  sub_51 = unsqueeze_193 = None
    mul_312: "f32[8, 1024, 7, 7]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_199);  sub_52 = unsqueeze_199 = None
    mul_313: "f32[1024]" = torch.ops.aten.mul.Tensor(sum_7, squeeze_115);  sum_7 = squeeze_115 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(mul_312, relu_37, primals_115, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_312 = primals_115 = None
    getitem_88: "f32[8, 960, 7, 7]" = convolution_backward_2[0]
    getitem_89: "f32[1024, 960, 3, 3]" = convolution_backward_2[1];  convolution_backward_2 = None
    le_3: "b8[8, 960, 7, 7]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
    where_3: "f32[8, 960, 7, 7]" = torch.ops.aten.where.self(le_3, full_default, getitem_88);  le_3 = getitem_88 = None
    sum_8: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_53: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_202);  convolution_37 = unsqueeze_202 = None
    mul_314: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_53)
    sum_9: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3]);  mul_314 = None
    mul_315: "f32[960]" = torch.ops.aten.mul.Tensor(sum_8, 0.002551020408163265)
    unsqueeze_203: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_315, 0);  mul_315 = None
    unsqueeze_204: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 2);  unsqueeze_203 = None
    unsqueeze_205: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, 3);  unsqueeze_204 = None
    mul_316: "f32[960]" = torch.ops.aten.mul.Tensor(sum_9, 0.002551020408163265)
    mul_317: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_112, squeeze_112)
    mul_318: "f32[960]" = torch.ops.aten.mul.Tensor(mul_316, mul_317);  mul_316 = mul_317 = None
    unsqueeze_206: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_318, 0);  mul_318 = None
    unsqueeze_207: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, 2);  unsqueeze_206 = None
    unsqueeze_208: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_207, 3);  unsqueeze_207 = None
    mul_319: "f32[960]" = torch.ops.aten.mul.Tensor(squeeze_112, primals_113);  primals_113 = None
    unsqueeze_209: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
    unsqueeze_210: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 2);  unsqueeze_209 = None
    unsqueeze_211: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, 3);  unsqueeze_210 = None
    mul_320: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_208);  sub_53 = unsqueeze_208 = None
    sub_55: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(where_3, mul_320);  where_3 = mul_320 = None
    sub_56: "f32[8, 960, 7, 7]" = torch.ops.aten.sub.Tensor(sub_55, unsqueeze_205);  sub_55 = unsqueeze_205 = None
    mul_321: "f32[8, 960, 7, 7]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_211);  sub_56 = unsqueeze_211 = None
    mul_322: "f32[960]" = torch.ops.aten.mul.Tensor(sum_9, squeeze_112);  sum_9 = squeeze_112 = None
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(mul_321, relu_36, primals_112, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_321 = primals_112 = None
    getitem_91: "f32[8, 480, 14, 14]" = convolution_backward_3[0]
    getitem_92: "f32[960, 480, 3, 3]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:107, code: return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]
    le_4: "b8[8, 480, 14, 14]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
    where_4: "f32[8, 480, 14, 14]" = torch.ops.aten.where.self(le_4, full_default, getitem_91);  le_4 = getitem_91 = None
    sum_10: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    sub_57: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_214);  convolution_36 = unsqueeze_214 = None
    mul_323: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_4, sub_57)
    sum_11: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 2, 3]);  mul_323 = None
    mul_324: "f32[480]" = torch.ops.aten.mul.Tensor(sum_10, 0.0006377551020408163)
    unsqueeze_215: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_324, 0);  mul_324 = None
    unsqueeze_216: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_215, 2);  unsqueeze_215 = None
    unsqueeze_217: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, 3);  unsqueeze_216 = None
    mul_325: "f32[480]" = torch.ops.aten.mul.Tensor(sum_11, 0.0006377551020408163)
    mul_326: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_109, squeeze_109)
    mul_327: "f32[480]" = torch.ops.aten.mul.Tensor(mul_325, mul_326);  mul_325 = mul_326 = None
    unsqueeze_218: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_327, 0);  mul_327 = None
    unsqueeze_219: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, 2);  unsqueeze_218 = None
    unsqueeze_220: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_219, 3);  unsqueeze_219 = None
    mul_328: "f32[480]" = torch.ops.aten.mul.Tensor(squeeze_109, primals_110);  primals_110 = None
    unsqueeze_221: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_328, 0);  mul_328 = None
    unsqueeze_222: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_221, 2);  unsqueeze_221 = None
    unsqueeze_223: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, 3);  unsqueeze_222 = None
    mul_329: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_220);  sub_57 = unsqueeze_220 = None
    sub_59: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(where_4, mul_329);  where_4 = mul_329 = None
    sub_60: "f32[8, 480, 14, 14]" = torch.ops.aten.sub.Tensor(sub_59, unsqueeze_217);  sub_59 = unsqueeze_217 = None
    mul_330: "f32[8, 480, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_223);  sub_60 = unsqueeze_223 = None
    mul_331: "f32[480]" = torch.ops.aten.mul.Tensor(sum_11, squeeze_109);  sum_11 = squeeze_109 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_330, cat_5, primals_109, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_330 = cat_5 = primals_109 = None
    getitem_94: "f32[8, 912, 14, 14]" = convolution_backward_4[0]
    getitem_95: "f32[480, 912, 1, 1]" = convolution_backward_4[1];  convolution_backward_4 = None
    slice_1: "f32[8, 304, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_94, 1, 0, 304)
    slice_2: "f32[8, 152, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_94, 1, 304, 456)
    slice_3: "f32[8, 152, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_94, 1, 456, 608)
    slice_4: "f32[8, 304, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_94, 1, 608, 912);  getitem_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    where_5: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_5, full_default, slice_3);  le_5 = slice_3 = None
    sum_12: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    sub_61: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_226);  convolution_35 = unsqueeze_226 = None
    mul_332: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_5, sub_61)
    sum_13: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 2, 3]);  mul_332 = None
    mul_333: "f32[152]" = torch.ops.aten.mul.Tensor(sum_12, 0.0006377551020408163)
    unsqueeze_227: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_333, 0);  mul_333 = None
    unsqueeze_228: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_227, 2);  unsqueeze_227 = None
    unsqueeze_229: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 3);  unsqueeze_228 = None
    mul_334: "f32[152]" = torch.ops.aten.mul.Tensor(sum_13, 0.0006377551020408163)
    mul_335: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_106, squeeze_106)
    mul_336: "f32[152]" = torch.ops.aten.mul.Tensor(mul_334, mul_335);  mul_334 = mul_335 = None
    unsqueeze_230: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_336, 0);  mul_336 = None
    unsqueeze_231: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 2);  unsqueeze_230 = None
    unsqueeze_232: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_231, 3);  unsqueeze_231 = None
    mul_337: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_106, primals_107);  primals_107 = None
    unsqueeze_233: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_337, 0);  mul_337 = None
    unsqueeze_234: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_233, 2);  unsqueeze_233 = None
    unsqueeze_235: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 3);  unsqueeze_234 = None
    mul_338: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_232);  sub_61 = unsqueeze_232 = None
    sub_63: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_5, mul_338);  where_5 = mul_338 = None
    sub_64: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_63, unsqueeze_229);  sub_63 = unsqueeze_229 = None
    mul_339: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_235);  sub_64 = unsqueeze_235 = None
    mul_340: "f32[152]" = torch.ops.aten.mul.Tensor(sum_13, squeeze_106);  sum_13 = squeeze_106 = None
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_339, relu_34, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_339 = primals_106 = None
    getitem_97: "f32[8, 304, 14, 14]" = convolution_backward_5[0]
    getitem_98: "f32[152, 304, 3, 3]" = convolution_backward_5[1];  convolution_backward_5 = None
    le_6: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
    where_6: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_6, full_default, getitem_97);  le_6 = getitem_97 = None
    sum_14: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_6, [0, 2, 3])
    sub_65: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_238);  convolution_34 = unsqueeze_238 = None
    mul_341: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_6, sub_65)
    sum_15: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_341, [0, 2, 3]);  mul_341 = None
    mul_342: "f32[304]" = torch.ops.aten.mul.Tensor(sum_14, 0.0006377551020408163)
    unsqueeze_239: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_342, 0);  mul_342 = None
    unsqueeze_240: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_239, 2);  unsqueeze_239 = None
    unsqueeze_241: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, 3);  unsqueeze_240 = None
    mul_343: "f32[304]" = torch.ops.aten.mul.Tensor(sum_15, 0.0006377551020408163)
    mul_344: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_103, squeeze_103)
    mul_345: "f32[304]" = torch.ops.aten.mul.Tensor(mul_343, mul_344);  mul_343 = mul_344 = None
    unsqueeze_242: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_345, 0);  mul_345 = None
    unsqueeze_243: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 2);  unsqueeze_242 = None
    unsqueeze_244: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_243, 3);  unsqueeze_243 = None
    mul_346: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_103, primals_104);  primals_104 = None
    unsqueeze_245: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_346, 0);  mul_346 = None
    unsqueeze_246: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_245, 2);  unsqueeze_245 = None
    unsqueeze_247: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, 3);  unsqueeze_246 = None
    mul_347: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_244);  sub_65 = unsqueeze_244 = None
    sub_67: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_6, mul_347);  where_6 = mul_347 = None
    sub_68: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_67, unsqueeze_241);  sub_67 = unsqueeze_241 = None
    mul_348: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_247);  sub_68 = unsqueeze_247 = None
    mul_349: "f32[304]" = torch.ops.aten.mul.Tensor(sum_15, squeeze_103);  sum_15 = squeeze_103 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_348, relu_33, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_348 = primals_103 = None
    getitem_100: "f32[8, 152, 14, 14]" = convolution_backward_6[0]
    getitem_101: "f32[304, 152, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    add_205: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(slice_2, getitem_100);  slice_2 = getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    le_7: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
    where_7: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_7, full_default, add_205);  le_7 = add_205 = None
    sum_16: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_69: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_250);  convolution_33 = unsqueeze_250 = None
    mul_350: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_7, sub_69)
    sum_17: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 2, 3]);  mul_350 = None
    mul_351: "f32[152]" = torch.ops.aten.mul.Tensor(sum_16, 0.0006377551020408163)
    unsqueeze_251: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_351, 0);  mul_351 = None
    unsqueeze_252: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 2);  unsqueeze_251 = None
    unsqueeze_253: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, 3);  unsqueeze_252 = None
    mul_352: "f32[152]" = torch.ops.aten.mul.Tensor(sum_17, 0.0006377551020408163)
    mul_353: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_100, squeeze_100)
    mul_354: "f32[152]" = torch.ops.aten.mul.Tensor(mul_352, mul_353);  mul_352 = mul_353 = None
    unsqueeze_254: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_354, 0);  mul_354 = None
    unsqueeze_255: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, 2);  unsqueeze_254 = None
    unsqueeze_256: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 3);  unsqueeze_255 = None
    mul_355: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_100, primals_101);  primals_101 = None
    unsqueeze_257: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_355, 0);  mul_355 = None
    unsqueeze_258: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_257, 2);  unsqueeze_257 = None
    unsqueeze_259: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, 3);  unsqueeze_258 = None
    mul_356: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_256);  sub_69 = unsqueeze_256 = None
    sub_71: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_7, mul_356);  where_7 = mul_356 = None
    sub_72: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_71, unsqueeze_253);  sub_71 = unsqueeze_253 = None
    mul_357: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_259);  sub_72 = unsqueeze_259 = None
    mul_358: "f32[152]" = torch.ops.aten.mul.Tensor(sum_17, squeeze_100);  sum_17 = squeeze_100 = None
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(mul_357, relu_32, primals_100, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_357 = primals_100 = None
    getitem_103: "f32[8, 304, 14, 14]" = convolution_backward_7[0]
    getitem_104: "f32[152, 304, 3, 3]" = convolution_backward_7[1];  convolution_backward_7 = None
    le_8: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
    where_8: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_8, full_default, getitem_103);  le_8 = getitem_103 = None
    sum_18: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_8, [0, 2, 3])
    sub_73: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_262);  convolution_32 = unsqueeze_262 = None
    mul_359: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_8, sub_73)
    sum_19: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 2, 3]);  mul_359 = None
    mul_360: "f32[304]" = torch.ops.aten.mul.Tensor(sum_18, 0.0006377551020408163)
    unsqueeze_263: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_360, 0);  mul_360 = None
    unsqueeze_264: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 2);  unsqueeze_263 = None
    unsqueeze_265: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, 3);  unsqueeze_264 = None
    mul_361: "f32[304]" = torch.ops.aten.mul.Tensor(sum_19, 0.0006377551020408163)
    mul_362: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_97, squeeze_97)
    mul_363: "f32[304]" = torch.ops.aten.mul.Tensor(mul_361, mul_362);  mul_361 = mul_362 = None
    unsqueeze_266: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_363, 0);  mul_363 = None
    unsqueeze_267: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, 2);  unsqueeze_266 = None
    unsqueeze_268: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 3);  unsqueeze_267 = None
    mul_364: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_97, primals_98);  primals_98 = None
    unsqueeze_269: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_364, 0);  mul_364 = None
    unsqueeze_270: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_269, 2);  unsqueeze_269 = None
    unsqueeze_271: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, 3);  unsqueeze_270 = None
    mul_365: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_268);  sub_73 = unsqueeze_268 = None
    sub_75: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_8, mul_365);  where_8 = mul_365 = None
    sub_76: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_75, unsqueeze_265);  sub_75 = unsqueeze_265 = None
    mul_366: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_271);  sub_76 = unsqueeze_271 = None
    mul_367: "f32[304]" = torch.ops.aten.mul.Tensor(sum_19, squeeze_97);  sum_19 = squeeze_97 = None
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(mul_366, relu_31, primals_97, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_366 = primals_97 = None
    getitem_106: "f32[8, 304, 14, 14]" = convolution_backward_8[0]
    getitem_107: "f32[304, 304, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    add_206: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(slice_1, getitem_106);  slice_1 = getitem_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    le_9: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
    where_9: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_9, full_default, add_206);  le_9 = add_206 = None
    sum_20: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_77: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_274);  convolution_31 = unsqueeze_274 = None
    mul_368: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_9, sub_77)
    sum_21: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_368, [0, 2, 3]);  mul_368 = None
    mul_369: "f32[304]" = torch.ops.aten.mul.Tensor(sum_20, 0.0006377551020408163)
    unsqueeze_275: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_369, 0);  mul_369 = None
    unsqueeze_276: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_275, 2);  unsqueeze_275 = None
    unsqueeze_277: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 3);  unsqueeze_276 = None
    mul_370: "f32[304]" = torch.ops.aten.mul.Tensor(sum_21, 0.0006377551020408163)
    mul_371: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_94, squeeze_94)
    mul_372: "f32[304]" = torch.ops.aten.mul.Tensor(mul_370, mul_371);  mul_370 = mul_371 = None
    unsqueeze_278: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_372, 0);  mul_372 = None
    unsqueeze_279: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 2);  unsqueeze_278 = None
    unsqueeze_280: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_279, 3);  unsqueeze_279 = None
    mul_373: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_94, primals_95);  primals_95 = None
    unsqueeze_281: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_373, 0);  mul_373 = None
    unsqueeze_282: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_281, 2);  unsqueeze_281 = None
    unsqueeze_283: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, 3);  unsqueeze_282 = None
    mul_374: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_280);  sub_77 = unsqueeze_280 = None
    sub_79: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_9, mul_374);  where_9 = mul_374 = None
    sub_80: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_79, unsqueeze_277);  sub_79 = unsqueeze_277 = None
    mul_375: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_283);  sub_80 = unsqueeze_283 = None
    mul_376: "f32[304]" = torch.ops.aten.mul.Tensor(sum_21, squeeze_94);  sum_21 = squeeze_94 = None
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_375, relu_30, primals_94, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_375 = primals_94 = None
    getitem_109: "f32[8, 304, 14, 14]" = convolution_backward_9[0]
    getitem_110: "f32[304, 304, 3, 3]" = convolution_backward_9[1];  convolution_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    add_207: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(slice_4, getitem_109);  slice_4 = getitem_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:104, code: out = self.conv6(torch.cat([d1, d2, d3], 1))
    le_10: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
    where_10: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_10, full_default, add_207);  le_10 = add_207 = None
    sum_22: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    sub_81: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_286);  convolution_30 = unsqueeze_286 = None
    mul_377: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_10, sub_81)
    sum_23: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_377, [0, 2, 3]);  mul_377 = None
    mul_378: "f32[304]" = torch.ops.aten.mul.Tensor(sum_22, 0.0006377551020408163)
    unsqueeze_287: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_378, 0);  mul_378 = None
    unsqueeze_288: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_287, 2);  unsqueeze_287 = None
    unsqueeze_289: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 3);  unsqueeze_288 = None
    mul_379: "f32[304]" = torch.ops.aten.mul.Tensor(sum_23, 0.0006377551020408163)
    mul_380: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_91, squeeze_91)
    mul_381: "f32[304]" = torch.ops.aten.mul.Tensor(mul_379, mul_380);  mul_379 = mul_380 = None
    unsqueeze_290: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_381, 0);  mul_381 = None
    unsqueeze_291: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, 2);  unsqueeze_290 = None
    unsqueeze_292: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_291, 3);  unsqueeze_291 = None
    mul_382: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_91, primals_92);  primals_92 = None
    unsqueeze_293: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_382, 0);  mul_382 = None
    unsqueeze_294: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_293, 2);  unsqueeze_293 = None
    unsqueeze_295: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, 3);  unsqueeze_294 = None
    mul_383: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_292);  sub_81 = unsqueeze_292 = None
    sub_83: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_10, mul_383);  where_10 = mul_383 = None
    sub_84: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_83, unsqueeze_289);  sub_83 = unsqueeze_289 = None
    mul_384: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_295);  sub_84 = unsqueeze_295 = None
    mul_385: "f32[304]" = torch.ops.aten.mul.Tensor(sum_23, squeeze_91);  sum_23 = squeeze_91 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_384, cat_4, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_384 = cat_4 = primals_91 = None
    getitem_112: "f32[8, 608, 14, 14]" = convolution_backward_10[0]
    getitem_113: "f32[304, 608, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    slice_5: "f32[8, 304, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_112, 1, 0, 304)
    slice_6: "f32[8, 152, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_112, 1, 304, 456)
    slice_7: "f32[8, 152, 14, 14]" = torch.ops.aten.slice.Tensor(getitem_112, 1, 456, 608);  getitem_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    where_11: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_11, full_default, slice_7);  le_11 = slice_7 = None
    sum_24: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    sub_85: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_298);  convolution_29 = unsqueeze_298 = None
    mul_386: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_11, sub_85)
    sum_25: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_386, [0, 2, 3]);  mul_386 = None
    mul_387: "f32[152]" = torch.ops.aten.mul.Tensor(sum_24, 0.0006377551020408163)
    unsqueeze_299: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_387, 0);  mul_387 = None
    unsqueeze_300: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_299, 2);  unsqueeze_299 = None
    unsqueeze_301: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, 3);  unsqueeze_300 = None
    mul_388: "f32[152]" = torch.ops.aten.mul.Tensor(sum_25, 0.0006377551020408163)
    mul_389: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_88, squeeze_88)
    mul_390: "f32[152]" = torch.ops.aten.mul.Tensor(mul_388, mul_389);  mul_388 = mul_389 = None
    unsqueeze_302: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_390, 0);  mul_390 = None
    unsqueeze_303: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, 2);  unsqueeze_302 = None
    unsqueeze_304: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 3);  unsqueeze_303 = None
    mul_391: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_88, primals_89);  primals_89 = None
    unsqueeze_305: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_391, 0);  mul_391 = None
    unsqueeze_306: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 2);  unsqueeze_305 = None
    unsqueeze_307: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, 3);  unsqueeze_306 = None
    mul_392: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_304);  sub_85 = unsqueeze_304 = None
    sub_87: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_11, mul_392);  where_11 = mul_392 = None
    sub_88: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_87, unsqueeze_301);  sub_87 = unsqueeze_301 = None
    mul_393: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_307);  sub_88 = unsqueeze_307 = None
    mul_394: "f32[152]" = torch.ops.aten.mul.Tensor(sum_25, squeeze_88);  sum_25 = squeeze_88 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_393, relu_28, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_393 = primals_88 = None
    getitem_115: "f32[8, 304, 14, 14]" = convolution_backward_11[0]
    getitem_116: "f32[152, 304, 3, 3]" = convolution_backward_11[1];  convolution_backward_11 = None
    le_12: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
    where_12: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_12, full_default, getitem_115);  le_12 = getitem_115 = None
    sum_26: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_12, [0, 2, 3])
    sub_89: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_310);  convolution_28 = unsqueeze_310 = None
    mul_395: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_12, sub_89)
    sum_27: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_395, [0, 2, 3]);  mul_395 = None
    mul_396: "f32[304]" = torch.ops.aten.mul.Tensor(sum_26, 0.0006377551020408163)
    unsqueeze_311: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_396, 0);  mul_396 = None
    unsqueeze_312: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_311, 2);  unsqueeze_311 = None
    unsqueeze_313: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, 3);  unsqueeze_312 = None
    mul_397: "f32[304]" = torch.ops.aten.mul.Tensor(sum_27, 0.0006377551020408163)
    mul_398: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_85, squeeze_85)
    mul_399: "f32[304]" = torch.ops.aten.mul.Tensor(mul_397, mul_398);  mul_397 = mul_398 = None
    unsqueeze_314: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_399, 0);  mul_399 = None
    unsqueeze_315: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, 2);  unsqueeze_314 = None
    unsqueeze_316: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_315, 3);  unsqueeze_315 = None
    mul_400: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_85, primals_86);  primals_86 = None
    unsqueeze_317: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_400, 0);  mul_400 = None
    unsqueeze_318: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 2);  unsqueeze_317 = None
    unsqueeze_319: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, 3);  unsqueeze_318 = None
    mul_401: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_316);  sub_89 = unsqueeze_316 = None
    sub_91: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_12, mul_401);  where_12 = mul_401 = None
    sub_92: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_91, unsqueeze_313);  sub_91 = unsqueeze_313 = None
    mul_402: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_319);  sub_92 = unsqueeze_319 = None
    mul_403: "f32[304]" = torch.ops.aten.mul.Tensor(sum_27, squeeze_85);  sum_27 = squeeze_85 = None
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(mul_402, relu_27, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_402 = primals_85 = None
    getitem_118: "f32[8, 152, 14, 14]" = convolution_backward_12[0]
    getitem_119: "f32[304, 152, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    add_208: "f32[8, 152, 14, 14]" = torch.ops.aten.add.Tensor(slice_6, getitem_118);  slice_6 = getitem_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    le_13: "b8[8, 152, 14, 14]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
    where_13: "f32[8, 152, 14, 14]" = torch.ops.aten.where.self(le_13, full_default, add_208);  le_13 = add_208 = None
    sum_28: "f32[152]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_93: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_322);  convolution_27 = unsqueeze_322 = None
    mul_404: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(where_13, sub_93)
    sum_29: "f32[152]" = torch.ops.aten.sum.dim_IntList(mul_404, [0, 2, 3]);  mul_404 = None
    mul_405: "f32[152]" = torch.ops.aten.mul.Tensor(sum_28, 0.0006377551020408163)
    unsqueeze_323: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_324: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_323, 2);  unsqueeze_323 = None
    unsqueeze_325: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, 3);  unsqueeze_324 = None
    mul_406: "f32[152]" = torch.ops.aten.mul.Tensor(sum_29, 0.0006377551020408163)
    mul_407: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_82, squeeze_82)
    mul_408: "f32[152]" = torch.ops.aten.mul.Tensor(mul_406, mul_407);  mul_406 = mul_407 = None
    unsqueeze_326: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_408, 0);  mul_408 = None
    unsqueeze_327: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 2);  unsqueeze_326 = None
    unsqueeze_328: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_327, 3);  unsqueeze_327 = None
    mul_409: "f32[152]" = torch.ops.aten.mul.Tensor(squeeze_82, primals_83);  primals_83 = None
    unsqueeze_329: "f32[1, 152]" = torch.ops.aten.unsqueeze.default(mul_409, 0);  mul_409 = None
    unsqueeze_330: "f32[1, 152, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_329, 2);  unsqueeze_329 = None
    unsqueeze_331: "f32[1, 152, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 3);  unsqueeze_330 = None
    mul_410: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_328);  sub_93 = unsqueeze_328 = None
    sub_95: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(where_13, mul_410);  where_13 = mul_410 = None
    sub_96: "f32[8, 152, 14, 14]" = torch.ops.aten.sub.Tensor(sub_95, unsqueeze_325);  sub_95 = unsqueeze_325 = None
    mul_411: "f32[8, 152, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_331);  sub_96 = unsqueeze_331 = None
    mul_412: "f32[152]" = torch.ops.aten.mul.Tensor(sum_29, squeeze_82);  sum_29 = squeeze_82 = None
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(mul_411, relu_26, primals_82, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_411 = primals_82 = None
    getitem_121: "f32[8, 304, 14, 14]" = convolution_backward_13[0]
    getitem_122: "f32[152, 304, 3, 3]" = convolution_backward_13[1];  convolution_backward_13 = None
    le_14: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
    where_14: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_14, full_default, getitem_121);  le_14 = getitem_121 = None
    sum_30: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_14, [0, 2, 3])
    sub_97: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_334);  convolution_26 = unsqueeze_334 = None
    mul_413: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_14, sub_97)
    sum_31: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_413, [0, 2, 3]);  mul_413 = None
    mul_414: "f32[304]" = torch.ops.aten.mul.Tensor(sum_30, 0.0006377551020408163)
    unsqueeze_335: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
    unsqueeze_336: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_335, 2);  unsqueeze_335 = None
    unsqueeze_337: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, 3);  unsqueeze_336 = None
    mul_415: "f32[304]" = torch.ops.aten.mul.Tensor(sum_31, 0.0006377551020408163)
    mul_416: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_79, squeeze_79)
    mul_417: "f32[304]" = torch.ops.aten.mul.Tensor(mul_415, mul_416);  mul_415 = mul_416 = None
    unsqueeze_338: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_417, 0);  mul_417 = None
    unsqueeze_339: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 2);  unsqueeze_338 = None
    unsqueeze_340: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_339, 3);  unsqueeze_339 = None
    mul_418: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_79, primals_80);  primals_80 = None
    unsqueeze_341: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_418, 0);  mul_418 = None
    unsqueeze_342: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_341, 2);  unsqueeze_341 = None
    unsqueeze_343: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 3);  unsqueeze_342 = None
    mul_419: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_340);  sub_97 = unsqueeze_340 = None
    sub_99: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_14, mul_419);  where_14 = mul_419 = None
    sub_100: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_99, unsqueeze_337);  sub_99 = unsqueeze_337 = None
    mul_420: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_343);  sub_100 = unsqueeze_343 = None
    mul_421: "f32[304]" = torch.ops.aten.mul.Tensor(sum_31, squeeze_79);  sum_31 = squeeze_79 = None
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_420, relu_25, primals_79, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_420 = primals_79 = None
    getitem_124: "f32[8, 304, 14, 14]" = convolution_backward_14[0]
    getitem_125: "f32[304, 304, 1, 1]" = convolution_backward_14[1];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    add_209: "f32[8, 304, 14, 14]" = torch.ops.aten.add.Tensor(slice_5, getitem_124);  slice_5 = getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    le_15: "b8[8, 304, 14, 14]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
    where_15: "f32[8, 304, 14, 14]" = torch.ops.aten.where.self(le_15, full_default, add_209);  le_15 = add_209 = None
    sum_32: "f32[304]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_101: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_346);  convolution_25 = unsqueeze_346 = None
    mul_422: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(where_15, sub_101)
    sum_33: "f32[304]" = torch.ops.aten.sum.dim_IntList(mul_422, [0, 2, 3]);  mul_422 = None
    mul_423: "f32[304]" = torch.ops.aten.mul.Tensor(sum_32, 0.0006377551020408163)
    unsqueeze_347: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_348: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_347, 2);  unsqueeze_347 = None
    unsqueeze_349: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, 3);  unsqueeze_348 = None
    mul_424: "f32[304]" = torch.ops.aten.mul.Tensor(sum_33, 0.0006377551020408163)
    mul_425: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_76, squeeze_76)
    mul_426: "f32[304]" = torch.ops.aten.mul.Tensor(mul_424, mul_425);  mul_424 = mul_425 = None
    unsqueeze_350: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_426, 0);  mul_426 = None
    unsqueeze_351: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, 2);  unsqueeze_350 = None
    unsqueeze_352: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 3);  unsqueeze_351 = None
    mul_427: "f32[304]" = torch.ops.aten.mul.Tensor(squeeze_76, primals_77);  primals_77 = None
    unsqueeze_353: "f32[1, 304]" = torch.ops.aten.unsqueeze.default(mul_427, 0);  mul_427 = None
    unsqueeze_354: "f32[1, 304, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 2);  unsqueeze_353 = None
    unsqueeze_355: "f32[1, 304, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, 3);  unsqueeze_354 = None
    mul_428: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_352);  sub_101 = unsqueeze_352 = None
    sub_103: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(where_15, mul_428);  where_15 = mul_428 = None
    sub_104: "f32[8, 304, 14, 14]" = torch.ops.aten.sub.Tensor(sub_103, unsqueeze_349);  sub_103 = unsqueeze_349 = None
    mul_429: "f32[8, 304, 14, 14]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_355);  sub_104 = unsqueeze_355 = None
    mul_430: "f32[304]" = torch.ops.aten.mul.Tensor(sum_33, squeeze_76);  sum_33 = squeeze_76 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_429, relu_24, primals_76, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_429 = primals_76 = None
    getitem_127: "f32[8, 288, 28, 28]" = convolution_backward_15[0]
    getitem_128: "f32[304, 288, 3, 3]" = convolution_backward_15[1];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:107, code: return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]
    le_16: "b8[8, 288, 28, 28]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
    where_16: "f32[8, 288, 28, 28]" = torch.ops.aten.where.self(le_16, full_default, getitem_127);  le_16 = getitem_127 = None
    sum_34: "f32[288]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    sub_105: "f32[8, 288, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_358);  convolution_24 = unsqueeze_358 = None
    mul_431: "f32[8, 288, 28, 28]" = torch.ops.aten.mul.Tensor(where_16, sub_105)
    sum_35: "f32[288]" = torch.ops.aten.sum.dim_IntList(mul_431, [0, 2, 3]);  mul_431 = None
    mul_432: "f32[288]" = torch.ops.aten.mul.Tensor(sum_34, 0.00015943877551020407)
    unsqueeze_359: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_432, 0);  mul_432 = None
    unsqueeze_360: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 2);  unsqueeze_359 = None
    unsqueeze_361: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, 3);  unsqueeze_360 = None
    mul_433: "f32[288]" = torch.ops.aten.mul.Tensor(sum_35, 0.00015943877551020407)
    mul_434: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_73, squeeze_73)
    mul_435: "f32[288]" = torch.ops.aten.mul.Tensor(mul_433, mul_434);  mul_433 = mul_434 = None
    unsqueeze_362: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_435, 0);  mul_435 = None
    unsqueeze_363: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, 2);  unsqueeze_362 = None
    unsqueeze_364: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 3);  unsqueeze_363 = None
    mul_436: "f32[288]" = torch.ops.aten.mul.Tensor(squeeze_73, primals_74);  primals_74 = None
    unsqueeze_365: "f32[1, 288]" = torch.ops.aten.unsqueeze.default(mul_436, 0);  mul_436 = None
    unsqueeze_366: "f32[1, 288, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_365, 2);  unsqueeze_365 = None
    unsqueeze_367: "f32[1, 288, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, 3);  unsqueeze_366 = None
    mul_437: "f32[8, 288, 28, 28]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_364);  sub_105 = unsqueeze_364 = None
    sub_107: "f32[8, 288, 28, 28]" = torch.ops.aten.sub.Tensor(where_16, mul_437);  where_16 = mul_437 = None
    sub_108: "f32[8, 288, 28, 28]" = torch.ops.aten.sub.Tensor(sub_107, unsqueeze_361);  sub_107 = unsqueeze_361 = None
    mul_438: "f32[8, 288, 28, 28]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_367);  sub_108 = unsqueeze_367 = None
    mul_439: "f32[288]" = torch.ops.aten.mul.Tensor(sum_35, squeeze_73);  sum_35 = squeeze_73 = None
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_438, cat_3, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_438 = cat_3 = primals_73 = None
    getitem_130: "f32[8, 432, 28, 28]" = convolution_backward_16[0]
    getitem_131: "f32[288, 432, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    slice_8: "f32[8, 144, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_130, 1, 0, 144)
    slice_9: "f32[8, 72, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_130, 1, 144, 216)
    slice_10: "f32[8, 72, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_130, 1, 216, 288)
    slice_11: "f32[8, 144, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_130, 1, 288, 432);  getitem_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    where_17: "f32[8, 72, 28, 28]" = torch.ops.aten.where.self(le_17, full_default, slice_10);  le_17 = slice_10 = None
    sum_36: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    sub_109: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_370);  convolution_23 = unsqueeze_370 = None
    mul_440: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_17, sub_109)
    sum_37: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_440, [0, 2, 3]);  mul_440 = None
    mul_441: "f32[72]" = torch.ops.aten.mul.Tensor(sum_36, 0.00015943877551020407)
    unsqueeze_371: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_441, 0);  mul_441 = None
    unsqueeze_372: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_371, 2);  unsqueeze_371 = None
    unsqueeze_373: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, 3);  unsqueeze_372 = None
    mul_442: "f32[72]" = torch.ops.aten.mul.Tensor(sum_37, 0.00015943877551020407)
    mul_443: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_70, squeeze_70)
    mul_444: "f32[72]" = torch.ops.aten.mul.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    unsqueeze_374: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_444, 0);  mul_444 = None
    unsqueeze_375: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, 2);  unsqueeze_374 = None
    unsqueeze_376: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_375, 3);  unsqueeze_375 = None
    mul_445: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_70, primals_71);  primals_71 = None
    unsqueeze_377: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_445, 0);  mul_445 = None
    unsqueeze_378: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    mul_446: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_376);  sub_109 = unsqueeze_376 = None
    sub_111: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_17, mul_446);  where_17 = mul_446 = None
    sub_112: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_111, unsqueeze_373);  sub_111 = unsqueeze_373 = None
    mul_447: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_379);  sub_112 = unsqueeze_379 = None
    mul_448: "f32[72]" = torch.ops.aten.mul.Tensor(sum_37, squeeze_70);  sum_37 = squeeze_70 = None
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(mul_447, relu_22, primals_70, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_447 = primals_70 = None
    getitem_133: "f32[8, 144, 28, 28]" = convolution_backward_17[0]
    getitem_134: "f32[72, 144, 3, 3]" = convolution_backward_17[1];  convolution_backward_17 = None
    le_18: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
    where_18: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_18, full_default, getitem_133);  le_18 = getitem_133 = None
    sum_38: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_18, [0, 2, 3])
    sub_113: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_382);  convolution_22 = unsqueeze_382 = None
    mul_449: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_18, sub_113)
    sum_39: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_449, [0, 2, 3]);  mul_449 = None
    mul_450: "f32[144]" = torch.ops.aten.mul.Tensor(sum_38, 0.00015943877551020407)
    unsqueeze_383: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_384: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_383, 2);  unsqueeze_383 = None
    unsqueeze_385: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 3);  unsqueeze_384 = None
    mul_451: "f32[144]" = torch.ops.aten.mul.Tensor(sum_39, 0.00015943877551020407)
    mul_452: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_67, squeeze_67)
    mul_453: "f32[144]" = torch.ops.aten.mul.Tensor(mul_451, mul_452);  mul_451 = mul_452 = None
    unsqueeze_386: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_453, 0);  mul_453 = None
    unsqueeze_387: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, 2);  unsqueeze_386 = None
    unsqueeze_388: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_387, 3);  unsqueeze_387 = None
    mul_454: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_67, primals_68);  primals_68 = None
    unsqueeze_389: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_454, 0);  mul_454 = None
    unsqueeze_390: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    mul_455: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_388);  sub_113 = unsqueeze_388 = None
    sub_115: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_18, mul_455);  where_18 = mul_455 = None
    sub_116: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_115, unsqueeze_385);  sub_115 = unsqueeze_385 = None
    mul_456: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_391);  sub_116 = unsqueeze_391 = None
    mul_457: "f32[144]" = torch.ops.aten.mul.Tensor(sum_39, squeeze_67);  sum_39 = squeeze_67 = None
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(mul_456, relu_21, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_456 = primals_67 = None
    getitem_136: "f32[8, 72, 28, 28]" = convolution_backward_18[0]
    getitem_137: "f32[144, 72, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    add_210: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(slice_9, getitem_136);  slice_9 = getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    le_19: "b8[8, 72, 28, 28]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
    where_19: "f32[8, 72, 28, 28]" = torch.ops.aten.where.self(le_19, full_default, add_210);  le_19 = add_210 = None
    sum_40: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_117: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_394);  convolution_21 = unsqueeze_394 = None
    mul_458: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_19, sub_117)
    sum_41: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_458, [0, 2, 3]);  mul_458 = None
    mul_459: "f32[72]" = torch.ops.aten.mul.Tensor(sum_40, 0.00015943877551020407)
    unsqueeze_395: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_459, 0);  mul_459 = None
    unsqueeze_396: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_395, 2);  unsqueeze_395 = None
    unsqueeze_397: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, 3);  unsqueeze_396 = None
    mul_460: "f32[72]" = torch.ops.aten.mul.Tensor(sum_41, 0.00015943877551020407)
    mul_461: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_64, squeeze_64)
    mul_462: "f32[72]" = torch.ops.aten.mul.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    unsqueeze_398: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_462, 0);  mul_462 = None
    unsqueeze_399: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, 2);  unsqueeze_398 = None
    unsqueeze_400: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_399, 3);  unsqueeze_399 = None
    mul_463: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_64, primals_65);  primals_65 = None
    unsqueeze_401: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_463, 0);  mul_463 = None
    unsqueeze_402: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    mul_464: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_400);  sub_117 = unsqueeze_400 = None
    sub_119: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_19, mul_464);  where_19 = mul_464 = None
    sub_120: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_119, unsqueeze_397);  sub_119 = unsqueeze_397 = None
    mul_465: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_403);  sub_120 = unsqueeze_403 = None
    mul_466: "f32[72]" = torch.ops.aten.mul.Tensor(sum_41, squeeze_64);  sum_41 = squeeze_64 = None
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_465, relu_20, primals_64, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_465 = primals_64 = None
    getitem_139: "f32[8, 144, 28, 28]" = convolution_backward_19[0]
    getitem_140: "f32[72, 144, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    le_20: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
    where_20: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_20, full_default, getitem_139);  le_20 = getitem_139 = None
    sum_42: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_20, [0, 2, 3])
    sub_121: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_406);  convolution_20 = unsqueeze_406 = None
    mul_467: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_20, sub_121)
    sum_43: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_467, [0, 2, 3]);  mul_467 = None
    mul_468: "f32[144]" = torch.ops.aten.mul.Tensor(sum_42, 0.00015943877551020407)
    unsqueeze_407: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_468, 0);  mul_468 = None
    unsqueeze_408: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_407, 2);  unsqueeze_407 = None
    unsqueeze_409: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, 3);  unsqueeze_408 = None
    mul_469: "f32[144]" = torch.ops.aten.mul.Tensor(sum_43, 0.00015943877551020407)
    mul_470: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_61, squeeze_61)
    mul_471: "f32[144]" = torch.ops.aten.mul.Tensor(mul_469, mul_470);  mul_469 = mul_470 = None
    unsqueeze_410: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_471, 0);  mul_471 = None
    unsqueeze_411: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, 2);  unsqueeze_410 = None
    unsqueeze_412: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_411, 3);  unsqueeze_411 = None
    mul_472: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_61, primals_62);  primals_62 = None
    unsqueeze_413: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_472, 0);  mul_472 = None
    unsqueeze_414: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    mul_473: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_412);  sub_121 = unsqueeze_412 = None
    sub_123: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_20, mul_473);  where_20 = mul_473 = None
    sub_124: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_123, unsqueeze_409);  sub_123 = unsqueeze_409 = None
    mul_474: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_415);  sub_124 = unsqueeze_415 = None
    mul_475: "f32[144]" = torch.ops.aten.mul.Tensor(sum_43, squeeze_61);  sum_43 = squeeze_61 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_474, relu_19, primals_61, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_474 = primals_61 = None
    getitem_142: "f32[8, 144, 28, 28]" = convolution_backward_20[0]
    getitem_143: "f32[144, 144, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    add_211: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(slice_8, getitem_142);  slice_8 = getitem_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    le_21: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
    where_21: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_21, full_default, add_211);  le_21 = add_211 = None
    sum_44: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_125: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_418);  convolution_19 = unsqueeze_418 = None
    mul_476: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_21, sub_125)
    sum_45: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 2, 3]);  mul_476 = None
    mul_477: "f32[144]" = torch.ops.aten.mul.Tensor(sum_44, 0.00015943877551020407)
    unsqueeze_419: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_420: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_419, 2);  unsqueeze_419 = None
    unsqueeze_421: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, 3);  unsqueeze_420 = None
    mul_478: "f32[144]" = torch.ops.aten.mul.Tensor(sum_45, 0.00015943877551020407)
    mul_479: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_58, squeeze_58)
    mul_480: "f32[144]" = torch.ops.aten.mul.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
    unsqueeze_422: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_480, 0);  mul_480 = None
    unsqueeze_423: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, 2);  unsqueeze_422 = None
    unsqueeze_424: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_423, 3);  unsqueeze_423 = None
    mul_481: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_58, primals_59);  primals_59 = None
    unsqueeze_425: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_481, 0);  mul_481 = None
    unsqueeze_426: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_482: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_424);  sub_125 = unsqueeze_424 = None
    sub_127: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_21, mul_482);  where_21 = mul_482 = None
    sub_128: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_127, unsqueeze_421);  sub_127 = unsqueeze_421 = None
    mul_483: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_427);  sub_128 = unsqueeze_427 = None
    mul_484: "f32[144]" = torch.ops.aten.mul.Tensor(sum_45, squeeze_58);  sum_45 = squeeze_58 = None
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_483, relu_18, primals_58, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_483 = primals_58 = None
    getitem_145: "f32[8, 144, 28, 28]" = convolution_backward_21[0]
    getitem_146: "f32[144, 144, 3, 3]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    add_212: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(slice_11, getitem_145);  slice_11 = getitem_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:104, code: out = self.conv6(torch.cat([d1, d2, d3], 1))
    le_22: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    where_22: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_22, full_default, add_212);  le_22 = add_212 = None
    sum_46: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    sub_129: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_430);  convolution_18 = unsqueeze_430 = None
    mul_485: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_22, sub_129)
    sum_47: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_485, [0, 2, 3]);  mul_485 = None
    mul_486: "f32[144]" = torch.ops.aten.mul.Tensor(sum_46, 0.00015943877551020407)
    unsqueeze_431: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_486, 0);  mul_486 = None
    unsqueeze_432: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_431, 2);  unsqueeze_431 = None
    unsqueeze_433: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, 3);  unsqueeze_432 = None
    mul_487: "f32[144]" = torch.ops.aten.mul.Tensor(sum_47, 0.00015943877551020407)
    mul_488: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_55, squeeze_55)
    mul_489: "f32[144]" = torch.ops.aten.mul.Tensor(mul_487, mul_488);  mul_487 = mul_488 = None
    unsqueeze_434: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_489, 0);  mul_489 = None
    unsqueeze_435: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 2);  unsqueeze_434 = None
    unsqueeze_436: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_435, 3);  unsqueeze_435 = None
    mul_490: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_55, primals_56);  primals_56 = None
    unsqueeze_437: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_490, 0);  mul_490 = None
    unsqueeze_438: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_491: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_436);  sub_129 = unsqueeze_436 = None
    sub_131: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_22, mul_491);  where_22 = mul_491 = None
    sub_132: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_131, unsqueeze_433);  sub_131 = unsqueeze_433 = None
    mul_492: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_439);  sub_132 = unsqueeze_439 = None
    mul_493: "f32[144]" = torch.ops.aten.mul.Tensor(sum_47, squeeze_55);  sum_47 = squeeze_55 = None
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(mul_492, cat_2, primals_55, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_492 = cat_2 = primals_55 = None
    getitem_148: "f32[8, 288, 28, 28]" = convolution_backward_22[0]
    getitem_149: "f32[144, 288, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    slice_12: "f32[8, 144, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_148, 1, 0, 144)
    slice_13: "f32[8, 72, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_148, 1, 144, 216)
    slice_14: "f32[8, 72, 28, 28]" = torch.ops.aten.slice.Tensor(getitem_148, 1, 216, 288);  getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    where_23: "f32[8, 72, 28, 28]" = torch.ops.aten.where.self(le_23, full_default, slice_14);  le_23 = slice_14 = None
    sum_48: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    sub_133: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_442);  convolution_17 = unsqueeze_442 = None
    mul_494: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_23, sub_133)
    sum_49: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_494, [0, 2, 3]);  mul_494 = None
    mul_495: "f32[72]" = torch.ops.aten.mul.Tensor(sum_48, 0.00015943877551020407)
    unsqueeze_443: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_495, 0);  mul_495 = None
    unsqueeze_444: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_443, 2);  unsqueeze_443 = None
    unsqueeze_445: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, 3);  unsqueeze_444 = None
    mul_496: "f32[72]" = torch.ops.aten.mul.Tensor(sum_49, 0.00015943877551020407)
    mul_497: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_52, squeeze_52)
    mul_498: "f32[72]" = torch.ops.aten.mul.Tensor(mul_496, mul_497);  mul_496 = mul_497 = None
    unsqueeze_446: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_498, 0);  mul_498 = None
    unsqueeze_447: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, 2);  unsqueeze_446 = None
    unsqueeze_448: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_447, 3);  unsqueeze_447 = None
    mul_499: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_52, primals_53);  primals_53 = None
    unsqueeze_449: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_499, 0);  mul_499 = None
    unsqueeze_450: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_500: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_448);  sub_133 = unsqueeze_448 = None
    sub_135: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_23, mul_500);  where_23 = mul_500 = None
    sub_136: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_135, unsqueeze_445);  sub_135 = unsqueeze_445 = None
    mul_501: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_451);  sub_136 = unsqueeze_451 = None
    mul_502: "f32[72]" = torch.ops.aten.mul.Tensor(sum_49, squeeze_52);  sum_49 = squeeze_52 = None
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(mul_501, relu_16, primals_52, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_501 = primals_52 = None
    getitem_151: "f32[8, 144, 28, 28]" = convolution_backward_23[0]
    getitem_152: "f32[72, 144, 3, 3]" = convolution_backward_23[1];  convolution_backward_23 = None
    le_24: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    where_24: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_24, full_default, getitem_151);  le_24 = getitem_151 = None
    sum_50: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_24, [0, 2, 3])
    sub_137: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_454);  convolution_16 = unsqueeze_454 = None
    mul_503: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_24, sub_137)
    sum_51: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_503, [0, 2, 3]);  mul_503 = None
    mul_504: "f32[144]" = torch.ops.aten.mul.Tensor(sum_50, 0.00015943877551020407)
    unsqueeze_455: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_456: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 2);  unsqueeze_455 = None
    unsqueeze_457: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, 3);  unsqueeze_456 = None
    mul_505: "f32[144]" = torch.ops.aten.mul.Tensor(sum_51, 0.00015943877551020407)
    mul_506: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_49, squeeze_49)
    mul_507: "f32[144]" = torch.ops.aten.mul.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
    unsqueeze_458: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_507, 0);  mul_507 = None
    unsqueeze_459: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, 2);  unsqueeze_458 = None
    unsqueeze_460: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 3);  unsqueeze_459 = None
    mul_508: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_49, primals_50);  primals_50 = None
    unsqueeze_461: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_508, 0);  mul_508 = None
    unsqueeze_462: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_509: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_460);  sub_137 = unsqueeze_460 = None
    sub_139: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_24, mul_509);  where_24 = mul_509 = None
    sub_140: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_139, unsqueeze_457);  sub_139 = unsqueeze_457 = None
    mul_510: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_463);  sub_140 = unsqueeze_463 = None
    mul_511: "f32[144]" = torch.ops.aten.mul.Tensor(sum_51, squeeze_49);  sum_51 = squeeze_49 = None
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_510, relu_15, primals_49, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_510 = primals_49 = None
    getitem_154: "f32[8, 72, 28, 28]" = convolution_backward_24[0]
    getitem_155: "f32[144, 72, 1, 1]" = convolution_backward_24[1];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    add_213: "f32[8, 72, 28, 28]" = torch.ops.aten.add.Tensor(slice_13, getitem_154);  slice_13 = getitem_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    le_25: "b8[8, 72, 28, 28]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_25: "f32[8, 72, 28, 28]" = torch.ops.aten.where.self(le_25, full_default, add_213);  le_25 = add_213 = None
    sum_52: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_141: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_466);  convolution_15 = unsqueeze_466 = None
    mul_512: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_25, sub_141)
    sum_53: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 2, 3]);  mul_512 = None
    mul_513: "f32[72]" = torch.ops.aten.mul.Tensor(sum_52, 0.00015943877551020407)
    unsqueeze_467: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_513, 0);  mul_513 = None
    unsqueeze_468: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 2);  unsqueeze_467 = None
    unsqueeze_469: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, 3);  unsqueeze_468 = None
    mul_514: "f32[72]" = torch.ops.aten.mul.Tensor(sum_53, 0.00015943877551020407)
    mul_515: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_46, squeeze_46)
    mul_516: "f32[72]" = torch.ops.aten.mul.Tensor(mul_514, mul_515);  mul_514 = mul_515 = None
    unsqueeze_470: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_516, 0);  mul_516 = None
    unsqueeze_471: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, 2);  unsqueeze_470 = None
    unsqueeze_472: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_471, 3);  unsqueeze_471 = None
    mul_517: "f32[72]" = torch.ops.aten.mul.Tensor(squeeze_46, primals_47);  primals_47 = None
    unsqueeze_473: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_517, 0);  mul_517 = None
    unsqueeze_474: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_518: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_472);  sub_141 = unsqueeze_472 = None
    sub_143: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(where_25, mul_518);  where_25 = mul_518 = None
    sub_144: "f32[8, 72, 28, 28]" = torch.ops.aten.sub.Tensor(sub_143, unsqueeze_469);  sub_143 = unsqueeze_469 = None
    mul_519: "f32[8, 72, 28, 28]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_475);  sub_144 = unsqueeze_475 = None
    mul_520: "f32[72]" = torch.ops.aten.mul.Tensor(sum_53, squeeze_46);  sum_53 = squeeze_46 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_519, relu_14, primals_46, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_519 = primals_46 = None
    getitem_157: "f32[8, 144, 28, 28]" = convolution_backward_25[0]
    getitem_158: "f32[72, 144, 3, 3]" = convolution_backward_25[1];  convolution_backward_25 = None
    le_26: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_26: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_26, full_default, getitem_157);  le_26 = getitem_157 = None
    sum_54: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_26, [0, 2, 3])
    sub_145: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_478);  convolution_14 = unsqueeze_478 = None
    mul_521: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_26, sub_145)
    sum_55: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 2, 3]);  mul_521 = None
    mul_522: "f32[144]" = torch.ops.aten.mul.Tensor(sum_54, 0.00015943877551020407)
    unsqueeze_479: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_522, 0);  mul_522 = None
    unsqueeze_480: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_479, 2);  unsqueeze_479 = None
    unsqueeze_481: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 3);  unsqueeze_480 = None
    mul_523: "f32[144]" = torch.ops.aten.mul.Tensor(sum_55, 0.00015943877551020407)
    mul_524: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_43, squeeze_43)
    mul_525: "f32[144]" = torch.ops.aten.mul.Tensor(mul_523, mul_524);  mul_523 = mul_524 = None
    unsqueeze_482: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_525, 0);  mul_525 = None
    unsqueeze_483: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, 2);  unsqueeze_482 = None
    unsqueeze_484: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_483, 3);  unsqueeze_483 = None
    mul_526: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_43, primals_44);  primals_44 = None
    unsqueeze_485: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_526, 0);  mul_526 = None
    unsqueeze_486: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_527: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_484);  sub_145 = unsqueeze_484 = None
    sub_147: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_26, mul_527);  where_26 = mul_527 = None
    sub_148: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_147, unsqueeze_481);  sub_147 = unsqueeze_481 = None
    mul_528: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_487);  sub_148 = unsqueeze_487 = None
    mul_529: "f32[144]" = torch.ops.aten.mul.Tensor(sum_55, squeeze_43);  sum_55 = squeeze_43 = None
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_528, relu_13, primals_43, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_528 = primals_43 = None
    getitem_160: "f32[8, 144, 28, 28]" = convolution_backward_26[0]
    getitem_161: "f32[144, 144, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    add_214: "f32[8, 144, 28, 28]" = torch.ops.aten.add.Tensor(slice_12, getitem_160);  slice_12 = getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    le_27: "b8[8, 144, 28, 28]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_27: "f32[8, 144, 28, 28]" = torch.ops.aten.where.self(le_27, full_default, add_214);  le_27 = add_214 = None
    sum_56: "f32[144]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_149: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_490);  convolution_13 = unsqueeze_490 = None
    mul_530: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(where_27, sub_149)
    sum_57: "f32[144]" = torch.ops.aten.sum.dim_IntList(mul_530, [0, 2, 3]);  mul_530 = None
    mul_531: "f32[144]" = torch.ops.aten.mul.Tensor(sum_56, 0.00015943877551020407)
    unsqueeze_491: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_531, 0);  mul_531 = None
    unsqueeze_492: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_491, 2);  unsqueeze_491 = None
    unsqueeze_493: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 3);  unsqueeze_492 = None
    mul_532: "f32[144]" = torch.ops.aten.mul.Tensor(sum_57, 0.00015943877551020407)
    mul_533: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_40, squeeze_40)
    mul_534: "f32[144]" = torch.ops.aten.mul.Tensor(mul_532, mul_533);  mul_532 = mul_533 = None
    unsqueeze_494: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_534, 0);  mul_534 = None
    unsqueeze_495: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, 2);  unsqueeze_494 = None
    unsqueeze_496: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_495, 3);  unsqueeze_495 = None
    mul_535: "f32[144]" = torch.ops.aten.mul.Tensor(squeeze_40, primals_41);  primals_41 = None
    unsqueeze_497: "f32[1, 144]" = torch.ops.aten.unsqueeze.default(mul_535, 0);  mul_535 = None
    unsqueeze_498: "f32[1, 144, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 144, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_536: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_496);  sub_149 = unsqueeze_496 = None
    sub_151: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(where_27, mul_536);  where_27 = mul_536 = None
    sub_152: "f32[8, 144, 28, 28]" = torch.ops.aten.sub.Tensor(sub_151, unsqueeze_493);  sub_151 = unsqueeze_493 = None
    mul_537: "f32[8, 144, 28, 28]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_499);  sub_152 = unsqueeze_499 = None
    mul_538: "f32[144]" = torch.ops.aten.mul.Tensor(sum_57, squeeze_40);  sum_57 = squeeze_40 = None
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_537, relu_12, primals_40, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_537 = primals_40 = None
    getitem_163: "f32[8, 128, 56, 56]" = convolution_backward_27[0]
    getitem_164: "f32[144, 128, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:107, code: return [self.conv6(torch.cat([d1, d2, d3, x[1]], 1)), x[1]]
    le_28: "b8[8, 128, 56, 56]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_28: "f32[8, 128, 56, 56]" = torch.ops.aten.where.self(le_28, full_default, getitem_163);  le_28 = getitem_163 = None
    sum_58: "f32[128]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    sub_153: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_502);  convolution_12 = unsqueeze_502 = None
    mul_539: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(where_28, sub_153)
    sum_59: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 2, 3]);  mul_539 = None
    mul_540: "f32[128]" = torch.ops.aten.mul.Tensor(sum_58, 3.985969387755102e-05)
    unsqueeze_503: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_540, 0);  mul_540 = None
    unsqueeze_504: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 2);  unsqueeze_503 = None
    unsqueeze_505: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, 3);  unsqueeze_504 = None
    mul_541: "f32[128]" = torch.ops.aten.mul.Tensor(sum_59, 3.985969387755102e-05)
    mul_542: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, squeeze_37)
    mul_543: "f32[128]" = torch.ops.aten.mul.Tensor(mul_541, mul_542);  mul_541 = mul_542 = None
    unsqueeze_506: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_543, 0);  mul_543 = None
    unsqueeze_507: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, 2);  unsqueeze_506 = None
    unsqueeze_508: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_507, 3);  unsqueeze_507 = None
    mul_544: "f32[128]" = torch.ops.aten.mul.Tensor(squeeze_37, primals_38);  primals_38 = None
    unsqueeze_509: "f32[1, 128]" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
    unsqueeze_510: "f32[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_545: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_508);  sub_153 = unsqueeze_508 = None
    sub_155: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(where_28, mul_545);  where_28 = mul_545 = None
    sub_156: "f32[8, 128, 56, 56]" = torch.ops.aten.sub.Tensor(sub_155, unsqueeze_505);  sub_155 = unsqueeze_505 = None
    mul_546: "f32[8, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_156, unsqueeze_511);  sub_156 = unsqueeze_511 = None
    mul_547: "f32[128]" = torch.ops.aten.mul.Tensor(sum_59, squeeze_37);  sum_59 = squeeze_37 = None
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_546, cat_1, primals_37, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_546 = cat_1 = primals_37 = None
    getitem_166: "f32[8, 192, 56, 56]" = convolution_backward_28[0]
    getitem_167: "f32[128, 192, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    slice_15: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_166, 1, 0, 64)
    slice_16: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_166, 1, 64, 96)
    slice_17: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_166, 1, 96, 128)
    slice_18: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_166, 1, 128, 192);  getitem_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    where_29: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_29, full_default, slice_17);  le_29 = slice_17 = None
    sum_60: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    sub_157: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_514);  convolution_11 = unsqueeze_514 = None
    mul_548: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_29, sub_157)
    sum_61: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_548, [0, 2, 3]);  mul_548 = None
    mul_549: "f32[32]" = torch.ops.aten.mul.Tensor(sum_60, 3.985969387755102e-05)
    unsqueeze_515: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_549, 0);  mul_549 = None
    unsqueeze_516: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_515, 2);  unsqueeze_515 = None
    unsqueeze_517: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, 3);  unsqueeze_516 = None
    mul_550: "f32[32]" = torch.ops.aten.mul.Tensor(sum_61, 3.985969387755102e-05)
    mul_551: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_34, squeeze_34)
    mul_552: "f32[32]" = torch.ops.aten.mul.Tensor(mul_550, mul_551);  mul_550 = mul_551 = None
    unsqueeze_518: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_519: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, 2);  unsqueeze_518 = None
    unsqueeze_520: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_519, 3);  unsqueeze_519 = None
    mul_553: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_34, primals_35);  primals_35 = None
    unsqueeze_521: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_553, 0);  mul_553 = None
    unsqueeze_522: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_554: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_157, unsqueeze_520);  sub_157 = unsqueeze_520 = None
    sub_159: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_29, mul_554);  where_29 = mul_554 = None
    sub_160: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_159, unsqueeze_517);  sub_159 = unsqueeze_517 = None
    mul_555: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_160, unsqueeze_523);  sub_160 = unsqueeze_523 = None
    mul_556: "f32[32]" = torch.ops.aten.mul.Tensor(sum_61, squeeze_34);  sum_61 = squeeze_34 = None
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_555, relu_10, primals_34, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_555 = primals_34 = None
    getitem_169: "f32[8, 64, 56, 56]" = convolution_backward_29[0]
    getitem_170: "f32[32, 64, 3, 3]" = convolution_backward_29[1];  convolution_backward_29 = None
    le_30: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_30: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_30, full_default, getitem_169);  le_30 = getitem_169 = None
    sum_62: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_30, [0, 2, 3])
    sub_161: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_526);  convolution_10 = unsqueeze_526 = None
    mul_557: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_30, sub_161)
    sum_63: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_557, [0, 2, 3]);  mul_557 = None
    mul_558: "f32[64]" = torch.ops.aten.mul.Tensor(sum_62, 3.985969387755102e-05)
    unsqueeze_527: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_558, 0);  mul_558 = None
    unsqueeze_528: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_527, 2);  unsqueeze_527 = None
    unsqueeze_529: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 3);  unsqueeze_528 = None
    mul_559: "f32[64]" = torch.ops.aten.mul.Tensor(sum_63, 3.985969387755102e-05)
    mul_560: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, squeeze_31)
    mul_561: "f32[64]" = torch.ops.aten.mul.Tensor(mul_559, mul_560);  mul_559 = mul_560 = None
    unsqueeze_530: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_561, 0);  mul_561 = None
    unsqueeze_531: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 2);  unsqueeze_530 = None
    unsqueeze_532: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_531, 3);  unsqueeze_531 = None
    mul_562: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_31, primals_32);  primals_32 = None
    unsqueeze_533: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_562, 0);  mul_562 = None
    unsqueeze_534: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_563: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_161, unsqueeze_532);  sub_161 = unsqueeze_532 = None
    sub_163: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_30, mul_563);  where_30 = mul_563 = None
    sub_164: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_163, unsqueeze_529);  sub_163 = unsqueeze_529 = None
    mul_564: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_164, unsqueeze_535);  sub_164 = unsqueeze_535 = None
    mul_565: "f32[64]" = torch.ops.aten.mul.Tensor(sum_63, squeeze_31);  sum_63 = squeeze_31 = None
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_564, relu_9, primals_31, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_564 = primals_31 = None
    getitem_172: "f32[8, 32, 56, 56]" = convolution_backward_30[0]
    getitem_173: "f32[64, 32, 1, 1]" = convolution_backward_30[1];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    add_215: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_16, getitem_172);  slice_16 = getitem_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    le_31: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_31: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_31, full_default, add_215);  le_31 = add_215 = None
    sum_64: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_165: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_538);  convolution_9 = unsqueeze_538 = None
    mul_566: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_31, sub_165)
    sum_65: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 2, 3]);  mul_566 = None
    mul_567: "f32[32]" = torch.ops.aten.mul.Tensor(sum_64, 3.985969387755102e-05)
    unsqueeze_539: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_567, 0);  mul_567 = None
    unsqueeze_540: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_539, 2);  unsqueeze_539 = None
    unsqueeze_541: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, 3);  unsqueeze_540 = None
    mul_568: "f32[32]" = torch.ops.aten.mul.Tensor(sum_65, 3.985969387755102e-05)
    mul_569: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_28, squeeze_28)
    mul_570: "f32[32]" = torch.ops.aten.mul.Tensor(mul_568, mul_569);  mul_568 = mul_569 = None
    unsqueeze_542: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_570, 0);  mul_570 = None
    unsqueeze_543: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 2);  unsqueeze_542 = None
    unsqueeze_544: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_543, 3);  unsqueeze_543 = None
    mul_571: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_28, primals_29);  primals_29 = None
    unsqueeze_545: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_571, 0);  mul_571 = None
    unsqueeze_546: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_572: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_165, unsqueeze_544);  sub_165 = unsqueeze_544 = None
    sub_167: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_31, mul_572);  where_31 = mul_572 = None
    sub_168: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_167, unsqueeze_541);  sub_167 = unsqueeze_541 = None
    mul_573: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_168, unsqueeze_547);  sub_168 = unsqueeze_547 = None
    mul_574: "f32[32]" = torch.ops.aten.mul.Tensor(sum_65, squeeze_28);  sum_65 = squeeze_28 = None
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_573, relu_8, primals_28, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_573 = primals_28 = None
    getitem_175: "f32[8, 64, 56, 56]" = convolution_backward_31[0]
    getitem_176: "f32[32, 64, 3, 3]" = convolution_backward_31[1];  convolution_backward_31 = None
    le_32: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_32: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_32, full_default, getitem_175);  le_32 = getitem_175 = None
    sum_66: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_32, [0, 2, 3])
    sub_169: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_550);  convolution_8 = unsqueeze_550 = None
    mul_575: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_32, sub_169)
    sum_67: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_575, [0, 2, 3]);  mul_575 = None
    mul_576: "f32[64]" = torch.ops.aten.mul.Tensor(sum_66, 3.985969387755102e-05)
    unsqueeze_551: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_576, 0);  mul_576 = None
    unsqueeze_552: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 2);  unsqueeze_551 = None
    unsqueeze_553: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, 3);  unsqueeze_552 = None
    mul_577: "f32[64]" = torch.ops.aten.mul.Tensor(sum_67, 3.985969387755102e-05)
    mul_578: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, squeeze_25)
    mul_579: "f32[64]" = torch.ops.aten.mul.Tensor(mul_577, mul_578);  mul_577 = mul_578 = None
    unsqueeze_554: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_579, 0);  mul_579 = None
    unsqueeze_555: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, 2);  unsqueeze_554 = None
    unsqueeze_556: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 3);  unsqueeze_555 = None
    mul_580: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_25, primals_26);  primals_26 = None
    unsqueeze_557: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_580, 0);  mul_580 = None
    unsqueeze_558: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_581: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_169, unsqueeze_556);  sub_169 = unsqueeze_556 = None
    sub_171: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_32, mul_581);  where_32 = mul_581 = None
    sub_172: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_171, unsqueeze_553);  sub_171 = unsqueeze_553 = None
    mul_582: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_172, unsqueeze_559);  sub_172 = unsqueeze_559 = None
    mul_583: "f32[64]" = torch.ops.aten.mul.Tensor(sum_67, squeeze_25);  sum_67 = squeeze_25 = None
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_582, relu_7, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_582 = primals_25 = None
    getitem_178: "f32[8, 64, 56, 56]" = convolution_backward_32[0]
    getitem_179: "f32[64, 64, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    add_216: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(slice_15, getitem_178);  slice_15 = getitem_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    le_33: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_33: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_33, full_default, add_216);  le_33 = add_216 = None
    sum_68: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_173: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_562);  convolution_7 = unsqueeze_562 = None
    mul_584: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_33, sub_173)
    sum_69: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 2, 3]);  mul_584 = None
    mul_585: "f32[64]" = torch.ops.aten.mul.Tensor(sum_68, 3.985969387755102e-05)
    unsqueeze_563: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_585, 0);  mul_585 = None
    unsqueeze_564: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 2);  unsqueeze_563 = None
    unsqueeze_565: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, 3);  unsqueeze_564 = None
    mul_586: "f32[64]" = torch.ops.aten.mul.Tensor(sum_69, 3.985969387755102e-05)
    mul_587: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, squeeze_22)
    mul_588: "f32[64]" = torch.ops.aten.mul.Tensor(mul_586, mul_587);  mul_586 = mul_587 = None
    unsqueeze_566: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_588, 0);  mul_588 = None
    unsqueeze_567: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, 2);  unsqueeze_566 = None
    unsqueeze_568: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 3);  unsqueeze_567 = None
    mul_589: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_22, primals_23);  primals_23 = None
    unsqueeze_569: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_589, 0);  mul_589 = None
    unsqueeze_570: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_590: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_173, unsqueeze_568);  sub_173 = unsqueeze_568 = None
    sub_175: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_33, mul_590);  where_33 = mul_590 = None
    sub_176: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_175, unsqueeze_565);  sub_175 = unsqueeze_565 = None
    mul_591: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_176, unsqueeze_571);  sub_176 = unsqueeze_571 = None
    mul_592: "f32[64]" = torch.ops.aten.mul.Tensor(sum_69, squeeze_22);  sum_69 = squeeze_22 = None
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_591, relu_6, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_591 = primals_22 = None
    getitem_181: "f32[8, 64, 56, 56]" = convolution_backward_33[0]
    getitem_182: "f32[64, 64, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    add_217: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(slice_18, getitem_181);  slice_18 = getitem_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:104, code: out = self.conv6(torch.cat([d1, d2, d3], 1))
    le_34: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_34: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_34, full_default, add_217);  le_34 = add_217 = None
    sum_70: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_34, [0, 2, 3])
    sub_177: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_574);  convolution_6 = unsqueeze_574 = None
    mul_593: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_34, sub_177)
    sum_71: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_593, [0, 2, 3]);  mul_593 = None
    mul_594: "f32[64]" = torch.ops.aten.mul.Tensor(sum_70, 3.985969387755102e-05)
    unsqueeze_575: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_594, 0);  mul_594 = None
    unsqueeze_576: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_575, 2);  unsqueeze_575 = None
    unsqueeze_577: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 3);  unsqueeze_576 = None
    mul_595: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, 3.985969387755102e-05)
    mul_596: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, squeeze_19)
    mul_597: "f32[64]" = torch.ops.aten.mul.Tensor(mul_595, mul_596);  mul_595 = mul_596 = None
    unsqueeze_578: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_597, 0);  mul_597 = None
    unsqueeze_579: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 2);  unsqueeze_578 = None
    unsqueeze_580: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_579, 3);  unsqueeze_579 = None
    mul_598: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_19, primals_20);  primals_20 = None
    unsqueeze_581: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_598, 0);  mul_598 = None
    unsqueeze_582: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_599: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_177, unsqueeze_580);  sub_177 = unsqueeze_580 = None
    sub_179: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_34, mul_599);  where_34 = mul_599 = None
    sub_180: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_179, unsqueeze_577);  sub_179 = unsqueeze_577 = None
    mul_600: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_180, unsqueeze_583);  sub_180 = unsqueeze_583 = None
    mul_601: "f32[64]" = torch.ops.aten.mul.Tensor(sum_71, squeeze_19);  sum_71 = squeeze_19 = None
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_600, cat, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_600 = cat = primals_19 = None
    getitem_184: "f32[8, 128, 56, 56]" = convolution_backward_34[0]
    getitem_185: "f32[64, 128, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    slice_19: "f32[8, 64, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_184, 1, 0, 64)
    slice_20: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_184, 1, 64, 96)
    slice_21: "f32[8, 32, 56, 56]" = torch.ops.aten.slice.Tensor(getitem_184, 1, 96, 128);  getitem_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    where_35: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_35, full_default, slice_21);  le_35 = slice_21 = None
    sum_72: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_181: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_586);  convolution_5 = unsqueeze_586 = None
    mul_602: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_35, sub_181)
    sum_73: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_602, [0, 2, 3]);  mul_602 = None
    mul_603: "f32[32]" = torch.ops.aten.mul.Tensor(sum_72, 3.985969387755102e-05)
    unsqueeze_587: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_603, 0);  mul_603 = None
    unsqueeze_588: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_587, 2);  unsqueeze_587 = None
    unsqueeze_589: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 3);  unsqueeze_588 = None
    mul_604: "f32[32]" = torch.ops.aten.mul.Tensor(sum_73, 3.985969387755102e-05)
    mul_605: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_16, squeeze_16)
    mul_606: "f32[32]" = torch.ops.aten.mul.Tensor(mul_604, mul_605);  mul_604 = mul_605 = None
    unsqueeze_590: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_606, 0);  mul_606 = None
    unsqueeze_591: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, 2);  unsqueeze_590 = None
    unsqueeze_592: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_591, 3);  unsqueeze_591 = None
    mul_607: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_16, primals_17);  primals_17 = None
    unsqueeze_593: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_607, 0);  mul_607 = None
    unsqueeze_594: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_608: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_181, unsqueeze_592);  sub_181 = unsqueeze_592 = None
    sub_183: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_35, mul_608);  where_35 = mul_608 = None
    sub_184: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_183, unsqueeze_589);  sub_183 = unsqueeze_589 = None
    mul_609: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_184, unsqueeze_595);  sub_184 = unsqueeze_595 = None
    mul_610: "f32[32]" = torch.ops.aten.mul.Tensor(sum_73, squeeze_16);  sum_73 = squeeze_16 = None
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_609, relu_4, primals_16, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_609 = primals_16 = None
    getitem_187: "f32[8, 64, 56, 56]" = convolution_backward_35[0]
    getitem_188: "f32[32, 64, 3, 3]" = convolution_backward_35[1];  convolution_backward_35 = None
    le_36: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_36: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_36, full_default, getitem_187);  le_36 = getitem_187 = None
    sum_74: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_36, [0, 2, 3])
    sub_185: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_598);  convolution_4 = unsqueeze_598 = None
    mul_611: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_36, sub_185)
    sum_75: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_611, [0, 2, 3]);  mul_611 = None
    mul_612: "f32[64]" = torch.ops.aten.mul.Tensor(sum_74, 3.985969387755102e-05)
    unsqueeze_599: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_612, 0);  mul_612 = None
    unsqueeze_600: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_599, 2);  unsqueeze_599 = None
    unsqueeze_601: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, 3);  unsqueeze_600 = None
    mul_613: "f32[64]" = torch.ops.aten.mul.Tensor(sum_75, 3.985969387755102e-05)
    mul_614: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, squeeze_13)
    mul_615: "f32[64]" = torch.ops.aten.mul.Tensor(mul_613, mul_614);  mul_613 = mul_614 = None
    unsqueeze_602: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_615, 0);  mul_615 = None
    unsqueeze_603: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, 2);  unsqueeze_602 = None
    unsqueeze_604: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_603, 3);  unsqueeze_603 = None
    mul_616: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_13, primals_14);  primals_14 = None
    unsqueeze_605: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_616, 0);  mul_616 = None
    unsqueeze_606: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_617: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_185, unsqueeze_604);  sub_185 = unsqueeze_604 = None
    sub_187: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_36, mul_617);  where_36 = mul_617 = None
    sub_188: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_187, unsqueeze_601);  sub_187 = unsqueeze_601 = None
    mul_618: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_188, unsqueeze_607);  sub_188 = unsqueeze_607 = None
    mul_619: "f32[64]" = torch.ops.aten.mul.Tensor(sum_75, squeeze_13);  sum_75 = squeeze_13 = None
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_618, relu_3, primals_13, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_618 = primals_13 = None
    getitem_190: "f32[8, 32, 56, 56]" = convolution_backward_36[0]
    getitem_191: "f32[64, 32, 1, 1]" = convolution_backward_36[1];  convolution_backward_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:102, code: d3 = self.conv5(self.conv4(d2))
    add_218: "f32[8, 32, 56, 56]" = torch.ops.aten.add.Tensor(slice_20, getitem_190);  slice_20 = getitem_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    le_37: "b8[8, 32, 56, 56]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_37: "f32[8, 32, 56, 56]" = torch.ops.aten.where.self(le_37, full_default, add_218);  le_37 = add_218 = None
    sum_76: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_189: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_610);  convolution_3 = unsqueeze_610 = None
    mul_620: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(where_37, sub_189)
    sum_77: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_620, [0, 2, 3]);  mul_620 = None
    mul_621: "f32[32]" = torch.ops.aten.mul.Tensor(sum_76, 3.985969387755102e-05)
    unsqueeze_611: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_621, 0);  mul_621 = None
    unsqueeze_612: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_611, 2);  unsqueeze_611 = None
    unsqueeze_613: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, 3);  unsqueeze_612 = None
    mul_622: "f32[32]" = torch.ops.aten.mul.Tensor(sum_77, 3.985969387755102e-05)
    mul_623: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, squeeze_10)
    mul_624: "f32[32]" = torch.ops.aten.mul.Tensor(mul_622, mul_623);  mul_622 = mul_623 = None
    unsqueeze_614: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_624, 0);  mul_624 = None
    unsqueeze_615: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, 2);  unsqueeze_614 = None
    unsqueeze_616: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_615, 3);  unsqueeze_615 = None
    mul_625: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_10, primals_11);  primals_11 = None
    unsqueeze_617: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_625, 0);  mul_625 = None
    unsqueeze_618: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_626: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_189, unsqueeze_616);  sub_189 = unsqueeze_616 = None
    sub_191: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(where_37, mul_626);  where_37 = mul_626 = None
    sub_192: "f32[8, 32, 56, 56]" = torch.ops.aten.sub.Tensor(sub_191, unsqueeze_613);  sub_191 = unsqueeze_613 = None
    mul_627: "f32[8, 32, 56, 56]" = torch.ops.aten.mul.Tensor(sub_192, unsqueeze_619);  sub_192 = unsqueeze_619 = None
    mul_628: "f32[32]" = torch.ops.aten.mul.Tensor(sum_77, squeeze_10);  sum_77 = squeeze_10 = None
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_627, relu_2, primals_10, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_627 = primals_10 = None
    getitem_193: "f32[8, 64, 56, 56]" = convolution_backward_37[0]
    getitem_194: "f32[32, 64, 3, 3]" = convolution_backward_37[1];  convolution_backward_37 = None
    le_38: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_38: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_38, full_default, getitem_193);  le_38 = getitem_193 = None
    sum_78: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_38, [0, 2, 3])
    sub_193: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_622);  convolution_2 = unsqueeze_622 = None
    mul_629: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_38, sub_193)
    sum_79: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_629, [0, 2, 3]);  mul_629 = None
    mul_630: "f32[64]" = torch.ops.aten.mul.Tensor(sum_78, 3.985969387755102e-05)
    unsqueeze_623: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_630, 0);  mul_630 = None
    unsqueeze_624: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_623, 2);  unsqueeze_623 = None
    unsqueeze_625: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, 3);  unsqueeze_624 = None
    mul_631: "f32[64]" = torch.ops.aten.mul.Tensor(sum_79, 3.985969387755102e-05)
    mul_632: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, squeeze_7)
    mul_633: "f32[64]" = torch.ops.aten.mul.Tensor(mul_631, mul_632);  mul_631 = mul_632 = None
    unsqueeze_626: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_633, 0);  mul_633 = None
    unsqueeze_627: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, 2);  unsqueeze_626 = None
    unsqueeze_628: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_627, 3);  unsqueeze_627 = None
    mul_634: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_7, primals_8);  primals_8 = None
    unsqueeze_629: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_634, 0);  mul_634 = None
    unsqueeze_630: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_635: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_193, unsqueeze_628);  sub_193 = unsqueeze_628 = None
    sub_195: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_38, mul_635);  where_38 = mul_635 = None
    sub_196: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_195, unsqueeze_625);  sub_195 = unsqueeze_625 = None
    mul_636: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_196, unsqueeze_631);  sub_196 = unsqueeze_631 = None
    mul_637: "f32[64]" = torch.ops.aten.mul.Tensor(sum_79, squeeze_7);  sum_79 = squeeze_7 = None
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_636, relu_1, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_636 = primals_7 = None
    getitem_196: "f32[8, 64, 56, 56]" = convolution_backward_38[0]
    getitem_197: "f32[64, 64, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:101, code: d2 = self.conv3(self.conv2(d1))
    add_219: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(slice_19, getitem_196);  slice_19 = getitem_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:100, code: d1 = self.conv1(x[0])
    le_39: "b8[8, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_39: "f32[8, 64, 56, 56]" = torch.ops.aten.where.self(le_39, full_default, add_219);  le_39 = add_219 = None
    sum_80: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_197: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_634);  convolution_1 = unsqueeze_634 = None
    mul_638: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_39, sub_197)
    sum_81: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_638, [0, 2, 3]);  mul_638 = None
    mul_639: "f32[64]" = torch.ops.aten.mul.Tensor(sum_80, 3.985969387755102e-05)
    unsqueeze_635: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_639, 0);  mul_639 = None
    unsqueeze_636: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_635, 2);  unsqueeze_635 = None
    unsqueeze_637: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, 3);  unsqueeze_636 = None
    mul_640: "f32[64]" = torch.ops.aten.mul.Tensor(sum_81, 3.985969387755102e-05)
    mul_641: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, squeeze_4)
    mul_642: "f32[64]" = torch.ops.aten.mul.Tensor(mul_640, mul_641);  mul_640 = mul_641 = None
    unsqueeze_638: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_642, 0);  mul_642 = None
    unsqueeze_639: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, 2);  unsqueeze_638 = None
    unsqueeze_640: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_639, 3);  unsqueeze_639 = None
    mul_643: "f32[64]" = torch.ops.aten.mul.Tensor(squeeze_4, primals_5);  primals_5 = None
    unsqueeze_641: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_643, 0);  mul_643 = None
    unsqueeze_642: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_644: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_197, unsqueeze_640);  sub_197 = unsqueeze_640 = None
    sub_199: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(where_39, mul_644);  where_39 = mul_644 = None
    sub_200: "f32[8, 64, 56, 56]" = torch.ops.aten.sub.Tensor(sub_199, unsqueeze_637);  sub_199 = unsqueeze_637 = None
    mul_645: "f32[8, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_200, unsqueeze_643);  sub_200 = unsqueeze_643 = None
    mul_646: "f32[64]" = torch.ops.aten.mul.Tensor(sum_81, squeeze_4);  sum_81 = squeeze_4 = None
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(mul_645, relu, primals_4, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_645 = primals_4 = None
    getitem_199: "f32[8, 32, 112, 112]" = convolution_backward_39[0]
    getitem_200: "f32[64, 32, 3, 3]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/selecsls.py:169, code: x = self.stem(x)
    le_40: "b8[8, 32, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_40: "f32[8, 32, 112, 112]" = torch.ops.aten.where.self(le_40, full_default, getitem_199);  le_40 = full_default = getitem_199 = None
    sum_82: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_40, [0, 2, 3])
    sub_201: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_646);  convolution = unsqueeze_646 = None
    mul_647: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(where_40, sub_201)
    sum_83: "f32[32]" = torch.ops.aten.sum.dim_IntList(mul_647, [0, 2, 3]);  mul_647 = None
    mul_648: "f32[32]" = torch.ops.aten.mul.Tensor(sum_82, 9.964923469387754e-06)
    unsqueeze_647: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_648, 0);  mul_648 = None
    unsqueeze_648: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_647, 2);  unsqueeze_647 = None
    unsqueeze_649: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, 3);  unsqueeze_648 = None
    mul_649: "f32[32]" = torch.ops.aten.mul.Tensor(sum_83, 9.964923469387754e-06)
    mul_650: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, squeeze_1)
    mul_651: "f32[32]" = torch.ops.aten.mul.Tensor(mul_649, mul_650);  mul_649 = mul_650 = None
    unsqueeze_650: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_651, 0);  mul_651 = None
    unsqueeze_651: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, 2);  unsqueeze_650 = None
    unsqueeze_652: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_651, 3);  unsqueeze_651 = None
    mul_652: "f32[32]" = torch.ops.aten.mul.Tensor(squeeze_1, primals_2);  primals_2 = None
    unsqueeze_653: "f32[1, 32]" = torch.ops.aten.unsqueeze.default(mul_652, 0);  mul_652 = None
    unsqueeze_654: "f32[1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 32, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_653: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_201, unsqueeze_652);  sub_201 = unsqueeze_652 = None
    sub_203: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(where_40, mul_653);  where_40 = mul_653 = None
    sub_204: "f32[8, 32, 112, 112]" = torch.ops.aten.sub.Tensor(sub_203, unsqueeze_649);  sub_203 = unsqueeze_649 = None
    mul_654: "f32[8, 32, 112, 112]" = torch.ops.aten.mul.Tensor(sub_204, unsqueeze_655);  sub_204 = unsqueeze_655 = None
    mul_655: "f32[32]" = torch.ops.aten.mul.Tensor(sum_83, squeeze_1);  sum_83 = squeeze_1 = None
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(mul_654, primals_249, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_654 = primals_249 = primals_1 = None
    getitem_203: "f32[32, 3, 3, 3]" = convolution_backward_40[1];  convolution_backward_40 = None
    return [getitem_203, mul_655, sum_82, getitem_200, mul_646, sum_80, getitem_197, mul_637, sum_78, getitem_194, mul_628, sum_76, getitem_191, mul_619, sum_74, getitem_188, mul_610, sum_72, getitem_185, mul_601, sum_70, getitem_182, mul_592, sum_68, getitem_179, mul_583, sum_66, getitem_176, mul_574, sum_64, getitem_173, mul_565, sum_62, getitem_170, mul_556, sum_60, getitem_167, mul_547, sum_58, getitem_164, mul_538, sum_56, getitem_161, mul_529, sum_54, getitem_158, mul_520, sum_52, getitem_155, mul_511, sum_50, getitem_152, mul_502, sum_48, getitem_149, mul_493, sum_46, getitem_146, mul_484, sum_44, getitem_143, mul_475, sum_42, getitem_140, mul_466, sum_40, getitem_137, mul_457, sum_38, getitem_134, mul_448, sum_36, getitem_131, mul_439, sum_34, getitem_128, mul_430, sum_32, getitem_125, mul_421, sum_30, getitem_122, mul_412, sum_28, getitem_119, mul_403, sum_26, getitem_116, mul_394, sum_24, getitem_113, mul_385, sum_22, getitem_110, mul_376, sum_20, getitem_107, mul_367, sum_18, getitem_104, mul_358, sum_16, getitem_101, mul_349, sum_14, getitem_98, mul_340, sum_12, getitem_95, mul_331, sum_10, getitem_92, mul_322, sum_8, getitem_89, mul_313, sum_6, getitem_86, mul_304, sum_4, getitem_83, mul_295, sum_2, permute_4, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    