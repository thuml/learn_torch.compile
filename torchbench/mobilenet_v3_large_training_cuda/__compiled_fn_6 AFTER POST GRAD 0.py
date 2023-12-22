from __future__ import annotations



def forward(self, primals_1: "f32[16, 3, 3, 3]", primals_2: "f32[16]", primals_4: "f32[16, 1, 3, 3]", primals_5: "f32[16]", primals_7: "f32[16, 16, 1, 1]", primals_8: "f32[16]", primals_10: "f32[64, 16, 1, 1]", primals_11: "f32[64]", primals_13: "f32[64, 1, 3, 3]", primals_14: "f32[64]", primals_16: "f32[24, 64, 1, 1]", primals_17: "f32[24]", primals_19: "f32[72, 24, 1, 1]", primals_20: "f32[72]", primals_22: "f32[72, 1, 3, 3]", primals_23: "f32[72]", primals_25: "f32[24, 72, 1, 1]", primals_26: "f32[24]", primals_28: "f32[72, 24, 1, 1]", primals_29: "f32[72]", primals_31: "f32[72, 1, 5, 5]", primals_32: "f32[72]", primals_34: "f32[24, 72, 1, 1]", primals_36: "f32[72, 24, 1, 1]", primals_38: "f32[40, 72, 1, 1]", primals_39: "f32[40]", primals_41: "f32[120, 40, 1, 1]", primals_42: "f32[120]", primals_44: "f32[120, 1, 5, 5]", primals_45: "f32[120]", primals_47: "f32[32, 120, 1, 1]", primals_49: "f32[120, 32, 1, 1]", primals_51: "f32[40, 120, 1, 1]", primals_52: "f32[40]", primals_54: "f32[120, 40, 1, 1]", primals_55: "f32[120]", primals_57: "f32[120, 1, 5, 5]", primals_58: "f32[120]", primals_60: "f32[32, 120, 1, 1]", primals_62: "f32[120, 32, 1, 1]", primals_64: "f32[40, 120, 1, 1]", primals_65: "f32[40]", primals_67: "f32[240, 40, 1, 1]", primals_68: "f32[240]", primals_70: "f32[240, 1, 3, 3]", primals_71: "f32[240]", primals_73: "f32[80, 240, 1, 1]", primals_74: "f32[80]", primals_76: "f32[200, 80, 1, 1]", primals_77: "f32[200]", primals_79: "f32[200, 1, 3, 3]", primals_80: "f32[200]", primals_82: "f32[80, 200, 1, 1]", primals_83: "f32[80]", primals_85: "f32[184, 80, 1, 1]", primals_86: "f32[184]", primals_88: "f32[184, 1, 3, 3]", primals_89: "f32[184]", primals_91: "f32[80, 184, 1, 1]", primals_92: "f32[80]", primals_94: "f32[184, 80, 1, 1]", primals_95: "f32[184]", primals_97: "f32[184, 1, 3, 3]", primals_98: "f32[184]", primals_100: "f32[80, 184, 1, 1]", primals_101: "f32[80]", primals_103: "f32[480, 80, 1, 1]", primals_104: "f32[480]", primals_106: "f32[480, 1, 3, 3]", primals_107: "f32[480]", primals_109: "f32[120, 480, 1, 1]", primals_111: "f32[480, 120, 1, 1]", primals_113: "f32[112, 480, 1, 1]", primals_114: "f32[112]", primals_116: "f32[672, 112, 1, 1]", primals_117: "f32[672]", primals_119: "f32[672, 1, 3, 3]", primals_120: "f32[672]", primals_122: "f32[168, 672, 1, 1]", primals_124: "f32[672, 168, 1, 1]", primals_126: "f32[112, 672, 1, 1]", primals_127: "f32[112]", primals_129: "f32[672, 112, 1, 1]", primals_130: "f32[672]", primals_132: "f32[672, 1, 5, 5]", primals_133: "f32[672]", primals_135: "f32[168, 672, 1, 1]", primals_137: "f32[672, 168, 1, 1]", primals_139: "f32[160, 672, 1, 1]", primals_140: "f32[160]", primals_142: "f32[960, 160, 1, 1]", primals_143: "f32[960]", primals_145: "f32[960, 1, 5, 5]", primals_146: "f32[960]", primals_148: "f32[240, 960, 1, 1]", primals_150: "f32[960, 240, 1, 1]", primals_152: "f32[160, 960, 1, 1]", primals_153: "f32[160]", primals_155: "f32[960, 160, 1, 1]", primals_156: "f32[960]", primals_158: "f32[960, 1, 5, 5]", primals_159: "f32[960]", primals_161: "f32[240, 960, 1, 1]", primals_163: "f32[960, 240, 1, 1]", primals_165: "f32[160, 960, 1, 1]", primals_166: "f32[160]", primals_168: "f32[960, 160, 1, 1]", primals_169: "f32[960]", primals_175: "f32[16]", primals_176: "f32[16]", primals_178: "f32[16]", primals_179: "f32[16]", primals_181: "f32[16]", primals_182: "f32[16]", primals_184: "f32[64]", primals_185: "f32[64]", primals_187: "f32[64]", primals_188: "f32[64]", primals_190: "f32[24]", primals_191: "f32[24]", primals_193: "f32[72]", primals_194: "f32[72]", primals_196: "f32[72]", primals_197: "f32[72]", primals_199: "f32[24]", primals_200: "f32[24]", primals_202: "f32[72]", primals_203: "f32[72]", primals_205: "f32[72]", primals_206: "f32[72]", primals_208: "f32[40]", primals_209: "f32[40]", primals_211: "f32[120]", primals_212: "f32[120]", primals_214: "f32[120]", primals_215: "f32[120]", primals_217: "f32[40]", primals_218: "f32[40]", primals_220: "f32[120]", primals_221: "f32[120]", primals_223: "f32[120]", primals_224: "f32[120]", primals_226: "f32[40]", primals_227: "f32[40]", primals_229: "f32[240]", primals_230: "f32[240]", primals_232: "f32[240]", primals_233: "f32[240]", primals_235: "f32[80]", primals_236: "f32[80]", primals_238: "f32[200]", primals_239: "f32[200]", primals_241: "f32[200]", primals_242: "f32[200]", primals_244: "f32[80]", primals_245: "f32[80]", primals_247: "f32[184]", primals_248: "f32[184]", primals_250: "f32[184]", primals_251: "f32[184]", primals_253: "f32[80]", primals_254: "f32[80]", primals_256: "f32[184]", primals_257: "f32[184]", primals_259: "f32[184]", primals_260: "f32[184]", primals_262: "f32[80]", primals_263: "f32[80]", primals_265: "f32[480]", primals_266: "f32[480]", primals_268: "f32[480]", primals_269: "f32[480]", primals_271: "f32[112]", primals_272: "f32[112]", primals_274: "f32[672]", primals_275: "f32[672]", primals_277: "f32[672]", primals_278: "f32[672]", primals_280: "f32[112]", primals_281: "f32[112]", primals_283: "f32[672]", primals_284: "f32[672]", primals_286: "f32[672]", primals_287: "f32[672]", primals_289: "f32[160]", primals_290: "f32[160]", primals_292: "f32[960]", primals_293: "f32[960]", primals_295: "f32[960]", primals_296: "f32[960]", primals_298: "f32[160]", primals_299: "f32[160]", primals_301: "f32[960]", primals_302: "f32[960]", primals_304: "f32[960]", primals_305: "f32[960]", primals_307: "f32[160]", primals_308: "f32[160]", primals_310: "f32[960]", primals_311: "f32[960]", primals_313: "f32[4, 3, 224, 224]", convolution: "f32[4, 16, 112, 112]", clone: "f32[4, 16, 112, 112]", div: "f32[4, 16, 112, 112]", convolution_1: "f32[4, 16, 112, 112]", relu: "f32[4, 16, 112, 112]", convolution_2: "f32[4, 16, 112, 112]", add_7: "f32[4, 16, 112, 112]", convolution_3: "f32[4, 64, 112, 112]", relu_1: "f32[4, 64, 112, 112]", convolution_4: "f32[4, 64, 56, 56]", relu_2: "f32[4, 64, 56, 56]", convolution_5: "f32[4, 24, 56, 56]", add_13: "f32[4, 24, 56, 56]", convolution_6: "f32[4, 72, 56, 56]", relu_3: "f32[4, 72, 56, 56]", convolution_7: "f32[4, 72, 56, 56]", relu_4: "f32[4, 72, 56, 56]", convolution_8: "f32[4, 24, 56, 56]", add_20: "f32[4, 24, 56, 56]", convolution_9: "f32[4, 72, 56, 56]", relu_5: "f32[4, 72, 56, 56]", convolution_10: "f32[4, 72, 28, 28]", relu_6: "f32[4, 72, 28, 28]", mean: "f32[4, 72, 1, 1]", relu_7: "f32[4, 24, 1, 1]", div_1: "f32[4, 72, 1, 1]", mul_34: "f32[4, 72, 28, 28]", convolution_13: "f32[4, 40, 28, 28]", add_27: "f32[4, 40, 28, 28]", convolution_14: "f32[4, 120, 28, 28]", relu_8: "f32[4, 120, 28, 28]", convolution_15: "f32[4, 120, 28, 28]", relu_9: "f32[4, 120, 28, 28]", mean_1: "f32[4, 120, 1, 1]", relu_10: "f32[4, 32, 1, 1]", div_2: "f32[4, 120, 1, 1]", mul_44: "f32[4, 120, 28, 28]", convolution_18: "f32[4, 40, 28, 28]", add_35: "f32[4, 40, 28, 28]", convolution_19: "f32[4, 120, 28, 28]", relu_11: "f32[4, 120, 28, 28]", convolution_20: "f32[4, 120, 28, 28]", relu_12: "f32[4, 120, 28, 28]", mean_2: "f32[4, 120, 1, 1]", relu_13: "f32[4, 32, 1, 1]", div_3: "f32[4, 120, 1, 1]", mul_54: "f32[4, 120, 28, 28]", convolution_23: "f32[4, 40, 28, 28]", add_43: "f32[4, 40, 28, 28]", convolution_24: "f32[4, 240, 28, 28]", clone_1: "f32[4, 240, 28, 28]", div_4: "f32[4, 240, 28, 28]", convolution_25: "f32[4, 240, 14, 14]", clone_2: "f32[4, 240, 14, 14]", div_5: "f32[4, 240, 14, 14]", convolution_26: "f32[4, 80, 14, 14]", add_51: "f32[4, 80, 14, 14]", convolution_27: "f32[4, 200, 14, 14]", clone_3: "f32[4, 200, 14, 14]", div_6: "f32[4, 200, 14, 14]", convolution_28: "f32[4, 200, 14, 14]", clone_4: "f32[4, 200, 14, 14]", div_7: "f32[4, 200, 14, 14]", convolution_29: "f32[4, 80, 14, 14]", add_60: "f32[4, 80, 14, 14]", convolution_30: "f32[4, 184, 14, 14]", clone_5: "f32[4, 184, 14, 14]", div_8: "f32[4, 184, 14, 14]", convolution_31: "f32[4, 184, 14, 14]", clone_6: "f32[4, 184, 14, 14]", div_9: "f32[4, 184, 14, 14]", convolution_32: "f32[4, 80, 14, 14]", add_69: "f32[4, 80, 14, 14]", convolution_33: "f32[4, 184, 14, 14]", clone_7: "f32[4, 184, 14, 14]", div_10: "f32[4, 184, 14, 14]", convolution_34: "f32[4, 184, 14, 14]", clone_8: "f32[4, 184, 14, 14]", div_11: "f32[4, 184, 14, 14]", convolution_35: "f32[4, 80, 14, 14]", add_78: "f32[4, 80, 14, 14]", convolution_36: "f32[4, 480, 14, 14]", clone_9: "f32[4, 480, 14, 14]", div_12: "f32[4, 480, 14, 14]", convolution_37: "f32[4, 480, 14, 14]", clone_10: "f32[4, 480, 14, 14]", div_13: "f32[4, 480, 14, 14]", mean_3: "f32[4, 480, 1, 1]", relu_14: "f32[4, 120, 1, 1]", div_14: "f32[4, 480, 1, 1]", mul_110: "f32[4, 480, 14, 14]", convolution_40: "f32[4, 112, 14, 14]", add_87: "f32[4, 112, 14, 14]", convolution_41: "f32[4, 672, 14, 14]", clone_11: "f32[4, 672, 14, 14]", div_15: "f32[4, 672, 14, 14]", convolution_42: "f32[4, 672, 14, 14]", clone_12: "f32[4, 672, 14, 14]", div_16: "f32[4, 672, 14, 14]", mean_4: "f32[4, 672, 1, 1]", relu_15: "f32[4, 168, 1, 1]", div_17: "f32[4, 672, 1, 1]", mul_122: "f32[4, 672, 14, 14]", convolution_45: "f32[4, 112, 14, 14]", add_97: "f32[4, 112, 14, 14]", convolution_46: "f32[4, 672, 14, 14]", clone_13: "f32[4, 672, 14, 14]", div_18: "f32[4, 672, 14, 14]", convolution_47: "f32[4, 672, 7, 7]", clone_14: "f32[4, 672, 7, 7]", div_19: "f32[4, 672, 7, 7]", mean_5: "f32[4, 672, 1, 1]", relu_16: "f32[4, 168, 1, 1]", div_20: "f32[4, 672, 1, 1]", mul_134: "f32[4, 672, 7, 7]", convolution_50: "f32[4, 160, 7, 7]", add_106: "f32[4, 160, 7, 7]", convolution_51: "f32[4, 960, 7, 7]", clone_15: "f32[4, 960, 7, 7]", div_21: "f32[4, 960, 7, 7]", convolution_52: "f32[4, 960, 7, 7]", clone_16: "f32[4, 960, 7, 7]", div_22: "f32[4, 960, 7, 7]", mean_6: "f32[4, 960, 1, 1]", relu_17: "f32[4, 240, 1, 1]", div_23: "f32[4, 960, 1, 1]", mul_146: "f32[4, 960, 7, 7]", convolution_55: "f32[4, 160, 7, 7]", add_116: "f32[4, 160, 7, 7]", convolution_56: "f32[4, 960, 7, 7]", clone_17: "f32[4, 960, 7, 7]", div_24: "f32[4, 960, 7, 7]", convolution_57: "f32[4, 960, 7, 7]", clone_18: "f32[4, 960, 7, 7]", div_25: "f32[4, 960, 7, 7]", mean_7: "f32[4, 960, 1, 1]", relu_18: "f32[4, 240, 1, 1]", div_26: "f32[4, 960, 1, 1]", mul_158: "f32[4, 960, 7, 7]", convolution_60: "f32[4, 160, 7, 7]", add_126: "f32[4, 160, 7, 7]", convolution_61: "f32[4, 960, 7, 7]", clone_19: "f32[4, 960, 7, 7]", view: "f32[4, 960]", addmm: "f32[4, 1280]", div_28: "f32[4, 1280]", permute_2: "f32[1000, 1280]", permute_6: "f32[1280, 960]", bitwise_and: "b8[4, 960, 1, 1]", bitwise_and_1: "b8[4, 960, 1, 1]", bitwise_and_2: "b8[4, 672, 1, 1]", bitwise_and_3: "b8[4, 672, 1, 1]", bitwise_and_4: "b8[4, 480, 1, 1]", bitwise_and_5: "b8[4, 120, 1, 1]", bitwise_and_6: "b8[4, 120, 1, 1]", bitwise_and_7: "b8[4, 72, 1, 1]", tangents_1: "f32[4, 1000]"):
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:215, code: x = self.classifier(x)
    mm: "f32[4, 1280]" = torch.ops.aten.mm.default(tangents_1, permute_2);  permute_2 = None
    permute_3: "f32[1000, 4]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 1280]" = torch.ops.aten.mm.default(permute_3, div_28);  permute_3 = div_28 = None
    permute_4: "f32[1280, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_1: "f32[1000]" = torch.ops.aten.reshape.default(sum_1, [1000]);  sum_1 = None
    permute_5: "f32[1000, 1280]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    lt: "b8[4, 1280]" = torch.ops.aten.lt.Scalar(addmm, -3)
    le: "b8[4, 1280]" = torch.ops.aten.le.Scalar(addmm, 3)
    div_29: "f32[4, 1280]" = torch.ops.aten.div.Tensor(addmm, 3);  addmm = None
    add_131: "f32[4, 1280]" = torch.ops.aten.add.Tensor(div_29, 0.5);  div_29 = None
    mul_167: "f32[4, 1280]" = torch.ops.aten.mul.Tensor(mm, add_131);  add_131 = None
    where: "f32[4, 1280]" = torch.ops.aten.where.self(le, mul_167, mm);  le = mul_167 = mm = None
    full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    where_1: "f32[4, 1280]" = torch.ops.aten.where.self(lt, full_default, where);  lt = where = None
    mm_2: "f32[4, 960]" = torch.ops.aten.mm.default(where_1, permute_6);  permute_6 = None
    permute_7: "f32[1280, 4]" = torch.ops.aten.permute.default(where_1, [1, 0])
    mm_3: "f32[1280, 960]" = torch.ops.aten.mm.default(permute_7, view);  permute_7 = view = None
    permute_8: "f32[960, 1280]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_2: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(where_1, [0], True);  where_1 = None
    view_2: "f32[1280]" = torch.ops.aten.reshape.default(sum_2, [1280]);  sum_2 = None
    permute_9: "f32[1280, 960]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:213, code: x = torch.flatten(x, 1)
    view_3: "f32[4, 960, 1, 1]" = torch.ops.aten.reshape.default(mm_2, [4, 960, 1, 1]);  mm_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:212, code: x = self.avgpool(x)
    expand: "f32[4, 960, 7, 7]" = torch.ops.aten.expand.default(view_3, [4, 960, 7, 7]);  view_3 = None
    div_30: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand, 49);  expand = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
    lt_1: "b8[4, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_19, -3)
    le_1: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_19, 3)
    div_31: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_19, 3);  clone_19 = None
    add_132: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_31, 0.5);  div_31 = None
    mul_168: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(div_30, add_132);  add_132 = None
    where_2: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(le_1, mul_168, div_30);  le_1 = mul_168 = div_30 = None
    where_3: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(lt_1, full_default, where_2);  lt_1 = where_2 = None
    add_133: "f32[960]" = torch.ops.aten.add.Tensor(primals_311, 0.001);  primals_311 = None
    rsqrt: "f32[960]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
    unsqueeze_368: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_310, 0);  primals_310 = None
    unsqueeze_369: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, 2);  unsqueeze_368 = None
    unsqueeze_370: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_369, 3);  unsqueeze_369 = None
    sum_3: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_3, [0, 2, 3])
    sub_46: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_370);  convolution_61 = unsqueeze_370 = None
    mul_169: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, sub_46);  sub_46 = None
    sum_4: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 2, 3]);  mul_169 = None
    mul_174: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt, primals_169);  primals_169 = None
    unsqueeze_377: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_174, 0);  mul_174 = None
    unsqueeze_378: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_377, 2);  unsqueeze_377 = None
    unsqueeze_379: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 3);  unsqueeze_378 = None
    mul_175: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_3, unsqueeze_379);  where_3 = unsqueeze_379 = None
    mul_176: "f32[960]" = torch.ops.aten.mul.Tensor(sum_4, rsqrt);  sum_4 = rsqrt = None
    convolution_backward = torch.ops.aten.convolution_backward.default(mul_175, add_126, primals_168, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_175 = add_126 = primals_168 = None
    getitem: "f32[4, 160, 7, 7]" = convolution_backward[0]
    getitem_1: "f32[960, 160, 1, 1]" = convolution_backward[1];  convolution_backward = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_134: "f32[160]" = torch.ops.aten.add.Tensor(primals_308, 0.001);  primals_308 = None
    rsqrt_1: "f32[160]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    unsqueeze_380: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_307, 0);  primals_307 = None
    unsqueeze_381: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 2);  unsqueeze_380 = None
    unsqueeze_382: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_381, 3);  unsqueeze_381 = None
    sum_5: "f32[160]" = torch.ops.aten.sum.dim_IntList(getitem, [0, 2, 3])
    sub_47: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_382);  convolution_60 = unsqueeze_382 = None
    mul_177: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, sub_47);  sub_47 = None
    sum_6: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_177, [0, 2, 3]);  mul_177 = None
    mul_182: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_1, primals_166);  primals_166 = None
    unsqueeze_389: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_182, 0);  mul_182 = None
    unsqueeze_390: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_389, 2);  unsqueeze_389 = None
    unsqueeze_391: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, 3);  unsqueeze_390 = None
    mul_183: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(getitem, unsqueeze_391);  unsqueeze_391 = None
    mul_184: "f32[160]" = torch.ops.aten.mul.Tensor(sum_6, rsqrt_1);  sum_6 = rsqrt_1 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(mul_183, mul_158, primals_165, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_183 = mul_158 = primals_165 = None
    getitem_3: "f32[4, 960, 7, 7]" = convolution_backward_1[0]
    getitem_4: "f32[160, 960, 1, 1]" = convolution_backward_1[1];  convolution_backward_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_185: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_3, div_26);  div_26 = None
    mul_186: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_3, div_25);  getitem_3 = div_25 = None
    sum_7: "f32[4, 960, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2, 3], True);  mul_186 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    mul_187: "f32[4, 960, 1, 1]" = torch.ops.aten.mul.Tensor(sum_7, 0.16666666666666666);  sum_7 = None
    where_4: "f32[4, 960, 1, 1]" = torch.ops.aten.where.self(bitwise_and, mul_187, full_default);  bitwise_and = mul_187 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_8: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_4, [0, 2, 3])
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(where_4, relu_18, primals_163, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_4 = primals_163 = None
    getitem_6: "f32[4, 240, 1, 1]" = convolution_backward_2[0]
    getitem_7: "f32[960, 240, 1, 1]" = convolution_backward_2[1];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    le_2: "b8[4, 240, 1, 1]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
    where_5: "f32[4, 240, 1, 1]" = torch.ops.aten.where.self(le_2, full_default, getitem_6);  le_2 = getitem_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_9: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_5, [0, 2, 3])
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(where_5, mean_7, primals_161, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_5 = mean_7 = primals_161 = None
    getitem_9: "f32[4, 960, 1, 1]" = convolution_backward_3[0]
    getitem_10: "f32[240, 960, 1, 1]" = convolution_backward_3[1];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_1: "f32[4, 960, 7, 7]" = torch.ops.aten.expand.default(getitem_9, [4, 960, 7, 7]);  getitem_9 = None
    div_32: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand_1, 49);  expand_1 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_135: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_185, div_32);  mul_185 = div_32 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    lt_3: "b8[4, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_18, -3)
    le_3: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_18, 3)
    div_33: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_18, 3);  clone_18 = None
    add_136: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_33, 0.5);  div_33 = None
    mul_188: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_135, add_136);  add_136 = None
    where_6: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(le_3, mul_188, add_135);  le_3 = mul_188 = add_135 = None
    where_7: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(lt_3, full_default, where_6);  lt_3 = where_6 = None
    add_137: "f32[960]" = torch.ops.aten.add.Tensor(primals_305, 0.001);  primals_305 = None
    rsqrt_2: "f32[960]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    unsqueeze_392: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_304, 0);  primals_304 = None
    unsqueeze_393: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 2);  unsqueeze_392 = None
    unsqueeze_394: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_393, 3);  unsqueeze_393 = None
    sum_10: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_7, [0, 2, 3])
    sub_48: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_394);  convolution_57 = unsqueeze_394 = None
    mul_189: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, sub_48);  sub_48 = None
    sum_11: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 2, 3]);  mul_189 = None
    mul_194: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_2, primals_159);  primals_159 = None
    unsqueeze_401: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_194, 0);  mul_194 = None
    unsqueeze_402: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 2);  unsqueeze_401 = None
    unsqueeze_403: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, 3);  unsqueeze_402 = None
    mul_195: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_7, unsqueeze_403);  where_7 = unsqueeze_403 = None
    mul_196: "f32[960]" = torch.ops.aten.mul.Tensor(sum_11, rsqrt_2);  sum_11 = rsqrt_2 = None
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(mul_195, div_24, primals_158, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False]);  mul_195 = div_24 = primals_158 = None
    getitem_12: "f32[4, 960, 7, 7]" = convolution_backward_4[0]
    getitem_13: "f32[960, 1, 5, 5]" = convolution_backward_4[1];  convolution_backward_4 = None
    lt_4: "b8[4, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_17, -3)
    le_4: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_17, 3)
    div_34: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_17, 3);  clone_17 = None
    add_138: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_34, 0.5);  div_34 = None
    mul_197: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_12, add_138);  add_138 = None
    where_8: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(le_4, mul_197, getitem_12);  le_4 = mul_197 = getitem_12 = None
    where_9: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(lt_4, full_default, where_8);  lt_4 = where_8 = None
    add_139: "f32[960]" = torch.ops.aten.add.Tensor(primals_302, 0.001);  primals_302 = None
    rsqrt_3: "f32[960]" = torch.ops.aten.rsqrt.default(add_139);  add_139 = None
    unsqueeze_404: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_301, 0);  primals_301 = None
    unsqueeze_405: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, 2);  unsqueeze_404 = None
    unsqueeze_406: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 3);  unsqueeze_405 = None
    sum_12: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_9, [0, 2, 3])
    sub_49: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_406);  convolution_56 = unsqueeze_406 = None
    mul_198: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, sub_49);  sub_49 = None
    sum_13: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 2, 3]);  mul_198 = None
    mul_203: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_3, primals_156);  primals_156 = None
    unsqueeze_413: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_203, 0);  mul_203 = None
    unsqueeze_414: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 2);  unsqueeze_413 = None
    unsqueeze_415: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, 3);  unsqueeze_414 = None
    mul_204: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_9, unsqueeze_415);  where_9 = unsqueeze_415 = None
    mul_205: "f32[960]" = torch.ops.aten.mul.Tensor(sum_13, rsqrt_3);  sum_13 = rsqrt_3 = None
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(mul_204, add_116, primals_155, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_204 = add_116 = primals_155 = None
    getitem_15: "f32[4, 160, 7, 7]" = convolution_backward_5[0]
    getitem_16: "f32[960, 160, 1, 1]" = convolution_backward_5[1];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_140: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(getitem, getitem_15);  getitem = getitem_15 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_141: "f32[160]" = torch.ops.aten.add.Tensor(primals_299, 0.001);  primals_299 = None
    rsqrt_4: "f32[160]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    unsqueeze_416: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_298, 0);  primals_298 = None
    unsqueeze_417: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, 2);  unsqueeze_416 = None
    unsqueeze_418: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 3);  unsqueeze_417 = None
    sum_14: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_140, [0, 2, 3])
    sub_50: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_418);  convolution_55 = unsqueeze_418 = None
    mul_206: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_140, sub_50);  sub_50 = None
    sum_15: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 2, 3]);  mul_206 = None
    mul_211: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_4, primals_153);  primals_153 = None
    unsqueeze_425: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_211, 0);  mul_211 = None
    unsqueeze_426: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_425, 2);  unsqueeze_425 = None
    unsqueeze_427: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 3);  unsqueeze_426 = None
    mul_212: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_140, unsqueeze_427);  unsqueeze_427 = None
    mul_213: "f32[160]" = torch.ops.aten.mul.Tensor(sum_15, rsqrt_4);  sum_15 = rsqrt_4 = None
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(mul_212, mul_146, primals_152, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_212 = mul_146 = primals_152 = None
    getitem_18: "f32[4, 960, 7, 7]" = convolution_backward_6[0]
    getitem_19: "f32[160, 960, 1, 1]" = convolution_backward_6[1];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_214: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_18, div_23);  div_23 = None
    mul_215: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_18, div_22);  getitem_18 = div_22 = None
    sum_16: "f32[4, 960, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2, 3], True);  mul_215 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    mul_216: "f32[4, 960, 1, 1]" = torch.ops.aten.mul.Tensor(sum_16, 0.16666666666666666);  sum_16 = None
    where_10: "f32[4, 960, 1, 1]" = torch.ops.aten.where.self(bitwise_and_1, mul_216, full_default);  bitwise_and_1 = mul_216 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_17: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_10, [0, 2, 3])
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(where_10, relu_17, primals_150, [960], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_10 = primals_150 = None
    getitem_21: "f32[4, 240, 1, 1]" = convolution_backward_7[0]
    getitem_22: "f32[960, 240, 1, 1]" = convolution_backward_7[1];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    le_5: "b8[4, 240, 1, 1]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
    where_11: "f32[4, 240, 1, 1]" = torch.ops.aten.where.self(le_5, full_default, getitem_21);  le_5 = getitem_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_18: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_11, [0, 2, 3])
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(where_11, mean_6, primals_148, [240], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_11 = mean_6 = primals_148 = None
    getitem_24: "f32[4, 960, 1, 1]" = convolution_backward_8[0]
    getitem_25: "f32[240, 960, 1, 1]" = convolution_backward_8[1];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_2: "f32[4, 960, 7, 7]" = torch.ops.aten.expand.default(getitem_24, [4, 960, 7, 7]);  getitem_24 = None
    div_35: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Scalar(expand_2, 49);  expand_2 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_142: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(mul_214, div_35);  mul_214 = div_35 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    lt_6: "b8[4, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_16, -3)
    le_6: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_16, 3)
    div_36: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_16, 3);  clone_16 = None
    add_143: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_36, 0.5);  div_36 = None
    mul_217: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(add_142, add_143);  add_143 = None
    where_12: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(le_6, mul_217, add_142);  le_6 = mul_217 = add_142 = None
    where_13: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(lt_6, full_default, where_12);  lt_6 = where_12 = None
    add_144: "f32[960]" = torch.ops.aten.add.Tensor(primals_296, 0.001);  primals_296 = None
    rsqrt_5: "f32[960]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    unsqueeze_428: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_295, 0);  primals_295 = None
    unsqueeze_429: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 2);  unsqueeze_428 = None
    unsqueeze_430: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_429, 3);  unsqueeze_429 = None
    sum_19: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_13, [0, 2, 3])
    sub_51: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_430);  convolution_52 = unsqueeze_430 = None
    mul_218: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, sub_51);  sub_51 = None
    sum_20: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 2, 3]);  mul_218 = None
    mul_223: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_5, primals_146);  primals_146 = None
    unsqueeze_437: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_223, 0);  mul_223 = None
    unsqueeze_438: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_437, 2);  unsqueeze_437 = None
    unsqueeze_439: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 3);  unsqueeze_438 = None
    mul_224: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_13, unsqueeze_439);  where_13 = unsqueeze_439 = None
    mul_225: "f32[960]" = torch.ops.aten.mul.Tensor(sum_20, rsqrt_5);  sum_20 = rsqrt_5 = None
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(mul_224, div_21, primals_145, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 960, [True, True, False]);  mul_224 = div_21 = primals_145 = None
    getitem_27: "f32[4, 960, 7, 7]" = convolution_backward_9[0]
    getitem_28: "f32[960, 1, 5, 5]" = convolution_backward_9[1];  convolution_backward_9 = None
    lt_7: "b8[4, 960, 7, 7]" = torch.ops.aten.lt.Scalar(clone_15, -3)
    le_7: "b8[4, 960, 7, 7]" = torch.ops.aten.le.Scalar(clone_15, 3)
    div_37: "f32[4, 960, 7, 7]" = torch.ops.aten.div.Tensor(clone_15, 3);  clone_15 = None
    add_145: "f32[4, 960, 7, 7]" = torch.ops.aten.add.Tensor(div_37, 0.5);  div_37 = None
    mul_226: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_27, add_145);  add_145 = None
    where_14: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(le_7, mul_226, getitem_27);  le_7 = mul_226 = getitem_27 = None
    where_15: "f32[4, 960, 7, 7]" = torch.ops.aten.where.self(lt_7, full_default, where_14);  lt_7 = where_14 = None
    add_146: "f32[960]" = torch.ops.aten.add.Tensor(primals_293, 0.001);  primals_293 = None
    rsqrt_6: "f32[960]" = torch.ops.aten.rsqrt.default(add_146);  add_146 = None
    unsqueeze_440: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(primals_292, 0);  primals_292 = None
    unsqueeze_441: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, 2);  unsqueeze_440 = None
    unsqueeze_442: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_441, 3);  unsqueeze_441 = None
    sum_21: "f32[960]" = torch.ops.aten.sum.dim_IntList(where_15, [0, 2, 3])
    sub_52: "f32[4, 960, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_442);  convolution_51 = unsqueeze_442 = None
    mul_227: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, sub_52);  sub_52 = None
    sum_22: "f32[960]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 2, 3]);  mul_227 = None
    mul_232: "f32[960]" = torch.ops.aten.mul.Tensor(rsqrt_6, primals_143);  primals_143 = None
    unsqueeze_449: "f32[1, 960]" = torch.ops.aten.unsqueeze.default(mul_232, 0);  mul_232 = None
    unsqueeze_450: "f32[1, 960, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_449, 2);  unsqueeze_449 = None
    unsqueeze_451: "f32[1, 960, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, 3);  unsqueeze_450 = None
    mul_233: "f32[4, 960, 7, 7]" = torch.ops.aten.mul.Tensor(where_15, unsqueeze_451);  where_15 = unsqueeze_451 = None
    mul_234: "f32[960]" = torch.ops.aten.mul.Tensor(sum_22, rsqrt_6);  sum_22 = rsqrt_6 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(mul_233, add_106, primals_142, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_233 = add_106 = primals_142 = None
    getitem_30: "f32[4, 160, 7, 7]" = convolution_backward_10[0]
    getitem_31: "f32[960, 160, 1, 1]" = convolution_backward_10[1];  convolution_backward_10 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_147: "f32[4, 160, 7, 7]" = torch.ops.aten.add.Tensor(add_140, getitem_30);  add_140 = getitem_30 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_148: "f32[160]" = torch.ops.aten.add.Tensor(primals_290, 0.001);  primals_290 = None
    rsqrt_7: "f32[160]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
    unsqueeze_452: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(primals_289, 0);  primals_289 = None
    unsqueeze_453: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, 2);  unsqueeze_452 = None
    unsqueeze_454: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 3);  unsqueeze_453 = None
    sum_23: "f32[160]" = torch.ops.aten.sum.dim_IntList(add_147, [0, 2, 3])
    sub_53: "f32[4, 160, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_454);  convolution_50 = unsqueeze_454 = None
    mul_235: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_147, sub_53);  sub_53 = None
    sum_24: "f32[160]" = torch.ops.aten.sum.dim_IntList(mul_235, [0, 2, 3]);  mul_235 = None
    mul_240: "f32[160]" = torch.ops.aten.mul.Tensor(rsqrt_7, primals_140);  primals_140 = None
    unsqueeze_461: "f32[1, 160]" = torch.ops.aten.unsqueeze.default(mul_240, 0);  mul_240 = None
    unsqueeze_462: "f32[1, 160, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_461, 2);  unsqueeze_461 = None
    unsqueeze_463: "f32[1, 160, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, 3);  unsqueeze_462 = None
    mul_241: "f32[4, 160, 7, 7]" = torch.ops.aten.mul.Tensor(add_147, unsqueeze_463);  add_147 = unsqueeze_463 = None
    mul_242: "f32[160]" = torch.ops.aten.mul.Tensor(sum_24, rsqrt_7);  sum_24 = rsqrt_7 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(mul_241, mul_134, primals_139, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_241 = mul_134 = primals_139 = None
    getitem_33: "f32[4, 672, 7, 7]" = convolution_backward_11[0]
    getitem_34: "f32[160, 672, 1, 1]" = convolution_backward_11[1];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_243: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_33, div_20);  div_20 = None
    mul_244: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(getitem_33, div_19);  getitem_33 = div_19 = None
    sum_25: "f32[4, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [2, 3], True);  mul_244 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    mul_245: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_25, 0.16666666666666666);  sum_25 = None
    where_16: "f32[4, 672, 1, 1]" = torch.ops.aten.where.self(bitwise_and_2, mul_245, full_default);  bitwise_and_2 = mul_245 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_26: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_16, [0, 2, 3])
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(where_16, relu_16, primals_137, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_16 = primals_137 = None
    getitem_36: "f32[4, 168, 1, 1]" = convolution_backward_12[0]
    getitem_37: "f32[672, 168, 1, 1]" = convolution_backward_12[1];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    le_8: "b8[4, 168, 1, 1]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
    where_17: "f32[4, 168, 1, 1]" = torch.ops.aten.where.self(le_8, full_default, getitem_36);  le_8 = getitem_36 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_27: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_17, [0, 2, 3])
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(where_17, mean_5, primals_135, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_17 = mean_5 = primals_135 = None
    getitem_39: "f32[4, 672, 1, 1]" = convolution_backward_13[0]
    getitem_40: "f32[168, 672, 1, 1]" = convolution_backward_13[1];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_3: "f32[4, 672, 7, 7]" = torch.ops.aten.expand.default(getitem_39, [4, 672, 7, 7]);  getitem_39 = None
    div_38: "f32[4, 672, 7, 7]" = torch.ops.aten.div.Scalar(expand_3, 49);  expand_3 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_149: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(mul_243, div_38);  mul_243 = div_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    lt_9: "b8[4, 672, 7, 7]" = torch.ops.aten.lt.Scalar(clone_14, -3)
    le_9: "b8[4, 672, 7, 7]" = torch.ops.aten.le.Scalar(clone_14, 3)
    div_39: "f32[4, 672, 7, 7]" = torch.ops.aten.div.Tensor(clone_14, 3);  clone_14 = None
    add_150: "f32[4, 672, 7, 7]" = torch.ops.aten.add.Tensor(div_39, 0.5);  div_39 = None
    mul_246: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(add_149, add_150);  add_150 = None
    where_18: "f32[4, 672, 7, 7]" = torch.ops.aten.where.self(le_9, mul_246, add_149);  le_9 = mul_246 = add_149 = None
    where_19: "f32[4, 672, 7, 7]" = torch.ops.aten.where.self(lt_9, full_default, where_18);  lt_9 = where_18 = None
    add_151: "f32[672]" = torch.ops.aten.add.Tensor(primals_287, 0.001);  primals_287 = None
    rsqrt_8: "f32[672]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    unsqueeze_464: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_286, 0);  primals_286 = None
    unsqueeze_465: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, 2);  unsqueeze_464 = None
    unsqueeze_466: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_465, 3);  unsqueeze_465 = None
    sum_28: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_19, [0, 2, 3])
    sub_54: "f32[4, 672, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_466);  convolution_47 = unsqueeze_466 = None
    mul_247: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(where_19, sub_54);  sub_54 = None
    sum_29: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_247, [0, 2, 3]);  mul_247 = None
    mul_252: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_8, primals_133);  primals_133 = None
    unsqueeze_473: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_252, 0);  mul_252 = None
    unsqueeze_474: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_473, 2);  unsqueeze_473 = None
    unsqueeze_475: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, 3);  unsqueeze_474 = None
    mul_253: "f32[4, 672, 7, 7]" = torch.ops.aten.mul.Tensor(where_19, unsqueeze_475);  where_19 = unsqueeze_475 = None
    mul_254: "f32[672]" = torch.ops.aten.mul.Tensor(sum_29, rsqrt_8);  sum_29 = rsqrt_8 = None
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(mul_253, div_18, primals_132, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_253 = div_18 = primals_132 = None
    getitem_42: "f32[4, 672, 14, 14]" = convolution_backward_14[0]
    getitem_43: "f32[672, 1, 5, 5]" = convolution_backward_14[1];  convolution_backward_14 = None
    lt_10: "b8[4, 672, 14, 14]" = torch.ops.aten.lt.Scalar(clone_13, -3)
    le_10: "b8[4, 672, 14, 14]" = torch.ops.aten.le.Scalar(clone_13, 3)
    div_40: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Tensor(clone_13, 3);  clone_13 = None
    add_152: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(div_40, 0.5);  div_40 = None
    mul_255: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_42, add_152);  add_152 = None
    where_20: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(le_10, mul_255, getitem_42);  le_10 = mul_255 = getitem_42 = None
    where_21: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(lt_10, full_default, where_20);  lt_10 = where_20 = None
    add_153: "f32[672]" = torch.ops.aten.add.Tensor(primals_284, 0.001);  primals_284 = None
    rsqrt_9: "f32[672]" = torch.ops.aten.rsqrt.default(add_153);  add_153 = None
    unsqueeze_476: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_283, 0);  primals_283 = None
    unsqueeze_477: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 2);  unsqueeze_476 = None
    unsqueeze_478: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_477, 3);  unsqueeze_477 = None
    sum_30: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_21, [0, 2, 3])
    sub_55: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_478);  convolution_46 = unsqueeze_478 = None
    mul_256: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, sub_55);  sub_55 = None
    sum_31: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 2, 3]);  mul_256 = None
    mul_261: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_9, primals_130);  primals_130 = None
    unsqueeze_485: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_261, 0);  mul_261 = None
    unsqueeze_486: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_485, 2);  unsqueeze_485 = None
    unsqueeze_487: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, 3);  unsqueeze_486 = None
    mul_262: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_21, unsqueeze_487);  where_21 = unsqueeze_487 = None
    mul_263: "f32[672]" = torch.ops.aten.mul.Tensor(sum_31, rsqrt_9);  sum_31 = rsqrt_9 = None
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(mul_262, add_97, primals_129, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_262 = add_97 = primals_129 = None
    getitem_45: "f32[4, 112, 14, 14]" = convolution_backward_15[0]
    getitem_46: "f32[672, 112, 1, 1]" = convolution_backward_15[1];  convolution_backward_15 = None
    add_154: "f32[112]" = torch.ops.aten.add.Tensor(primals_281, 0.001);  primals_281 = None
    rsqrt_10: "f32[112]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    unsqueeze_488: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(primals_280, 0);  primals_280 = None
    unsqueeze_489: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 2);  unsqueeze_488 = None
    unsqueeze_490: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_489, 3);  unsqueeze_489 = None
    sum_32: "f32[112]" = torch.ops.aten.sum.dim_IntList(getitem_45, [0, 2, 3])
    sub_56: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_490);  convolution_45 = unsqueeze_490 = None
    mul_264: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_45, sub_56);  sub_56 = None
    sum_33: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_264, [0, 2, 3]);  mul_264 = None
    mul_269: "f32[112]" = torch.ops.aten.mul.Tensor(rsqrt_10, primals_127);  primals_127 = None
    unsqueeze_497: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_269, 0);  mul_269 = None
    unsqueeze_498: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_497, 2);  unsqueeze_497 = None
    unsqueeze_499: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, 3);  unsqueeze_498 = None
    mul_270: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_45, unsqueeze_499);  unsqueeze_499 = None
    mul_271: "f32[112]" = torch.ops.aten.mul.Tensor(sum_33, rsqrt_10);  sum_33 = rsqrt_10 = None
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(mul_270, mul_122, primals_126, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_270 = mul_122 = primals_126 = None
    getitem_48: "f32[4, 672, 14, 14]" = convolution_backward_16[0]
    getitem_49: "f32[112, 672, 1, 1]" = convolution_backward_16[1];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_272: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_48, div_17);  div_17 = None
    mul_273: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_48, div_16);  getitem_48 = div_16 = None
    sum_34: "f32[4, 672, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2, 3], True);  mul_273 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    mul_274: "f32[4, 672, 1, 1]" = torch.ops.aten.mul.Tensor(sum_34, 0.16666666666666666);  sum_34 = None
    where_22: "f32[4, 672, 1, 1]" = torch.ops.aten.where.self(bitwise_and_3, mul_274, full_default);  bitwise_and_3 = mul_274 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_35: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_22, [0, 2, 3])
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(where_22, relu_15, primals_124, [672], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_22 = primals_124 = None
    getitem_51: "f32[4, 168, 1, 1]" = convolution_backward_17[0]
    getitem_52: "f32[672, 168, 1, 1]" = convolution_backward_17[1];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    le_11: "b8[4, 168, 1, 1]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
    where_23: "f32[4, 168, 1, 1]" = torch.ops.aten.where.self(le_11, full_default, getitem_51);  le_11 = getitem_51 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_36: "f32[168]" = torch.ops.aten.sum.dim_IntList(where_23, [0, 2, 3])
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(where_23, mean_4, primals_122, [168], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_23 = mean_4 = primals_122 = None
    getitem_54: "f32[4, 672, 1, 1]" = convolution_backward_18[0]
    getitem_55: "f32[168, 672, 1, 1]" = convolution_backward_18[1];  convolution_backward_18 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_4: "f32[4, 672, 14, 14]" = torch.ops.aten.expand.default(getitem_54, [4, 672, 14, 14]);  getitem_54 = None
    div_41: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Scalar(expand_4, 196);  expand_4 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_155: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(mul_272, div_41);  mul_272 = div_41 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    lt_12: "b8[4, 672, 14, 14]" = torch.ops.aten.lt.Scalar(clone_12, -3)
    le_12: "b8[4, 672, 14, 14]" = torch.ops.aten.le.Scalar(clone_12, 3)
    div_42: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Tensor(clone_12, 3);  clone_12 = None
    add_156: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(div_42, 0.5);  div_42 = None
    mul_275: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(add_155, add_156);  add_156 = None
    where_24: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(le_12, mul_275, add_155);  le_12 = mul_275 = add_155 = None
    where_25: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(lt_12, full_default, where_24);  lt_12 = where_24 = None
    add_157: "f32[672]" = torch.ops.aten.add.Tensor(primals_278, 0.001);  primals_278 = None
    rsqrt_11: "f32[672]" = torch.ops.aten.rsqrt.default(add_157);  add_157 = None
    unsqueeze_500: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_277, 0);  primals_277 = None
    unsqueeze_501: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, 2);  unsqueeze_500 = None
    unsqueeze_502: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 3);  unsqueeze_501 = None
    sum_37: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_25, [0, 2, 3])
    sub_57: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_502);  convolution_42 = unsqueeze_502 = None
    mul_276: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, sub_57);  sub_57 = None
    sum_38: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 2, 3]);  mul_276 = None
    mul_281: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_11, primals_120);  primals_120 = None
    unsqueeze_509: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_281, 0);  mul_281 = None
    unsqueeze_510: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 2);  unsqueeze_509 = None
    unsqueeze_511: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, 3);  unsqueeze_510 = None
    mul_282: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_25, unsqueeze_511);  where_25 = unsqueeze_511 = None
    mul_283: "f32[672]" = torch.ops.aten.mul.Tensor(sum_38, rsqrt_11);  sum_38 = rsqrt_11 = None
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(mul_282, div_15, primals_119, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 672, [True, True, False]);  mul_282 = div_15 = primals_119 = None
    getitem_57: "f32[4, 672, 14, 14]" = convolution_backward_19[0]
    getitem_58: "f32[672, 1, 3, 3]" = convolution_backward_19[1];  convolution_backward_19 = None
    lt_13: "b8[4, 672, 14, 14]" = torch.ops.aten.lt.Scalar(clone_11, -3)
    le_13: "b8[4, 672, 14, 14]" = torch.ops.aten.le.Scalar(clone_11, 3)
    div_43: "f32[4, 672, 14, 14]" = torch.ops.aten.div.Tensor(clone_11, 3);  clone_11 = None
    add_158: "f32[4, 672, 14, 14]" = torch.ops.aten.add.Tensor(div_43, 0.5);  div_43 = None
    mul_284: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_57, add_158);  add_158 = None
    where_26: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(le_13, mul_284, getitem_57);  le_13 = mul_284 = getitem_57 = None
    where_27: "f32[4, 672, 14, 14]" = torch.ops.aten.where.self(lt_13, full_default, where_26);  lt_13 = where_26 = None
    add_159: "f32[672]" = torch.ops.aten.add.Tensor(primals_275, 0.001);  primals_275 = None
    rsqrt_12: "f32[672]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
    unsqueeze_512: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(primals_274, 0);  primals_274 = None
    unsqueeze_513: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, 2);  unsqueeze_512 = None
    unsqueeze_514: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 3);  unsqueeze_513 = None
    sum_39: "f32[672]" = torch.ops.aten.sum.dim_IntList(where_27, [0, 2, 3])
    sub_58: "f32[4, 672, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_514);  convolution_41 = unsqueeze_514 = None
    mul_285: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, sub_58);  sub_58 = None
    sum_40: "f32[672]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 2, 3]);  mul_285 = None
    mul_290: "f32[672]" = torch.ops.aten.mul.Tensor(rsqrt_12, primals_117);  primals_117 = None
    unsqueeze_521: "f32[1, 672]" = torch.ops.aten.unsqueeze.default(mul_290, 0);  mul_290 = None
    unsqueeze_522: "f32[1, 672, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_521, 2);  unsqueeze_521 = None
    unsqueeze_523: "f32[1, 672, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, 3);  unsqueeze_522 = None
    mul_291: "f32[4, 672, 14, 14]" = torch.ops.aten.mul.Tensor(where_27, unsqueeze_523);  where_27 = unsqueeze_523 = None
    mul_292: "f32[672]" = torch.ops.aten.mul.Tensor(sum_40, rsqrt_12);  sum_40 = rsqrt_12 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(mul_291, add_87, primals_116, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_291 = add_87 = primals_116 = None
    getitem_60: "f32[4, 112, 14, 14]" = convolution_backward_20[0]
    getitem_61: "f32[672, 112, 1, 1]" = convolution_backward_20[1];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_160: "f32[4, 112, 14, 14]" = torch.ops.aten.add.Tensor(getitem_45, getitem_60);  getitem_45 = getitem_60 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_161: "f32[112]" = torch.ops.aten.add.Tensor(primals_272, 0.001);  primals_272 = None
    rsqrt_13: "f32[112]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    unsqueeze_524: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(primals_271, 0);  primals_271 = None
    unsqueeze_525: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, 2);  unsqueeze_524 = None
    unsqueeze_526: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_525, 3);  unsqueeze_525 = None
    sum_41: "f32[112]" = torch.ops.aten.sum.dim_IntList(add_160, [0, 2, 3])
    sub_59: "f32[4, 112, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_526);  convolution_40 = unsqueeze_526 = None
    mul_293: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_160, sub_59);  sub_59 = None
    sum_42: "f32[112]" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 2, 3]);  mul_293 = None
    mul_298: "f32[112]" = torch.ops.aten.mul.Tensor(rsqrt_13, primals_114);  primals_114 = None
    unsqueeze_533: "f32[1, 112]" = torch.ops.aten.unsqueeze.default(mul_298, 0);  mul_298 = None
    unsqueeze_534: "f32[1, 112, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_533, 2);  unsqueeze_533 = None
    unsqueeze_535: "f32[1, 112, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 3);  unsqueeze_534 = None
    mul_299: "f32[4, 112, 14, 14]" = torch.ops.aten.mul.Tensor(add_160, unsqueeze_535);  add_160 = unsqueeze_535 = None
    mul_300: "f32[112]" = torch.ops.aten.mul.Tensor(sum_42, rsqrt_13);  sum_42 = rsqrt_13 = None
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(mul_299, mul_110, primals_113, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_299 = mul_110 = primals_113 = None
    getitem_63: "f32[4, 480, 14, 14]" = convolution_backward_21[0]
    getitem_64: "f32[112, 480, 1, 1]" = convolution_backward_21[1];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_301: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_63, div_14);  div_14 = None
    mul_302: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_63, div_13);  getitem_63 = div_13 = None
    sum_43: "f32[4, 480, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2, 3], True);  mul_302 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    mul_303: "f32[4, 480, 1, 1]" = torch.ops.aten.mul.Tensor(sum_43, 0.16666666666666666);  sum_43 = None
    where_28: "f32[4, 480, 1, 1]" = torch.ops.aten.where.self(bitwise_and_4, mul_303, full_default);  bitwise_and_4 = mul_303 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_44: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_28, [0, 2, 3])
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(where_28, relu_14, primals_111, [480], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_28 = primals_111 = None
    getitem_66: "f32[4, 120, 1, 1]" = convolution_backward_22[0]
    getitem_67: "f32[480, 120, 1, 1]" = convolution_backward_22[1];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    le_14: "b8[4, 120, 1, 1]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
    where_29: "f32[4, 120, 1, 1]" = torch.ops.aten.where.self(le_14, full_default, getitem_66);  le_14 = getitem_66 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_45: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_29, [0, 2, 3])
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(where_29, mean_3, primals_109, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_29 = mean_3 = primals_109 = None
    getitem_69: "f32[4, 480, 1, 1]" = convolution_backward_23[0]
    getitem_70: "f32[120, 480, 1, 1]" = convolution_backward_23[1];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_5: "f32[4, 480, 14, 14]" = torch.ops.aten.expand.default(getitem_69, [4, 480, 14, 14]);  getitem_69 = None
    div_44: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Scalar(expand_5, 196);  expand_5 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_162: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(mul_301, div_44);  mul_301 = div_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    lt_15: "b8[4, 480, 14, 14]" = torch.ops.aten.lt.Scalar(clone_10, -3)
    le_15: "b8[4, 480, 14, 14]" = torch.ops.aten.le.Scalar(clone_10, 3)
    div_45: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Tensor(clone_10, 3);  clone_10 = None
    add_163: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(div_45, 0.5);  div_45 = None
    mul_304: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(add_162, add_163);  add_163 = None
    where_30: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(le_15, mul_304, add_162);  le_15 = mul_304 = add_162 = None
    where_31: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(lt_15, full_default, where_30);  lt_15 = where_30 = None
    add_164: "f32[480]" = torch.ops.aten.add.Tensor(primals_269, 0.001);  primals_269 = None
    rsqrt_14: "f32[480]" = torch.ops.aten.rsqrt.default(add_164);  add_164 = None
    unsqueeze_536: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_268, 0);  primals_268 = None
    unsqueeze_537: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, 2);  unsqueeze_536 = None
    unsqueeze_538: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_537, 3);  unsqueeze_537 = None
    sum_46: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_31, [0, 2, 3])
    sub_60: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_538);  convolution_37 = unsqueeze_538 = None
    mul_305: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, sub_60);  sub_60 = None
    sum_47: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 2, 3]);  mul_305 = None
    mul_310: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_14, primals_107);  primals_107 = None
    unsqueeze_545: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_310, 0);  mul_310 = None
    unsqueeze_546: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_545, 2);  unsqueeze_545 = None
    unsqueeze_547: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, 3);  unsqueeze_546 = None
    mul_311: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_31, unsqueeze_547);  where_31 = unsqueeze_547 = None
    mul_312: "f32[480]" = torch.ops.aten.mul.Tensor(sum_47, rsqrt_14);  sum_47 = rsqrt_14 = None
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(mul_311, div_12, primals_106, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 480, [True, True, False]);  mul_311 = div_12 = primals_106 = None
    getitem_72: "f32[4, 480, 14, 14]" = convolution_backward_24[0]
    getitem_73: "f32[480, 1, 3, 3]" = convolution_backward_24[1];  convolution_backward_24 = None
    lt_16: "b8[4, 480, 14, 14]" = torch.ops.aten.lt.Scalar(clone_9, -3)
    le_16: "b8[4, 480, 14, 14]" = torch.ops.aten.le.Scalar(clone_9, 3)
    div_46: "f32[4, 480, 14, 14]" = torch.ops.aten.div.Tensor(clone_9, 3);  clone_9 = None
    add_165: "f32[4, 480, 14, 14]" = torch.ops.aten.add.Tensor(div_46, 0.5);  div_46 = None
    mul_313: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_72, add_165);  add_165 = None
    where_32: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(le_16, mul_313, getitem_72);  le_16 = mul_313 = getitem_72 = None
    where_33: "f32[4, 480, 14, 14]" = torch.ops.aten.where.self(lt_16, full_default, where_32);  lt_16 = where_32 = None
    add_166: "f32[480]" = torch.ops.aten.add.Tensor(primals_266, 0.001);  primals_266 = None
    rsqrt_15: "f32[480]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
    unsqueeze_548: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(primals_265, 0);  primals_265 = None
    unsqueeze_549: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, 2);  unsqueeze_548 = None
    unsqueeze_550: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_549, 3);  unsqueeze_549 = None
    sum_48: "f32[480]" = torch.ops.aten.sum.dim_IntList(where_33, [0, 2, 3])
    sub_61: "f32[4, 480, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_550);  convolution_36 = unsqueeze_550 = None
    mul_314: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, sub_61);  sub_61 = None
    sum_49: "f32[480]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 2, 3]);  mul_314 = None
    mul_319: "f32[480]" = torch.ops.aten.mul.Tensor(rsqrt_15, primals_104);  primals_104 = None
    unsqueeze_557: "f32[1, 480]" = torch.ops.aten.unsqueeze.default(mul_319, 0);  mul_319 = None
    unsqueeze_558: "f32[1, 480, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_557, 2);  unsqueeze_557 = None
    unsqueeze_559: "f32[1, 480, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, 3);  unsqueeze_558 = None
    mul_320: "f32[4, 480, 14, 14]" = torch.ops.aten.mul.Tensor(where_33, unsqueeze_559);  where_33 = unsqueeze_559 = None
    mul_321: "f32[480]" = torch.ops.aten.mul.Tensor(sum_49, rsqrt_15);  sum_49 = rsqrt_15 = None
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(mul_320, add_78, primals_103, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_320 = add_78 = primals_103 = None
    getitem_75: "f32[4, 80, 14, 14]" = convolution_backward_25[0]
    getitem_76: "f32[480, 80, 1, 1]" = convolution_backward_25[1];  convolution_backward_25 = None
    add_167: "f32[80]" = torch.ops.aten.add.Tensor(primals_263, 0.001);  primals_263 = None
    rsqrt_16: "f32[80]" = torch.ops.aten.rsqrt.default(add_167);  add_167 = None
    unsqueeze_560: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_262, 0);  primals_262 = None
    unsqueeze_561: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, 2);  unsqueeze_560 = None
    unsqueeze_562: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_561, 3);  unsqueeze_561 = None
    sum_50: "f32[80]" = torch.ops.aten.sum.dim_IntList(getitem_75, [0, 2, 3])
    sub_62: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_562);  convolution_35 = unsqueeze_562 = None
    mul_322: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_75, sub_62);  sub_62 = None
    sum_51: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_322, [0, 2, 3]);  mul_322 = None
    mul_327: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_16, primals_101);  primals_101 = None
    unsqueeze_569: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_327, 0);  mul_327 = None
    unsqueeze_570: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_569, 2);  unsqueeze_569 = None
    unsqueeze_571: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, 3);  unsqueeze_570 = None
    mul_328: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_75, unsqueeze_571);  unsqueeze_571 = None
    mul_329: "f32[80]" = torch.ops.aten.mul.Tensor(sum_51, rsqrt_16);  sum_51 = rsqrt_16 = None
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(mul_328, div_11, primals_100, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_328 = div_11 = primals_100 = None
    getitem_78: "f32[4, 184, 14, 14]" = convolution_backward_26[0]
    getitem_79: "f32[80, 184, 1, 1]" = convolution_backward_26[1];  convolution_backward_26 = None
    lt_17: "b8[4, 184, 14, 14]" = torch.ops.aten.lt.Scalar(clone_8, -3)
    le_17: "b8[4, 184, 14, 14]" = torch.ops.aten.le.Scalar(clone_8, 3)
    div_47: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(clone_8, 3);  clone_8 = None
    add_168: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(div_47, 0.5);  div_47 = None
    mul_330: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_78, add_168);  add_168 = None
    where_34: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(le_17, mul_330, getitem_78);  le_17 = mul_330 = getitem_78 = None
    where_35: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(lt_17, full_default, where_34);  lt_17 = where_34 = None
    add_169: "f32[184]" = torch.ops.aten.add.Tensor(primals_260, 0.001);  primals_260 = None
    rsqrt_17: "f32[184]" = torch.ops.aten.rsqrt.default(add_169);  add_169 = None
    unsqueeze_572: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(primals_259, 0);  primals_259 = None
    unsqueeze_573: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, 2);  unsqueeze_572 = None
    unsqueeze_574: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_573, 3);  unsqueeze_573 = None
    sum_52: "f32[184]" = torch.ops.aten.sum.dim_IntList(where_35, [0, 2, 3])
    sub_63: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_574);  convolution_34 = unsqueeze_574 = None
    mul_331: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, sub_63);  sub_63 = None
    sum_53: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 2, 3]);  mul_331 = None
    mul_336: "f32[184]" = torch.ops.aten.mul.Tensor(rsqrt_17, primals_98);  primals_98 = None
    unsqueeze_581: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_336, 0);  mul_336 = None
    unsqueeze_582: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_581, 2);  unsqueeze_581 = None
    unsqueeze_583: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, 3);  unsqueeze_582 = None
    mul_337: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_35, unsqueeze_583);  where_35 = unsqueeze_583 = None
    mul_338: "f32[184]" = torch.ops.aten.mul.Tensor(sum_53, rsqrt_17);  sum_53 = rsqrt_17 = None
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(mul_337, div_10, primals_97, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False]);  mul_337 = div_10 = primals_97 = None
    getitem_81: "f32[4, 184, 14, 14]" = convolution_backward_27[0]
    getitem_82: "f32[184, 1, 3, 3]" = convolution_backward_27[1];  convolution_backward_27 = None
    lt_18: "b8[4, 184, 14, 14]" = torch.ops.aten.lt.Scalar(clone_7, -3)
    le_18: "b8[4, 184, 14, 14]" = torch.ops.aten.le.Scalar(clone_7, 3)
    div_48: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(clone_7, 3);  clone_7 = None
    add_170: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(div_48, 0.5);  div_48 = None
    mul_339: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_81, add_170);  add_170 = None
    where_36: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(le_18, mul_339, getitem_81);  le_18 = mul_339 = getitem_81 = None
    where_37: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(lt_18, full_default, where_36);  lt_18 = where_36 = None
    add_171: "f32[184]" = torch.ops.aten.add.Tensor(primals_257, 0.001);  primals_257 = None
    rsqrt_18: "f32[184]" = torch.ops.aten.rsqrt.default(add_171);  add_171 = None
    unsqueeze_584: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(primals_256, 0);  primals_256 = None
    unsqueeze_585: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 2);  unsqueeze_584 = None
    unsqueeze_586: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_585, 3);  unsqueeze_585 = None
    sum_54: "f32[184]" = torch.ops.aten.sum.dim_IntList(where_37, [0, 2, 3])
    sub_64: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_586);  convolution_33 = unsqueeze_586 = None
    mul_340: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, sub_64);  sub_64 = None
    sum_55: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 2, 3]);  mul_340 = None
    mul_345: "f32[184]" = torch.ops.aten.mul.Tensor(rsqrt_18, primals_95);  primals_95 = None
    unsqueeze_593: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_345, 0);  mul_345 = None
    unsqueeze_594: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_593, 2);  unsqueeze_593 = None
    unsqueeze_595: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, 3);  unsqueeze_594 = None
    mul_346: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_37, unsqueeze_595);  where_37 = unsqueeze_595 = None
    mul_347: "f32[184]" = torch.ops.aten.mul.Tensor(sum_55, rsqrt_18);  sum_55 = rsqrt_18 = None
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(mul_346, add_69, primals_94, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_346 = add_69 = primals_94 = None
    getitem_84: "f32[4, 80, 14, 14]" = convolution_backward_28[0]
    getitem_85: "f32[184, 80, 1, 1]" = convolution_backward_28[1];  convolution_backward_28 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_172: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(getitem_75, getitem_84);  getitem_75 = getitem_84 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_173: "f32[80]" = torch.ops.aten.add.Tensor(primals_254, 0.001);  primals_254 = None
    rsqrt_19: "f32[80]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
    unsqueeze_596: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_253, 0);  primals_253 = None
    unsqueeze_597: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, 2);  unsqueeze_596 = None
    unsqueeze_598: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_597, 3);  unsqueeze_597 = None
    sum_56: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_172, [0, 2, 3])
    sub_65: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_598);  convolution_32 = unsqueeze_598 = None
    mul_348: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_172, sub_65);  sub_65 = None
    sum_57: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_348, [0, 2, 3]);  mul_348 = None
    mul_353: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_19, primals_92);  primals_92 = None
    unsqueeze_605: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_353, 0);  mul_353 = None
    unsqueeze_606: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_605, 2);  unsqueeze_605 = None
    unsqueeze_607: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, 3);  unsqueeze_606 = None
    mul_354: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_172, unsqueeze_607);  unsqueeze_607 = None
    mul_355: "f32[80]" = torch.ops.aten.mul.Tensor(sum_57, rsqrt_19);  sum_57 = rsqrt_19 = None
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(mul_354, div_9, primals_91, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_354 = div_9 = primals_91 = None
    getitem_87: "f32[4, 184, 14, 14]" = convolution_backward_29[0]
    getitem_88: "f32[80, 184, 1, 1]" = convolution_backward_29[1];  convolution_backward_29 = None
    lt_19: "b8[4, 184, 14, 14]" = torch.ops.aten.lt.Scalar(clone_6, -3)
    le_19: "b8[4, 184, 14, 14]" = torch.ops.aten.le.Scalar(clone_6, 3)
    div_49: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(clone_6, 3);  clone_6 = None
    add_174: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(div_49, 0.5);  div_49 = None
    mul_356: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_87, add_174);  add_174 = None
    where_38: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(le_19, mul_356, getitem_87);  le_19 = mul_356 = getitem_87 = None
    where_39: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(lt_19, full_default, where_38);  lt_19 = where_38 = None
    add_175: "f32[184]" = torch.ops.aten.add.Tensor(primals_251, 0.001);  primals_251 = None
    rsqrt_20: "f32[184]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    unsqueeze_608: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(primals_250, 0);  primals_250 = None
    unsqueeze_609: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, 2);  unsqueeze_608 = None
    unsqueeze_610: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_609, 3);  unsqueeze_609 = None
    sum_58: "f32[184]" = torch.ops.aten.sum.dim_IntList(where_39, [0, 2, 3])
    sub_66: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_610);  convolution_31 = unsqueeze_610 = None
    mul_357: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, sub_66);  sub_66 = None
    sum_59: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_357, [0, 2, 3]);  mul_357 = None
    mul_362: "f32[184]" = torch.ops.aten.mul.Tensor(rsqrt_20, primals_89);  primals_89 = None
    unsqueeze_617: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_362, 0);  mul_362 = None
    unsqueeze_618: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_617, 2);  unsqueeze_617 = None
    unsqueeze_619: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, 3);  unsqueeze_618 = None
    mul_363: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_39, unsqueeze_619);  where_39 = unsqueeze_619 = None
    mul_364: "f32[184]" = torch.ops.aten.mul.Tensor(sum_59, rsqrt_20);  sum_59 = rsqrt_20 = None
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(mul_363, div_8, primals_88, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 184, [True, True, False]);  mul_363 = div_8 = primals_88 = None
    getitem_90: "f32[4, 184, 14, 14]" = convolution_backward_30[0]
    getitem_91: "f32[184, 1, 3, 3]" = convolution_backward_30[1];  convolution_backward_30 = None
    lt_20: "b8[4, 184, 14, 14]" = torch.ops.aten.lt.Scalar(clone_5, -3)
    le_20: "b8[4, 184, 14, 14]" = torch.ops.aten.le.Scalar(clone_5, 3)
    div_50: "f32[4, 184, 14, 14]" = torch.ops.aten.div.Tensor(clone_5, 3);  clone_5 = None
    add_176: "f32[4, 184, 14, 14]" = torch.ops.aten.add.Tensor(div_50, 0.5);  div_50 = None
    mul_365: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_90, add_176);  add_176 = None
    where_40: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(le_20, mul_365, getitem_90);  le_20 = mul_365 = getitem_90 = None
    where_41: "f32[4, 184, 14, 14]" = torch.ops.aten.where.self(lt_20, full_default, where_40);  lt_20 = where_40 = None
    add_177: "f32[184]" = torch.ops.aten.add.Tensor(primals_248, 0.001);  primals_248 = None
    rsqrt_21: "f32[184]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
    unsqueeze_620: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(primals_247, 0);  primals_247 = None
    unsqueeze_621: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, 2);  unsqueeze_620 = None
    unsqueeze_622: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_621, 3);  unsqueeze_621 = None
    sum_60: "f32[184]" = torch.ops.aten.sum.dim_IntList(where_41, [0, 2, 3])
    sub_67: "f32[4, 184, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_622);  convolution_30 = unsqueeze_622 = None
    mul_366: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, sub_67);  sub_67 = None
    sum_61: "f32[184]" = torch.ops.aten.sum.dim_IntList(mul_366, [0, 2, 3]);  mul_366 = None
    mul_371: "f32[184]" = torch.ops.aten.mul.Tensor(rsqrt_21, primals_86);  primals_86 = None
    unsqueeze_629: "f32[1, 184]" = torch.ops.aten.unsqueeze.default(mul_371, 0);  mul_371 = None
    unsqueeze_630: "f32[1, 184, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_629, 2);  unsqueeze_629 = None
    unsqueeze_631: "f32[1, 184, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, 3);  unsqueeze_630 = None
    mul_372: "f32[4, 184, 14, 14]" = torch.ops.aten.mul.Tensor(where_41, unsqueeze_631);  where_41 = unsqueeze_631 = None
    mul_373: "f32[184]" = torch.ops.aten.mul.Tensor(sum_61, rsqrt_21);  sum_61 = rsqrt_21 = None
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(mul_372, add_60, primals_85, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_372 = add_60 = primals_85 = None
    getitem_93: "f32[4, 80, 14, 14]" = convolution_backward_31[0]
    getitem_94: "f32[184, 80, 1, 1]" = convolution_backward_31[1];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_178: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_172, getitem_93);  add_172 = getitem_93 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_179: "f32[80]" = torch.ops.aten.add.Tensor(primals_245, 0.001);  primals_245 = None
    rsqrt_22: "f32[80]" = torch.ops.aten.rsqrt.default(add_179);  add_179 = None
    unsqueeze_632: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_244, 0);  primals_244 = None
    unsqueeze_633: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, 2);  unsqueeze_632 = None
    unsqueeze_634: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_633, 3);  unsqueeze_633 = None
    sum_62: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_178, [0, 2, 3])
    sub_68: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_634);  convolution_29 = unsqueeze_634 = None
    mul_374: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_178, sub_68);  sub_68 = None
    sum_63: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 2, 3]);  mul_374 = None
    mul_379: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_22, primals_83);  primals_83 = None
    unsqueeze_641: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_379, 0);  mul_379 = None
    unsqueeze_642: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_641, 2);  unsqueeze_641 = None
    unsqueeze_643: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, 3);  unsqueeze_642 = None
    mul_380: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_178, unsqueeze_643);  unsqueeze_643 = None
    mul_381: "f32[80]" = torch.ops.aten.mul.Tensor(sum_63, rsqrt_22);  sum_63 = rsqrt_22 = None
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(mul_380, div_7, primals_82, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_380 = div_7 = primals_82 = None
    getitem_96: "f32[4, 200, 14, 14]" = convolution_backward_32[0]
    getitem_97: "f32[80, 200, 1, 1]" = convolution_backward_32[1];  convolution_backward_32 = None
    lt_21: "b8[4, 200, 14, 14]" = torch.ops.aten.lt.Scalar(clone_4, -3)
    le_21: "b8[4, 200, 14, 14]" = torch.ops.aten.le.Scalar(clone_4, 3)
    div_51: "f32[4, 200, 14, 14]" = torch.ops.aten.div.Tensor(clone_4, 3);  clone_4 = None
    add_180: "f32[4, 200, 14, 14]" = torch.ops.aten.add.Tensor(div_51, 0.5);  div_51 = None
    mul_382: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_96, add_180);  add_180 = None
    where_42: "f32[4, 200, 14, 14]" = torch.ops.aten.where.self(le_21, mul_382, getitem_96);  le_21 = mul_382 = getitem_96 = None
    where_43: "f32[4, 200, 14, 14]" = torch.ops.aten.where.self(lt_21, full_default, where_42);  lt_21 = where_42 = None
    add_181: "f32[200]" = torch.ops.aten.add.Tensor(primals_242, 0.001);  primals_242 = None
    rsqrt_23: "f32[200]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
    unsqueeze_644: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(primals_241, 0);  primals_241 = None
    unsqueeze_645: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, 2);  unsqueeze_644 = None
    unsqueeze_646: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_645, 3);  unsqueeze_645 = None
    sum_64: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_43, [0, 2, 3])
    sub_69: "f32[4, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_646);  convolution_28 = unsqueeze_646 = None
    mul_383: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, sub_69);  sub_69 = None
    sum_65: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_383, [0, 2, 3]);  mul_383 = None
    mul_388: "f32[200]" = torch.ops.aten.mul.Tensor(rsqrt_23, primals_80);  primals_80 = None
    unsqueeze_653: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_388, 0);  mul_388 = None
    unsqueeze_654: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_653, 2);  unsqueeze_653 = None
    unsqueeze_655: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, 3);  unsqueeze_654 = None
    mul_389: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_43, unsqueeze_655);  where_43 = unsqueeze_655 = None
    mul_390: "f32[200]" = torch.ops.aten.mul.Tensor(sum_65, rsqrt_23);  sum_65 = rsqrt_23 = None
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(mul_389, div_6, primals_79, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 200, [True, True, False]);  mul_389 = div_6 = primals_79 = None
    getitem_99: "f32[4, 200, 14, 14]" = convolution_backward_33[0]
    getitem_100: "f32[200, 1, 3, 3]" = convolution_backward_33[1];  convolution_backward_33 = None
    lt_22: "b8[4, 200, 14, 14]" = torch.ops.aten.lt.Scalar(clone_3, -3)
    le_22: "b8[4, 200, 14, 14]" = torch.ops.aten.le.Scalar(clone_3, 3)
    div_52: "f32[4, 200, 14, 14]" = torch.ops.aten.div.Tensor(clone_3, 3);  clone_3 = None
    add_182: "f32[4, 200, 14, 14]" = torch.ops.aten.add.Tensor(div_52, 0.5);  div_52 = None
    mul_391: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_99, add_182);  add_182 = None
    where_44: "f32[4, 200, 14, 14]" = torch.ops.aten.where.self(le_22, mul_391, getitem_99);  le_22 = mul_391 = getitem_99 = None
    where_45: "f32[4, 200, 14, 14]" = torch.ops.aten.where.self(lt_22, full_default, where_44);  lt_22 = where_44 = None
    add_183: "f32[200]" = torch.ops.aten.add.Tensor(primals_239, 0.001);  primals_239 = None
    rsqrt_24: "f32[200]" = torch.ops.aten.rsqrt.default(add_183);  add_183 = None
    unsqueeze_656: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(primals_238, 0);  primals_238 = None
    unsqueeze_657: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, 2);  unsqueeze_656 = None
    unsqueeze_658: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_657, 3);  unsqueeze_657 = None
    sum_66: "f32[200]" = torch.ops.aten.sum.dim_IntList(where_45, [0, 2, 3])
    sub_70: "f32[4, 200, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_658);  convolution_27 = unsqueeze_658 = None
    mul_392: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, sub_70);  sub_70 = None
    sum_67: "f32[200]" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 2, 3]);  mul_392 = None
    mul_397: "f32[200]" = torch.ops.aten.mul.Tensor(rsqrt_24, primals_77);  primals_77 = None
    unsqueeze_665: "f32[1, 200]" = torch.ops.aten.unsqueeze.default(mul_397, 0);  mul_397 = None
    unsqueeze_666: "f32[1, 200, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_665, 2);  unsqueeze_665 = None
    unsqueeze_667: "f32[1, 200, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, 3);  unsqueeze_666 = None
    mul_398: "f32[4, 200, 14, 14]" = torch.ops.aten.mul.Tensor(where_45, unsqueeze_667);  where_45 = unsqueeze_667 = None
    mul_399: "f32[200]" = torch.ops.aten.mul.Tensor(sum_67, rsqrt_24);  sum_67 = rsqrt_24 = None
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(mul_398, add_51, primals_76, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_398 = add_51 = primals_76 = None
    getitem_102: "f32[4, 80, 14, 14]" = convolution_backward_34[0]
    getitem_103: "f32[200, 80, 1, 1]" = convolution_backward_34[1];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_184: "f32[4, 80, 14, 14]" = torch.ops.aten.add.Tensor(add_178, getitem_102);  add_178 = getitem_102 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_185: "f32[80]" = torch.ops.aten.add.Tensor(primals_236, 0.001);  primals_236 = None
    rsqrt_25: "f32[80]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    unsqueeze_668: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(primals_235, 0);  primals_235 = None
    unsqueeze_669: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, 2);  unsqueeze_668 = None
    unsqueeze_670: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_669, 3);  unsqueeze_669 = None
    sum_68: "f32[80]" = torch.ops.aten.sum.dim_IntList(add_184, [0, 2, 3])
    sub_71: "f32[4, 80, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_670);  convolution_26 = unsqueeze_670 = None
    mul_400: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_184, sub_71);  sub_71 = None
    sum_69: "f32[80]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 2, 3]);  mul_400 = None
    mul_405: "f32[80]" = torch.ops.aten.mul.Tensor(rsqrt_25, primals_74);  primals_74 = None
    unsqueeze_677: "f32[1, 80]" = torch.ops.aten.unsqueeze.default(mul_405, 0);  mul_405 = None
    unsqueeze_678: "f32[1, 80, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_677, 2);  unsqueeze_677 = None
    unsqueeze_679: "f32[1, 80, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, 3);  unsqueeze_678 = None
    mul_406: "f32[4, 80, 14, 14]" = torch.ops.aten.mul.Tensor(add_184, unsqueeze_679);  add_184 = unsqueeze_679 = None
    mul_407: "f32[80]" = torch.ops.aten.mul.Tensor(sum_69, rsqrt_25);  sum_69 = rsqrt_25 = None
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(mul_406, div_5, primals_73, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_406 = div_5 = primals_73 = None
    getitem_105: "f32[4, 240, 14, 14]" = convolution_backward_35[0]
    getitem_106: "f32[80, 240, 1, 1]" = convolution_backward_35[1];  convolution_backward_35 = None
    lt_23: "b8[4, 240, 14, 14]" = torch.ops.aten.lt.Scalar(clone_2, -3)
    le_23: "b8[4, 240, 14, 14]" = torch.ops.aten.le.Scalar(clone_2, 3)
    div_53: "f32[4, 240, 14, 14]" = torch.ops.aten.div.Tensor(clone_2, 3);  clone_2 = None
    add_186: "f32[4, 240, 14, 14]" = torch.ops.aten.add.Tensor(div_53, 0.5);  div_53 = None
    mul_408: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(getitem_105, add_186);  add_186 = None
    where_46: "f32[4, 240, 14, 14]" = torch.ops.aten.where.self(le_23, mul_408, getitem_105);  le_23 = mul_408 = getitem_105 = None
    where_47: "f32[4, 240, 14, 14]" = torch.ops.aten.where.self(lt_23, full_default, where_46);  lt_23 = where_46 = None
    add_187: "f32[240]" = torch.ops.aten.add.Tensor(primals_233, 0.001);  primals_233 = None
    rsqrt_26: "f32[240]" = torch.ops.aten.rsqrt.default(add_187);  add_187 = None
    unsqueeze_680: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_232, 0);  primals_232 = None
    unsqueeze_681: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, 2);  unsqueeze_680 = None
    unsqueeze_682: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_681, 3);  unsqueeze_681 = None
    sum_70: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_47, [0, 2, 3])
    sub_72: "f32[4, 240, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_682);  convolution_25 = unsqueeze_682 = None
    mul_409: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, sub_72);  sub_72 = None
    sum_71: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_409, [0, 2, 3]);  mul_409 = None
    mul_414: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_26, primals_71);  primals_71 = None
    unsqueeze_689: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_414, 0);  mul_414 = None
    unsqueeze_690: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_689, 2);  unsqueeze_689 = None
    unsqueeze_691: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, 3);  unsqueeze_690 = None
    mul_415: "f32[4, 240, 14, 14]" = torch.ops.aten.mul.Tensor(where_47, unsqueeze_691);  where_47 = unsqueeze_691 = None
    mul_416: "f32[240]" = torch.ops.aten.mul.Tensor(sum_71, rsqrt_26);  sum_71 = rsqrt_26 = None
    convolution_backward_36 = torch.ops.aten.convolution_backward.default(mul_415, div_4, primals_70, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 240, [True, True, False]);  mul_415 = div_4 = primals_70 = None
    getitem_108: "f32[4, 240, 28, 28]" = convolution_backward_36[0]
    getitem_109: "f32[240, 1, 3, 3]" = convolution_backward_36[1];  convolution_backward_36 = None
    lt_24: "b8[4, 240, 28, 28]" = torch.ops.aten.lt.Scalar(clone_1, -3)
    le_24: "b8[4, 240, 28, 28]" = torch.ops.aten.le.Scalar(clone_1, 3)
    div_54: "f32[4, 240, 28, 28]" = torch.ops.aten.div.Tensor(clone_1, 3);  clone_1 = None
    add_188: "f32[4, 240, 28, 28]" = torch.ops.aten.add.Tensor(div_54, 0.5);  div_54 = None
    mul_417: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_108, add_188);  add_188 = None
    where_48: "f32[4, 240, 28, 28]" = torch.ops.aten.where.self(le_24, mul_417, getitem_108);  le_24 = mul_417 = getitem_108 = None
    where_49: "f32[4, 240, 28, 28]" = torch.ops.aten.where.self(lt_24, full_default, where_48);  lt_24 = where_48 = None
    add_189: "f32[240]" = torch.ops.aten.add.Tensor(primals_230, 0.001);  primals_230 = None
    rsqrt_27: "f32[240]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    unsqueeze_692: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(primals_229, 0);  primals_229 = None
    unsqueeze_693: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, 2);  unsqueeze_692 = None
    unsqueeze_694: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_693, 3);  unsqueeze_693 = None
    sum_72: "f32[240]" = torch.ops.aten.sum.dim_IntList(where_49, [0, 2, 3])
    sub_73: "f32[4, 240, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_694);  convolution_24 = unsqueeze_694 = None
    mul_418: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(where_49, sub_73);  sub_73 = None
    sum_73: "f32[240]" = torch.ops.aten.sum.dim_IntList(mul_418, [0, 2, 3]);  mul_418 = None
    mul_423: "f32[240]" = torch.ops.aten.mul.Tensor(rsqrt_27, primals_68);  primals_68 = None
    unsqueeze_701: "f32[1, 240]" = torch.ops.aten.unsqueeze.default(mul_423, 0);  mul_423 = None
    unsqueeze_702: "f32[1, 240, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_701, 2);  unsqueeze_701 = None
    unsqueeze_703: "f32[1, 240, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, 3);  unsqueeze_702 = None
    mul_424: "f32[4, 240, 28, 28]" = torch.ops.aten.mul.Tensor(where_49, unsqueeze_703);  where_49 = unsqueeze_703 = None
    mul_425: "f32[240]" = torch.ops.aten.mul.Tensor(sum_73, rsqrt_27);  sum_73 = rsqrt_27 = None
    convolution_backward_37 = torch.ops.aten.convolution_backward.default(mul_424, add_43, primals_67, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_424 = add_43 = primals_67 = None
    getitem_111: "f32[4, 40, 28, 28]" = convolution_backward_37[0]
    getitem_112: "f32[240, 40, 1, 1]" = convolution_backward_37[1];  convolution_backward_37 = None
    add_190: "f32[40]" = torch.ops.aten.add.Tensor(primals_227, 0.001);  primals_227 = None
    rsqrt_28: "f32[40]" = torch.ops.aten.rsqrt.default(add_190);  add_190 = None
    unsqueeze_704: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_226, 0);  primals_226 = None
    unsqueeze_705: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, 2);  unsqueeze_704 = None
    unsqueeze_706: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_705, 3);  unsqueeze_705 = None
    sum_74: "f32[40]" = torch.ops.aten.sum.dim_IntList(getitem_111, [0, 2, 3])
    sub_74: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_706);  convolution_23 = unsqueeze_706 = None
    mul_426: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_111, sub_74);  sub_74 = None
    sum_75: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 2, 3]);  mul_426 = None
    mul_431: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_28, primals_65);  primals_65 = None
    unsqueeze_713: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_431, 0);  mul_431 = None
    unsqueeze_714: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_713, 2);  unsqueeze_713 = None
    unsqueeze_715: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, 3);  unsqueeze_714 = None
    mul_432: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_111, unsqueeze_715);  unsqueeze_715 = None
    mul_433: "f32[40]" = torch.ops.aten.mul.Tensor(sum_75, rsqrt_28);  sum_75 = rsqrt_28 = None
    convolution_backward_38 = torch.ops.aten.convolution_backward.default(mul_432, mul_54, primals_64, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_432 = mul_54 = primals_64 = None
    getitem_114: "f32[4, 120, 28, 28]" = convolution_backward_38[0]
    getitem_115: "f32[40, 120, 1, 1]" = convolution_backward_38[1];  convolution_backward_38 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_434: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_114, div_3);  div_3 = None
    mul_435: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_114, relu_12);  getitem_114 = None
    sum_76: "f32[4, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_435, [2, 3], True);  mul_435 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    mul_436: "f32[4, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_76, 0.16666666666666666);  sum_76 = None
    where_50: "f32[4, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_5, mul_436, full_default);  bitwise_and_5 = mul_436 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_77: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_50, [0, 2, 3])
    convolution_backward_39 = torch.ops.aten.convolution_backward.default(where_50, relu_13, primals_62, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_50 = primals_62 = None
    getitem_117: "f32[4, 32, 1, 1]" = convolution_backward_39[0]
    getitem_118: "f32[120, 32, 1, 1]" = convolution_backward_39[1];  convolution_backward_39 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    le_25: "b8[4, 32, 1, 1]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
    where_51: "f32[4, 32, 1, 1]" = torch.ops.aten.where.self(le_25, full_default, getitem_117);  le_25 = getitem_117 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_78: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_51, [0, 2, 3])
    convolution_backward_40 = torch.ops.aten.convolution_backward.default(where_51, mean_2, primals_60, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_51 = mean_2 = primals_60 = None
    getitem_120: "f32[4, 120, 1, 1]" = convolution_backward_40[0]
    getitem_121: "f32[32, 120, 1, 1]" = convolution_backward_40[1];  convolution_backward_40 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_6: "f32[4, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_120, [4, 120, 28, 28]);  getitem_120 = None
    div_55: "f32[4, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_6, 784);  expand_6 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_191: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_434, div_55);  mul_434 = div_55 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    le_26: "b8[4, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
    where_52: "f32[4, 120, 28, 28]" = torch.ops.aten.where.self(le_26, full_default, add_191);  le_26 = add_191 = None
    add_192: "f32[120]" = torch.ops.aten.add.Tensor(primals_224, 0.001);  primals_224 = None
    rsqrt_29: "f32[120]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    unsqueeze_716: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(primals_223, 0);  primals_223 = None
    unsqueeze_717: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, 2);  unsqueeze_716 = None
    unsqueeze_718: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_717, 3);  unsqueeze_717 = None
    sum_79: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_52, [0, 2, 3])
    sub_75: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_718);  convolution_20 = unsqueeze_718 = None
    mul_437: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, sub_75);  sub_75 = None
    sum_80: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 2, 3]);  mul_437 = None
    mul_442: "f32[120]" = torch.ops.aten.mul.Tensor(rsqrt_29, primals_58);  primals_58 = None
    unsqueeze_725: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_442, 0);  mul_442 = None
    unsqueeze_726: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_725, 2);  unsqueeze_725 = None
    unsqueeze_727: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, 3);  unsqueeze_726 = None
    mul_443: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_52, unsqueeze_727);  where_52 = unsqueeze_727 = None
    mul_444: "f32[120]" = torch.ops.aten.mul.Tensor(sum_80, rsqrt_29);  sum_80 = rsqrt_29 = None
    convolution_backward_41 = torch.ops.aten.convolution_backward.default(mul_443, relu_11, primals_57, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_443 = primals_57 = None
    getitem_123: "f32[4, 120, 28, 28]" = convolution_backward_41[0]
    getitem_124: "f32[120, 1, 5, 5]" = convolution_backward_41[1];  convolution_backward_41 = None
    le_27: "b8[4, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
    where_53: "f32[4, 120, 28, 28]" = torch.ops.aten.where.self(le_27, full_default, getitem_123);  le_27 = getitem_123 = None
    add_193: "f32[120]" = torch.ops.aten.add.Tensor(primals_221, 0.001);  primals_221 = None
    rsqrt_30: "f32[120]" = torch.ops.aten.rsqrt.default(add_193);  add_193 = None
    unsqueeze_728: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(primals_220, 0);  primals_220 = None
    unsqueeze_729: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, 2);  unsqueeze_728 = None
    unsqueeze_730: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_729, 3);  unsqueeze_729 = None
    sum_81: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_53, [0, 2, 3])
    sub_76: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_730);  convolution_19 = unsqueeze_730 = None
    mul_445: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_53, sub_76);  sub_76 = None
    sum_82: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 2, 3]);  mul_445 = None
    mul_450: "f32[120]" = torch.ops.aten.mul.Tensor(rsqrt_30, primals_55);  primals_55 = None
    unsqueeze_737: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_450, 0);  mul_450 = None
    unsqueeze_738: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_737, 2);  unsqueeze_737 = None
    unsqueeze_739: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, 3);  unsqueeze_738 = None
    mul_451: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_53, unsqueeze_739);  where_53 = unsqueeze_739 = None
    mul_452: "f32[120]" = torch.ops.aten.mul.Tensor(sum_82, rsqrt_30);  sum_82 = rsqrt_30 = None
    convolution_backward_42 = torch.ops.aten.convolution_backward.default(mul_451, add_35, primals_54, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_451 = add_35 = primals_54 = None
    getitem_126: "f32[4, 40, 28, 28]" = convolution_backward_42[0]
    getitem_127: "f32[120, 40, 1, 1]" = convolution_backward_42[1];  convolution_backward_42 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_194: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(getitem_111, getitem_126);  getitem_111 = getitem_126 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_195: "f32[40]" = torch.ops.aten.add.Tensor(primals_218, 0.001);  primals_218 = None
    rsqrt_31: "f32[40]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
    unsqueeze_740: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_217, 0);  primals_217 = None
    unsqueeze_741: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, 2);  unsqueeze_740 = None
    unsqueeze_742: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_741, 3);  unsqueeze_741 = None
    sum_83: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_194, [0, 2, 3])
    sub_77: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_742);  convolution_18 = unsqueeze_742 = None
    mul_453: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_194, sub_77);  sub_77 = None
    sum_84: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_453, [0, 2, 3]);  mul_453 = None
    mul_458: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_31, primals_52);  primals_52 = None
    unsqueeze_749: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_458, 0);  mul_458 = None
    unsqueeze_750: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_749, 2);  unsqueeze_749 = None
    unsqueeze_751: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, 3);  unsqueeze_750 = None
    mul_459: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_194, unsqueeze_751);  unsqueeze_751 = None
    mul_460: "f32[40]" = torch.ops.aten.mul.Tensor(sum_84, rsqrt_31);  sum_84 = rsqrt_31 = None
    convolution_backward_43 = torch.ops.aten.convolution_backward.default(mul_459, mul_44, primals_51, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_459 = mul_44 = primals_51 = None
    getitem_129: "f32[4, 120, 28, 28]" = convolution_backward_43[0]
    getitem_130: "f32[40, 120, 1, 1]" = convolution_backward_43[1];  convolution_backward_43 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_461: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_129, div_2);  div_2 = None
    mul_462: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_129, relu_9);  getitem_129 = None
    sum_85: "f32[4, 120, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [2, 3], True);  mul_462 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    mul_463: "f32[4, 120, 1, 1]" = torch.ops.aten.mul.Tensor(sum_85, 0.16666666666666666);  sum_85 = None
    where_54: "f32[4, 120, 1, 1]" = torch.ops.aten.where.self(bitwise_and_6, mul_463, full_default);  bitwise_and_6 = mul_463 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_86: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_54, [0, 2, 3])
    convolution_backward_44 = torch.ops.aten.convolution_backward.default(where_54, relu_10, primals_49, [120], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_54 = primals_49 = None
    getitem_132: "f32[4, 32, 1, 1]" = convolution_backward_44[0]
    getitem_133: "f32[120, 32, 1, 1]" = convolution_backward_44[1];  convolution_backward_44 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    le_28: "b8[4, 32, 1, 1]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
    where_55: "f32[4, 32, 1, 1]" = torch.ops.aten.where.self(le_28, full_default, getitem_132);  le_28 = getitem_132 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_87: "f32[32]" = torch.ops.aten.sum.dim_IntList(where_55, [0, 2, 3])
    convolution_backward_45 = torch.ops.aten.convolution_backward.default(where_55, mean_1, primals_47, [32], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_55 = mean_1 = primals_47 = None
    getitem_135: "f32[4, 120, 1, 1]" = convolution_backward_45[0]
    getitem_136: "f32[32, 120, 1, 1]" = convolution_backward_45[1];  convolution_backward_45 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_7: "f32[4, 120, 28, 28]" = torch.ops.aten.expand.default(getitem_135, [4, 120, 28, 28]);  getitem_135 = None
    div_56: "f32[4, 120, 28, 28]" = torch.ops.aten.div.Scalar(expand_7, 784);  expand_7 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_196: "f32[4, 120, 28, 28]" = torch.ops.aten.add.Tensor(mul_461, div_56);  mul_461 = div_56 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    le_29: "b8[4, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
    where_56: "f32[4, 120, 28, 28]" = torch.ops.aten.where.self(le_29, full_default, add_196);  le_29 = add_196 = None
    add_197: "f32[120]" = torch.ops.aten.add.Tensor(primals_215, 0.001);  primals_215 = None
    rsqrt_32: "f32[120]" = torch.ops.aten.rsqrt.default(add_197);  add_197 = None
    unsqueeze_752: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(primals_214, 0);  primals_214 = None
    unsqueeze_753: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, 2);  unsqueeze_752 = None
    unsqueeze_754: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_753, 3);  unsqueeze_753 = None
    sum_88: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_56, [0, 2, 3])
    sub_78: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_754);  convolution_15 = unsqueeze_754 = None
    mul_464: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_56, sub_78);  sub_78 = None
    sum_89: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 2, 3]);  mul_464 = None
    mul_469: "f32[120]" = torch.ops.aten.mul.Tensor(rsqrt_32, primals_45);  primals_45 = None
    unsqueeze_761: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_469, 0);  mul_469 = None
    unsqueeze_762: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_761, 2);  unsqueeze_761 = None
    unsqueeze_763: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, 3);  unsqueeze_762 = None
    mul_470: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_56, unsqueeze_763);  where_56 = unsqueeze_763 = None
    mul_471: "f32[120]" = torch.ops.aten.mul.Tensor(sum_89, rsqrt_32);  sum_89 = rsqrt_32 = None
    convolution_backward_46 = torch.ops.aten.convolution_backward.default(mul_470, relu_8, primals_44, [0], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, False]);  mul_470 = primals_44 = None
    getitem_138: "f32[4, 120, 28, 28]" = convolution_backward_46[0]
    getitem_139: "f32[120, 1, 5, 5]" = convolution_backward_46[1];  convolution_backward_46 = None
    le_30: "b8[4, 120, 28, 28]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
    where_57: "f32[4, 120, 28, 28]" = torch.ops.aten.where.self(le_30, full_default, getitem_138);  le_30 = getitem_138 = None
    add_198: "f32[120]" = torch.ops.aten.add.Tensor(primals_212, 0.001);  primals_212 = None
    rsqrt_33: "f32[120]" = torch.ops.aten.rsqrt.default(add_198);  add_198 = None
    unsqueeze_764: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(primals_211, 0);  primals_211 = None
    unsqueeze_765: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, 2);  unsqueeze_764 = None
    unsqueeze_766: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_765, 3);  unsqueeze_765 = None
    sum_90: "f32[120]" = torch.ops.aten.sum.dim_IntList(where_57, [0, 2, 3])
    sub_79: "f32[4, 120, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_766);  convolution_14 = unsqueeze_766 = None
    mul_472: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_57, sub_79);  sub_79 = None
    sum_91: "f32[120]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 2, 3]);  mul_472 = None
    mul_477: "f32[120]" = torch.ops.aten.mul.Tensor(rsqrt_33, primals_42);  primals_42 = None
    unsqueeze_773: "f32[1, 120]" = torch.ops.aten.unsqueeze.default(mul_477, 0);  mul_477 = None
    unsqueeze_774: "f32[1, 120, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_773, 2);  unsqueeze_773 = None
    unsqueeze_775: "f32[1, 120, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, 3);  unsqueeze_774 = None
    mul_478: "f32[4, 120, 28, 28]" = torch.ops.aten.mul.Tensor(where_57, unsqueeze_775);  where_57 = unsqueeze_775 = None
    mul_479: "f32[120]" = torch.ops.aten.mul.Tensor(sum_91, rsqrt_33);  sum_91 = rsqrt_33 = None
    convolution_backward_47 = torch.ops.aten.convolution_backward.default(mul_478, add_27, primals_41, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_478 = add_27 = primals_41 = None
    getitem_141: "f32[4, 40, 28, 28]" = convolution_backward_47[0]
    getitem_142: "f32[120, 40, 1, 1]" = convolution_backward_47[1];  convolution_backward_47 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_199: "f32[4, 40, 28, 28]" = torch.ops.aten.add.Tensor(add_194, getitem_141);  add_194 = getitem_141 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_200: "f32[40]" = torch.ops.aten.add.Tensor(primals_209, 0.001);  primals_209 = None
    rsqrt_34: "f32[40]" = torch.ops.aten.rsqrt.default(add_200);  add_200 = None
    unsqueeze_776: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(primals_208, 0);  primals_208 = None
    unsqueeze_777: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, 2);  unsqueeze_776 = None
    unsqueeze_778: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_777, 3);  unsqueeze_777 = None
    sum_92: "f32[40]" = torch.ops.aten.sum.dim_IntList(add_199, [0, 2, 3])
    sub_80: "f32[4, 40, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_778);  convolution_13 = unsqueeze_778 = None
    mul_480: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_199, sub_80);  sub_80 = None
    sum_93: "f32[40]" = torch.ops.aten.sum.dim_IntList(mul_480, [0, 2, 3]);  mul_480 = None
    mul_485: "f32[40]" = torch.ops.aten.mul.Tensor(rsqrt_34, primals_39);  primals_39 = None
    unsqueeze_785: "f32[1, 40]" = torch.ops.aten.unsqueeze.default(mul_485, 0);  mul_485 = None
    unsqueeze_786: "f32[1, 40, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_785, 2);  unsqueeze_785 = None
    unsqueeze_787: "f32[1, 40, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, 3);  unsqueeze_786 = None
    mul_486: "f32[4, 40, 28, 28]" = torch.ops.aten.mul.Tensor(add_199, unsqueeze_787);  add_199 = unsqueeze_787 = None
    mul_487: "f32[40]" = torch.ops.aten.mul.Tensor(sum_93, rsqrt_34);  sum_93 = rsqrt_34 = None
    convolution_backward_48 = torch.ops.aten.convolution_backward.default(mul_486, mul_34, primals_38, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_486 = mul_34 = primals_38 = None
    getitem_144: "f32[4, 72, 28, 28]" = convolution_backward_48[0]
    getitem_145: "f32[40, 72, 1, 1]" = convolution_backward_48[1];  convolution_backward_48 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:260, code: return scale * input
    mul_488: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_144, div_1);  div_1 = None
    mul_489: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(getitem_144, relu_6);  getitem_144 = None
    sum_94: "f32[4, 72, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_489, [2, 3], True);  mul_489 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:256, code: return self.scale_activation(scale)
    mul_490: "f32[4, 72, 1, 1]" = torch.ops.aten.mul.Tensor(sum_94, 0.16666666666666666);  sum_94 = None
    where_58: "f32[4, 72, 1, 1]" = torch.ops.aten.where.self(bitwise_and_7, mul_490, full_default);  bitwise_and_7 = mul_490 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:255, code: scale = self.fc2(scale)
    sum_95: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_58, [0, 2, 3])
    convolution_backward_49 = torch.ops.aten.convolution_backward.default(where_58, relu_7, primals_36, [72], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_58 = primals_36 = None
    getitem_147: "f32[4, 24, 1, 1]" = convolution_backward_49[0]
    getitem_148: "f32[72, 24, 1, 1]" = convolution_backward_49[1];  convolution_backward_49 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:254, code: scale = self.activation(scale)
    le_31: "b8[4, 24, 1, 1]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
    where_59: "f32[4, 24, 1, 1]" = torch.ops.aten.where.self(le_31, full_default, getitem_147);  le_31 = getitem_147 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:253, code: scale = self.fc1(scale)
    sum_96: "f32[24]" = torch.ops.aten.sum.dim_IntList(where_59, [0, 2, 3])
    convolution_backward_50 = torch.ops.aten.convolution_backward.default(where_59, mean, primals_34, [24], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  where_59 = mean = primals_34 = None
    getitem_150: "f32[4, 72, 1, 1]" = convolution_backward_50[0]
    getitem_151: "f32[24, 72, 1, 1]" = convolution_backward_50[1];  convolution_backward_50 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    expand_8: "f32[4, 72, 28, 28]" = torch.ops.aten.expand.default(getitem_150, [4, 72, 28, 28]);  getitem_150 = None
    div_57: "f32[4, 72, 28, 28]" = torch.ops.aten.div.Scalar(expand_8, 784);  expand_8 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/ops/misc.py:252, code: scale = self.avgpool(input)
    add_201: "f32[4, 72, 28, 28]" = torch.ops.aten.add.Tensor(mul_488, div_57);  mul_488 = div_57 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    le_32: "b8[4, 72, 28, 28]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
    where_60: "f32[4, 72, 28, 28]" = torch.ops.aten.where.self(le_32, full_default, add_201);  le_32 = add_201 = None
    add_202: "f32[72]" = torch.ops.aten.add.Tensor(primals_206, 0.001);  primals_206 = None
    rsqrt_35: "f32[72]" = torch.ops.aten.rsqrt.default(add_202);  add_202 = None
    unsqueeze_788: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_205, 0);  primals_205 = None
    unsqueeze_789: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, 2);  unsqueeze_788 = None
    unsqueeze_790: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_789, 3);  unsqueeze_789 = None
    sum_97: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_60, [0, 2, 3])
    sub_81: "f32[4, 72, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_790);  convolution_10 = unsqueeze_790 = None
    mul_491: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, sub_81);  sub_81 = None
    sum_98: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_491, [0, 2, 3]);  mul_491 = None
    mul_496: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_35, primals_32);  primals_32 = None
    unsqueeze_797: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_496, 0);  mul_496 = None
    unsqueeze_798: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_797, 2);  unsqueeze_797 = None
    unsqueeze_799: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, 3);  unsqueeze_798 = None
    mul_497: "f32[4, 72, 28, 28]" = torch.ops.aten.mul.Tensor(where_60, unsqueeze_799);  where_60 = unsqueeze_799 = None
    mul_498: "f32[72]" = torch.ops.aten.mul.Tensor(sum_98, rsqrt_35);  sum_98 = rsqrt_35 = None
    convolution_backward_51 = torch.ops.aten.convolution_backward.default(mul_497, relu_5, primals_31, [0], [2, 2], [2, 2], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_497 = primals_31 = None
    getitem_153: "f32[4, 72, 56, 56]" = convolution_backward_51[0]
    getitem_154: "f32[72, 1, 5, 5]" = convolution_backward_51[1];  convolution_backward_51 = None
    le_33: "b8[4, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
    where_61: "f32[4, 72, 56, 56]" = torch.ops.aten.where.self(le_33, full_default, getitem_153);  le_33 = getitem_153 = None
    add_203: "f32[72]" = torch.ops.aten.add.Tensor(primals_203, 0.001);  primals_203 = None
    rsqrt_36: "f32[72]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
    unsqueeze_800: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_202, 0);  primals_202 = None
    unsqueeze_801: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, 2);  unsqueeze_800 = None
    unsqueeze_802: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_801, 3);  unsqueeze_801 = None
    sum_99: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_61, [0, 2, 3])
    sub_82: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_802);  convolution_9 = unsqueeze_802 = None
    mul_499: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_61, sub_82);  sub_82 = None
    sum_100: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_499, [0, 2, 3]);  mul_499 = None
    mul_504: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_36, primals_29);  primals_29 = None
    unsqueeze_809: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_504, 0);  mul_504 = None
    unsqueeze_810: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_809, 2);  unsqueeze_809 = None
    unsqueeze_811: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, 3);  unsqueeze_810 = None
    mul_505: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_61, unsqueeze_811);  where_61 = unsqueeze_811 = None
    mul_506: "f32[72]" = torch.ops.aten.mul.Tensor(sum_100, rsqrt_36);  sum_100 = rsqrt_36 = None
    convolution_backward_52 = torch.ops.aten.convolution_backward.default(mul_505, add_20, primals_28, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_505 = add_20 = primals_28 = None
    getitem_156: "f32[4, 24, 56, 56]" = convolution_backward_52[0]
    getitem_157: "f32[72, 24, 1, 1]" = convolution_backward_52[1];  convolution_backward_52 = None
    add_204: "f32[24]" = torch.ops.aten.add.Tensor(primals_200, 0.001);  primals_200 = None
    rsqrt_37: "f32[24]" = torch.ops.aten.rsqrt.default(add_204);  add_204 = None
    unsqueeze_812: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_199, 0);  primals_199 = None
    unsqueeze_813: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, 2);  unsqueeze_812 = None
    unsqueeze_814: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_813, 3);  unsqueeze_813 = None
    sum_101: "f32[24]" = torch.ops.aten.sum.dim_IntList(getitem_156, [0, 2, 3])
    sub_83: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_814);  convolution_8 = unsqueeze_814 = None
    mul_507: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_156, sub_83);  sub_83 = None
    sum_102: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_507, [0, 2, 3]);  mul_507 = None
    mul_512: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_37, primals_26);  primals_26 = None
    unsqueeze_821: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_512, 0);  mul_512 = None
    unsqueeze_822: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_821, 2);  unsqueeze_821 = None
    unsqueeze_823: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, 3);  unsqueeze_822 = None
    mul_513: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(getitem_156, unsqueeze_823);  unsqueeze_823 = None
    mul_514: "f32[24]" = torch.ops.aten.mul.Tensor(sum_102, rsqrt_37);  sum_102 = rsqrt_37 = None
    convolution_backward_53 = torch.ops.aten.convolution_backward.default(mul_513, relu_4, primals_25, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_513 = primals_25 = None
    getitem_159: "f32[4, 72, 56, 56]" = convolution_backward_53[0]
    getitem_160: "f32[24, 72, 1, 1]" = convolution_backward_53[1];  convolution_backward_53 = None
    le_34: "b8[4, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
    where_62: "f32[4, 72, 56, 56]" = torch.ops.aten.where.self(le_34, full_default, getitem_159);  le_34 = getitem_159 = None
    add_205: "f32[72]" = torch.ops.aten.add.Tensor(primals_197, 0.001);  primals_197 = None
    rsqrt_38: "f32[72]" = torch.ops.aten.rsqrt.default(add_205);  add_205 = None
    unsqueeze_824: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_196, 0);  primals_196 = None
    unsqueeze_825: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, 2);  unsqueeze_824 = None
    unsqueeze_826: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_825, 3);  unsqueeze_825 = None
    sum_103: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_62, [0, 2, 3])
    sub_84: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_826);  convolution_7 = unsqueeze_826 = None
    mul_515: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_62, sub_84);  sub_84 = None
    sum_104: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_515, [0, 2, 3]);  mul_515 = None
    mul_520: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_38, primals_23);  primals_23 = None
    unsqueeze_833: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_520, 0);  mul_520 = None
    unsqueeze_834: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_833, 2);  unsqueeze_833 = None
    unsqueeze_835: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, 3);  unsqueeze_834 = None
    mul_521: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_62, unsqueeze_835);  where_62 = unsqueeze_835 = None
    mul_522: "f32[72]" = torch.ops.aten.mul.Tensor(sum_104, rsqrt_38);  sum_104 = rsqrt_38 = None
    convolution_backward_54 = torch.ops.aten.convolution_backward.default(mul_521, relu_3, primals_22, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 72, [True, True, False]);  mul_521 = primals_22 = None
    getitem_162: "f32[4, 72, 56, 56]" = convolution_backward_54[0]
    getitem_163: "f32[72, 1, 3, 3]" = convolution_backward_54[1];  convolution_backward_54 = None
    le_35: "b8[4, 72, 56, 56]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
    where_63: "f32[4, 72, 56, 56]" = torch.ops.aten.where.self(le_35, full_default, getitem_162);  le_35 = getitem_162 = None
    add_206: "f32[72]" = torch.ops.aten.add.Tensor(primals_194, 0.001);  primals_194 = None
    rsqrt_39: "f32[72]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
    unsqueeze_836: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(primals_193, 0);  primals_193 = None
    unsqueeze_837: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, 2);  unsqueeze_836 = None
    unsqueeze_838: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_837, 3);  unsqueeze_837 = None
    sum_105: "f32[72]" = torch.ops.aten.sum.dim_IntList(where_63, [0, 2, 3])
    sub_85: "f32[4, 72, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_838);  convolution_6 = unsqueeze_838 = None
    mul_523: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_63, sub_85);  sub_85 = None
    sum_106: "f32[72]" = torch.ops.aten.sum.dim_IntList(mul_523, [0, 2, 3]);  mul_523 = None
    mul_528: "f32[72]" = torch.ops.aten.mul.Tensor(rsqrt_39, primals_20);  primals_20 = None
    unsqueeze_845: "f32[1, 72]" = torch.ops.aten.unsqueeze.default(mul_528, 0);  mul_528 = None
    unsqueeze_846: "f32[1, 72, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_845, 2);  unsqueeze_845 = None
    unsqueeze_847: "f32[1, 72, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, 3);  unsqueeze_846 = None
    mul_529: "f32[4, 72, 56, 56]" = torch.ops.aten.mul.Tensor(where_63, unsqueeze_847);  where_63 = unsqueeze_847 = None
    mul_530: "f32[72]" = torch.ops.aten.mul.Tensor(sum_106, rsqrt_39);  sum_106 = rsqrt_39 = None
    convolution_backward_55 = torch.ops.aten.convolution_backward.default(mul_529, add_13, primals_19, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_529 = add_13 = primals_19 = None
    getitem_165: "f32[4, 24, 56, 56]" = convolution_backward_55[0]
    getitem_166: "f32[72, 24, 1, 1]" = convolution_backward_55[1];  convolution_backward_55 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_207: "f32[4, 24, 56, 56]" = torch.ops.aten.add.Tensor(getitem_156, getitem_165);  getitem_156 = getitem_165 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_208: "f32[24]" = torch.ops.aten.add.Tensor(primals_191, 0.001);  primals_191 = None
    rsqrt_40: "f32[24]" = torch.ops.aten.rsqrt.default(add_208);  add_208 = None
    unsqueeze_848: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(primals_190, 0);  primals_190 = None
    unsqueeze_849: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, 2);  unsqueeze_848 = None
    unsqueeze_850: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_849, 3);  unsqueeze_849 = None
    sum_107: "f32[24]" = torch.ops.aten.sum.dim_IntList(add_207, [0, 2, 3])
    sub_86: "f32[4, 24, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_850);  convolution_5 = unsqueeze_850 = None
    mul_531: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_207, sub_86);  sub_86 = None
    sum_108: "f32[24]" = torch.ops.aten.sum.dim_IntList(mul_531, [0, 2, 3]);  mul_531 = None
    mul_536: "f32[24]" = torch.ops.aten.mul.Tensor(rsqrt_40, primals_17);  primals_17 = None
    unsqueeze_857: "f32[1, 24]" = torch.ops.aten.unsqueeze.default(mul_536, 0);  mul_536 = None
    unsqueeze_858: "f32[1, 24, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_857, 2);  unsqueeze_857 = None
    unsqueeze_859: "f32[1, 24, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, 3);  unsqueeze_858 = None
    mul_537: "f32[4, 24, 56, 56]" = torch.ops.aten.mul.Tensor(add_207, unsqueeze_859);  add_207 = unsqueeze_859 = None
    mul_538: "f32[24]" = torch.ops.aten.mul.Tensor(sum_108, rsqrt_40);  sum_108 = rsqrt_40 = None
    convolution_backward_56 = torch.ops.aten.convolution_backward.default(mul_537, relu_2, primals_16, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_537 = primals_16 = None
    getitem_168: "f32[4, 64, 56, 56]" = convolution_backward_56[0]
    getitem_169: "f32[24, 64, 1, 1]" = convolution_backward_56[1];  convolution_backward_56 = None
    le_36: "b8[4, 64, 56, 56]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
    where_64: "f32[4, 64, 56, 56]" = torch.ops.aten.where.self(le_36, full_default, getitem_168);  le_36 = getitem_168 = None
    add_209: "f32[64]" = torch.ops.aten.add.Tensor(primals_188, 0.001);  primals_188 = None
    rsqrt_41: "f32[64]" = torch.ops.aten.rsqrt.default(add_209);  add_209 = None
    unsqueeze_860: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_187, 0);  primals_187 = None
    unsqueeze_861: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, 2);  unsqueeze_860 = None
    unsqueeze_862: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_861, 3);  unsqueeze_861 = None
    sum_109: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_64, [0, 2, 3])
    sub_87: "f32[4, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_862);  convolution_4 = unsqueeze_862 = None
    mul_539: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_64, sub_87);  sub_87 = None
    sum_110: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 2, 3]);  mul_539 = None
    mul_544: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_41, primals_14);  primals_14 = None
    unsqueeze_869: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_544, 0);  mul_544 = None
    unsqueeze_870: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_869, 2);  unsqueeze_869 = None
    unsqueeze_871: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, 3);  unsqueeze_870 = None
    mul_545: "f32[4, 64, 56, 56]" = torch.ops.aten.mul.Tensor(where_64, unsqueeze_871);  where_64 = unsqueeze_871 = None
    mul_546: "f32[64]" = torch.ops.aten.mul.Tensor(sum_110, rsqrt_41);  sum_110 = rsqrt_41 = None
    convolution_backward_57 = torch.ops.aten.convolution_backward.default(mul_545, relu_1, primals_13, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 64, [True, True, False]);  mul_545 = primals_13 = None
    getitem_171: "f32[4, 64, 112, 112]" = convolution_backward_57[0]
    getitem_172: "f32[64, 1, 3, 3]" = convolution_backward_57[1];  convolution_backward_57 = None
    le_37: "b8[4, 64, 112, 112]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
    where_65: "f32[4, 64, 112, 112]" = torch.ops.aten.where.self(le_37, full_default, getitem_171);  le_37 = getitem_171 = None
    add_210: "f32[64]" = torch.ops.aten.add.Tensor(primals_185, 0.001);  primals_185 = None
    rsqrt_42: "f32[64]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
    unsqueeze_872: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(primals_184, 0);  primals_184 = None
    unsqueeze_873: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, 2);  unsqueeze_872 = None
    unsqueeze_874: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_873, 3);  unsqueeze_873 = None
    sum_111: "f32[64]" = torch.ops.aten.sum.dim_IntList(where_65, [0, 2, 3])
    sub_88: "f32[4, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_874);  convolution_3 = unsqueeze_874 = None
    mul_547: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_65, sub_88);  sub_88 = None
    sum_112: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 2, 3]);  mul_547 = None
    mul_552: "f32[64]" = torch.ops.aten.mul.Tensor(rsqrt_42, primals_11);  primals_11 = None
    unsqueeze_881: "f32[1, 64]" = torch.ops.aten.unsqueeze.default(mul_552, 0);  mul_552 = None
    unsqueeze_882: "f32[1, 64, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_881, 2);  unsqueeze_881 = None
    unsqueeze_883: "f32[1, 64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, 3);  unsqueeze_882 = None
    mul_553: "f32[4, 64, 112, 112]" = torch.ops.aten.mul.Tensor(where_65, unsqueeze_883);  where_65 = unsqueeze_883 = None
    mul_554: "f32[64]" = torch.ops.aten.mul.Tensor(sum_112, rsqrt_42);  sum_112 = rsqrt_42 = None
    convolution_backward_58 = torch.ops.aten.convolution_backward.default(mul_553, add_7, primals_10, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_553 = add_7 = primals_10 = None
    getitem_174: "f32[4, 16, 112, 112]" = convolution_backward_58[0]
    getitem_175: "f32[64, 16, 1, 1]" = convolution_backward_58[1];  convolution_backward_58 = None
    add_211: "f32[16]" = torch.ops.aten.add.Tensor(primals_182, 0.001);  primals_182 = None
    rsqrt_43: "f32[16]" = torch.ops.aten.rsqrt.default(add_211);  add_211 = None
    unsqueeze_884: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(primals_181, 0);  primals_181 = None
    unsqueeze_885: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, 2);  unsqueeze_884 = None
    unsqueeze_886: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_885, 3);  unsqueeze_885 = None
    sum_113: "f32[16]" = torch.ops.aten.sum.dim_IntList(getitem_174, [0, 2, 3])
    sub_89: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_886);  convolution_2 = unsqueeze_886 = None
    mul_555: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_174, sub_89);  sub_89 = None
    sum_114: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_555, [0, 2, 3]);  mul_555 = None
    mul_560: "f32[16]" = torch.ops.aten.mul.Tensor(rsqrt_43, primals_8);  primals_8 = None
    unsqueeze_893: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_560, 0);  mul_560 = None
    unsqueeze_894: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_893, 2);  unsqueeze_893 = None
    unsqueeze_895: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, 3);  unsqueeze_894 = None
    mul_561: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(getitem_174, unsqueeze_895);  unsqueeze_895 = None
    mul_562: "f32[16]" = torch.ops.aten.mul.Tensor(sum_114, rsqrt_43);  sum_114 = rsqrt_43 = None
    convolution_backward_59 = torch.ops.aten.convolution_backward.default(mul_561, relu, primals_7, [0], [1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, False]);  mul_561 = primals_7 = None
    getitem_177: "f32[4, 16, 112, 112]" = convolution_backward_59[0]
    getitem_178: "f32[16, 16, 1, 1]" = convolution_backward_59[1];  convolution_backward_59 = None
    le_38: "b8[4, 16, 112, 112]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
    where_66: "f32[4, 16, 112, 112]" = torch.ops.aten.where.self(le_38, full_default, getitem_177);  le_38 = getitem_177 = None
    add_212: "f32[16]" = torch.ops.aten.add.Tensor(primals_179, 0.001);  primals_179 = None
    rsqrt_44: "f32[16]" = torch.ops.aten.rsqrt.default(add_212);  add_212 = None
    unsqueeze_896: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(primals_178, 0);  primals_178 = None
    unsqueeze_897: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, 2);  unsqueeze_896 = None
    unsqueeze_898: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_897, 3);  unsqueeze_897 = None
    sum_115: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_66, [0, 2, 3])
    sub_90: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_898);  convolution_1 = unsqueeze_898 = None
    mul_563: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_66, sub_90);  sub_90 = None
    sum_116: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_563, [0, 2, 3]);  mul_563 = None
    mul_568: "f32[16]" = torch.ops.aten.mul.Tensor(rsqrt_44, primals_5);  primals_5 = None
    unsqueeze_905: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_568, 0);  mul_568 = None
    unsqueeze_906: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_905, 2);  unsqueeze_905 = None
    unsqueeze_907: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, 3);  unsqueeze_906 = None
    mul_569: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_66, unsqueeze_907);  where_66 = unsqueeze_907 = None
    mul_570: "f32[16]" = torch.ops.aten.mul.Tensor(sum_116, rsqrt_44);  sum_116 = rsqrt_44 = None
    convolution_backward_60 = torch.ops.aten.convolution_backward.default(mul_569, div, primals_4, [0], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, False]);  mul_569 = div = primals_4 = None
    getitem_180: "f32[4, 16, 112, 112]" = convolution_backward_60[0]
    getitem_181: "f32[16, 1, 3, 3]" = convolution_backward_60[1];  convolution_backward_60 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:111, code: result = self.block(input)
    add_213: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(getitem_174, getitem_180);  getitem_174 = getitem_180 = None
    
    # File: /workspace/youkaichao/code/torchvision/torchvision/models/mobilenetv3.py:210, code: x = self.features(x)
    lt_28: "b8[4, 16, 112, 112]" = torch.ops.aten.lt.Scalar(clone, -3)
    le_39: "b8[4, 16, 112, 112]" = torch.ops.aten.le.Scalar(clone, 3)
    div_58: "f32[4, 16, 112, 112]" = torch.ops.aten.div.Tensor(clone, 3);  clone = None
    add_214: "f32[4, 16, 112, 112]" = torch.ops.aten.add.Tensor(div_58, 0.5);  div_58 = None
    mul_571: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(add_213, add_214);  add_214 = None
    where_67: "f32[4, 16, 112, 112]" = torch.ops.aten.where.self(le_39, mul_571, add_213);  le_39 = mul_571 = add_213 = None
    where_68: "f32[4, 16, 112, 112]" = torch.ops.aten.where.self(lt_28, full_default, where_67);  lt_28 = full_default = where_67 = None
    add_215: "f32[16]" = torch.ops.aten.add.Tensor(primals_176, 0.001);  primals_176 = None
    rsqrt_45: "f32[16]" = torch.ops.aten.rsqrt.default(add_215);  add_215 = None
    unsqueeze_908: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(primals_175, 0);  primals_175 = None
    unsqueeze_909: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, 2);  unsqueeze_908 = None
    unsqueeze_910: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_909, 3);  unsqueeze_909 = None
    sum_117: "f32[16]" = torch.ops.aten.sum.dim_IntList(where_68, [0, 2, 3])
    sub_91: "f32[4, 16, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_910);  convolution = unsqueeze_910 = None
    mul_572: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_68, sub_91);  sub_91 = None
    sum_118: "f32[16]" = torch.ops.aten.sum.dim_IntList(mul_572, [0, 2, 3]);  mul_572 = None
    mul_577: "f32[16]" = torch.ops.aten.mul.Tensor(rsqrt_45, primals_2);  primals_2 = None
    unsqueeze_917: "f32[1, 16]" = torch.ops.aten.unsqueeze.default(mul_577, 0);  mul_577 = None
    unsqueeze_918: "f32[1, 16, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_917, 2);  unsqueeze_917 = None
    unsqueeze_919: "f32[1, 16, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, 3);  unsqueeze_918 = None
    mul_578: "f32[4, 16, 112, 112]" = torch.ops.aten.mul.Tensor(where_68, unsqueeze_919);  where_68 = unsqueeze_919 = None
    mul_579: "f32[16]" = torch.ops.aten.mul.Tensor(sum_118, rsqrt_45);  sum_118 = rsqrt_45 = None
    convolution_backward_61 = torch.ops.aten.convolution_backward.default(mul_578, primals_313, primals_1, [0], [2, 2], [1, 1], [1, 1], False, [0, 0], 1, [False, True, False]);  mul_578 = primals_313 = primals_1 = None
    getitem_184: "f32[16, 3, 3, 3]" = convolution_backward_61[1];  convolution_backward_61 = None
    return [getitem_184, mul_579, sum_117, getitem_181, mul_570, sum_115, getitem_178, mul_562, sum_113, getitem_175, mul_554, sum_111, getitem_172, mul_546, sum_109, getitem_169, mul_538, sum_107, getitem_166, mul_530, sum_105, getitem_163, mul_522, sum_103, getitem_160, mul_514, sum_101, getitem_157, mul_506, sum_99, getitem_154, mul_498, sum_97, getitem_151, sum_96, getitem_148, sum_95, getitem_145, mul_487, sum_92, getitem_142, mul_479, sum_90, getitem_139, mul_471, sum_88, getitem_136, sum_87, getitem_133, sum_86, getitem_130, mul_460, sum_83, getitem_127, mul_452, sum_81, getitem_124, mul_444, sum_79, getitem_121, sum_78, getitem_118, sum_77, getitem_115, mul_433, sum_74, getitem_112, mul_425, sum_72, getitem_109, mul_416, sum_70, getitem_106, mul_407, sum_68, getitem_103, mul_399, sum_66, getitem_100, mul_390, sum_64, getitem_97, mul_381, sum_62, getitem_94, mul_373, sum_60, getitem_91, mul_364, sum_58, getitem_88, mul_355, sum_56, getitem_85, mul_347, sum_54, getitem_82, mul_338, sum_52, getitem_79, mul_329, sum_50, getitem_76, mul_321, sum_48, getitem_73, mul_312, sum_46, getitem_70, sum_45, getitem_67, sum_44, getitem_64, mul_300, sum_41, getitem_61, mul_292, sum_39, getitem_58, mul_283, sum_37, getitem_55, sum_36, getitem_52, sum_35, getitem_49, mul_271, sum_32, getitem_46, mul_263, sum_30, getitem_43, mul_254, sum_28, getitem_40, sum_27, getitem_37, sum_26, getitem_34, mul_242, sum_23, getitem_31, mul_234, sum_21, getitem_28, mul_225, sum_19, getitem_25, sum_18, getitem_22, sum_17, getitem_19, mul_213, sum_14, getitem_16, mul_205, sum_12, getitem_13, mul_196, sum_10, getitem_10, sum_9, getitem_7, sum_8, getitem_4, mul_184, sum_5, getitem_1, mul_176, sum_3, permute_9, view_2, permute_5, view_1, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]
    