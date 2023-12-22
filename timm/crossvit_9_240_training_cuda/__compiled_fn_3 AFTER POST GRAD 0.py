from __future__ import annotations



def forward(self, primals_1: "f32[1, 1, 128]", primals_2: "f32[1, 401, 128]", primals_3: "f32[1, 1, 256]", primals_4: "f32[1, 197, 256]", primals_5: "f32[128, 3, 12, 12]", primals_6: "f32[128]", primals_7: "f32[256, 3, 16, 16]", primals_8: "f32[256]", primals_9: "f32[128]", primals_10: "f32[128]", primals_11: "f32[384, 128]", primals_12: "f32[384]", primals_13: "f32[128, 128]", primals_14: "f32[128]", primals_15: "f32[128]", primals_16: "f32[128]", primals_17: "f32[384, 128]", primals_18: "f32[384]", primals_19: "f32[128, 384]", primals_20: "f32[128]", primals_21: "f32[256]", primals_22: "f32[256]", primals_23: "f32[768, 256]", primals_24: "f32[768]", primals_25: "f32[256, 256]", primals_26: "f32[256]", primals_27: "f32[256]", primals_28: "f32[256]", primals_29: "f32[768, 256]", primals_30: "f32[768]", primals_31: "f32[256, 768]", primals_32: "f32[256]", primals_33: "f32[256]", primals_34: "f32[256]", primals_35: "f32[768, 256]", primals_36: "f32[768]", primals_37: "f32[256, 256]", primals_38: "f32[256]", primals_39: "f32[256]", primals_40: "f32[256]", primals_41: "f32[768, 256]", primals_42: "f32[768]", primals_43: "f32[256, 768]", primals_44: "f32[256]", primals_45: "f32[256]", primals_46: "f32[256]", primals_47: "f32[768, 256]", primals_48: "f32[768]", primals_49: "f32[256, 256]", primals_50: "f32[256]", primals_51: "f32[256]", primals_52: "f32[256]", primals_53: "f32[768, 256]", primals_54: "f32[768]", primals_55: "f32[256, 768]", primals_56: "f32[256]", primals_57: "f32[128]", primals_58: "f32[128]", primals_59: "f32[256, 128]", primals_60: "f32[256]", primals_61: "f32[256]", primals_62: "f32[256]", primals_63: "f32[128, 256]", primals_64: "f32[128]", primals_65: "f32[256]", primals_66: "f32[256]", primals_67: "f32[256, 256]", primals_68: "f32[256]", primals_69: "f32[256, 256]", primals_70: "f32[256]", primals_71: "f32[256, 256]", primals_72: "f32[256]", primals_73: "f32[256, 256]", primals_74: "f32[256]", primals_75: "f32[256]", primals_76: "f32[256]", primals_77: "f32[128, 256]", primals_78: "f32[128]", primals_79: "f32[128]", primals_80: "f32[128]", primals_81: "f32[128, 128]", primals_82: "f32[128]", primals_83: "f32[128, 128]", primals_84: "f32[128]", primals_85: "f32[128, 128]", primals_86: "f32[128]", primals_87: "f32[128, 128]", primals_88: "f32[128]", primals_89: "f32[128]", primals_90: "f32[128]", primals_91: "f32[256, 128]", primals_92: "f32[256]", primals_93: "f32[128]", primals_94: "f32[128]", primals_95: "f32[384, 128]", primals_96: "f32[384]", primals_97: "f32[128, 128]", primals_98: "f32[128]", primals_99: "f32[128]", primals_100: "f32[128]", primals_101: "f32[384, 128]", primals_102: "f32[384]", primals_103: "f32[128, 384]", primals_104: "f32[128]", primals_105: "f32[256]", primals_106: "f32[256]", primals_107: "f32[768, 256]", primals_108: "f32[768]", primals_109: "f32[256, 256]", primals_110: "f32[256]", primals_111: "f32[256]", primals_112: "f32[256]", primals_113: "f32[768, 256]", primals_114: "f32[768]", primals_115: "f32[256, 768]", primals_116: "f32[256]", primals_117: "f32[256]", primals_118: "f32[256]", primals_119: "f32[768, 256]", primals_120: "f32[768]", primals_121: "f32[256, 256]", primals_122: "f32[256]", primals_123: "f32[256]", primals_124: "f32[256]", primals_125: "f32[768, 256]", primals_126: "f32[768]", primals_127: "f32[256, 768]", primals_128: "f32[256]", primals_129: "f32[256]", primals_130: "f32[256]", primals_131: "f32[768, 256]", primals_132: "f32[768]", primals_133: "f32[256, 256]", primals_134: "f32[256]", primals_135: "f32[256]", primals_136: "f32[256]", primals_137: "f32[768, 256]", primals_138: "f32[768]", primals_139: "f32[256, 768]", primals_140: "f32[256]", primals_141: "f32[128]", primals_142: "f32[128]", primals_143: "f32[256, 128]", primals_144: "f32[256]", primals_145: "f32[256]", primals_146: "f32[256]", primals_147: "f32[128, 256]", primals_148: "f32[128]", primals_149: "f32[256]", primals_150: "f32[256]", primals_151: "f32[256, 256]", primals_152: "f32[256]", primals_153: "f32[256, 256]", primals_154: "f32[256]", primals_155: "f32[256, 256]", primals_156: "f32[256]", primals_157: "f32[256, 256]", primals_158: "f32[256]", primals_159: "f32[256]", primals_160: "f32[256]", primals_161: "f32[128, 256]", primals_162: "f32[128]", primals_163: "f32[128]", primals_164: "f32[128]", primals_165: "f32[128, 128]", primals_166: "f32[128]", primals_167: "f32[128, 128]", primals_168: "f32[128]", primals_169: "f32[128, 128]", primals_170: "f32[128]", primals_171: "f32[128, 128]", primals_172: "f32[128]", primals_173: "f32[128]", primals_174: "f32[128]", primals_175: "f32[256, 128]", primals_176: "f32[256]", primals_177: "f32[128]", primals_178: "f32[128]", primals_179: "f32[384, 128]", primals_180: "f32[384]", primals_181: "f32[128, 128]", primals_182: "f32[128]", primals_183: "f32[128]", primals_184: "f32[128]", primals_185: "f32[384, 128]", primals_186: "f32[384]", primals_187: "f32[128, 384]", primals_188: "f32[128]", primals_189: "f32[256]", primals_190: "f32[256]", primals_191: "f32[768, 256]", primals_192: "f32[768]", primals_193: "f32[256, 256]", primals_194: "f32[256]", primals_195: "f32[256]", primals_196: "f32[256]", primals_197: "f32[768, 256]", primals_198: "f32[768]", primals_199: "f32[256, 768]", primals_200: "f32[256]", primals_201: "f32[256]", primals_202: "f32[256]", primals_203: "f32[768, 256]", primals_204: "f32[768]", primals_205: "f32[256, 256]", primals_206: "f32[256]", primals_207: "f32[256]", primals_208: "f32[256]", primals_209: "f32[768, 256]", primals_210: "f32[768]", primals_211: "f32[256, 768]", primals_212: "f32[256]", primals_213: "f32[256]", primals_214: "f32[256]", primals_215: "f32[768, 256]", primals_216: "f32[768]", primals_217: "f32[256, 256]", primals_218: "f32[256]", primals_219: "f32[256]", primals_220: "f32[256]", primals_221: "f32[768, 256]", primals_222: "f32[768]", primals_223: "f32[256, 768]", primals_224: "f32[256]", primals_225: "f32[128]", primals_226: "f32[128]", primals_227: "f32[256, 128]", primals_228: "f32[256]", primals_229: "f32[256]", primals_230: "f32[256]", primals_231: "f32[128, 256]", primals_232: "f32[128]", primals_233: "f32[256]", primals_234: "f32[256]", primals_235: "f32[256, 256]", primals_236: "f32[256]", primals_237: "f32[256, 256]", primals_238: "f32[256]", primals_239: "f32[256, 256]", primals_240: "f32[256]", primals_241: "f32[256, 256]", primals_242: "f32[256]", primals_243: "f32[256]", primals_244: "f32[256]", primals_245: "f32[128, 256]", primals_246: "f32[128]", primals_247: "f32[128]", primals_248: "f32[128]", primals_249: "f32[128, 128]", primals_250: "f32[128]", primals_251: "f32[128, 128]", primals_252: "f32[128]", primals_253: "f32[128, 128]", primals_254: "f32[128]", primals_255: "f32[128, 128]", primals_256: "f32[128]", primals_257: "f32[128]", primals_258: "f32[128]", primals_259: "f32[256, 128]", primals_260: "f32[256]", primals_261: "f32[128]", primals_262: "f32[128]", primals_263: "f32[256]", primals_264: "f32[256]", primals_265: "f32[1000, 128]", primals_266: "f32[1000]", primals_267: "f32[1000, 256]", primals_268: "f32[1000]", primals_269: "f32[8, 3, 240, 240]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:84, code: x = self.proj(x).flatten(2).transpose(1, 2)
    convolution: "f32[8, 128, 20, 20]" = torch.ops.aten.convolution.default(primals_269, primals_5, primals_6, [12, 12], [0, 0], [1, 1], False, [0, 0], 1);  primals_6 = None
    view: "f32[8, 128, 400]" = torch.ops.aten.reshape.default(convolution, [8, 128, 400]);  convolution = None
    permute: "f32[8, 400, 128]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:440, code: cls_tokens = cls_tokens.expand(B, -1, -1)
    expand: "f32[8, 1, 128]" = torch.ops.aten.expand.default(primals_1, [8, -1, -1]);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:441, code: x_ = torch.cat((cls_tokens, x_), dim=1)
    cat: "f32[8, 401, 128]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:443, code: x_ = x_ + pos_embed
    add: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(cat, primals_2);  cat = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:292, code: x = torch.nn.functional.interpolate(x, size=ss, mode='bicubic', align_corners=False)
    iota: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    view_1: "i64[8, 1, 1, 1]" = torch.ops.aten.reshape.default(iota, [8, 1, 1, 1]);  iota = None
    iota_1: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    view_2: "i64[1, 3, 1, 1]" = torch.ops.aten.reshape.default(iota_1, [1, 3, 1, 1]);  iota_1 = None
    iota_2: "i64[224]" = torch.ops.prims.iota.default(224, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    view_3: "i64[1, 1, 224, 1]" = torch.ops.aten.reshape.default(iota_2, [1, 1, 224, 1])
    view_4: "i64[1, 1, 1, 224]" = torch.ops.aten.reshape.default(iota_2, [1, 1, 1, 224]);  iota_2 = None
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
    clamp_min: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(sub_4, 0);  sub_4 = None
    clamp_max: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min, 239);  clamp_min = None
    clamp_min_1: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(sub_5, 0);  sub_5 = None
    clamp_max_1: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_1, 239);  clamp_min_1 = None
    _unsafe_index: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max, clamp_max_1])
    clamp_min_3: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(convert_element_type, 0);  convert_element_type = None
    clamp_max_3: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_3, 239);  clamp_min_3 = None
    _unsafe_index_1: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max, clamp_max_3])
    clamp_min_5: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_5: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_5, 239);  clamp_min_5 = None
    _unsafe_index_2: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max, clamp_max_5])
    clamp_min_7: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_6, 0);  add_6 = None
    clamp_max_7: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_7, 239);  clamp_min_7 = None
    _unsafe_index_3: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max, clamp_max_7]);  clamp_max = None
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
    sub_11: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(2.0, sub_1);  sub_1 = None
    mul_11: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_11, -0.75)
    sub_12: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_11, -3.75);  mul_11 = None
    mul_12: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_12, sub_11);  sub_12 = None
    add_11: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_12, -6.0);  mul_12 = None
    mul_13: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_11, sub_11);  add_11 = sub_11 = None
    sub_13: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_13, -3.0);  mul_13 = None
    mul_14: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index, sub_7);  _unsafe_index = None
    mul_15: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_1, add_9);  _unsafe_index_1 = None
    add_12: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None
    mul_16: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_2, add_10);  _unsafe_index_2 = None
    add_13: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_12, mul_16);  add_12 = mul_16 = None
    mul_17: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_3, sub_13);  _unsafe_index_3 = None
    add_14: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_13, mul_17);  add_13 = mul_17 = None
    clamp_min_8: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(convert_element_type_1, 0);  convert_element_type_1 = None
    clamp_max_8: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_8, 239);  clamp_min_8 = None
    _unsafe_index_4: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_8, clamp_max_1])
    _unsafe_index_5: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_8, clamp_max_3])
    _unsafe_index_6: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_8, clamp_max_5])
    _unsafe_index_7: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_8, clamp_max_7]);  clamp_max_8 = None
    mul_30: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_4, sub_7);  _unsafe_index_4 = None
    mul_31: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_5, add_9);  _unsafe_index_5 = None
    add_20: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
    mul_32: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_6, add_10);  _unsafe_index_6 = None
    add_21: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_20, mul_32);  add_20 = mul_32 = None
    mul_33: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_7, sub_13);  _unsafe_index_7 = None
    add_22: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_21, mul_33);  add_21 = mul_33 = None
    clamp_min_16: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
    clamp_max_16: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_16, 239);  clamp_min_16 = None
    _unsafe_index_8: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_16, clamp_max_1])
    _unsafe_index_9: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_16, clamp_max_3])
    _unsafe_index_10: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_16, clamp_max_5])
    _unsafe_index_11: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_16, clamp_max_7]);  clamp_max_16 = None
    mul_46: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_8, sub_7);  _unsafe_index_8 = None
    mul_47: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_9, add_9);  _unsafe_index_9 = None
    add_28: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    mul_48: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_10, add_10);  _unsafe_index_10 = None
    add_29: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_28, mul_48);  add_28 = mul_48 = None
    mul_49: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_11, sub_13);  _unsafe_index_11 = None
    add_30: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_29, mul_49);  add_29 = mul_49 = None
    clamp_min_24: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_4, 0);  add_4 = None
    clamp_max_24: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_24, 239);  clamp_min_24 = None
    _unsafe_index_12: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_24, clamp_max_1]);  clamp_max_1 = None
    _unsafe_index_13: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_24, clamp_max_3]);  clamp_max_3 = None
    _unsafe_index_14: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_24, clamp_max_5]);  clamp_max_5 = None
    _unsafe_index_15: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_24, clamp_max_7]);  view_1 = view_2 = clamp_max_24 = clamp_max_7 = None
    mul_62: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_12, sub_7);  _unsafe_index_12 = sub_7 = None
    mul_63: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_13, add_9);  _unsafe_index_13 = add_9 = None
    add_36: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
    mul_64: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_14, add_10);  _unsafe_index_14 = add_10 = None
    add_37: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_36, mul_64);  add_36 = mul_64 = None
    mul_65: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_15, sub_13);  _unsafe_index_15 = sub_13 = None
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
    convolution_1: "f32[8, 256, 14, 14]" = torch.ops.aten.convolution.default(add_46, primals_7, primals_8, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_8 = None
    view_5: "f32[8, 256, 196]" = torch.ops.aten.reshape.default(convolution_1, [8, 256, 196]);  convolution_1 = None
    permute_1: "f32[8, 196, 256]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:440, code: cls_tokens = cls_tokens.expand(B, -1, -1)
    expand_1: "f32[8, 1, 256]" = torch.ops.aten.expand.default(primals_3, [8, -1, -1]);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:441, code: x_ = torch.cat((cls_tokens, x_), dim=1)
    cat_1: "f32[8, 197, 256]" = torch.ops.aten.cat.default([expand_1, permute_1], 1);  expand_1 = permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:443, code: x_ = x_ + pos_embed
    add_47: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(cat_1, primals_4);  cat_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 401, 1]" = var_mean[0]
    getitem_1: "f32[8, 401, 1]" = var_mean[1];  var_mean = None
    add_48: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_46: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
    mul_82: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt);  sub_46 = None
    mul_83: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_82, primals_9)
    add_49: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_83, primals_10);  mul_83 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_6: "f32[3208, 128]" = torch.ops.aten.reshape.default(add_49, [3208, 128]);  add_49 = None
    permute_2: "f32[128, 384]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm: "f32[3208, 384]" = torch.ops.aten.addmm.default(primals_12, view_6, permute_2);  primals_12 = None
    view_7: "f32[8, 401, 384]" = torch.ops.aten.reshape.default(addmm, [8, 401, 384]);  addmm = None
    view_8: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.reshape.default(view_7, [8, 401, 3, 4, 32]);  view_7 = None
    permute_3: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.permute.default(view_8, [2, 0, 3, 1, 4]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
    getitem_2: "f32[8, 4, 401, 32]" = unbind[0]
    getitem_3: "f32[8, 4, 401, 32]" = unbind[1]
    getitem_4: "f32[8, 4, 401, 32]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_2, getitem_3, getitem_4, None, True)
    getitem_5: "f32[8, 4, 401, 32]" = _scaled_dot_product_efficient_attention[0]
    getitem_6: "f32[8, 4, 416]" = _scaled_dot_product_efficient_attention[1]
    getitem_7: "i64[]" = _scaled_dot_product_efficient_attention[2]
    getitem_8: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
    alias: "f32[8, 4, 401, 32]" = torch.ops.aten.alias.default(getitem_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_4: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    view_9: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(permute_4, [8, 401, 128]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_10: "f32[3208, 128]" = torch.ops.aten.reshape.default(view_9, [3208, 128]);  view_9 = None
    permute_5: "f32[128, 128]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    
    # No stacktrace found for following nodes
    mm_default_41: "f32[3208, 128]" = torch.ops.aten.mm.default(view_10, permute_5)
    add_tensor_41: "f32[3208, 128]" = torch.ops.aten.add.Tensor(mm_default_41, primals_14);  mm_default_41 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_11: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(add_tensor_41, [8, 401, 128]);  add_tensor_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_50: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add, view_11);  add = view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_1 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_9: "f32[8, 401, 1]" = var_mean_1[0]
    getitem_10: "f32[8, 401, 1]" = var_mean_1[1];  var_mean_1 = None
    add_51: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-06);  getitem_9 = None
    rsqrt_1: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_47: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add_50, getitem_10);  getitem_10 = None
    mul_84: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_1);  sub_47 = None
    mul_85: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_84, primals_15)
    add_52: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_85, primals_16);  mul_85 = primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_12: "f32[3208, 128]" = torch.ops.aten.reshape.default(add_52, [3208, 128]);  add_52 = None
    permute_6: "f32[128, 384]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_2: "f32[3208, 384]" = torch.ops.aten.addmm.default(primals_18, view_12, permute_6);  primals_18 = None
    view_13: "f32[8, 401, 384]" = torch.ops.aten.reshape.default(addmm_2, [8, 401, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
    mul_87: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476);  view_13 = None
    erf: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_53: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_88: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_86, add_53);  mul_86 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_14: "f32[3208, 384]" = torch.ops.aten.reshape.default(mul_88, [3208, 384]);  mul_88 = None
    permute_7: "f32[384, 128]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    
    # No stacktrace found for following nodes
    mm_default_40: "f32[3208, 128]" = torch.ops.aten.mm.default(view_14, permute_7)
    add_tensor_40: "f32[3208, 128]" = torch.ops.aten.add.Tensor(mm_default_40, primals_20);  mm_default_40 = primals_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_15: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(add_tensor_40, [8, 401, 128]);  add_tensor_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_54: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_50, view_15);  add_50 = view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_2 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_11: "f32[8, 197, 1]" = var_mean_2[0]
    getitem_12: "f32[8, 197, 1]" = var_mean_2[1];  var_mean_2 = None
    add_55: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-06);  getitem_11 = None
    rsqrt_2: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_48: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_47, getitem_12);  getitem_12 = None
    mul_89: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_2);  sub_48 = None
    mul_90: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_89, primals_21)
    add_56: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_90, primals_22);  mul_90 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_16: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_56, [1576, 256]);  add_56 = None
    permute_8: "f32[256, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_4: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_24, view_16, permute_8);  primals_24 = None
    view_17: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_4, [8, 197, 768]);  addmm_4 = None
    view_18: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.reshape.default(view_17, [8, 197, 3, 4, 64]);  view_17 = None
    permute_9: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_18, [2, 0, 3, 1, 4]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
    getitem_13: "f32[8, 4, 197, 64]" = unbind_1[0]
    getitem_14: "f32[8, 4, 197, 64]" = unbind_1[1]
    getitem_15: "f32[8, 4, 197, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_13, getitem_14, getitem_15, None, True)
    getitem_16: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_1[0]
    getitem_17: "f32[8, 4, 224]" = _scaled_dot_product_efficient_attention_1[1]
    getitem_18: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
    getitem_19: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
    alias_1: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_10: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
    view_19: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(permute_10, [8, 197, 256]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_20: "f32[1576, 256]" = torch.ops.aten.reshape.default(view_19, [1576, 256]);  view_19 = None
    permute_11: "f32[256, 256]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    
    # No stacktrace found for following nodes
    mm_default_39: "f32[1576, 256]" = torch.ops.aten.mm.default(view_20, permute_11)
    add_tensor_39: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_39, primals_26);  mm_default_39 = primals_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_21: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_39, [8, 197, 256]);  add_tensor_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_57: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_47, view_21);  add_47 = view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_3 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 197, 1]" = var_mean_3[0]
    getitem_21: "f32[8, 197, 1]" = var_mean_3[1];  var_mean_3 = None
    add_58: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_3: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_49: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_57, getitem_21);  getitem_21 = None
    mul_91: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_3);  sub_49 = None
    mul_92: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_91, primals_27)
    add_59: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_92, primals_28);  mul_92 = primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_22: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_59, [1576, 256]);  add_59 = None
    permute_12: "f32[256, 768]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
    addmm_6: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_30, view_22, permute_12);  primals_30 = None
    view_23: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_6, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_93: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_23, 0.5)
    mul_94: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_23, 0.7071067811865476);  view_23 = None
    erf_1: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_60: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_95: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_93, add_60);  mul_93 = add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_24: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_95, [1576, 768]);  mul_95 = None
    permute_13: "f32[768, 256]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    
    # No stacktrace found for following nodes
    mm_default_38: "f32[1576, 256]" = torch.ops.aten.mm.default(view_24, permute_13)
    add_tensor_38: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_38, primals_32);  mm_default_38 = primals_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_25: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_38, [8, 197, 256]);  add_tensor_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_61: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_57, view_25);  add_57 = view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_4 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 197, 1]" = var_mean_4[0]
    getitem_23: "f32[8, 197, 1]" = var_mean_4[1];  var_mean_4 = None
    add_62: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_4: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_50: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_61, getitem_23);  getitem_23 = None
    mul_96: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_4);  sub_50 = None
    mul_97: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_96, primals_33)
    add_63: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_97, primals_34);  mul_97 = primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_26: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_63, [1576, 256]);  add_63 = None
    permute_14: "f32[256, 768]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
    addmm_8: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_36, view_26, permute_14);  primals_36 = None
    view_27: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_8, [8, 197, 768]);  addmm_8 = None
    view_28: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.reshape.default(view_27, [8, 197, 3, 4, 64]);  view_27 = None
    permute_15: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_28, [2, 0, 3, 1, 4]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_15);  permute_15 = None
    getitem_24: "f32[8, 4, 197, 64]" = unbind_2[0]
    getitem_25: "f32[8, 4, 197, 64]" = unbind_2[1]
    getitem_26: "f32[8, 4, 197, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_24, getitem_25, getitem_26, None, True)
    getitem_27: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_2[0]
    getitem_28: "f32[8, 4, 224]" = _scaled_dot_product_efficient_attention_2[1]
    getitem_29: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
    getitem_30: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
    alias_2: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_16: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
    view_29: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(permute_16, [8, 197, 256]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_30: "f32[1576, 256]" = torch.ops.aten.reshape.default(view_29, [1576, 256]);  view_29 = None
    permute_17: "f32[256, 256]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    
    # No stacktrace found for following nodes
    mm_default_37: "f32[1576, 256]" = torch.ops.aten.mm.default(view_30, permute_17)
    add_tensor_37: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_37, primals_38);  mm_default_37 = primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_31: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_37, [8, 197, 256]);  add_tensor_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_64: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_61, view_31);  add_61 = view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_5 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_31: "f32[8, 197, 1]" = var_mean_5[0]
    getitem_32: "f32[8, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    add_65: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-06);  getitem_31 = None
    rsqrt_5: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_51: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_64, getitem_32);  getitem_32 = None
    mul_98: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_5);  sub_51 = None
    mul_99: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_98, primals_39)
    add_66: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_99, primals_40);  mul_99 = primals_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_32: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_66, [1576, 256]);  add_66 = None
    permute_18: "f32[256, 768]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    addmm_10: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_42, view_32, permute_18);  primals_42 = None
    view_33: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_10, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
    mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476);  view_33 = None
    erf_2: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
    add_67: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_102: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_100, add_67);  mul_100 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_34: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_102, [1576, 768]);  mul_102 = None
    permute_19: "f32[768, 256]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    
    # No stacktrace found for following nodes
    mm_default_36: "f32[1576, 256]" = torch.ops.aten.mm.default(view_34, permute_19)
    add_tensor_36: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_36, primals_44);  mm_default_36 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_35: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_36, [8, 197, 256]);  add_tensor_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_68: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_64, view_35);  add_64 = view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_6 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
    getitem_33: "f32[8, 197, 1]" = var_mean_6[0]
    getitem_34: "f32[8, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    add_69: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
    rsqrt_6: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_52: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_68, getitem_34);  getitem_34 = None
    mul_103: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_6);  sub_52 = None
    mul_104: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_103, primals_45)
    add_70: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_104, primals_46);  mul_104 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_36: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_70, [1576, 256]);  add_70 = None
    permute_20: "f32[256, 768]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_12: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_48, view_36, permute_20);  primals_48 = None
    view_37: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_12, [8, 197, 768]);  addmm_12 = None
    view_38: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.reshape.default(view_37, [8, 197, 3, 4, 64]);  view_37 = None
    permute_21: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_38, [2, 0, 3, 1, 4]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_21);  permute_21 = None
    getitem_35: "f32[8, 4, 197, 64]" = unbind_3[0]
    getitem_36: "f32[8, 4, 197, 64]" = unbind_3[1]
    getitem_37: "f32[8, 4, 197, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_35, getitem_36, getitem_37, None, True)
    getitem_38: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_3[0]
    getitem_39: "f32[8, 4, 224]" = _scaled_dot_product_efficient_attention_3[1]
    getitem_40: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
    getitem_41: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
    alias_3: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_22: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
    view_39: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(permute_22, [8, 197, 256]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_40: "f32[1576, 256]" = torch.ops.aten.reshape.default(view_39, [1576, 256]);  view_39 = None
    permute_23: "f32[256, 256]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    
    # No stacktrace found for following nodes
    mm_default_35: "f32[1576, 256]" = torch.ops.aten.mm.default(view_40, permute_23)
    add_tensor_35: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_35, primals_50);  mm_default_35 = primals_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_41: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_35, [8, 197, 256]);  add_tensor_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_71: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_68, view_41);  add_68 = view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_7 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 197, 1]" = var_mean_7[0]
    getitem_43: "f32[8, 197, 1]" = var_mean_7[1];  var_mean_7 = None
    add_72: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_7: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_53: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_71, getitem_43);  getitem_43 = None
    mul_105: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_7);  sub_53 = None
    mul_106: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_105, primals_51)
    add_73: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_106, primals_52);  mul_106 = primals_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_42: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_73, [1576, 256]);  add_73 = None
    permute_24: "f32[256, 768]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    addmm_14: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_54, view_42, permute_24);  primals_54 = None
    view_43: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_14, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_107: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_108: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476);  view_43 = None
    erf_3: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
    add_74: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_107, add_74);  mul_107 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_44: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_109, [1576, 768]);  mul_109 = None
    permute_25: "f32[768, 256]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    
    # No stacktrace found for following nodes
    mm_default_34: "f32[1576, 256]" = torch.ops.aten.mm.default(view_44, permute_25)
    add_tensor_34: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_34, primals_56);  mm_default_34 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_45: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_34, [8, 197, 256]);  add_tensor_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_75: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_71, view_45);  add_71 = view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    slice_2: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_54, 1, 0, 1)
    clone_14: "f32[8, 1, 128]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_14, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 1, 1]" = var_mean_8[0]
    getitem_45: "f32[8, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_76: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_8: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_54: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(clone_14, getitem_45);  clone_14 = getitem_45 = None
    mul_110: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_8);  sub_54 = None
    mul_111: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_110, primals_57)
    add_77: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_111, primals_58);  mul_111 = None
    mul_112: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_77, 0.5)
    mul_113: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_77, 0.7071067811865476);  add_77 = None
    erf_4: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_113);  mul_113 = None
    add_78: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_114: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_112, add_78);  mul_112 = add_78 = None
    view_46: "f32[8, 128]" = torch.ops.aten.reshape.default(mul_114, [8, 128]);  mul_114 = None
    permute_26: "f32[128, 256]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    addmm_16: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_60, view_46, permute_26);  primals_60 = None
    view_47: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(addmm_16, [8, 1, 256]);  addmm_16 = None
    slice_4: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_75, 1, 0, 1)
    clone_15: "f32[8, 1, 256]" = torch.ops.aten.clone.default(slice_4, memory_format = torch.contiguous_format);  slice_4 = None
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_15, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 1, 1]" = var_mean_9[0]
    getitem_47: "f32[8, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_79: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_9: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_55: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(clone_15, getitem_47);  clone_15 = getitem_47 = None
    mul_115: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_9);  sub_55 = None
    mul_116: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_115, primals_61)
    add_80: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_116, primals_62);  mul_116 = None
    mul_117: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_80, 0.5)
    mul_118: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_80, 0.7071067811865476);  add_80 = None
    erf_5: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_81: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_119: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_117, add_81);  mul_117 = add_81 = None
    view_48: "f32[8, 256]" = torch.ops.aten.reshape.default(mul_119, [8, 256]);  mul_119 = None
    permute_27: "f32[256, 128]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    addmm_17: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_64, view_48, permute_27);  primals_64 = None
    view_49: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(addmm_17, [8, 1, 128]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_6: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_75, 1, 1, 9223372036854775807);  add_75 = None
    cat_2: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_47, slice_6], 1);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_7: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(cat_2, 0, 0, 9223372036854775807)
    slice_8: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 1);  slice_7 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(cat_2, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 197, 1]" = var_mean_10[0]
    getitem_49: "f32[8, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    add_82: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_10: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_56: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_2, getitem_49)
    mul_120: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_10);  sub_56 = None
    mul_121: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_120, primals_65);  mul_120 = None
    add_83: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_121, primals_66);  mul_121 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_9: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_83, 0, 0, 9223372036854775807)
    slice_10: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 1);  slice_9 = None
    permute_28: "f32[256, 256]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    view_50: "f32[8, 256]" = torch.ops.aten.reshape.default(slice_10, [8, 256]);  slice_10 = None
    mm: "f32[8, 256]" = torch.ops.aten.mm.default(view_50, permute_28)
    view_51: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(mm, [8, 1, 256]);  mm = None
    add_84: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(view_51, primals_68);  view_51 = primals_68 = None
    view_52: "f32[8, 1, 4, 64]" = torch.ops.aten.reshape.default(add_84, [8, 1, 4, 64]);  add_84 = None
    permute_29: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_53: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_83, [1576, 256]);  add_83 = None
    permute_30: "f32[256, 256]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    
    # No stacktrace found for following nodes
    mm_default_33: "f32[1576, 256]" = torch.ops.aten.mm.default(view_53, permute_30)
    add_tensor_33: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_33, primals_70);  mm_default_33 = primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_54: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_33, [8, 197, 256]);  add_tensor_33 = None
    view_55: "f32[8, 197, 4, 64]" = torch.ops.aten.reshape.default(view_54, [8, 197, 4, 64]);  view_54 = None
    permute_31: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_32: "f32[256, 256]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    
    # No stacktrace found for following nodes
    mm_default_32: "f32[1576, 256]" = torch.ops.aten.mm.default(view_53, permute_32)
    add_tensor_32: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_32, primals_72);  mm_default_32 = primals_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_57: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_32, [8, 197, 256]);  add_tensor_32 = None
    view_58: "f32[8, 197, 4, 64]" = torch.ops.aten.reshape.default(view_57, [8, 197, 4, 64]);  view_57 = None
    permute_33: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_34: "f32[8, 4, 64, 197]" = torch.ops.aten.permute.default(permute_31, [0, 1, 3, 2]);  permute_31 = None
    expand_2: "f32[8, 4, 1, 64]" = torch.ops.aten.expand.default(permute_29, [8, 4, 1, 64]);  permute_29 = None
    view_59: "f32[32, 1, 64]" = torch.ops.aten.reshape.default(expand_2, [32, 1, 64]);  expand_2 = None
    expand_3: "f32[8, 4, 64, 197]" = torch.ops.aten.expand.default(permute_34, [8, 4, 64, 197]);  permute_34 = None
    clone_16: "f32[8, 4, 64, 197]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_60: "f32[32, 64, 197]" = torch.ops.aten.reshape.default(clone_16, [32, 64, 197]);  clone_16 = None
    bmm: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_59, view_60)
    view_61: "f32[8, 4, 1, 197]" = torch.ops.aten.reshape.default(bmm, [8, 4, 1, 197]);  bmm = None
    mul_122: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(view_61, 0.125);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    amax: "f32[8, 4, 1, 1]" = torch.ops.aten.amax.default(mul_122, [-1], True)
    sub_57: "f32[8, 4, 1, 197]" = torch.ops.aten.sub.Tensor(mul_122, amax);  mul_122 = amax = None
    exp: "f32[8, 4, 1, 197]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
    sum_1: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 4, 1, 197]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias_4: "f32[8, 4, 1, 197]" = torch.ops.aten.alias.default(div)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    clone_17: "f32[8, 4, 1, 197]" = torch.ops.aten.clone.default(div);  div = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    expand_4: "f32[8, 4, 1, 197]" = torch.ops.aten.expand.default(clone_17, [8, 4, 1, 197]);  clone_17 = None
    view_62: "f32[32, 1, 197]" = torch.ops.aten.reshape.default(expand_4, [32, 1, 197]);  expand_4 = None
    expand_5: "f32[8, 4, 197, 64]" = torch.ops.aten.expand.default(permute_33, [8, 4, 197, 64]);  permute_33 = None
    clone_18: "f32[8, 4, 197, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_63: "f32[32, 197, 64]" = torch.ops.aten.reshape.default(clone_18, [32, 197, 64]);  clone_18 = None
    bmm_1: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_62, view_63)
    view_64: "f32[8, 4, 1, 64]" = torch.ops.aten.reshape.default(bmm_1, [8, 4, 1, 64]);  bmm_1 = None
    permute_35: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
    view_65: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(permute_35, [8, 1, 256]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_66: "f32[8, 256]" = torch.ops.aten.reshape.default(view_65, [8, 256]);  view_65 = None
    permute_36: "f32[256, 256]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    
    # No stacktrace found for following nodes
    mm_default_31: "f32[8, 256]" = torch.ops.aten.mm.default(view_66, permute_36)
    add_tensor_31: "f32[8, 256]" = torch.ops.aten.add.Tensor(mm_default_31, primals_74);  mm_default_31 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_67: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(add_tensor_31, [8, 1, 256]);  add_tensor_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_85: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(slice_8, view_67);  slice_8 = view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    var_mean_11 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
    getitem_50: "f32[8, 1, 1]" = var_mean_11[0]
    getitem_51: "f32[8, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_86: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-06);  getitem_50 = None
    rsqrt_11: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_58: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(add_85, getitem_51);  add_85 = getitem_51 = None
    mul_123: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_11);  sub_58 = None
    mul_124: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_123, primals_75)
    add_87: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_124, primals_76);  mul_124 = None
    mul_125: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_87, 0.5)
    mul_126: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_87, 0.7071067811865476);  add_87 = None
    erf_6: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
    add_88: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_127: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_125, add_88);  mul_125 = add_88 = None
    view_68: "f32[8, 256]" = torch.ops.aten.reshape.default(mul_127, [8, 256]);  mul_127 = None
    permute_37: "f32[256, 128]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm_21: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_78, view_68, permute_37);  primals_78 = None
    view_69: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(addmm_21, [8, 1, 128]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_13: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_54, 1, 1, 9223372036854775807);  add_54 = None
    cat_3: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_69, slice_13], 1);  view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    cat_4: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_49, slice_13], 1);  view_49 = slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_16: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(cat_4, 0, 0, 9223372036854775807)
    slice_17: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 1);  slice_16 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(cat_4, [2], correction = 0, keepdim = True)
    getitem_52: "f32[8, 401, 1]" = var_mean_12[0]
    getitem_53: "f32[8, 401, 1]" = var_mean_12[1];  var_mean_12 = None
    add_89: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_52, 1e-06);  getitem_52 = None
    rsqrt_12: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_59: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_4, getitem_53)
    mul_128: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_12);  sub_59 = None
    mul_129: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_128, primals_79);  mul_128 = None
    add_90: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_129, primals_80);  mul_129 = primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_18: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_90, 0, 0, 9223372036854775807)
    slice_19: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_18, 1, 0, 1);  slice_18 = None
    permute_38: "f32[128, 128]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    view_70: "f32[8, 128]" = torch.ops.aten.reshape.default(slice_19, [8, 128]);  slice_19 = None
    mm_1: "f32[8, 128]" = torch.ops.aten.mm.default(view_70, permute_38)
    view_71: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(mm_1, [8, 1, 128]);  mm_1 = None
    add_91: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(view_71, primals_82);  view_71 = primals_82 = None
    view_72: "f32[8, 1, 4, 32]" = torch.ops.aten.reshape.default(add_91, [8, 1, 4, 32]);  add_91 = None
    permute_39: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_73: "f32[3208, 128]" = torch.ops.aten.reshape.default(add_90, [3208, 128]);  add_90 = None
    permute_40: "f32[128, 128]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    
    # No stacktrace found for following nodes
    mm_default_30: "f32[3208, 128]" = torch.ops.aten.mm.default(view_73, permute_40)
    add_tensor_30: "f32[3208, 128]" = torch.ops.aten.add.Tensor(mm_default_30, primals_84);  mm_default_30 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_74: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(add_tensor_30, [8, 401, 128]);  add_tensor_30 = None
    view_75: "f32[8, 401, 4, 32]" = torch.ops.aten.reshape.default(view_74, [8, 401, 4, 32]);  view_74 = None
    permute_41: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_42: "f32[128, 128]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    
    # No stacktrace found for following nodes
    mm_default_29: "f32[3208, 128]" = torch.ops.aten.mm.default(view_73, permute_42)
    add_tensor_29: "f32[3208, 128]" = torch.ops.aten.add.Tensor(mm_default_29, primals_86);  mm_default_29 = primals_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_77: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(add_tensor_29, [8, 401, 128]);  add_tensor_29 = None
    view_78: "f32[8, 401, 4, 32]" = torch.ops.aten.reshape.default(view_77, [8, 401, 4, 32]);  view_77 = None
    permute_43: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_44: "f32[8, 4, 32, 401]" = torch.ops.aten.permute.default(permute_41, [0, 1, 3, 2]);  permute_41 = None
    expand_6: "f32[8, 4, 1, 32]" = torch.ops.aten.expand.default(permute_39, [8, 4, 1, 32]);  permute_39 = None
    view_79: "f32[32, 1, 32]" = torch.ops.aten.reshape.default(expand_6, [32, 1, 32]);  expand_6 = None
    expand_7: "f32[8, 4, 32, 401]" = torch.ops.aten.expand.default(permute_44, [8, 4, 32, 401]);  permute_44 = None
    clone_20: "f32[8, 4, 32, 401]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_80: "f32[32, 32, 401]" = torch.ops.aten.reshape.default(clone_20, [32, 32, 401]);  clone_20 = None
    bmm_2: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_79, view_80)
    view_81: "f32[8, 4, 1, 401]" = torch.ops.aten.reshape.default(bmm_2, [8, 4, 1, 401]);  bmm_2 = None
    mul_130: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(view_81, 0.1767766952966369);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    amax_1: "f32[8, 4, 1, 1]" = torch.ops.aten.amax.default(mul_130, [-1], True)
    sub_60: "f32[8, 4, 1, 401]" = torch.ops.aten.sub.Tensor(mul_130, amax_1);  mul_130 = amax_1 = None
    exp_1: "f32[8, 4, 1, 401]" = torch.ops.aten.exp.default(sub_60);  sub_60 = None
    sum_2: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 4, 1, 401]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_5: "f32[8, 4, 1, 401]" = torch.ops.aten.alias.default(div_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    clone_21: "f32[8, 4, 1, 401]" = torch.ops.aten.clone.default(div_1);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    expand_8: "f32[8, 4, 1, 401]" = torch.ops.aten.expand.default(clone_21, [8, 4, 1, 401]);  clone_21 = None
    view_82: "f32[32, 1, 401]" = torch.ops.aten.reshape.default(expand_8, [32, 1, 401]);  expand_8 = None
    expand_9: "f32[8, 4, 401, 32]" = torch.ops.aten.expand.default(permute_43, [8, 4, 401, 32]);  permute_43 = None
    clone_22: "f32[8, 4, 401, 32]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_83: "f32[32, 401, 32]" = torch.ops.aten.reshape.default(clone_22, [32, 401, 32]);  clone_22 = None
    bmm_3: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_82, view_83)
    view_84: "f32[8, 4, 1, 32]" = torch.ops.aten.reshape.default(bmm_3, [8, 4, 1, 32]);  bmm_3 = None
    permute_45: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    view_85: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(permute_45, [8, 1, 128]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_86: "f32[8, 128]" = torch.ops.aten.reshape.default(view_85, [8, 128]);  view_85 = None
    permute_46: "f32[128, 128]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    
    # No stacktrace found for following nodes
    mm_default_28: "f32[8, 128]" = torch.ops.aten.mm.default(view_86, permute_46)
    add_tensor_28: "f32[8, 128]" = torch.ops.aten.add.Tensor(mm_default_28, primals_88);  mm_default_28 = primals_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_87: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(add_tensor_28, [8, 1, 128]);  add_tensor_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_92: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(slice_17, view_87);  slice_17 = view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    var_mean_13 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 1, 1]" = var_mean_13[0]
    getitem_55: "f32[8, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_93: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_13: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_61: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(add_92, getitem_55);  add_92 = getitem_55 = None
    mul_131: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_13);  sub_61 = None
    mul_132: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_131, primals_89)
    add_94: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_132, primals_90);  mul_132 = None
    mul_133: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_94, 0.5)
    mul_134: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_94, 0.7071067811865476);  add_94 = None
    erf_7: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_134);  mul_134 = None
    add_95: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_135: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_133, add_95);  mul_133 = add_95 = None
    view_88: "f32[8, 128]" = torch.ops.aten.reshape.default(mul_135, [8, 128]);  mul_135 = None
    permute_47: "f32[128, 256]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_25: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_92, view_88, permute_47);  primals_92 = None
    view_89: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(addmm_25, [8, 1, 256]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    cat_5: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_89, slice_6], 1);  view_89 = slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_14 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
    getitem_56: "f32[8, 401, 1]" = var_mean_14[0]
    getitem_57: "f32[8, 401, 1]" = var_mean_14[1];  var_mean_14 = None
    add_96: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_14: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_62: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_3, getitem_57)
    mul_136: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_14);  sub_62 = None
    mul_137: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_136, primals_93);  mul_136 = None
    add_97: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_137, primals_94);  mul_137 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_90: "f32[3208, 128]" = torch.ops.aten.reshape.default(add_97, [3208, 128]);  add_97 = None
    permute_48: "f32[128, 384]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_26: "f32[3208, 384]" = torch.ops.aten.addmm.default(primals_96, view_90, permute_48);  primals_96 = None
    view_91: "f32[8, 401, 384]" = torch.ops.aten.reshape.default(addmm_26, [8, 401, 384]);  addmm_26 = None
    view_92: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.reshape.default(view_91, [8, 401, 3, 4, 32]);  view_91 = None
    permute_49: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.permute.default(view_92, [2, 0, 3, 1, 4]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_49);  permute_49 = None
    getitem_58: "f32[8, 4, 401, 32]" = unbind_4[0]
    getitem_59: "f32[8, 4, 401, 32]" = unbind_4[1]
    getitem_60: "f32[8, 4, 401, 32]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_58, getitem_59, getitem_60, None, True)
    getitem_61: "f32[8, 4, 401, 32]" = _scaled_dot_product_efficient_attention_4[0]
    getitem_62: "f32[8, 4, 416]" = _scaled_dot_product_efficient_attention_4[1]
    getitem_63: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
    getitem_64: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
    alias_6: "f32[8, 4, 401, 32]" = torch.ops.aten.alias.default(getitem_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_50: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(getitem_61, [0, 2, 1, 3]);  getitem_61 = None
    view_93: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(permute_50, [8, 401, 128]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_94: "f32[3208, 128]" = torch.ops.aten.reshape.default(view_93, [3208, 128]);  view_93 = None
    permute_51: "f32[128, 128]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    
    # No stacktrace found for following nodes
    mm_default_27: "f32[3208, 128]" = torch.ops.aten.mm.default(view_94, permute_51)
    add_tensor_27: "f32[3208, 128]" = torch.ops.aten.add.Tensor(mm_default_27, primals_98);  mm_default_27 = primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_95: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(add_tensor_27, [8, 401, 128]);  add_tensor_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_98: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(cat_3, view_95);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_15 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_65: "f32[8, 401, 1]" = var_mean_15[0]
    getitem_66: "f32[8, 401, 1]" = var_mean_15[1];  var_mean_15 = None
    add_99: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_65, 1e-06);  getitem_65 = None
    rsqrt_15: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_63: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add_98, getitem_66);  getitem_66 = None
    mul_138: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_15);  sub_63 = None
    mul_139: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_138, primals_99)
    add_100: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_139, primals_100);  mul_139 = primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_96: "f32[3208, 128]" = torch.ops.aten.reshape.default(add_100, [3208, 128]);  add_100 = None
    permute_52: "f32[128, 384]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_28: "f32[3208, 384]" = torch.ops.aten.addmm.default(primals_102, view_96, permute_52);  primals_102 = None
    view_97: "f32[8, 401, 384]" = torch.ops.aten.reshape.default(addmm_28, [8, 401, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_140: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_97, 0.5)
    mul_141: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_97, 0.7071067811865476);  view_97 = None
    erf_8: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_101: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_142: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_140, add_101);  mul_140 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_98: "f32[3208, 384]" = torch.ops.aten.reshape.default(mul_142, [3208, 384]);  mul_142 = None
    permute_53: "f32[384, 128]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    
    # No stacktrace found for following nodes
    mm_default_26: "f32[3208, 128]" = torch.ops.aten.mm.default(view_98, permute_53)
    add_tensor_26: "f32[3208, 128]" = torch.ops.aten.add.Tensor(mm_default_26, primals_104);  mm_default_26 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_99: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(add_tensor_26, [8, 401, 128]);  add_tensor_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_102: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_98, view_99);  add_98 = view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_16 = torch.ops.aten.var_mean.correction(cat_5, [2], correction = 0, keepdim = True)
    getitem_67: "f32[8, 197, 1]" = var_mean_16[0]
    getitem_68: "f32[8, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    add_103: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_67, 1e-06);  getitem_67 = None
    rsqrt_16: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    sub_64: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_5, getitem_68)
    mul_143: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_16);  sub_64 = None
    mul_144: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_143, primals_105);  mul_143 = None
    add_104: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_144, primals_106);  mul_144 = primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_100: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_104, [1576, 256]);  add_104 = None
    permute_54: "f32[256, 768]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_30: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_108, view_100, permute_54);  primals_108 = None
    view_101: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_30, [8, 197, 768]);  addmm_30 = None
    view_102: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.reshape.default(view_101, [8, 197, 3, 4, 64]);  view_101 = None
    permute_55: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_102, [2, 0, 3, 1, 4]);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_55);  permute_55 = None
    getitem_69: "f32[8, 4, 197, 64]" = unbind_5[0]
    getitem_70: "f32[8, 4, 197, 64]" = unbind_5[1]
    getitem_71: "f32[8, 4, 197, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_69, getitem_70, getitem_71, None, True)
    getitem_72: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_5[0]
    getitem_73: "f32[8, 4, 224]" = _scaled_dot_product_efficient_attention_5[1]
    getitem_74: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
    getitem_75: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
    alias_7: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_56: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None
    view_103: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(permute_56, [8, 197, 256]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_104: "f32[1576, 256]" = torch.ops.aten.reshape.default(view_103, [1576, 256]);  view_103 = None
    permute_57: "f32[256, 256]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    
    # No stacktrace found for following nodes
    mm_default_25: "f32[1576, 256]" = torch.ops.aten.mm.default(view_104, permute_57)
    add_tensor_25: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_25, primals_110);  mm_default_25 = primals_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_105: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_25, [8, 197, 256]);  add_tensor_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_105: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(cat_5, view_105);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_17 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 197, 1]" = var_mean_17[0]
    getitem_77: "f32[8, 197, 1]" = var_mean_17[1];  var_mean_17 = None
    add_106: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_17: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_65: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_105, getitem_77);  getitem_77 = None
    mul_145: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_17);  sub_65 = None
    mul_146: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_145, primals_111)
    add_107: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_146, primals_112);  mul_146 = primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_106: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_107, [1576, 256]);  add_107 = None
    permute_58: "f32[256, 768]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    addmm_32: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_114, view_106, permute_58);  primals_114 = None
    view_107: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_32, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_147: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_148: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
    erf_9: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
    add_108: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_149: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_147, add_108);  mul_147 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_108: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_149, [1576, 768]);  mul_149 = None
    permute_59: "f32[768, 256]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    
    # No stacktrace found for following nodes
    mm_default_24: "f32[1576, 256]" = torch.ops.aten.mm.default(view_108, permute_59)
    add_tensor_24: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_24, primals_116);  mm_default_24 = primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_109: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_24, [8, 197, 256]);  add_tensor_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_109: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_105, view_109);  add_105 = view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_18 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_78: "f32[8, 197, 1]" = var_mean_18[0]
    getitem_79: "f32[8, 197, 1]" = var_mean_18[1];  var_mean_18 = None
    add_110: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_18: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_66: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_109, getitem_79);  getitem_79 = None
    mul_150: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_18);  sub_66 = None
    mul_151: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_150, primals_117)
    add_111: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_151, primals_118);  mul_151 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_110: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_111, [1576, 256]);  add_111 = None
    permute_60: "f32[256, 768]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_34: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_120, view_110, permute_60);  primals_120 = None
    view_111: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_34, [8, 197, 768]);  addmm_34 = None
    view_112: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.reshape.default(view_111, [8, 197, 3, 4, 64]);  view_111 = None
    permute_61: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_112, [2, 0, 3, 1, 4]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_61);  permute_61 = None
    getitem_80: "f32[8, 4, 197, 64]" = unbind_6[0]
    getitem_81: "f32[8, 4, 197, 64]" = unbind_6[1]
    getitem_82: "f32[8, 4, 197, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_80, getitem_81, getitem_82, None, True)
    getitem_83: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_6[0]
    getitem_84: "f32[8, 4, 224]" = _scaled_dot_product_efficient_attention_6[1]
    getitem_85: "i64[]" = _scaled_dot_product_efficient_attention_6[2]
    getitem_86: "i64[]" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None
    alias_8: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_83)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_62: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_83, [0, 2, 1, 3]);  getitem_83 = None
    view_113: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(permute_62, [8, 197, 256]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_114: "f32[1576, 256]" = torch.ops.aten.reshape.default(view_113, [1576, 256]);  view_113 = None
    permute_63: "f32[256, 256]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    
    # No stacktrace found for following nodes
    mm_default_23: "f32[1576, 256]" = torch.ops.aten.mm.default(view_114, permute_63)
    add_tensor_23: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_23, primals_122);  mm_default_23 = primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_115: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_23, [8, 197, 256]);  add_tensor_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_112: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_109, view_115);  add_109 = view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
    getitem_87: "f32[8, 197, 1]" = var_mean_19[0]
    getitem_88: "f32[8, 197, 1]" = var_mean_19[1];  var_mean_19 = None
    add_113: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_87, 1e-06);  getitem_87 = None
    rsqrt_19: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_67: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_112, getitem_88);  getitem_88 = None
    mul_152: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_19);  sub_67 = None
    mul_153: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_152, primals_123)
    add_114: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_153, primals_124);  mul_153 = primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_116: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_114, [1576, 256]);  add_114 = None
    permute_64: "f32[256, 768]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_36: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_126, view_116, permute_64);  primals_126 = None
    view_117: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_36, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_154: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_117, 0.5)
    mul_155: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_117, 0.7071067811865476);  view_117 = None
    erf_10: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_115: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_156: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_154, add_115);  mul_154 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_118: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_156, [1576, 768]);  mul_156 = None
    permute_65: "f32[768, 256]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    
    # No stacktrace found for following nodes
    mm_default_22: "f32[1576, 256]" = torch.ops.aten.mm.default(view_118, permute_65)
    add_tensor_22: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_22, primals_128);  mm_default_22 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_119: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_22, [8, 197, 256]);  add_tensor_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_116: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_112, view_119);  add_112 = view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_20 = torch.ops.aten.var_mean.correction(add_116, [2], correction = 0, keepdim = True)
    getitem_89: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_90: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_117: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_89, 1e-06);  getitem_89 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_68: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_116, getitem_90);  getitem_90 = None
    mul_157: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_20);  sub_68 = None
    mul_158: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_157, primals_129)
    add_118: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_158, primals_130);  mul_158 = primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_120: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_118, [1576, 256]);  add_118 = None
    permute_66: "f32[256, 768]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_38: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_132, view_120, permute_66);  primals_132 = None
    view_121: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_38, [8, 197, 768]);  addmm_38 = None
    view_122: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.reshape.default(view_121, [8, 197, 3, 4, 64]);  view_121 = None
    permute_67: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_122, [2, 0, 3, 1, 4]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
    getitem_91: "f32[8, 4, 197, 64]" = unbind_7[0]
    getitem_92: "f32[8, 4, 197, 64]" = unbind_7[1]
    getitem_93: "f32[8, 4, 197, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_91, getitem_92, getitem_93, None, True)
    getitem_94: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_7[0]
    getitem_95: "f32[8, 4, 224]" = _scaled_dot_product_efficient_attention_7[1]
    getitem_96: "i64[]" = _scaled_dot_product_efficient_attention_7[2]
    getitem_97: "i64[]" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None
    alias_9: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_68: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_94, [0, 2, 1, 3]);  getitem_94 = None
    view_123: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(permute_68, [8, 197, 256]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_124: "f32[1576, 256]" = torch.ops.aten.reshape.default(view_123, [1576, 256]);  view_123 = None
    permute_69: "f32[256, 256]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    
    # No stacktrace found for following nodes
    mm_default_21: "f32[1576, 256]" = torch.ops.aten.mm.default(view_124, permute_69)
    add_tensor_21: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_21, primals_134);  mm_default_21 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_125: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_21, [8, 197, 256]);  add_tensor_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_119: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_116, view_125);  add_116 = view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_21 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
    getitem_98: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_99: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_120: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-06);  getitem_98 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    sub_69: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_119, getitem_99);  getitem_99 = None
    mul_159: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_21);  sub_69 = None
    mul_160: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_159, primals_135)
    add_121: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_160, primals_136);  mul_160 = primals_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_126: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_121, [1576, 256]);  add_121 = None
    permute_70: "f32[256, 768]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_40: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_138, view_126, permute_70);  primals_138 = None
    view_127: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_40, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_161: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, 0.5)
    mul_162: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, 0.7071067811865476);  view_127 = None
    erf_11: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_162);  mul_162 = None
    add_122: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_163: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_161, add_122);  mul_161 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_128: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_163, [1576, 768]);  mul_163 = None
    permute_71: "f32[768, 256]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    
    # No stacktrace found for following nodes
    mm_default_20: "f32[1576, 256]" = torch.ops.aten.mm.default(view_128, permute_71)
    add_tensor_20: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_20, primals_140);  mm_default_20 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_129: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_20, [8, 197, 256]);  add_tensor_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_123: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_119, view_129);  add_119 = view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    slice_24: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_102, 1, 0, 1)
    clone_36: "f32[8, 1, 128]" = torch.ops.aten.clone.default(slice_24, memory_format = torch.contiguous_format);  slice_24 = None
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_36, [2], correction = 0, keepdim = True)
    getitem_100: "f32[8, 1, 1]" = var_mean_22[0]
    getitem_101: "f32[8, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_124: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-06);  getitem_100 = None
    rsqrt_22: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    sub_70: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(clone_36, getitem_101);  clone_36 = getitem_101 = None
    mul_164: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_22);  sub_70 = None
    mul_165: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_164, primals_141)
    add_125: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_165, primals_142);  mul_165 = None
    mul_166: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_125, 0.5)
    mul_167: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_125, 0.7071067811865476);  add_125 = None
    erf_12: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_126: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_168: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_166, add_126);  mul_166 = add_126 = None
    view_130: "f32[8, 128]" = torch.ops.aten.reshape.default(mul_168, [8, 128]);  mul_168 = None
    permute_72: "f32[128, 256]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_42: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_144, view_130, permute_72);  primals_144 = None
    view_131: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(addmm_42, [8, 1, 256]);  addmm_42 = None
    slice_26: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_123, 1, 0, 1)
    clone_37: "f32[8, 1, 256]" = torch.ops.aten.clone.default(slice_26, memory_format = torch.contiguous_format);  slice_26 = None
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_37, [2], correction = 0, keepdim = True)
    getitem_102: "f32[8, 1, 1]" = var_mean_23[0]
    getitem_103: "f32[8, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_127: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_102, 1e-06);  getitem_102 = None
    rsqrt_23: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_71: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(clone_37, getitem_103);  clone_37 = getitem_103 = None
    mul_169: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_23);  sub_71 = None
    mul_170: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_169, primals_145)
    add_128: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_170, primals_146);  mul_170 = None
    mul_171: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_128, 0.5)
    mul_172: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_128, 0.7071067811865476);  add_128 = None
    erf_13: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
    add_129: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_173: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_171, add_129);  mul_171 = add_129 = None
    view_132: "f32[8, 256]" = torch.ops.aten.reshape.default(mul_173, [8, 256]);  mul_173 = None
    permute_73: "f32[256, 128]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm_43: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_148, view_132, permute_73);  primals_148 = None
    view_133: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(addmm_43, [8, 1, 128]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_28: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_123, 1, 1, 9223372036854775807);  add_123 = None
    cat_6: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_131, slice_28], 1);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_29: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(cat_6, 0, 0, 9223372036854775807)
    slice_30: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_29, 1, 0, 1);  slice_29 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(cat_6, [2], correction = 0, keepdim = True)
    getitem_104: "f32[8, 197, 1]" = var_mean_24[0]
    getitem_105: "f32[8, 197, 1]" = var_mean_24[1];  var_mean_24 = None
    add_130: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_104, 1e-06);  getitem_104 = None
    rsqrt_24: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_72: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_6, getitem_105)
    mul_174: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_24);  sub_72 = None
    mul_175: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_174, primals_149);  mul_174 = None
    add_131: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_175, primals_150);  mul_175 = primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_31: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_131, 0, 0, 9223372036854775807)
    slice_32: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_31, 1, 0, 1);  slice_31 = None
    permute_74: "f32[256, 256]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    view_134: "f32[8, 256]" = torch.ops.aten.reshape.default(slice_32, [8, 256]);  slice_32 = None
    mm_2: "f32[8, 256]" = torch.ops.aten.mm.default(view_134, permute_74)
    view_135: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(mm_2, [8, 1, 256]);  mm_2 = None
    add_132: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(view_135, primals_152);  view_135 = primals_152 = None
    view_136: "f32[8, 1, 4, 64]" = torch.ops.aten.reshape.default(add_132, [8, 1, 4, 64]);  add_132 = None
    permute_75: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_137: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_131, [1576, 256]);  add_131 = None
    permute_76: "f32[256, 256]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    
    # No stacktrace found for following nodes
    mm_default_19: "f32[1576, 256]" = torch.ops.aten.mm.default(view_137, permute_76)
    add_tensor_19: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_19, primals_154);  mm_default_19 = primals_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_138: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_19, [8, 197, 256]);  add_tensor_19 = None
    view_139: "f32[8, 197, 4, 64]" = torch.ops.aten.reshape.default(view_138, [8, 197, 4, 64]);  view_138 = None
    permute_77: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_78: "f32[256, 256]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    
    # No stacktrace found for following nodes
    mm_default_18: "f32[1576, 256]" = torch.ops.aten.mm.default(view_137, permute_78)
    add_tensor_18: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_18, primals_156);  mm_default_18 = primals_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_141: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_18, [8, 197, 256]);  add_tensor_18 = None
    view_142: "f32[8, 197, 4, 64]" = torch.ops.aten.reshape.default(view_141, [8, 197, 4, 64]);  view_141 = None
    permute_79: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_80: "f32[8, 4, 64, 197]" = torch.ops.aten.permute.default(permute_77, [0, 1, 3, 2]);  permute_77 = None
    expand_10: "f32[8, 4, 1, 64]" = torch.ops.aten.expand.default(permute_75, [8, 4, 1, 64]);  permute_75 = None
    view_143: "f32[32, 1, 64]" = torch.ops.aten.reshape.default(expand_10, [32, 1, 64]);  expand_10 = None
    expand_11: "f32[8, 4, 64, 197]" = torch.ops.aten.expand.default(permute_80, [8, 4, 64, 197]);  permute_80 = None
    clone_38: "f32[8, 4, 64, 197]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_144: "f32[32, 64, 197]" = torch.ops.aten.reshape.default(clone_38, [32, 64, 197]);  clone_38 = None
    bmm_4: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_143, view_144)
    view_145: "f32[8, 4, 1, 197]" = torch.ops.aten.reshape.default(bmm_4, [8, 4, 1, 197]);  bmm_4 = None
    mul_176: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(view_145, 0.125);  view_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    amax_2: "f32[8, 4, 1, 1]" = torch.ops.aten.amax.default(mul_176, [-1], True)
    sub_73: "f32[8, 4, 1, 197]" = torch.ops.aten.sub.Tensor(mul_176, amax_2);  mul_176 = amax_2 = None
    exp_2: "f32[8, 4, 1, 197]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
    sum_3: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 4, 1, 197]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_10: "f32[8, 4, 1, 197]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    clone_39: "f32[8, 4, 1, 197]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    expand_12: "f32[8, 4, 1, 197]" = torch.ops.aten.expand.default(clone_39, [8, 4, 1, 197]);  clone_39 = None
    view_146: "f32[32, 1, 197]" = torch.ops.aten.reshape.default(expand_12, [32, 1, 197]);  expand_12 = None
    expand_13: "f32[8, 4, 197, 64]" = torch.ops.aten.expand.default(permute_79, [8, 4, 197, 64]);  permute_79 = None
    clone_40: "f32[8, 4, 197, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_147: "f32[32, 197, 64]" = torch.ops.aten.reshape.default(clone_40, [32, 197, 64]);  clone_40 = None
    bmm_5: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_146, view_147)
    view_148: "f32[8, 4, 1, 64]" = torch.ops.aten.reshape.default(bmm_5, [8, 4, 1, 64]);  bmm_5 = None
    permute_81: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    view_149: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(permute_81, [8, 1, 256]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_150: "f32[8, 256]" = torch.ops.aten.reshape.default(view_149, [8, 256]);  view_149 = None
    permute_82: "f32[256, 256]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    
    # No stacktrace found for following nodes
    mm_default_17: "f32[8, 256]" = torch.ops.aten.mm.default(view_150, permute_82)
    add_tensor_17: "f32[8, 256]" = torch.ops.aten.add.Tensor(mm_default_17, primals_158);  mm_default_17 = primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_151: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(add_tensor_17, [8, 1, 256]);  add_tensor_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_133: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(slice_30, view_151);  slice_30 = view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    var_mean_25 = torch.ops.aten.var_mean.correction(add_133, [2], correction = 0, keepdim = True)
    getitem_106: "f32[8, 1, 1]" = var_mean_25[0]
    getitem_107: "f32[8, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_134: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
    rsqrt_25: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_74: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(add_133, getitem_107);  add_133 = getitem_107 = None
    mul_177: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_25);  sub_74 = None
    mul_178: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_177, primals_159)
    add_135: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_178, primals_160);  mul_178 = None
    mul_179: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_135, 0.5)
    mul_180: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_135, 0.7071067811865476);  add_135 = None
    erf_14: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_180);  mul_180 = None
    add_136: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_181: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_179, add_136);  mul_179 = add_136 = None
    view_152: "f32[8, 256]" = torch.ops.aten.reshape.default(mul_181, [8, 256]);  mul_181 = None
    permute_83: "f32[256, 128]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_47: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_162, view_152, permute_83);  primals_162 = None
    view_153: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(addmm_47, [8, 1, 128]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_35: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_102, 1, 1, 9223372036854775807);  add_102 = None
    cat_7: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_153, slice_35], 1);  view_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    cat_8: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_133, slice_35], 1);  view_133 = slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_38: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(cat_8, 0, 0, 9223372036854775807)
    slice_39: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_38, 1, 0, 1);  slice_38 = None
    var_mean_26 = torch.ops.aten.var_mean.correction(cat_8, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 401, 1]" = var_mean_26[0]
    getitem_109: "f32[8, 401, 1]" = var_mean_26[1];  var_mean_26 = None
    add_137: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_26: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_75: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_8, getitem_109)
    mul_182: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_26);  sub_75 = None
    mul_183: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_182, primals_163);  mul_182 = None
    add_138: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_183, primals_164);  mul_183 = primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_40: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_138, 0, 0, 9223372036854775807)
    slice_41: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_40, 1, 0, 1);  slice_40 = None
    permute_84: "f32[128, 128]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    view_154: "f32[8, 128]" = torch.ops.aten.reshape.default(slice_41, [8, 128]);  slice_41 = None
    mm_3: "f32[8, 128]" = torch.ops.aten.mm.default(view_154, permute_84)
    view_155: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(mm_3, [8, 1, 128]);  mm_3 = None
    add_139: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(view_155, primals_166);  view_155 = primals_166 = None
    view_156: "f32[8, 1, 4, 32]" = torch.ops.aten.reshape.default(add_139, [8, 1, 4, 32]);  add_139 = None
    permute_85: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_157: "f32[3208, 128]" = torch.ops.aten.reshape.default(add_138, [3208, 128]);  add_138 = None
    permute_86: "f32[128, 128]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
    
    # No stacktrace found for following nodes
    mm_default_16: "f32[3208, 128]" = torch.ops.aten.mm.default(view_157, permute_86)
    add_tensor_16: "f32[3208, 128]" = torch.ops.aten.add.Tensor(mm_default_16, primals_168);  mm_default_16 = primals_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_158: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(add_tensor_16, [8, 401, 128]);  add_tensor_16 = None
    view_159: "f32[8, 401, 4, 32]" = torch.ops.aten.reshape.default(view_158, [8, 401, 4, 32]);  view_158 = None
    permute_87: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_88: "f32[128, 128]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    
    # No stacktrace found for following nodes
    mm_default_15: "f32[3208, 128]" = torch.ops.aten.mm.default(view_157, permute_88)
    add_tensor_15: "f32[3208, 128]" = torch.ops.aten.add.Tensor(mm_default_15, primals_170);  mm_default_15 = primals_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_161: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(add_tensor_15, [8, 401, 128]);  add_tensor_15 = None
    view_162: "f32[8, 401, 4, 32]" = torch.ops.aten.reshape.default(view_161, [8, 401, 4, 32]);  view_161 = None
    permute_89: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_90: "f32[8, 4, 32, 401]" = torch.ops.aten.permute.default(permute_87, [0, 1, 3, 2]);  permute_87 = None
    expand_14: "f32[8, 4, 1, 32]" = torch.ops.aten.expand.default(permute_85, [8, 4, 1, 32]);  permute_85 = None
    view_163: "f32[32, 1, 32]" = torch.ops.aten.reshape.default(expand_14, [32, 1, 32]);  expand_14 = None
    expand_15: "f32[8, 4, 32, 401]" = torch.ops.aten.expand.default(permute_90, [8, 4, 32, 401]);  permute_90 = None
    clone_42: "f32[8, 4, 32, 401]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_164: "f32[32, 32, 401]" = torch.ops.aten.reshape.default(clone_42, [32, 32, 401]);  clone_42 = None
    bmm_6: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_163, view_164)
    view_165: "f32[8, 4, 1, 401]" = torch.ops.aten.reshape.default(bmm_6, [8, 4, 1, 401]);  bmm_6 = None
    mul_184: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(view_165, 0.1767766952966369);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    amax_3: "f32[8, 4, 1, 1]" = torch.ops.aten.amax.default(mul_184, [-1], True)
    sub_76: "f32[8, 4, 1, 401]" = torch.ops.aten.sub.Tensor(mul_184, amax_3);  mul_184 = amax_3 = None
    exp_3: "f32[8, 4, 1, 401]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
    sum_4: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 4, 1, 401]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_11: "f32[8, 4, 1, 401]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    clone_43: "f32[8, 4, 1, 401]" = torch.ops.aten.clone.default(div_3);  div_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    expand_16: "f32[8, 4, 1, 401]" = torch.ops.aten.expand.default(clone_43, [8, 4, 1, 401]);  clone_43 = None
    view_166: "f32[32, 1, 401]" = torch.ops.aten.reshape.default(expand_16, [32, 1, 401]);  expand_16 = None
    expand_17: "f32[8, 4, 401, 32]" = torch.ops.aten.expand.default(permute_89, [8, 4, 401, 32]);  permute_89 = None
    clone_44: "f32[8, 4, 401, 32]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_167: "f32[32, 401, 32]" = torch.ops.aten.reshape.default(clone_44, [32, 401, 32]);  clone_44 = None
    bmm_7: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_166, view_167)
    view_168: "f32[8, 4, 1, 32]" = torch.ops.aten.reshape.default(bmm_7, [8, 4, 1, 32]);  bmm_7 = None
    permute_91: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    view_169: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(permute_91, [8, 1, 128]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_170: "f32[8, 128]" = torch.ops.aten.reshape.default(view_169, [8, 128]);  view_169 = None
    permute_92: "f32[128, 128]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    
    # No stacktrace found for following nodes
    mm_default_14: "f32[8, 128]" = torch.ops.aten.mm.default(view_170, permute_92)
    add_tensor_14: "f32[8, 128]" = torch.ops.aten.add.Tensor(mm_default_14, primals_172);  mm_default_14 = primals_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_171: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(add_tensor_14, [8, 1, 128]);  add_tensor_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_140: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(slice_39, view_171);  slice_39 = view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    var_mean_27 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 1, 1]" = var_mean_27[0]
    getitem_111: "f32[8, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_141: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_27: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_77: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(add_140, getitem_111);  add_140 = getitem_111 = None
    mul_185: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_27);  sub_77 = None
    mul_186: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_185, primals_173)
    add_142: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_186, primals_174);  mul_186 = None
    mul_187: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_142, 0.5)
    mul_188: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_142, 0.7071067811865476);  add_142 = None
    erf_15: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_188);  mul_188 = None
    add_143: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_189: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_187, add_143);  mul_187 = add_143 = None
    view_172: "f32[8, 128]" = torch.ops.aten.reshape.default(mul_189, [8, 128]);  mul_189 = None
    permute_93: "f32[128, 256]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_51: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_176, view_172, permute_93);  primals_176 = None
    view_173: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(addmm_51, [8, 1, 256]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    cat_9: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_173, slice_28], 1);  view_173 = slice_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_28 = torch.ops.aten.var_mean.correction(cat_7, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 401, 1]" = var_mean_28[0]
    getitem_113: "f32[8, 401, 1]" = var_mean_28[1];  var_mean_28 = None
    add_144: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
    rsqrt_28: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_78: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_7, getitem_113)
    mul_190: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_28);  sub_78 = None
    mul_191: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_190, primals_177);  mul_190 = None
    add_145: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_191, primals_178);  mul_191 = primals_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_174: "f32[3208, 128]" = torch.ops.aten.reshape.default(add_145, [3208, 128]);  add_145 = None
    permute_94: "f32[128, 384]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    addmm_52: "f32[3208, 384]" = torch.ops.aten.addmm.default(primals_180, view_174, permute_94);  primals_180 = None
    view_175: "f32[8, 401, 384]" = torch.ops.aten.reshape.default(addmm_52, [8, 401, 384]);  addmm_52 = None
    view_176: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.reshape.default(view_175, [8, 401, 3, 4, 32]);  view_175 = None
    permute_95: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.permute.default(view_176, [2, 0, 3, 1, 4]);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_95);  permute_95 = None
    getitem_114: "f32[8, 4, 401, 32]" = unbind_8[0]
    getitem_115: "f32[8, 4, 401, 32]" = unbind_8[1]
    getitem_116: "f32[8, 4, 401, 32]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_114, getitem_115, getitem_116, None, True)
    getitem_117: "f32[8, 4, 401, 32]" = _scaled_dot_product_efficient_attention_8[0]
    getitem_118: "f32[8, 4, 416]" = _scaled_dot_product_efficient_attention_8[1]
    getitem_119: "i64[]" = _scaled_dot_product_efficient_attention_8[2]
    getitem_120: "i64[]" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None
    alias_12: "f32[8, 4, 401, 32]" = torch.ops.aten.alias.default(getitem_117)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_96: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
    view_177: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(permute_96, [8, 401, 128]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_178: "f32[3208, 128]" = torch.ops.aten.reshape.default(view_177, [3208, 128]);  view_177 = None
    permute_97: "f32[128, 128]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    
    # No stacktrace found for following nodes
    mm_default_13: "f32[3208, 128]" = torch.ops.aten.mm.default(view_178, permute_97)
    add_tensor_13: "f32[3208, 128]" = torch.ops.aten.add.Tensor(mm_default_13, primals_182);  mm_default_13 = primals_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_179: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(add_tensor_13, [8, 401, 128]);  add_tensor_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_146: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(cat_7, view_179);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_29 = torch.ops.aten.var_mean.correction(add_146, [2], correction = 0, keepdim = True)
    getitem_121: "f32[8, 401, 1]" = var_mean_29[0]
    getitem_122: "f32[8, 401, 1]" = var_mean_29[1];  var_mean_29 = None
    add_147: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-06);  getitem_121 = None
    rsqrt_29: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_79: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add_146, getitem_122);  getitem_122 = None
    mul_192: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_29);  sub_79 = None
    mul_193: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_192, primals_183)
    add_148: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_193, primals_184);  mul_193 = primals_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_180: "f32[3208, 128]" = torch.ops.aten.reshape.default(add_148, [3208, 128]);  add_148 = None
    permute_98: "f32[128, 384]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    addmm_54: "f32[3208, 384]" = torch.ops.aten.addmm.default(primals_186, view_180, permute_98);  primals_186 = None
    view_181: "f32[8, 401, 384]" = torch.ops.aten.reshape.default(addmm_54, [8, 401, 384])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_194: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_181, 0.5)
    mul_195: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476);  view_181 = None
    erf_16: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_149: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_196: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_194, add_149);  mul_194 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_182: "f32[3208, 384]" = torch.ops.aten.reshape.default(mul_196, [3208, 384]);  mul_196 = None
    permute_99: "f32[384, 128]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    
    # No stacktrace found for following nodes
    mm_default_12: "f32[3208, 128]" = torch.ops.aten.mm.default(view_182, permute_99)
    add_tensor_12: "f32[3208, 128]" = torch.ops.aten.add.Tensor(mm_default_12, primals_188);  mm_default_12 = primals_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_183: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(add_tensor_12, [8, 401, 128]);  add_tensor_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_150: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_146, view_183);  add_146 = view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_30 = torch.ops.aten.var_mean.correction(cat_9, [2], correction = 0, keepdim = True)
    getitem_123: "f32[8, 197, 1]" = var_mean_30[0]
    getitem_124: "f32[8, 197, 1]" = var_mean_30[1];  var_mean_30 = None
    add_151: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_123, 1e-06);  getitem_123 = None
    rsqrt_30: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_80: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_9, getitem_124)
    mul_197: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_30);  sub_80 = None
    mul_198: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_197, primals_189);  mul_197 = None
    add_152: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_198, primals_190);  mul_198 = primals_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_184: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_152, [1576, 256]);  add_152 = None
    permute_100: "f32[256, 768]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
    addmm_56: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_192, view_184, permute_100);  primals_192 = None
    view_185: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_56, [8, 197, 768]);  addmm_56 = None
    view_186: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.reshape.default(view_185, [8, 197, 3, 4, 64]);  view_185 = None
    permute_101: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_186, [2, 0, 3, 1, 4]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_101);  permute_101 = None
    getitem_125: "f32[8, 4, 197, 64]" = unbind_9[0]
    getitem_126: "f32[8, 4, 197, 64]" = unbind_9[1]
    getitem_127: "f32[8, 4, 197, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_125, getitem_126, getitem_127, None, True)
    getitem_128: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_9[0]
    getitem_129: "f32[8, 4, 224]" = _scaled_dot_product_efficient_attention_9[1]
    getitem_130: "i64[]" = _scaled_dot_product_efficient_attention_9[2]
    getitem_131: "i64[]" = _scaled_dot_product_efficient_attention_9[3];  _scaled_dot_product_efficient_attention_9 = None
    alias_13: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_102: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_128, [0, 2, 1, 3]);  getitem_128 = None
    view_187: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(permute_102, [8, 197, 256]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_188: "f32[1576, 256]" = torch.ops.aten.reshape.default(view_187, [1576, 256]);  view_187 = None
    permute_103: "f32[256, 256]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
    
    # No stacktrace found for following nodes
    mm_default_11: "f32[1576, 256]" = torch.ops.aten.mm.default(view_188, permute_103)
    add_tensor_11: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_11, primals_194);  mm_default_11 = primals_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_189: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_11, [8, 197, 256]);  add_tensor_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_153: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(cat_9, view_189);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_31 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 197, 1]" = var_mean_31[0]
    getitem_133: "f32[8, 197, 1]" = var_mean_31[1];  var_mean_31 = None
    add_154: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_31: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_81: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_153, getitem_133);  getitem_133 = None
    mul_199: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_31);  sub_81 = None
    mul_200: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_199, primals_195)
    add_155: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_200, primals_196);  mul_200 = primals_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_190: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_155, [1576, 256]);  add_155 = None
    permute_104: "f32[256, 768]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
    addmm_58: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_198, view_190, permute_104);  primals_198 = None
    view_191: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_58, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_201: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_191, 0.5)
    mul_202: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_191, 0.7071067811865476);  view_191 = None
    erf_17: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_202);  mul_202 = None
    add_156: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_203: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_201, add_156);  mul_201 = add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_192: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_203, [1576, 768]);  mul_203 = None
    permute_105: "f32[768, 256]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    
    # No stacktrace found for following nodes
    mm_default_10: "f32[1576, 256]" = torch.ops.aten.mm.default(view_192, permute_105)
    add_tensor_10: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_10, primals_200);  mm_default_10 = primals_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_193: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_10, [8, 197, 256]);  add_tensor_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_157: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_153, view_193);  add_153 = view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_32 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
    getitem_134: "f32[8, 197, 1]" = var_mean_32[0]
    getitem_135: "f32[8, 197, 1]" = var_mean_32[1];  var_mean_32 = None
    add_158: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_134, 1e-06);  getitem_134 = None
    rsqrt_32: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_82: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_157, getitem_135);  getitem_135 = None
    mul_204: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_32);  sub_82 = None
    mul_205: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_204, primals_201)
    add_159: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_205, primals_202);  mul_205 = primals_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_194: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_159, [1576, 256]);  add_159 = None
    permute_106: "f32[256, 768]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    addmm_60: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_204, view_194, permute_106);  primals_204 = None
    view_195: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_60, [8, 197, 768]);  addmm_60 = None
    view_196: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.reshape.default(view_195, [8, 197, 3, 4, 64]);  view_195 = None
    permute_107: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_196, [2, 0, 3, 1, 4]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_107);  permute_107 = None
    getitem_136: "f32[8, 4, 197, 64]" = unbind_10[0]
    getitem_137: "f32[8, 4, 197, 64]" = unbind_10[1]
    getitem_138: "f32[8, 4, 197, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_136, getitem_137, getitem_138, None, True)
    getitem_139: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_10[0]
    getitem_140: "f32[8, 4, 224]" = _scaled_dot_product_efficient_attention_10[1]
    getitem_141: "i64[]" = _scaled_dot_product_efficient_attention_10[2]
    getitem_142: "i64[]" = _scaled_dot_product_efficient_attention_10[3];  _scaled_dot_product_efficient_attention_10 = None
    alias_14: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_139)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_108: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None
    view_197: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(permute_108, [8, 197, 256]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_198: "f32[1576, 256]" = torch.ops.aten.reshape.default(view_197, [1576, 256]);  view_197 = None
    permute_109: "f32[256, 256]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    
    # No stacktrace found for following nodes
    mm_default_9: "f32[1576, 256]" = torch.ops.aten.mm.default(view_198, permute_109)
    add_tensor_9: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_9, primals_206);  mm_default_9 = primals_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_199: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_9, [8, 197, 256]);  add_tensor_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_160: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_157, view_199);  add_157 = view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_33 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
    getitem_143: "f32[8, 197, 1]" = var_mean_33[0]
    getitem_144: "f32[8, 197, 1]" = var_mean_33[1];  var_mean_33 = None
    add_161: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_143, 1e-06);  getitem_143 = None
    rsqrt_33: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_83: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_160, getitem_144);  getitem_144 = None
    mul_206: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_33);  sub_83 = None
    mul_207: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_206, primals_207)
    add_162: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_207, primals_208);  mul_207 = primals_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_200: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_162, [1576, 256]);  add_162 = None
    permute_110: "f32[256, 768]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    addmm_62: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_210, view_200, permute_110);  primals_210 = None
    view_201: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_62, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_208: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_201, 0.5)
    mul_209: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_201, 0.7071067811865476);  view_201 = None
    erf_18: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_209);  mul_209 = None
    add_163: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_210: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_208, add_163);  mul_208 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_202: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_210, [1576, 768]);  mul_210 = None
    permute_111: "f32[768, 256]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    
    # No stacktrace found for following nodes
    mm_default_8: "f32[1576, 256]" = torch.ops.aten.mm.default(view_202, permute_111)
    add_tensor_8: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_8, primals_212);  mm_default_8 = primals_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_203: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_8, [8, 197, 256]);  add_tensor_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_164: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_160, view_203);  add_160 = view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_34 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
    getitem_145: "f32[8, 197, 1]" = var_mean_34[0]
    getitem_146: "f32[8, 197, 1]" = var_mean_34[1];  var_mean_34 = None
    add_165: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_145, 1e-06);  getitem_145 = None
    rsqrt_34: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_84: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_164, getitem_146);  getitem_146 = None
    mul_211: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_34);  sub_84 = None
    mul_212: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_211, primals_213)
    add_166: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_212, primals_214);  mul_212 = primals_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_204: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_166, [1576, 256]);  add_166 = None
    permute_112: "f32[256, 768]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    addmm_64: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_216, view_204, permute_112);  primals_216 = None
    view_205: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_64, [8, 197, 768]);  addmm_64 = None
    view_206: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.reshape.default(view_205, [8, 197, 3, 4, 64]);  view_205 = None
    permute_113: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_206, [2, 0, 3, 1, 4]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_113);  permute_113 = None
    getitem_147: "f32[8, 4, 197, 64]" = unbind_11[0]
    getitem_148: "f32[8, 4, 197, 64]" = unbind_11[1]
    getitem_149: "f32[8, 4, 197, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_147, getitem_148, getitem_149, None, True)
    getitem_150: "f32[8, 4, 197, 64]" = _scaled_dot_product_efficient_attention_11[0]
    getitem_151: "f32[8, 4, 224]" = _scaled_dot_product_efficient_attention_11[1]
    getitem_152: "i64[]" = _scaled_dot_product_efficient_attention_11[2]
    getitem_153: "i64[]" = _scaled_dot_product_efficient_attention_11[3];  _scaled_dot_product_efficient_attention_11 = None
    alias_15: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_150)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_114: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_150, [0, 2, 1, 3]);  getitem_150 = None
    view_207: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(permute_114, [8, 197, 256]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_208: "f32[1576, 256]" = torch.ops.aten.reshape.default(view_207, [1576, 256]);  view_207 = None
    permute_115: "f32[256, 256]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
    
    # No stacktrace found for following nodes
    mm_default_7: "f32[1576, 256]" = torch.ops.aten.mm.default(view_208, permute_115)
    add_tensor_7: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_7, primals_218);  mm_default_7 = primals_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_209: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_7, [8, 197, 256]);  add_tensor_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_167: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_164, view_209);  add_164 = view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_35 = torch.ops.aten.var_mean.correction(add_167, [2], correction = 0, keepdim = True)
    getitem_154: "f32[8, 197, 1]" = var_mean_35[0]
    getitem_155: "f32[8, 197, 1]" = var_mean_35[1];  var_mean_35 = None
    add_168: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_154, 1e-06);  getitem_154 = None
    rsqrt_35: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_85: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_167, getitem_155);  getitem_155 = None
    mul_213: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_35);  sub_85 = None
    mul_214: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_213, primals_219)
    add_169: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_214, primals_220);  mul_214 = primals_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_210: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_169, [1576, 256]);  add_169 = None
    permute_116: "f32[256, 768]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
    addmm_66: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_222, view_210, permute_116);  primals_222 = None
    view_211: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_66, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_215: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, 0.5)
    mul_216: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, 0.7071067811865476);  view_211 = None
    erf_19: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_216);  mul_216 = None
    add_170: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_217: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_215, add_170);  mul_215 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_212: "f32[1576, 768]" = torch.ops.aten.reshape.default(mul_217, [1576, 768]);  mul_217 = None
    permute_117: "f32[768, 256]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
    
    # No stacktrace found for following nodes
    mm_default_6: "f32[1576, 256]" = torch.ops.aten.mm.default(view_212, permute_117)
    add_tensor_6: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_6, primals_224);  mm_default_6 = primals_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_213: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_6, [8, 197, 256]);  add_tensor_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_171: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_167, view_213);  add_167 = view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    slice_46: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_150, 1, 0, 1)
    clone_58: "f32[8, 1, 128]" = torch.ops.aten.clone.default(slice_46, memory_format = torch.contiguous_format);  slice_46 = None
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_58, [2], correction = 0, keepdim = True)
    getitem_156: "f32[8, 1, 1]" = var_mean_36[0]
    getitem_157: "f32[8, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_172: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-06);  getitem_156 = None
    rsqrt_36: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_86: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(clone_58, getitem_157);  clone_58 = getitem_157 = None
    mul_218: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_36);  sub_86 = None
    mul_219: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_218, primals_225)
    add_173: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_219, primals_226);  mul_219 = None
    mul_220: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_173, 0.5)
    mul_221: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_173, 0.7071067811865476);  add_173 = None
    erf_20: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
    add_174: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_222: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_220, add_174);  mul_220 = add_174 = None
    view_214: "f32[8, 128]" = torch.ops.aten.reshape.default(mul_222, [8, 128]);  mul_222 = None
    permute_118: "f32[128, 256]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
    addmm_68: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_228, view_214, permute_118);  primals_228 = None
    view_215: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(addmm_68, [8, 1, 256]);  addmm_68 = None
    slice_48: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_171, 1, 0, 1)
    clone_59: "f32[8, 1, 256]" = torch.ops.aten.clone.default(slice_48, memory_format = torch.contiguous_format);  slice_48 = None
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_59, [2], correction = 0, keepdim = True)
    getitem_158: "f32[8, 1, 1]" = var_mean_37[0]
    getitem_159: "f32[8, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_175: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
    rsqrt_37: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_87: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(clone_59, getitem_159);  clone_59 = getitem_159 = None
    mul_223: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_37);  sub_87 = None
    mul_224: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_223, primals_229)
    add_176: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_224, primals_230);  mul_224 = None
    mul_225: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_176, 0.5)
    mul_226: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_176, 0.7071067811865476);  add_176 = None
    erf_21: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_226);  mul_226 = None
    add_177: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_227: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_225, add_177);  mul_225 = add_177 = None
    view_216: "f32[8, 256]" = torch.ops.aten.reshape.default(mul_227, [8, 256]);  mul_227 = None
    permute_119: "f32[256, 128]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    addmm_69: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_232, view_216, permute_119);  primals_232 = None
    view_217: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(addmm_69, [8, 1, 128]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_50: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_171, 1, 1, 9223372036854775807);  add_171 = None
    cat_10: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_215, slice_50], 1);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_51: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(cat_10, 0, 0, 9223372036854775807)
    slice_52: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_51, 1, 0, 1);  slice_51 = None
    var_mean_38 = torch.ops.aten.var_mean.correction(cat_10, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 197, 1]" = var_mean_38[0]
    getitem_161: "f32[8, 197, 1]" = var_mean_38[1];  var_mean_38 = None
    add_178: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_38: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_88: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_10, getitem_161)
    mul_228: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_38);  sub_88 = None
    mul_229: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_228, primals_233);  mul_228 = None
    add_179: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_229, primals_234);  mul_229 = primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_53: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_179, 0, 0, 9223372036854775807)
    slice_54: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_53, 1, 0, 1);  slice_53 = None
    permute_120: "f32[256, 256]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    view_218: "f32[8, 256]" = torch.ops.aten.reshape.default(slice_54, [8, 256]);  slice_54 = None
    mm_4: "f32[8, 256]" = torch.ops.aten.mm.default(view_218, permute_120)
    view_219: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(mm_4, [8, 1, 256]);  mm_4 = None
    add_180: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(view_219, primals_236);  view_219 = primals_236 = None
    view_220: "f32[8, 1, 4, 64]" = torch.ops.aten.reshape.default(add_180, [8, 1, 4, 64]);  add_180 = None
    permute_121: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_221: "f32[1576, 256]" = torch.ops.aten.reshape.default(add_179, [1576, 256]);  add_179 = None
    permute_122: "f32[256, 256]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    
    # No stacktrace found for following nodes
    mm_default_5: "f32[1576, 256]" = torch.ops.aten.mm.default(view_221, permute_122)
    add_tensor_5: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_5, primals_238);  mm_default_5 = primals_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_222: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_5, [8, 197, 256]);  add_tensor_5 = None
    view_223: "f32[8, 197, 4, 64]" = torch.ops.aten.reshape.default(view_222, [8, 197, 4, 64]);  view_222 = None
    permute_123: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_124: "f32[256, 256]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    
    # No stacktrace found for following nodes
    mm_default_4: "f32[1576, 256]" = torch.ops.aten.mm.default(view_221, permute_124)
    add_tensor_4: "f32[1576, 256]" = torch.ops.aten.add.Tensor(mm_default_4, primals_240);  mm_default_4 = primals_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_225: "f32[8, 197, 256]" = torch.ops.aten.reshape.default(add_tensor_4, [8, 197, 256]);  add_tensor_4 = None
    view_226: "f32[8, 197, 4, 64]" = torch.ops.aten.reshape.default(view_225, [8, 197, 4, 64]);  view_225 = None
    permute_125: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_126: "f32[8, 4, 64, 197]" = torch.ops.aten.permute.default(permute_123, [0, 1, 3, 2]);  permute_123 = None
    expand_18: "f32[8, 4, 1, 64]" = torch.ops.aten.expand.default(permute_121, [8, 4, 1, 64]);  permute_121 = None
    view_227: "f32[32, 1, 64]" = torch.ops.aten.reshape.default(expand_18, [32, 1, 64]);  expand_18 = None
    expand_19: "f32[8, 4, 64, 197]" = torch.ops.aten.expand.default(permute_126, [8, 4, 64, 197]);  permute_126 = None
    clone_60: "f32[8, 4, 64, 197]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_228: "f32[32, 64, 197]" = torch.ops.aten.reshape.default(clone_60, [32, 64, 197]);  clone_60 = None
    bmm_8: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_227, view_228)
    view_229: "f32[8, 4, 1, 197]" = torch.ops.aten.reshape.default(bmm_8, [8, 4, 1, 197]);  bmm_8 = None
    mul_230: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(view_229, 0.125);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    amax_4: "f32[8, 4, 1, 1]" = torch.ops.aten.amax.default(mul_230, [-1], True)
    sub_89: "f32[8, 4, 1, 197]" = torch.ops.aten.sub.Tensor(mul_230, amax_4);  mul_230 = amax_4 = None
    exp_4: "f32[8, 4, 1, 197]" = torch.ops.aten.exp.default(sub_89);  sub_89 = None
    sum_5: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 4, 1, 197]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_16: "f32[8, 4, 1, 197]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    clone_61: "f32[8, 4, 1, 197]" = torch.ops.aten.clone.default(div_4);  div_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    expand_20: "f32[8, 4, 1, 197]" = torch.ops.aten.expand.default(clone_61, [8, 4, 1, 197]);  clone_61 = None
    view_230: "f32[32, 1, 197]" = torch.ops.aten.reshape.default(expand_20, [32, 1, 197]);  expand_20 = None
    expand_21: "f32[8, 4, 197, 64]" = torch.ops.aten.expand.default(permute_125, [8, 4, 197, 64]);  permute_125 = None
    clone_62: "f32[8, 4, 197, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_231: "f32[32, 197, 64]" = torch.ops.aten.reshape.default(clone_62, [32, 197, 64]);  clone_62 = None
    bmm_9: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_230, view_231)
    view_232: "f32[8, 4, 1, 64]" = torch.ops.aten.reshape.default(bmm_9, [8, 4, 1, 64]);  bmm_9 = None
    permute_127: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
    view_233: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(permute_127, [8, 1, 256]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_234: "f32[8, 256]" = torch.ops.aten.reshape.default(view_233, [8, 256]);  view_233 = None
    permute_128: "f32[256, 256]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    
    # No stacktrace found for following nodes
    mm_default_3: "f32[8, 256]" = torch.ops.aten.mm.default(view_234, permute_128)
    add_tensor_3: "f32[8, 256]" = torch.ops.aten.add.Tensor(mm_default_3, primals_242);  mm_default_3 = primals_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_235: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(add_tensor_3, [8, 1, 256]);  add_tensor_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_181: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(slice_52, view_235);  slice_52 = view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    var_mean_39 = torch.ops.aten.var_mean.correction(add_181, [2], correction = 0, keepdim = True)
    getitem_162: "f32[8, 1, 1]" = var_mean_39[0]
    getitem_163: "f32[8, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_182: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_162, 1e-06);  getitem_162 = None
    rsqrt_39: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_90: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(add_181, getitem_163);  add_181 = getitem_163 = None
    mul_231: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_39);  sub_90 = None
    mul_232: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_231, primals_243)
    add_183: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_232, primals_244);  mul_232 = None
    mul_233: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_183, 0.5)
    mul_234: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_183, 0.7071067811865476);  add_183 = None
    erf_22: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_234);  mul_234 = None
    add_184: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_235: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_233, add_184);  mul_233 = add_184 = None
    view_236: "f32[8, 256]" = torch.ops.aten.reshape.default(mul_235, [8, 256]);  mul_235 = None
    permute_129: "f32[256, 128]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
    addmm_73: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_246, view_236, permute_129);  primals_246 = None
    view_237: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(addmm_73, [8, 1, 128]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_57: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_150, 1, 1, 9223372036854775807);  add_150 = None
    cat_11: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_237, slice_57], 1);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    cat_12: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_217, slice_57], 1);  view_217 = slice_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_60: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(cat_12, 0, 0, 9223372036854775807)
    slice_61: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_60, 1, 0, 1);  slice_60 = None
    var_mean_40 = torch.ops.aten.var_mean.correction(cat_12, [2], correction = 0, keepdim = True)
    getitem_164: "f32[8, 401, 1]" = var_mean_40[0]
    getitem_165: "f32[8, 401, 1]" = var_mean_40[1];  var_mean_40 = None
    add_185: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_164, 1e-06);  getitem_164 = None
    rsqrt_40: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_91: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_12, getitem_165)
    mul_236: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_40);  sub_91 = None
    mul_237: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_236, primals_247);  mul_236 = None
    add_186: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_237, primals_248);  mul_237 = primals_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_62: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_186, 0, 0, 9223372036854775807)
    slice_63: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_62, 1, 0, 1);  slice_62 = None
    permute_130: "f32[128, 128]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
    view_238: "f32[8, 128]" = torch.ops.aten.reshape.default(slice_63, [8, 128]);  slice_63 = None
    mm_5: "f32[8, 128]" = torch.ops.aten.mm.default(view_238, permute_130)
    view_239: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(mm_5, [8, 1, 128]);  mm_5 = None
    add_187: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(view_239, primals_250);  view_239 = primals_250 = None
    view_240: "f32[8, 1, 4, 32]" = torch.ops.aten.reshape.default(add_187, [8, 1, 4, 32]);  add_187 = None
    permute_131: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_241: "f32[3208, 128]" = torch.ops.aten.reshape.default(add_186, [3208, 128]);  add_186 = None
    permute_132: "f32[128, 128]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
    
    # No stacktrace found for following nodes
    mm_default_2: "f32[3208, 128]" = torch.ops.aten.mm.default(view_241, permute_132)
    add_tensor_2: "f32[3208, 128]" = torch.ops.aten.add.Tensor(mm_default_2, primals_252);  mm_default_2 = primals_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_242: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(add_tensor_2, [8, 401, 128]);  add_tensor_2 = None
    view_243: "f32[8, 401, 4, 32]" = torch.ops.aten.reshape.default(view_242, [8, 401, 4, 32]);  view_242 = None
    permute_133: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_134: "f32[128, 128]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
    
    # No stacktrace found for following nodes
    mm_default_1: "f32[3208, 128]" = torch.ops.aten.mm.default(view_241, permute_134)
    add_tensor_1: "f32[3208, 128]" = torch.ops.aten.add.Tensor(mm_default_1, primals_254);  mm_default_1 = primals_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_245: "f32[8, 401, 128]" = torch.ops.aten.reshape.default(add_tensor_1, [8, 401, 128]);  add_tensor_1 = None
    view_246: "f32[8, 401, 4, 32]" = torch.ops.aten.reshape.default(view_245, [8, 401, 4, 32]);  view_245 = None
    permute_135: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_136: "f32[8, 4, 32, 401]" = torch.ops.aten.permute.default(permute_133, [0, 1, 3, 2]);  permute_133 = None
    expand_22: "f32[8, 4, 1, 32]" = torch.ops.aten.expand.default(permute_131, [8, 4, 1, 32]);  permute_131 = None
    view_247: "f32[32, 1, 32]" = torch.ops.aten.reshape.default(expand_22, [32, 1, 32]);  expand_22 = None
    expand_23: "f32[8, 4, 32, 401]" = torch.ops.aten.expand.default(permute_136, [8, 4, 32, 401]);  permute_136 = None
    clone_64: "f32[8, 4, 32, 401]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_248: "f32[32, 32, 401]" = torch.ops.aten.reshape.default(clone_64, [32, 32, 401]);  clone_64 = None
    bmm_10: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_247, view_248)
    view_249: "f32[8, 4, 1, 401]" = torch.ops.aten.reshape.default(bmm_10, [8, 4, 1, 401]);  bmm_10 = None
    mul_238: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(view_249, 0.1767766952966369);  view_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    amax_5: "f32[8, 4, 1, 1]" = torch.ops.aten.amax.default(mul_238, [-1], True)
    sub_92: "f32[8, 4, 1, 401]" = torch.ops.aten.sub.Tensor(mul_238, amax_5);  mul_238 = amax_5 = None
    exp_5: "f32[8, 4, 1, 401]" = torch.ops.aten.exp.default(sub_92);  sub_92 = None
    sum_6: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 4, 1, 401]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_17: "f32[8, 4, 1, 401]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    clone_65: "f32[8, 4, 1, 401]" = torch.ops.aten.clone.default(div_5);  div_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    expand_24: "f32[8, 4, 1, 401]" = torch.ops.aten.expand.default(clone_65, [8, 4, 1, 401]);  clone_65 = None
    view_250: "f32[32, 1, 401]" = torch.ops.aten.reshape.default(expand_24, [32, 1, 401]);  expand_24 = None
    expand_25: "f32[8, 4, 401, 32]" = torch.ops.aten.expand.default(permute_135, [8, 4, 401, 32]);  permute_135 = None
    clone_66: "f32[8, 4, 401, 32]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_251: "f32[32, 401, 32]" = torch.ops.aten.reshape.default(clone_66, [32, 401, 32]);  clone_66 = None
    bmm_11: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_250, view_251)
    view_252: "f32[8, 4, 1, 32]" = torch.ops.aten.reshape.default(bmm_11, [8, 4, 1, 32]);  bmm_11 = None
    permute_137: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    view_253: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(permute_137, [8, 1, 128]);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_254: "f32[8, 128]" = torch.ops.aten.reshape.default(view_253, [8, 128]);  view_253 = None
    permute_138: "f32[128, 128]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    
    # No stacktrace found for following nodes
    mm_default: "f32[8, 128]" = torch.ops.aten.mm.default(view_254, permute_138)
    add_tensor: "f32[8, 128]" = torch.ops.aten.add.Tensor(mm_default, primals_256);  mm_default = primals_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_255: "f32[8, 1, 128]" = torch.ops.aten.reshape.default(add_tensor, [8, 1, 128]);  add_tensor = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_188: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(slice_61, view_255);  slice_61 = view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    var_mean_41 = torch.ops.aten.var_mean.correction(add_188, [2], correction = 0, keepdim = True)
    getitem_166: "f32[8, 1, 1]" = var_mean_41[0]
    getitem_167: "f32[8, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_189: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-06);  getitem_166 = None
    rsqrt_41: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_93: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(add_188, getitem_167);  add_188 = getitem_167 = None
    mul_239: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_41);  sub_93 = None
    mul_240: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_239, primals_257)
    add_190: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_240, primals_258);  mul_240 = None
    mul_241: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_190, 0.5)
    mul_242: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_190, 0.7071067811865476);  add_190 = None
    erf_23: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_242);  mul_242 = None
    add_191: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_243: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_241, add_191);  mul_241 = add_191 = None
    view_256: "f32[8, 128]" = torch.ops.aten.reshape.default(mul_243, [8, 128]);  mul_243 = None
    permute_139: "f32[128, 256]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
    addmm_77: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_260, view_256, permute_139);  primals_260 = None
    view_257: "f32[8, 1, 256]" = torch.ops.aten.reshape.default(addmm_77, [8, 1, 256]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    cat_13: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_257, slice_50], 1);  view_257 = slice_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:451, code: xs = [norm(xs[i]) for i, norm in enumerate(self.norm)]
    var_mean_42 = torch.ops.aten.var_mean.correction(cat_11, [2], correction = 0, keepdim = True)
    getitem_168: "f32[8, 401, 1]" = var_mean_42[0]
    getitem_169: "f32[8, 401, 1]" = var_mean_42[1];  var_mean_42 = None
    add_192: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
    rsqrt_42: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_94: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_11, getitem_169)
    mul_244: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_42);  sub_94 = None
    mul_245: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_244, primals_261);  mul_244 = None
    add_193: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_245, primals_262);  mul_245 = primals_262 = None
    var_mean_43 = torch.ops.aten.var_mean.correction(cat_13, [2], correction = 0, keepdim = True)
    getitem_170: "f32[8, 197, 1]" = var_mean_43[0]
    getitem_171: "f32[8, 197, 1]" = var_mean_43[1];  var_mean_43 = None
    add_194: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-06);  getitem_170 = None
    rsqrt_43: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_95: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_13, getitem_171)
    mul_246: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_43);  sub_95 = None
    mul_247: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_246, primals_263);  mul_246 = None
    add_195: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_247, primals_264);  mul_247 = primals_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:455, code: xs = [x[:, 1:].mean(dim=1) for x in xs] if self.global_pool == 'avg' else [x[:, 0] for x in xs]
    select: "f32[8, 128]" = torch.ops.aten.select.int(add_193, 1, 0);  add_193 = None
    select_1: "f32[8, 256]" = torch.ops.aten.select.int(add_195, 1, 0);  add_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:456, code: xs = [self.head_drop(x) for x in xs]
    clone_68: "f32[8, 128]" = torch.ops.aten.clone.default(select);  select = None
    clone_69: "f32[8, 256]" = torch.ops.aten.clone.default(select_1);  select_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:459, code: return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)
    permute_140: "f32[128, 1000]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
    addmm_78: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_266, clone_68, permute_140);  primals_266 = None
    permute_141: "f32[256, 1000]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
    addmm_79: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_268, clone_69, permute_141);  primals_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:459, code: return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)
    cat_14: "f32[16, 1000]" = torch.ops.aten.cat.default([addmm_78, addmm_79]);  addmm_78 = addmm_79 = None
    view_258: "f32[2, 8, 1000]" = torch.ops.aten.reshape.default(cat_14, [2, 8, 1000]);  cat_14 = None
    mean: "f32[8, 1000]" = torch.ops.aten.mean.dim(view_258, [0]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:459, code: return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)
    permute_142: "f32[1000, 256]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    permute_146: "f32[1000, 128]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    permute_150: "f32[256, 128]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    div_9: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 128);  rsqrt_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    permute_154: "f32[128, 128]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    permute_159: "f32[32, 401, 1]" = torch.ops.aten.permute.default(view_250, [0, 2, 1]);  view_250 = None
    permute_160: "f32[32, 32, 401]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    alias_18: "f32[8, 4, 1, 401]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_161: "f32[32, 32, 1]" = torch.ops.aten.permute.default(view_247, [0, 2, 1]);  view_247 = None
    permute_162: "f32[32, 401, 32]" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_165: "f32[128, 128]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_170: "f32[128, 128]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_177: "f32[128, 128]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    permute_179: "f32[128, 256]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    div_11: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 256);  rsqrt_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    permute_183: "f32[256, 256]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    permute_188: "f32[32, 197, 1]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
    permute_189: "f32[32, 64, 197]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    alias_19: "f32[8, 4, 1, 197]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_190: "f32[32, 64, 1]" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
    permute_191: "f32[32, 197, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_194: "f32[256, 256]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_199: "f32[256, 256]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_206: "f32[256, 256]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    permute_208: "f32[128, 256]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    div_13: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 256);  rsqrt_37 = None
    permute_212: "f32[256, 128]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    div_14: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 128);  rsqrt_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_216: "f32[256, 768]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_220: "f32[768, 256]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_15: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 256);  rsqrt_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_224: "f32[256, 256]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_20: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_230: "f32[768, 256]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_16: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 256);  rsqrt_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_234: "f32[256, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_238: "f32[768, 256]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_17: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 256);  rsqrt_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_242: "f32[256, 256]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_21: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_248: "f32[768, 256]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_18: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 256);  rsqrt_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_252: "f32[256, 768]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_256: "f32[768, 256]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_19: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 256);  rsqrt_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_260: "f32[256, 256]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_22: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_266: "f32[768, 256]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_270: "f32[128, 384]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_274: "f32[384, 128]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_21: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 128);  rsqrt_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_278: "f32[128, 128]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_23: "f32[8, 4, 401, 32]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_284: "f32[384, 128]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    permute_288: "f32[256, 128]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    div_23: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 128);  rsqrt_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    permute_292: "f32[128, 128]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    permute_297: "f32[32, 401, 1]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    permute_298: "f32[32, 32, 401]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    alias_24: "f32[8, 4, 1, 401]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_299: "f32[32, 32, 1]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
    permute_300: "f32[32, 401, 32]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_303: "f32[128, 128]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_308: "f32[128, 128]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_315: "f32[128, 128]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    permute_317: "f32[128, 256]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    div_25: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 256);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    permute_321: "f32[256, 256]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    permute_326: "f32[32, 197, 1]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
    permute_327: "f32[32, 64, 197]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    alias_25: "f32[8, 4, 1, 197]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_328: "f32[32, 64, 1]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    permute_329: "f32[32, 197, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_332: "f32[256, 256]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_337: "f32[256, 256]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_344: "f32[256, 256]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    permute_346: "f32[128, 256]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    div_27: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 256);  rsqrt_23 = None
    permute_350: "f32[256, 128]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    div_28: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 128);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_354: "f32[256, 768]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_358: "f32[768, 256]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_29: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 256);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_362: "f32[256, 256]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_26: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_368: "f32[768, 256]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_30: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 256);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_372: "f32[256, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_376: "f32[768, 256]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_31: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 256);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_380: "f32[256, 256]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_27: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_386: "f32[768, 256]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_32: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 256);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_390: "f32[256, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_394: "f32[768, 256]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_33: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 256);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_398: "f32[256, 256]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_28: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_404: "f32[768, 256]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_408: "f32[128, 384]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_412: "f32[384, 128]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_35: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 128);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_416: "f32[128, 128]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_29: "f32[8, 4, 401, 32]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_422: "f32[384, 128]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    permute_426: "f32[256, 128]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    div_37: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 128);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    permute_430: "f32[128, 128]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    permute_435: "f32[32, 401, 1]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    permute_436: "f32[32, 32, 401]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    alias_30: "f32[8, 4, 1, 401]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_437: "f32[32, 32, 1]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    permute_438: "f32[32, 401, 32]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_441: "f32[128, 128]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_446: "f32[128, 128]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_453: "f32[128, 128]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    permute_455: "f32[128, 256]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    div_39: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 256);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    permute_459: "f32[256, 256]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    permute_464: "f32[32, 197, 1]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    permute_465: "f32[32, 64, 197]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    alias_31: "f32[8, 4, 1, 197]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_466: "f32[32, 64, 1]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    permute_467: "f32[32, 197, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_470: "f32[256, 256]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_475: "f32[256, 256]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_482: "f32[256, 256]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    permute_484: "f32[128, 256]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    div_41: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 256);  rsqrt_9 = None
    permute_488: "f32[256, 128]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    div_42: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 128);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_492: "f32[256, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_496: "f32[768, 256]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_43: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_500: "f32[256, 256]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_32: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_506: "f32[768, 256]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_44: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_510: "f32[256, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_514: "f32[768, 256]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_45: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_518: "f32[256, 256]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_33: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_524: "f32[768, 256]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_46: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 256);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_528: "f32[256, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_532: "f32[768, 256]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_47: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 256);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_536: "f32[256, 256]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_34: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_542: "f32[768, 256]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_48: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 256);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_546: "f32[128, 384]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_550: "f32[384, 128]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_49: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_554: "f32[128, 128]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_35: "f32[8, 4, 401, 32]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_560: "f32[384, 128]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_50: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
    return [mean, primals_5, primals_7, primals_9, primals_15, primals_21, primals_27, primals_33, primals_39, primals_45, primals_51, primals_57, primals_58, primals_61, primals_62, primals_65, primals_75, primals_76, primals_79, primals_89, primals_90, primals_93, primals_99, primals_105, primals_111, primals_117, primals_123, primals_129, primals_135, primals_141, primals_142, primals_145, primals_146, primals_149, primals_159, primals_160, primals_163, primals_173, primals_174, primals_177, primals_183, primals_189, primals_195, primals_201, primals_207, primals_213, primals_219, primals_225, primals_226, primals_229, primals_230, primals_233, primals_243, primals_244, primals_247, primals_257, primals_258, primals_261, primals_263, primals_269, add_46, mul_82, view_6, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, view_10, mul_84, view_12, addmm_2, view_14, mul_89, view_16, getitem_13, getitem_14, getitem_15, getitem_17, getitem_18, getitem_19, view_20, mul_91, view_22, addmm_6, view_24, mul_96, view_26, getitem_24, getitem_25, getitem_26, getitem_28, getitem_29, getitem_30, view_30, mul_98, view_32, addmm_10, view_34, mul_103, view_36, getitem_35, getitem_36, getitem_37, getitem_39, getitem_40, getitem_41, view_40, mul_105, view_42, addmm_14, view_44, mul_110, view_46, mul_115, view_48, cat_2, getitem_49, rsqrt_10, view_50, view_53, view_66, mul_123, view_68, cat_3, cat_4, getitem_53, rsqrt_12, view_70, view_73, view_86, mul_131, view_88, cat_5, getitem_57, rsqrt_14, view_90, getitem_58, getitem_59, getitem_60, getitem_62, getitem_63, getitem_64, view_94, mul_138, view_96, addmm_28, view_98, getitem_68, rsqrt_16, view_100, getitem_69, getitem_70, getitem_71, getitem_73, getitem_74, getitem_75, view_104, mul_145, view_106, addmm_32, view_108, mul_150, view_110, getitem_80, getitem_81, getitem_82, getitem_84, getitem_85, getitem_86, view_114, mul_152, view_116, addmm_36, view_118, mul_157, view_120, getitem_91, getitem_92, getitem_93, getitem_95, getitem_96, getitem_97, view_124, mul_159, view_126, addmm_40, view_128, mul_164, view_130, mul_169, view_132, cat_6, getitem_105, rsqrt_24, view_134, view_137, view_150, mul_177, view_152, cat_7, cat_8, getitem_109, rsqrt_26, view_154, view_157, view_170, mul_185, view_172, cat_9, getitem_113, rsqrt_28, view_174, getitem_114, getitem_115, getitem_116, getitem_118, getitem_119, getitem_120, view_178, mul_192, view_180, addmm_54, view_182, getitem_124, rsqrt_30, view_184, getitem_125, getitem_126, getitem_127, getitem_129, getitem_130, getitem_131, view_188, mul_199, view_190, addmm_58, view_192, mul_204, view_194, getitem_136, getitem_137, getitem_138, getitem_140, getitem_141, getitem_142, view_198, mul_206, view_200, addmm_62, view_202, mul_211, view_204, getitem_147, getitem_148, getitem_149, getitem_151, getitem_152, getitem_153, view_208, mul_213, view_210, addmm_66, view_212, mul_218, view_214, mul_223, view_216, cat_10, getitem_161, rsqrt_38, view_218, view_221, view_234, mul_231, view_236, cat_11, cat_12, getitem_165, rsqrt_40, view_238, view_241, view_254, mul_239, view_256, cat_13, getitem_169, rsqrt_42, getitem_171, rsqrt_43, clone_68, clone_69, permute_142, permute_146, permute_150, div_9, permute_154, permute_159, permute_160, alias_18, permute_161, permute_162, permute_165, permute_170, permute_177, permute_179, div_11, permute_183, permute_188, permute_189, alias_19, permute_190, permute_191, permute_194, permute_199, permute_206, permute_208, div_13, permute_212, div_14, permute_216, permute_220, div_15, permute_224, alias_20, permute_230, div_16, permute_234, permute_238, div_17, permute_242, alias_21, permute_248, div_18, permute_252, permute_256, div_19, permute_260, alias_22, permute_266, permute_270, permute_274, div_21, permute_278, alias_23, permute_284, permute_288, div_23, permute_292, permute_297, permute_298, alias_24, permute_299, permute_300, permute_303, permute_308, permute_315, permute_317, div_25, permute_321, permute_326, permute_327, alias_25, permute_328, permute_329, permute_332, permute_337, permute_344, permute_346, div_27, permute_350, div_28, permute_354, permute_358, div_29, permute_362, alias_26, permute_368, div_30, permute_372, permute_376, div_31, permute_380, alias_27, permute_386, div_32, permute_390, permute_394, div_33, permute_398, alias_28, permute_404, permute_408, permute_412, div_35, permute_416, alias_29, permute_422, permute_426, div_37, permute_430, permute_435, permute_436, alias_30, permute_437, permute_438, permute_441, permute_446, permute_453, permute_455, div_39, permute_459, permute_464, permute_465, alias_31, permute_466, permute_467, permute_470, permute_475, permute_482, permute_484, div_41, permute_488, div_42, permute_492, permute_496, div_43, permute_500, alias_32, permute_506, div_44, permute_510, permute_514, div_45, permute_518, alias_33, permute_524, div_46, permute_528, permute_532, div_47, permute_536, alias_34, permute_542, div_48, permute_546, permute_550, div_49, permute_554, alias_35, permute_560, div_50]
    