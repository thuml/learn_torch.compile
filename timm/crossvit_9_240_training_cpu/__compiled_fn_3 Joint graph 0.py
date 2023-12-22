from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1, 1, 128]"; primals_2: "f32[1, 401, 128]"; primals_3: "f32[1, 1, 256]"; primals_4: "f32[1, 197, 256]"; primals_5: "f32[128, 3, 12, 12]"; primals_6: "f32[128]"; primals_7: "f32[256, 3, 16, 16]"; primals_8: "f32[256]"; primals_9: "f32[128]"; primals_10: "f32[128]"; primals_11: "f32[384, 128]"; primals_12: "f32[384]"; primals_13: "f32[128, 128]"; primals_14: "f32[128]"; primals_15: "f32[128]"; primals_16: "f32[128]"; primals_17: "f32[384, 128]"; primals_18: "f32[384]"; primals_19: "f32[128, 384]"; primals_20: "f32[128]"; primals_21: "f32[256]"; primals_22: "f32[256]"; primals_23: "f32[768, 256]"; primals_24: "f32[768]"; primals_25: "f32[256, 256]"; primals_26: "f32[256]"; primals_27: "f32[256]"; primals_28: "f32[256]"; primals_29: "f32[768, 256]"; primals_30: "f32[768]"; primals_31: "f32[256, 768]"; primals_32: "f32[256]"; primals_33: "f32[256]"; primals_34: "f32[256]"; primals_35: "f32[768, 256]"; primals_36: "f32[768]"; primals_37: "f32[256, 256]"; primals_38: "f32[256]"; primals_39: "f32[256]"; primals_40: "f32[256]"; primals_41: "f32[768, 256]"; primals_42: "f32[768]"; primals_43: "f32[256, 768]"; primals_44: "f32[256]"; primals_45: "f32[256]"; primals_46: "f32[256]"; primals_47: "f32[768, 256]"; primals_48: "f32[768]"; primals_49: "f32[256, 256]"; primals_50: "f32[256]"; primals_51: "f32[256]"; primals_52: "f32[256]"; primals_53: "f32[768, 256]"; primals_54: "f32[768]"; primals_55: "f32[256, 768]"; primals_56: "f32[256]"; primals_57: "f32[128]"; primals_58: "f32[128]"; primals_59: "f32[256, 128]"; primals_60: "f32[256]"; primals_61: "f32[256]"; primals_62: "f32[256]"; primals_63: "f32[128, 256]"; primals_64: "f32[128]"; primals_65: "f32[256]"; primals_66: "f32[256]"; primals_67: "f32[256, 256]"; primals_68: "f32[256]"; primals_69: "f32[256, 256]"; primals_70: "f32[256]"; primals_71: "f32[256, 256]"; primals_72: "f32[256]"; primals_73: "f32[256, 256]"; primals_74: "f32[256]"; primals_75: "f32[256]"; primals_76: "f32[256]"; primals_77: "f32[128, 256]"; primals_78: "f32[128]"; primals_79: "f32[128]"; primals_80: "f32[128]"; primals_81: "f32[128, 128]"; primals_82: "f32[128]"; primals_83: "f32[128, 128]"; primals_84: "f32[128]"; primals_85: "f32[128, 128]"; primals_86: "f32[128]"; primals_87: "f32[128, 128]"; primals_88: "f32[128]"; primals_89: "f32[128]"; primals_90: "f32[128]"; primals_91: "f32[256, 128]"; primals_92: "f32[256]"; primals_93: "f32[128]"; primals_94: "f32[128]"; primals_95: "f32[384, 128]"; primals_96: "f32[384]"; primals_97: "f32[128, 128]"; primals_98: "f32[128]"; primals_99: "f32[128]"; primals_100: "f32[128]"; primals_101: "f32[384, 128]"; primals_102: "f32[384]"; primals_103: "f32[128, 384]"; primals_104: "f32[128]"; primals_105: "f32[256]"; primals_106: "f32[256]"; primals_107: "f32[768, 256]"; primals_108: "f32[768]"; primals_109: "f32[256, 256]"; primals_110: "f32[256]"; primals_111: "f32[256]"; primals_112: "f32[256]"; primals_113: "f32[768, 256]"; primals_114: "f32[768]"; primals_115: "f32[256, 768]"; primals_116: "f32[256]"; primals_117: "f32[256]"; primals_118: "f32[256]"; primals_119: "f32[768, 256]"; primals_120: "f32[768]"; primals_121: "f32[256, 256]"; primals_122: "f32[256]"; primals_123: "f32[256]"; primals_124: "f32[256]"; primals_125: "f32[768, 256]"; primals_126: "f32[768]"; primals_127: "f32[256, 768]"; primals_128: "f32[256]"; primals_129: "f32[256]"; primals_130: "f32[256]"; primals_131: "f32[768, 256]"; primals_132: "f32[768]"; primals_133: "f32[256, 256]"; primals_134: "f32[256]"; primals_135: "f32[256]"; primals_136: "f32[256]"; primals_137: "f32[768, 256]"; primals_138: "f32[768]"; primals_139: "f32[256, 768]"; primals_140: "f32[256]"; primals_141: "f32[128]"; primals_142: "f32[128]"; primals_143: "f32[256, 128]"; primals_144: "f32[256]"; primals_145: "f32[256]"; primals_146: "f32[256]"; primals_147: "f32[128, 256]"; primals_148: "f32[128]"; primals_149: "f32[256]"; primals_150: "f32[256]"; primals_151: "f32[256, 256]"; primals_152: "f32[256]"; primals_153: "f32[256, 256]"; primals_154: "f32[256]"; primals_155: "f32[256, 256]"; primals_156: "f32[256]"; primals_157: "f32[256, 256]"; primals_158: "f32[256]"; primals_159: "f32[256]"; primals_160: "f32[256]"; primals_161: "f32[128, 256]"; primals_162: "f32[128]"; primals_163: "f32[128]"; primals_164: "f32[128]"; primals_165: "f32[128, 128]"; primals_166: "f32[128]"; primals_167: "f32[128, 128]"; primals_168: "f32[128]"; primals_169: "f32[128, 128]"; primals_170: "f32[128]"; primals_171: "f32[128, 128]"; primals_172: "f32[128]"; primals_173: "f32[128]"; primals_174: "f32[128]"; primals_175: "f32[256, 128]"; primals_176: "f32[256]"; primals_177: "f32[128]"; primals_178: "f32[128]"; primals_179: "f32[384, 128]"; primals_180: "f32[384]"; primals_181: "f32[128, 128]"; primals_182: "f32[128]"; primals_183: "f32[128]"; primals_184: "f32[128]"; primals_185: "f32[384, 128]"; primals_186: "f32[384]"; primals_187: "f32[128, 384]"; primals_188: "f32[128]"; primals_189: "f32[256]"; primals_190: "f32[256]"; primals_191: "f32[768, 256]"; primals_192: "f32[768]"; primals_193: "f32[256, 256]"; primals_194: "f32[256]"; primals_195: "f32[256]"; primals_196: "f32[256]"; primals_197: "f32[768, 256]"; primals_198: "f32[768]"; primals_199: "f32[256, 768]"; primals_200: "f32[256]"; primals_201: "f32[256]"; primals_202: "f32[256]"; primals_203: "f32[768, 256]"; primals_204: "f32[768]"; primals_205: "f32[256, 256]"; primals_206: "f32[256]"; primals_207: "f32[256]"; primals_208: "f32[256]"; primals_209: "f32[768, 256]"; primals_210: "f32[768]"; primals_211: "f32[256, 768]"; primals_212: "f32[256]"; primals_213: "f32[256]"; primals_214: "f32[256]"; primals_215: "f32[768, 256]"; primals_216: "f32[768]"; primals_217: "f32[256, 256]"; primals_218: "f32[256]"; primals_219: "f32[256]"; primals_220: "f32[256]"; primals_221: "f32[768, 256]"; primals_222: "f32[768]"; primals_223: "f32[256, 768]"; primals_224: "f32[256]"; primals_225: "f32[128]"; primals_226: "f32[128]"; primals_227: "f32[256, 128]"; primals_228: "f32[256]"; primals_229: "f32[256]"; primals_230: "f32[256]"; primals_231: "f32[128, 256]"; primals_232: "f32[128]"; primals_233: "f32[256]"; primals_234: "f32[256]"; primals_235: "f32[256, 256]"; primals_236: "f32[256]"; primals_237: "f32[256, 256]"; primals_238: "f32[256]"; primals_239: "f32[256, 256]"; primals_240: "f32[256]"; primals_241: "f32[256, 256]"; primals_242: "f32[256]"; primals_243: "f32[256]"; primals_244: "f32[256]"; primals_245: "f32[128, 256]"; primals_246: "f32[128]"; primals_247: "f32[128]"; primals_248: "f32[128]"; primals_249: "f32[128, 128]"; primals_250: "f32[128]"; primals_251: "f32[128, 128]"; primals_252: "f32[128]"; primals_253: "f32[128, 128]"; primals_254: "f32[128]"; primals_255: "f32[128, 128]"; primals_256: "f32[128]"; primals_257: "f32[128]"; primals_258: "f32[128]"; primals_259: "f32[256, 128]"; primals_260: "f32[256]"; primals_261: "f32[128]"; primals_262: "f32[128]"; primals_263: "f32[256]"; primals_264: "f32[256]"; primals_265: "f32[1000, 128]"; primals_266: "f32[1000]"; primals_267: "f32[1000, 256]"; primals_268: "f32[1000]"; primals_269: "f32[8, 3, 240, 240]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:84, code: x = self.proj(x).flatten(2).transpose(1, 2)
    convolution: "f32[8, 128, 20, 20]" = torch.ops.aten.convolution.default(primals_269, primals_5, primals_6, [12, 12], [0, 0], [1, 1], False, [0, 0], 1);  primals_6 = None
    view: "f32[8, 128, 400]" = torch.ops.aten.view.default(convolution, [8, 128, 400]);  convolution = None
    permute: "f32[8, 400, 128]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:440, code: cls_tokens = cls_tokens.expand(B, -1, -1)
    expand: "f32[8, 1, 128]" = torch.ops.aten.expand.default(primals_1, [8, -1, -1]);  primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:441, code: x_ = torch.cat((cls_tokens, x_), dim=1)
    cat: "f32[8, 401, 128]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:443, code: x_ = x_ + pos_embed
    add: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(cat, primals_2);  cat = primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:444, code: x_ = self.pos_drop(x_)
    clone: "f32[8, 401, 128]" = torch.ops.aten.clone.default(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:292, code: x = torch.nn.functional.interpolate(x, size=ss, mode='bicubic', align_corners=False)
    iota: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_1: "i64[8, 1, 1, 1]" = torch.ops.aten.view.default(iota, [8, 1, 1, 1]);  iota = None
    iota_1: "i64[3]" = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_2: "i64[1, 3, 1, 1]" = torch.ops.aten.view.default(iota_1, [1, 3, 1, 1]);  iota_1 = None
    iota_2: "i64[224]" = torch.ops.prims.iota.default(224, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_3: "i64[1, 1, 224, 1]" = torch.ops.aten.view.default(iota_2, [1, 1, 224, 1]);  iota_2 = None
    iota_3: "i64[224]" = torch.ops.prims.iota.default(224, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    view_4: "i64[1, 1, 1, 224]" = torch.ops.aten.view.default(iota_3, [1, 1, 1, 224]);  iota_3 = None
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
    clamp_min: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(sub_4, 0)
    clamp_max: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min, 239);  clamp_min = None
    clamp_min_1: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(sub_5, 0)
    clamp_max_1: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_1, 239);  clamp_min_1 = None
    _unsafe_index: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max, clamp_max_1]);  clamp_max = clamp_max_1 = None
    clamp_min_2: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(sub_4, 0)
    clamp_max_2: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_2, 239);  clamp_min_2 = None
    clamp_min_3: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(convert_element_type, 0)
    clamp_max_3: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_3, 239);  clamp_min_3 = None
    _unsafe_index_1: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_2, clamp_max_3]);  clamp_max_2 = clamp_max_3 = None
    clamp_min_4: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(sub_4, 0)
    clamp_max_4: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_4, 239);  clamp_min_4 = None
    clamp_min_5: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_5, 0)
    clamp_max_5: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_5, 239);  clamp_min_5 = None
    _unsafe_index_2: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_4, clamp_max_5]);  clamp_max_4 = clamp_max_5 = None
    clamp_min_6: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(sub_4, 0);  sub_4 = None
    clamp_max_6: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_6, 239);  clamp_min_6 = None
    clamp_min_7: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_6, 0)
    clamp_max_7: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_7, 239);  clamp_min_7 = None
    _unsafe_index_3: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_6, clamp_max_7]);  clamp_max_6 = clamp_max_7 = None
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
    sub_11: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(2.0, sub_1)
    mul_11: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_11, -0.75)
    sub_12: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_11, -3.75);  mul_11 = None
    mul_12: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_12, sub_11);  sub_12 = None
    add_11: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_12, -6.0);  mul_12 = None
    mul_13: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_11, sub_11);  add_11 = sub_11 = None
    sub_13: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_13, -3.0);  mul_13 = None
    mul_14: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index, sub_7);  _unsafe_index = sub_7 = None
    mul_15: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_1, add_9);  _unsafe_index_1 = add_9 = None
    add_12: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_14, mul_15);  mul_14 = mul_15 = None
    mul_16: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_2, add_10);  _unsafe_index_2 = add_10 = None
    add_13: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_12, mul_16);  add_12 = mul_16 = None
    mul_17: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_3, sub_13);  _unsafe_index_3 = sub_13 = None
    add_14: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_13, mul_17);  add_13 = mul_17 = None
    clamp_min_8: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(convert_element_type_1, 0)
    clamp_max_8: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_8, 239);  clamp_min_8 = None
    clamp_min_9: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(sub_5, 0)
    clamp_max_9: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_9, 239);  clamp_min_9 = None
    _unsafe_index_4: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_8, clamp_max_9]);  clamp_max_8 = clamp_max_9 = None
    clamp_min_10: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(convert_element_type_1, 0)
    clamp_max_10: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_10, 239);  clamp_min_10 = None
    clamp_min_11: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(convert_element_type, 0)
    clamp_max_11: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_11, 239);  clamp_min_11 = None
    _unsafe_index_5: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_10, clamp_max_11]);  clamp_max_10 = clamp_max_11 = None
    clamp_min_12: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(convert_element_type_1, 0)
    clamp_max_12: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_12, 239);  clamp_min_12 = None
    clamp_min_13: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_5, 0)
    clamp_max_13: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_13, 239);  clamp_min_13 = None
    _unsafe_index_6: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_12, clamp_max_13]);  clamp_max_12 = clamp_max_13 = None
    clamp_min_14: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(convert_element_type_1, 0);  convert_element_type_1 = None
    clamp_max_14: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_14, 239);  clamp_min_14 = None
    clamp_min_15: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_6, 0)
    clamp_max_15: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_15, 239);  clamp_min_15 = None
    _unsafe_index_7: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_14, clamp_max_15]);  clamp_max_14 = clamp_max_15 = None
    add_15: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(sub_1, 1.0)
    mul_18: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_15, -0.75)
    sub_14: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_18, -3.75);  mul_18 = None
    mul_19: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_14, add_15);  sub_14 = None
    add_16: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_19, -6.0);  mul_19 = None
    mul_20: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_16, add_15);  add_16 = add_15 = None
    sub_15: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_20, -3.0);  mul_20 = None
    mul_21: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_1, 1.25)
    sub_16: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_21, 2.25);  mul_21 = None
    mul_22: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_16, sub_1);  sub_16 = None
    mul_23: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_22, sub_1);  mul_22 = None
    add_17: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_23, 1);  mul_23 = None
    sub_17: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(1.0, sub_1)
    mul_24: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_17, 1.25)
    sub_18: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_24, 2.25);  mul_24 = None
    mul_25: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_18, sub_17);  sub_18 = None
    mul_26: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_25, sub_17);  mul_25 = sub_17 = None
    add_18: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_26, 1);  mul_26 = None
    sub_19: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(2.0, sub_1)
    mul_27: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_19, -0.75)
    sub_20: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_27, -3.75);  mul_27 = None
    mul_28: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_20, sub_19);  sub_20 = None
    add_19: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_28, -6.0);  mul_28 = None
    mul_29: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_19, sub_19);  add_19 = sub_19 = None
    sub_21: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_29, -3.0);  mul_29 = None
    mul_30: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_4, sub_15);  _unsafe_index_4 = sub_15 = None
    mul_31: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_5, add_17);  _unsafe_index_5 = add_17 = None
    add_20: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_30, mul_31);  mul_30 = mul_31 = None
    mul_32: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_6, add_18);  _unsafe_index_6 = add_18 = None
    add_21: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_20, mul_32);  add_20 = mul_32 = None
    mul_33: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_7, sub_21);  _unsafe_index_7 = sub_21 = None
    add_22: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_21, mul_33);  add_21 = mul_33 = None
    clamp_min_16: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_3, 0)
    clamp_max_16: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_16, 239);  clamp_min_16 = None
    clamp_min_17: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(sub_5, 0)
    clamp_max_17: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_17, 239);  clamp_min_17 = None
    _unsafe_index_8: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_16, clamp_max_17]);  clamp_max_16 = clamp_max_17 = None
    clamp_min_18: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_3, 0)
    clamp_max_18: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_18, 239);  clamp_min_18 = None
    clamp_min_19: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(convert_element_type, 0)
    clamp_max_19: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_19, 239);  clamp_min_19 = None
    _unsafe_index_9: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_18, clamp_max_19]);  clamp_max_18 = clamp_max_19 = None
    clamp_min_20: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_3, 0)
    clamp_max_20: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_20, 239);  clamp_min_20 = None
    clamp_min_21: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_5, 0)
    clamp_max_21: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_21, 239);  clamp_min_21 = None
    _unsafe_index_10: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_20, clamp_max_21]);  clamp_max_20 = clamp_max_21 = None
    clamp_min_22: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_3, 0);  add_3 = None
    clamp_max_22: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_22, 239);  clamp_min_22 = None
    clamp_min_23: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_6, 0)
    clamp_max_23: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_23, 239);  clamp_min_23 = None
    _unsafe_index_11: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_22, clamp_max_23]);  clamp_max_22 = clamp_max_23 = None
    add_23: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(sub_1, 1.0)
    mul_34: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_23, -0.75)
    sub_22: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_34, -3.75);  mul_34 = None
    mul_35: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_22, add_23);  sub_22 = None
    add_24: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_35, -6.0);  mul_35 = None
    mul_36: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_24, add_23);  add_24 = add_23 = None
    sub_23: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_36, -3.0);  mul_36 = None
    mul_37: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_1, 1.25)
    sub_24: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_37, 2.25);  mul_37 = None
    mul_38: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_24, sub_1);  sub_24 = None
    mul_39: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_38, sub_1);  mul_38 = None
    add_25: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_39, 1);  mul_39 = None
    sub_25: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(1.0, sub_1)
    mul_40: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_25, 1.25)
    sub_26: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_40, 2.25);  mul_40 = None
    mul_41: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_26, sub_25);  sub_26 = None
    mul_42: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_41, sub_25);  mul_41 = sub_25 = None
    add_26: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_42, 1);  mul_42 = None
    sub_27: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(2.0, sub_1)
    mul_43: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_27, -0.75)
    sub_28: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_43, -3.75);  mul_43 = None
    mul_44: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_28, sub_27);  sub_28 = None
    add_27: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_44, -6.0);  mul_44 = None
    mul_45: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_27, sub_27);  add_27 = sub_27 = None
    sub_29: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_45, -3.0);  mul_45 = None
    mul_46: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_8, sub_23);  _unsafe_index_8 = sub_23 = None
    mul_47: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_9, add_25);  _unsafe_index_9 = add_25 = None
    add_28: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_46, mul_47);  mul_46 = mul_47 = None
    mul_48: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_10, add_26);  _unsafe_index_10 = add_26 = None
    add_29: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_28, mul_48);  add_28 = mul_48 = None
    mul_49: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_11, sub_29);  _unsafe_index_11 = sub_29 = None
    add_30: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_29, mul_49);  add_29 = mul_49 = None
    clamp_min_24: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_4, 0)
    clamp_max_24: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_24, 239);  clamp_min_24 = None
    clamp_min_25: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(sub_5, 0);  sub_5 = None
    clamp_max_25: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_25, 239);  clamp_min_25 = None
    _unsafe_index_12: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_24, clamp_max_25]);  clamp_max_24 = clamp_max_25 = None
    clamp_min_26: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_4, 0)
    clamp_max_26: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_26, 239);  clamp_min_26 = None
    clamp_min_27: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(convert_element_type, 0);  convert_element_type = None
    clamp_max_27: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_27, 239);  clamp_min_27 = None
    _unsafe_index_13: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_26, clamp_max_27]);  clamp_max_26 = clamp_max_27 = None
    clamp_min_28: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_4, 0)
    clamp_max_28: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_28, 239);  clamp_min_28 = None
    clamp_min_29: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_5, 0);  add_5 = None
    clamp_max_29: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_29, 239);  clamp_min_29 = None
    _unsafe_index_14: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_28, clamp_max_29]);  clamp_max_28 = clamp_max_29 = None
    clamp_min_30: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_min.default(add_4, 0);  add_4 = None
    clamp_max_30: "i64[1, 1, 224, 1]" = torch.ops.aten.clamp_max.default(clamp_min_30, 239);  clamp_min_30 = None
    clamp_min_31: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_min.default(add_6, 0);  add_6 = None
    clamp_max_31: "i64[1, 1, 1, 224]" = torch.ops.aten.clamp_max.default(clamp_min_31, 239);  clamp_min_31 = None
    _unsafe_index_15: "f32[8, 3, 224, 224]" = torch.ops.aten._unsafe_index.Tensor(primals_269, [view_1, view_2, clamp_max_30, clamp_max_31]);  view_1 = view_2 = clamp_max_30 = clamp_max_31 = None
    add_31: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(sub_1, 1.0)
    mul_50: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_31, -0.75)
    sub_30: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_50, -3.75);  mul_50 = None
    mul_51: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_30, add_31);  sub_30 = None
    add_32: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_51, -6.0);  mul_51 = None
    mul_52: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_32, add_31);  add_32 = add_31 = None
    sub_31: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_52, -3.0);  mul_52 = None
    mul_53: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_1, 1.25)
    sub_32: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_53, 2.25);  mul_53 = None
    mul_54: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_32, sub_1);  sub_32 = None
    mul_55: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_54, sub_1);  mul_54 = None
    add_33: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_55, 1);  mul_55 = None
    sub_33: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(1.0, sub_1)
    mul_56: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_33, 1.25)
    sub_34: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_56, 2.25);  mul_56 = None
    mul_57: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_34, sub_33);  sub_34 = None
    mul_58: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(mul_57, sub_33);  mul_57 = sub_33 = None
    add_34: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_58, 1);  mul_58 = None
    sub_35: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(2.0, sub_1);  sub_1 = None
    mul_59: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_35, -0.75)
    sub_36: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_59, -3.75);  mul_59 = None
    mul_60: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(sub_36, sub_35);  sub_36 = None
    add_35: "f32[1, 1, 1, 224]" = torch.ops.aten.add.Tensor(mul_60, -6.0);  mul_60 = None
    mul_61: "f32[1, 1, 1, 224]" = torch.ops.aten.mul.Tensor(add_35, sub_35);  add_35 = sub_35 = None
    sub_37: "f32[1, 1, 1, 224]" = torch.ops.aten.sub.Tensor(mul_61, -3.0);  mul_61 = None
    mul_62: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_12, sub_31);  _unsafe_index_12 = sub_31 = None
    mul_63: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_13, add_33);  _unsafe_index_13 = add_33 = None
    add_36: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(mul_62, mul_63);  mul_62 = mul_63 = None
    mul_64: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_14, add_34);  _unsafe_index_14 = add_34 = None
    add_37: "f32[8, 3, 224, 224]" = torch.ops.aten.add.Tensor(add_36, mul_64);  add_36 = mul_64 = None
    mul_65: "f32[8, 3, 224, 224]" = torch.ops.aten.mul.Tensor(_unsafe_index_15, sub_37);  _unsafe_index_15 = sub_37 = None
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
    view_5: "f32[8, 256, 196]" = torch.ops.aten.view.default(convolution_1, [8, 256, 196]);  convolution_1 = None
    permute_1: "f32[8, 196, 256]" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:440, code: cls_tokens = cls_tokens.expand(B, -1, -1)
    expand_1: "f32[8, 1, 256]" = torch.ops.aten.expand.default(primals_3, [8, -1, -1]);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:441, code: x_ = torch.cat((cls_tokens, x_), dim=1)
    cat_1: "f32[8, 197, 256]" = torch.ops.aten.cat.default([expand_1, permute_1], 1);  expand_1 = permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:443, code: x_ = x_ + pos_embed
    add_47: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(cat_1, primals_4);  cat_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:444, code: x_ = self.pos_drop(x_)
    clone_1: "f32[8, 197, 256]" = torch.ops.aten.clone.default(add_47);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 401, 1]" = var_mean[0]
    getitem_1: "f32[8, 401, 1]" = var_mean[1];  var_mean = None
    add_48: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_46: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(clone, getitem_1)
    mul_82: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt);  sub_46 = None
    mul_83: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_82, primals_9);  mul_82 = None
    add_49: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_83, primals_10);  mul_83 = primals_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_6: "f32[3208, 128]" = torch.ops.aten.view.default(add_49, [3208, 128]);  add_49 = None
    permute_2: "f32[128, 384]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
    addmm: "f32[3208, 384]" = torch.ops.aten.addmm.default(primals_12, view_6, permute_2);  primals_12 = None
    view_7: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm, [8, 401, 384]);  addmm = None
    view_8: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.view.default(view_7, [8, 401, 3, 4, 32]);  view_7 = None
    permute_3: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.permute.default(view_8, [2, 0, 3, 1, 4]);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_3);  permute_3 = None
    getitem_2: "f32[8, 4, 401, 32]" = unbind[0]
    getitem_3: "f32[8, 4, 401, 32]" = unbind[1]
    getitem_4: "f32[8, 4, 401, 32]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_2, getitem_3, getitem_4)
    getitem_5: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention[0]
    getitem_6: "f32[8, 4, 401]" = _scaled_dot_product_flash_attention[1]
    getitem_7: "i32[]" = _scaled_dot_product_flash_attention[2]
    getitem_8: "i32[]" = _scaled_dot_product_flash_attention[3]
    getitem_11: "i64[]" = _scaled_dot_product_flash_attention[6]
    getitem_12: "i64[]" = _scaled_dot_product_flash_attention[7];  _scaled_dot_product_flash_attention = None
    alias: "f32[8, 4, 401, 32]" = torch.ops.aten.alias.default(getitem_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_4: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    view_9: "f32[8, 401, 128]" = torch.ops.aten.view.default(permute_4, [8, 401, 128]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_10: "f32[3208, 128]" = torch.ops.aten.view.default(view_9, [3208, 128]);  view_9 = None
    permute_5: "f32[128, 128]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_1: "f32[3208, 128]" = torch.ops.aten.addmm.default(primals_14, view_10, permute_5);  primals_14 = None
    view_11: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_1, [8, 401, 128]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_2: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_11);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_50: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(clone, clone_2);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_1 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 401, 1]" = var_mean_1[0]
    getitem_15: "f32[8, 401, 1]" = var_mean_1[1];  var_mean_1 = None
    add_51: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_1: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_47: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add_50, getitem_15)
    mul_84: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_1);  sub_47 = None
    mul_85: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_84, primals_15);  mul_84 = None
    add_52: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_85, primals_16);  mul_85 = primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_12: "f32[3208, 128]" = torch.ops.aten.view.default(add_52, [3208, 128]);  add_52 = None
    permute_6: "f32[128, 384]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_2: "f32[3208, 384]" = torch.ops.aten.addmm.default(primals_18, view_12, permute_6);  primals_18 = None
    view_13: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_2, [8, 401, 384]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_13, 0.5)
    mul_87: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476)
    erf: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_53: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_88: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_86, add_53);  mul_86 = add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_3: "f32[8, 401, 384]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_14: "f32[3208, 384]" = torch.ops.aten.view.default(clone_3, [3208, 384]);  clone_3 = None
    permute_7: "f32[384, 128]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    addmm_3: "f32[3208, 128]" = torch.ops.aten.addmm.default(primals_20, view_14, permute_7);  primals_20 = None
    view_15: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_3, [8, 401, 128]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_4: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_15);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_54: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_50, clone_4);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_2 = torch.ops.aten.var_mean.correction(clone_1, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 197, 1]" = var_mean_2[0]
    getitem_17: "f32[8, 197, 1]" = var_mean_2[1];  var_mean_2 = None
    add_55: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_2: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_55);  add_55 = None
    sub_48: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(clone_1, getitem_17)
    mul_89: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_2);  sub_48 = None
    mul_90: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_89, primals_21);  mul_89 = None
    add_56: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_90, primals_22);  mul_90 = primals_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_16: "f32[1576, 256]" = torch.ops.aten.view.default(add_56, [1576, 256]);  add_56 = None
    permute_8: "f32[256, 768]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
    addmm_4: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_24, view_16, permute_8);  primals_24 = None
    view_17: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_4, [8, 197, 768]);  addmm_4 = None
    view_18: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_17, [8, 197, 3, 4, 64]);  view_17 = None
    permute_9: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_18, [2, 0, 3, 1, 4]);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
    getitem_18: "f32[8, 4, 197, 64]" = unbind_1[0]
    getitem_19: "f32[8, 4, 197, 64]" = unbind_1[1]
    getitem_20: "f32[8, 4, 197, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_18, getitem_19, getitem_20)
    getitem_21: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_1[0]
    getitem_22: "f32[8, 4, 197]" = _scaled_dot_product_flash_attention_1[1]
    getitem_23: "i32[]" = _scaled_dot_product_flash_attention_1[2]
    getitem_24: "i32[]" = _scaled_dot_product_flash_attention_1[3]
    getitem_27: "i64[]" = _scaled_dot_product_flash_attention_1[6]
    getitem_28: "i64[]" = _scaled_dot_product_flash_attention_1[7];  _scaled_dot_product_flash_attention_1 = None
    alias_1: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_10: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
    view_19: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_10, [8, 197, 256]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_20: "f32[1576, 256]" = torch.ops.aten.view.default(view_19, [1576, 256]);  view_19 = None
    permute_11: "f32[256, 256]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_5: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_26, view_20, permute_11);  primals_26 = None
    view_21: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_5, [8, 197, 256]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_5: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_21);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_57: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(clone_1, clone_5);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_3 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 197, 1]" = var_mean_3[0]
    getitem_31: "f32[8, 197, 1]" = var_mean_3[1];  var_mean_3 = None
    add_58: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_3: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_49: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_57, getitem_31)
    mul_91: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_3);  sub_49 = None
    mul_92: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_91, primals_27);  mul_91 = None
    add_59: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_92, primals_28);  mul_92 = primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_22: "f32[1576, 256]" = torch.ops.aten.view.default(add_59, [1576, 256]);  add_59 = None
    permute_12: "f32[256, 768]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
    addmm_6: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_30, view_22, permute_12);  primals_30 = None
    view_23: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_6, [8, 197, 768]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_93: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_23, 0.5)
    mul_94: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_23, 0.7071067811865476)
    erf_1: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
    add_60: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_95: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_93, add_60);  mul_93 = add_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_6: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_95);  mul_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_24: "f32[1576, 768]" = torch.ops.aten.view.default(clone_6, [1576, 768]);  clone_6 = None
    permute_13: "f32[768, 256]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    addmm_7: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_32, view_24, permute_13);  primals_32 = None
    view_25: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_7, [8, 197, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_7: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_25);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_61: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_57, clone_7);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_4 = torch.ops.aten.var_mean.correction(add_61, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 197, 1]" = var_mean_4[0]
    getitem_33: "f32[8, 197, 1]" = var_mean_4[1];  var_mean_4 = None
    add_62: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_4: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_62);  add_62 = None
    sub_50: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_61, getitem_33)
    mul_96: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_4);  sub_50 = None
    mul_97: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_96, primals_33);  mul_96 = None
    add_63: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_97, primals_34);  mul_97 = primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_26: "f32[1576, 256]" = torch.ops.aten.view.default(add_63, [1576, 256]);  add_63 = None
    permute_14: "f32[256, 768]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
    addmm_8: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_36, view_26, permute_14);  primals_36 = None
    view_27: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_8, [8, 197, 768]);  addmm_8 = None
    view_28: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_27, [8, 197, 3, 4, 64]);  view_27 = None
    permute_15: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_28, [2, 0, 3, 1, 4]);  view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_15);  permute_15 = None
    getitem_34: "f32[8, 4, 197, 64]" = unbind_2[0]
    getitem_35: "f32[8, 4, 197, 64]" = unbind_2[1]
    getitem_36: "f32[8, 4, 197, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_34, getitem_35, getitem_36)
    getitem_37: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_2[0]
    getitem_38: "f32[8, 4, 197]" = _scaled_dot_product_flash_attention_2[1]
    getitem_39: "i32[]" = _scaled_dot_product_flash_attention_2[2]
    getitem_40: "i32[]" = _scaled_dot_product_flash_attention_2[3]
    getitem_43: "i64[]" = _scaled_dot_product_flash_attention_2[6]
    getitem_44: "i64[]" = _scaled_dot_product_flash_attention_2[7];  _scaled_dot_product_flash_attention_2 = None
    alias_2: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_16: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3]);  getitem_37 = None
    view_29: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_16, [8, 197, 256]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_30: "f32[1576, 256]" = torch.ops.aten.view.default(view_29, [1576, 256]);  view_29 = None
    permute_17: "f32[256, 256]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_9: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_38, view_30, permute_17);  primals_38 = None
    view_31: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_9, [8, 197, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_8: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_31);  view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_64: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_61, clone_8);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_5 = torch.ops.aten.var_mean.correction(add_64, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 197, 1]" = var_mean_5[0]
    getitem_47: "f32[8, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    add_65: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_5: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_65);  add_65 = None
    sub_51: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_64, getitem_47)
    mul_98: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_5);  sub_51 = None
    mul_99: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_98, primals_39);  mul_98 = None
    add_66: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_99, primals_40);  mul_99 = primals_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_32: "f32[1576, 256]" = torch.ops.aten.view.default(add_66, [1576, 256]);  add_66 = None
    permute_18: "f32[256, 768]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
    addmm_10: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_42, view_32, permute_18);  primals_42 = None
    view_33: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_10, [8, 197, 768]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_33, 0.5)
    mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf_2: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_101);  mul_101 = None
    add_67: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_102: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_100, add_67);  mul_100 = add_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_9: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_102);  mul_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_34: "f32[1576, 768]" = torch.ops.aten.view.default(clone_9, [1576, 768]);  clone_9 = None
    permute_19: "f32[768, 256]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    addmm_11: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_44, view_34, permute_19);  primals_44 = None
    view_35: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_11, [8, 197, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_10: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_35);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_68: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_64, clone_10);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_6 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 197, 1]" = var_mean_6[0]
    getitem_49: "f32[8, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    add_69: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_6: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_52: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_68, getitem_49)
    mul_103: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_6);  sub_52 = None
    mul_104: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_103, primals_45);  mul_103 = None
    add_70: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_104, primals_46);  mul_104 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_36: "f32[1576, 256]" = torch.ops.aten.view.default(add_70, [1576, 256]);  add_70 = None
    permute_20: "f32[256, 768]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_12: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_48, view_36, permute_20);  primals_48 = None
    view_37: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_12, [8, 197, 768]);  addmm_12 = None
    view_38: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_37, [8, 197, 3, 4, 64]);  view_37 = None
    permute_21: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_38, [2, 0, 3, 1, 4]);  view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_21);  permute_21 = None
    getitem_50: "f32[8, 4, 197, 64]" = unbind_3[0]
    getitem_51: "f32[8, 4, 197, 64]" = unbind_3[1]
    getitem_52: "f32[8, 4, 197, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_50, getitem_51, getitem_52)
    getitem_53: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_3[0]
    getitem_54: "f32[8, 4, 197]" = _scaled_dot_product_flash_attention_3[1]
    getitem_55: "i32[]" = _scaled_dot_product_flash_attention_3[2]
    getitem_56: "i32[]" = _scaled_dot_product_flash_attention_3[3]
    getitem_59: "i64[]" = _scaled_dot_product_flash_attention_3[6]
    getitem_60: "i64[]" = _scaled_dot_product_flash_attention_3[7];  _scaled_dot_product_flash_attention_3 = None
    alias_3: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_22: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
    view_39: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_22, [8, 197, 256]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_40: "f32[1576, 256]" = torch.ops.aten.view.default(view_39, [1576, 256]);  view_39 = None
    permute_23: "f32[256, 256]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    addmm_13: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_50, view_40, permute_23);  primals_50 = None
    view_41: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_13, [8, 197, 256]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_11: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_41);  view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_71: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_68, clone_11);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_7 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_62: "f32[8, 197, 1]" = var_mean_7[0]
    getitem_63: "f32[8, 197, 1]" = var_mean_7[1];  var_mean_7 = None
    add_72: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
    rsqrt_7: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_53: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_71, getitem_63)
    mul_105: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_7);  sub_53 = None
    mul_106: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_105, primals_51);  mul_105 = None
    add_73: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_106, primals_52);  mul_106 = primals_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_42: "f32[1576, 256]" = torch.ops.aten.view.default(add_73, [1576, 256]);  add_73 = None
    permute_24: "f32[256, 768]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    addmm_14: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_54, view_42, permute_24);  primals_54 = None
    view_43: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_14, [8, 197, 768]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_107: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_43, 0.5)
    mul_108: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
    erf_3: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
    add_74: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_107, add_74);  mul_107 = add_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_12: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_109);  mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_44: "f32[1576, 768]" = torch.ops.aten.view.default(clone_12, [1576, 768]);  clone_12 = None
    permute_25: "f32[768, 256]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_15: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_56, view_44, permute_25);  primals_56 = None
    view_45: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_15, [8, 197, 256]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_13: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_45);  view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_75: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_71, clone_13);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    slice_1: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_54, 0, 0, 9223372036854775807)
    slice_2: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 1);  slice_1 = None
    clone_14: "f32[8, 1, 128]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format)
    var_mean_8 = torch.ops.aten.var_mean.correction(clone_14, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 1, 1]" = var_mean_8[0]
    getitem_65: "f32[8, 1, 1]" = var_mean_8[1];  var_mean_8 = None
    add_76: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_8: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_54: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(clone_14, getitem_65);  clone_14 = None
    mul_110: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_8);  sub_54 = None
    mul_111: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_110, primals_57);  mul_110 = None
    add_77: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_111, primals_58);  mul_111 = primals_58 = None
    mul_112: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_77, 0.5)
    mul_113: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_77, 0.7071067811865476)
    erf_4: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_113);  mul_113 = None
    add_78: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_114: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_112, add_78);  mul_112 = add_78 = None
    view_46: "f32[8, 128]" = torch.ops.aten.view.default(mul_114, [8, 128]);  mul_114 = None
    permute_26: "f32[128, 256]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    addmm_16: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_60, view_46, permute_26);  primals_60 = None
    view_47: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_16, [8, 1, 256]);  addmm_16 = None
    slice_3: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_75, 0, 0, 9223372036854775807)
    slice_4: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_3, 1, 0, 1);  slice_3 = None
    clone_15: "f32[8, 1, 256]" = torch.ops.aten.clone.default(slice_4, memory_format = torch.contiguous_format)
    var_mean_9 = torch.ops.aten.var_mean.correction(clone_15, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 1, 1]" = var_mean_9[0]
    getitem_67: "f32[8, 1, 1]" = var_mean_9[1];  var_mean_9 = None
    add_79: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_9: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_79);  add_79 = None
    sub_55: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(clone_15, getitem_67);  clone_15 = None
    mul_115: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_9);  sub_55 = None
    mul_116: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_115, primals_61);  mul_115 = None
    add_80: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_116, primals_62);  mul_116 = primals_62 = None
    mul_117: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_80, 0.5)
    mul_118: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_80, 0.7071067811865476)
    erf_5: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_81: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_119: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_117, add_81);  mul_117 = add_81 = None
    view_48: "f32[8, 256]" = torch.ops.aten.view.default(mul_119, [8, 256]);  mul_119 = None
    permute_27: "f32[256, 128]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    addmm_17: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_64, view_48, permute_27);  primals_64 = None
    view_49: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_17, [8, 1, 128]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_5: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_75, 0, 0, 9223372036854775807)
    slice_6: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(slice_5, 1, 1, 9223372036854775807);  slice_5 = None
    cat_2: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_47, slice_6], 1);  view_47 = slice_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_7: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(cat_2, 0, 0, 9223372036854775807)
    slice_8: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_7, 1, 0, 1);  slice_7 = None
    var_mean_10 = torch.ops.aten.var_mean.correction(cat_2, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 197, 1]" = var_mean_10[0]
    getitem_69: "f32[8, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    add_82: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_10: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
    sub_56: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_2, getitem_69)
    mul_120: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_10);  sub_56 = None
    mul_121: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_120, primals_65);  mul_120 = None
    add_83: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_121, primals_66);  mul_121 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_9: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_83, 0, 0, 9223372036854775807)
    slice_10: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 1);  slice_9 = None
    permute_28: "f32[256, 256]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    view_50: "f32[8, 256]" = torch.ops.aten.view.default(slice_10, [8, 256]);  slice_10 = None
    mm: "f32[8, 256]" = torch.ops.aten.mm.default(view_50, permute_28)
    view_51: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm, [8, 1, 256]);  mm = None
    add_84: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(view_51, primals_68);  view_51 = primals_68 = None
    view_52: "f32[8, 1, 4, 64]" = torch.ops.aten.view.default(add_84, [8, 1, 4, 64]);  add_84 = None
    permute_29: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_53: "f32[1576, 256]" = torch.ops.aten.view.default(add_83, [1576, 256])
    permute_30: "f32[256, 256]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    addmm_18: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_70, view_53, permute_30);  primals_70 = None
    view_54: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_18, [8, 197, 256]);  addmm_18 = None
    view_55: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_54, [8, 197, 4, 64]);  view_54 = None
    permute_31: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_56: "f32[1576, 256]" = torch.ops.aten.view.default(add_83, [1576, 256]);  add_83 = None
    permute_32: "f32[256, 256]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_19: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_72, view_56, permute_32);  primals_72 = None
    view_57: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_19, [8, 197, 256]);  addmm_19 = None
    view_58: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_57, [8, 197, 4, 64]);  view_57 = None
    permute_33: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_34: "f32[8, 4, 64, 197]" = torch.ops.aten.permute.default(permute_31, [0, 1, 3, 2]);  permute_31 = None
    expand_2: "f32[8, 4, 1, 64]" = torch.ops.aten.expand.default(permute_29, [8, 4, 1, 64]);  permute_29 = None
    view_59: "f32[32, 1, 64]" = torch.ops.aten.view.default(expand_2, [32, 1, 64]);  expand_2 = None
    expand_3: "f32[8, 4, 64, 197]" = torch.ops.aten.expand.default(permute_34, [8, 4, 64, 197]);  permute_34 = None
    clone_16: "f32[8, 4, 64, 197]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_60: "f32[32, 64, 197]" = torch.ops.aten.view.default(clone_16, [32, 64, 197]);  clone_16 = None
    bmm: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_59, view_60)
    view_61: "f32[8, 4, 1, 197]" = torch.ops.aten.view.default(bmm, [8, 4, 1, 197]);  bmm = None
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
    view_62: "f32[32, 1, 197]" = torch.ops.aten.view.default(expand_4, [32, 1, 197]);  expand_4 = None
    expand_5: "f32[8, 4, 197, 64]" = torch.ops.aten.expand.default(permute_33, [8, 4, 197, 64]);  permute_33 = None
    clone_18: "f32[8, 4, 197, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_63: "f32[32, 197, 64]" = torch.ops.aten.view.default(clone_18, [32, 197, 64]);  clone_18 = None
    bmm_1: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_62, view_63)
    view_64: "f32[8, 4, 1, 64]" = torch.ops.aten.view.default(bmm_1, [8, 4, 1, 64]);  bmm_1 = None
    permute_35: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None
    view_65: "f32[8, 1, 256]" = torch.ops.aten.view.default(permute_35, [8, 1, 256]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_66: "f32[8, 256]" = torch.ops.aten.view.default(view_65, [8, 256]);  view_65 = None
    permute_36: "f32[256, 256]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_20: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_74, view_66, permute_36);  primals_74 = None
    view_67: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_20, [8, 1, 256]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    clone_19: "f32[8, 1, 256]" = torch.ops.aten.clone.default(view_67);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_85: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(slice_8, clone_19);  slice_8 = clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    slice_11: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_85, 0, 0, 9223372036854775807);  add_85 = None
    var_mean_11 = torch.ops.aten.var_mean.correction(slice_11, [2], correction = 0, keepdim = True)
    getitem_70: "f32[8, 1, 1]" = var_mean_11[0]
    getitem_71: "f32[8, 1, 1]" = var_mean_11[1];  var_mean_11 = None
    add_86: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
    rsqrt_11: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
    sub_58: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(slice_11, getitem_71)
    mul_123: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_11);  sub_58 = None
    mul_124: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_123, primals_75);  mul_123 = None
    add_87: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_124, primals_76);  mul_124 = primals_76 = None
    mul_125: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_87, 0.5)
    mul_126: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_87, 0.7071067811865476)
    erf_6: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
    add_88: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_127: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_125, add_88);  mul_125 = add_88 = None
    view_68: "f32[8, 256]" = torch.ops.aten.view.default(mul_127, [8, 256]);  mul_127 = None
    permute_37: "f32[256, 128]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm_21: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_78, view_68, permute_37);  primals_78 = None
    view_69: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_21, [8, 1, 128]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_12: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_54, 0, 0, 9223372036854775807)
    slice_13: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(slice_12, 1, 1, 9223372036854775807);  slice_12 = None
    cat_3: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_69, slice_13], 1);  view_69 = slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_14: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_54, 0, 0, 9223372036854775807);  add_54 = None
    slice_15: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(slice_14, 1, 1, 9223372036854775807);  slice_14 = None
    cat_4: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_49, slice_15], 1);  view_49 = slice_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_16: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(cat_4, 0, 0, 9223372036854775807)
    slice_17: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_16, 1, 0, 1);  slice_16 = None
    var_mean_12 = torch.ops.aten.var_mean.correction(cat_4, [2], correction = 0, keepdim = True)
    getitem_72: "f32[8, 401, 1]" = var_mean_12[0]
    getitem_73: "f32[8, 401, 1]" = var_mean_12[1];  var_mean_12 = None
    add_89: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_72, 1e-06);  getitem_72 = None
    rsqrt_12: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
    sub_59: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_4, getitem_73)
    mul_128: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_12);  sub_59 = None
    mul_129: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_128, primals_79);  mul_128 = None
    add_90: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_129, primals_80);  mul_129 = primals_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_18: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_90, 0, 0, 9223372036854775807)
    slice_19: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_18, 1, 0, 1);  slice_18 = None
    permute_38: "f32[128, 128]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    view_70: "f32[8, 128]" = torch.ops.aten.view.default(slice_19, [8, 128]);  slice_19 = None
    mm_1: "f32[8, 128]" = torch.ops.aten.mm.default(view_70, permute_38)
    view_71: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_1, [8, 1, 128]);  mm_1 = None
    add_91: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(view_71, primals_82);  view_71 = primals_82 = None
    view_72: "f32[8, 1, 4, 32]" = torch.ops.aten.view.default(add_91, [8, 1, 4, 32]);  add_91 = None
    permute_39: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_72, [0, 2, 1, 3]);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_73: "f32[3208, 128]" = torch.ops.aten.view.default(add_90, [3208, 128])
    permute_40: "f32[128, 128]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    addmm_22: "f32[3208, 128]" = torch.ops.aten.addmm.default(primals_84, view_73, permute_40);  primals_84 = None
    view_74: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_22, [8, 401, 128]);  addmm_22 = None
    view_75: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_74, [8, 401, 4, 32]);  view_74 = None
    permute_41: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_75, [0, 2, 1, 3]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_76: "f32[3208, 128]" = torch.ops.aten.view.default(add_90, [3208, 128]);  add_90 = None
    permute_42: "f32[128, 128]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_23: "f32[3208, 128]" = torch.ops.aten.addmm.default(primals_86, view_76, permute_42);  primals_86 = None
    view_77: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_23, [8, 401, 128]);  addmm_23 = None
    view_78: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_77, [8, 401, 4, 32]);  view_77 = None
    permute_43: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_78, [0, 2, 1, 3]);  view_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_44: "f32[8, 4, 32, 401]" = torch.ops.aten.permute.default(permute_41, [0, 1, 3, 2]);  permute_41 = None
    expand_6: "f32[8, 4, 1, 32]" = torch.ops.aten.expand.default(permute_39, [8, 4, 1, 32]);  permute_39 = None
    view_79: "f32[32, 1, 32]" = torch.ops.aten.view.default(expand_6, [32, 1, 32]);  expand_6 = None
    expand_7: "f32[8, 4, 32, 401]" = torch.ops.aten.expand.default(permute_44, [8, 4, 32, 401]);  permute_44 = None
    clone_20: "f32[8, 4, 32, 401]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_80: "f32[32, 32, 401]" = torch.ops.aten.view.default(clone_20, [32, 32, 401]);  clone_20 = None
    bmm_2: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_79, view_80)
    view_81: "f32[8, 4, 1, 401]" = torch.ops.aten.view.default(bmm_2, [8, 4, 1, 401]);  bmm_2 = None
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
    view_82: "f32[32, 1, 401]" = torch.ops.aten.view.default(expand_8, [32, 1, 401]);  expand_8 = None
    expand_9: "f32[8, 4, 401, 32]" = torch.ops.aten.expand.default(permute_43, [8, 4, 401, 32]);  permute_43 = None
    clone_22: "f32[8, 4, 401, 32]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_83: "f32[32, 401, 32]" = torch.ops.aten.view.default(clone_22, [32, 401, 32]);  clone_22 = None
    bmm_3: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_82, view_83)
    view_84: "f32[8, 4, 1, 32]" = torch.ops.aten.view.default(bmm_3, [8, 4, 1, 32]);  bmm_3 = None
    permute_45: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    view_85: "f32[8, 1, 128]" = torch.ops.aten.view.default(permute_45, [8, 1, 128]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_86: "f32[8, 128]" = torch.ops.aten.view.default(view_85, [8, 128]);  view_85 = None
    permute_46: "f32[128, 128]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_24: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_88, view_86, permute_46);  primals_88 = None
    view_87: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_24, [8, 1, 128]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    clone_23: "f32[8, 1, 128]" = torch.ops.aten.clone.default(view_87);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_92: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(slice_17, clone_23);  slice_17 = clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    slice_20: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_92, 0, 0, 9223372036854775807);  add_92 = None
    var_mean_13 = torch.ops.aten.var_mean.correction(slice_20, [2], correction = 0, keepdim = True)
    getitem_74: "f32[8, 1, 1]" = var_mean_13[0]
    getitem_75: "f32[8, 1, 1]" = var_mean_13[1];  var_mean_13 = None
    add_93: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_74, 1e-06);  getitem_74 = None
    rsqrt_13: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
    sub_61: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(slice_20, getitem_75)
    mul_131: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_13);  sub_61 = None
    mul_132: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_131, primals_89);  mul_131 = None
    add_94: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_132, primals_90);  mul_132 = primals_90 = None
    mul_133: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_94, 0.5)
    mul_134: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_94, 0.7071067811865476)
    erf_7: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_134);  mul_134 = None
    add_95: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_135: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_133, add_95);  mul_133 = add_95 = None
    view_88: "f32[8, 128]" = torch.ops.aten.view.default(mul_135, [8, 128]);  mul_135 = None
    permute_47: "f32[128, 256]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_25: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_92, view_88, permute_47);  primals_92 = None
    view_89: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_25, [8, 1, 256]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_21: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_75, 0, 0, 9223372036854775807);  add_75 = None
    slice_22: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(slice_21, 1, 1, 9223372036854775807);  slice_21 = None
    cat_5: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_89, slice_22], 1);  view_89 = slice_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_14 = torch.ops.aten.var_mean.correction(cat_3, [2], correction = 0, keepdim = True)
    getitem_76: "f32[8, 401, 1]" = var_mean_14[0]
    getitem_77: "f32[8, 401, 1]" = var_mean_14[1];  var_mean_14 = None
    add_96: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-06);  getitem_76 = None
    rsqrt_14: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_62: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_3, getitem_77)
    mul_136: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_14);  sub_62 = None
    mul_137: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_136, primals_93);  mul_136 = None
    add_97: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_137, primals_94);  mul_137 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_90: "f32[3208, 128]" = torch.ops.aten.view.default(add_97, [3208, 128]);  add_97 = None
    permute_48: "f32[128, 384]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_26: "f32[3208, 384]" = torch.ops.aten.addmm.default(primals_96, view_90, permute_48);  primals_96 = None
    view_91: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_26, [8, 401, 384]);  addmm_26 = None
    view_92: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.view.default(view_91, [8, 401, 3, 4, 32]);  view_91 = None
    permute_49: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.permute.default(view_92, [2, 0, 3, 1, 4]);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_49);  permute_49 = None
    getitem_78: "f32[8, 4, 401, 32]" = unbind_4[0]
    getitem_79: "f32[8, 4, 401, 32]" = unbind_4[1]
    getitem_80: "f32[8, 4, 401, 32]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_78, getitem_79, getitem_80)
    getitem_81: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_4[0]
    getitem_82: "f32[8, 4, 401]" = _scaled_dot_product_flash_attention_4[1]
    getitem_83: "i32[]" = _scaled_dot_product_flash_attention_4[2]
    getitem_84: "i32[]" = _scaled_dot_product_flash_attention_4[3]
    getitem_87: "i64[]" = _scaled_dot_product_flash_attention_4[6]
    getitem_88: "i64[]" = _scaled_dot_product_flash_attention_4[7];  _scaled_dot_product_flash_attention_4 = None
    alias_6: "f32[8, 4, 401, 32]" = torch.ops.aten.alias.default(getitem_81)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_50: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(getitem_81, [0, 2, 1, 3]);  getitem_81 = None
    view_93: "f32[8, 401, 128]" = torch.ops.aten.view.default(permute_50, [8, 401, 128]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_94: "f32[3208, 128]" = torch.ops.aten.view.default(view_93, [3208, 128]);  view_93 = None
    permute_51: "f32[128, 128]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_27: "f32[3208, 128]" = torch.ops.aten.addmm.default(primals_98, view_94, permute_51);  primals_98 = None
    view_95: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_27, [8, 401, 128]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_24: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_95);  view_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_98: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(cat_3, clone_24);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_15 = torch.ops.aten.var_mean.correction(add_98, [2], correction = 0, keepdim = True)
    getitem_90: "f32[8, 401, 1]" = var_mean_15[0]
    getitem_91: "f32[8, 401, 1]" = var_mean_15[1];  var_mean_15 = None
    add_99: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-06);  getitem_90 = None
    rsqrt_15: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_99);  add_99 = None
    sub_63: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add_98, getitem_91)
    mul_138: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_15);  sub_63 = None
    mul_139: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_138, primals_99);  mul_138 = None
    add_100: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_139, primals_100);  mul_139 = primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_96: "f32[3208, 128]" = torch.ops.aten.view.default(add_100, [3208, 128]);  add_100 = None
    permute_52: "f32[128, 384]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_28: "f32[3208, 384]" = torch.ops.aten.addmm.default(primals_102, view_96, permute_52);  primals_102 = None
    view_97: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_28, [8, 401, 384]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_140: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_97, 0.5)
    mul_141: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_97, 0.7071067811865476)
    erf_8: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
    add_101: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_142: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_140, add_101);  mul_140 = add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_25: "f32[8, 401, 384]" = torch.ops.aten.clone.default(mul_142);  mul_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_98: "f32[3208, 384]" = torch.ops.aten.view.default(clone_25, [3208, 384]);  clone_25 = None
    permute_53: "f32[384, 128]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_29: "f32[3208, 128]" = torch.ops.aten.addmm.default(primals_104, view_98, permute_53);  primals_104 = None
    view_99: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_29, [8, 401, 128]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_26: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_99);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_102: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_98, clone_26);  clone_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_16 = torch.ops.aten.var_mean.correction(cat_5, [2], correction = 0, keepdim = True)
    getitem_92: "f32[8, 197, 1]" = var_mean_16[0]
    getitem_93: "f32[8, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    add_103: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_92, 1e-06);  getitem_92 = None
    rsqrt_16: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_103);  add_103 = None
    sub_64: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_5, getitem_93)
    mul_143: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_16);  sub_64 = None
    mul_144: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_143, primals_105);  mul_143 = None
    add_104: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_144, primals_106);  mul_144 = primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_100: "f32[1576, 256]" = torch.ops.aten.view.default(add_104, [1576, 256]);  add_104 = None
    permute_54: "f32[256, 768]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_30: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_108, view_100, permute_54);  primals_108 = None
    view_101: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_30, [8, 197, 768]);  addmm_30 = None
    view_102: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_101, [8, 197, 3, 4, 64]);  view_101 = None
    permute_55: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_102, [2, 0, 3, 1, 4]);  view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_55);  permute_55 = None
    getitem_94: "f32[8, 4, 197, 64]" = unbind_5[0]
    getitem_95: "f32[8, 4, 197, 64]" = unbind_5[1]
    getitem_96: "f32[8, 4, 197, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_94, getitem_95, getitem_96)
    getitem_97: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_5[0]
    getitem_98: "f32[8, 4, 197]" = _scaled_dot_product_flash_attention_5[1]
    getitem_99: "i32[]" = _scaled_dot_product_flash_attention_5[2]
    getitem_100: "i32[]" = _scaled_dot_product_flash_attention_5[3]
    getitem_103: "i64[]" = _scaled_dot_product_flash_attention_5[6]
    getitem_104: "i64[]" = _scaled_dot_product_flash_attention_5[7];  _scaled_dot_product_flash_attention_5 = None
    alias_7: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_56: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_97, [0, 2, 1, 3]);  getitem_97 = None
    view_103: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_56, [8, 197, 256]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_104: "f32[1576, 256]" = torch.ops.aten.view.default(view_103, [1576, 256]);  view_103 = None
    permute_57: "f32[256, 256]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    addmm_31: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_110, view_104, permute_57);  primals_110 = None
    view_105: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_31, [8, 197, 256]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_27: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_105);  view_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_105: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(cat_5, clone_27);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_17 = torch.ops.aten.var_mean.correction(add_105, [2], correction = 0, keepdim = True)
    getitem_106: "f32[8, 197, 1]" = var_mean_17[0]
    getitem_107: "f32[8, 197, 1]" = var_mean_17[1];  var_mean_17 = None
    add_106: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-06);  getitem_106 = None
    rsqrt_17: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_106);  add_106 = None
    sub_65: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_105, getitem_107)
    mul_145: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_17);  sub_65 = None
    mul_146: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_145, primals_111);  mul_145 = None
    add_107: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_146, primals_112);  mul_146 = primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_106: "f32[1576, 256]" = torch.ops.aten.view.default(add_107, [1576, 256]);  add_107 = None
    permute_58: "f32[256, 768]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    addmm_32: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_114, view_106, permute_58);  primals_114 = None
    view_107: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_32, [8, 197, 768]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_147: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_148: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_9: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_148);  mul_148 = None
    add_108: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_149: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_147, add_108);  mul_147 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_28: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_149);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_108: "f32[1576, 768]" = torch.ops.aten.view.default(clone_28, [1576, 768]);  clone_28 = None
    permute_59: "f32[768, 256]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_33: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_116, view_108, permute_59);  primals_116 = None
    view_109: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_33, [8, 197, 256]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_29: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_109);  view_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_109: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_105, clone_29);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_18 = torch.ops.aten.var_mean.correction(add_109, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 197, 1]" = var_mean_18[0]
    getitem_109: "f32[8, 197, 1]" = var_mean_18[1];  var_mean_18 = None
    add_110: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_18: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_110);  add_110 = None
    sub_66: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_109, getitem_109)
    mul_150: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_18);  sub_66 = None
    mul_151: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_150, primals_117);  mul_150 = None
    add_111: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_151, primals_118);  mul_151 = primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_110: "f32[1576, 256]" = torch.ops.aten.view.default(add_111, [1576, 256]);  add_111 = None
    permute_60: "f32[256, 768]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_34: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_120, view_110, permute_60);  primals_120 = None
    view_111: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_34, [8, 197, 768]);  addmm_34 = None
    view_112: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_111, [8, 197, 3, 4, 64]);  view_111 = None
    permute_61: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_112, [2, 0, 3, 1, 4]);  view_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_61);  permute_61 = None
    getitem_110: "f32[8, 4, 197, 64]" = unbind_6[0]
    getitem_111: "f32[8, 4, 197, 64]" = unbind_6[1]
    getitem_112: "f32[8, 4, 197, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_110, getitem_111, getitem_112)
    getitem_113: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_6[0]
    getitem_114: "f32[8, 4, 197]" = _scaled_dot_product_flash_attention_6[1]
    getitem_115: "i32[]" = _scaled_dot_product_flash_attention_6[2]
    getitem_116: "i32[]" = _scaled_dot_product_flash_attention_6[3]
    getitem_119: "i64[]" = _scaled_dot_product_flash_attention_6[6]
    getitem_120: "i64[]" = _scaled_dot_product_flash_attention_6[7];  _scaled_dot_product_flash_attention_6 = None
    alias_8: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_113)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_62: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_113, [0, 2, 1, 3]);  getitem_113 = None
    view_113: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_62, [8, 197, 256]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_114: "f32[1576, 256]" = torch.ops.aten.view.default(view_113, [1576, 256]);  view_113 = None
    permute_63: "f32[256, 256]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_35: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_122, view_114, permute_63);  primals_122 = None
    view_115: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_35, [8, 197, 256]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_30: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_115);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_112: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_109, clone_30);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_112, [2], correction = 0, keepdim = True)
    getitem_122: "f32[8, 197, 1]" = var_mean_19[0]
    getitem_123: "f32[8, 197, 1]" = var_mean_19[1];  var_mean_19 = None
    add_113: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_122, 1e-06);  getitem_122 = None
    rsqrt_19: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_113);  add_113 = None
    sub_67: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_112, getitem_123)
    mul_152: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_19);  sub_67 = None
    mul_153: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_152, primals_123);  mul_152 = None
    add_114: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_153, primals_124);  mul_153 = primals_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_116: "f32[1576, 256]" = torch.ops.aten.view.default(add_114, [1576, 256]);  add_114 = None
    permute_64: "f32[256, 768]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_36: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_126, view_116, permute_64);  primals_126 = None
    view_117: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_36, [8, 197, 768]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_154: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_117, 0.5)
    mul_155: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_117, 0.7071067811865476)
    erf_10: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
    add_115: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_156: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_154, add_115);  mul_154 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_31: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_156);  mul_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_118: "f32[1576, 768]" = torch.ops.aten.view.default(clone_31, [1576, 768]);  clone_31 = None
    permute_65: "f32[768, 256]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    addmm_37: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_128, view_118, permute_65);  primals_128 = None
    view_119: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_37, [8, 197, 256]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_119);  view_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_116: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_112, clone_32);  clone_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_20 = torch.ops.aten.var_mean.correction(add_116, [2], correction = 0, keepdim = True)
    getitem_124: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_125: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_117: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_124, 1e-06);  getitem_124 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_117);  add_117 = None
    sub_68: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_116, getitem_125)
    mul_157: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_20);  sub_68 = None
    mul_158: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_157, primals_129);  mul_157 = None
    add_118: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_158, primals_130);  mul_158 = primals_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_120: "f32[1576, 256]" = torch.ops.aten.view.default(add_118, [1576, 256]);  add_118 = None
    permute_66: "f32[256, 768]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_38: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_132, view_120, permute_66);  primals_132 = None
    view_121: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_38, [8, 197, 768]);  addmm_38 = None
    view_122: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_121, [8, 197, 3, 4, 64]);  view_121 = None
    permute_67: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_122, [2, 0, 3, 1, 4]);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_67);  permute_67 = None
    getitem_126: "f32[8, 4, 197, 64]" = unbind_7[0]
    getitem_127: "f32[8, 4, 197, 64]" = unbind_7[1]
    getitem_128: "f32[8, 4, 197, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_126, getitem_127, getitem_128)
    getitem_129: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_7[0]
    getitem_130: "f32[8, 4, 197]" = _scaled_dot_product_flash_attention_7[1]
    getitem_131: "i32[]" = _scaled_dot_product_flash_attention_7[2]
    getitem_132: "i32[]" = _scaled_dot_product_flash_attention_7[3]
    getitem_135: "i64[]" = _scaled_dot_product_flash_attention_7[6]
    getitem_136: "i64[]" = _scaled_dot_product_flash_attention_7[7];  _scaled_dot_product_flash_attention_7 = None
    alias_9: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_129)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_68: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_129, [0, 2, 1, 3]);  getitem_129 = None
    view_123: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_68, [8, 197, 256]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_124: "f32[1576, 256]" = torch.ops.aten.view.default(view_123, [1576, 256]);  view_123 = None
    permute_69: "f32[256, 256]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    addmm_39: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_134, view_124, permute_69);  primals_134 = None
    view_125: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_39, [8, 197, 256]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_33: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_125);  view_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_119: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_116, clone_33);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_21 = torch.ops.aten.var_mean.correction(add_119, [2], correction = 0, keepdim = True)
    getitem_138: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_139: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_120: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-06);  getitem_138 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_120);  add_120 = None
    sub_69: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_119, getitem_139)
    mul_159: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_69, rsqrt_21);  sub_69 = None
    mul_160: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_159, primals_135);  mul_159 = None
    add_121: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_160, primals_136);  mul_160 = primals_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_126: "f32[1576, 256]" = torch.ops.aten.view.default(add_121, [1576, 256]);  add_121 = None
    permute_70: "f32[256, 768]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_40: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_138, view_126, permute_70);  primals_138 = None
    view_127: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_40, [8, 197, 768]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_161: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, 0.5)
    mul_162: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, 0.7071067811865476)
    erf_11: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_162);  mul_162 = None
    add_122: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_163: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_161, add_122);  mul_161 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_34: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_163);  mul_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_128: "f32[1576, 768]" = torch.ops.aten.view.default(clone_34, [1576, 768]);  clone_34 = None
    permute_71: "f32[768, 256]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    addmm_41: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_140, view_128, permute_71);  primals_140 = None
    view_129: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_41, [8, 197, 256]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_35: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_129);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_123: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_119, clone_35);  clone_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    slice_23: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_102, 0, 0, 9223372036854775807)
    slice_24: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_23, 1, 0, 1);  slice_23 = None
    clone_36: "f32[8, 1, 128]" = torch.ops.aten.clone.default(slice_24, memory_format = torch.contiguous_format)
    var_mean_22 = torch.ops.aten.var_mean.correction(clone_36, [2], correction = 0, keepdim = True)
    getitem_140: "f32[8, 1, 1]" = var_mean_22[0]
    getitem_141: "f32[8, 1, 1]" = var_mean_22[1];  var_mean_22 = None
    add_124: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-06);  getitem_140 = None
    rsqrt_22: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_124);  add_124 = None
    sub_70: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(clone_36, getitem_141);  clone_36 = None
    mul_164: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_22);  sub_70 = None
    mul_165: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_164, primals_141);  mul_164 = None
    add_125: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_165, primals_142);  mul_165 = primals_142 = None
    mul_166: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_125, 0.5)
    mul_167: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_125, 0.7071067811865476)
    erf_12: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_167);  mul_167 = None
    add_126: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_168: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_166, add_126);  mul_166 = add_126 = None
    view_130: "f32[8, 128]" = torch.ops.aten.view.default(mul_168, [8, 128]);  mul_168 = None
    permute_72: "f32[128, 256]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_42: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_144, view_130, permute_72);  primals_144 = None
    view_131: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_42, [8, 1, 256]);  addmm_42 = None
    slice_25: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_123, 0, 0, 9223372036854775807)
    slice_26: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_25, 1, 0, 1);  slice_25 = None
    clone_37: "f32[8, 1, 256]" = torch.ops.aten.clone.default(slice_26, memory_format = torch.contiguous_format)
    var_mean_23 = torch.ops.aten.var_mean.correction(clone_37, [2], correction = 0, keepdim = True)
    getitem_142: "f32[8, 1, 1]" = var_mean_23[0]
    getitem_143: "f32[8, 1, 1]" = var_mean_23[1];  var_mean_23 = None
    add_127: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
    rsqrt_23: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_127);  add_127 = None
    sub_71: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(clone_37, getitem_143);  clone_37 = None
    mul_169: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_23);  sub_71 = None
    mul_170: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_169, primals_145);  mul_169 = None
    add_128: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_170, primals_146);  mul_170 = primals_146 = None
    mul_171: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_128, 0.5)
    mul_172: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_128, 0.7071067811865476)
    erf_13: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
    add_129: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_173: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_171, add_129);  mul_171 = add_129 = None
    view_132: "f32[8, 256]" = torch.ops.aten.view.default(mul_173, [8, 256]);  mul_173 = None
    permute_73: "f32[256, 128]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm_43: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_148, view_132, permute_73);  primals_148 = None
    view_133: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_43, [8, 1, 128]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_27: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_123, 0, 0, 9223372036854775807)
    slice_28: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(slice_27, 1, 1, 9223372036854775807);  slice_27 = None
    cat_6: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_131, slice_28], 1);  view_131 = slice_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_29: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(cat_6, 0, 0, 9223372036854775807)
    slice_30: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_29, 1, 0, 1);  slice_29 = None
    var_mean_24 = torch.ops.aten.var_mean.correction(cat_6, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 197, 1]" = var_mean_24[0]
    getitem_145: "f32[8, 197, 1]" = var_mean_24[1];  var_mean_24 = None
    add_130: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_24: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
    sub_72: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_6, getitem_145)
    mul_174: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_24);  sub_72 = None
    mul_175: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_174, primals_149);  mul_174 = None
    add_131: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_175, primals_150);  mul_175 = primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_31: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_131, 0, 0, 9223372036854775807)
    slice_32: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_31, 1, 0, 1);  slice_31 = None
    permute_74: "f32[256, 256]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    view_134: "f32[8, 256]" = torch.ops.aten.view.default(slice_32, [8, 256]);  slice_32 = None
    mm_2: "f32[8, 256]" = torch.ops.aten.mm.default(view_134, permute_74)
    view_135: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_2, [8, 1, 256]);  mm_2 = None
    add_132: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(view_135, primals_152);  view_135 = primals_152 = None
    view_136: "f32[8, 1, 4, 64]" = torch.ops.aten.view.default(add_132, [8, 1, 4, 64]);  add_132 = None
    permute_75: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_137: "f32[1576, 256]" = torch.ops.aten.view.default(add_131, [1576, 256])
    permute_76: "f32[256, 256]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    addmm_44: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_154, view_137, permute_76);  primals_154 = None
    view_138: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_44, [8, 197, 256]);  addmm_44 = None
    view_139: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_138, [8, 197, 4, 64]);  view_138 = None
    permute_77: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_140: "f32[1576, 256]" = torch.ops.aten.view.default(add_131, [1576, 256]);  add_131 = None
    permute_78: "f32[256, 256]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_45: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_156, view_140, permute_78);  primals_156 = None
    view_141: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_45, [8, 197, 256]);  addmm_45 = None
    view_142: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_141, [8, 197, 4, 64]);  view_141 = None
    permute_79: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1, 3]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_80: "f32[8, 4, 64, 197]" = torch.ops.aten.permute.default(permute_77, [0, 1, 3, 2]);  permute_77 = None
    expand_10: "f32[8, 4, 1, 64]" = torch.ops.aten.expand.default(permute_75, [8, 4, 1, 64]);  permute_75 = None
    view_143: "f32[32, 1, 64]" = torch.ops.aten.view.default(expand_10, [32, 1, 64]);  expand_10 = None
    expand_11: "f32[8, 4, 64, 197]" = torch.ops.aten.expand.default(permute_80, [8, 4, 64, 197]);  permute_80 = None
    clone_38: "f32[8, 4, 64, 197]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_144: "f32[32, 64, 197]" = torch.ops.aten.view.default(clone_38, [32, 64, 197]);  clone_38 = None
    bmm_4: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_143, view_144)
    view_145: "f32[8, 4, 1, 197]" = torch.ops.aten.view.default(bmm_4, [8, 4, 1, 197]);  bmm_4 = None
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
    view_146: "f32[32, 1, 197]" = torch.ops.aten.view.default(expand_12, [32, 1, 197]);  expand_12 = None
    expand_13: "f32[8, 4, 197, 64]" = torch.ops.aten.expand.default(permute_79, [8, 4, 197, 64]);  permute_79 = None
    clone_40: "f32[8, 4, 197, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_147: "f32[32, 197, 64]" = torch.ops.aten.view.default(clone_40, [32, 197, 64]);  clone_40 = None
    bmm_5: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_146, view_147)
    view_148: "f32[8, 4, 1, 64]" = torch.ops.aten.view.default(bmm_5, [8, 4, 1, 64]);  bmm_5 = None
    permute_81: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None
    view_149: "f32[8, 1, 256]" = torch.ops.aten.view.default(permute_81, [8, 1, 256]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_150: "f32[8, 256]" = torch.ops.aten.view.default(view_149, [8, 256]);  view_149 = None
    permute_82: "f32[256, 256]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_46: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_158, view_150, permute_82);  primals_158 = None
    view_151: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_46, [8, 1, 256]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    clone_41: "f32[8, 1, 256]" = torch.ops.aten.clone.default(view_151);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_133: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(slice_30, clone_41);  slice_30 = clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    slice_33: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_133, 0, 0, 9223372036854775807);  add_133 = None
    var_mean_25 = torch.ops.aten.var_mean.correction(slice_33, [2], correction = 0, keepdim = True)
    getitem_146: "f32[8, 1, 1]" = var_mean_25[0]
    getitem_147: "f32[8, 1, 1]" = var_mean_25[1];  var_mean_25 = None
    add_134: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-06);  getitem_146 = None
    rsqrt_25: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_134);  add_134 = None
    sub_74: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(slice_33, getitem_147)
    mul_177: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_25);  sub_74 = None
    mul_178: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_177, primals_159);  mul_177 = None
    add_135: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_178, primals_160);  mul_178 = primals_160 = None
    mul_179: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_135, 0.5)
    mul_180: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_135, 0.7071067811865476)
    erf_14: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_180);  mul_180 = None
    add_136: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_181: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_179, add_136);  mul_179 = add_136 = None
    view_152: "f32[8, 256]" = torch.ops.aten.view.default(mul_181, [8, 256]);  mul_181 = None
    permute_83: "f32[256, 128]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_47: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_162, view_152, permute_83);  primals_162 = None
    view_153: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_47, [8, 1, 128]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_34: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_102, 0, 0, 9223372036854775807)
    slice_35: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(slice_34, 1, 1, 9223372036854775807);  slice_34 = None
    cat_7: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_153, slice_35], 1);  view_153 = slice_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_36: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_102, 0, 0, 9223372036854775807);  add_102 = None
    slice_37: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(slice_36, 1, 1, 9223372036854775807);  slice_36 = None
    cat_8: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_133, slice_37], 1);  view_133 = slice_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_38: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(cat_8, 0, 0, 9223372036854775807)
    slice_39: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_38, 1, 0, 1);  slice_38 = None
    var_mean_26 = torch.ops.aten.var_mean.correction(cat_8, [2], correction = 0, keepdim = True)
    getitem_148: "f32[8, 401, 1]" = var_mean_26[0]
    getitem_149: "f32[8, 401, 1]" = var_mean_26[1];  var_mean_26 = None
    add_137: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-06);  getitem_148 = None
    rsqrt_26: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
    sub_75: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_8, getitem_149)
    mul_182: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_26);  sub_75 = None
    mul_183: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_182, primals_163);  mul_182 = None
    add_138: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_183, primals_164);  mul_183 = primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_40: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_138, 0, 0, 9223372036854775807)
    slice_41: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_40, 1, 0, 1);  slice_40 = None
    permute_84: "f32[128, 128]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    view_154: "f32[8, 128]" = torch.ops.aten.view.default(slice_41, [8, 128]);  slice_41 = None
    mm_3: "f32[8, 128]" = torch.ops.aten.mm.default(view_154, permute_84)
    view_155: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_3, [8, 1, 128]);  mm_3 = None
    add_139: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(view_155, primals_166);  view_155 = primals_166 = None
    view_156: "f32[8, 1, 4, 32]" = torch.ops.aten.view.default(add_139, [8, 1, 4, 32]);  add_139 = None
    permute_85: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_157: "f32[3208, 128]" = torch.ops.aten.view.default(add_138, [3208, 128])
    permute_86: "f32[128, 128]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
    addmm_48: "f32[3208, 128]" = torch.ops.aten.addmm.default(primals_168, view_157, permute_86);  primals_168 = None
    view_158: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_48, [8, 401, 128]);  addmm_48 = None
    view_159: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_158, [8, 401, 4, 32]);  view_158 = None
    permute_87: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_160: "f32[3208, 128]" = torch.ops.aten.view.default(add_138, [3208, 128]);  add_138 = None
    permute_88: "f32[128, 128]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    addmm_49: "f32[3208, 128]" = torch.ops.aten.addmm.default(primals_170, view_160, permute_88);  primals_170 = None
    view_161: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_49, [8, 401, 128]);  addmm_49 = None
    view_162: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_161, [8, 401, 4, 32]);  view_161 = None
    permute_89: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_90: "f32[8, 4, 32, 401]" = torch.ops.aten.permute.default(permute_87, [0, 1, 3, 2]);  permute_87 = None
    expand_14: "f32[8, 4, 1, 32]" = torch.ops.aten.expand.default(permute_85, [8, 4, 1, 32]);  permute_85 = None
    view_163: "f32[32, 1, 32]" = torch.ops.aten.view.default(expand_14, [32, 1, 32]);  expand_14 = None
    expand_15: "f32[8, 4, 32, 401]" = torch.ops.aten.expand.default(permute_90, [8, 4, 32, 401]);  permute_90 = None
    clone_42: "f32[8, 4, 32, 401]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_164: "f32[32, 32, 401]" = torch.ops.aten.view.default(clone_42, [32, 32, 401]);  clone_42 = None
    bmm_6: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_163, view_164)
    view_165: "f32[8, 4, 1, 401]" = torch.ops.aten.view.default(bmm_6, [8, 4, 1, 401]);  bmm_6 = None
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
    view_166: "f32[32, 1, 401]" = torch.ops.aten.view.default(expand_16, [32, 1, 401]);  expand_16 = None
    expand_17: "f32[8, 4, 401, 32]" = torch.ops.aten.expand.default(permute_89, [8, 4, 401, 32]);  permute_89 = None
    clone_44: "f32[8, 4, 401, 32]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_167: "f32[32, 401, 32]" = torch.ops.aten.view.default(clone_44, [32, 401, 32]);  clone_44 = None
    bmm_7: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_166, view_167)
    view_168: "f32[8, 4, 1, 32]" = torch.ops.aten.view.default(bmm_7, [8, 4, 1, 32]);  bmm_7 = None
    permute_91: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    view_169: "f32[8, 1, 128]" = torch.ops.aten.view.default(permute_91, [8, 1, 128]);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_170: "f32[8, 128]" = torch.ops.aten.view.default(view_169, [8, 128]);  view_169 = None
    permute_92: "f32[128, 128]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm_50: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_172, view_170, permute_92);  primals_172 = None
    view_171: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_50, [8, 1, 128]);  addmm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    clone_45: "f32[8, 1, 128]" = torch.ops.aten.clone.default(view_171);  view_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_140: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(slice_39, clone_45);  slice_39 = clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    slice_42: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_140, 0, 0, 9223372036854775807);  add_140 = None
    var_mean_27 = torch.ops.aten.var_mean.correction(slice_42, [2], correction = 0, keepdim = True)
    getitem_150: "f32[8, 1, 1]" = var_mean_27[0]
    getitem_151: "f32[8, 1, 1]" = var_mean_27[1];  var_mean_27 = None
    add_141: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-06);  getitem_150 = None
    rsqrt_27: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    sub_77: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(slice_42, getitem_151)
    mul_185: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_77, rsqrt_27);  sub_77 = None
    mul_186: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_185, primals_173);  mul_185 = None
    add_142: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_186, primals_174);  mul_186 = primals_174 = None
    mul_187: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_142, 0.5)
    mul_188: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_142, 0.7071067811865476)
    erf_15: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_188);  mul_188 = None
    add_143: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_189: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_187, add_143);  mul_187 = add_143 = None
    view_172: "f32[8, 128]" = torch.ops.aten.view.default(mul_189, [8, 128]);  mul_189 = None
    permute_93: "f32[128, 256]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_51: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_176, view_172, permute_93);  primals_176 = None
    view_173: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_51, [8, 1, 256]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_43: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_123, 0, 0, 9223372036854775807);  add_123 = None
    slice_44: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(slice_43, 1, 1, 9223372036854775807);  slice_43 = None
    cat_9: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_173, slice_44], 1);  view_173 = slice_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_28 = torch.ops.aten.var_mean.correction(cat_7, [2], correction = 0, keepdim = True)
    getitem_152: "f32[8, 401, 1]" = var_mean_28[0]
    getitem_153: "f32[8, 401, 1]" = var_mean_28[1];  var_mean_28 = None
    add_144: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_152, 1e-06);  getitem_152 = None
    rsqrt_28: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
    sub_78: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_7, getitem_153)
    mul_190: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_28);  sub_78 = None
    mul_191: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_190, primals_177);  mul_190 = None
    add_145: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_191, primals_178);  mul_191 = primals_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_174: "f32[3208, 128]" = torch.ops.aten.view.default(add_145, [3208, 128]);  add_145 = None
    permute_94: "f32[128, 384]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    addmm_52: "f32[3208, 384]" = torch.ops.aten.addmm.default(primals_180, view_174, permute_94);  primals_180 = None
    view_175: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_52, [8, 401, 384]);  addmm_52 = None
    view_176: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.view.default(view_175, [8, 401, 3, 4, 32]);  view_175 = None
    permute_95: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.permute.default(view_176, [2, 0, 3, 1, 4]);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_95);  permute_95 = None
    getitem_154: "f32[8, 4, 401, 32]" = unbind_8[0]
    getitem_155: "f32[8, 4, 401, 32]" = unbind_8[1]
    getitem_156: "f32[8, 4, 401, 32]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_154, getitem_155, getitem_156)
    getitem_157: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_8[0]
    getitem_158: "f32[8, 4, 401]" = _scaled_dot_product_flash_attention_8[1]
    getitem_159: "i32[]" = _scaled_dot_product_flash_attention_8[2]
    getitem_160: "i32[]" = _scaled_dot_product_flash_attention_8[3]
    getitem_163: "i64[]" = _scaled_dot_product_flash_attention_8[6]
    getitem_164: "i64[]" = _scaled_dot_product_flash_attention_8[7];  _scaled_dot_product_flash_attention_8 = None
    alias_12: "f32[8, 4, 401, 32]" = torch.ops.aten.alias.default(getitem_157)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_96: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(getitem_157, [0, 2, 1, 3]);  getitem_157 = None
    view_177: "f32[8, 401, 128]" = torch.ops.aten.view.default(permute_96, [8, 401, 128]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_178: "f32[3208, 128]" = torch.ops.aten.view.default(view_177, [3208, 128]);  view_177 = None
    permute_97: "f32[128, 128]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_53: "f32[3208, 128]" = torch.ops.aten.addmm.default(primals_182, view_178, permute_97);  primals_182 = None
    view_179: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_53, [8, 401, 128]);  addmm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_46: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_179);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_146: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(cat_7, clone_46);  clone_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_29 = torch.ops.aten.var_mean.correction(add_146, [2], correction = 0, keepdim = True)
    getitem_166: "f32[8, 401, 1]" = var_mean_29[0]
    getitem_167: "f32[8, 401, 1]" = var_mean_29[1];  var_mean_29 = None
    add_147: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-06);  getitem_166 = None
    rsqrt_29: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_147);  add_147 = None
    sub_79: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add_146, getitem_167)
    mul_192: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_29);  sub_79 = None
    mul_193: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_192, primals_183);  mul_192 = None
    add_148: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_193, primals_184);  mul_193 = primals_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_180: "f32[3208, 128]" = torch.ops.aten.view.default(add_148, [3208, 128]);  add_148 = None
    permute_98: "f32[128, 384]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    addmm_54: "f32[3208, 384]" = torch.ops.aten.addmm.default(primals_186, view_180, permute_98);  primals_186 = None
    view_181: "f32[8, 401, 384]" = torch.ops.aten.view.default(addmm_54, [8, 401, 384]);  addmm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_194: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_181, 0.5)
    mul_195: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476)
    erf_16: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_195);  mul_195 = None
    add_149: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_196: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_194, add_149);  mul_194 = add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_47: "f32[8, 401, 384]" = torch.ops.aten.clone.default(mul_196);  mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_182: "f32[3208, 384]" = torch.ops.aten.view.default(clone_47, [3208, 384]);  clone_47 = None
    permute_99: "f32[384, 128]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    addmm_55: "f32[3208, 128]" = torch.ops.aten.addmm.default(primals_188, view_182, permute_99);  primals_188 = None
    view_183: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_55, [8, 401, 128]);  addmm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_183);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_150: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_146, clone_48);  clone_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_30 = torch.ops.aten.var_mean.correction(cat_9, [2], correction = 0, keepdim = True)
    getitem_168: "f32[8, 197, 1]" = var_mean_30[0]
    getitem_169: "f32[8, 197, 1]" = var_mean_30[1];  var_mean_30 = None
    add_151: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-06);  getitem_168 = None
    rsqrt_30: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
    sub_80: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_9, getitem_169)
    mul_197: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_30);  sub_80 = None
    mul_198: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_197, primals_189);  mul_197 = None
    add_152: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_198, primals_190);  mul_198 = primals_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_184: "f32[1576, 256]" = torch.ops.aten.view.default(add_152, [1576, 256]);  add_152 = None
    permute_100: "f32[256, 768]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
    addmm_56: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_192, view_184, permute_100);  primals_192 = None
    view_185: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_56, [8, 197, 768]);  addmm_56 = None
    view_186: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_185, [8, 197, 3, 4, 64]);  view_185 = None
    permute_101: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_186, [2, 0, 3, 1, 4]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_101);  permute_101 = None
    getitem_170: "f32[8, 4, 197, 64]" = unbind_9[0]
    getitem_171: "f32[8, 4, 197, 64]" = unbind_9[1]
    getitem_172: "f32[8, 4, 197, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_170, getitem_171, getitem_172)
    getitem_173: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_9[0]
    getitem_174: "f32[8, 4, 197]" = _scaled_dot_product_flash_attention_9[1]
    getitem_175: "i32[]" = _scaled_dot_product_flash_attention_9[2]
    getitem_176: "i32[]" = _scaled_dot_product_flash_attention_9[3]
    getitem_179: "i64[]" = _scaled_dot_product_flash_attention_9[6]
    getitem_180: "i64[]" = _scaled_dot_product_flash_attention_9[7];  _scaled_dot_product_flash_attention_9 = None
    alias_13: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_173)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_102: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_173, [0, 2, 1, 3]);  getitem_173 = None
    view_187: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_102, [8, 197, 256]);  permute_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_188: "f32[1576, 256]" = torch.ops.aten.view.default(view_187, [1576, 256]);  view_187 = None
    permute_103: "f32[256, 256]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
    addmm_57: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_194, view_188, permute_103);  primals_194 = None
    view_189: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_57, [8, 197, 256]);  addmm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_49: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_189);  view_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_153: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(cat_9, clone_49);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_31 = torch.ops.aten.var_mean.correction(add_153, [2], correction = 0, keepdim = True)
    getitem_182: "f32[8, 197, 1]" = var_mean_31[0]
    getitem_183: "f32[8, 197, 1]" = var_mean_31[1];  var_mean_31 = None
    add_154: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_182, 1e-06);  getitem_182 = None
    rsqrt_31: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_154);  add_154 = None
    sub_81: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_153, getitem_183)
    mul_199: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_81, rsqrt_31);  sub_81 = None
    mul_200: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_199, primals_195);  mul_199 = None
    add_155: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_200, primals_196);  mul_200 = primals_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_190: "f32[1576, 256]" = torch.ops.aten.view.default(add_155, [1576, 256]);  add_155 = None
    permute_104: "f32[256, 768]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
    addmm_58: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_198, view_190, permute_104);  primals_198 = None
    view_191: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_58, [8, 197, 768]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_201: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_191, 0.5)
    mul_202: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_191, 0.7071067811865476)
    erf_17: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_202);  mul_202 = None
    add_156: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_203: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_201, add_156);  mul_201 = add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_50: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_203);  mul_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_192: "f32[1576, 768]" = torch.ops.aten.view.default(clone_50, [1576, 768]);  clone_50 = None
    permute_105: "f32[768, 256]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
    addmm_59: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_200, view_192, permute_105);  primals_200 = None
    view_193: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_59, [8, 197, 256]);  addmm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_51: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_193);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_157: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_153, clone_51);  clone_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_32 = torch.ops.aten.var_mean.correction(add_157, [2], correction = 0, keepdim = True)
    getitem_184: "f32[8, 197, 1]" = var_mean_32[0]
    getitem_185: "f32[8, 197, 1]" = var_mean_32[1];  var_mean_32 = None
    add_158: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_184, 1e-06);  getitem_184 = None
    rsqrt_32: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_158);  add_158 = None
    sub_82: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_157, getitem_185)
    mul_204: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_32);  sub_82 = None
    mul_205: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_204, primals_201);  mul_204 = None
    add_159: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_205, primals_202);  mul_205 = primals_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_194: "f32[1576, 256]" = torch.ops.aten.view.default(add_159, [1576, 256]);  add_159 = None
    permute_106: "f32[256, 768]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
    addmm_60: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_204, view_194, permute_106);  primals_204 = None
    view_195: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_60, [8, 197, 768]);  addmm_60 = None
    view_196: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_195, [8, 197, 3, 4, 64]);  view_195 = None
    permute_107: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_196, [2, 0, 3, 1, 4]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_107);  permute_107 = None
    getitem_186: "f32[8, 4, 197, 64]" = unbind_10[0]
    getitem_187: "f32[8, 4, 197, 64]" = unbind_10[1]
    getitem_188: "f32[8, 4, 197, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_186, getitem_187, getitem_188)
    getitem_189: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_10[0]
    getitem_190: "f32[8, 4, 197]" = _scaled_dot_product_flash_attention_10[1]
    getitem_191: "i32[]" = _scaled_dot_product_flash_attention_10[2]
    getitem_192: "i32[]" = _scaled_dot_product_flash_attention_10[3]
    getitem_195: "i64[]" = _scaled_dot_product_flash_attention_10[6]
    getitem_196: "i64[]" = _scaled_dot_product_flash_attention_10[7];  _scaled_dot_product_flash_attention_10 = None
    alias_14: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_189)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_108: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3]);  getitem_189 = None
    view_197: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_108, [8, 197, 256]);  permute_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_198: "f32[1576, 256]" = torch.ops.aten.view.default(view_197, [1576, 256]);  view_197 = None
    permute_109: "f32[256, 256]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
    addmm_61: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_206, view_198, permute_109);  primals_206 = None
    view_199: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_61, [8, 197, 256]);  addmm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_52: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_199);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_160: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_157, clone_52);  clone_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_33 = torch.ops.aten.var_mean.correction(add_160, [2], correction = 0, keepdim = True)
    getitem_198: "f32[8, 197, 1]" = var_mean_33[0]
    getitem_199: "f32[8, 197, 1]" = var_mean_33[1];  var_mean_33 = None
    add_161: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-06);  getitem_198 = None
    rsqrt_33: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_161);  add_161 = None
    sub_83: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_160, getitem_199)
    mul_206: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_33);  sub_83 = None
    mul_207: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_206, primals_207);  mul_206 = None
    add_162: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_207, primals_208);  mul_207 = primals_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_200: "f32[1576, 256]" = torch.ops.aten.view.default(add_162, [1576, 256]);  add_162 = None
    permute_110: "f32[256, 768]" = torch.ops.aten.permute.default(primals_209, [1, 0]);  primals_209 = None
    addmm_62: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_210, view_200, permute_110);  primals_210 = None
    view_201: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_62, [8, 197, 768]);  addmm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_208: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_201, 0.5)
    mul_209: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_201, 0.7071067811865476)
    erf_18: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_209);  mul_209 = None
    add_163: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_210: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_208, add_163);  mul_208 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_53: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_210);  mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_202: "f32[1576, 768]" = torch.ops.aten.view.default(clone_53, [1576, 768]);  clone_53 = None
    permute_111: "f32[768, 256]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
    addmm_63: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_212, view_202, permute_111);  primals_212 = None
    view_203: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_63, [8, 197, 256]);  addmm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_54: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_203);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_164: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_160, clone_54);  clone_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_34 = torch.ops.aten.var_mean.correction(add_164, [2], correction = 0, keepdim = True)
    getitem_200: "f32[8, 197, 1]" = var_mean_34[0]
    getitem_201: "f32[8, 197, 1]" = var_mean_34[1];  var_mean_34 = None
    add_165: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-06);  getitem_200 = None
    rsqrt_34: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_165);  add_165 = None
    sub_84: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_164, getitem_201)
    mul_211: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_34);  sub_84 = None
    mul_212: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_211, primals_213);  mul_211 = None
    add_166: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_212, primals_214);  mul_212 = primals_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_204: "f32[1576, 256]" = torch.ops.aten.view.default(add_166, [1576, 256]);  add_166 = None
    permute_112: "f32[256, 768]" = torch.ops.aten.permute.default(primals_215, [1, 0]);  primals_215 = None
    addmm_64: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_216, view_204, permute_112);  primals_216 = None
    view_205: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_64, [8, 197, 768]);  addmm_64 = None
    view_206: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.view.default(view_205, [8, 197, 3, 4, 64]);  view_205 = None
    permute_113: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.permute.default(view_206, [2, 0, 3, 1, 4]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_113);  permute_113 = None
    getitem_202: "f32[8, 4, 197, 64]" = unbind_11[0]
    getitem_203: "f32[8, 4, 197, 64]" = unbind_11[1]
    getitem_204: "f32[8, 4, 197, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_202, getitem_203, getitem_204)
    getitem_205: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_11[0]
    getitem_206: "f32[8, 4, 197]" = _scaled_dot_product_flash_attention_11[1]
    getitem_207: "i32[]" = _scaled_dot_product_flash_attention_11[2]
    getitem_208: "i32[]" = _scaled_dot_product_flash_attention_11[3]
    getitem_211: "i64[]" = _scaled_dot_product_flash_attention_11[6]
    getitem_212: "i64[]" = _scaled_dot_product_flash_attention_11[7];  _scaled_dot_product_flash_attention_11 = None
    alias_15: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(getitem_205)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_114: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(getitem_205, [0, 2, 1, 3]);  getitem_205 = None
    view_207: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_114, [8, 197, 256]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_208: "f32[1576, 256]" = torch.ops.aten.view.default(view_207, [1576, 256]);  view_207 = None
    permute_115: "f32[256, 256]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
    addmm_65: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_218, view_208, permute_115);  primals_218 = None
    view_209: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_65, [8, 197, 256]);  addmm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_55: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_209);  view_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_167: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_164, clone_55);  clone_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_35 = torch.ops.aten.var_mean.correction(add_167, [2], correction = 0, keepdim = True)
    getitem_214: "f32[8, 197, 1]" = var_mean_35[0]
    getitem_215: "f32[8, 197, 1]" = var_mean_35[1];  var_mean_35 = None
    add_168: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_214, 1e-06);  getitem_214 = None
    rsqrt_35: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_168);  add_168 = None
    sub_85: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_167, getitem_215)
    mul_213: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_35);  sub_85 = None
    mul_214: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_213, primals_219);  mul_213 = None
    add_169: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_214, primals_220);  mul_214 = primals_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_210: "f32[1576, 256]" = torch.ops.aten.view.default(add_169, [1576, 256]);  add_169 = None
    permute_116: "f32[256, 768]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
    addmm_66: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_222, view_210, permute_116);  primals_222 = None
    view_211: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_66, [8, 197, 768]);  addmm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_215: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, 0.5)
    mul_216: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, 0.7071067811865476)
    erf_19: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_216);  mul_216 = None
    add_170: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_217: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_215, add_170);  mul_215 = add_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_56: "f32[8, 197, 768]" = torch.ops.aten.clone.default(mul_217);  mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_212: "f32[1576, 768]" = torch.ops.aten.view.default(clone_56, [1576, 768]);  clone_56 = None
    permute_117: "f32[768, 256]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
    addmm_67: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_224, view_212, permute_117);  primals_224 = None
    view_213: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_67, [8, 197, 256]);  addmm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_57: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_213);  view_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_171: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_167, clone_57);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    slice_45: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_150, 0, 0, 9223372036854775807)
    slice_46: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_45, 1, 0, 1);  slice_45 = None
    clone_58: "f32[8, 1, 128]" = torch.ops.aten.clone.default(slice_46, memory_format = torch.contiguous_format)
    var_mean_36 = torch.ops.aten.var_mean.correction(clone_58, [2], correction = 0, keepdim = True)
    getitem_216: "f32[8, 1, 1]" = var_mean_36[0]
    getitem_217: "f32[8, 1, 1]" = var_mean_36[1];  var_mean_36 = None
    add_172: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-06);  getitem_216 = None
    rsqrt_36: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_172);  add_172 = None
    sub_86: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(clone_58, getitem_217);  clone_58 = None
    mul_218: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_36);  sub_86 = None
    mul_219: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_218, primals_225);  mul_218 = None
    add_173: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_219, primals_226);  mul_219 = primals_226 = None
    mul_220: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_173, 0.5)
    mul_221: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_173, 0.7071067811865476)
    erf_20: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_221);  mul_221 = None
    add_174: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_222: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_220, add_174);  mul_220 = add_174 = None
    view_214: "f32[8, 128]" = torch.ops.aten.view.default(mul_222, [8, 128]);  mul_222 = None
    permute_118: "f32[128, 256]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
    addmm_68: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_228, view_214, permute_118);  primals_228 = None
    view_215: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_68, [8, 1, 256]);  addmm_68 = None
    slice_47: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_171, 0, 0, 9223372036854775807)
    slice_48: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_47, 1, 0, 1);  slice_47 = None
    clone_59: "f32[8, 1, 256]" = torch.ops.aten.clone.default(slice_48, memory_format = torch.contiguous_format)
    var_mean_37 = torch.ops.aten.var_mean.correction(clone_59, [2], correction = 0, keepdim = True)
    getitem_218: "f32[8, 1, 1]" = var_mean_37[0]
    getitem_219: "f32[8, 1, 1]" = var_mean_37[1];  var_mean_37 = None
    add_175: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-06);  getitem_218 = None
    rsqrt_37: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_175);  add_175 = None
    sub_87: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(clone_59, getitem_219);  clone_59 = None
    mul_223: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_37);  sub_87 = None
    mul_224: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_223, primals_229);  mul_223 = None
    add_176: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_224, primals_230);  mul_224 = primals_230 = None
    mul_225: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_176, 0.5)
    mul_226: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_176, 0.7071067811865476)
    erf_21: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_226);  mul_226 = None
    add_177: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_227: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_225, add_177);  mul_225 = add_177 = None
    view_216: "f32[8, 256]" = torch.ops.aten.view.default(mul_227, [8, 256]);  mul_227 = None
    permute_119: "f32[256, 128]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
    addmm_69: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_232, view_216, permute_119);  primals_232 = None
    view_217: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_69, [8, 1, 128]);  addmm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_49: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_171, 0, 0, 9223372036854775807)
    slice_50: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(slice_49, 1, 1, 9223372036854775807);  slice_49 = None
    cat_10: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_215, slice_50], 1);  view_215 = slice_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_51: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(cat_10, 0, 0, 9223372036854775807)
    slice_52: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_51, 1, 0, 1);  slice_51 = None
    var_mean_38 = torch.ops.aten.var_mean.correction(cat_10, [2], correction = 0, keepdim = True)
    getitem_220: "f32[8, 197, 1]" = var_mean_38[0]
    getitem_221: "f32[8, 197, 1]" = var_mean_38[1];  var_mean_38 = None
    add_178: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-06);  getitem_220 = None
    rsqrt_38: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_178);  add_178 = None
    sub_88: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_10, getitem_221)
    mul_228: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_38);  sub_88 = None
    mul_229: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_228, primals_233);  mul_228 = None
    add_179: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_229, primals_234);  mul_229 = primals_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_53: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_179, 0, 0, 9223372036854775807)
    slice_54: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_53, 1, 0, 1);  slice_53 = None
    permute_120: "f32[256, 256]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
    view_218: "f32[8, 256]" = torch.ops.aten.view.default(slice_54, [8, 256]);  slice_54 = None
    mm_4: "f32[8, 256]" = torch.ops.aten.mm.default(view_218, permute_120)
    view_219: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_4, [8, 1, 256]);  mm_4 = None
    add_180: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(view_219, primals_236);  view_219 = primals_236 = None
    view_220: "f32[8, 1, 4, 64]" = torch.ops.aten.view.default(add_180, [8, 1, 4, 64]);  add_180 = None
    permute_121: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1, 3]);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_221: "f32[1576, 256]" = torch.ops.aten.view.default(add_179, [1576, 256])
    permute_122: "f32[256, 256]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
    addmm_70: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_238, view_221, permute_122);  primals_238 = None
    view_222: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_70, [8, 197, 256]);  addmm_70 = None
    view_223: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_222, [8, 197, 4, 64]);  view_222 = None
    permute_123: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_223, [0, 2, 1, 3]);  view_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_224: "f32[1576, 256]" = torch.ops.aten.view.default(add_179, [1576, 256]);  add_179 = None
    permute_124: "f32[256, 256]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
    addmm_71: "f32[1576, 256]" = torch.ops.aten.addmm.default(primals_240, view_224, permute_124);  primals_240 = None
    view_225: "f32[8, 197, 256]" = torch.ops.aten.view.default(addmm_71, [8, 197, 256]);  addmm_71 = None
    view_226: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_225, [8, 197, 4, 64]);  view_225 = None
    permute_125: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_126: "f32[8, 4, 64, 197]" = torch.ops.aten.permute.default(permute_123, [0, 1, 3, 2]);  permute_123 = None
    expand_18: "f32[8, 4, 1, 64]" = torch.ops.aten.expand.default(permute_121, [8, 4, 1, 64]);  permute_121 = None
    view_227: "f32[32, 1, 64]" = torch.ops.aten.view.default(expand_18, [32, 1, 64]);  expand_18 = None
    expand_19: "f32[8, 4, 64, 197]" = torch.ops.aten.expand.default(permute_126, [8, 4, 64, 197]);  permute_126 = None
    clone_60: "f32[8, 4, 64, 197]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_228: "f32[32, 64, 197]" = torch.ops.aten.view.default(clone_60, [32, 64, 197]);  clone_60 = None
    bmm_8: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_227, view_228)
    view_229: "f32[8, 4, 1, 197]" = torch.ops.aten.view.default(bmm_8, [8, 4, 1, 197]);  bmm_8 = None
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
    view_230: "f32[32, 1, 197]" = torch.ops.aten.view.default(expand_20, [32, 1, 197]);  expand_20 = None
    expand_21: "f32[8, 4, 197, 64]" = torch.ops.aten.expand.default(permute_125, [8, 4, 197, 64]);  permute_125 = None
    clone_62: "f32[8, 4, 197, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_231: "f32[32, 197, 64]" = torch.ops.aten.view.default(clone_62, [32, 197, 64]);  clone_62 = None
    bmm_9: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_230, view_231)
    view_232: "f32[8, 4, 1, 64]" = torch.ops.aten.view.default(bmm_9, [8, 4, 1, 64]);  bmm_9 = None
    permute_127: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
    view_233: "f32[8, 1, 256]" = torch.ops.aten.view.default(permute_127, [8, 1, 256]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_234: "f32[8, 256]" = torch.ops.aten.view.default(view_233, [8, 256]);  view_233 = None
    permute_128: "f32[256, 256]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
    addmm_72: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_242, view_234, permute_128);  primals_242 = None
    view_235: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_72, [8, 1, 256]);  addmm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    clone_63: "f32[8, 1, 256]" = torch.ops.aten.clone.default(view_235);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_181: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(slice_52, clone_63);  slice_52 = clone_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    slice_55: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_181, 0, 0, 9223372036854775807);  add_181 = None
    var_mean_39 = torch.ops.aten.var_mean.correction(slice_55, [2], correction = 0, keepdim = True)
    getitem_222: "f32[8, 1, 1]" = var_mean_39[0]
    getitem_223: "f32[8, 1, 1]" = var_mean_39[1];  var_mean_39 = None
    add_182: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_222, 1e-06);  getitem_222 = None
    rsqrt_39: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_182);  add_182 = None
    sub_90: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(slice_55, getitem_223)
    mul_231: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_39);  sub_90 = None
    mul_232: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_231, primals_243);  mul_231 = None
    add_183: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_232, primals_244);  mul_232 = primals_244 = None
    mul_233: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_183, 0.5)
    mul_234: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_183, 0.7071067811865476)
    erf_22: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_234);  mul_234 = None
    add_184: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_235: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_233, add_184);  mul_233 = add_184 = None
    view_236: "f32[8, 256]" = torch.ops.aten.view.default(mul_235, [8, 256]);  mul_235 = None
    permute_129: "f32[256, 128]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
    addmm_73: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_246, view_236, permute_129);  primals_246 = None
    view_237: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_73, [8, 1, 128]);  addmm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_56: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_150, 0, 0, 9223372036854775807)
    slice_57: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(slice_56, 1, 1, 9223372036854775807);  slice_56 = None
    cat_11: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_237, slice_57], 1);  view_237 = slice_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_58: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_150, 0, 0, 9223372036854775807);  add_150 = None
    slice_59: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(slice_58, 1, 1, 9223372036854775807);  slice_58 = None
    cat_12: "f32[8, 401, 128]" = torch.ops.aten.cat.default([view_217, slice_59], 1);  view_217 = slice_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    slice_60: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(cat_12, 0, 0, 9223372036854775807)
    slice_61: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_60, 1, 0, 1);  slice_60 = None
    var_mean_40 = torch.ops.aten.var_mean.correction(cat_12, [2], correction = 0, keepdim = True)
    getitem_224: "f32[8, 401, 1]" = var_mean_40[0]
    getitem_225: "f32[8, 401, 1]" = var_mean_40[1];  var_mean_40 = None
    add_185: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_224, 1e-06);  getitem_224 = None
    rsqrt_40: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
    sub_91: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_12, getitem_225)
    mul_236: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_40);  sub_91 = None
    mul_237: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_236, primals_247);  mul_236 = None
    add_186: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_237, primals_248);  mul_237 = primals_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    slice_62: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_186, 0, 0, 9223372036854775807)
    slice_63: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(slice_62, 1, 0, 1);  slice_62 = None
    permute_130: "f32[128, 128]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
    view_238: "f32[8, 128]" = torch.ops.aten.view.default(slice_63, [8, 128]);  slice_63 = None
    mm_5: "f32[8, 128]" = torch.ops.aten.mm.default(view_238, permute_130)
    view_239: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_5, [8, 1, 128]);  mm_5 = None
    add_187: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(view_239, primals_250);  view_239 = primals_250 = None
    view_240: "f32[8, 1, 4, 32]" = torch.ops.aten.view.default(add_187, [8, 1, 4, 32]);  add_187 = None
    permute_131: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_241: "f32[3208, 128]" = torch.ops.aten.view.default(add_186, [3208, 128])
    permute_132: "f32[128, 128]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
    addmm_74: "f32[3208, 128]" = torch.ops.aten.addmm.default(primals_252, view_241, permute_132);  primals_252 = None
    view_242: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_74, [8, 401, 128]);  addmm_74 = None
    view_243: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_242, [8, 401, 4, 32]);  view_242 = None
    permute_133: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_243, [0, 2, 1, 3]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    view_244: "f32[3208, 128]" = torch.ops.aten.view.default(add_186, [3208, 128]);  add_186 = None
    permute_134: "f32[128, 128]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
    addmm_75: "f32[3208, 128]" = torch.ops.aten.addmm.default(primals_254, view_244, permute_134);  primals_254 = None
    view_245: "f32[8, 401, 128]" = torch.ops.aten.view.default(addmm_75, [8, 401, 128]);  addmm_75 = None
    view_246: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_245, [8, 401, 4, 32]);  view_245 = None
    permute_135: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    permute_136: "f32[8, 4, 32, 401]" = torch.ops.aten.permute.default(permute_133, [0, 1, 3, 2]);  permute_133 = None
    expand_22: "f32[8, 4, 1, 32]" = torch.ops.aten.expand.default(permute_131, [8, 4, 1, 32]);  permute_131 = None
    view_247: "f32[32, 1, 32]" = torch.ops.aten.view.default(expand_22, [32, 1, 32]);  expand_22 = None
    expand_23: "f32[8, 4, 32, 401]" = torch.ops.aten.expand.default(permute_136, [8, 4, 32, 401]);  permute_136 = None
    clone_64: "f32[8, 4, 32, 401]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_248: "f32[32, 32, 401]" = torch.ops.aten.view.default(clone_64, [32, 32, 401]);  clone_64 = None
    bmm_10: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_247, view_248)
    view_249: "f32[8, 4, 1, 401]" = torch.ops.aten.view.default(bmm_10, [8, 4, 1, 401]);  bmm_10 = None
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
    view_250: "f32[32, 1, 401]" = torch.ops.aten.view.default(expand_24, [32, 1, 401]);  expand_24 = None
    expand_25: "f32[8, 4, 401, 32]" = torch.ops.aten.expand.default(permute_135, [8, 4, 401, 32]);  permute_135 = None
    clone_66: "f32[8, 4, 401, 32]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_251: "f32[32, 401, 32]" = torch.ops.aten.view.default(clone_66, [32, 401, 32]);  clone_66 = None
    bmm_11: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_250, view_251)
    view_252: "f32[8, 4, 1, 32]" = torch.ops.aten.view.default(bmm_11, [8, 4, 1, 32]);  bmm_11 = None
    permute_137: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    view_253: "f32[8, 1, 128]" = torch.ops.aten.view.default(permute_137, [8, 1, 128]);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_254: "f32[8, 128]" = torch.ops.aten.view.default(view_253, [8, 128]);  view_253 = None
    permute_138: "f32[128, 128]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
    addmm_76: "f32[8, 128]" = torch.ops.aten.addmm.default(primals_256, view_254, permute_138);  primals_256 = None
    view_255: "f32[8, 1, 128]" = torch.ops.aten.view.default(addmm_76, [8, 1, 128]);  addmm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    clone_67: "f32[8, 1, 128]" = torch.ops.aten.clone.default(view_255);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_188: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(slice_61, clone_67);  slice_61 = clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    slice_64: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_188, 0, 0, 9223372036854775807);  add_188 = None
    var_mean_41 = torch.ops.aten.var_mean.correction(slice_64, [2], correction = 0, keepdim = True)
    getitem_226: "f32[8, 1, 1]" = var_mean_41[0]
    getitem_227: "f32[8, 1, 1]" = var_mean_41[1];  var_mean_41 = None
    add_189: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-06);  getitem_226 = None
    rsqrt_41: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_189);  add_189 = None
    sub_93: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(slice_64, getitem_227)
    mul_239: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_41);  sub_93 = None
    mul_240: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_239, primals_257);  mul_239 = None
    add_190: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_240, primals_258);  mul_240 = primals_258 = None
    mul_241: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_190, 0.5)
    mul_242: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_190, 0.7071067811865476)
    erf_23: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_242);  mul_242 = None
    add_191: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_243: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_241, add_191);  mul_241 = add_191 = None
    view_256: "f32[8, 128]" = torch.ops.aten.view.default(mul_243, [8, 128]);  mul_243 = None
    permute_139: "f32[128, 256]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
    addmm_77: "f32[8, 256]" = torch.ops.aten.addmm.default(primals_260, view_256, permute_139);  primals_260 = None
    view_257: "f32[8, 1, 256]" = torch.ops.aten.view.default(addmm_77, [8, 1, 256]);  addmm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_65: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_171, 0, 0, 9223372036854775807);  add_171 = None
    slice_66: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(slice_65, 1, 1, 9223372036854775807);  slice_65 = None
    cat_13: "f32[8, 197, 256]" = torch.ops.aten.cat.default([view_257, slice_66], 1);  view_257 = slice_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:451, code: xs = [norm(xs[i]) for i, norm in enumerate(self.norm)]
    var_mean_42 = torch.ops.aten.var_mean.correction(cat_11, [2], correction = 0, keepdim = True)
    getitem_228: "f32[8, 401, 1]" = var_mean_42[0]
    getitem_229: "f32[8, 401, 1]" = var_mean_42[1];  var_mean_42 = None
    add_192: "f32[8, 401, 1]" = torch.ops.aten.add.Tensor(getitem_228, 1e-06);  getitem_228 = None
    rsqrt_42: "f32[8, 401, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
    sub_94: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_11, getitem_229)
    mul_244: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_42);  sub_94 = None
    mul_245: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_244, primals_261);  mul_244 = None
    add_193: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_245, primals_262);  mul_245 = primals_262 = None
    var_mean_43 = torch.ops.aten.var_mean.correction(cat_13, [2], correction = 0, keepdim = True)
    getitem_230: "f32[8, 197, 1]" = var_mean_43[0]
    getitem_231: "f32[8, 197, 1]" = var_mean_43[1];  var_mean_43 = None
    add_194: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_230, 1e-06);  getitem_230 = None
    rsqrt_43: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_194);  add_194 = None
    sub_95: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_13, getitem_231)
    mul_246: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_43);  sub_95 = None
    mul_247: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_246, primals_263);  mul_246 = None
    add_195: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_247, primals_264);  mul_247 = primals_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:455, code: xs = [x[:, 1:].mean(dim=1) for x in xs] if self.global_pool == 'avg' else [x[:, 0] for x in xs]
    slice_67: "f32[8, 401, 128]" = torch.ops.aten.slice.Tensor(add_193, 0, 0, 9223372036854775807);  add_193 = None
    select: "f32[8, 128]" = torch.ops.aten.select.int(slice_67, 1, 0);  slice_67 = None
    slice_68: "f32[8, 197, 256]" = torch.ops.aten.slice.Tensor(add_195, 0, 0, 9223372036854775807);  add_195 = None
    select_1: "f32[8, 256]" = torch.ops.aten.select.int(slice_68, 1, 0);  slice_68 = None
    
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
    view_258: "f32[2, 8, 1000]" = torch.ops.aten.view.default(cat_14, [2, 8, 1000]);  cat_14 = None
    mean: "f32[8, 1000]" = torch.ops.aten.mean.dim(view_258, [0]);  view_258 = None
    unsqueeze: "f32[1, 8, 1000]" = torch.ops.aten.unsqueeze.default(tangents_1, 0);  tangents_1 = None
    expand_26: "f32[2, 8, 1000]" = torch.ops.aten.expand.default(unsqueeze, [2, 8, 1000]);  unsqueeze = None
    div_6: "f32[2, 8, 1000]" = torch.ops.aten.div.Scalar(expand_26, 2);  expand_26 = None
    select_2: "f32[8, 1000]" = torch.ops.aten.select.int(div_6, 0, 0)
    select_3: "f32[8, 1000]" = torch.ops.aten.select.int(div_6, 0, 1);  div_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:459, code: return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)
    permute_142: "f32[1000, 256]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
    mm_6: "f32[8, 256]" = torch.ops.aten.mm.default(select_3, permute_142);  permute_142 = None
    permute_143: "f32[1000, 8]" = torch.ops.aten.permute.default(select_3, [1, 0])
    mm_7: "f32[1000, 256]" = torch.ops.aten.mm.default(permute_143, clone_69);  permute_143 = clone_69 = None
    permute_144: "f32[256, 1000]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_7: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(select_3, [0], True);  select_3 = None
    view_259: "f32[1000]" = torch.ops.aten.view.default(sum_7, [1000]);  sum_7 = None
    permute_145: "f32[1000, 256]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    permute_146: "f32[1000, 128]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    mm_8: "f32[8, 128]" = torch.ops.aten.mm.default(select_2, permute_146);  permute_146 = None
    permute_147: "f32[1000, 8]" = torch.ops.aten.permute.default(select_2, [1, 0])
    mm_9: "f32[1000, 128]" = torch.ops.aten.mm.default(permute_147, clone_68);  permute_147 = clone_68 = None
    permute_148: "f32[128, 1000]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_8: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(select_2, [0], True);  select_2 = None
    view_260: "f32[1000]" = torch.ops.aten.view.default(sum_8, [1000]);  sum_8 = None
    permute_149: "f32[1000, 128]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:455, code: xs = [x[:, 1:].mean(dim=1) for x in xs] if self.global_pool == 'avg' else [x[:, 0] for x in xs]
    full: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter: "f32[8, 197, 256]" = torch.ops.aten.select_scatter.default(full, mm_6, 1, 0);  full = mm_6 = None
    full_1: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_1, select_scatter, 0, 0, 9223372036854775807);  full_1 = select_scatter = None
    full_2: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter_1: "f32[8, 401, 128]" = torch.ops.aten.select_scatter.default(full_2, mm_8, 1, 0);  full_2 = mm_8 = None
    full_3: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_3, select_scatter_1, 0, 0, 9223372036854775807);  full_3 = select_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:451, code: xs = [norm(xs[i]) for i, norm in enumerate(self.norm)]
    sub_96: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_13, getitem_231);  cat_13 = getitem_231 = None
    mul_248: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_43);  sub_96 = None
    mul_249: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(slice_scatter, primals_263);  primals_263 = None
    mul_250: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_249, 256)
    sum_9: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_249, [2], True)
    mul_251: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_249, mul_248);  mul_249 = None
    sum_10: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True);  mul_251 = None
    mul_252: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_248, sum_10);  sum_10 = None
    sub_97: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_250, sum_9);  mul_250 = sum_9 = None
    sub_98: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_97, mul_252);  sub_97 = mul_252 = None
    div_7: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_43, 256);  rsqrt_43 = None
    mul_253: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_7, sub_98);  div_7 = sub_98 = None
    mul_254: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(slice_scatter, mul_248);  mul_248 = None
    sum_11: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_254, [0, 1]);  mul_254 = None
    sum_12: "f32[256]" = torch.ops.aten.sum.dim_IntList(slice_scatter, [0, 1]);  slice_scatter = None
    sub_99: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_11, getitem_229);  cat_11 = getitem_229 = None
    mul_255: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_42);  sub_99 = None
    mul_256: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(slice_scatter_1, primals_261);  primals_261 = None
    mul_257: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_256, 128)
    sum_13: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True)
    mul_258: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_256, mul_255);  mul_256 = None
    sum_14: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_258, [2], True);  mul_258 = None
    mul_259: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_255, sum_14);  sum_14 = None
    sub_100: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_257, sum_13);  mul_257 = sum_13 = None
    sub_101: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_100, mul_259);  sub_100 = mul_259 = None
    div_8: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_42, 128);  rsqrt_42 = None
    mul_260: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_8, sub_101);  div_8 = sub_101 = None
    mul_261: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(slice_scatter_1, mul_255);  mul_255 = None
    sum_15: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_261, [0, 1]);  mul_261 = None
    sum_16: "f32[128]" = torch.ops.aten.sum.dim_IntList(slice_scatter_1, [0, 1]);  slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_69: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(mul_253, 1, 0, 1)
    slice_70: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(mul_253, 1, 1, 197);  mul_253 = None
    full_4: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_2: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_4, slice_70, 1, 1, 9223372036854775807);  full_4 = slice_70 = None
    full_5: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_3: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_5, slice_scatter_2, 0, 0, 9223372036854775807);  full_5 = slice_scatter_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    view_261: "f32[8, 256]" = torch.ops.aten.view.default(slice_69, [8, 256]);  slice_69 = None
    permute_150: "f32[256, 128]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    mm_10: "f32[8, 128]" = torch.ops.aten.mm.default(view_261, permute_150);  permute_150 = None
    permute_151: "f32[256, 8]" = torch.ops.aten.permute.default(view_261, [1, 0])
    mm_11: "f32[256, 128]" = torch.ops.aten.mm.default(permute_151, view_256);  permute_151 = view_256 = None
    permute_152: "f32[128, 256]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_17: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
    view_262: "f32[256]" = torch.ops.aten.view.default(sum_17, [256]);  sum_17 = None
    permute_153: "f32[256, 128]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_263: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_10, [8, 1, 128]);  mm_10 = None
    mul_262: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_190, 0.7071067811865476)
    erf_24: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_262);  mul_262 = None
    add_196: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
    mul_263: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_196, 0.5);  add_196 = None
    mul_264: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_190, add_190)
    mul_265: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_264, -0.5);  mul_264 = None
    exp_6: "f32[8, 1, 128]" = torch.ops.aten.exp.default(mul_265);  mul_265 = None
    mul_266: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_267: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_190, mul_266);  add_190 = mul_266 = None
    add_197: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_263, mul_267);  mul_263 = mul_267 = None
    mul_268: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_263, add_197);  view_263 = add_197 = None
    sub_102: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(slice_64, getitem_227);  slice_64 = getitem_227 = None
    mul_269: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_41);  sub_102 = None
    mul_270: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_268, primals_257);  primals_257 = None
    mul_271: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_270, 128)
    sum_18: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_270, [2], True)
    mul_272: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_270, mul_269);  mul_270 = None
    sum_19: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_272, [2], True);  mul_272 = None
    mul_273: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_269, sum_19);  sum_19 = None
    sub_103: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(mul_271, sum_18);  mul_271 = sum_18 = None
    sub_104: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(sub_103, mul_273);  sub_103 = mul_273 = None
    div_9: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_41, 128);  rsqrt_41 = None
    mul_274: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(div_9, sub_104);  div_9 = sub_104 = None
    mul_275: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_268, mul_269);  mul_269 = None
    sum_20: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_275, [0, 1]);  mul_275 = None
    sum_21: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_268, [0, 1]);  mul_268 = None
    full_6: "f32[8, 1, 128]" = torch.ops.aten.full.default([8, 1, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_4: "f32[8, 1, 128]" = torch.ops.aten.slice_scatter.default(full_6, mul_274, 0, 0, 9223372036854775807);  full_6 = mul_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_264: "f32[8, 128]" = torch.ops.aten.view.default(slice_scatter_4, [8, 128])
    permute_154: "f32[128, 128]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    mm_12: "f32[8, 128]" = torch.ops.aten.mm.default(view_264, permute_154);  permute_154 = None
    permute_155: "f32[128, 8]" = torch.ops.aten.permute.default(view_264, [1, 0])
    mm_13: "f32[128, 128]" = torch.ops.aten.mm.default(permute_155, view_254);  permute_155 = view_254 = None
    permute_156: "f32[128, 128]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_22: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
    view_265: "f32[128]" = torch.ops.aten.view.default(sum_22, [128]);  sum_22 = None
    permute_157: "f32[128, 128]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    view_266: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_12, [8, 1, 128]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    view_267: "f32[8, 1, 4, 32]" = torch.ops.aten.view.default(view_266, [8, 1, 4, 32]);  view_266 = None
    permute_158: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
    view_268: "f32[32, 1, 32]" = torch.ops.aten.view.default(permute_158, [32, 1, 32]);  permute_158 = None
    permute_159: "f32[32, 401, 1]" = torch.ops.aten.permute.default(view_250, [0, 2, 1]);  view_250 = None
    bmm_12: "f32[32, 401, 32]" = torch.ops.aten.bmm.default(permute_159, view_268);  permute_159 = None
    permute_160: "f32[32, 32, 401]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    bmm_13: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_268, permute_160);  view_268 = permute_160 = None
    view_269: "f32[8, 4, 401, 32]" = torch.ops.aten.view.default(bmm_12, [8, 4, 401, 32]);  bmm_12 = None
    view_270: "f32[8, 4, 1, 401]" = torch.ops.aten.view.default(bmm_13, [8, 4, 1, 401]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    alias_18: "f32[8, 4, 1, 401]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_276: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(view_270, alias_18);  view_270 = None
    sum_23: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [-1], True)
    mul_277: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(alias_18, sum_23);  alias_18 = sum_23 = None
    sub_105: "f32[8, 4, 1, 401]" = torch.ops.aten.sub.Tensor(mul_276, mul_277);  mul_276 = mul_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    mul_278: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(sub_105, 0.1767766952966369);  sub_105 = None
    view_271: "f32[32, 1, 401]" = torch.ops.aten.view.default(mul_278, [32, 1, 401]);  mul_278 = None
    permute_161: "f32[32, 32, 1]" = torch.ops.aten.permute.default(view_247, [0, 2, 1]);  view_247 = None
    bmm_14: "f32[32, 32, 401]" = torch.ops.aten.bmm.default(permute_161, view_271);  permute_161 = None
    permute_162: "f32[32, 401, 32]" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
    bmm_15: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_271, permute_162);  view_271 = permute_162 = None
    view_272: "f32[8, 4, 32, 401]" = torch.ops.aten.view.default(bmm_14, [8, 4, 32, 401]);  bmm_14 = None
    view_273: "f32[8, 4, 1, 32]" = torch.ops.aten.view.default(bmm_15, [8, 4, 1, 32]);  bmm_15 = None
    permute_163: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_272, [0, 1, 3, 2]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_164: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(view_269, [0, 2, 1, 3]);  view_269 = None
    clone_70: "f32[8, 401, 4, 32]" = torch.ops.aten.clone.default(permute_164, memory_format = torch.contiguous_format);  permute_164 = None
    view_274: "f32[8, 401, 128]" = torch.ops.aten.view.default(clone_70, [8, 401, 128]);  clone_70 = None
    view_275: "f32[3208, 128]" = torch.ops.aten.view.default(view_274, [3208, 128]);  view_274 = None
    permute_165: "f32[128, 128]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    mm_14: "f32[3208, 128]" = torch.ops.aten.mm.default(view_275, permute_165);  permute_165 = None
    permute_166: "f32[128, 3208]" = torch.ops.aten.permute.default(view_275, [1, 0])
    mm_15: "f32[128, 128]" = torch.ops.aten.mm.default(permute_166, view_244);  permute_166 = view_244 = None
    permute_167: "f32[128, 128]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_24: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_275, [0], True);  view_275 = None
    view_276: "f32[128]" = torch.ops.aten.view.default(sum_24, [128]);  sum_24 = None
    permute_168: "f32[128, 128]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    view_277: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_14, [8, 401, 128]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_169: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(permute_163, [0, 2, 1, 3]);  permute_163 = None
    view_278: "f32[8, 401, 128]" = torch.ops.aten.view.default(permute_169, [8, 401, 128]);  permute_169 = None
    clone_71: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_278, memory_format = torch.contiguous_format);  view_278 = None
    view_279: "f32[3208, 128]" = torch.ops.aten.view.default(clone_71, [3208, 128]);  clone_71 = None
    permute_170: "f32[128, 128]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    mm_16: "f32[3208, 128]" = torch.ops.aten.mm.default(view_279, permute_170);  permute_170 = None
    permute_171: "f32[128, 3208]" = torch.ops.aten.permute.default(view_279, [1, 0])
    mm_17: "f32[128, 128]" = torch.ops.aten.mm.default(permute_171, view_241);  permute_171 = view_241 = None
    permute_172: "f32[128, 128]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_25: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
    view_280: "f32[128]" = torch.ops.aten.view.default(sum_25, [128]);  sum_25 = None
    permute_173: "f32[128, 128]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_281: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_16, [8, 401, 128]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_198: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(view_277, view_281);  view_277 = view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_174: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_273, [0, 2, 1, 3]);  view_273 = None
    view_282: "f32[8, 1, 128]" = torch.ops.aten.view.default(permute_174, [8, 1, 128]);  permute_174 = None
    sum_26: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(view_282, [0, 1], True)
    view_283: "f32[128]" = torch.ops.aten.view.default(sum_26, [128]);  sum_26 = None
    view_284: "f32[8, 128]" = torch.ops.aten.view.default(view_282, [8, 128]);  view_282 = None
    permute_175: "f32[128, 8]" = torch.ops.aten.permute.default(view_284, [1, 0])
    mm_18: "f32[128, 128]" = torch.ops.aten.mm.default(permute_175, view_238);  permute_175 = view_238 = None
    permute_176: "f32[128, 128]" = torch.ops.aten.permute.default(mm_18, [1, 0]);  mm_18 = None
    permute_177: "f32[128, 128]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_19: "f32[8, 128]" = torch.ops.aten.mm.default(view_284, permute_177);  view_284 = permute_177 = None
    view_285: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_19, [8, 1, 128]);  mm_19 = None
    permute_178: "f32[128, 128]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    full_7: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_5: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_7, view_285, 1, 0, 1);  full_7 = view_285 = None
    full_8: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_6: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_8, slice_scatter_5, 0, 0, 9223372036854775807);  full_8 = slice_scatter_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_199: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_198, slice_scatter_6);  add_198 = slice_scatter_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    sub_106: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_12, getitem_225);  cat_12 = getitem_225 = None
    mul_279: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_106, rsqrt_40);  sub_106 = None
    mul_280: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(add_199, primals_247);  primals_247 = None
    mul_281: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_280, 128)
    sum_27: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True)
    mul_282: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_280, mul_279);  mul_280 = None
    sum_28: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True);  mul_282 = None
    mul_283: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_279, sum_28);  sum_28 = None
    sub_107: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_281, sum_27);  mul_281 = sum_27 = None
    sub_108: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_107, mul_283);  sub_107 = mul_283 = None
    div_10: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_40, 128);  rsqrt_40 = None
    mul_284: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_10, sub_108);  div_10 = sub_108 = None
    mul_285: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(add_199, mul_279);  mul_279 = None
    sum_29: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 1]);  mul_285 = None
    sum_30: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_199, [0, 1]);  add_199 = None
    full_9: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_7: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_9, slice_scatter_4, 1, 0, 1);  full_9 = slice_scatter_4 = None
    full_10: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_8: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_10, slice_scatter_7, 0, 0, 9223372036854775807);  full_10 = slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_200: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_284, slice_scatter_8);  mul_284 = slice_scatter_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_71: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_200, 1, 0, 1)
    slice_72: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_200, 1, 1, 401);  add_200 = None
    full_11: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_9: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_11, slice_72, 1, 1, 9223372036854775807);  full_11 = slice_72 = None
    full_12: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_10: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_12, slice_scatter_9, 0, 0, 9223372036854775807);  full_12 = slice_scatter_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_73: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(mul_260, 1, 0, 1)
    slice_74: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(mul_260, 1, 1, 401);  mul_260 = None
    full_13: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_11: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_13, slice_74, 1, 1, 9223372036854775807);  full_13 = slice_74 = None
    full_14: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_12: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_14, slice_scatter_11, 0, 0, 9223372036854775807);  full_14 = slice_scatter_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    add_201: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_12);  slice_scatter_10 = slice_scatter_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    view_286: "f32[8, 128]" = torch.ops.aten.view.default(slice_73, [8, 128]);  slice_73 = None
    permute_179: "f32[128, 256]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    mm_20: "f32[8, 256]" = torch.ops.aten.mm.default(view_286, permute_179);  permute_179 = None
    permute_180: "f32[128, 8]" = torch.ops.aten.permute.default(view_286, [1, 0])
    mm_21: "f32[128, 256]" = torch.ops.aten.mm.default(permute_180, view_236);  permute_180 = view_236 = None
    permute_181: "f32[256, 128]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_31: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_286, [0], True);  view_286 = None
    view_287: "f32[128]" = torch.ops.aten.view.default(sum_31, [128]);  sum_31 = None
    permute_182: "f32[128, 256]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    view_288: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_20, [8, 1, 256]);  mm_20 = None
    mul_286: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_183, 0.7071067811865476)
    erf_25: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_286);  mul_286 = None
    add_202: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
    mul_287: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_202, 0.5);  add_202 = None
    mul_288: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_183, add_183)
    mul_289: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_288, -0.5);  mul_288 = None
    exp_7: "f32[8, 1, 256]" = torch.ops.aten.exp.default(mul_289);  mul_289 = None
    mul_290: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_291: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_183, mul_290);  add_183 = mul_290 = None
    add_203: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_287, mul_291);  mul_287 = mul_291 = None
    mul_292: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_288, add_203);  view_288 = add_203 = None
    sub_109: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(slice_55, getitem_223);  slice_55 = getitem_223 = None
    mul_293: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_109, rsqrt_39);  sub_109 = None
    mul_294: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_292, primals_243);  primals_243 = None
    mul_295: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_294, 256)
    sum_32: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_294, [2], True)
    mul_296: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_294, mul_293);  mul_294 = None
    sum_33: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True);  mul_296 = None
    mul_297: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_293, sum_33);  sum_33 = None
    sub_110: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(mul_295, sum_32);  mul_295 = sum_32 = None
    sub_111: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(sub_110, mul_297);  sub_110 = mul_297 = None
    div_11: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_39, 256);  rsqrt_39 = None
    mul_298: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(div_11, sub_111);  div_11 = sub_111 = None
    mul_299: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_292, mul_293);  mul_293 = None
    sum_34: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_299, [0, 1]);  mul_299 = None
    sum_35: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1]);  mul_292 = None
    full_15: "f32[8, 1, 256]" = torch.ops.aten.full.default([8, 1, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_13: "f32[8, 1, 256]" = torch.ops.aten.slice_scatter.default(full_15, mul_298, 0, 0, 9223372036854775807);  full_15 = mul_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_289: "f32[8, 256]" = torch.ops.aten.view.default(slice_scatter_13, [8, 256])
    permute_183: "f32[256, 256]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    mm_22: "f32[8, 256]" = torch.ops.aten.mm.default(view_289, permute_183);  permute_183 = None
    permute_184: "f32[256, 8]" = torch.ops.aten.permute.default(view_289, [1, 0])
    mm_23: "f32[256, 256]" = torch.ops.aten.mm.default(permute_184, view_234);  permute_184 = view_234 = None
    permute_185: "f32[256, 256]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_36: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_289, [0], True);  view_289 = None
    view_290: "f32[256]" = torch.ops.aten.view.default(sum_36, [256]);  sum_36 = None
    permute_186: "f32[256, 256]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
    view_291: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_22, [8, 1, 256]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    view_292: "f32[8, 1, 4, 64]" = torch.ops.aten.view.default(view_291, [8, 1, 4, 64]);  view_291 = None
    permute_187: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    view_293: "f32[32, 1, 64]" = torch.ops.aten.view.default(permute_187, [32, 1, 64]);  permute_187 = None
    permute_188: "f32[32, 197, 1]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
    bmm_16: "f32[32, 197, 64]" = torch.ops.aten.bmm.default(permute_188, view_293);  permute_188 = None
    permute_189: "f32[32, 64, 197]" = torch.ops.aten.permute.default(view_231, [0, 2, 1]);  view_231 = None
    bmm_17: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_293, permute_189);  view_293 = permute_189 = None
    view_294: "f32[8, 4, 197, 64]" = torch.ops.aten.view.default(bmm_16, [8, 4, 197, 64]);  bmm_16 = None
    view_295: "f32[8, 4, 1, 197]" = torch.ops.aten.view.default(bmm_17, [8, 4, 1, 197]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    alias_19: "f32[8, 4, 1, 197]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    mul_300: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(view_295, alias_19);  view_295 = None
    sum_37: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [-1], True)
    mul_301: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(alias_19, sum_37);  alias_19 = sum_37 = None
    sub_112: "f32[8, 4, 1, 197]" = torch.ops.aten.sub.Tensor(mul_300, mul_301);  mul_300 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    mul_302: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(sub_112, 0.125);  sub_112 = None
    view_296: "f32[32, 1, 197]" = torch.ops.aten.view.default(mul_302, [32, 1, 197]);  mul_302 = None
    permute_190: "f32[32, 64, 1]" = torch.ops.aten.permute.default(view_227, [0, 2, 1]);  view_227 = None
    bmm_18: "f32[32, 64, 197]" = torch.ops.aten.bmm.default(permute_190, view_296);  permute_190 = None
    permute_191: "f32[32, 197, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
    bmm_19: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_296, permute_191);  view_296 = permute_191 = None
    view_297: "f32[8, 4, 64, 197]" = torch.ops.aten.view.default(bmm_18, [8, 4, 64, 197]);  bmm_18 = None
    view_298: "f32[8, 4, 1, 64]" = torch.ops.aten.view.default(bmm_19, [8, 4, 1, 64]);  bmm_19 = None
    permute_192: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_297, [0, 1, 3, 2]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_193: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(view_294, [0, 2, 1, 3]);  view_294 = None
    clone_72: "f32[8, 197, 4, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    view_299: "f32[8, 197, 256]" = torch.ops.aten.view.default(clone_72, [8, 197, 256]);  clone_72 = None
    view_300: "f32[1576, 256]" = torch.ops.aten.view.default(view_299, [1576, 256]);  view_299 = None
    permute_194: "f32[256, 256]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    mm_24: "f32[1576, 256]" = torch.ops.aten.mm.default(view_300, permute_194);  permute_194 = None
    permute_195: "f32[256, 1576]" = torch.ops.aten.permute.default(view_300, [1, 0])
    mm_25: "f32[256, 256]" = torch.ops.aten.mm.default(permute_195, view_224);  permute_195 = view_224 = None
    permute_196: "f32[256, 256]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_38: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
    view_301: "f32[256]" = torch.ops.aten.view.default(sum_38, [256]);  sum_38 = None
    permute_197: "f32[256, 256]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_302: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_24, [8, 197, 256]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_198: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(permute_192, [0, 2, 1, 3]);  permute_192 = None
    view_303: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_198, [8, 197, 256]);  permute_198 = None
    clone_73: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_303, memory_format = torch.contiguous_format);  view_303 = None
    view_304: "f32[1576, 256]" = torch.ops.aten.view.default(clone_73, [1576, 256]);  clone_73 = None
    permute_199: "f32[256, 256]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
    mm_26: "f32[1576, 256]" = torch.ops.aten.mm.default(view_304, permute_199);  permute_199 = None
    permute_200: "f32[256, 1576]" = torch.ops.aten.permute.default(view_304, [1, 0])
    mm_27: "f32[256, 256]" = torch.ops.aten.mm.default(permute_200, view_221);  permute_200 = view_221 = None
    permute_201: "f32[256, 256]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_39: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
    view_305: "f32[256]" = torch.ops.aten.view.default(sum_39, [256]);  sum_39 = None
    permute_202: "f32[256, 256]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_306: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_26, [8, 197, 256]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_204: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(view_302, view_306);  view_302 = view_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_203: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
    view_307: "f32[8, 1, 256]" = torch.ops.aten.view.default(permute_203, [8, 1, 256]);  permute_203 = None
    sum_40: "f32[1, 1, 256]" = torch.ops.aten.sum.dim_IntList(view_307, [0, 1], True)
    view_308: "f32[256]" = torch.ops.aten.view.default(sum_40, [256]);  sum_40 = None
    view_309: "f32[8, 256]" = torch.ops.aten.view.default(view_307, [8, 256]);  view_307 = None
    permute_204: "f32[256, 8]" = torch.ops.aten.permute.default(view_309, [1, 0])
    mm_28: "f32[256, 256]" = torch.ops.aten.mm.default(permute_204, view_218);  permute_204 = view_218 = None
    permute_205: "f32[256, 256]" = torch.ops.aten.permute.default(mm_28, [1, 0]);  mm_28 = None
    permute_206: "f32[256, 256]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_29: "f32[8, 256]" = torch.ops.aten.mm.default(view_309, permute_206);  view_309 = permute_206 = None
    view_310: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_29, [8, 1, 256]);  mm_29 = None
    permute_207: "f32[256, 256]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
    full_16: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_14: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_16, view_310, 1, 0, 1);  full_16 = view_310 = None
    full_17: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_15: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_17, slice_scatter_14, 0, 0, 9223372036854775807);  full_17 = slice_scatter_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_205: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_204, slice_scatter_15);  add_204 = slice_scatter_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    sub_113: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_10, getitem_221);  cat_10 = getitem_221 = None
    mul_303: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_113, rsqrt_38);  sub_113 = None
    mul_304: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(add_205, primals_233);  primals_233 = None
    mul_305: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_304, 256)
    sum_41: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [2], True)
    mul_306: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_304, mul_303);  mul_304 = None
    sum_42: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [2], True);  mul_306 = None
    mul_307: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_303, sum_42);  sum_42 = None
    sub_114: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_305, sum_41);  mul_305 = sum_41 = None
    sub_115: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_114, mul_307);  sub_114 = mul_307 = None
    div_12: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_38, 256);  rsqrt_38 = None
    mul_308: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_12, sub_115);  div_12 = sub_115 = None
    mul_309: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(add_205, mul_303);  mul_303 = None
    sum_43: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_309, [0, 1]);  mul_309 = None
    sum_44: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_205, [0, 1]);  add_205 = None
    full_18: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_16: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_18, slice_scatter_13, 1, 0, 1);  full_18 = slice_scatter_13 = None
    full_19: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_17: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_19, slice_scatter_16, 0, 0, 9223372036854775807);  full_19 = slice_scatter_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_206: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_308, slice_scatter_17);  mul_308 = slice_scatter_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_75: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_206, 1, 0, 1)
    slice_76: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_206, 1, 1, 197);  add_206 = None
    full_20: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_18: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_20, slice_76, 1, 1, 9223372036854775807);  full_20 = slice_76 = None
    full_21: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_19: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_21, slice_scatter_18, 0, 0, 9223372036854775807);  full_21 = slice_scatter_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    add_207: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(slice_scatter_3, slice_scatter_19);  slice_scatter_3 = slice_scatter_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    view_311: "f32[8, 128]" = torch.ops.aten.view.default(slice_71, [8, 128]);  slice_71 = None
    permute_208: "f32[128, 256]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_30: "f32[8, 256]" = torch.ops.aten.mm.default(view_311, permute_208);  permute_208 = None
    permute_209: "f32[128, 8]" = torch.ops.aten.permute.default(view_311, [1, 0])
    mm_31: "f32[128, 256]" = torch.ops.aten.mm.default(permute_209, view_216);  permute_209 = view_216 = None
    permute_210: "f32[256, 128]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_45: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
    view_312: "f32[128]" = torch.ops.aten.view.default(sum_45, [128]);  sum_45 = None
    permute_211: "f32[128, 256]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_313: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_30, [8, 1, 256]);  mm_30 = None
    mul_310: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_176, 0.7071067811865476)
    erf_26: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_310);  mul_310 = None
    add_208: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_26, 1);  erf_26 = None
    mul_311: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_208, 0.5);  add_208 = None
    mul_312: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_176, add_176)
    mul_313: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_312, -0.5);  mul_312 = None
    exp_8: "f32[8, 1, 256]" = torch.ops.aten.exp.default(mul_313);  mul_313 = None
    mul_314: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_315: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_176, mul_314);  add_176 = mul_314 = None
    add_209: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_311, mul_315);  mul_311 = mul_315 = None
    mul_316: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_313, add_209);  view_313 = add_209 = None
    clone_74: "f32[8, 1, 256]" = torch.ops.aten.clone.default(slice_48, memory_format = torch.contiguous_format);  slice_48 = None
    sub_116: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(clone_74, getitem_219);  clone_74 = getitem_219 = None
    mul_317: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_116, rsqrt_37);  sub_116 = None
    mul_318: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_316, primals_229);  primals_229 = None
    mul_319: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_318, 256)
    sum_46: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
    mul_320: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_318, mul_317);  mul_318 = None
    sum_47: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
    mul_321: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_317, sum_47);  sum_47 = None
    sub_117: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(mul_319, sum_46);  mul_319 = sum_46 = None
    sub_118: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(sub_117, mul_321);  sub_117 = mul_321 = None
    div_13: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_37, 256);  rsqrt_37 = None
    mul_322: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(div_13, sub_118);  div_13 = sub_118 = None
    mul_323: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_316, mul_317);  mul_317 = None
    sum_48: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
    sum_49: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
    full_22: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_20: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_22, mul_322, 1, 0, 1);  full_22 = mul_322 = None
    full_23: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_21: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_23, slice_scatter_20, 0, 0, 9223372036854775807);  full_23 = slice_scatter_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    add_210: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_207, slice_scatter_21);  add_207 = slice_scatter_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    view_314: "f32[8, 256]" = torch.ops.aten.view.default(slice_75, [8, 256]);  slice_75 = None
    permute_212: "f32[256, 128]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_32: "f32[8, 128]" = torch.ops.aten.mm.default(view_314, permute_212);  permute_212 = None
    permute_213: "f32[256, 8]" = torch.ops.aten.permute.default(view_314, [1, 0])
    mm_33: "f32[256, 128]" = torch.ops.aten.mm.default(permute_213, view_214);  permute_213 = view_214 = None
    permute_214: "f32[128, 256]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_50: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_314, [0], True);  view_314 = None
    view_315: "f32[256]" = torch.ops.aten.view.default(sum_50, [256]);  sum_50 = None
    permute_215: "f32[256, 128]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_316: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_32, [8, 1, 128]);  mm_32 = None
    mul_324: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_173, 0.7071067811865476)
    erf_27: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_324);  mul_324 = None
    add_211: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_27, 1);  erf_27 = None
    mul_325: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_211, 0.5);  add_211 = None
    mul_326: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_173, add_173)
    mul_327: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_326, -0.5);  mul_326 = None
    exp_9: "f32[8, 1, 128]" = torch.ops.aten.exp.default(mul_327);  mul_327 = None
    mul_328: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_329: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_173, mul_328);  add_173 = mul_328 = None
    add_212: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_325, mul_329);  mul_325 = mul_329 = None
    mul_330: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_316, add_212);  view_316 = add_212 = None
    clone_75: "f32[8, 1, 128]" = torch.ops.aten.clone.default(slice_46, memory_format = torch.contiguous_format);  slice_46 = None
    sub_119: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(clone_75, getitem_217);  clone_75 = getitem_217 = None
    mul_331: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_119, rsqrt_36);  sub_119 = None
    mul_332: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_330, primals_225);  primals_225 = None
    mul_333: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_332, 128)
    sum_51: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True)
    mul_334: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_332, mul_331);  mul_332 = None
    sum_52: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True);  mul_334 = None
    mul_335: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_331, sum_52);  sum_52 = None
    sub_120: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(mul_333, sum_51);  mul_333 = sum_51 = None
    sub_121: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(sub_120, mul_335);  sub_120 = mul_335 = None
    div_14: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_36, 128);  rsqrt_36 = None
    mul_336: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(div_14, sub_121);  div_14 = sub_121 = None
    mul_337: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_330, mul_331);  mul_331 = None
    sum_53: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 1]);  mul_337 = None
    sum_54: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
    full_24: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_22: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_24, mul_336, 1, 0, 1);  full_24 = mul_336 = None
    full_25: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_23: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_25, slice_scatter_22, 0, 0, 9223372036854775807);  full_25 = slice_scatter_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    add_213: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_201, slice_scatter_23);  add_201 = slice_scatter_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_317: "f32[1576, 256]" = torch.ops.aten.view.default(add_210, [1576, 256])
    permute_216: "f32[256, 768]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    mm_34: "f32[1576, 768]" = torch.ops.aten.mm.default(view_317, permute_216);  permute_216 = None
    permute_217: "f32[256, 1576]" = torch.ops.aten.permute.default(view_317, [1, 0])
    mm_35: "f32[256, 768]" = torch.ops.aten.mm.default(permute_217, view_212);  permute_217 = view_212 = None
    permute_218: "f32[768, 256]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_55: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_317, [0], True);  view_317 = None
    view_318: "f32[256]" = torch.ops.aten.view.default(sum_55, [256]);  sum_55 = None
    permute_219: "f32[256, 768]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
    view_319: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_34, [8, 197, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_338: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, 0.7071067811865476)
    erf_28: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_338);  mul_338 = None
    add_214: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_28, 1);  erf_28 = None
    mul_339: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_214, 0.5);  add_214 = None
    mul_340: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, view_211)
    mul_341: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_340, -0.5);  mul_340 = None
    exp_10: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_341);  mul_341 = None
    mul_342: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_343: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, mul_342);  view_211 = mul_342 = None
    add_215: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_339, mul_343);  mul_339 = mul_343 = None
    mul_344: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_319, add_215);  view_319 = add_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_320: "f32[1576, 768]" = torch.ops.aten.view.default(mul_344, [1576, 768]);  mul_344 = None
    permute_220: "f32[768, 256]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    mm_36: "f32[1576, 256]" = torch.ops.aten.mm.default(view_320, permute_220);  permute_220 = None
    permute_221: "f32[768, 1576]" = torch.ops.aten.permute.default(view_320, [1, 0])
    mm_37: "f32[768, 256]" = torch.ops.aten.mm.default(permute_221, view_210);  permute_221 = view_210 = None
    permute_222: "f32[256, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_320, [0], True);  view_320 = None
    view_321: "f32[768]" = torch.ops.aten.view.default(sum_56, [768]);  sum_56 = None
    permute_223: "f32[768, 256]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    view_322: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_36, [8, 197, 256]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_122: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_167, getitem_215);  add_167 = getitem_215 = None
    mul_345: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_122, rsqrt_35);  sub_122 = None
    mul_346: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_322, primals_219);  primals_219 = None
    mul_347: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_346, 256)
    sum_57: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_346, [2], True)
    mul_348: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_346, mul_345);  mul_346 = None
    sum_58: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_348, [2], True);  mul_348 = None
    mul_349: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_345, sum_58);  sum_58 = None
    sub_123: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_347, sum_57);  mul_347 = sum_57 = None
    sub_124: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_123, mul_349);  sub_123 = mul_349 = None
    div_15: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_35, 256);  rsqrt_35 = None
    mul_350: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_15, sub_124);  div_15 = sub_124 = None
    mul_351: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_322, mul_345);  mul_345 = None
    sum_59: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_351, [0, 1]);  mul_351 = None
    sum_60: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_322, [0, 1]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_216: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_210, mul_350);  add_210 = mul_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_323: "f32[1576, 256]" = torch.ops.aten.view.default(add_216, [1576, 256])
    permute_224: "f32[256, 256]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    mm_38: "f32[1576, 256]" = torch.ops.aten.mm.default(view_323, permute_224);  permute_224 = None
    permute_225: "f32[256, 1576]" = torch.ops.aten.permute.default(view_323, [1, 0])
    mm_39: "f32[256, 256]" = torch.ops.aten.mm.default(permute_225, view_208);  permute_225 = view_208 = None
    permute_226: "f32[256, 256]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_61: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
    view_324: "f32[256]" = torch.ops.aten.view.default(sum_61, [256]);  sum_61 = None
    permute_227: "f32[256, 256]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    view_325: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_38, [8, 197, 256]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_326: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_325, [8, 197, 4, 64]);  view_325 = None
    permute_228: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_20: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_228, getitem_202, getitem_203, getitem_204, alias_20, getitem_206, getitem_207, getitem_208, 0, 0, 0.0, False, getitem_211, getitem_212);  permute_228 = getitem_202 = getitem_203 = getitem_204 = alias_20 = getitem_206 = getitem_207 = getitem_208 = getitem_211 = getitem_212 = None
    getitem_232: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward[0]
    getitem_233: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward[1]
    getitem_234: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_15: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_232, getitem_233, getitem_234]);  getitem_232 = getitem_233 = getitem_234 = None
    view_327: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_15, [3, 8, 4, 197, 64]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_229: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_327, [1, 3, 0, 2, 4]);  view_327 = None
    clone_76: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
    view_328: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_76, [8, 197, 768]);  clone_76 = None
    view_329: "f32[1576, 768]" = torch.ops.aten.view.default(view_328, [1576, 768]);  view_328 = None
    permute_230: "f32[768, 256]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_40: "f32[1576, 256]" = torch.ops.aten.mm.default(view_329, permute_230);  permute_230 = None
    permute_231: "f32[768, 1576]" = torch.ops.aten.permute.default(view_329, [1, 0])
    mm_41: "f32[768, 256]" = torch.ops.aten.mm.default(permute_231, view_204);  permute_231 = view_204 = None
    permute_232: "f32[256, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_329, [0], True);  view_329 = None
    view_330: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    permute_233: "f32[768, 256]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    view_331: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_40, [8, 197, 256]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_125: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_164, getitem_201);  add_164 = getitem_201 = None
    mul_352: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_125, rsqrt_34);  sub_125 = None
    mul_353: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_331, primals_213);  primals_213 = None
    mul_354: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_353, 256)
    sum_63: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_353, [2], True)
    mul_355: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_353, mul_352);  mul_353 = None
    sum_64: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True);  mul_355 = None
    mul_356: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_352, sum_64);  sum_64 = None
    sub_126: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_354, sum_63);  mul_354 = sum_63 = None
    sub_127: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_126, mul_356);  sub_126 = mul_356 = None
    div_16: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_34, 256);  rsqrt_34 = None
    mul_357: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_16, sub_127);  div_16 = sub_127 = None
    mul_358: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_331, mul_352);  mul_352 = None
    sum_65: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_358, [0, 1]);  mul_358 = None
    sum_66: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_331, [0, 1]);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_217: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_216, mul_357);  add_216 = mul_357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_332: "f32[1576, 256]" = torch.ops.aten.view.default(add_217, [1576, 256])
    permute_234: "f32[256, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    mm_42: "f32[1576, 768]" = torch.ops.aten.mm.default(view_332, permute_234);  permute_234 = None
    permute_235: "f32[256, 1576]" = torch.ops.aten.permute.default(view_332, [1, 0])
    mm_43: "f32[256, 768]" = torch.ops.aten.mm.default(permute_235, view_202);  permute_235 = view_202 = None
    permute_236: "f32[768, 256]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_67: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_332, [0], True);  view_332 = None
    view_333: "f32[256]" = torch.ops.aten.view.default(sum_67, [256]);  sum_67 = None
    permute_237: "f32[256, 768]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_334: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_42, [8, 197, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_359: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_201, 0.7071067811865476)
    erf_29: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_359);  mul_359 = None
    add_218: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_29, 1);  erf_29 = None
    mul_360: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_218, 0.5);  add_218 = None
    mul_361: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_201, view_201)
    mul_362: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_361, -0.5);  mul_361 = None
    exp_11: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_362);  mul_362 = None
    mul_363: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_364: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_201, mul_363);  view_201 = mul_363 = None
    add_219: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_360, mul_364);  mul_360 = mul_364 = None
    mul_365: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_334, add_219);  view_334 = add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_335: "f32[1576, 768]" = torch.ops.aten.view.default(mul_365, [1576, 768]);  mul_365 = None
    permute_238: "f32[768, 256]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_44: "f32[1576, 256]" = torch.ops.aten.mm.default(view_335, permute_238);  permute_238 = None
    permute_239: "f32[768, 1576]" = torch.ops.aten.permute.default(view_335, [1, 0])
    mm_45: "f32[768, 256]" = torch.ops.aten.mm.default(permute_239, view_200);  permute_239 = view_200 = None
    permute_240: "f32[256, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
    view_336: "f32[768]" = torch.ops.aten.view.default(sum_68, [768]);  sum_68 = None
    permute_241: "f32[768, 256]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
    view_337: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_44, [8, 197, 256]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_128: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_160, getitem_199);  add_160 = getitem_199 = None
    mul_366: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_128, rsqrt_33);  sub_128 = None
    mul_367: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_337, primals_207);  primals_207 = None
    mul_368: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_367, 256)
    sum_69: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
    mul_369: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_367, mul_366);  mul_367 = None
    sum_70: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
    mul_370: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_366, sum_70);  sum_70 = None
    sub_129: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_368, sum_69);  mul_368 = sum_69 = None
    sub_130: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_129, mul_370);  sub_129 = mul_370 = None
    div_17: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_33, 256);  rsqrt_33 = None
    mul_371: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_17, sub_130);  div_17 = sub_130 = None
    mul_372: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_337, mul_366);  mul_366 = None
    sum_71: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
    sum_72: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_337, [0, 1]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_220: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_217, mul_371);  add_217 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_338: "f32[1576, 256]" = torch.ops.aten.view.default(add_220, [1576, 256])
    permute_242: "f32[256, 256]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_46: "f32[1576, 256]" = torch.ops.aten.mm.default(view_338, permute_242);  permute_242 = None
    permute_243: "f32[256, 1576]" = torch.ops.aten.permute.default(view_338, [1, 0])
    mm_47: "f32[256, 256]" = torch.ops.aten.mm.default(permute_243, view_198);  permute_243 = view_198 = None
    permute_244: "f32[256, 256]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_73: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_338, [0], True);  view_338 = None
    view_339: "f32[256]" = torch.ops.aten.view.default(sum_73, [256]);  sum_73 = None
    permute_245: "f32[256, 256]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    view_340: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_46, [8, 197, 256]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_341: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_340, [8, 197, 4, 64]);  view_340 = None
    permute_246: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_21: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_246, getitem_186, getitem_187, getitem_188, alias_21, getitem_190, getitem_191, getitem_192, 0, 0, 0.0, False, getitem_195, getitem_196);  permute_246 = getitem_186 = getitem_187 = getitem_188 = alias_21 = getitem_190 = getitem_191 = getitem_192 = getitem_195 = getitem_196 = None
    getitem_235: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_1[0]
    getitem_236: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_1[1]
    getitem_237: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_16: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_235, getitem_236, getitem_237]);  getitem_235 = getitem_236 = getitem_237 = None
    view_342: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_16, [3, 8, 4, 197, 64]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_247: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_342, [1, 3, 0, 2, 4]);  view_342 = None
    clone_77: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_247, memory_format = torch.contiguous_format);  permute_247 = None
    view_343: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_77, [8, 197, 768]);  clone_77 = None
    view_344: "f32[1576, 768]" = torch.ops.aten.view.default(view_343, [1576, 768]);  view_343 = None
    permute_248: "f32[768, 256]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    mm_48: "f32[1576, 256]" = torch.ops.aten.mm.default(view_344, permute_248);  permute_248 = None
    permute_249: "f32[768, 1576]" = torch.ops.aten.permute.default(view_344, [1, 0])
    mm_49: "f32[768, 256]" = torch.ops.aten.mm.default(permute_249, view_194);  permute_249 = view_194 = None
    permute_250: "f32[256, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_344, [0], True);  view_344 = None
    view_345: "f32[768]" = torch.ops.aten.view.default(sum_74, [768]);  sum_74 = None
    permute_251: "f32[768, 256]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    view_346: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_48, [8, 197, 256]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_131: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_157, getitem_185);  add_157 = getitem_185 = None
    mul_373: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_131, rsqrt_32);  sub_131 = None
    mul_374: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_346, primals_201);  primals_201 = None
    mul_375: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_374, 256)
    sum_75: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True)
    mul_376: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_374, mul_373);  mul_374 = None
    sum_76: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [2], True);  mul_376 = None
    mul_377: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_373, sum_76);  sum_76 = None
    sub_132: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_375, sum_75);  mul_375 = sum_75 = None
    sub_133: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_132, mul_377);  sub_132 = mul_377 = None
    div_18: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_32, 256);  rsqrt_32 = None
    mul_378: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_18, sub_133);  div_18 = sub_133 = None
    mul_379: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_346, mul_373);  mul_373 = None
    sum_77: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 1]);  mul_379 = None
    sum_78: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_346, [0, 1]);  view_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_221: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_220, mul_378);  add_220 = mul_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_347: "f32[1576, 256]" = torch.ops.aten.view.default(add_221, [1576, 256])
    permute_252: "f32[256, 768]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
    mm_50: "f32[1576, 768]" = torch.ops.aten.mm.default(view_347, permute_252);  permute_252 = None
    permute_253: "f32[256, 1576]" = torch.ops.aten.permute.default(view_347, [1, 0])
    mm_51: "f32[256, 768]" = torch.ops.aten.mm.default(permute_253, view_192);  permute_253 = view_192 = None
    permute_254: "f32[768, 256]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_79: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_347, [0], True);  view_347 = None
    view_348: "f32[256]" = torch.ops.aten.view.default(sum_79, [256]);  sum_79 = None
    permute_255: "f32[256, 768]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    view_349: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_50, [8, 197, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_380: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_191, 0.7071067811865476)
    erf_30: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_380);  mul_380 = None
    add_222: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_30, 1);  erf_30 = None
    mul_381: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_222, 0.5);  add_222 = None
    mul_382: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_191, view_191)
    mul_383: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_382, -0.5);  mul_382 = None
    exp_12: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_383);  mul_383 = None
    mul_384: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_385: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_191, mul_384);  view_191 = mul_384 = None
    add_223: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_381, mul_385);  mul_381 = mul_385 = None
    mul_386: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_349, add_223);  view_349 = add_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_350: "f32[1576, 768]" = torch.ops.aten.view.default(mul_386, [1576, 768]);  mul_386 = None
    permute_256: "f32[768, 256]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    mm_52: "f32[1576, 256]" = torch.ops.aten.mm.default(view_350, permute_256);  permute_256 = None
    permute_257: "f32[768, 1576]" = torch.ops.aten.permute.default(view_350, [1, 0])
    mm_53: "f32[768, 256]" = torch.ops.aten.mm.default(permute_257, view_190);  permute_257 = view_190 = None
    permute_258: "f32[256, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_80: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
    view_351: "f32[768]" = torch.ops.aten.view.default(sum_80, [768]);  sum_80 = None
    permute_259: "f32[768, 256]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
    view_352: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_52, [8, 197, 256]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_134: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_153, getitem_183);  add_153 = getitem_183 = None
    mul_387: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_134, rsqrt_31);  sub_134 = None
    mul_388: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_352, primals_195);  primals_195 = None
    mul_389: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_388, 256)
    sum_81: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_388, [2], True)
    mul_390: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_388, mul_387);  mul_388 = None
    sum_82: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True);  mul_390 = None
    mul_391: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_387, sum_82);  sum_82 = None
    sub_135: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_389, sum_81);  mul_389 = sum_81 = None
    sub_136: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_135, mul_391);  sub_135 = mul_391 = None
    div_19: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_31, 256);  rsqrt_31 = None
    mul_392: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_19, sub_136);  div_19 = sub_136 = None
    mul_393: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_352, mul_387);  mul_387 = None
    sum_83: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_393, [0, 1]);  mul_393 = None
    sum_84: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_352, [0, 1]);  view_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_224: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_221, mul_392);  add_221 = mul_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_353: "f32[1576, 256]" = torch.ops.aten.view.default(add_224, [1576, 256])
    permute_260: "f32[256, 256]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    mm_54: "f32[1576, 256]" = torch.ops.aten.mm.default(view_353, permute_260);  permute_260 = None
    permute_261: "f32[256, 1576]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_55: "f32[256, 256]" = torch.ops.aten.mm.default(permute_261, view_188);  permute_261 = view_188 = None
    permute_262: "f32[256, 256]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_85: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[256]" = torch.ops.aten.view.default(sum_85, [256]);  sum_85 = None
    permute_263: "f32[256, 256]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    view_355: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_54, [8, 197, 256]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_356: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_355, [8, 197, 4, 64]);  view_355 = None
    permute_264: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_22: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    _scaled_dot_product_flash_attention_backward_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_264, getitem_170, getitem_171, getitem_172, alias_22, getitem_174, getitem_175, getitem_176, 0, 0, 0.0, False, getitem_179, getitem_180);  permute_264 = getitem_170 = getitem_171 = getitem_172 = alias_22 = getitem_174 = getitem_175 = getitem_176 = getitem_179 = getitem_180 = None
    getitem_238: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_2[0]
    getitem_239: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_2[1]
    getitem_240: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_2[2];  _scaled_dot_product_flash_attention_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_17: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_238, getitem_239, getitem_240]);  getitem_238 = getitem_239 = getitem_240 = None
    view_357: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_17, [3, 8, 4, 197, 64]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_265: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_357, [1, 3, 0, 2, 4]);  view_357 = None
    clone_78: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    view_358: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_78, [8, 197, 768]);  clone_78 = None
    view_359: "f32[1576, 768]" = torch.ops.aten.view.default(view_358, [1576, 768]);  view_358 = None
    permute_266: "f32[768, 256]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    mm_56: "f32[1576, 256]" = torch.ops.aten.mm.default(view_359, permute_266);  permute_266 = None
    permute_267: "f32[768, 1576]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_57: "f32[768, 256]" = torch.ops.aten.mm.default(permute_267, view_184);  permute_267 = view_184 = None
    permute_268: "f32[256, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[768]" = torch.ops.aten.view.default(sum_86, [768]);  sum_86 = None
    permute_269: "f32[768, 256]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_361: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_56, [8, 197, 256]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_137: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_9, getitem_169);  cat_9 = getitem_169 = None
    mul_394: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_137, rsqrt_30);  sub_137 = None
    mul_395: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_361, primals_189);  primals_189 = None
    mul_396: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_395, 256)
    sum_87: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
    mul_397: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_395, mul_394);  mul_395 = None
    sum_88: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
    mul_398: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_394, sum_88);  sum_88 = None
    sub_138: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_396, sum_87);  mul_396 = sum_87 = None
    sub_139: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_138, mul_398);  sub_138 = mul_398 = None
    div_20: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_30, 256);  rsqrt_30 = None
    mul_399: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_20, sub_139);  div_20 = sub_139 = None
    mul_400: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_361, mul_394);  mul_394 = None
    sum_89: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
    sum_90: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_361, [0, 1]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_225: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_224, mul_399);  add_224 = mul_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_362: "f32[3208, 128]" = torch.ops.aten.view.default(add_213, [3208, 128])
    permute_270: "f32[128, 384]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_58: "f32[3208, 384]" = torch.ops.aten.mm.default(view_362, permute_270);  permute_270 = None
    permute_271: "f32[128, 3208]" = torch.ops.aten.permute.default(view_362, [1, 0])
    mm_59: "f32[128, 384]" = torch.ops.aten.mm.default(permute_271, view_182);  permute_271 = view_182 = None
    permute_272: "f32[384, 128]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_91: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_362, [0], True);  view_362 = None
    view_363: "f32[128]" = torch.ops.aten.view.default(sum_91, [128]);  sum_91 = None
    permute_273: "f32[128, 384]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_364: "f32[8, 401, 384]" = torch.ops.aten.view.default(mm_58, [8, 401, 384]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_401: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_181, 0.7071067811865476)
    erf_31: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_401);  mul_401 = None
    add_226: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf_31, 1);  erf_31 = None
    mul_402: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(add_226, 0.5);  add_226 = None
    mul_403: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_181, view_181)
    mul_404: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_403, -0.5);  mul_403 = None
    exp_13: "f32[8, 401, 384]" = torch.ops.aten.exp.default(mul_404);  mul_404 = None
    mul_405: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_406: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_181, mul_405);  view_181 = mul_405 = None
    add_227: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(mul_402, mul_406);  mul_402 = mul_406 = None
    mul_407: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_364, add_227);  view_364 = add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_365: "f32[3208, 384]" = torch.ops.aten.view.default(mul_407, [3208, 384]);  mul_407 = None
    permute_274: "f32[384, 128]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_60: "f32[3208, 128]" = torch.ops.aten.mm.default(view_365, permute_274);  permute_274 = None
    permute_275: "f32[384, 3208]" = torch.ops.aten.permute.default(view_365, [1, 0])
    mm_61: "f32[384, 128]" = torch.ops.aten.mm.default(permute_275, view_180);  permute_275 = view_180 = None
    permute_276: "f32[128, 384]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_92: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_365, [0], True);  view_365 = None
    view_366: "f32[384]" = torch.ops.aten.view.default(sum_92, [384]);  sum_92 = None
    permute_277: "f32[384, 128]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_367: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_60, [8, 401, 128]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_140: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add_146, getitem_167);  add_146 = getitem_167 = None
    mul_408: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_140, rsqrt_29);  sub_140 = None
    mul_409: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_367, primals_183);  primals_183 = None
    mul_410: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_409, 128)
    sum_93: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_409, [2], True)
    mul_411: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_409, mul_408);  mul_409 = None
    sum_94: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True);  mul_411 = None
    mul_412: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_408, sum_94);  sum_94 = None
    sub_141: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_410, sum_93);  mul_410 = sum_93 = None
    sub_142: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_141, mul_412);  sub_141 = mul_412 = None
    div_21: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_29, 128);  rsqrt_29 = None
    mul_413: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_21, sub_142);  div_21 = sub_142 = None
    mul_414: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_367, mul_408);  mul_408 = None
    sum_95: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_414, [0, 1]);  mul_414 = None
    sum_96: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_367, [0, 1]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_228: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_213, mul_413);  add_213 = mul_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_368: "f32[3208, 128]" = torch.ops.aten.view.default(add_228, [3208, 128])
    permute_278: "f32[128, 128]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_62: "f32[3208, 128]" = torch.ops.aten.mm.default(view_368, permute_278);  permute_278 = None
    permute_279: "f32[128, 3208]" = torch.ops.aten.permute.default(view_368, [1, 0])
    mm_63: "f32[128, 128]" = torch.ops.aten.mm.default(permute_279, view_178);  permute_279 = view_178 = None
    permute_280: "f32[128, 128]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_97: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_368, [0], True);  view_368 = None
    view_369: "f32[128]" = torch.ops.aten.view.default(sum_97, [128]);  sum_97 = None
    permute_281: "f32[128, 128]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_370: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_62, [8, 401, 128]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_371: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_370, [8, 401, 4, 32]);  view_370 = None
    permute_282: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_23: "f32[8, 4, 401, 32]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    _scaled_dot_product_flash_attention_backward_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_282, getitem_154, getitem_155, getitem_156, alias_23, getitem_158, getitem_159, getitem_160, 0, 0, 0.0, False, getitem_163, getitem_164);  permute_282 = getitem_154 = getitem_155 = getitem_156 = alias_23 = getitem_158 = getitem_159 = getitem_160 = getitem_163 = getitem_164 = None
    getitem_241: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_3[0]
    getitem_242: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_3[1]
    getitem_243: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_3[2];  _scaled_dot_product_flash_attention_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_18: "f32[24, 4, 401, 32]" = torch.ops.aten.cat.default([getitem_241, getitem_242, getitem_243]);  getitem_241 = getitem_242 = getitem_243 = None
    view_372: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.view.default(cat_18, [3, 8, 4, 401, 32]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_283: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.permute.default(view_372, [1, 3, 0, 2, 4]);  view_372 = None
    clone_79: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.clone.default(permute_283, memory_format = torch.contiguous_format);  permute_283 = None
    view_373: "f32[8, 401, 384]" = torch.ops.aten.view.default(clone_79, [8, 401, 384]);  clone_79 = None
    view_374: "f32[3208, 384]" = torch.ops.aten.view.default(view_373, [3208, 384]);  view_373 = None
    permute_284: "f32[384, 128]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    mm_64: "f32[3208, 128]" = torch.ops.aten.mm.default(view_374, permute_284);  permute_284 = None
    permute_285: "f32[384, 3208]" = torch.ops.aten.permute.default(view_374, [1, 0])
    mm_65: "f32[384, 128]" = torch.ops.aten.mm.default(permute_285, view_174);  permute_285 = view_174 = None
    permute_286: "f32[128, 384]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_98: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_374, [0], True);  view_374 = None
    view_375: "f32[384]" = torch.ops.aten.view.default(sum_98, [384]);  sum_98 = None
    permute_287: "f32[384, 128]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
    view_376: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_64, [8, 401, 128]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_143: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_7, getitem_153);  cat_7 = getitem_153 = None
    mul_415: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_143, rsqrt_28);  sub_143 = None
    mul_416: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_376, primals_177);  primals_177 = None
    mul_417: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_416, 128)
    sum_99: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
    mul_418: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_416, mul_415);  mul_416 = None
    sum_100: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
    mul_419: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_415, sum_100);  sum_100 = None
    sub_144: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_417, sum_99);  mul_417 = sum_99 = None
    sub_145: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_144, mul_419);  sub_144 = mul_419 = None
    div_22: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_28, 128);  rsqrt_28 = None
    mul_420: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_22, sub_145);  div_22 = sub_145 = None
    mul_421: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_376, mul_415);  mul_415 = None
    sum_101: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
    sum_102: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_376, [0, 1]);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_229: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_228, mul_420);  add_228 = mul_420 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_77: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_225, 1, 0, 1)
    slice_78: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_225, 1, 1, 197);  add_225 = None
    full_26: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_24: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_26, slice_78, 1, 1, 9223372036854775807);  full_26 = slice_78 = None
    full_27: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_25: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_27, slice_scatter_24, 0, 0, 9223372036854775807);  full_27 = slice_scatter_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    view_377: "f32[8, 256]" = torch.ops.aten.view.default(slice_77, [8, 256]);  slice_77 = None
    permute_288: "f32[256, 128]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    mm_66: "f32[8, 128]" = torch.ops.aten.mm.default(view_377, permute_288);  permute_288 = None
    permute_289: "f32[256, 8]" = torch.ops.aten.permute.default(view_377, [1, 0])
    mm_67: "f32[256, 128]" = torch.ops.aten.mm.default(permute_289, view_172);  permute_289 = view_172 = None
    permute_290: "f32[128, 256]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_103: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
    view_378: "f32[256]" = torch.ops.aten.view.default(sum_103, [256]);  sum_103 = None
    permute_291: "f32[256, 128]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
    view_379: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_66, [8, 1, 128]);  mm_66 = None
    mul_422: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_142, 0.7071067811865476)
    erf_32: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_422);  mul_422 = None
    add_230: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_32, 1);  erf_32 = None
    mul_423: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_230, 0.5);  add_230 = None
    mul_424: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_142, add_142)
    mul_425: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_424, -0.5);  mul_424 = None
    exp_14: "f32[8, 1, 128]" = torch.ops.aten.exp.default(mul_425);  mul_425 = None
    mul_426: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_427: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_142, mul_426);  add_142 = mul_426 = None
    add_231: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_423, mul_427);  mul_423 = mul_427 = None
    mul_428: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_379, add_231);  view_379 = add_231 = None
    sub_146: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(slice_42, getitem_151);  slice_42 = getitem_151 = None
    mul_429: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_146, rsqrt_27);  sub_146 = None
    mul_430: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_428, primals_173);  primals_173 = None
    mul_431: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_430, 128)
    sum_104: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_430, [2], True)
    mul_432: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_430, mul_429);  mul_430 = None
    sum_105: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True);  mul_432 = None
    mul_433: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_429, sum_105);  sum_105 = None
    sub_147: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(mul_431, sum_104);  mul_431 = sum_104 = None
    sub_148: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(sub_147, mul_433);  sub_147 = mul_433 = None
    div_23: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_27, 128);  rsqrt_27 = None
    mul_434: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(div_23, sub_148);  div_23 = sub_148 = None
    mul_435: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_428, mul_429);  mul_429 = None
    sum_106: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_435, [0, 1]);  mul_435 = None
    sum_107: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 1]);  mul_428 = None
    full_28: "f32[8, 1, 128]" = torch.ops.aten.full.default([8, 1, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_26: "f32[8, 1, 128]" = torch.ops.aten.slice_scatter.default(full_28, mul_434, 0, 0, 9223372036854775807);  full_28 = mul_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_380: "f32[8, 128]" = torch.ops.aten.view.default(slice_scatter_26, [8, 128])
    permute_292: "f32[128, 128]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_68: "f32[8, 128]" = torch.ops.aten.mm.default(view_380, permute_292);  permute_292 = None
    permute_293: "f32[128, 8]" = torch.ops.aten.permute.default(view_380, [1, 0])
    mm_69: "f32[128, 128]" = torch.ops.aten.mm.default(permute_293, view_170);  permute_293 = view_170 = None
    permute_294: "f32[128, 128]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_108: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
    view_381: "f32[128]" = torch.ops.aten.view.default(sum_108, [128]);  sum_108 = None
    permute_295: "f32[128, 128]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
    view_382: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_68, [8, 1, 128]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    view_383: "f32[8, 1, 4, 32]" = torch.ops.aten.view.default(view_382, [8, 1, 4, 32]);  view_382 = None
    permute_296: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_383, [0, 2, 1, 3]);  view_383 = None
    view_384: "f32[32, 1, 32]" = torch.ops.aten.view.default(permute_296, [32, 1, 32]);  permute_296 = None
    permute_297: "f32[32, 401, 1]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_20: "f32[32, 401, 32]" = torch.ops.aten.bmm.default(permute_297, view_384);  permute_297 = None
    permute_298: "f32[32, 32, 401]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    bmm_21: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_384, permute_298);  view_384 = permute_298 = None
    view_385: "f32[8, 4, 401, 32]" = torch.ops.aten.view.default(bmm_20, [8, 4, 401, 32]);  bmm_20 = None
    view_386: "f32[8, 4, 1, 401]" = torch.ops.aten.view.default(bmm_21, [8, 4, 1, 401]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    alias_24: "f32[8, 4, 1, 401]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_436: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(view_386, alias_24);  view_386 = None
    sum_109: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [-1], True)
    mul_437: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(alias_24, sum_109);  alias_24 = sum_109 = None
    sub_149: "f32[8, 4, 1, 401]" = torch.ops.aten.sub.Tensor(mul_436, mul_437);  mul_436 = mul_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    mul_438: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(sub_149, 0.1767766952966369);  sub_149 = None
    view_387: "f32[32, 1, 401]" = torch.ops.aten.view.default(mul_438, [32, 1, 401]);  mul_438 = None
    permute_299: "f32[32, 32, 1]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
    bmm_22: "f32[32, 32, 401]" = torch.ops.aten.bmm.default(permute_299, view_387);  permute_299 = None
    permute_300: "f32[32, 401, 32]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
    bmm_23: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_387, permute_300);  view_387 = permute_300 = None
    view_388: "f32[8, 4, 32, 401]" = torch.ops.aten.view.default(bmm_22, [8, 4, 32, 401]);  bmm_22 = None
    view_389: "f32[8, 4, 1, 32]" = torch.ops.aten.view.default(bmm_23, [8, 4, 1, 32]);  bmm_23 = None
    permute_301: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_388, [0, 1, 3, 2]);  view_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_302: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(view_385, [0, 2, 1, 3]);  view_385 = None
    clone_80: "f32[8, 401, 4, 32]" = torch.ops.aten.clone.default(permute_302, memory_format = torch.contiguous_format);  permute_302 = None
    view_390: "f32[8, 401, 128]" = torch.ops.aten.view.default(clone_80, [8, 401, 128]);  clone_80 = None
    view_391: "f32[3208, 128]" = torch.ops.aten.view.default(view_390, [3208, 128]);  view_390 = None
    permute_303: "f32[128, 128]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_70: "f32[3208, 128]" = torch.ops.aten.mm.default(view_391, permute_303);  permute_303 = None
    permute_304: "f32[128, 3208]" = torch.ops.aten.permute.default(view_391, [1, 0])
    mm_71: "f32[128, 128]" = torch.ops.aten.mm.default(permute_304, view_160);  permute_304 = view_160 = None
    permute_305: "f32[128, 128]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_110: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_391, [0], True);  view_391 = None
    view_392: "f32[128]" = torch.ops.aten.view.default(sum_110, [128]);  sum_110 = None
    permute_306: "f32[128, 128]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
    view_393: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_70, [8, 401, 128]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_307: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(permute_301, [0, 2, 1, 3]);  permute_301 = None
    view_394: "f32[8, 401, 128]" = torch.ops.aten.view.default(permute_307, [8, 401, 128]);  permute_307 = None
    clone_81: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_394, memory_format = torch.contiguous_format);  view_394 = None
    view_395: "f32[3208, 128]" = torch.ops.aten.view.default(clone_81, [3208, 128]);  clone_81 = None
    permute_308: "f32[128, 128]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_72: "f32[3208, 128]" = torch.ops.aten.mm.default(view_395, permute_308);  permute_308 = None
    permute_309: "f32[128, 3208]" = torch.ops.aten.permute.default(view_395, [1, 0])
    mm_73: "f32[128, 128]" = torch.ops.aten.mm.default(permute_309, view_157);  permute_309 = view_157 = None
    permute_310: "f32[128, 128]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_111: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
    view_396: "f32[128]" = torch.ops.aten.view.default(sum_111, [128]);  sum_111 = None
    permute_311: "f32[128, 128]" = torch.ops.aten.permute.default(permute_310, [1, 0]);  permute_310 = None
    view_397: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_72, [8, 401, 128]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_232: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(view_393, view_397);  view_393 = view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_312: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_389, [0, 2, 1, 3]);  view_389 = None
    view_398: "f32[8, 1, 128]" = torch.ops.aten.view.default(permute_312, [8, 1, 128]);  permute_312 = None
    sum_112: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(view_398, [0, 1], True)
    view_399: "f32[128]" = torch.ops.aten.view.default(sum_112, [128]);  sum_112 = None
    view_400: "f32[8, 128]" = torch.ops.aten.view.default(view_398, [8, 128]);  view_398 = None
    permute_313: "f32[128, 8]" = torch.ops.aten.permute.default(view_400, [1, 0])
    mm_74: "f32[128, 128]" = torch.ops.aten.mm.default(permute_313, view_154);  permute_313 = view_154 = None
    permute_314: "f32[128, 128]" = torch.ops.aten.permute.default(mm_74, [1, 0]);  mm_74 = None
    permute_315: "f32[128, 128]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    mm_75: "f32[8, 128]" = torch.ops.aten.mm.default(view_400, permute_315);  view_400 = permute_315 = None
    view_401: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_75, [8, 1, 128]);  mm_75 = None
    permute_316: "f32[128, 128]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
    full_29: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_27: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_29, view_401, 1, 0, 1);  full_29 = view_401 = None
    full_30: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_28: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_30, slice_scatter_27, 0, 0, 9223372036854775807);  full_30 = slice_scatter_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_233: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_232, slice_scatter_28);  add_232 = slice_scatter_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    sub_150: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_8, getitem_149);  cat_8 = getitem_149 = None
    mul_439: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_150, rsqrt_26);  sub_150 = None
    mul_440: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(add_233, primals_163);  primals_163 = None
    mul_441: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_440, 128)
    sum_113: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_440, [2], True)
    mul_442: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_440, mul_439);  mul_440 = None
    sum_114: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True);  mul_442 = None
    mul_443: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_439, sum_114);  sum_114 = None
    sub_151: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_441, sum_113);  mul_441 = sum_113 = None
    sub_152: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_151, mul_443);  sub_151 = mul_443 = None
    div_24: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 128);  rsqrt_26 = None
    mul_444: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_24, sub_152);  div_24 = sub_152 = None
    mul_445: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(add_233, mul_439);  mul_439 = None
    sum_115: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 1]);  mul_445 = None
    sum_116: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
    full_31: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_29: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_31, slice_scatter_26, 1, 0, 1);  full_31 = slice_scatter_26 = None
    full_32: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_30: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_32, slice_scatter_29, 0, 0, 9223372036854775807);  full_32 = slice_scatter_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_234: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_444, slice_scatter_30);  mul_444 = slice_scatter_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_79: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_234, 1, 0, 1)
    slice_80: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_234, 1, 1, 401);  add_234 = None
    full_33: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_31: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_33, slice_80, 1, 1, 9223372036854775807);  full_33 = slice_80 = None
    full_34: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_32: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_34, slice_scatter_31, 0, 0, 9223372036854775807);  full_34 = slice_scatter_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_81: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_229, 1, 0, 1)
    slice_82: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_229, 1, 1, 401);  add_229 = None
    full_35: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_33: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_35, slice_82, 1, 1, 9223372036854775807);  full_35 = slice_82 = None
    full_36: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_34: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_36, slice_scatter_33, 0, 0, 9223372036854775807);  full_36 = slice_scatter_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    add_235: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(slice_scatter_32, slice_scatter_34);  slice_scatter_32 = slice_scatter_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    view_402: "f32[8, 128]" = torch.ops.aten.view.default(slice_81, [8, 128]);  slice_81 = None
    permute_317: "f32[128, 256]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    mm_76: "f32[8, 256]" = torch.ops.aten.mm.default(view_402, permute_317);  permute_317 = None
    permute_318: "f32[128, 8]" = torch.ops.aten.permute.default(view_402, [1, 0])
    mm_77: "f32[128, 256]" = torch.ops.aten.mm.default(permute_318, view_152);  permute_318 = view_152 = None
    permute_319: "f32[256, 128]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_117: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_402, [0], True);  view_402 = None
    view_403: "f32[128]" = torch.ops.aten.view.default(sum_117, [128]);  sum_117 = None
    permute_320: "f32[128, 256]" = torch.ops.aten.permute.default(permute_319, [1, 0]);  permute_319 = None
    view_404: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_76, [8, 1, 256]);  mm_76 = None
    mul_446: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_135, 0.7071067811865476)
    erf_33: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_446);  mul_446 = None
    add_236: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_33, 1);  erf_33 = None
    mul_447: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_236, 0.5);  add_236 = None
    mul_448: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_135, add_135)
    mul_449: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_448, -0.5);  mul_448 = None
    exp_15: "f32[8, 1, 256]" = torch.ops.aten.exp.default(mul_449);  mul_449 = None
    mul_450: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_451: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_135, mul_450);  add_135 = mul_450 = None
    add_237: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_447, mul_451);  mul_447 = mul_451 = None
    mul_452: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_404, add_237);  view_404 = add_237 = None
    sub_153: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(slice_33, getitem_147);  slice_33 = getitem_147 = None
    mul_453: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_153, rsqrt_25);  sub_153 = None
    mul_454: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_452, primals_159);  primals_159 = None
    mul_455: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_454, 256)
    sum_118: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [2], True)
    mul_456: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_454, mul_453);  mul_454 = None
    sum_119: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [2], True);  mul_456 = None
    mul_457: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_453, sum_119);  sum_119 = None
    sub_154: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(mul_455, sum_118);  mul_455 = sum_118 = None
    sub_155: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(sub_154, mul_457);  sub_154 = mul_457 = None
    div_25: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 256);  rsqrt_25 = None
    mul_458: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(div_25, sub_155);  div_25 = sub_155 = None
    mul_459: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_452, mul_453);  mul_453 = None
    sum_120: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_459, [0, 1]);  mul_459 = None
    sum_121: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 1]);  mul_452 = None
    full_37: "f32[8, 1, 256]" = torch.ops.aten.full.default([8, 1, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_35: "f32[8, 1, 256]" = torch.ops.aten.slice_scatter.default(full_37, mul_458, 0, 0, 9223372036854775807);  full_37 = mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_405: "f32[8, 256]" = torch.ops.aten.view.default(slice_scatter_35, [8, 256])
    permute_321: "f32[256, 256]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    mm_78: "f32[8, 256]" = torch.ops.aten.mm.default(view_405, permute_321);  permute_321 = None
    permute_322: "f32[256, 8]" = torch.ops.aten.permute.default(view_405, [1, 0])
    mm_79: "f32[256, 256]" = torch.ops.aten.mm.default(permute_322, view_150);  permute_322 = view_150 = None
    permute_323: "f32[256, 256]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_122: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_405, [0], True);  view_405 = None
    view_406: "f32[256]" = torch.ops.aten.view.default(sum_122, [256]);  sum_122 = None
    permute_324: "f32[256, 256]" = torch.ops.aten.permute.default(permute_323, [1, 0]);  permute_323 = None
    view_407: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_78, [8, 1, 256]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    view_408: "f32[8, 1, 4, 64]" = torch.ops.aten.view.default(view_407, [8, 1, 4, 64]);  view_407 = None
    permute_325: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_408, [0, 2, 1, 3]);  view_408 = None
    view_409: "f32[32, 1, 64]" = torch.ops.aten.view.default(permute_325, [32, 1, 64]);  permute_325 = None
    permute_326: "f32[32, 197, 1]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
    bmm_24: "f32[32, 197, 64]" = torch.ops.aten.bmm.default(permute_326, view_409);  permute_326 = None
    permute_327: "f32[32, 64, 197]" = torch.ops.aten.permute.default(view_147, [0, 2, 1]);  view_147 = None
    bmm_25: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_409, permute_327);  view_409 = permute_327 = None
    view_410: "f32[8, 4, 197, 64]" = torch.ops.aten.view.default(bmm_24, [8, 4, 197, 64]);  bmm_24 = None
    view_411: "f32[8, 4, 1, 197]" = torch.ops.aten.view.default(bmm_25, [8, 4, 1, 197]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    alias_25: "f32[8, 4, 1, 197]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_460: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(view_411, alias_25);  view_411 = None
    sum_123: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_460, [-1], True)
    mul_461: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(alias_25, sum_123);  alias_25 = sum_123 = None
    sub_156: "f32[8, 4, 1, 197]" = torch.ops.aten.sub.Tensor(mul_460, mul_461);  mul_460 = mul_461 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    mul_462: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(sub_156, 0.125);  sub_156 = None
    view_412: "f32[32, 1, 197]" = torch.ops.aten.view.default(mul_462, [32, 1, 197]);  mul_462 = None
    permute_328: "f32[32, 64, 1]" = torch.ops.aten.permute.default(view_143, [0, 2, 1]);  view_143 = None
    bmm_26: "f32[32, 64, 197]" = torch.ops.aten.bmm.default(permute_328, view_412);  permute_328 = None
    permute_329: "f32[32, 197, 64]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_27: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_412, permute_329);  view_412 = permute_329 = None
    view_413: "f32[8, 4, 64, 197]" = torch.ops.aten.view.default(bmm_26, [8, 4, 64, 197]);  bmm_26 = None
    view_414: "f32[8, 4, 1, 64]" = torch.ops.aten.view.default(bmm_27, [8, 4, 1, 64]);  bmm_27 = None
    permute_330: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_413, [0, 1, 3, 2]);  view_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_331: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
    clone_82: "f32[8, 197, 4, 64]" = torch.ops.aten.clone.default(permute_331, memory_format = torch.contiguous_format);  permute_331 = None
    view_415: "f32[8, 197, 256]" = torch.ops.aten.view.default(clone_82, [8, 197, 256]);  clone_82 = None
    view_416: "f32[1576, 256]" = torch.ops.aten.view.default(view_415, [1576, 256]);  view_415 = None
    permute_332: "f32[256, 256]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_80: "f32[1576, 256]" = torch.ops.aten.mm.default(view_416, permute_332);  permute_332 = None
    permute_333: "f32[256, 1576]" = torch.ops.aten.permute.default(view_416, [1, 0])
    mm_81: "f32[256, 256]" = torch.ops.aten.mm.default(permute_333, view_140);  permute_333 = view_140 = None
    permute_334: "f32[256, 256]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_124: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_416, [0], True);  view_416 = None
    view_417: "f32[256]" = torch.ops.aten.view.default(sum_124, [256]);  sum_124 = None
    permute_335: "f32[256, 256]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
    view_418: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_80, [8, 197, 256]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_336: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(permute_330, [0, 2, 1, 3]);  permute_330 = None
    view_419: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_336, [8, 197, 256]);  permute_336 = None
    clone_83: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_419, memory_format = torch.contiguous_format);  view_419 = None
    view_420: "f32[1576, 256]" = torch.ops.aten.view.default(clone_83, [1576, 256]);  clone_83 = None
    permute_337: "f32[256, 256]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_82: "f32[1576, 256]" = torch.ops.aten.mm.default(view_420, permute_337);  permute_337 = None
    permute_338: "f32[256, 1576]" = torch.ops.aten.permute.default(view_420, [1, 0])
    mm_83: "f32[256, 256]" = torch.ops.aten.mm.default(permute_338, view_137);  permute_338 = view_137 = None
    permute_339: "f32[256, 256]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_125: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_420, [0], True);  view_420 = None
    view_421: "f32[256]" = torch.ops.aten.view.default(sum_125, [256]);  sum_125 = None
    permute_340: "f32[256, 256]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
    view_422: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_82, [8, 197, 256]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_238: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(view_418, view_422);  view_418 = view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_341: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
    view_423: "f32[8, 1, 256]" = torch.ops.aten.view.default(permute_341, [8, 1, 256]);  permute_341 = None
    sum_126: "f32[1, 1, 256]" = torch.ops.aten.sum.dim_IntList(view_423, [0, 1], True)
    view_424: "f32[256]" = torch.ops.aten.view.default(sum_126, [256]);  sum_126 = None
    view_425: "f32[8, 256]" = torch.ops.aten.view.default(view_423, [8, 256]);  view_423 = None
    permute_342: "f32[256, 8]" = torch.ops.aten.permute.default(view_425, [1, 0])
    mm_84: "f32[256, 256]" = torch.ops.aten.mm.default(permute_342, view_134);  permute_342 = view_134 = None
    permute_343: "f32[256, 256]" = torch.ops.aten.permute.default(mm_84, [1, 0]);  mm_84 = None
    permute_344: "f32[256, 256]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    mm_85: "f32[8, 256]" = torch.ops.aten.mm.default(view_425, permute_344);  view_425 = permute_344 = None
    view_426: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_85, [8, 1, 256]);  mm_85 = None
    permute_345: "f32[256, 256]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
    full_38: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_36: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_38, view_426, 1, 0, 1);  full_38 = view_426 = None
    full_39: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_37: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_39, slice_scatter_36, 0, 0, 9223372036854775807);  full_39 = slice_scatter_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_239: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_238, slice_scatter_37);  add_238 = slice_scatter_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    sub_157: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_6, getitem_145);  cat_6 = getitem_145 = None
    mul_463: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_157, rsqrt_24);  sub_157 = None
    mul_464: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(add_239, primals_149);  primals_149 = None
    mul_465: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_464, 256)
    sum_127: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True)
    mul_466: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_464, mul_463);  mul_464 = None
    sum_128: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_466, [2], True);  mul_466 = None
    mul_467: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_463, sum_128);  sum_128 = None
    sub_158: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_465, sum_127);  mul_465 = sum_127 = None
    sub_159: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_158, mul_467);  sub_158 = mul_467 = None
    div_26: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 256);  rsqrt_24 = None
    mul_468: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_26, sub_159);  div_26 = sub_159 = None
    mul_469: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(add_239, mul_463);  mul_463 = None
    sum_129: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1]);  mul_469 = None
    sum_130: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None
    full_40: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_38: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_40, slice_scatter_35, 1, 0, 1);  full_40 = slice_scatter_35 = None
    full_41: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_39: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_41, slice_scatter_38, 0, 0, 9223372036854775807);  full_41 = slice_scatter_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_240: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_468, slice_scatter_39);  mul_468 = slice_scatter_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_83: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_240, 1, 0, 1)
    slice_84: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_240, 1, 1, 197);  add_240 = None
    full_42: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_40: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_42, slice_84, 1, 1, 9223372036854775807);  full_42 = slice_84 = None
    full_43: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_41: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_43, slice_scatter_40, 0, 0, 9223372036854775807);  full_43 = slice_scatter_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    add_241: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(slice_scatter_25, slice_scatter_41);  slice_scatter_25 = slice_scatter_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    view_427: "f32[8, 128]" = torch.ops.aten.view.default(slice_79, [8, 128]);  slice_79 = None
    permute_346: "f32[128, 256]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    mm_86: "f32[8, 256]" = torch.ops.aten.mm.default(view_427, permute_346);  permute_346 = None
    permute_347: "f32[128, 8]" = torch.ops.aten.permute.default(view_427, [1, 0])
    mm_87: "f32[128, 256]" = torch.ops.aten.mm.default(permute_347, view_132);  permute_347 = view_132 = None
    permute_348: "f32[256, 128]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_131: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_427, [0], True);  view_427 = None
    view_428: "f32[128]" = torch.ops.aten.view.default(sum_131, [128]);  sum_131 = None
    permute_349: "f32[128, 256]" = torch.ops.aten.permute.default(permute_348, [1, 0]);  permute_348 = None
    view_429: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_86, [8, 1, 256]);  mm_86 = None
    mul_470: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_128, 0.7071067811865476)
    erf_34: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_470);  mul_470 = None
    add_242: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_34, 1);  erf_34 = None
    mul_471: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_242, 0.5);  add_242 = None
    mul_472: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_128, add_128)
    mul_473: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_472, -0.5);  mul_472 = None
    exp_16: "f32[8, 1, 256]" = torch.ops.aten.exp.default(mul_473);  mul_473 = None
    mul_474: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_475: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_128, mul_474);  add_128 = mul_474 = None
    add_243: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_471, mul_475);  mul_471 = mul_475 = None
    mul_476: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_429, add_243);  view_429 = add_243 = None
    clone_84: "f32[8, 1, 256]" = torch.ops.aten.clone.default(slice_26, memory_format = torch.contiguous_format);  slice_26 = None
    sub_160: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(clone_84, getitem_143);  clone_84 = getitem_143 = None
    mul_477: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_160, rsqrt_23);  sub_160 = None
    mul_478: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_476, primals_145);  primals_145 = None
    mul_479: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_478, 256)
    sum_132: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True)
    mul_480: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_478, mul_477);  mul_478 = None
    sum_133: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_480, [2], True);  mul_480 = None
    mul_481: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_477, sum_133);  sum_133 = None
    sub_161: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(mul_479, sum_132);  mul_479 = sum_132 = None
    sub_162: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(sub_161, mul_481);  sub_161 = mul_481 = None
    div_27: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 256);  rsqrt_23 = None
    mul_482: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(div_27, sub_162);  div_27 = sub_162 = None
    mul_483: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_476, mul_477);  mul_477 = None
    sum_134: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_483, [0, 1]);  mul_483 = None
    sum_135: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_476, [0, 1]);  mul_476 = None
    full_44: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_42: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_44, mul_482, 1, 0, 1);  full_44 = mul_482 = None
    full_45: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_43: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_45, slice_scatter_42, 0, 0, 9223372036854775807);  full_45 = slice_scatter_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    add_244: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_241, slice_scatter_43);  add_241 = slice_scatter_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    view_430: "f32[8, 256]" = torch.ops.aten.view.default(slice_83, [8, 256]);  slice_83 = None
    permute_350: "f32[256, 128]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    mm_88: "f32[8, 128]" = torch.ops.aten.mm.default(view_430, permute_350);  permute_350 = None
    permute_351: "f32[256, 8]" = torch.ops.aten.permute.default(view_430, [1, 0])
    mm_89: "f32[256, 128]" = torch.ops.aten.mm.default(permute_351, view_130);  permute_351 = view_130 = None
    permute_352: "f32[128, 256]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_136: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_430, [0], True);  view_430 = None
    view_431: "f32[256]" = torch.ops.aten.view.default(sum_136, [256]);  sum_136 = None
    permute_353: "f32[256, 128]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    view_432: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_88, [8, 1, 128]);  mm_88 = None
    mul_484: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_125, 0.7071067811865476)
    erf_35: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_484);  mul_484 = None
    add_245: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_35, 1);  erf_35 = None
    mul_485: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_245, 0.5);  add_245 = None
    mul_486: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_125, add_125)
    mul_487: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_486, -0.5);  mul_486 = None
    exp_17: "f32[8, 1, 128]" = torch.ops.aten.exp.default(mul_487);  mul_487 = None
    mul_488: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_489: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_125, mul_488);  add_125 = mul_488 = None
    add_246: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_485, mul_489);  mul_485 = mul_489 = None
    mul_490: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_432, add_246);  view_432 = add_246 = None
    clone_85: "f32[8, 1, 128]" = torch.ops.aten.clone.default(slice_24, memory_format = torch.contiguous_format);  slice_24 = None
    sub_163: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(clone_85, getitem_141);  clone_85 = getitem_141 = None
    mul_491: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_163, rsqrt_22);  sub_163 = None
    mul_492: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_490, primals_141);  primals_141 = None
    mul_493: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_492, 128)
    sum_137: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_492, [2], True)
    mul_494: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_492, mul_491);  mul_492 = None
    sum_138: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_494, [2], True);  mul_494 = None
    mul_495: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_491, sum_138);  sum_138 = None
    sub_164: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(mul_493, sum_137);  mul_493 = sum_137 = None
    sub_165: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(sub_164, mul_495);  sub_164 = mul_495 = None
    div_28: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 128);  rsqrt_22 = None
    mul_496: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(div_28, sub_165);  div_28 = sub_165 = None
    mul_497: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_490, mul_491);  mul_491 = None
    sum_139: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_497, [0, 1]);  mul_497 = None
    sum_140: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 1]);  mul_490 = None
    full_46: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_44: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_46, mul_496, 1, 0, 1);  full_46 = mul_496 = None
    full_47: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_45: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_47, slice_scatter_44, 0, 0, 9223372036854775807);  full_47 = slice_scatter_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    add_247: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_235, slice_scatter_45);  add_235 = slice_scatter_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_433: "f32[1576, 256]" = torch.ops.aten.view.default(add_244, [1576, 256])
    permute_354: "f32[256, 768]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_90: "f32[1576, 768]" = torch.ops.aten.mm.default(view_433, permute_354);  permute_354 = None
    permute_355: "f32[256, 1576]" = torch.ops.aten.permute.default(view_433, [1, 0])
    mm_91: "f32[256, 768]" = torch.ops.aten.mm.default(permute_355, view_128);  permute_355 = view_128 = None
    permute_356: "f32[768, 256]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_141: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_433, [0], True);  view_433 = None
    view_434: "f32[256]" = torch.ops.aten.view.default(sum_141, [256]);  sum_141 = None
    permute_357: "f32[256, 768]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
    view_435: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_90, [8, 197, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_498: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, 0.7071067811865476)
    erf_36: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_498);  mul_498 = None
    add_248: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_36, 1);  erf_36 = None
    mul_499: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_248, 0.5);  add_248 = None
    mul_500: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, view_127)
    mul_501: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_500, -0.5);  mul_500 = None
    exp_18: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_501);  mul_501 = None
    mul_502: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_503: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, mul_502);  view_127 = mul_502 = None
    add_249: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_499, mul_503);  mul_499 = mul_503 = None
    mul_504: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_435, add_249);  view_435 = add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_436: "f32[1576, 768]" = torch.ops.aten.view.default(mul_504, [1576, 768]);  mul_504 = None
    permute_358: "f32[768, 256]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_92: "f32[1576, 256]" = torch.ops.aten.mm.default(view_436, permute_358);  permute_358 = None
    permute_359: "f32[768, 1576]" = torch.ops.aten.permute.default(view_436, [1, 0])
    mm_93: "f32[768, 256]" = torch.ops.aten.mm.default(permute_359, view_126);  permute_359 = view_126 = None
    permute_360: "f32[256, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_142: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_436, [0], True);  view_436 = None
    view_437: "f32[768]" = torch.ops.aten.view.default(sum_142, [768]);  sum_142 = None
    permute_361: "f32[768, 256]" = torch.ops.aten.permute.default(permute_360, [1, 0]);  permute_360 = None
    view_438: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_92, [8, 197, 256]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_166: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_119, getitem_139);  add_119 = getitem_139 = None
    mul_505: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_166, rsqrt_21);  sub_166 = None
    mul_506: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_438, primals_135);  primals_135 = None
    mul_507: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_506, 256)
    sum_143: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True)
    mul_508: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_506, mul_505);  mul_506 = None
    sum_144: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_508, [2], True);  mul_508 = None
    mul_509: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_505, sum_144);  sum_144 = None
    sub_167: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_507, sum_143);  mul_507 = sum_143 = None
    sub_168: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_167, mul_509);  sub_167 = mul_509 = None
    div_29: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 256);  rsqrt_21 = None
    mul_510: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_29, sub_168);  div_29 = sub_168 = None
    mul_511: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_438, mul_505);  mul_505 = None
    sum_145: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_511, [0, 1]);  mul_511 = None
    sum_146: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_438, [0, 1]);  view_438 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_250: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_244, mul_510);  add_244 = mul_510 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_439: "f32[1576, 256]" = torch.ops.aten.view.default(add_250, [1576, 256])
    permute_362: "f32[256, 256]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    mm_94: "f32[1576, 256]" = torch.ops.aten.mm.default(view_439, permute_362);  permute_362 = None
    permute_363: "f32[256, 1576]" = torch.ops.aten.permute.default(view_439, [1, 0])
    mm_95: "f32[256, 256]" = torch.ops.aten.mm.default(permute_363, view_124);  permute_363 = view_124 = None
    permute_364: "f32[256, 256]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_147: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
    view_440: "f32[256]" = torch.ops.aten.view.default(sum_147, [256]);  sum_147 = None
    permute_365: "f32[256, 256]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    view_441: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_94, [8, 197, 256]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_442: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_441, [8, 197, 4, 64]);  view_441 = None
    permute_366: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_26: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    _scaled_dot_product_flash_attention_backward_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_366, getitem_126, getitem_127, getitem_128, alias_26, getitem_130, getitem_131, getitem_132, 0, 0, 0.0, False, getitem_135, getitem_136);  permute_366 = getitem_126 = getitem_127 = getitem_128 = alias_26 = getitem_130 = getitem_131 = getitem_132 = getitem_135 = getitem_136 = None
    getitem_244: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_4[0]
    getitem_245: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_4[1]
    getitem_246: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_4[2];  _scaled_dot_product_flash_attention_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_19: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_244, getitem_245, getitem_246]);  getitem_244 = getitem_245 = getitem_246 = None
    view_443: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_19, [3, 8, 4, 197, 64]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_367: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_443, [1, 3, 0, 2, 4]);  view_443 = None
    clone_86: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_367, memory_format = torch.contiguous_format);  permute_367 = None
    view_444: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_86, [8, 197, 768]);  clone_86 = None
    view_445: "f32[1576, 768]" = torch.ops.aten.view.default(view_444, [1576, 768]);  view_444 = None
    permute_368: "f32[768, 256]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_96: "f32[1576, 256]" = torch.ops.aten.mm.default(view_445, permute_368);  permute_368 = None
    permute_369: "f32[768, 1576]" = torch.ops.aten.permute.default(view_445, [1, 0])
    mm_97: "f32[768, 256]" = torch.ops.aten.mm.default(permute_369, view_120);  permute_369 = view_120 = None
    permute_370: "f32[256, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_148: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
    view_446: "f32[768]" = torch.ops.aten.view.default(sum_148, [768]);  sum_148 = None
    permute_371: "f32[768, 256]" = torch.ops.aten.permute.default(permute_370, [1, 0]);  permute_370 = None
    view_447: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_96, [8, 197, 256]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_169: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_116, getitem_125);  add_116 = getitem_125 = None
    mul_512: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_169, rsqrt_20);  sub_169 = None
    mul_513: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_447, primals_129);  primals_129 = None
    mul_514: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_513, 256)
    sum_149: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_513, [2], True)
    mul_515: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_513, mul_512);  mul_513 = None
    sum_150: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_515, [2], True);  mul_515 = None
    mul_516: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_512, sum_150);  sum_150 = None
    sub_170: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_514, sum_149);  mul_514 = sum_149 = None
    sub_171: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_170, mul_516);  sub_170 = mul_516 = None
    div_30: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 256);  rsqrt_20 = None
    mul_517: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_30, sub_171);  div_30 = sub_171 = None
    mul_518: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_447, mul_512);  mul_512 = None
    sum_151: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_518, [0, 1]);  mul_518 = None
    sum_152: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_447, [0, 1]);  view_447 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_251: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_250, mul_517);  add_250 = mul_517 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_448: "f32[1576, 256]" = torch.ops.aten.view.default(add_251, [1576, 256])
    permute_372: "f32[256, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_98: "f32[1576, 768]" = torch.ops.aten.mm.default(view_448, permute_372);  permute_372 = None
    permute_373: "f32[256, 1576]" = torch.ops.aten.permute.default(view_448, [1, 0])
    mm_99: "f32[256, 768]" = torch.ops.aten.mm.default(permute_373, view_118);  permute_373 = view_118 = None
    permute_374: "f32[768, 256]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_153: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_448, [0], True);  view_448 = None
    view_449: "f32[256]" = torch.ops.aten.view.default(sum_153, [256]);  sum_153 = None
    permute_375: "f32[256, 768]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
    view_450: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_98, [8, 197, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_519: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_117, 0.7071067811865476)
    erf_37: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_519);  mul_519 = None
    add_252: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_37, 1);  erf_37 = None
    mul_520: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_252, 0.5);  add_252 = None
    mul_521: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_117, view_117)
    mul_522: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_521, -0.5);  mul_521 = None
    exp_19: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_522);  mul_522 = None
    mul_523: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_524: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_117, mul_523);  view_117 = mul_523 = None
    add_253: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_520, mul_524);  mul_520 = mul_524 = None
    mul_525: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_450, add_253);  view_450 = add_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_451: "f32[1576, 768]" = torch.ops.aten.view.default(mul_525, [1576, 768]);  mul_525 = None
    permute_376: "f32[768, 256]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_100: "f32[1576, 256]" = torch.ops.aten.mm.default(view_451, permute_376);  permute_376 = None
    permute_377: "f32[768, 1576]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_101: "f32[768, 256]" = torch.ops.aten.mm.default(permute_377, view_116);  permute_377 = view_116 = None
    permute_378: "f32[256, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_154: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[768]" = torch.ops.aten.view.default(sum_154, [768]);  sum_154 = None
    permute_379: "f32[768, 256]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
    view_453: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_100, [8, 197, 256]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_172: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_112, getitem_123);  add_112 = getitem_123 = None
    mul_526: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_172, rsqrt_19);  sub_172 = None
    mul_527: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_453, primals_123);  primals_123 = None
    mul_528: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_527, 256)
    sum_155: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_527, [2], True)
    mul_529: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_527, mul_526);  mul_527 = None
    sum_156: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_529, [2], True);  mul_529 = None
    mul_530: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_526, sum_156);  sum_156 = None
    sub_173: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_528, sum_155);  mul_528 = sum_155 = None
    sub_174: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_173, mul_530);  sub_173 = mul_530 = None
    div_31: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 256);  rsqrt_19 = None
    mul_531: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_31, sub_174);  div_31 = sub_174 = None
    mul_532: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_453, mul_526);  mul_526 = None
    sum_157: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_532, [0, 1]);  mul_532 = None
    sum_158: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_453, [0, 1]);  view_453 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_254: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_251, mul_531);  add_251 = mul_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_454: "f32[1576, 256]" = torch.ops.aten.view.default(add_254, [1576, 256])
    permute_380: "f32[256, 256]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_102: "f32[1576, 256]" = torch.ops.aten.mm.default(view_454, permute_380);  permute_380 = None
    permute_381: "f32[256, 1576]" = torch.ops.aten.permute.default(view_454, [1, 0])
    mm_103: "f32[256, 256]" = torch.ops.aten.mm.default(permute_381, view_114);  permute_381 = view_114 = None
    permute_382: "f32[256, 256]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_159: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_454, [0], True);  view_454 = None
    view_455: "f32[256]" = torch.ops.aten.view.default(sum_159, [256]);  sum_159 = None
    permute_383: "f32[256, 256]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
    view_456: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_102, [8, 197, 256]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_457: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_456, [8, 197, 4, 64]);  view_456 = None
    permute_384: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_457, [0, 2, 1, 3]);  view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_27: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    _scaled_dot_product_flash_attention_backward_5 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_384, getitem_110, getitem_111, getitem_112, alias_27, getitem_114, getitem_115, getitem_116, 0, 0, 0.0, False, getitem_119, getitem_120);  permute_384 = getitem_110 = getitem_111 = getitem_112 = alias_27 = getitem_114 = getitem_115 = getitem_116 = getitem_119 = getitem_120 = None
    getitem_247: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_5[0]
    getitem_248: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_5[1]
    getitem_249: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_5[2];  _scaled_dot_product_flash_attention_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_20: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_247, getitem_248, getitem_249]);  getitem_247 = getitem_248 = getitem_249 = None
    view_458: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_20, [3, 8, 4, 197, 64]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_385: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_458, [1, 3, 0, 2, 4]);  view_458 = None
    clone_87: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
    view_459: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_87, [8, 197, 768]);  clone_87 = None
    view_460: "f32[1576, 768]" = torch.ops.aten.view.default(view_459, [1576, 768]);  view_459 = None
    permute_386: "f32[768, 256]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    mm_104: "f32[1576, 256]" = torch.ops.aten.mm.default(view_460, permute_386);  permute_386 = None
    permute_387: "f32[768, 1576]" = torch.ops.aten.permute.default(view_460, [1, 0])
    mm_105: "f32[768, 256]" = torch.ops.aten.mm.default(permute_387, view_110);  permute_387 = view_110 = None
    permute_388: "f32[256, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_160: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_460, [0], True);  view_460 = None
    view_461: "f32[768]" = torch.ops.aten.view.default(sum_160, [768]);  sum_160 = None
    permute_389: "f32[768, 256]" = torch.ops.aten.permute.default(permute_388, [1, 0]);  permute_388 = None
    view_462: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_104, [8, 197, 256]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_175: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_109, getitem_109);  add_109 = getitem_109 = None
    mul_533: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_175, rsqrt_18);  sub_175 = None
    mul_534: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_462, primals_117);  primals_117 = None
    mul_535: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_534, 256)
    sum_161: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_534, [2], True)
    mul_536: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_534, mul_533);  mul_534 = None
    sum_162: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_536, [2], True);  mul_536 = None
    mul_537: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_533, sum_162);  sum_162 = None
    sub_176: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_535, sum_161);  mul_535 = sum_161 = None
    sub_177: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_176, mul_537);  sub_176 = mul_537 = None
    div_32: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 256);  rsqrt_18 = None
    mul_538: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_32, sub_177);  div_32 = sub_177 = None
    mul_539: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_462, mul_533);  mul_533 = None
    sum_163: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_539, [0, 1]);  mul_539 = None
    sum_164: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_462, [0, 1]);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_255: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_254, mul_538);  add_254 = mul_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_463: "f32[1576, 256]" = torch.ops.aten.view.default(add_255, [1576, 256])
    permute_390: "f32[256, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_106: "f32[1576, 768]" = torch.ops.aten.mm.default(view_463, permute_390);  permute_390 = None
    permute_391: "f32[256, 1576]" = torch.ops.aten.permute.default(view_463, [1, 0])
    mm_107: "f32[256, 768]" = torch.ops.aten.mm.default(permute_391, view_108);  permute_391 = view_108 = None
    permute_392: "f32[768, 256]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_165: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_463, [0], True);  view_463 = None
    view_464: "f32[256]" = torch.ops.aten.view.default(sum_165, [256]);  sum_165 = None
    permute_393: "f32[256, 768]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
    view_465: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_106, [8, 197, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_540: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_38: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_540);  mul_540 = None
    add_256: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_38, 1);  erf_38 = None
    mul_541: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_256, 0.5);  add_256 = None
    mul_542: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_543: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_542, -0.5);  mul_542 = None
    exp_20: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_543);  mul_543 = None
    mul_544: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_545: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_107, mul_544);  view_107 = mul_544 = None
    add_257: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_541, mul_545);  mul_541 = mul_545 = None
    mul_546: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_465, add_257);  view_465 = add_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_466: "f32[1576, 768]" = torch.ops.aten.view.default(mul_546, [1576, 768]);  mul_546 = None
    permute_394: "f32[768, 256]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_108: "f32[1576, 256]" = torch.ops.aten.mm.default(view_466, permute_394);  permute_394 = None
    permute_395: "f32[768, 1576]" = torch.ops.aten.permute.default(view_466, [1, 0])
    mm_109: "f32[768, 256]" = torch.ops.aten.mm.default(permute_395, view_106);  permute_395 = view_106 = None
    permute_396: "f32[256, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_166: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_466, [0], True);  view_466 = None
    view_467: "f32[768]" = torch.ops.aten.view.default(sum_166, [768]);  sum_166 = None
    permute_397: "f32[768, 256]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
    view_468: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_108, [8, 197, 256]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_178: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_105, getitem_107);  add_105 = getitem_107 = None
    mul_547: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_178, rsqrt_17);  sub_178 = None
    mul_548: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_468, primals_111);  primals_111 = None
    mul_549: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_548, 256)
    sum_167: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_548, [2], True)
    mul_550: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_548, mul_547);  mul_548 = None
    sum_168: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_550, [2], True);  mul_550 = None
    mul_551: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_547, sum_168);  sum_168 = None
    sub_179: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_549, sum_167);  mul_549 = sum_167 = None
    sub_180: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_179, mul_551);  sub_179 = mul_551 = None
    div_33: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 256);  rsqrt_17 = None
    mul_552: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_33, sub_180);  div_33 = sub_180 = None
    mul_553: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_468, mul_547);  mul_547 = None
    sum_169: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_553, [0, 1]);  mul_553 = None
    sum_170: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_468, [0, 1]);  view_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_258: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_255, mul_552);  add_255 = mul_552 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_469: "f32[1576, 256]" = torch.ops.aten.view.default(add_258, [1576, 256])
    permute_398: "f32[256, 256]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_110: "f32[1576, 256]" = torch.ops.aten.mm.default(view_469, permute_398);  permute_398 = None
    permute_399: "f32[256, 1576]" = torch.ops.aten.permute.default(view_469, [1, 0])
    mm_111: "f32[256, 256]" = torch.ops.aten.mm.default(permute_399, view_104);  permute_399 = view_104 = None
    permute_400: "f32[256, 256]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_171: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_469, [0], True);  view_469 = None
    view_470: "f32[256]" = torch.ops.aten.view.default(sum_171, [256]);  sum_171 = None
    permute_401: "f32[256, 256]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
    view_471: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_110, [8, 197, 256]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_472: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_471, [8, 197, 4, 64]);  view_471 = None
    permute_402: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_28: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    _scaled_dot_product_flash_attention_backward_6 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_402, getitem_94, getitem_95, getitem_96, alias_28, getitem_98, getitem_99, getitem_100, 0, 0, 0.0, False, getitem_103, getitem_104);  permute_402 = getitem_94 = getitem_95 = getitem_96 = alias_28 = getitem_98 = getitem_99 = getitem_100 = getitem_103 = getitem_104 = None
    getitem_250: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_6[0]
    getitem_251: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_6[1]
    getitem_252: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_6[2];  _scaled_dot_product_flash_attention_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_21: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_250, getitem_251, getitem_252]);  getitem_250 = getitem_251 = getitem_252 = None
    view_473: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_21, [3, 8, 4, 197, 64]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_403: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_473, [1, 3, 0, 2, 4]);  view_473 = None
    clone_88: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_403, memory_format = torch.contiguous_format);  permute_403 = None
    view_474: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_88, [8, 197, 768]);  clone_88 = None
    view_475: "f32[1576, 768]" = torch.ops.aten.view.default(view_474, [1576, 768]);  view_474 = None
    permute_404: "f32[768, 256]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_112: "f32[1576, 256]" = torch.ops.aten.mm.default(view_475, permute_404);  permute_404 = None
    permute_405: "f32[768, 1576]" = torch.ops.aten.permute.default(view_475, [1, 0])
    mm_113: "f32[768, 256]" = torch.ops.aten.mm.default(permute_405, view_100);  permute_405 = view_100 = None
    permute_406: "f32[256, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
    view_476: "f32[768]" = torch.ops.aten.view.default(sum_172, [768]);  sum_172 = None
    permute_407: "f32[768, 256]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_477: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_112, [8, 197, 256]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_181: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_5, getitem_93);  cat_5 = getitem_93 = None
    mul_554: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_181, rsqrt_16);  sub_181 = None
    mul_555: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_477, primals_105);  primals_105 = None
    mul_556: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_555, 256)
    sum_173: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_555, [2], True)
    mul_557: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_555, mul_554);  mul_555 = None
    sum_174: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_557, [2], True);  mul_557 = None
    mul_558: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_554, sum_174);  sum_174 = None
    sub_182: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_556, sum_173);  mul_556 = sum_173 = None
    sub_183: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_182, mul_558);  sub_182 = mul_558 = None
    div_34: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 256);  rsqrt_16 = None
    mul_559: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_34, sub_183);  div_34 = sub_183 = None
    mul_560: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_477, mul_554);  mul_554 = None
    sum_175: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_560, [0, 1]);  mul_560 = None
    sum_176: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_477, [0, 1]);  view_477 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_259: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_258, mul_559);  add_258 = mul_559 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_478: "f32[3208, 128]" = torch.ops.aten.view.default(add_247, [3208, 128])
    permute_408: "f32[128, 384]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_114: "f32[3208, 384]" = torch.ops.aten.mm.default(view_478, permute_408);  permute_408 = None
    permute_409: "f32[128, 3208]" = torch.ops.aten.permute.default(view_478, [1, 0])
    mm_115: "f32[128, 384]" = torch.ops.aten.mm.default(permute_409, view_98);  permute_409 = view_98 = None
    permute_410: "f32[384, 128]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_177: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
    view_479: "f32[128]" = torch.ops.aten.view.default(sum_177, [128]);  sum_177 = None
    permute_411: "f32[128, 384]" = torch.ops.aten.permute.default(permute_410, [1, 0]);  permute_410 = None
    view_480: "f32[8, 401, 384]" = torch.ops.aten.view.default(mm_114, [8, 401, 384]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_561: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_97, 0.7071067811865476)
    erf_39: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_561);  mul_561 = None
    add_260: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf_39, 1);  erf_39 = None
    mul_562: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(add_260, 0.5);  add_260 = None
    mul_563: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_97, view_97)
    mul_564: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_563, -0.5);  mul_563 = None
    exp_21: "f32[8, 401, 384]" = torch.ops.aten.exp.default(mul_564);  mul_564 = None
    mul_565: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_566: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_97, mul_565);  view_97 = mul_565 = None
    add_261: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(mul_562, mul_566);  mul_562 = mul_566 = None
    mul_567: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_480, add_261);  view_480 = add_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_481: "f32[3208, 384]" = torch.ops.aten.view.default(mul_567, [3208, 384]);  mul_567 = None
    permute_412: "f32[384, 128]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_116: "f32[3208, 128]" = torch.ops.aten.mm.default(view_481, permute_412);  permute_412 = None
    permute_413: "f32[384, 3208]" = torch.ops.aten.permute.default(view_481, [1, 0])
    mm_117: "f32[384, 128]" = torch.ops.aten.mm.default(permute_413, view_96);  permute_413 = view_96 = None
    permute_414: "f32[128, 384]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_178: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
    view_482: "f32[384]" = torch.ops.aten.view.default(sum_178, [384]);  sum_178 = None
    permute_415: "f32[384, 128]" = torch.ops.aten.permute.default(permute_414, [1, 0]);  permute_414 = None
    view_483: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_116, [8, 401, 128]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_184: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add_98, getitem_91);  add_98 = getitem_91 = None
    mul_568: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_184, rsqrt_15);  sub_184 = None
    mul_569: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_483, primals_99);  primals_99 = None
    mul_570: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_569, 128)
    sum_179: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_569, [2], True)
    mul_571: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_569, mul_568);  mul_569 = None
    sum_180: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_571, [2], True);  mul_571 = None
    mul_572: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_568, sum_180);  sum_180 = None
    sub_185: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_570, sum_179);  mul_570 = sum_179 = None
    sub_186: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_185, mul_572);  sub_185 = mul_572 = None
    div_35: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 128);  rsqrt_15 = None
    mul_573: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_35, sub_186);  div_35 = sub_186 = None
    mul_574: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_483, mul_568);  mul_568 = None
    sum_181: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_574, [0, 1]);  mul_574 = None
    sum_182: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_483, [0, 1]);  view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_262: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_247, mul_573);  add_247 = mul_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_484: "f32[3208, 128]" = torch.ops.aten.view.default(add_262, [3208, 128])
    permute_416: "f32[128, 128]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    mm_118: "f32[3208, 128]" = torch.ops.aten.mm.default(view_484, permute_416);  permute_416 = None
    permute_417: "f32[128, 3208]" = torch.ops.aten.permute.default(view_484, [1, 0])
    mm_119: "f32[128, 128]" = torch.ops.aten.mm.default(permute_417, view_94);  permute_417 = view_94 = None
    permute_418: "f32[128, 128]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_183: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
    view_485: "f32[128]" = torch.ops.aten.view.default(sum_183, [128]);  sum_183 = None
    permute_419: "f32[128, 128]" = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
    view_486: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_118, [8, 401, 128]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_487: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_486, [8, 401, 4, 32]);  view_486 = None
    permute_420: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_487, [0, 2, 1, 3]);  view_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_29: "f32[8, 4, 401, 32]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    _scaled_dot_product_flash_attention_backward_7 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_420, getitem_78, getitem_79, getitem_80, alias_29, getitem_82, getitem_83, getitem_84, 0, 0, 0.0, False, getitem_87, getitem_88);  permute_420 = getitem_78 = getitem_79 = getitem_80 = alias_29 = getitem_82 = getitem_83 = getitem_84 = getitem_87 = getitem_88 = None
    getitem_253: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_7[0]
    getitem_254: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_7[1]
    getitem_255: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_7[2];  _scaled_dot_product_flash_attention_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_22: "f32[24, 4, 401, 32]" = torch.ops.aten.cat.default([getitem_253, getitem_254, getitem_255]);  getitem_253 = getitem_254 = getitem_255 = None
    view_488: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.view.default(cat_22, [3, 8, 4, 401, 32]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_421: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.permute.default(view_488, [1, 3, 0, 2, 4]);  view_488 = None
    clone_89: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.clone.default(permute_421, memory_format = torch.contiguous_format);  permute_421 = None
    view_489: "f32[8, 401, 384]" = torch.ops.aten.view.default(clone_89, [8, 401, 384]);  clone_89 = None
    view_490: "f32[3208, 384]" = torch.ops.aten.view.default(view_489, [3208, 384]);  view_489 = None
    permute_422: "f32[384, 128]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_120: "f32[3208, 128]" = torch.ops.aten.mm.default(view_490, permute_422);  permute_422 = None
    permute_423: "f32[384, 3208]" = torch.ops.aten.permute.default(view_490, [1, 0])
    mm_121: "f32[384, 128]" = torch.ops.aten.mm.default(permute_423, view_90);  permute_423 = view_90 = None
    permute_424: "f32[128, 384]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_184: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
    view_491: "f32[384]" = torch.ops.aten.view.default(sum_184, [384]);  sum_184 = None
    permute_425: "f32[384, 128]" = torch.ops.aten.permute.default(permute_424, [1, 0]);  permute_424 = None
    view_492: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_120, [8, 401, 128]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_187: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_3, getitem_77);  cat_3 = getitem_77 = None
    mul_575: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_187, rsqrt_14);  sub_187 = None
    mul_576: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_492, primals_93);  primals_93 = None
    mul_577: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_576, 128)
    sum_185: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_576, [2], True)
    mul_578: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_576, mul_575);  mul_576 = None
    sum_186: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_578, [2], True);  mul_578 = None
    mul_579: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_575, sum_186);  sum_186 = None
    sub_188: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_577, sum_185);  mul_577 = sum_185 = None
    sub_189: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_188, mul_579);  sub_188 = mul_579 = None
    div_36: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 128);  rsqrt_14 = None
    mul_580: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_36, sub_189);  div_36 = sub_189 = None
    mul_581: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_492, mul_575);  mul_575 = None
    sum_187: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_581, [0, 1]);  mul_581 = None
    sum_188: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_492, [0, 1]);  view_492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_263: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_262, mul_580);  add_262 = mul_580 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_85: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_259, 1, 0, 1)
    slice_86: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_259, 1, 1, 197);  add_259 = None
    full_48: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_46: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_48, slice_86, 1, 1, 9223372036854775807);  full_48 = slice_86 = None
    full_49: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_47: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_49, slice_scatter_46, 0, 0, 9223372036854775807);  full_49 = slice_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    view_493: "f32[8, 256]" = torch.ops.aten.view.default(slice_85, [8, 256]);  slice_85 = None
    permute_426: "f32[256, 128]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_122: "f32[8, 128]" = torch.ops.aten.mm.default(view_493, permute_426);  permute_426 = None
    permute_427: "f32[256, 8]" = torch.ops.aten.permute.default(view_493, [1, 0])
    mm_123: "f32[256, 128]" = torch.ops.aten.mm.default(permute_427, view_88);  permute_427 = view_88 = None
    permute_428: "f32[128, 256]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_189: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
    view_494: "f32[256]" = torch.ops.aten.view.default(sum_189, [256]);  sum_189 = None
    permute_429: "f32[256, 128]" = torch.ops.aten.permute.default(permute_428, [1, 0]);  permute_428 = None
    view_495: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_122, [8, 1, 128]);  mm_122 = None
    mul_582: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_94, 0.7071067811865476)
    erf_40: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_582);  mul_582 = None
    add_264: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_40, 1);  erf_40 = None
    mul_583: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_264, 0.5);  add_264 = None
    mul_584: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_94, add_94)
    mul_585: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_584, -0.5);  mul_584 = None
    exp_22: "f32[8, 1, 128]" = torch.ops.aten.exp.default(mul_585);  mul_585 = None
    mul_586: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_587: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_94, mul_586);  add_94 = mul_586 = None
    add_265: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_583, mul_587);  mul_583 = mul_587 = None
    mul_588: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_495, add_265);  view_495 = add_265 = None
    sub_190: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(slice_20, getitem_75);  slice_20 = getitem_75 = None
    mul_589: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_190, rsqrt_13);  sub_190 = None
    mul_590: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_588, primals_89);  primals_89 = None
    mul_591: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_590, 128)
    sum_190: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_590, [2], True)
    mul_592: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_590, mul_589);  mul_590 = None
    sum_191: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_592, [2], True);  mul_592 = None
    mul_593: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_589, sum_191);  sum_191 = None
    sub_191: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(mul_591, sum_190);  mul_591 = sum_190 = None
    sub_192: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(sub_191, mul_593);  sub_191 = mul_593 = None
    div_37: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 128);  rsqrt_13 = None
    mul_594: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(div_37, sub_192);  div_37 = sub_192 = None
    mul_595: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_588, mul_589);  mul_589 = None
    sum_192: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 1]);  mul_595 = None
    sum_193: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_588, [0, 1]);  mul_588 = None
    full_50: "f32[8, 1, 128]" = torch.ops.aten.full.default([8, 1, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_48: "f32[8, 1, 128]" = torch.ops.aten.slice_scatter.default(full_50, mul_594, 0, 0, 9223372036854775807);  full_50 = mul_594 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_496: "f32[8, 128]" = torch.ops.aten.view.default(slice_scatter_48, [8, 128])
    permute_430: "f32[128, 128]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_124: "f32[8, 128]" = torch.ops.aten.mm.default(view_496, permute_430);  permute_430 = None
    permute_431: "f32[128, 8]" = torch.ops.aten.permute.default(view_496, [1, 0])
    mm_125: "f32[128, 128]" = torch.ops.aten.mm.default(permute_431, view_86);  permute_431 = view_86 = None
    permute_432: "f32[128, 128]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_194: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
    view_497: "f32[128]" = torch.ops.aten.view.default(sum_194, [128]);  sum_194 = None
    permute_433: "f32[128, 128]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
    view_498: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_124, [8, 1, 128]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    view_499: "f32[8, 1, 4, 32]" = torch.ops.aten.view.default(view_498, [8, 1, 4, 32]);  view_498 = None
    permute_434: "f32[8, 4, 1, 32]" = torch.ops.aten.permute.default(view_499, [0, 2, 1, 3]);  view_499 = None
    view_500: "f32[32, 1, 32]" = torch.ops.aten.view.default(permute_434, [32, 1, 32]);  permute_434 = None
    permute_435: "f32[32, 401, 1]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    bmm_28: "f32[32, 401, 32]" = torch.ops.aten.bmm.default(permute_435, view_500);  permute_435 = None
    permute_436: "f32[32, 32, 401]" = torch.ops.aten.permute.default(view_83, [0, 2, 1]);  view_83 = None
    bmm_29: "f32[32, 1, 401]" = torch.ops.aten.bmm.default(view_500, permute_436);  view_500 = permute_436 = None
    view_501: "f32[8, 4, 401, 32]" = torch.ops.aten.view.default(bmm_28, [8, 4, 401, 32]);  bmm_28 = None
    view_502: "f32[8, 4, 1, 401]" = torch.ops.aten.view.default(bmm_29, [8, 4, 1, 401]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    alias_30: "f32[8, 4, 1, 401]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_596: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(view_502, alias_30);  view_502 = None
    sum_195: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_596, [-1], True)
    mul_597: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(alias_30, sum_195);  alias_30 = sum_195 = None
    sub_193: "f32[8, 4, 1, 401]" = torch.ops.aten.sub.Tensor(mul_596, mul_597);  mul_596 = mul_597 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    mul_598: "f32[8, 4, 1, 401]" = torch.ops.aten.mul.Tensor(sub_193, 0.1767766952966369);  sub_193 = None
    view_503: "f32[32, 1, 401]" = torch.ops.aten.view.default(mul_598, [32, 1, 401]);  mul_598 = None
    permute_437: "f32[32, 32, 1]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    bmm_30: "f32[32, 32, 401]" = torch.ops.aten.bmm.default(permute_437, view_503);  permute_437 = None
    permute_438: "f32[32, 401, 32]" = torch.ops.aten.permute.default(view_80, [0, 2, 1]);  view_80 = None
    bmm_31: "f32[32, 1, 32]" = torch.ops.aten.bmm.default(view_503, permute_438);  view_503 = permute_438 = None
    view_504: "f32[8, 4, 32, 401]" = torch.ops.aten.view.default(bmm_30, [8, 4, 32, 401]);  bmm_30 = None
    view_505: "f32[8, 4, 1, 32]" = torch.ops.aten.view.default(bmm_31, [8, 4, 1, 32]);  bmm_31 = None
    permute_439: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_504, [0, 1, 3, 2]);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_440: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(view_501, [0, 2, 1, 3]);  view_501 = None
    clone_90: "f32[8, 401, 4, 32]" = torch.ops.aten.clone.default(permute_440, memory_format = torch.contiguous_format);  permute_440 = None
    view_506: "f32[8, 401, 128]" = torch.ops.aten.view.default(clone_90, [8, 401, 128]);  clone_90 = None
    view_507: "f32[3208, 128]" = torch.ops.aten.view.default(view_506, [3208, 128]);  view_506 = None
    permute_441: "f32[128, 128]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_126: "f32[3208, 128]" = torch.ops.aten.mm.default(view_507, permute_441);  permute_441 = None
    permute_442: "f32[128, 3208]" = torch.ops.aten.permute.default(view_507, [1, 0])
    mm_127: "f32[128, 128]" = torch.ops.aten.mm.default(permute_442, view_76);  permute_442 = view_76 = None
    permute_443: "f32[128, 128]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_196: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
    view_508: "f32[128]" = torch.ops.aten.view.default(sum_196, [128]);  sum_196 = None
    permute_444: "f32[128, 128]" = torch.ops.aten.permute.default(permute_443, [1, 0]);  permute_443 = None
    view_509: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_126, [8, 401, 128]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_445: "f32[8, 401, 4, 32]" = torch.ops.aten.permute.default(permute_439, [0, 2, 1, 3]);  permute_439 = None
    view_510: "f32[8, 401, 128]" = torch.ops.aten.view.default(permute_445, [8, 401, 128]);  permute_445 = None
    clone_91: "f32[8, 401, 128]" = torch.ops.aten.clone.default(view_510, memory_format = torch.contiguous_format);  view_510 = None
    view_511: "f32[3208, 128]" = torch.ops.aten.view.default(clone_91, [3208, 128]);  clone_91 = None
    permute_446: "f32[128, 128]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    mm_128: "f32[3208, 128]" = torch.ops.aten.mm.default(view_511, permute_446);  permute_446 = None
    permute_447: "f32[128, 3208]" = torch.ops.aten.permute.default(view_511, [1, 0])
    mm_129: "f32[128, 128]" = torch.ops.aten.mm.default(permute_447, view_73);  permute_447 = view_73 = None
    permute_448: "f32[128, 128]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_197: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_511, [0], True);  view_511 = None
    view_512: "f32[128]" = torch.ops.aten.view.default(sum_197, [128]);  sum_197 = None
    permute_449: "f32[128, 128]" = torch.ops.aten.permute.default(permute_448, [1, 0]);  permute_448 = None
    view_513: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_128, [8, 401, 128]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_266: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(view_509, view_513);  view_509 = view_513 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_450: "f32[8, 1, 4, 32]" = torch.ops.aten.permute.default(view_505, [0, 2, 1, 3]);  view_505 = None
    view_514: "f32[8, 1, 128]" = torch.ops.aten.view.default(permute_450, [8, 1, 128]);  permute_450 = None
    sum_198: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(view_514, [0, 1], True)
    view_515: "f32[128]" = torch.ops.aten.view.default(sum_198, [128]);  sum_198 = None
    view_516: "f32[8, 128]" = torch.ops.aten.view.default(view_514, [8, 128]);  view_514 = None
    permute_451: "f32[128, 8]" = torch.ops.aten.permute.default(view_516, [1, 0])
    mm_130: "f32[128, 128]" = torch.ops.aten.mm.default(permute_451, view_70);  permute_451 = view_70 = None
    permute_452: "f32[128, 128]" = torch.ops.aten.permute.default(mm_130, [1, 0]);  mm_130 = None
    permute_453: "f32[128, 128]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    mm_131: "f32[8, 128]" = torch.ops.aten.mm.default(view_516, permute_453);  view_516 = permute_453 = None
    view_517: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_131, [8, 1, 128]);  mm_131 = None
    permute_454: "f32[128, 128]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
    full_51: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_49: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_51, view_517, 1, 0, 1);  full_51 = view_517 = None
    full_52: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_50: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_52, slice_scatter_49, 0, 0, 9223372036854775807);  full_52 = slice_scatter_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_267: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_266, slice_scatter_50);  add_266 = slice_scatter_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    sub_194: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(cat_4, getitem_73);  cat_4 = getitem_73 = None
    mul_599: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_194, rsqrt_12);  sub_194 = None
    mul_600: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(add_267, primals_79);  primals_79 = None
    mul_601: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_600, 128)
    sum_199: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_600, [2], True)
    mul_602: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_600, mul_599);  mul_600 = None
    sum_200: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_602, [2], True);  mul_602 = None
    mul_603: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_599, sum_200);  sum_200 = None
    sub_195: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_601, sum_199);  mul_601 = sum_199 = None
    sub_196: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_195, mul_603);  sub_195 = mul_603 = None
    div_38: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 128);  rsqrt_12 = None
    mul_604: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_38, sub_196);  div_38 = sub_196 = None
    mul_605: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(add_267, mul_599);  mul_599 = None
    sum_201: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_605, [0, 1]);  mul_605 = None
    sum_202: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_267, [0, 1]);  add_267 = None
    full_53: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_51: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_53, slice_scatter_48, 1, 0, 1);  full_53 = slice_scatter_48 = None
    full_54: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_52: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_54, slice_scatter_51, 0, 0, 9223372036854775807);  full_54 = slice_scatter_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_268: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(mul_604, slice_scatter_52);  mul_604 = slice_scatter_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_87: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_268, 1, 0, 1)
    slice_88: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_268, 1, 1, 401);  add_268 = None
    full_55: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_53: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_55, slice_88, 1, 1, 9223372036854775807);  full_55 = slice_88 = None
    full_56: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_54: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_56, slice_scatter_53, 0, 0, 9223372036854775807);  full_56 = slice_scatter_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    slice_89: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_263, 1, 0, 1)
    slice_90: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_263, 1, 1, 401);  add_263 = None
    full_57: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_55: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_57, slice_90, 1, 1, 9223372036854775807);  full_57 = slice_90 = None
    full_58: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_56: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_58, slice_scatter_55, 0, 0, 9223372036854775807);  full_58 = slice_scatter_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    add_269: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(slice_scatter_54, slice_scatter_56);  slice_scatter_54 = slice_scatter_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    view_518: "f32[8, 128]" = torch.ops.aten.view.default(slice_89, [8, 128]);  slice_89 = None
    permute_455: "f32[128, 256]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    mm_132: "f32[8, 256]" = torch.ops.aten.mm.default(view_518, permute_455);  permute_455 = None
    permute_456: "f32[128, 8]" = torch.ops.aten.permute.default(view_518, [1, 0])
    mm_133: "f32[128, 256]" = torch.ops.aten.mm.default(permute_456, view_68);  permute_456 = view_68 = None
    permute_457: "f32[256, 128]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_203: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_518, [0], True);  view_518 = None
    view_519: "f32[128]" = torch.ops.aten.view.default(sum_203, [128]);  sum_203 = None
    permute_458: "f32[128, 256]" = torch.ops.aten.permute.default(permute_457, [1, 0]);  permute_457 = None
    view_520: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_132, [8, 1, 256]);  mm_132 = None
    mul_606: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_87, 0.7071067811865476)
    erf_41: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_606);  mul_606 = None
    add_270: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_41, 1);  erf_41 = None
    mul_607: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_270, 0.5);  add_270 = None
    mul_608: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_87, add_87)
    mul_609: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_608, -0.5);  mul_608 = None
    exp_23: "f32[8, 1, 256]" = torch.ops.aten.exp.default(mul_609);  mul_609 = None
    mul_610: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_611: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_87, mul_610);  add_87 = mul_610 = None
    add_271: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_607, mul_611);  mul_607 = mul_611 = None
    mul_612: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_520, add_271);  view_520 = add_271 = None
    sub_197: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(slice_11, getitem_71);  slice_11 = getitem_71 = None
    mul_613: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_197, rsqrt_11);  sub_197 = None
    mul_614: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_612, primals_75);  primals_75 = None
    mul_615: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_614, 256)
    sum_204: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_614, [2], True)
    mul_616: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_614, mul_613);  mul_614 = None
    sum_205: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_616, [2], True);  mul_616 = None
    mul_617: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_613, sum_205);  sum_205 = None
    sub_198: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(mul_615, sum_204);  mul_615 = sum_204 = None
    sub_199: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(sub_198, mul_617);  sub_198 = mul_617 = None
    div_39: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 256);  rsqrt_11 = None
    mul_618: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(div_39, sub_199);  div_39 = sub_199 = None
    mul_619: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_612, mul_613);  mul_613 = None
    sum_206: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_619, [0, 1]);  mul_619 = None
    sum_207: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_612, [0, 1]);  mul_612 = None
    full_59: "f32[8, 1, 256]" = torch.ops.aten.full.default([8, 1, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_57: "f32[8, 1, 256]" = torch.ops.aten.slice_scatter.default(full_59, mul_618, 0, 0, 9223372036854775807);  full_59 = mul_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    view_521: "f32[8, 256]" = torch.ops.aten.view.default(slice_scatter_57, [8, 256])
    permute_459: "f32[256, 256]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_134: "f32[8, 256]" = torch.ops.aten.mm.default(view_521, permute_459);  permute_459 = None
    permute_460: "f32[256, 8]" = torch.ops.aten.permute.default(view_521, [1, 0])
    mm_135: "f32[256, 256]" = torch.ops.aten.mm.default(permute_460, view_66);  permute_460 = view_66 = None
    permute_461: "f32[256, 256]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_208: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_521, [0], True);  view_521 = None
    view_522: "f32[256]" = torch.ops.aten.view.default(sum_208, [256]);  sum_208 = None
    permute_462: "f32[256, 256]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
    view_523: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_134, [8, 1, 256]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    view_524: "f32[8, 1, 4, 64]" = torch.ops.aten.view.default(view_523, [8, 1, 4, 64]);  view_523 = None
    permute_463: "f32[8, 4, 1, 64]" = torch.ops.aten.permute.default(view_524, [0, 2, 1, 3]);  view_524 = None
    view_525: "f32[32, 1, 64]" = torch.ops.aten.view.default(permute_463, [32, 1, 64]);  permute_463 = None
    permute_464: "f32[32, 197, 1]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    bmm_32: "f32[32, 197, 64]" = torch.ops.aten.bmm.default(permute_464, view_525);  permute_464 = None
    permute_465: "f32[32, 64, 197]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    bmm_33: "f32[32, 1, 197]" = torch.ops.aten.bmm.default(view_525, permute_465);  view_525 = permute_465 = None
    view_526: "f32[8, 4, 197, 64]" = torch.ops.aten.view.default(bmm_32, [8, 4, 197, 64]);  bmm_32 = None
    view_527: "f32[8, 4, 1, 197]" = torch.ops.aten.view.default(bmm_33, [8, 4, 1, 197]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    alias_31: "f32[8, 4, 1, 197]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_620: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(view_527, alias_31);  view_527 = None
    sum_209: "f32[8, 4, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_620, [-1], True)
    mul_621: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(alias_31, sum_209);  alias_31 = sum_209 = None
    sub_200: "f32[8, 4, 1, 197]" = torch.ops.aten.sub.Tensor(mul_620, mul_621);  mul_620 = mul_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    mul_622: "f32[8, 4, 1, 197]" = torch.ops.aten.mul.Tensor(sub_200, 0.125);  sub_200 = None
    view_528: "f32[32, 1, 197]" = torch.ops.aten.view.default(mul_622, [32, 1, 197]);  mul_622 = None
    permute_466: "f32[32, 64, 1]" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None
    bmm_34: "f32[32, 64, 197]" = torch.ops.aten.bmm.default(permute_466, view_528);  permute_466 = None
    permute_467: "f32[32, 197, 64]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    bmm_35: "f32[32, 1, 64]" = torch.ops.aten.bmm.default(view_528, permute_467);  view_528 = permute_467 = None
    view_529: "f32[8, 4, 64, 197]" = torch.ops.aten.view.default(bmm_34, [8, 4, 64, 197]);  bmm_34 = None
    view_530: "f32[8, 4, 1, 64]" = torch.ops.aten.view.default(bmm_35, [8, 4, 1, 64]);  bmm_35 = None
    permute_468: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_529, [0, 1, 3, 2]);  view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_469: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None
    clone_92: "f32[8, 197, 4, 64]" = torch.ops.aten.clone.default(permute_469, memory_format = torch.contiguous_format);  permute_469 = None
    view_531: "f32[8, 197, 256]" = torch.ops.aten.view.default(clone_92, [8, 197, 256]);  clone_92 = None
    view_532: "f32[1576, 256]" = torch.ops.aten.view.default(view_531, [1576, 256]);  view_531 = None
    permute_470: "f32[256, 256]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_136: "f32[1576, 256]" = torch.ops.aten.mm.default(view_532, permute_470);  permute_470 = None
    permute_471: "f32[256, 1576]" = torch.ops.aten.permute.default(view_532, [1, 0])
    mm_137: "f32[256, 256]" = torch.ops.aten.mm.default(permute_471, view_56);  permute_471 = view_56 = None
    permute_472: "f32[256, 256]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_210: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
    view_533: "f32[256]" = torch.ops.aten.view.default(sum_210, [256]);  sum_210 = None
    permute_473: "f32[256, 256]" = torch.ops.aten.permute.default(permute_472, [1, 0]);  permute_472 = None
    view_534: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_136, [8, 197, 256]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_474: "f32[8, 197, 4, 64]" = torch.ops.aten.permute.default(permute_468, [0, 2, 1, 3]);  permute_468 = None
    view_535: "f32[8, 197, 256]" = torch.ops.aten.view.default(permute_474, [8, 197, 256]);  permute_474 = None
    clone_93: "f32[8, 197, 256]" = torch.ops.aten.clone.default(view_535, memory_format = torch.contiguous_format);  view_535 = None
    view_536: "f32[1576, 256]" = torch.ops.aten.view.default(clone_93, [1576, 256]);  clone_93 = None
    permute_475: "f32[256, 256]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_138: "f32[1576, 256]" = torch.ops.aten.mm.default(view_536, permute_475);  permute_475 = None
    permute_476: "f32[256, 1576]" = torch.ops.aten.permute.default(view_536, [1, 0])
    mm_139: "f32[256, 256]" = torch.ops.aten.mm.default(permute_476, view_53);  permute_476 = view_53 = None
    permute_477: "f32[256, 256]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_211: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_536, [0], True);  view_536 = None
    view_537: "f32[256]" = torch.ops.aten.view.default(sum_211, [256]);  sum_211 = None
    permute_478: "f32[256, 256]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
    view_538: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_138, [8, 197, 256]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_272: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(view_534, view_538);  view_534 = view_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    permute_479: "f32[8, 1, 4, 64]" = torch.ops.aten.permute.default(view_530, [0, 2, 1, 3]);  view_530 = None
    view_539: "f32[8, 1, 256]" = torch.ops.aten.view.default(permute_479, [8, 1, 256]);  permute_479 = None
    sum_212: "f32[1, 1, 256]" = torch.ops.aten.sum.dim_IntList(view_539, [0, 1], True)
    view_540: "f32[256]" = torch.ops.aten.view.default(sum_212, [256]);  sum_212 = None
    view_541: "f32[8, 256]" = torch.ops.aten.view.default(view_539, [8, 256]);  view_539 = None
    permute_480: "f32[256, 8]" = torch.ops.aten.permute.default(view_541, [1, 0])
    mm_140: "f32[256, 256]" = torch.ops.aten.mm.default(permute_480, view_50);  permute_480 = view_50 = None
    permute_481: "f32[256, 256]" = torch.ops.aten.permute.default(mm_140, [1, 0]);  mm_140 = None
    permute_482: "f32[256, 256]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    mm_141: "f32[8, 256]" = torch.ops.aten.mm.default(view_541, permute_482);  view_541 = permute_482 = None
    view_542: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_141, [8, 1, 256]);  mm_141 = None
    permute_483: "f32[256, 256]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
    full_60: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_58: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_60, view_542, 1, 0, 1);  full_60 = view_542 = None
    full_61: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_59: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_61, slice_scatter_58, 0, 0, 9223372036854775807);  full_61 = slice_scatter_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    add_273: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_272, slice_scatter_59);  add_272 = slice_scatter_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    sub_201: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(cat_2, getitem_69);  cat_2 = getitem_69 = None
    mul_623: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_201, rsqrt_10);  sub_201 = None
    mul_624: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(add_273, primals_65);  primals_65 = None
    mul_625: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_624, 256)
    sum_213: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_624, [2], True)
    mul_626: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_624, mul_623);  mul_624 = None
    sum_214: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_626, [2], True);  mul_626 = None
    mul_627: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_623, sum_214);  sum_214 = None
    sub_202: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_625, sum_213);  mul_625 = sum_213 = None
    sub_203: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_202, mul_627);  sub_202 = mul_627 = None
    div_40: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 256);  rsqrt_10 = None
    mul_628: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_40, sub_203);  div_40 = sub_203 = None
    mul_629: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(add_273, mul_623);  mul_623 = None
    sum_215: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_629, [0, 1]);  mul_629 = None
    sum_216: "f32[256]" = torch.ops.aten.sum.dim_IntList(add_273, [0, 1]);  add_273 = None
    full_62: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_60: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_62, slice_scatter_57, 1, 0, 1);  full_62 = slice_scatter_57 = None
    full_63: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_61: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_63, slice_scatter_60, 0, 0, 9223372036854775807);  full_63 = slice_scatter_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    add_274: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(mul_628, slice_scatter_61);  mul_628 = slice_scatter_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    slice_91: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_274, 1, 0, 1)
    slice_92: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_274, 1, 1, 197);  add_274 = None
    full_64: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_62: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_64, slice_92, 1, 1, 9223372036854775807);  full_64 = slice_92 = None
    full_65: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_63: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_65, slice_scatter_62, 0, 0, 9223372036854775807);  full_65 = slice_scatter_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    add_275: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(slice_scatter_47, slice_scatter_63);  slice_scatter_47 = slice_scatter_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    view_543: "f32[8, 128]" = torch.ops.aten.view.default(slice_87, [8, 128]);  slice_87 = None
    permute_484: "f32[128, 256]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    mm_142: "f32[8, 256]" = torch.ops.aten.mm.default(view_543, permute_484);  permute_484 = None
    permute_485: "f32[128, 8]" = torch.ops.aten.permute.default(view_543, [1, 0])
    mm_143: "f32[128, 256]" = torch.ops.aten.mm.default(permute_485, view_48);  permute_485 = view_48 = None
    permute_486: "f32[256, 128]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_217: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_543, [0], True);  view_543 = None
    view_544: "f32[128]" = torch.ops.aten.view.default(sum_217, [128]);  sum_217 = None
    permute_487: "f32[128, 256]" = torch.ops.aten.permute.default(permute_486, [1, 0]);  permute_486 = None
    view_545: "f32[8, 1, 256]" = torch.ops.aten.view.default(mm_142, [8, 1, 256]);  mm_142 = None
    mul_630: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_80, 0.7071067811865476)
    erf_42: "f32[8, 1, 256]" = torch.ops.aten.erf.default(mul_630);  mul_630 = None
    add_276: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(erf_42, 1);  erf_42 = None
    mul_631: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_276, 0.5);  add_276 = None
    mul_632: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_80, add_80)
    mul_633: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_632, -0.5);  mul_632 = None
    exp_24: "f32[8, 1, 256]" = torch.ops.aten.exp.default(mul_633);  mul_633 = None
    mul_634: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(exp_24, 0.3989422804014327);  exp_24 = None
    mul_635: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(add_80, mul_634);  add_80 = mul_634 = None
    add_277: "f32[8, 1, 256]" = torch.ops.aten.add.Tensor(mul_631, mul_635);  mul_631 = mul_635 = None
    mul_636: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(view_545, add_277);  view_545 = add_277 = None
    clone_94: "f32[8, 1, 256]" = torch.ops.aten.clone.default(slice_4, memory_format = torch.contiguous_format);  slice_4 = None
    sub_204: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(clone_94, getitem_67);  clone_94 = getitem_67 = None
    mul_637: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(sub_204, rsqrt_9);  sub_204 = None
    mul_638: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_636, primals_61);  primals_61 = None
    mul_639: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_638, 256)
    sum_218: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_638, [2], True)
    mul_640: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_638, mul_637);  mul_638 = None
    sum_219: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_640, [2], True);  mul_640 = None
    mul_641: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_637, sum_219);  sum_219 = None
    sub_205: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(mul_639, sum_218);  mul_639 = sum_218 = None
    sub_206: "f32[8, 1, 256]" = torch.ops.aten.sub.Tensor(sub_205, mul_641);  sub_205 = mul_641 = None
    div_41: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 256);  rsqrt_9 = None
    mul_642: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(div_41, sub_206);  div_41 = sub_206 = None
    mul_643: "f32[8, 1, 256]" = torch.ops.aten.mul.Tensor(mul_636, mul_637);  mul_637 = None
    sum_220: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_643, [0, 1]);  mul_643 = None
    sum_221: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_636, [0, 1]);  mul_636 = None
    full_66: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_64: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_66, mul_642, 1, 0, 1);  full_66 = mul_642 = None
    full_67: "f32[8, 197, 256]" = torch.ops.aten.full.default([8, 197, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_65: "f32[8, 197, 256]" = torch.ops.aten.slice_scatter.default(full_67, slice_scatter_64, 0, 0, 9223372036854775807);  full_67 = slice_scatter_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    add_278: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_275, slice_scatter_65);  add_275 = slice_scatter_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    view_546: "f32[8, 256]" = torch.ops.aten.view.default(slice_91, [8, 256]);  slice_91 = None
    permute_488: "f32[256, 128]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    mm_144: "f32[8, 128]" = torch.ops.aten.mm.default(view_546, permute_488);  permute_488 = None
    permute_489: "f32[256, 8]" = torch.ops.aten.permute.default(view_546, [1, 0])
    mm_145: "f32[256, 128]" = torch.ops.aten.mm.default(permute_489, view_46);  permute_489 = view_46 = None
    permute_490: "f32[128, 256]" = torch.ops.aten.permute.default(mm_145, [1, 0]);  mm_145 = None
    sum_222: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_546, [0], True);  view_546 = None
    view_547: "f32[256]" = torch.ops.aten.view.default(sum_222, [256]);  sum_222 = None
    permute_491: "f32[256, 128]" = torch.ops.aten.permute.default(permute_490, [1, 0]);  permute_490 = None
    view_548: "f32[8, 1, 128]" = torch.ops.aten.view.default(mm_144, [8, 1, 128]);  mm_144 = None
    mul_644: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_77, 0.7071067811865476)
    erf_43: "f32[8, 1, 128]" = torch.ops.aten.erf.default(mul_644);  mul_644 = None
    add_279: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(erf_43, 1);  erf_43 = None
    mul_645: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_279, 0.5);  add_279 = None
    mul_646: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_77, add_77)
    mul_647: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_646, -0.5);  mul_646 = None
    exp_25: "f32[8, 1, 128]" = torch.ops.aten.exp.default(mul_647);  mul_647 = None
    mul_648: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(exp_25, 0.3989422804014327);  exp_25 = None
    mul_649: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(add_77, mul_648);  add_77 = mul_648 = None
    add_280: "f32[8, 1, 128]" = torch.ops.aten.add.Tensor(mul_645, mul_649);  mul_645 = mul_649 = None
    mul_650: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(view_548, add_280);  view_548 = add_280 = None
    clone_95: "f32[8, 1, 128]" = torch.ops.aten.clone.default(slice_2, memory_format = torch.contiguous_format);  slice_2 = None
    sub_207: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(clone_95, getitem_65);  clone_95 = getitem_65 = None
    mul_651: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(sub_207, rsqrt_8);  sub_207 = None
    mul_652: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_650, primals_57);  primals_57 = None
    mul_653: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_652, 128)
    sum_223: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_652, [2], True)
    mul_654: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_652, mul_651);  mul_652 = None
    sum_224: "f32[8, 1, 1]" = torch.ops.aten.sum.dim_IntList(mul_654, [2], True);  mul_654 = None
    mul_655: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_651, sum_224);  sum_224 = None
    sub_208: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(mul_653, sum_223);  mul_653 = sum_223 = None
    sub_209: "f32[8, 1, 128]" = torch.ops.aten.sub.Tensor(sub_208, mul_655);  sub_208 = mul_655 = None
    div_42: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 128);  rsqrt_8 = None
    mul_656: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(div_42, sub_209);  div_42 = sub_209 = None
    mul_657: "f32[8, 1, 128]" = torch.ops.aten.mul.Tensor(mul_650, mul_651);  mul_651 = None
    sum_225: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_657, [0, 1]);  mul_657 = None
    sum_226: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_650, [0, 1]);  mul_650 = None
    full_68: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_66: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_68, mul_656, 1, 0, 1);  full_68 = mul_656 = None
    full_69: "f32[8, 401, 128]" = torch.ops.aten.full.default([8, 401, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_67: "f32[8, 401, 128]" = torch.ops.aten.slice_scatter.default(full_69, slice_scatter_66, 0, 0, 9223372036854775807);  full_69 = slice_scatter_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    add_281: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_269, slice_scatter_67);  add_269 = slice_scatter_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_549: "f32[1576, 256]" = torch.ops.aten.view.default(add_278, [1576, 256])
    permute_492: "f32[256, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_146: "f32[1576, 768]" = torch.ops.aten.mm.default(view_549, permute_492);  permute_492 = None
    permute_493: "f32[256, 1576]" = torch.ops.aten.permute.default(view_549, [1, 0])
    mm_147: "f32[256, 768]" = torch.ops.aten.mm.default(permute_493, view_44);  permute_493 = view_44 = None
    permute_494: "f32[768, 256]" = torch.ops.aten.permute.default(mm_147, [1, 0]);  mm_147 = None
    sum_227: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_549, [0], True);  view_549 = None
    view_550: "f32[256]" = torch.ops.aten.view.default(sum_227, [256]);  sum_227 = None
    permute_495: "f32[256, 768]" = torch.ops.aten.permute.default(permute_494, [1, 0]);  permute_494 = None
    view_551: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_146, [8, 197, 768]);  mm_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_658: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_43, 0.7071067811865476)
    erf_44: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_658);  mul_658 = None
    add_282: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_44, 1);  erf_44 = None
    mul_659: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_282, 0.5);  add_282 = None
    mul_660: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_43, view_43)
    mul_661: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_660, -0.5);  mul_660 = None
    exp_26: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_661);  mul_661 = None
    mul_662: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
    mul_663: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_43, mul_662);  view_43 = mul_662 = None
    add_283: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_659, mul_663);  mul_659 = mul_663 = None
    mul_664: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_551, add_283);  view_551 = add_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_552: "f32[1576, 768]" = torch.ops.aten.view.default(mul_664, [1576, 768]);  mul_664 = None
    permute_496: "f32[768, 256]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_148: "f32[1576, 256]" = torch.ops.aten.mm.default(view_552, permute_496);  permute_496 = None
    permute_497: "f32[768, 1576]" = torch.ops.aten.permute.default(view_552, [1, 0])
    mm_149: "f32[768, 256]" = torch.ops.aten.mm.default(permute_497, view_42);  permute_497 = view_42 = None
    permute_498: "f32[256, 768]" = torch.ops.aten.permute.default(mm_149, [1, 0]);  mm_149 = None
    sum_228: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True);  view_552 = None
    view_553: "f32[768]" = torch.ops.aten.view.default(sum_228, [768]);  sum_228 = None
    permute_499: "f32[768, 256]" = torch.ops.aten.permute.default(permute_498, [1, 0]);  permute_498 = None
    view_554: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_148, [8, 197, 256]);  mm_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_210: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_71, getitem_63);  add_71 = getitem_63 = None
    mul_665: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_210, rsqrt_7);  sub_210 = None
    mul_666: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_554, primals_51);  primals_51 = None
    mul_667: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_666, 256)
    sum_229: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_666, [2], True)
    mul_668: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_666, mul_665);  mul_666 = None
    sum_230: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_668, [2], True);  mul_668 = None
    mul_669: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_665, sum_230);  sum_230 = None
    sub_211: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_667, sum_229);  mul_667 = sum_229 = None
    sub_212: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_211, mul_669);  sub_211 = mul_669 = None
    div_43: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 256);  rsqrt_7 = None
    mul_670: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_43, sub_212);  div_43 = sub_212 = None
    mul_671: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_554, mul_665);  mul_665 = None
    sum_231: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_671, [0, 1]);  mul_671 = None
    sum_232: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_554, [0, 1]);  view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_284: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_278, mul_670);  add_278 = mul_670 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_555: "f32[1576, 256]" = torch.ops.aten.view.default(add_284, [1576, 256])
    permute_500: "f32[256, 256]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_150: "f32[1576, 256]" = torch.ops.aten.mm.default(view_555, permute_500);  permute_500 = None
    permute_501: "f32[256, 1576]" = torch.ops.aten.permute.default(view_555, [1, 0])
    mm_151: "f32[256, 256]" = torch.ops.aten.mm.default(permute_501, view_40);  permute_501 = view_40 = None
    permute_502: "f32[256, 256]" = torch.ops.aten.permute.default(mm_151, [1, 0]);  mm_151 = None
    sum_233: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_555, [0], True);  view_555 = None
    view_556: "f32[256]" = torch.ops.aten.view.default(sum_233, [256]);  sum_233 = None
    permute_503: "f32[256, 256]" = torch.ops.aten.permute.default(permute_502, [1, 0]);  permute_502 = None
    view_557: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_150, [8, 197, 256]);  mm_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_558: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_557, [8, 197, 4, 64]);  view_557 = None
    permute_504: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_558, [0, 2, 1, 3]);  view_558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_32: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    _scaled_dot_product_flash_attention_backward_8 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_504, getitem_50, getitem_51, getitem_52, alias_32, getitem_54, getitem_55, getitem_56, 0, 0, 0.0, False, getitem_59, getitem_60);  permute_504 = getitem_50 = getitem_51 = getitem_52 = alias_32 = getitem_54 = getitem_55 = getitem_56 = getitem_59 = getitem_60 = None
    getitem_256: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_8[0]
    getitem_257: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_8[1]
    getitem_258: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_8[2];  _scaled_dot_product_flash_attention_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_23: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_256, getitem_257, getitem_258]);  getitem_256 = getitem_257 = getitem_258 = None
    view_559: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_23, [3, 8, 4, 197, 64]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_505: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_559, [1, 3, 0, 2, 4]);  view_559 = None
    clone_96: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_505, memory_format = torch.contiguous_format);  permute_505 = None
    view_560: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_96, [8, 197, 768]);  clone_96 = None
    view_561: "f32[1576, 768]" = torch.ops.aten.view.default(view_560, [1576, 768]);  view_560 = None
    permute_506: "f32[768, 256]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_152: "f32[1576, 256]" = torch.ops.aten.mm.default(view_561, permute_506);  permute_506 = None
    permute_507: "f32[768, 1576]" = torch.ops.aten.permute.default(view_561, [1, 0])
    mm_153: "f32[768, 256]" = torch.ops.aten.mm.default(permute_507, view_36);  permute_507 = view_36 = None
    permute_508: "f32[256, 768]" = torch.ops.aten.permute.default(mm_153, [1, 0]);  mm_153 = None
    sum_234: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
    view_562: "f32[768]" = torch.ops.aten.view.default(sum_234, [768]);  sum_234 = None
    permute_509: "f32[768, 256]" = torch.ops.aten.permute.default(permute_508, [1, 0]);  permute_508 = None
    view_563: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_152, [8, 197, 256]);  mm_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_213: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_68, getitem_49);  add_68 = getitem_49 = None
    mul_672: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_213, rsqrt_6);  sub_213 = None
    mul_673: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_563, primals_45);  primals_45 = None
    mul_674: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_673, 256)
    sum_235: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_673, [2], True)
    mul_675: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_673, mul_672);  mul_673 = None
    sum_236: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_675, [2], True);  mul_675 = None
    mul_676: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_672, sum_236);  sum_236 = None
    sub_214: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_674, sum_235);  mul_674 = sum_235 = None
    sub_215: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_214, mul_676);  sub_214 = mul_676 = None
    div_44: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 256);  rsqrt_6 = None
    mul_677: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_44, sub_215);  div_44 = sub_215 = None
    mul_678: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_563, mul_672);  mul_672 = None
    sum_237: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_678, [0, 1]);  mul_678 = None
    sum_238: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_563, [0, 1]);  view_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_285: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_284, mul_677);  add_284 = mul_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_564: "f32[1576, 256]" = torch.ops.aten.view.default(add_285, [1576, 256])
    permute_510: "f32[256, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_154: "f32[1576, 768]" = torch.ops.aten.mm.default(view_564, permute_510);  permute_510 = None
    permute_511: "f32[256, 1576]" = torch.ops.aten.permute.default(view_564, [1, 0])
    mm_155: "f32[256, 768]" = torch.ops.aten.mm.default(permute_511, view_34);  permute_511 = view_34 = None
    permute_512: "f32[768, 256]" = torch.ops.aten.permute.default(mm_155, [1, 0]);  mm_155 = None
    sum_239: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True);  view_564 = None
    view_565: "f32[256]" = torch.ops.aten.view.default(sum_239, [256]);  sum_239 = None
    permute_513: "f32[256, 768]" = torch.ops.aten.permute.default(permute_512, [1, 0]);  permute_512 = None
    view_566: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_154, [8, 197, 768]);  mm_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_679: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_33, 0.7071067811865476)
    erf_45: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_679);  mul_679 = None
    add_286: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_45, 1);  erf_45 = None
    mul_680: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_286, 0.5);  add_286 = None
    mul_681: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_33, view_33)
    mul_682: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_681, -0.5);  mul_681 = None
    exp_27: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_682);  mul_682 = None
    mul_683: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
    mul_684: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_33, mul_683);  view_33 = mul_683 = None
    add_287: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_680, mul_684);  mul_680 = mul_684 = None
    mul_685: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_566, add_287);  view_566 = add_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_567: "f32[1576, 768]" = torch.ops.aten.view.default(mul_685, [1576, 768]);  mul_685 = None
    permute_514: "f32[768, 256]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_156: "f32[1576, 256]" = torch.ops.aten.mm.default(view_567, permute_514);  permute_514 = None
    permute_515: "f32[768, 1576]" = torch.ops.aten.permute.default(view_567, [1, 0])
    mm_157: "f32[768, 256]" = torch.ops.aten.mm.default(permute_515, view_32);  permute_515 = view_32 = None
    permute_516: "f32[256, 768]" = torch.ops.aten.permute.default(mm_157, [1, 0]);  mm_157 = None
    sum_240: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
    view_568: "f32[768]" = torch.ops.aten.view.default(sum_240, [768]);  sum_240 = None
    permute_517: "f32[768, 256]" = torch.ops.aten.permute.default(permute_516, [1, 0]);  permute_516 = None
    view_569: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_156, [8, 197, 256]);  mm_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_216: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_64, getitem_47);  add_64 = getitem_47 = None
    mul_686: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_216, rsqrt_5);  sub_216 = None
    mul_687: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_569, primals_39);  primals_39 = None
    mul_688: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_687, 256)
    sum_241: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_687, [2], True)
    mul_689: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_687, mul_686);  mul_687 = None
    sum_242: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_689, [2], True);  mul_689 = None
    mul_690: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_686, sum_242);  sum_242 = None
    sub_217: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_688, sum_241);  mul_688 = sum_241 = None
    sub_218: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_217, mul_690);  sub_217 = mul_690 = None
    div_45: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None
    mul_691: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_45, sub_218);  div_45 = sub_218 = None
    mul_692: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_569, mul_686);  mul_686 = None
    sum_243: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_692, [0, 1]);  mul_692 = None
    sum_244: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_569, [0, 1]);  view_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_288: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_285, mul_691);  add_285 = mul_691 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_570: "f32[1576, 256]" = torch.ops.aten.view.default(add_288, [1576, 256])
    permute_518: "f32[256, 256]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    mm_158: "f32[1576, 256]" = torch.ops.aten.mm.default(view_570, permute_518);  permute_518 = None
    permute_519: "f32[256, 1576]" = torch.ops.aten.permute.default(view_570, [1, 0])
    mm_159: "f32[256, 256]" = torch.ops.aten.mm.default(permute_519, view_30);  permute_519 = view_30 = None
    permute_520: "f32[256, 256]" = torch.ops.aten.permute.default(mm_159, [1, 0]);  mm_159 = None
    sum_245: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
    view_571: "f32[256]" = torch.ops.aten.view.default(sum_245, [256]);  sum_245 = None
    permute_521: "f32[256, 256]" = torch.ops.aten.permute.default(permute_520, [1, 0]);  permute_520 = None
    view_572: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_158, [8, 197, 256]);  mm_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_573: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_572, [8, 197, 4, 64]);  view_572 = None
    permute_522: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_33: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    _scaled_dot_product_flash_attention_backward_9 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_522, getitem_34, getitem_35, getitem_36, alias_33, getitem_38, getitem_39, getitem_40, 0, 0, 0.0, False, getitem_43, getitem_44);  permute_522 = getitem_34 = getitem_35 = getitem_36 = alias_33 = getitem_38 = getitem_39 = getitem_40 = getitem_43 = getitem_44 = None
    getitem_259: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_9[0]
    getitem_260: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_9[1]
    getitem_261: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_9[2];  _scaled_dot_product_flash_attention_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_24: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_259, getitem_260, getitem_261]);  getitem_259 = getitem_260 = getitem_261 = None
    view_574: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_24, [3, 8, 4, 197, 64]);  cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_523: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_574, [1, 3, 0, 2, 4]);  view_574 = None
    clone_97: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
    view_575: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_97, [8, 197, 768]);  clone_97 = None
    view_576: "f32[1576, 768]" = torch.ops.aten.view.default(view_575, [1576, 768]);  view_575 = None
    permute_524: "f32[768, 256]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_160: "f32[1576, 256]" = torch.ops.aten.mm.default(view_576, permute_524);  permute_524 = None
    permute_525: "f32[768, 1576]" = torch.ops.aten.permute.default(view_576, [1, 0])
    mm_161: "f32[768, 256]" = torch.ops.aten.mm.default(permute_525, view_26);  permute_525 = view_26 = None
    permute_526: "f32[256, 768]" = torch.ops.aten.permute.default(mm_161, [1, 0]);  mm_161 = None
    sum_246: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
    view_577: "f32[768]" = torch.ops.aten.view.default(sum_246, [768]);  sum_246 = None
    permute_527: "f32[768, 256]" = torch.ops.aten.permute.default(permute_526, [1, 0]);  permute_526 = None
    view_578: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_160, [8, 197, 256]);  mm_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_219: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_61, getitem_33);  add_61 = getitem_33 = None
    mul_693: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_219, rsqrt_4);  sub_219 = None
    mul_694: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_578, primals_33);  primals_33 = None
    mul_695: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_694, 256)
    sum_247: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_694, [2], True)
    mul_696: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_694, mul_693);  mul_694 = None
    sum_248: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_696, [2], True);  mul_696 = None
    mul_697: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_693, sum_248);  sum_248 = None
    sub_220: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_695, sum_247);  mul_695 = sum_247 = None
    sub_221: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_220, mul_697);  sub_220 = mul_697 = None
    div_46: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 256);  rsqrt_4 = None
    mul_698: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_46, sub_221);  div_46 = sub_221 = None
    mul_699: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_578, mul_693);  mul_693 = None
    sum_249: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_699, [0, 1]);  mul_699 = None
    sum_250: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_578, [0, 1]);  view_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_289: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_288, mul_698);  add_288 = mul_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_579: "f32[1576, 256]" = torch.ops.aten.view.default(add_289, [1576, 256])
    permute_528: "f32[256, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_162: "f32[1576, 768]" = torch.ops.aten.mm.default(view_579, permute_528);  permute_528 = None
    permute_529: "f32[256, 1576]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_163: "f32[256, 768]" = torch.ops.aten.mm.default(permute_529, view_24);  permute_529 = view_24 = None
    permute_530: "f32[768, 256]" = torch.ops.aten.permute.default(mm_163, [1, 0]);  mm_163 = None
    sum_251: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[256]" = torch.ops.aten.view.default(sum_251, [256]);  sum_251 = None
    permute_531: "f32[256, 768]" = torch.ops.aten.permute.default(permute_530, [1, 0]);  permute_530 = None
    view_581: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_162, [8, 197, 768]);  mm_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_700: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_23, 0.7071067811865476)
    erf_46: "f32[8, 197, 768]" = torch.ops.aten.erf.default(mul_700);  mul_700 = None
    add_290: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(erf_46, 1);  erf_46 = None
    mul_701: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_290, 0.5);  add_290 = None
    mul_702: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_23, view_23)
    mul_703: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_702, -0.5);  mul_702 = None
    exp_28: "f32[8, 197, 768]" = torch.ops.aten.exp.default(mul_703);  mul_703 = None
    mul_704: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
    mul_705: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_23, mul_704);  view_23 = mul_704 = None
    add_291: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_701, mul_705);  mul_701 = mul_705 = None
    mul_706: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_581, add_291);  view_581 = add_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_582: "f32[1576, 768]" = torch.ops.aten.view.default(mul_706, [1576, 768]);  mul_706 = None
    permute_532: "f32[768, 256]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_164: "f32[1576, 256]" = torch.ops.aten.mm.default(view_582, permute_532);  permute_532 = None
    permute_533: "f32[768, 1576]" = torch.ops.aten.permute.default(view_582, [1, 0])
    mm_165: "f32[768, 256]" = torch.ops.aten.mm.default(permute_533, view_22);  permute_533 = view_22 = None
    permute_534: "f32[256, 768]" = torch.ops.aten.permute.default(mm_165, [1, 0]);  mm_165 = None
    sum_252: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_582, [0], True);  view_582 = None
    view_583: "f32[768]" = torch.ops.aten.view.default(sum_252, [768]);  sum_252 = None
    permute_535: "f32[768, 256]" = torch.ops.aten.permute.default(permute_534, [1, 0]);  permute_534 = None
    view_584: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_164, [8, 197, 256]);  mm_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_222: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(add_57, getitem_31);  add_57 = getitem_31 = None
    mul_707: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_222, rsqrt_3);  sub_222 = None
    mul_708: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_584, primals_27);  primals_27 = None
    mul_709: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_708, 256)
    sum_253: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_708, [2], True)
    mul_710: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_708, mul_707);  mul_708 = None
    sum_254: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_710, [2], True);  mul_710 = None
    mul_711: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_707, sum_254);  sum_254 = None
    sub_223: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_709, sum_253);  mul_709 = sum_253 = None
    sub_224: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_223, mul_711);  sub_223 = mul_711 = None
    div_47: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 256);  rsqrt_3 = None
    mul_712: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_47, sub_224);  div_47 = sub_224 = None
    mul_713: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_584, mul_707);  mul_707 = None
    sum_255: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_713, [0, 1]);  mul_713 = None
    sum_256: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_584, [0, 1]);  view_584 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_292: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_289, mul_712);  add_289 = mul_712 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_585: "f32[1576, 256]" = torch.ops.aten.view.default(add_292, [1576, 256])
    permute_536: "f32[256, 256]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_166: "f32[1576, 256]" = torch.ops.aten.mm.default(view_585, permute_536);  permute_536 = None
    permute_537: "f32[256, 1576]" = torch.ops.aten.permute.default(view_585, [1, 0])
    mm_167: "f32[256, 256]" = torch.ops.aten.mm.default(permute_537, view_20);  permute_537 = view_20 = None
    permute_538: "f32[256, 256]" = torch.ops.aten.permute.default(mm_167, [1, 0]);  mm_167 = None
    sum_257: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_585, [0], True);  view_585 = None
    view_586: "f32[256]" = torch.ops.aten.view.default(sum_257, [256]);  sum_257 = None
    permute_539: "f32[256, 256]" = torch.ops.aten.permute.default(permute_538, [1, 0]);  permute_538 = None
    view_587: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_166, [8, 197, 256]);  mm_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_588: "f32[8, 197, 4, 64]" = torch.ops.aten.view.default(view_587, [8, 197, 4, 64]);  view_587 = None
    permute_540: "f32[8, 4, 197, 64]" = torch.ops.aten.permute.default(view_588, [0, 2, 1, 3]);  view_588 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_34: "f32[8, 4, 197, 64]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    _scaled_dot_product_flash_attention_backward_10 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_540, getitem_18, getitem_19, getitem_20, alias_34, getitem_22, getitem_23, getitem_24, 0, 0, 0.0, False, getitem_27, getitem_28);  permute_540 = getitem_18 = getitem_19 = getitem_20 = alias_34 = getitem_22 = getitem_23 = getitem_24 = getitem_27 = getitem_28 = None
    getitem_262: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_10[0]
    getitem_263: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_10[1]
    getitem_264: "f32[8, 4, 197, 64]" = _scaled_dot_product_flash_attention_backward_10[2];  _scaled_dot_product_flash_attention_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_25: "f32[24, 4, 197, 64]" = torch.ops.aten.cat.default([getitem_262, getitem_263, getitem_264]);  getitem_262 = getitem_263 = getitem_264 = None
    view_589: "f32[3, 8, 4, 197, 64]" = torch.ops.aten.view.default(cat_25, [3, 8, 4, 197, 64]);  cat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_541: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.permute.default(view_589, [1, 3, 0, 2, 4]);  view_589 = None
    clone_98: "f32[8, 197, 3, 4, 64]" = torch.ops.aten.clone.default(permute_541, memory_format = torch.contiguous_format);  permute_541 = None
    view_590: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_98, [8, 197, 768]);  clone_98 = None
    view_591: "f32[1576, 768]" = torch.ops.aten.view.default(view_590, [1576, 768]);  view_590 = None
    permute_542: "f32[768, 256]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_168: "f32[1576, 256]" = torch.ops.aten.mm.default(view_591, permute_542);  permute_542 = None
    permute_543: "f32[768, 1576]" = torch.ops.aten.permute.default(view_591, [1, 0])
    mm_169: "f32[768, 256]" = torch.ops.aten.mm.default(permute_543, view_16);  permute_543 = view_16 = None
    permute_544: "f32[256, 768]" = torch.ops.aten.permute.default(mm_169, [1, 0]);  mm_169 = None
    sum_258: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_591, [0], True);  view_591 = None
    view_592: "f32[768]" = torch.ops.aten.view.default(sum_258, [768]);  sum_258 = None
    permute_545: "f32[768, 256]" = torch.ops.aten.permute.default(permute_544, [1, 0]);  permute_544 = None
    view_593: "f32[8, 197, 256]" = torch.ops.aten.view.default(mm_168, [8, 197, 256]);  mm_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_225: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(clone_1, getitem_17);  clone_1 = getitem_17 = None
    mul_714: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(sub_225, rsqrt_2);  sub_225 = None
    mul_715: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_593, primals_21);  primals_21 = None
    mul_716: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_715, 256)
    sum_259: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_715, [2], True)
    mul_717: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_715, mul_714);  mul_715 = None
    sum_260: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_717, [2], True);  mul_717 = None
    mul_718: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(mul_714, sum_260);  sum_260 = None
    sub_226: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(mul_716, sum_259);  mul_716 = sum_259 = None
    sub_227: "f32[8, 197, 256]" = torch.ops.aten.sub.Tensor(sub_226, mul_718);  sub_226 = mul_718 = None
    div_48: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 256);  rsqrt_2 = None
    mul_719: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(div_48, sub_227);  div_48 = sub_227 = None
    mul_720: "f32[8, 197, 256]" = torch.ops.aten.mul.Tensor(view_593, mul_714);  mul_714 = None
    sum_261: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_720, [0, 1]);  mul_720 = None
    sum_262: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_593, [0, 1]);  view_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_293: "f32[8, 197, 256]" = torch.ops.aten.add.Tensor(add_292, mul_719);  add_292 = mul_719 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_594: "f32[3208, 128]" = torch.ops.aten.view.default(add_281, [3208, 128])
    permute_546: "f32[128, 384]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    mm_170: "f32[3208, 384]" = torch.ops.aten.mm.default(view_594, permute_546);  permute_546 = None
    permute_547: "f32[128, 3208]" = torch.ops.aten.permute.default(view_594, [1, 0])
    mm_171: "f32[128, 384]" = torch.ops.aten.mm.default(permute_547, view_14);  permute_547 = view_14 = None
    permute_548: "f32[384, 128]" = torch.ops.aten.permute.default(mm_171, [1, 0]);  mm_171 = None
    sum_263: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_594, [0], True);  view_594 = None
    view_595: "f32[128]" = torch.ops.aten.view.default(sum_263, [128]);  sum_263 = None
    permute_549: "f32[128, 384]" = torch.ops.aten.permute.default(permute_548, [1, 0]);  permute_548 = None
    view_596: "f32[8, 401, 384]" = torch.ops.aten.view.default(mm_170, [8, 401, 384]);  mm_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_721: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_13, 0.7071067811865476)
    erf_47: "f32[8, 401, 384]" = torch.ops.aten.erf.default(mul_721);  mul_721 = None
    add_294: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(erf_47, 1);  erf_47 = None
    mul_722: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(add_294, 0.5);  add_294 = None
    mul_723: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_13, view_13)
    mul_724: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(mul_723, -0.5);  mul_723 = None
    exp_29: "f32[8, 401, 384]" = torch.ops.aten.exp.default(mul_724);  mul_724 = None
    mul_725: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
    mul_726: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_13, mul_725);  view_13 = mul_725 = None
    add_295: "f32[8, 401, 384]" = torch.ops.aten.add.Tensor(mul_722, mul_726);  mul_722 = mul_726 = None
    mul_727: "f32[8, 401, 384]" = torch.ops.aten.mul.Tensor(view_596, add_295);  view_596 = add_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_597: "f32[3208, 384]" = torch.ops.aten.view.default(mul_727, [3208, 384]);  mul_727 = None
    permute_550: "f32[384, 128]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    mm_172: "f32[3208, 128]" = torch.ops.aten.mm.default(view_597, permute_550);  permute_550 = None
    permute_551: "f32[384, 3208]" = torch.ops.aten.permute.default(view_597, [1, 0])
    mm_173: "f32[384, 128]" = torch.ops.aten.mm.default(permute_551, view_12);  permute_551 = view_12 = None
    permute_552: "f32[128, 384]" = torch.ops.aten.permute.default(mm_173, [1, 0]);  mm_173 = None
    sum_264: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_597, [0], True);  view_597 = None
    view_598: "f32[384]" = torch.ops.aten.view.default(sum_264, [384]);  sum_264 = None
    permute_553: "f32[384, 128]" = torch.ops.aten.permute.default(permute_552, [1, 0]);  permute_552 = None
    view_599: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_172, [8, 401, 128]);  mm_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_228: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(add_50, getitem_15);  add_50 = getitem_15 = None
    mul_728: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_228, rsqrt_1);  sub_228 = None
    mul_729: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_599, primals_15);  primals_15 = None
    mul_730: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_729, 128)
    sum_265: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_729, [2], True)
    mul_731: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_729, mul_728);  mul_729 = None
    sum_266: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_731, [2], True);  mul_731 = None
    mul_732: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_728, sum_266);  sum_266 = None
    sub_229: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_730, sum_265);  mul_730 = sum_265 = None
    sub_230: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_229, mul_732);  sub_229 = mul_732 = None
    div_49: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 128);  rsqrt_1 = None
    mul_733: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_49, sub_230);  div_49 = sub_230 = None
    mul_734: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_599, mul_728);  mul_728 = None
    sum_267: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_734, [0, 1]);  mul_734 = None
    sum_268: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_599, [0, 1]);  view_599 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_296: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_281, mul_733);  add_281 = mul_733 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_600: "f32[3208, 128]" = torch.ops.aten.view.default(add_296, [3208, 128])
    permute_554: "f32[128, 128]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm_174: "f32[3208, 128]" = torch.ops.aten.mm.default(view_600, permute_554);  permute_554 = None
    permute_555: "f32[128, 3208]" = torch.ops.aten.permute.default(view_600, [1, 0])
    mm_175: "f32[128, 128]" = torch.ops.aten.mm.default(permute_555, view_10);  permute_555 = view_10 = None
    permute_556: "f32[128, 128]" = torch.ops.aten.permute.default(mm_175, [1, 0]);  mm_175 = None
    sum_269: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True);  view_600 = None
    view_601: "f32[128]" = torch.ops.aten.view.default(sum_269, [128]);  sum_269 = None
    permute_557: "f32[128, 128]" = torch.ops.aten.permute.default(permute_556, [1, 0]);  permute_556 = None
    view_602: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_174, [8, 401, 128]);  mm_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_603: "f32[8, 401, 4, 32]" = torch.ops.aten.view.default(view_602, [8, 401, 4, 32]);  view_602 = None
    permute_558: "f32[8, 4, 401, 32]" = torch.ops.aten.permute.default(view_603, [0, 2, 1, 3]);  view_603 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_35: "f32[8, 4, 401, 32]" = torch.ops.aten.alias.default(alias);  alias = None
    _scaled_dot_product_flash_attention_backward_11 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_558, getitem_2, getitem_3, getitem_4, alias_35, getitem_6, getitem_7, getitem_8, 0, 0, 0.0, False, getitem_11, getitem_12);  permute_558 = getitem_2 = getitem_3 = getitem_4 = alias_35 = getitem_6 = getitem_7 = getitem_8 = getitem_11 = getitem_12 = None
    getitem_265: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_11[0]
    getitem_266: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_11[1]
    getitem_267: "f32[8, 4, 401, 32]" = _scaled_dot_product_flash_attention_backward_11[2];  _scaled_dot_product_flash_attention_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_26: "f32[24, 4, 401, 32]" = torch.ops.aten.cat.default([getitem_265, getitem_266, getitem_267]);  getitem_265 = getitem_266 = getitem_267 = None
    view_604: "f32[3, 8, 4, 401, 32]" = torch.ops.aten.view.default(cat_26, [3, 8, 4, 401, 32]);  cat_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_559: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.permute.default(view_604, [1, 3, 0, 2, 4]);  view_604 = None
    clone_99: "f32[8, 401, 3, 4, 32]" = torch.ops.aten.clone.default(permute_559, memory_format = torch.contiguous_format);  permute_559 = None
    view_605: "f32[8, 401, 384]" = torch.ops.aten.view.default(clone_99, [8, 401, 384]);  clone_99 = None
    view_606: "f32[3208, 384]" = torch.ops.aten.view.default(view_605, [3208, 384]);  view_605 = None
    permute_560: "f32[384, 128]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_176: "f32[3208, 128]" = torch.ops.aten.mm.default(view_606, permute_560);  permute_560 = None
    permute_561: "f32[384, 3208]" = torch.ops.aten.permute.default(view_606, [1, 0])
    mm_177: "f32[384, 128]" = torch.ops.aten.mm.default(permute_561, view_6);  permute_561 = view_6 = None
    permute_562: "f32[128, 384]" = torch.ops.aten.permute.default(mm_177, [1, 0]);  mm_177 = None
    sum_270: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_606, [0], True);  view_606 = None
    view_607: "f32[384]" = torch.ops.aten.view.default(sum_270, [384]);  sum_270 = None
    permute_563: "f32[384, 128]" = torch.ops.aten.permute.default(permute_562, [1, 0]);  permute_562 = None
    view_608: "f32[8, 401, 128]" = torch.ops.aten.view.default(mm_176, [8, 401, 128]);  mm_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_231: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul_735: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(sub_231, rsqrt);  sub_231 = None
    mul_736: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_608, primals_9);  primals_9 = None
    mul_737: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_736, 128)
    sum_271: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_736, [2], True)
    mul_738: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_736, mul_735);  mul_736 = None
    sum_272: "f32[8, 401, 1]" = torch.ops.aten.sum.dim_IntList(mul_738, [2], True);  mul_738 = None
    mul_739: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(mul_735, sum_272);  sum_272 = None
    sub_232: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(mul_737, sum_271);  mul_737 = sum_271 = None
    sub_233: "f32[8, 401, 128]" = torch.ops.aten.sub.Tensor(sub_232, mul_739);  sub_232 = mul_739 = None
    div_50: "f32[8, 401, 1]" = torch.ops.aten.div.Tensor(rsqrt, 128);  rsqrt = None
    mul_740: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(div_50, sub_233);  div_50 = sub_233 = None
    mul_741: "f32[8, 401, 128]" = torch.ops.aten.mul.Tensor(view_608, mul_735);  mul_735 = None
    sum_273: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_741, [0, 1]);  mul_741 = None
    sum_274: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_608, [0, 1]);  view_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_297: "f32[8, 401, 128]" = torch.ops.aten.add.Tensor(add_296, mul_740);  add_296 = mul_740 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:443, code: x_ = x_ + pos_embed
    sum_275: "f32[1, 197, 256]" = torch.ops.aten.sum.dim_IntList(add_293, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:441, code: x_ = torch.cat((cls_tokens, x_), dim=1)
    slice_93: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(add_293, 1, 0, 1)
    slice_94: "f32[8, 196, 256]" = torch.ops.aten.slice.Tensor(add_293, 1, 1, 197);  add_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:440, code: cls_tokens = cls_tokens.expand(B, -1, -1)
    sum_276: "f32[1, 1, 256]" = torch.ops.aten.sum.dim_IntList(slice_93, [0], True);  slice_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:84, code: x = self.proj(x).flatten(2).transpose(1, 2)
    permute_564: "f32[8, 256, 196]" = torch.ops.aten.permute.default(slice_94, [0, 2, 1]);  slice_94 = None
    view_609: "f32[8, 256, 14, 14]" = torch.ops.aten.view.default(permute_564, [8, 256, 14, 14]);  permute_564 = None
    convolution_backward = torch.ops.aten.convolution_backward.default(view_609, add_46, primals_7, [256], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_609 = add_46 = primals_7 = None
    getitem_269: "f32[256, 3, 16, 16]" = convolution_backward[1]
    getitem_270: "f32[256]" = convolution_backward[2];  convolution_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:443, code: x_ = x_ + pos_embed
    sum_277: "f32[1, 401, 128]" = torch.ops.aten.sum.dim_IntList(add_297, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:441, code: x_ = torch.cat((cls_tokens, x_), dim=1)
    slice_95: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_297, 1, 0, 1)
    slice_96: "f32[8, 400, 128]" = torch.ops.aten.slice.Tensor(add_297, 1, 1, 401);  add_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:440, code: cls_tokens = cls_tokens.expand(B, -1, -1)
    sum_278: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(slice_95, [0], True);  slice_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:84, code: x = self.proj(x).flatten(2).transpose(1, 2)
    permute_565: "f32[8, 128, 400]" = torch.ops.aten.permute.default(slice_96, [0, 2, 1]);  slice_96 = None
    view_610: "f32[8, 128, 20, 20]" = torch.ops.aten.view.default(permute_565, [8, 128, 20, 20]);  permute_565 = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(view_610, primals_269, primals_5, [128], [12, 12], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_610 = primals_269 = primals_5 = None
    getitem_272: "f32[128, 3, 12, 12]" = convolution_backward_1[1]
    getitem_273: "f32[128]" = convolution_backward_1[2];  convolution_backward_1 = None
    return pytree.tree_unflatten([mean, sum_278, sum_277, sum_276, sum_275, getitem_272, getitem_273, getitem_269, getitem_270, sum_273, sum_274, permute_563, view_607, permute_557, view_601, sum_267, sum_268, permute_553, view_598, permute_549, view_595, sum_261, sum_262, permute_545, view_592, permute_539, view_586, sum_255, sum_256, permute_535, view_583, permute_531, view_580, sum_249, sum_250, permute_527, view_577, permute_521, view_571, sum_243, sum_244, permute_517, view_568, permute_513, view_565, sum_237, sum_238, permute_509, view_562, permute_503, view_556, sum_231, sum_232, permute_499, view_553, permute_495, view_550, sum_225, sum_226, permute_491, view_547, sum_220, sum_221, permute_487, view_544, sum_215, sum_216, permute_483, view_540, permute_478, view_537, permute_473, view_533, permute_462, view_522, sum_206, sum_207, permute_458, view_519, sum_201, sum_202, permute_454, view_515, permute_449, view_512, permute_444, view_508, permute_433, view_497, sum_192, sum_193, permute_429, view_494, sum_187, sum_188, permute_425, view_491, permute_419, view_485, sum_181, sum_182, permute_415, view_482, permute_411, view_479, sum_175, sum_176, permute_407, view_476, permute_401, view_470, sum_169, sum_170, permute_397, view_467, permute_393, view_464, sum_163, sum_164, permute_389, view_461, permute_383, view_455, sum_157, sum_158, permute_379, view_452, permute_375, view_449, sum_151, sum_152, permute_371, view_446, permute_365, view_440, sum_145, sum_146, permute_361, view_437, permute_357, view_434, sum_139, sum_140, permute_353, view_431, sum_134, sum_135, permute_349, view_428, sum_129, sum_130, permute_345, view_424, permute_340, view_421, permute_335, view_417, permute_324, view_406, sum_120, sum_121, permute_320, view_403, sum_115, sum_116, permute_316, view_399, permute_311, view_396, permute_306, view_392, permute_295, view_381, sum_106, sum_107, permute_291, view_378, sum_101, sum_102, permute_287, view_375, permute_281, view_369, sum_95, sum_96, permute_277, view_366, permute_273, view_363, sum_89, sum_90, permute_269, view_360, permute_263, view_354, sum_83, sum_84, permute_259, view_351, permute_255, view_348, sum_77, sum_78, permute_251, view_345, permute_245, view_339, sum_71, sum_72, permute_241, view_336, permute_237, view_333, sum_65, sum_66, permute_233, view_330, permute_227, view_324, sum_59, sum_60, permute_223, view_321, permute_219, view_318, sum_53, sum_54, permute_215, view_315, sum_48, sum_49, permute_211, view_312, sum_43, sum_44, permute_207, view_308, permute_202, view_305, permute_197, view_301, permute_186, view_290, sum_34, sum_35, permute_182, view_287, sum_29, sum_30, permute_178, view_283, permute_173, view_280, permute_168, view_276, permute_157, view_265, sum_20, sum_21, permute_153, view_262, sum_15, sum_16, sum_11, sum_12, permute_149, view_260, permute_145, view_259, None], self._out_spec)
    