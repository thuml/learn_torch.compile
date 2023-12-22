from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1, 1, 768]"; primals_2: "f32[768]"; primals_3: "f32[768]"; primals_4: "f32[768]"; primals_5: "f32[768]"; primals_6: "f32[768]"; primals_7: "f32[2304, 768]"; primals_8: "f32[732, 12]"; primals_9: "f32[768]"; primals_10: "f32[768]"; primals_11: "f32[768]"; primals_12: "f32[768]"; primals_13: "f32[768]"; primals_14: "f32[768]"; primals_15: "f32[768]"; primals_16: "f32[768]"; primals_17: "f32[2304, 768]"; primals_18: "f32[732, 12]"; primals_19: "f32[768]"; primals_20: "f32[768]"; primals_21: "f32[768]"; primals_22: "f32[768]"; primals_23: "f32[768]"; primals_24: "f32[768]"; primals_25: "f32[768]"; primals_26: "f32[768]"; primals_27: "f32[2304, 768]"; primals_28: "f32[732, 12]"; primals_29: "f32[768]"; primals_30: "f32[768]"; primals_31: "f32[768]"; primals_32: "f32[768]"; primals_33: "f32[768]"; primals_34: "f32[768]"; primals_35: "f32[768]"; primals_36: "f32[768]"; primals_37: "f32[2304, 768]"; primals_38: "f32[732, 12]"; primals_39: "f32[768]"; primals_40: "f32[768]"; primals_41: "f32[768]"; primals_42: "f32[768]"; primals_43: "f32[768]"; primals_44: "f32[768]"; primals_45: "f32[768]"; primals_46: "f32[768]"; primals_47: "f32[2304, 768]"; primals_48: "f32[732, 12]"; primals_49: "f32[768]"; primals_50: "f32[768]"; primals_51: "f32[768]"; primals_52: "f32[768]"; primals_53: "f32[768]"; primals_54: "f32[768]"; primals_55: "f32[768]"; primals_56: "f32[768]"; primals_57: "f32[2304, 768]"; primals_58: "f32[732, 12]"; primals_59: "f32[768]"; primals_60: "f32[768]"; primals_61: "f32[768]"; primals_62: "f32[768]"; primals_63: "f32[768]"; primals_64: "f32[768]"; primals_65: "f32[768]"; primals_66: "f32[768]"; primals_67: "f32[2304, 768]"; primals_68: "f32[732, 12]"; primals_69: "f32[768]"; primals_70: "f32[768]"; primals_71: "f32[768]"; primals_72: "f32[768]"; primals_73: "f32[768]"; primals_74: "f32[768]"; primals_75: "f32[768]"; primals_76: "f32[768]"; primals_77: "f32[2304, 768]"; primals_78: "f32[732, 12]"; primals_79: "f32[768]"; primals_80: "f32[768]"; primals_81: "f32[768]"; primals_82: "f32[768]"; primals_83: "f32[768]"; primals_84: "f32[768]"; primals_85: "f32[768]"; primals_86: "f32[768]"; primals_87: "f32[2304, 768]"; primals_88: "f32[732, 12]"; primals_89: "f32[768]"; primals_90: "f32[768]"; primals_91: "f32[768]"; primals_92: "f32[768]"; primals_93: "f32[768]"; primals_94: "f32[768]"; primals_95: "f32[768]"; primals_96: "f32[768]"; primals_97: "f32[2304, 768]"; primals_98: "f32[732, 12]"; primals_99: "f32[768]"; primals_100: "f32[768]"; primals_101: "f32[768]"; primals_102: "f32[768]"; primals_103: "f32[768]"; primals_104: "f32[768]"; primals_105: "f32[768]"; primals_106: "f32[768]"; primals_107: "f32[2304, 768]"; primals_108: "f32[732, 12]"; primals_109: "f32[768]"; primals_110: "f32[768]"; primals_111: "f32[768]"; primals_112: "f32[768]"; primals_113: "f32[768]"; primals_114: "f32[768]"; primals_115: "f32[768]"; primals_116: "f32[768]"; primals_117: "f32[2304, 768]"; primals_118: "f32[732, 12]"; primals_119: "f32[768]"; primals_120: "f32[768]"; primals_121: "f32[768]"; primals_122: "f32[768]"; primals_123: "f32[768]"; primals_124: "f32[768, 3, 16, 16]"; primals_125: "f32[768]"; primals_126: "f32[768, 768]"; primals_127: "f32[768]"; primals_128: "f32[3072, 768]"; primals_129: "f32[3072]"; primals_130: "f32[768, 3072]"; primals_131: "f32[768]"; primals_132: "f32[768, 768]"; primals_133: "f32[768]"; primals_134: "f32[3072, 768]"; primals_135: "f32[3072]"; primals_136: "f32[768, 3072]"; primals_137: "f32[768]"; primals_138: "f32[768, 768]"; primals_139: "f32[768]"; primals_140: "f32[3072, 768]"; primals_141: "f32[3072]"; primals_142: "f32[768, 3072]"; primals_143: "f32[768]"; primals_144: "f32[768, 768]"; primals_145: "f32[768]"; primals_146: "f32[3072, 768]"; primals_147: "f32[3072]"; primals_148: "f32[768, 3072]"; primals_149: "f32[768]"; primals_150: "f32[768, 768]"; primals_151: "f32[768]"; primals_152: "f32[3072, 768]"; primals_153: "f32[3072]"; primals_154: "f32[768, 3072]"; primals_155: "f32[768]"; primals_156: "f32[768, 768]"; primals_157: "f32[768]"; primals_158: "f32[3072, 768]"; primals_159: "f32[3072]"; primals_160: "f32[768, 3072]"; primals_161: "f32[768]"; primals_162: "f32[768, 768]"; primals_163: "f32[768]"; primals_164: "f32[3072, 768]"; primals_165: "f32[3072]"; primals_166: "f32[768, 3072]"; primals_167: "f32[768]"; primals_168: "f32[768, 768]"; primals_169: "f32[768]"; primals_170: "f32[3072, 768]"; primals_171: "f32[3072]"; primals_172: "f32[768, 3072]"; primals_173: "f32[768]"; primals_174: "f32[768, 768]"; primals_175: "f32[768]"; primals_176: "f32[3072, 768]"; primals_177: "f32[3072]"; primals_178: "f32[768, 3072]"; primals_179: "f32[768]"; primals_180: "f32[768, 768]"; primals_181: "f32[768]"; primals_182: "f32[3072, 768]"; primals_183: "f32[3072]"; primals_184: "f32[768, 3072]"; primals_185: "f32[768]"; primals_186: "f32[768, 768]"; primals_187: "f32[768]"; primals_188: "f32[3072, 768]"; primals_189: "f32[3072]"; primals_190: "f32[768, 3072]"; primals_191: "f32[768]"; primals_192: "f32[768, 768]"; primals_193: "f32[768]"; primals_194: "f32[3072, 768]"; primals_195: "f32[3072]"; primals_196: "f32[768, 3072]"; primals_197: "f32[768]"; primals_198: "f32[1000, 768]"; primals_199: "f32[1000]"; primals_200: "f32[768]"; primals_201: "i64[197, 197]"; primals_202: "f32[768]"; primals_203: "i64[197, 197]"; primals_204: "f32[768]"; primals_205: "i64[197, 197]"; primals_206: "f32[768]"; primals_207: "i64[197, 197]"; primals_208: "f32[768]"; primals_209: "i64[197, 197]"; primals_210: "f32[768]"; primals_211: "i64[197, 197]"; primals_212: "f32[768]"; primals_213: "i64[197, 197]"; primals_214: "f32[768]"; primals_215: "i64[197, 197]"; primals_216: "f32[768]"; primals_217: "i64[197, 197]"; primals_218: "f32[768]"; primals_219: "i64[197, 197]"; primals_220: "f32[768]"; primals_221: "i64[197, 197]"; primals_222: "f32[768]"; primals_223: "i64[197, 197]"; primals_224: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(primals_224, primals_124, primals_125, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 196]" = torch.ops.aten.view.default(convolution, [8, 768, 196]);  convolution = None
    permute: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:405, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    expand: "f32[8, 1, 768]" = torch.ops.aten.expand.default(primals_1, [8, -1, -1]);  primals_1 = None
    cat: "f32[8, 197, 768]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:408, code: x = self.pos_drop(x)
    clone: "f32[8, 197, 768]" = torch.ops.aten.clone.default(cat);  cat = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 197, 1]" = var_mean[0]
    getitem_1: "f32[8, 197, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(clone, getitem_1)
    mul: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul, primals_3);  mul = None
    add_1: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_1: "f32[2304]" = torch.ops.aten.cat.default([primals_5, primals_200, primals_6]);  primals_5 = primals_200 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_1: "f32[1576, 768]" = torch.ops.aten.view.default(add_1, [1576, 768]);  add_1 = None
    permute_1: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_1, view_1, permute_1);  cat_1 = None
    view_2: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm, [8, 197, 2304]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_3: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_2, [8, 197, 3, 12, -1]);  view_2 = None
    permute_2: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
    getitem_2: "f32[8, 12, 197, 64]" = unbind[0]
    getitem_3: "f32[8, 12, 197, 64]" = unbind[1]
    getitem_4: "f32[8, 12, 197, 64]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_4: "i64[38809]" = torch.ops.aten.view.default(primals_201, [-1]);  primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_8, [view_4]);  primals_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_5: "f32[197, 197, 12]" = torch.ops.aten.view.default(index, [197, 197, -1]);  index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_3: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_5, [2, 0, 1]);  view_5 = None
    clone_1: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_3, memory_format = torch.contiguous_format);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_1, 0);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_2: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_2, 0.3535533905932738);  getitem_2 = None
    permute_4: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_3, [0, 1, 3, 2]);  getitem_3 = None
    mul_3: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_4, 0.3535533905932738);  permute_4 = None
    expand_1: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_2, [8, 12, 197, 64]);  mul_2 = None
    clone_2: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_6: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_2, [96, 197, 64]);  clone_2 = None
    expand_2: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_3, [8, 12, 64, 197]);  mul_3 = None
    clone_3: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_7: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_3, [96, 64, 197]);  clone_3 = None
    bmm: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_6, view_7)
    view_8: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm, [8, 12, 197, 197]);  bmm = None
    add_2: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_8, unsqueeze);  view_8 = unsqueeze = None
    amax: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_2, [-1], True)
    sub_1: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_2, amax);  add_2 = amax = None
    exp: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div)
    expand_3: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div, [8, 12, 197, 197]);  div = None
    view_9: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_3, [96, 197, 197]);  expand_3 = None
    expand_4: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_4, [8, 12, 197, 64]);  getitem_4 = None
    clone_4: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_10: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_4, [96, 197, 64]);  clone_4 = None
    bmm_1: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_1, [8, 12, 197, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_5: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
    clone_5: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_12: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_5, [8, 197, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_13: "f32[1576, 768]" = torch.ops.aten.view.default(view_12, [1576, 768]);  view_12 = None
    permute_6: "f32[768, 768]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    addmm_1: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_127, view_13, permute_6);  primals_127 = None
    view_14: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_1, [8, 197, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_6: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_14);  view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_4: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_2, clone_6)
    add_3: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(clone, mul_4);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_5: "f32[8, 197, 1]" = var_mean_1[0]
    getitem_6: "f32[8, 197, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_5, 1e-06);  getitem_5 = None
    rsqrt_1: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_6)
    mul_5: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_6: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_5, primals_10);  mul_5 = None
    add_5: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_6, primals_11);  mul_6 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_15: "f32[1576, 768]" = torch.ops.aten.view.default(add_5, [1576, 768]);  add_5 = None
    permute_7: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_2: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_129, view_15, permute_7);  primals_129 = None
    view_16: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_2, [8, 197, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_7: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.5)
    mul_8: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.7071067811865476)
    erf: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_6: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_9: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_6);  mul_7 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_7: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_9);  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_17: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_7, [1576, 3072]);  clone_7 = None
    permute_8: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_3: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_131, view_17, permute_8);  primals_131 = None
    view_18: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_3, [8, 197, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_8: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_18);  view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_10: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_9, clone_8)
    add_7: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_3, mul_10);  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_7: "f32[8, 197, 1]" = var_mean_2[0]
    getitem_8: "f32[8, 197, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-06);  getitem_7 = None
    rsqrt_2: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_8)
    mul_11: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_12: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_11, primals_13);  mul_11 = None
    add_9: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_12, primals_14);  mul_12 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_2: "f32[2304]" = torch.ops.aten.cat.default([primals_15, primals_202, primals_16]);  primals_15 = primals_202 = primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_19: "f32[1576, 768]" = torch.ops.aten.view.default(add_9, [1576, 768]);  add_9 = None
    permute_9: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_4: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_2, view_19, permute_9);  cat_2 = None
    view_20: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_4, [8, 197, 2304]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_21: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_20, [8, 197, 3, 12, -1]);  view_20 = None
    permute_10: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_21, [2, 0, 3, 1, 4]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_1 = torch.ops.aten.unbind.int(permute_10);  permute_10 = None
    getitem_9: "f32[8, 12, 197, 64]" = unbind_1[0]
    getitem_10: "f32[8, 12, 197, 64]" = unbind_1[1]
    getitem_11: "f32[8, 12, 197, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_22: "i64[38809]" = torch.ops.aten.view.default(primals_203, [-1]);  primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_1: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_18, [view_22]);  primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_23: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_1, [197, 197, -1]);  index_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_11: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_23, [2, 0, 1]);  view_23 = None
    clone_9: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_11, memory_format = torch.contiguous_format);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_1: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_9, 0);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_13: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_9, 0.3535533905932738);  getitem_9 = None
    permute_12: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_10, [0, 1, 3, 2]);  getitem_10 = None
    mul_14: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_12, 0.3535533905932738);  permute_12 = None
    expand_5: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_13, [8, 12, 197, 64]);  mul_13 = None
    clone_10: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_24: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_10, [96, 197, 64]);  clone_10 = None
    expand_6: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_14, [8, 12, 64, 197]);  mul_14 = None
    clone_11: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_25: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_11, [96, 64, 197]);  clone_11 = None
    bmm_2: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_24, view_25)
    view_26: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_2, [8, 12, 197, 197]);  bmm_2 = None
    add_10: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_26, unsqueeze_1);  view_26 = unsqueeze_1 = None
    amax_1: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_10, [-1], True)
    sub_4: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_10, amax_1);  add_10 = amax_1 = None
    exp_1: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_1)
    expand_7: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_1, [8, 12, 197, 197]);  div_1 = None
    view_27: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_7, [96, 197, 197]);  expand_7 = None
    expand_8: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_11, [8, 12, 197, 64]);  getitem_11 = None
    clone_12: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_28: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_12, [96, 197, 64]);  clone_12 = None
    bmm_3: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_27, view_28)
    view_29: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_3, [8, 12, 197, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_13: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_13: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    view_30: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_13, [8, 197, 768]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_31: "f32[1576, 768]" = torch.ops.aten.view.default(view_30, [1576, 768]);  view_30 = None
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_5: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_133, view_31, permute_14);  primals_133 = None
    view_32: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_5, [8, 197, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_14: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_32);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_15: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_12, clone_14)
    add_11: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_7, mul_15);  mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 197, 1]" = var_mean_3[0]
    getitem_13: "f32[8, 197, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_3: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_13)
    mul_16: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_17: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_20);  mul_16 = None
    add_13: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_21);  mul_17 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_33: "f32[1576, 768]" = torch.ops.aten.view.default(add_13, [1576, 768]);  add_13 = None
    permute_15: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_6: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_135, view_33, permute_15);  primals_135 = None
    view_34: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_6, [8, 197, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
    mul_19: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.7071067811865476)
    erf_1: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_14: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_20: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_14);  mul_18 = add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_15: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_35: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_15, [1576, 3072]);  clone_15 = None
    permute_16: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_7: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_137, view_35, permute_16);  primals_137 = None
    view_36: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_7, [8, 197, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_16: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_21: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_19, clone_16)
    add_15: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_11, mul_21);  mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 197, 1]" = var_mean_4[0]
    getitem_15: "f32[8, 197, 1]" = var_mean_4[1];  var_mean_4 = None
    add_16: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_4: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_6: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_15)
    mul_22: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_23: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_22, primals_23);  mul_22 = None
    add_17: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_23, primals_24);  mul_23 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_3: "f32[2304]" = torch.ops.aten.cat.default([primals_25, primals_204, primals_26]);  primals_25 = primals_204 = primals_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_37: "f32[1576, 768]" = torch.ops.aten.view.default(add_17, [1576, 768]);  add_17 = None
    permute_17: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm_8: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_3, view_37, permute_17);  cat_3 = None
    view_38: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_8, [8, 197, 2304]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_39: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_38, [8, 197, 3, 12, -1]);  view_38 = None
    permute_18: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_39, [2, 0, 3, 1, 4]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_2 = torch.ops.aten.unbind.int(permute_18);  permute_18 = None
    getitem_16: "f32[8, 12, 197, 64]" = unbind_2[0]
    getitem_17: "f32[8, 12, 197, 64]" = unbind_2[1]
    getitem_18: "f32[8, 12, 197, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_40: "i64[38809]" = torch.ops.aten.view.default(primals_205, [-1]);  primals_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_2: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_28, [view_40]);  primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_41: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_2, [197, 197, -1]);  index_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_19: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_41, [2, 0, 1]);  view_41 = None
    clone_17: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_19, memory_format = torch.contiguous_format);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_2: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_17, 0);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_24: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_16, 0.3535533905932738);  getitem_16 = None
    permute_20: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_17, [0, 1, 3, 2]);  getitem_17 = None
    mul_25: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_20, 0.3535533905932738);  permute_20 = None
    expand_9: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_24, [8, 12, 197, 64]);  mul_24 = None
    clone_18: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_42: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_18, [96, 197, 64]);  clone_18 = None
    expand_10: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_25, [8, 12, 64, 197]);  mul_25 = None
    clone_19: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_43: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_19, [96, 64, 197]);  clone_19 = None
    bmm_4: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_42, view_43)
    view_44: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_4, [8, 12, 197, 197]);  bmm_4 = None
    add_18: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_44, unsqueeze_2);  view_44 = unsqueeze_2 = None
    amax_2: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_18, [-1], True)
    sub_7: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_18, amax_2);  add_18 = amax_2 = None
    exp_2: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_2)
    expand_11: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_2, [8, 12, 197, 197]);  div_2 = None
    view_45: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_11, [96, 197, 197]);  expand_11 = None
    expand_12: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_18, [8, 12, 197, 64]);  getitem_18 = None
    clone_20: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_46: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_20, [96, 197, 64]);  clone_20 = None
    bmm_5: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_45, view_46)
    view_47: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_5, [8, 12, 197, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_21: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
    clone_21: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    view_48: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_21, [8, 197, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_49: "f32[1576, 768]" = torch.ops.aten.view.default(view_48, [1576, 768]);  view_48 = None
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_9: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_139, view_49, permute_22);  primals_139 = None
    view_50: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_9, [8, 197, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_22: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_50);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_26: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_22, clone_22)
    add_19: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_15, mul_26);  mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_19: "f32[8, 197, 1]" = var_mean_5[0]
    getitem_20: "f32[8, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    add_20: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_19, 1e-06);  getitem_19 = None
    rsqrt_5: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_8: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_20)
    mul_27: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_28: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_27, primals_30);  mul_27 = None
    add_21: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_28, primals_31);  mul_28 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_51: "f32[1576, 768]" = torch.ops.aten.view.default(add_21, [1576, 768]);  add_21 = None
    permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_10: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_141, view_51, permute_23);  primals_141 = None
    view_52: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 197, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_29: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_52, 0.5)
    mul_30: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_52, 0.7071067811865476)
    erf_2: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_30);  mul_30 = None
    add_22: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_31: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_29, add_22);  mul_29 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_31);  mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_53: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_23, [1576, 3072]);  clone_23 = None
    permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    addmm_11: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_143, view_53, permute_24);  primals_143 = None
    view_54: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_11, [8, 197, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_54);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_32: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_29, clone_24)
    add_23: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_19, mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_21: "f32[8, 197, 1]" = var_mean_6[0]
    getitem_22: "f32[8, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    add_24: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_21, 1e-06);  getitem_21 = None
    rsqrt_6: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_9: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_22)
    mul_33: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_34: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_33, primals_33);  mul_33 = None
    add_25: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_34);  mul_34 = primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_4: "f32[2304]" = torch.ops.aten.cat.default([primals_35, primals_206, primals_36]);  primals_35 = primals_206 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_55: "f32[1576, 768]" = torch.ops.aten.view.default(add_25, [1576, 768]);  add_25 = None
    permute_25: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_12: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_4, view_55, permute_25);  cat_4 = None
    view_56: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_12, [8, 197, 2304]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_57: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_56, [8, 197, 3, 12, -1]);  view_56 = None
    permute_26: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_57, [2, 0, 3, 1, 4]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_3 = torch.ops.aten.unbind.int(permute_26);  permute_26 = None
    getitem_23: "f32[8, 12, 197, 64]" = unbind_3[0]
    getitem_24: "f32[8, 12, 197, 64]" = unbind_3[1]
    getitem_25: "f32[8, 12, 197, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_58: "i64[38809]" = torch.ops.aten.view.default(primals_207, [-1]);  primals_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_3: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_38, [view_58]);  primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_59: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_3, [197, 197, -1]);  index_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_27: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_59, [2, 0, 1]);  view_59 = None
    clone_25: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_27, memory_format = torch.contiguous_format);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_3: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_25, 0);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_35: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_23, 0.3535533905932738);  getitem_23 = None
    permute_28: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_24, [0, 1, 3, 2]);  getitem_24 = None
    mul_36: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_28, 0.3535533905932738);  permute_28 = None
    expand_13: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_35, [8, 12, 197, 64]);  mul_35 = None
    clone_26: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_60: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_26, [96, 197, 64]);  clone_26 = None
    expand_14: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_36, [8, 12, 64, 197]);  mul_36 = None
    clone_27: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_61: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_27, [96, 64, 197]);  clone_27 = None
    bmm_6: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_60, view_61)
    view_62: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_6, [8, 12, 197, 197]);  bmm_6 = None
    add_26: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_62, unsqueeze_3);  view_62 = unsqueeze_3 = None
    amax_3: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_26, [-1], True)
    sub_10: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_26, amax_3);  add_26 = amax_3 = None
    exp_3: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_3)
    expand_15: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_3, [8, 12, 197, 197]);  div_3 = None
    view_63: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_15, [96, 197, 197]);  expand_15 = None
    expand_16: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_25, [8, 12, 197, 64]);  getitem_25 = None
    clone_28: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_64: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_28, [96, 197, 64]);  clone_28 = None
    bmm_7: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_63, view_64)
    view_65: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_7, [8, 12, 197, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_29: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_29: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_66: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_29, [8, 197, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_67: "f32[1576, 768]" = torch.ops.aten.view.default(view_66, [1576, 768]);  view_66 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_13: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_145, view_67, permute_30);  primals_145 = None
    view_68: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_13, [8, 197, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_30: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_37: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_32, clone_30)
    add_27: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_23, mul_37);  mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 197, 1]" = var_mean_7[0]
    getitem_27: "f32[8, 197, 1]" = var_mean_7[1];  var_mean_7 = None
    add_28: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_7: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_27)
    mul_38: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_39: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_38, primals_40);  mul_38 = None
    add_29: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_39, primals_41);  mul_39 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_69: "f32[1576, 768]" = torch.ops.aten.view.default(add_29, [1576, 768]);  add_29 = None
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_14: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_147, view_69, permute_31);  primals_147 = None
    view_70: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_14, [8, 197, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_40: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
    mul_41: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.7071067811865476)
    erf_3: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_30: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_42: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_40, add_30);  mul_40 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_31: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_42);  mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_71: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_31, [1576, 3072]);  clone_31 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_15: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_149, view_71, permute_32);  primals_149 = None
    view_72: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_15, [8, 197, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_43: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_39, clone_32)
    add_31: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_27, mul_43);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 197, 1]" = var_mean_8[0]
    getitem_29: "f32[8, 197, 1]" = var_mean_8[1];  var_mean_8 = None
    add_32: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_8: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_12: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_29)
    mul_44: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_45: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_43);  mul_44 = None
    add_33: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_44);  mul_45 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_5: "f32[2304]" = torch.ops.aten.cat.default([primals_45, primals_208, primals_46]);  primals_45 = primals_208 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_73: "f32[1576, 768]" = torch.ops.aten.view.default(add_33, [1576, 768]);  add_33 = None
    permute_33: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_16: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_5, view_73, permute_33);  cat_5 = None
    view_74: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_16, [8, 197, 2304]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_75: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_74, [8, 197, 3, 12, -1]);  view_74 = None
    permute_34: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_75, [2, 0, 3, 1, 4]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_4 = torch.ops.aten.unbind.int(permute_34);  permute_34 = None
    getitem_30: "f32[8, 12, 197, 64]" = unbind_4[0]
    getitem_31: "f32[8, 12, 197, 64]" = unbind_4[1]
    getitem_32: "f32[8, 12, 197, 64]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_76: "i64[38809]" = torch.ops.aten.view.default(primals_209, [-1]);  primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_4: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_48, [view_76]);  primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_77: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_4, [197, 197, -1]);  index_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_35: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_77, [2, 0, 1]);  view_77 = None
    clone_33: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_35, memory_format = torch.contiguous_format);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_4: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_33, 0);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_46: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_30, 0.3535533905932738);  getitem_30 = None
    permute_36: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_31, [0, 1, 3, 2]);  getitem_31 = None
    mul_47: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_36, 0.3535533905932738);  permute_36 = None
    expand_17: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_46, [8, 12, 197, 64]);  mul_46 = None
    clone_34: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_78: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_34, [96, 197, 64]);  clone_34 = None
    expand_18: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_47, [8, 12, 64, 197]);  mul_47 = None
    clone_35: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_79: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_35, [96, 64, 197]);  clone_35 = None
    bmm_8: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_8, [8, 12, 197, 197]);  bmm_8 = None
    add_34: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_80, unsqueeze_4);  view_80 = unsqueeze_4 = None
    amax_4: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_34, [-1], True)
    sub_13: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_34, amax_4);  add_34 = amax_4 = None
    exp_4: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_4)
    expand_19: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_4, [8, 12, 197, 197]);  div_4 = None
    view_81: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_19, [96, 197, 197]);  expand_19 = None
    expand_20: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_32, [8, 12, 197, 64]);  getitem_32 = None
    clone_36: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_82: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_36, [96, 197, 64]);  clone_36 = None
    bmm_9: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_81, view_82)
    view_83: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_9, [8, 12, 197, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_37: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
    clone_37: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_84: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_37, [8, 197, 768]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_85: "f32[1576, 768]" = torch.ops.aten.view.default(view_84, [1576, 768]);  view_84 = None
    permute_38: "f32[768, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_17: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_151, view_85, permute_38);  primals_151 = None
    view_86: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_17, [8, 197, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_38: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_86);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_48: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_42, clone_38)
    add_35: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_31, mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_33: "f32[8, 197, 1]" = var_mean_9[0]
    getitem_34: "f32[8, 197, 1]" = var_mean_9[1];  var_mean_9 = None
    add_36: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
    rsqrt_9: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_14: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_34)
    mul_49: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_50: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_50);  mul_49 = None
    add_37: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_51);  mul_50 = primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_87: "f32[1576, 768]" = torch.ops.aten.view.default(add_37, [1576, 768]);  add_37 = None
    permute_39: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_18: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_153, view_87, permute_39);  primals_153 = None
    view_88: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_18, [8, 197, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_51: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.5)
    mul_52: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476)
    erf_4: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
    add_38: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_53: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_51, add_38);  mul_51 = add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_39: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_53);  mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_89: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_39, [1576, 3072]);  clone_39 = None
    permute_40: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_19: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_155, view_89, permute_40);  primals_155 = None
    view_90: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_19, [8, 197, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_40: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_90);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_54: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_49, clone_40)
    add_39: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_35, mul_54);  mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_35: "f32[8, 197, 1]" = var_mean_10[0]
    getitem_36: "f32[8, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    add_40: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_35, 1e-06);  getitem_35 = None
    rsqrt_10: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_15: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_36)
    mul_55: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_56: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_55, primals_53);  mul_55 = None
    add_41: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_56, primals_54);  mul_56 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_6: "f32[2304]" = torch.ops.aten.cat.default([primals_55, primals_210, primals_56]);  primals_55 = primals_210 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_91: "f32[1576, 768]" = torch.ops.aten.view.default(add_41, [1576, 768]);  add_41 = None
    permute_41: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    addmm_20: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_6, view_91, permute_41);  cat_6 = None
    view_92: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_20, [8, 197, 2304]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_93: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_92, [8, 197, 3, 12, -1]);  view_92 = None
    permute_42: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_93, [2, 0, 3, 1, 4]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_5 = torch.ops.aten.unbind.int(permute_42);  permute_42 = None
    getitem_37: "f32[8, 12, 197, 64]" = unbind_5[0]
    getitem_38: "f32[8, 12, 197, 64]" = unbind_5[1]
    getitem_39: "f32[8, 12, 197, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_94: "i64[38809]" = torch.ops.aten.view.default(primals_211, [-1]);  primals_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_5: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_58, [view_94]);  primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_95: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_5, [197, 197, -1]);  index_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_43: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_95, [2, 0, 1]);  view_95 = None
    clone_41: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_43, memory_format = torch.contiguous_format);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_5: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_41, 0);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_57: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_37, 0.3535533905932738);  getitem_37 = None
    permute_44: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_38, [0, 1, 3, 2]);  getitem_38 = None
    mul_58: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_44, 0.3535533905932738);  permute_44 = None
    expand_21: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_57, [8, 12, 197, 64]);  mul_57 = None
    clone_42: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_96: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_42, [96, 197, 64]);  clone_42 = None
    expand_22: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_58, [8, 12, 64, 197]);  mul_58 = None
    clone_43: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_97: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_43, [96, 64, 197]);  clone_43 = None
    bmm_10: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_96, view_97)
    view_98: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_10, [8, 12, 197, 197]);  bmm_10 = None
    add_42: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_98, unsqueeze_5);  view_98 = unsqueeze_5 = None
    amax_5: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_42, [-1], True)
    sub_16: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_42, amax_5);  add_42 = amax_5 = None
    exp_5: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_5)
    expand_23: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_5, [8, 12, 197, 197]);  div_5 = None
    view_99: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_23, [96, 197, 197]);  expand_23 = None
    expand_24: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_39, [8, 12, 197, 64]);  getitem_39 = None
    clone_44: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_100: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_44, [96, 197, 64]);  clone_44 = None
    bmm_11: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_99, view_100)
    view_101: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_11, [8, 12, 197, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_45: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
    clone_45: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_102: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_45, [8, 197, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_103: "f32[1576, 768]" = torch.ops.aten.view.default(view_102, [1576, 768]);  view_102 = None
    permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_21: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_157, view_103, permute_46);  primals_157 = None
    view_104: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_21, [8, 197, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_46: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_59: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_52, clone_46)
    add_43: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_39, mul_59);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 197, 1]" = var_mean_11[0]
    getitem_41: "f32[8, 197, 1]" = var_mean_11[1];  var_mean_11 = None
    add_44: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_11: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_17: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_41)
    mul_60: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_61: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_60, primals_60);  mul_60 = None
    add_45: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_61);  mul_61 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_105: "f32[1576, 768]" = torch.ops.aten.view.default(add_45, [1576, 768]);  add_45 = None
    permute_47: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    addmm_22: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_159, view_105, permute_47);  primals_159 = None
    view_106: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 197, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_62: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
    mul_63: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476)
    erf_5: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_46: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_64: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_62, add_46);  mul_62 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_47: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_64);  mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_107: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_47, [1576, 3072]);  clone_47 = None
    permute_48: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_23: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_161, view_107, permute_48);  primals_161 = None
    view_108: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_23, [8, 197, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_65: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_59, clone_48)
    add_47: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_43, mul_65);  mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 197, 1]" = var_mean_12[0]
    getitem_43: "f32[8, 197, 1]" = var_mean_12[1];  var_mean_12 = None
    add_48: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_12: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_18: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_43)
    mul_66: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_67: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_63);  mul_66 = None
    add_49: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_64);  mul_67 = primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_7: "f32[2304]" = torch.ops.aten.cat.default([primals_65, primals_212, primals_66]);  primals_65 = primals_212 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_109: "f32[1576, 768]" = torch.ops.aten.view.default(add_49, [1576, 768]);  add_49 = None
    permute_49: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    addmm_24: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_7, view_109, permute_49);  cat_7 = None
    view_110: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_24, [8, 197, 2304]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_111: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_110, [8, 197, 3, 12, -1]);  view_110 = None
    permute_50: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_6 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
    getitem_44: "f32[8, 12, 197, 64]" = unbind_6[0]
    getitem_45: "f32[8, 12, 197, 64]" = unbind_6[1]
    getitem_46: "f32[8, 12, 197, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_112: "i64[38809]" = torch.ops.aten.view.default(primals_213, [-1]);  primals_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_6: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_68, [view_112]);  primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_113: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_6, [197, 197, -1]);  index_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_51: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_113, [2, 0, 1]);  view_113 = None
    clone_49: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_6: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_49, 0);  clone_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_68: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_44, 0.3535533905932738);  getitem_44 = None
    permute_52: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_45, [0, 1, 3, 2]);  getitem_45 = None
    mul_69: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_52, 0.3535533905932738);  permute_52 = None
    expand_25: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_68, [8, 12, 197, 64]);  mul_68 = None
    clone_50: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_114: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_50, [96, 197, 64]);  clone_50 = None
    expand_26: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_69, [8, 12, 64, 197]);  mul_69 = None
    clone_51: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_115: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_51, [96, 64, 197]);  clone_51 = None
    bmm_12: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_114, view_115)
    view_116: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_12, [8, 12, 197, 197]);  bmm_12 = None
    add_50: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_116, unsqueeze_6);  view_116 = unsqueeze_6 = None
    amax_6: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_50, [-1], True)
    sub_19: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_50, amax_6);  add_50 = amax_6 = None
    exp_6: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_6)
    expand_27: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_6, [8, 12, 197, 197]);  div_6 = None
    view_117: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_27, [96, 197, 197]);  expand_27 = None
    expand_28: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_46, [8, 12, 197, 64]);  getitem_46 = None
    clone_52: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_118: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_52, [96, 197, 64]);  clone_52 = None
    bmm_13: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_117, view_118)
    view_119: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_13, [8, 12, 197, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_53: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    clone_53: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
    view_120: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_53, [8, 197, 768]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_121: "f32[1576, 768]" = torch.ops.aten.view.default(view_120, [1576, 768]);  view_120 = None
    permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_25: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_163, view_121, permute_54);  primals_163 = None
    view_122: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_25, [8, 197, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_54: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_122);  view_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_70: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_62, clone_54)
    add_51: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_47, mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_47: "f32[8, 197, 1]" = var_mean_13[0]
    getitem_48: "f32[8, 197, 1]" = var_mean_13[1];  var_mean_13 = None
    add_52: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_47, 1e-06);  getitem_47 = None
    rsqrt_13: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_20: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_48)
    mul_71: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
    mul_72: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_71, primals_70);  mul_71 = None
    add_53: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_72, primals_71);  mul_72 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_123: "f32[1576, 768]" = torch.ops.aten.view.default(add_53, [1576, 768]);  add_53 = None
    permute_55: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
    addmm_26: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_165, view_123, permute_55);  primals_165 = None
    view_124: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_26, [8, 197, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_73: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, 0.5)
    mul_74: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, 0.7071067811865476)
    erf_6: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
    add_54: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_75: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_73, add_54);  mul_73 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_55: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_75);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_125: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_55, [1576, 3072]);  clone_55 = None
    permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_27: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_167, view_125, permute_56);  primals_167 = None
    view_126: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_27, [8, 197, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_56: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_126);  view_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_76: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_69, clone_56)
    add_55: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_51, mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_49: "f32[8, 197, 1]" = var_mean_14[0]
    getitem_50: "f32[8, 197, 1]" = var_mean_14[1];  var_mean_14 = None
    add_56: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_49, 1e-06);  getitem_49 = None
    rsqrt_14: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_21: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_50)
    mul_77: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
    mul_78: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_73);  mul_77 = None
    add_57: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_78, primals_74);  mul_78 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_8: "f32[2304]" = torch.ops.aten.cat.default([primals_75, primals_214, primals_76]);  primals_75 = primals_214 = primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_127: "f32[1576, 768]" = torch.ops.aten.view.default(add_57, [1576, 768]);  add_57 = None
    permute_57: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm_28: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_8, view_127, permute_57);  cat_8 = None
    view_128: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_28, [8, 197, 2304]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_129: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_128, [8, 197, 3, 12, -1]);  view_128 = None
    permute_58: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_129, [2, 0, 3, 1, 4]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_7 = torch.ops.aten.unbind.int(permute_58);  permute_58 = None
    getitem_51: "f32[8, 12, 197, 64]" = unbind_7[0]
    getitem_52: "f32[8, 12, 197, 64]" = unbind_7[1]
    getitem_53: "f32[8, 12, 197, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_130: "i64[38809]" = torch.ops.aten.view.default(primals_215, [-1]);  primals_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_7: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_78, [view_130]);  primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_131: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_7, [197, 197, -1]);  index_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_59: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_131, [2, 0, 1]);  view_131 = None
    clone_57: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_7: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_57, 0);  clone_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_79: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_51, 0.3535533905932738);  getitem_51 = None
    permute_60: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_52, [0, 1, 3, 2]);  getitem_52 = None
    mul_80: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_60, 0.3535533905932738);  permute_60 = None
    expand_29: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_79, [8, 12, 197, 64]);  mul_79 = None
    clone_58: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_132: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_58, [96, 197, 64]);  clone_58 = None
    expand_30: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_80, [8, 12, 64, 197]);  mul_80 = None
    clone_59: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_133: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_59, [96, 64, 197]);  clone_59 = None
    bmm_14: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_132, view_133)
    view_134: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_14, [8, 12, 197, 197]);  bmm_14 = None
    add_58: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_134, unsqueeze_7);  view_134 = unsqueeze_7 = None
    amax_7: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_58, [-1], True)
    sub_22: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_58, amax_7);  add_58 = amax_7 = None
    exp_7: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_7)
    expand_31: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_7, [8, 12, 197, 197]);  div_7 = None
    view_135: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_31, [96, 197, 197]);  expand_31 = None
    expand_32: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_53, [8, 12, 197, 64]);  getitem_53 = None
    clone_60: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_136: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_60, [96, 197, 64]);  clone_60 = None
    bmm_15: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_135, view_136)
    view_137: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_15, [8, 12, 197, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_61: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
    clone_61: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
    view_138: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_61, [8, 197, 768]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_139: "f32[1576, 768]" = torch.ops.aten.view.default(view_138, [1576, 768]);  view_138 = None
    permute_62: "f32[768, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_29: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_169, view_139, permute_62);  primals_169 = None
    view_140: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_29, [8, 197, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_62: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_81: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_72, clone_62)
    add_59: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_55, mul_81);  mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 197, 1]" = var_mean_15[0]
    getitem_55: "f32[8, 197, 1]" = var_mean_15[1];  var_mean_15 = None
    add_60: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_15: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_23: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_55)
    mul_82: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    mul_83: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_82, primals_80);  mul_82 = None
    add_61: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_83, primals_81);  mul_83 = primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_141: "f32[1576, 768]" = torch.ops.aten.view.default(add_61, [1576, 768]);  add_61 = None
    permute_63: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_30: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_171, view_141, permute_63);  primals_171 = None
    view_142: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_30, [8, 197, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_84: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
    mul_85: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.7071067811865476)
    erf_7: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_85);  mul_85 = None
    add_62: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_86: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_62);  mul_84 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_63: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_86);  mul_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_143: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_63, [1576, 3072]);  clone_63 = None
    permute_64: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_31: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_173, view_143, permute_64);  primals_173 = None
    view_144: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_31, [8, 197, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_64: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_87: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_79, clone_64)
    add_63: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_59, mul_87);  mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_56: "f32[8, 197, 1]" = var_mean_16[0]
    getitem_57: "f32[8, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    add_64: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_16: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_24: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_57)
    mul_88: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
    mul_89: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_88, primals_83);  mul_88 = None
    add_65: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_89, primals_84);  mul_89 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_9: "f32[2304]" = torch.ops.aten.cat.default([primals_85, primals_216, primals_86]);  primals_85 = primals_216 = primals_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_145: "f32[1576, 768]" = torch.ops.aten.view.default(add_65, [1576, 768]);  add_65 = None
    permute_65: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_32: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_9, view_145, permute_65);  cat_9 = None
    view_146: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_32, [8, 197, 2304]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_147: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_146, [8, 197, 3, 12, -1]);  view_146 = None
    permute_66: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_147, [2, 0, 3, 1, 4]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_8 = torch.ops.aten.unbind.int(permute_66);  permute_66 = None
    getitem_58: "f32[8, 12, 197, 64]" = unbind_8[0]
    getitem_59: "f32[8, 12, 197, 64]" = unbind_8[1]
    getitem_60: "f32[8, 12, 197, 64]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_148: "i64[38809]" = torch.ops.aten.view.default(primals_217, [-1]);  primals_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_8: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_88, [view_148]);  primals_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_149: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_8, [197, 197, -1]);  index_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_67: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_149, [2, 0, 1]);  view_149 = None
    clone_65: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_67, memory_format = torch.contiguous_format);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_8: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_65, 0);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_90: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_58, 0.3535533905932738);  getitem_58 = None
    permute_68: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_59, [0, 1, 3, 2]);  getitem_59 = None
    mul_91: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_68, 0.3535533905932738);  permute_68 = None
    expand_33: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_90, [8, 12, 197, 64]);  mul_90 = None
    clone_66: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_150: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_66, [96, 197, 64]);  clone_66 = None
    expand_34: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_91, [8, 12, 64, 197]);  mul_91 = None
    clone_67: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_151: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_67, [96, 64, 197]);  clone_67 = None
    bmm_16: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_150, view_151)
    view_152: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_16, [8, 12, 197, 197]);  bmm_16 = None
    add_66: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_152, unsqueeze_8);  view_152 = unsqueeze_8 = None
    amax_8: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_66, [-1], True)
    sub_25: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_66, amax_8);  add_66 = amax_8 = None
    exp_8: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_8)
    expand_35: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_8, [8, 12, 197, 197]);  div_8 = None
    view_153: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_35, [96, 197, 197]);  expand_35 = None
    expand_36: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_60, [8, 12, 197, 64]);  getitem_60 = None
    clone_68: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_154: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_68, [96, 197, 64]);  clone_68 = None
    bmm_17: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_153, view_154)
    view_155: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_17, [8, 12, 197, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_69: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
    clone_69: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    view_156: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_69, [8, 197, 768]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_157: "f32[1576, 768]" = torch.ops.aten.view.default(view_156, [1576, 768]);  view_156 = None
    permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    addmm_33: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_175, view_157, permute_70);  primals_175 = None
    view_158: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_33, [8, 197, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_70: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_158);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_92: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_82, clone_70)
    add_67: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_63, mul_92);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_61: "f32[8, 197, 1]" = var_mean_17[0]
    getitem_62: "f32[8, 197, 1]" = var_mean_17[1];  var_mean_17 = None
    add_68: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_61, 1e-06);  getitem_61 = None
    rsqrt_17: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_26: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_62)
    mul_93: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    mul_94: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_93, primals_90);  mul_93 = None
    add_69: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_94, primals_91);  mul_94 = primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_159: "f32[1576, 768]" = torch.ops.aten.view.default(add_69, [1576, 768]);  add_69 = None
    permute_71: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_34: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_177, view_159, permute_71);  primals_177 = None
    view_160: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 197, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_95: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_160, 0.5)
    mul_96: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_160, 0.7071067811865476)
    erf_8: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_70: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_97: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_95, add_70);  mul_95 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_71: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_97);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_161: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_71, [1576, 3072]);  clone_71 = None
    permute_72: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_35: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_179, view_161, permute_72);  primals_179 = None
    view_162: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_35, [8, 197, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_72: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_162);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_98: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_89, clone_72)
    add_71: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_67, mul_98);  mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_63: "f32[8, 197, 1]" = var_mean_18[0]
    getitem_64: "f32[8, 197, 1]" = var_mean_18[1];  var_mean_18 = None
    add_72: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-06);  getitem_63 = None
    rsqrt_18: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_27: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_64)
    mul_99: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_99, primals_93);  mul_99 = None
    add_73: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_100, primals_94);  mul_100 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_10: "f32[2304]" = torch.ops.aten.cat.default([primals_95, primals_218, primals_96]);  primals_95 = primals_218 = primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_163: "f32[1576, 768]" = torch.ops.aten.view.default(add_73, [1576, 768]);  add_73 = None
    permute_73: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_36: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_10, view_163, permute_73);  cat_10 = None
    view_164: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_36, [8, 197, 2304]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_165: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_164, [8, 197, 3, 12, -1]);  view_164 = None
    permute_74: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_165, [2, 0, 3, 1, 4]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_9 = torch.ops.aten.unbind.int(permute_74);  permute_74 = None
    getitem_65: "f32[8, 12, 197, 64]" = unbind_9[0]
    getitem_66: "f32[8, 12, 197, 64]" = unbind_9[1]
    getitem_67: "f32[8, 12, 197, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_166: "i64[38809]" = torch.ops.aten.view.default(primals_219, [-1]);  primals_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_9: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_98, [view_166]);  primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_167: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_9, [197, 197, -1]);  index_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_75: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_167, [2, 0, 1]);  view_167 = None
    clone_73: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_9: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_73, 0);  clone_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_101: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_65, 0.3535533905932738);  getitem_65 = None
    permute_76: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_66, [0, 1, 3, 2]);  getitem_66 = None
    mul_102: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_76, 0.3535533905932738);  permute_76 = None
    expand_37: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_101, [8, 12, 197, 64]);  mul_101 = None
    clone_74: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_168: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_74, [96, 197, 64]);  clone_74 = None
    expand_38: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_102, [8, 12, 64, 197]);  mul_102 = None
    clone_75: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_169: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_75, [96, 64, 197]);  clone_75 = None
    bmm_18: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_168, view_169)
    view_170: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_18, [8, 12, 197, 197]);  bmm_18 = None
    add_74: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_170, unsqueeze_9);  view_170 = unsqueeze_9 = None
    amax_9: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_74, [-1], True)
    sub_28: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_74, amax_9);  add_74 = amax_9 = None
    exp_9: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_9)
    expand_39: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_9, [8, 12, 197, 197]);  div_9 = None
    view_171: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_39, [96, 197, 197]);  expand_39 = None
    expand_40: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_67, [8, 12, 197, 64]);  getitem_67 = None
    clone_76: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_172: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_76, [96, 197, 64]);  clone_76 = None
    bmm_19: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_171, view_172)
    view_173: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_19, [8, 12, 197, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_77: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    clone_77: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_174: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_77, [8, 197, 768]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_175: "f32[1576, 768]" = torch.ops.aten.view.default(view_174, [1576, 768]);  view_174 = None
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    addmm_37: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_181, view_175, permute_78);  primals_181 = None
    view_176: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_37, [8, 197, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_78: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_176);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_103: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_92, clone_78)
    add_75: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_71, mul_103);  mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 197, 1]" = var_mean_19[0]
    getitem_69: "f32[8, 197, 1]" = var_mean_19[1];  var_mean_19 = None
    add_76: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_19: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_29: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_69)
    mul_104: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    mul_105: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_104, primals_100);  mul_104 = None
    add_77: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_105, primals_101);  mul_105 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_177: "f32[1576, 768]" = torch.ops.aten.view.default(add_77, [1576, 768]);  add_77 = None
    permute_79: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_38: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_183, view_177, permute_79);  primals_183 = None
    view_178: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_38, [8, 197, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_106: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
    mul_107: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476)
    erf_9: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_78: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_108: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_106, add_78);  mul_106 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_79: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_108);  mul_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_179: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_79, [1576, 3072]);  clone_79 = None
    permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_39: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_185, view_179, permute_80);  primals_185 = None
    view_180: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_39, [8, 197, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_80: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_180);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_99, clone_80)
    add_79: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_75, mul_109);  mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_70: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_71: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_80: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_30: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_71)
    mul_110: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
    mul_111: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_110, primals_103);  mul_110 = None
    add_81: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_111, primals_104);  mul_111 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_11: "f32[2304]" = torch.ops.aten.cat.default([primals_105, primals_220, primals_106]);  primals_105 = primals_220 = primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_181: "f32[1576, 768]" = torch.ops.aten.view.default(add_81, [1576, 768]);  add_81 = None
    permute_81: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_40: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_11, view_181, permute_81);  cat_11 = None
    view_182: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_40, [8, 197, 2304]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_183: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_182, [8, 197, 3, 12, -1]);  view_182 = None
    permute_82: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_183, [2, 0, 3, 1, 4]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_10 = torch.ops.aten.unbind.int(permute_82);  permute_82 = None
    getitem_72: "f32[8, 12, 197, 64]" = unbind_10[0]
    getitem_73: "f32[8, 12, 197, 64]" = unbind_10[1]
    getitem_74: "f32[8, 12, 197, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_184: "i64[38809]" = torch.ops.aten.view.default(primals_221, [-1]);  primals_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_10: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_108, [view_184]);  primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_185: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_10, [197, 197, -1]);  index_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_83: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_185, [2, 0, 1]);  view_185 = None
    clone_81: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_10: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_81, 0);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_112: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_72, 0.3535533905932738);  getitem_72 = None
    permute_84: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_73, [0, 1, 3, 2]);  getitem_73 = None
    mul_113: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_84, 0.3535533905932738);  permute_84 = None
    expand_41: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_112, [8, 12, 197, 64]);  mul_112 = None
    clone_82: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_186: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_82, [96, 197, 64]);  clone_82 = None
    expand_42: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_113, [8, 12, 64, 197]);  mul_113 = None
    clone_83: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_187: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_83, [96, 64, 197]);  clone_83 = None
    bmm_20: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_186, view_187)
    view_188: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_20, [8, 12, 197, 197]);  bmm_20 = None
    add_82: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_188, unsqueeze_10);  view_188 = unsqueeze_10 = None
    amax_10: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_82, [-1], True)
    sub_31: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_82, amax_10);  add_82 = amax_10 = None
    exp_10: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_10)
    expand_43: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_10, [8, 12, 197, 197]);  div_10 = None
    view_189: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_43, [96, 197, 197]);  expand_43 = None
    expand_44: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_74, [8, 12, 197, 64]);  getitem_74 = None
    clone_84: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_190: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_84, [96, 197, 64]);  clone_84 = None
    bmm_21: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_189, view_190)
    view_191: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_21, [8, 12, 197, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_85: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
    clone_85: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_192: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_85, [8, 197, 768]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_193: "f32[1576, 768]" = torch.ops.aten.view.default(view_192, [1576, 768]);  view_192 = None
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    addmm_41: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_187, view_193, permute_86);  primals_187 = None
    view_194: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_41, [8, 197, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_86: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_194);  view_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_114: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_102, clone_86)
    add_83: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_79, mul_114);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_75: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_76: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_84: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-06);  getitem_75 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_32: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_76)
    mul_115: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
    mul_116: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_115, primals_110);  mul_115 = None
    add_85: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_116, primals_111);  mul_116 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_195: "f32[1576, 768]" = torch.ops.aten.view.default(add_85, [1576, 768]);  add_85 = None
    permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_42: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_189, view_195, permute_87);  primals_189 = None
    view_196: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_42, [8, 197, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_117: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.5)
    mul_118: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.7071067811865476)
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_86: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_119: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_117, add_86);  mul_117 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_87: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_119);  mul_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_197: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_87, [1576, 3072]);  clone_87 = None
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_43: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_191, view_197, permute_88);  primals_191 = None
    view_198: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_43, [8, 197, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_88: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_198);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_120: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_109, clone_88)
    add_87: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_83, mul_120);  mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_77: "f32[8, 197, 1]" = var_mean_22[0]
    getitem_78: "f32[8, 197, 1]" = var_mean_22[1];  var_mean_22 = None
    add_88: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
    rsqrt_22: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_33: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_78)
    mul_121: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
    mul_122: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_121, primals_113);  mul_121 = None
    add_89: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_122, primals_114);  mul_122 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_12: "f32[2304]" = torch.ops.aten.cat.default([primals_115, primals_222, primals_116]);  primals_115 = primals_222 = primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_199: "f32[1576, 768]" = torch.ops.aten.view.default(add_89, [1576, 768]);  add_89 = None
    permute_89: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_44: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_12, view_199, permute_89);  cat_12 = None
    view_200: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_44, [8, 197, 2304]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_201: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_200, [8, 197, 3, 12, -1]);  view_200 = None
    permute_90: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_201, [2, 0, 3, 1, 4]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_11 = torch.ops.aten.unbind.int(permute_90);  permute_90 = None
    getitem_79: "f32[8, 12, 197, 64]" = unbind_11[0]
    getitem_80: "f32[8, 12, 197, 64]" = unbind_11[1]
    getitem_81: "f32[8, 12, 197, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_202: "i64[38809]" = torch.ops.aten.view.default(primals_223, [-1]);  primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_11: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_118, [view_202]);  primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_203: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_11, [197, 197, -1]);  index_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_91: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_203, [2, 0, 1]);  view_203 = None
    clone_89: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_11: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_89, 0);  clone_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    mul_123: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(getitem_79, 0.3535533905932738);  getitem_79 = None
    permute_92: "f32[8, 12, 64, 197]" = torch.ops.aten.permute.default(getitem_80, [0, 1, 3, 2]);  getitem_80 = None
    mul_124: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(permute_92, 0.3535533905932738);  permute_92 = None
    expand_45: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(mul_123, [8, 12, 197, 64]);  mul_123 = None
    clone_90: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_204: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_90, [96, 197, 64]);  clone_90 = None
    expand_46: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_124, [8, 12, 64, 197]);  mul_124 = None
    clone_91: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_205: "f32[96, 64, 197]" = torch.ops.aten.view.default(clone_91, [96, 64, 197]);  clone_91 = None
    bmm_22: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_204, view_205)
    view_206: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_22, [8, 12, 197, 197]);  bmm_22 = None
    add_90: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_206, unsqueeze_11);  view_206 = unsqueeze_11 = None
    amax_11: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_90, [-1], True)
    sub_34: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_90, amax_11);  add_90 = amax_11 = None
    exp_11: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_11)
    expand_47: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_11, [8, 12, 197, 197]);  div_11 = None
    view_207: "f32[96, 197, 197]" = torch.ops.aten.view.default(expand_47, [96, 197, 197]);  expand_47 = None
    expand_48: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_81, [8, 12, 197, 64]);  getitem_81 = None
    clone_92: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_208: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_92, [96, 197, 64]);  clone_92 = None
    bmm_23: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_207, view_208)
    view_209: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_23, [8, 12, 197, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_93: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
    clone_93: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    view_210: "f32[8, 197, 768]" = torch.ops.aten.view.default(clone_93, [8, 197, 768]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_211: "f32[1576, 768]" = torch.ops.aten.view.default(view_210, [1576, 768]);  view_210 = None
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_45: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_193, view_211, permute_94);  primals_193 = None
    view_212: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_45, [8, 197, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_94: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_212);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_125: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_112, clone_94)
    add_91: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_87, mul_125);  mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_82: "f32[8, 197, 1]" = var_mean_23[0]
    getitem_83: "f32[8, 197, 1]" = var_mean_23[1];  var_mean_23 = None
    add_92: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-06);  getitem_82 = None
    rsqrt_23: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_35: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_83)
    mul_126: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
    mul_127: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_126, primals_120);  mul_126 = None
    add_93: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_127, primals_121);  mul_127 = primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_213: "f32[1576, 768]" = torch.ops.aten.view.default(add_93, [1576, 768]);  add_93 = None
    permute_95: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_46: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_195, view_213, permute_95);  primals_195 = None
    view_214: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 197, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_128: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, 0.5)
    mul_129: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, 0.7071067811865476)
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
    add_94: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_130: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_128, add_94);  mul_128 = add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_95: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_130);  mul_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_215: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_95, [1576, 3072]);  clone_95 = None
    permute_96: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_47: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_197, view_215, permute_96);  primals_197 = None
    view_216: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_47, [8, 197, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_96: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_216);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_131: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_119, clone_96)
    add_95: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_91, mul_131);  mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:421, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    slice_1: "f32[8, 197, 768]" = torch.ops.aten.slice.Tensor(add_95, 0, 0, 9223372036854775807);  add_95 = None
    slice_2: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(slice_1, 1, 1, 9223372036854775807);  slice_1 = None
    mean: "f32[8, 768]" = torch.ops.aten.mean.dim(slice_2, [1]);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(mean, [1], correction = 0, keepdim = True)
    getitem_84: "f32[8, 1]" = var_mean_24[0]
    getitem_85: "f32[8, 1]" = var_mean_24[1];  var_mean_24 = None
    add_96: "f32[8, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
    rsqrt_24: "f32[8, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_36: "f32[8, 768]" = torch.ops.aten.sub.Tensor(mean, getitem_85)
    mul_132: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
    mul_133: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_132, primals_122);  mul_132 = None
    add_97: "f32[8, 768]" = torch.ops.aten.add.Tensor(mul_133, primals_123);  mul_133 = primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:423, code: x = self.head_drop(x)
    clone_97: "f32[8, 768]" = torch.ops.aten.clone.default(add_97);  add_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:424, code: return x if pre_logits else self.head(x)
    permute_97: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_48: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_199, clone_97, permute_97);  primals_199 = None
    permute_98: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_98);  permute_98 = None
    permute_99: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_99, clone_97);  permute_99 = clone_97 = None
    permute_100: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_13: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_217: "f32[1000]" = torch.ops.aten.view.default(sum_13, [1000]);  sum_13 = None
    permute_101: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_37: "f32[8, 768]" = torch.ops.aten.sub.Tensor(mean, getitem_85);  mean = getitem_85 = None
    mul_134: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
    mul_135: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mm, primals_122);  primals_122 = None
    mul_136: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_135, 768)
    sum_14: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [1], True)
    mul_137: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_135, mul_134);  mul_135 = None
    sum_15: "f32[8, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [1], True);  mul_137 = None
    mul_138: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_134, sum_15);  sum_15 = None
    sub_38: "f32[8, 768]" = torch.ops.aten.sub.Tensor(mul_136, sum_14);  mul_136 = sum_14 = None
    sub_39: "f32[8, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_138);  sub_38 = mul_138 = None
    div_12: "f32[8, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    mul_139: "f32[8, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_39);  div_12 = sub_39 = None
    mul_140: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mm, mul_134);  mul_134 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_140, [0]);  mul_140 = None
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mm, [0]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:421, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    unsqueeze_12: "f32[8, 1, 768]" = torch.ops.aten.unsqueeze.default(mul_139, 1);  mul_139 = None
    expand_49: "f32[8, 196, 768]" = torch.ops.aten.expand.default(unsqueeze_12, [8, 196, 768]);  unsqueeze_12 = None
    div_13: "f32[8, 196, 768]" = torch.ops.aten.div.Scalar(expand_49, 196);  expand_49 = None
    full: "f32[8, 197, 768]" = torch.ops.aten.full.default([8, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[8, 197, 768]" = torch.ops.aten.slice_scatter.default(full, div_13, 1, 1, 9223372036854775807);  full = div_13 = None
    full_1: "f32[8, 197, 768]" = torch.ops.aten.full.default([8, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[8, 197, 768]" = torch.ops.aten.slice_scatter.default(full_1, slice_scatter, 0, 0, 9223372036854775807);  full_1 = slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_141: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter_1, primals_119);  primals_119 = None
    mul_142: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter_1, clone_96);  clone_96 = None
    sum_18: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_142, [0, 1], True);  mul_142 = None
    view_218: "f32[768]" = torch.ops.aten.view.default(sum_18, [768]);  sum_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_219: "f32[1576, 768]" = torch.ops.aten.view.default(mul_141, [1576, 768]);  mul_141 = None
    permute_102: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_2: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_219, permute_102);  permute_102 = None
    permute_103: "f32[768, 1576]" = torch.ops.aten.permute.default(view_219, [1, 0])
    mm_3: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_103, view_215);  permute_103 = view_215 = None
    permute_104: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_19: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_219, [0], True);  view_219 = None
    view_220: "f32[768]" = torch.ops.aten.view.default(sum_19, [768]);  sum_19 = None
    permute_105: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_104, [1, 0]);  permute_104 = None
    view_221: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_2, [8, 197, 3072]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_143: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, 0.7071067811865476)
    erf_12: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_143);  mul_143 = None
    add_98: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_144: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_98, 0.5);  add_98 = None
    mul_145: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, view_214)
    mul_146: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_145, -0.5);  mul_145 = None
    exp_12: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_146);  mul_146 = None
    mul_147: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_148: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, mul_147);  view_214 = mul_147 = None
    add_99: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_144, mul_148);  mul_144 = mul_148 = None
    mul_149: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_221, add_99);  view_221 = add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_222: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_149, [1576, 3072]);  mul_149 = None
    permute_106: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    mm_4: "f32[1576, 768]" = torch.ops.aten.mm.default(view_222, permute_106);  permute_106 = None
    permute_107: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_222, [1, 0])
    mm_5: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_107, view_213);  permute_107 = view_213 = None
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_20: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_222, [0], True);  view_222 = None
    view_223: "f32[3072]" = torch.ops.aten.view.default(sum_20, [3072]);  sum_20 = None
    permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    view_224: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_4, [8, 197, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_40: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_83);  add_91 = getitem_83 = None
    mul_150: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_23);  sub_40 = None
    mul_151: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_224, primals_120);  primals_120 = None
    mul_152: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_151, 768)
    sum_21: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [2], True)
    mul_153: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_151, mul_150);  mul_151 = None
    sum_22: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_153, [2], True);  mul_153 = None
    mul_154: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_150, sum_22);  sum_22 = None
    sub_41: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_152, sum_21);  mul_152 = sum_21 = None
    sub_42: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_154);  sub_41 = mul_154 = None
    div_14: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    mul_155: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_42);  div_14 = sub_42 = None
    mul_156: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_224, mul_150);  mul_150 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_156, [0, 1]);  mul_156 = None
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_224, [0, 1]);  view_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_100: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(slice_scatter_1, mul_155);  slice_scatter_1 = mul_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_157: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_100, primals_112);  primals_112 = None
    mul_158: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_100, clone_94);  clone_94 = None
    sum_25: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_158, [0, 1], True);  mul_158 = None
    view_225: "f32[768]" = torch.ops.aten.view.default(sum_25, [768]);  sum_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_226: "f32[1576, 768]" = torch.ops.aten.view.default(mul_157, [1576, 768]);  mul_157 = None
    permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    mm_6: "f32[1576, 768]" = torch.ops.aten.mm.default(view_226, permute_110);  permute_110 = None
    permute_111: "f32[768, 1576]" = torch.ops.aten.permute.default(view_226, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_111, view_211);  permute_111 = view_211 = None
    permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_226, [0], True);  view_226 = None
    view_227: "f32[768]" = torch.ops.aten.view.default(sum_26, [768]);  sum_26 = None
    permute_113: "f32[768, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    view_228: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_6, [8, 197, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_229: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_228, [8, 197, 12, 64]);  view_228 = None
    permute_114: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_98: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_230: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_98, [96, 197, 64]);  clone_98 = None
    permute_115: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    bmm_24: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_115, view_230);  permute_115 = None
    permute_116: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    bmm_25: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_230, permute_116);  view_230 = permute_116 = None
    view_231: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_24, [8, 12, 197, 64]);  bmm_24 = None
    view_232: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_25, [8, 12, 197, 197]);  bmm_25 = None
    alias_12: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_159: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_232, alias_12);  view_232 = None
    sum_27: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [-1], True)
    mul_160: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_12, sum_27);  alias_12 = sum_27 = None
    sub_43: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_159, mul_160);  mul_159 = mul_160 = None
    sum_28: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_43, [0], True)
    view_233: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_43, [96, 197, 197]);  sub_43 = None
    permute_117: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
    bmm_26: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_117, view_233);  permute_117 = None
    permute_118: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    bmm_27: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_233, permute_118);  view_233 = permute_118 = None
    view_234: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_26, [8, 12, 64, 197]);  bmm_26 = None
    view_235: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_27, [8, 12, 197, 64]);  bmm_27 = None
    mul_161: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_234, 0.3535533905932738);  view_234 = None
    permute_119: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_161, [0, 1, 3, 2]);  mul_161 = None
    mul_162: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_235, 0.3535533905932738);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_28, 0);  sum_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_120: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze, [1, 2, 0]);  squeeze = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_236: "f32[38809, 12]" = torch.ops.aten.view.default(permute_120, [38809, 12]);  permute_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_2: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put: "f32[732, 12]" = torch.ops.aten.index_put.default(full_2, [view_202], view_236, True);  full_2 = view_202 = view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_13: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_162, permute_119, view_231]);  mul_162 = permute_119 = view_231 = None
    view_237: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_13, [3, 8, 12, 197, 64]);  cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_121: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_237, [1, 3, 0, 2, 4]);  view_237 = None
    clone_99: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_238: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_99, [8, 197, 2304]);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_239: "f32[1576, 2304]" = torch.ops.aten.view.default(view_238, [1576, 2304]);  view_238 = None
    permute_122: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    mm_8: "f32[1576, 768]" = torch.ops.aten.mm.default(view_239, permute_122);  permute_122 = None
    permute_123: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_239, [1, 0])
    mm_9: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_123, view_199);  permute_123 = view_199 = None
    permute_124: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_29: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_239, [0], True);  view_239 = None
    view_240: "f32[2304]" = torch.ops.aten.view.default(sum_29, [2304]);  sum_29 = None
    permute_125: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    view_241: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_8, [8, 197, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_3: "f32[768]" = torch.ops.aten.slice.Tensor(view_240, 0, 0, 768)
    slice_5: "f32[768]" = torch.ops.aten.slice.Tensor(view_240, 0, 1536, 2304);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_44: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_78);  add_87 = getitem_78 = None
    mul_163: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_22);  sub_44 = None
    mul_164: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_241, primals_113);  primals_113 = None
    mul_165: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_164, 768)
    sum_30: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
    mul_166: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_164, mul_163);  mul_164 = None
    sum_31: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
    mul_167: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_163, sum_31);  sum_31 = None
    sub_45: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_165, sum_30);  mul_165 = sum_30 = None
    sub_46: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_45, mul_167);  sub_45 = mul_167 = None
    div_15: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    mul_168: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_46);  div_15 = sub_46 = None
    mul_169: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_241, mul_163);  mul_163 = None
    sum_32: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
    sum_33: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_241, [0, 1]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_101: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_100, mul_168);  add_100 = mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_170: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_101, primals_109);  primals_109 = None
    mul_171: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_101, clone_88);  clone_88 = None
    sum_34: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_171, [0, 1], True);  mul_171 = None
    view_242: "f32[768]" = torch.ops.aten.view.default(sum_34, [768]);  sum_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_243: "f32[1576, 768]" = torch.ops.aten.view.default(mul_170, [1576, 768]);  mul_170 = None
    permute_126: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_10: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_243, permute_126);  permute_126 = None
    permute_127: "f32[768, 1576]" = torch.ops.aten.permute.default(view_243, [1, 0])
    mm_11: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_127, view_197);  permute_127 = view_197 = None
    permute_128: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_35: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
    view_244: "f32[768]" = torch.ops.aten.view.default(sum_35, [768]);  sum_35 = None
    permute_129: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    view_245: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_10, [8, 197, 3072]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_172: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.7071067811865476)
    erf_13: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_172);  mul_172 = None
    add_102: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_173: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_102, 0.5);  add_102 = None
    mul_174: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_196, view_196)
    mul_175: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_174, -0.5);  mul_174 = None
    exp_13: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_175);  mul_175 = None
    mul_176: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_177: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_196, mul_176);  view_196 = mul_176 = None
    add_103: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_173, mul_177);  mul_173 = mul_177 = None
    mul_178: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_245, add_103);  view_245 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_246: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_178, [1576, 3072]);  mul_178 = None
    permute_130: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_12: "f32[1576, 768]" = torch.ops.aten.mm.default(view_246, permute_130);  permute_130 = None
    permute_131: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_246, [1, 0])
    mm_13: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_131, view_195);  permute_131 = view_195 = None
    permute_132: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_36: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
    view_247: "f32[3072]" = torch.ops.aten.view.default(sum_36, [3072]);  sum_36 = None
    permute_133: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    view_248: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_12, [8, 197, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_47: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_76);  add_83 = getitem_76 = None
    mul_179: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_21);  sub_47 = None
    mul_180: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_248, primals_110);  primals_110 = None
    mul_181: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_180, 768)
    sum_37: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_180, [2], True)
    mul_182: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_180, mul_179);  mul_180 = None
    sum_38: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_182, [2], True);  mul_182 = None
    mul_183: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_179, sum_38);  sum_38 = None
    sub_48: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_181, sum_37);  mul_181 = sum_37 = None
    sub_49: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_48, mul_183);  sub_48 = mul_183 = None
    div_16: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    mul_184: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_49);  div_16 = sub_49 = None
    mul_185: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_248, mul_179);  mul_179 = None
    sum_39: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_185, [0, 1]);  mul_185 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_248, [0, 1]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_104: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_101, mul_184);  add_101 = mul_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_186: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_104, primals_102);  primals_102 = None
    mul_187: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_104, clone_86);  clone_86 = None
    sum_41: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_187, [0, 1], True);  mul_187 = None
    view_249: "f32[768]" = torch.ops.aten.view.default(sum_41, [768]);  sum_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_250: "f32[1576, 768]" = torch.ops.aten.view.default(mul_186, [1576, 768]);  mul_186 = None
    permute_134: "f32[768, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_14: "f32[1576, 768]" = torch.ops.aten.mm.default(view_250, permute_134);  permute_134 = None
    permute_135: "f32[768, 1576]" = torch.ops.aten.permute.default(view_250, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_135, view_193);  permute_135 = view_193 = None
    permute_136: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_250, [0], True);  view_250 = None
    view_251: "f32[768]" = torch.ops.aten.view.default(sum_42, [768]);  sum_42 = None
    permute_137: "f32[768, 768]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    view_252: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_14, [8, 197, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_253: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_252, [8, 197, 12, 64]);  view_252 = None
    permute_138: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_253, [0, 2, 1, 3]);  view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_100: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_138, memory_format = torch.contiguous_format);  permute_138 = None
    view_254: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_100, [96, 197, 64]);  clone_100 = None
    permute_139: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    bmm_28: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_139, view_254);  permute_139 = None
    permute_140: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    bmm_29: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_254, permute_140);  view_254 = permute_140 = None
    view_255: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_28, [8, 12, 197, 64]);  bmm_28 = None
    view_256: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_29, [8, 12, 197, 197]);  bmm_29 = None
    alias_13: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_188: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_256, alias_13);  view_256 = None
    sum_43: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_188, [-1], True)
    mul_189: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_13, sum_43);  alias_13 = sum_43 = None
    sub_50: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_188, mul_189);  mul_188 = mul_189 = None
    sum_44: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_50, [0], True)
    view_257: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_50, [96, 197, 197]);  sub_50 = None
    permute_141: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
    bmm_30: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_141, view_257);  permute_141 = None
    permute_142: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    bmm_31: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_257, permute_142);  view_257 = permute_142 = None
    view_258: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_30, [8, 12, 64, 197]);  bmm_30 = None
    view_259: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_31, [8, 12, 197, 64]);  bmm_31 = None
    mul_190: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_258, 0.3535533905932738);  view_258 = None
    permute_143: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_190, [0, 1, 3, 2]);  mul_190 = None
    mul_191: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_259, 0.3535533905932738);  view_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_1: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_44, 0);  sum_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_144: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_260: "f32[38809, 12]" = torch.ops.aten.view.default(permute_144, [38809, 12]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_3: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_1: "f32[732, 12]" = torch.ops.aten.index_put.default(full_3, [view_184], view_260, True);  full_3 = view_184 = view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_14: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_191, permute_143, view_255]);  mul_191 = permute_143 = view_255 = None
    view_261: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_14, [3, 8, 12, 197, 64]);  cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_145: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_261, [1, 3, 0, 2, 4]);  view_261 = None
    clone_101: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    view_262: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_101, [8, 197, 2304]);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_263: "f32[1576, 2304]" = torch.ops.aten.view.default(view_262, [1576, 2304]);  view_262 = None
    permute_146: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    mm_16: "f32[1576, 768]" = torch.ops.aten.mm.default(view_263, permute_146);  permute_146 = None
    permute_147: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_263, [1, 0])
    mm_17: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_147, view_181);  permute_147 = view_181 = None
    permute_148: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_45: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
    view_264: "f32[2304]" = torch.ops.aten.view.default(sum_45, [2304]);  sum_45 = None
    permute_149: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_265: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_16, [8, 197, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_6: "f32[768]" = torch.ops.aten.slice.Tensor(view_264, 0, 0, 768)
    slice_8: "f32[768]" = torch.ops.aten.slice.Tensor(view_264, 0, 1536, 2304);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_51: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_71);  add_79 = getitem_71 = None
    mul_192: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_20);  sub_51 = None
    mul_193: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_265, primals_103);  primals_103 = None
    mul_194: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_193, 768)
    sum_46: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [2], True)
    mul_195: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_193, mul_192);  mul_193 = None
    sum_47: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True);  mul_195 = None
    mul_196: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_192, sum_47);  sum_47 = None
    sub_52: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_194, sum_46);  mul_194 = sum_46 = None
    sub_53: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_52, mul_196);  sub_52 = mul_196 = None
    div_17: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    mul_197: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_53);  div_17 = sub_53 = None
    mul_198: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_265, mul_192);  mul_192 = None
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1]);  mul_198 = None
    sum_49: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_265, [0, 1]);  view_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_105: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_104, mul_197);  add_104 = mul_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_199: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_105, primals_99);  primals_99 = None
    mul_200: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_105, clone_80);  clone_80 = None
    sum_50: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_200, [0, 1], True);  mul_200 = None
    view_266: "f32[768]" = torch.ops.aten.view.default(sum_50, [768]);  sum_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_267: "f32[1576, 768]" = torch.ops.aten.view.default(mul_199, [1576, 768]);  mul_199 = None
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    mm_18: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_267, permute_150);  permute_150 = None
    permute_151: "f32[768, 1576]" = torch.ops.aten.permute.default(view_267, [1, 0])
    mm_19: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_151, view_179);  permute_151 = view_179 = None
    permute_152: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_267, [0], True);  view_267 = None
    view_268: "f32[768]" = torch.ops.aten.view.default(sum_51, [768]);  sum_51 = None
    permute_153: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_269: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_18, [8, 197, 3072]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_201: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476)
    erf_14: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_201);  mul_201 = None
    add_106: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_202: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
    mul_203: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_178, view_178)
    mul_204: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_203, -0.5);  mul_203 = None
    exp_14: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_204);  mul_204 = None
    mul_205: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_206: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_178, mul_205);  view_178 = mul_205 = None
    add_107: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_202, mul_206);  mul_202 = mul_206 = None
    mul_207: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_269, add_107);  view_269 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_270: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_207, [1576, 3072]);  mul_207 = None
    permute_154: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    mm_20: "f32[1576, 768]" = torch.ops.aten.mm.default(view_270, permute_154);  permute_154 = None
    permute_155: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_270, [1, 0])
    mm_21: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_155, view_177);  permute_155 = view_177 = None
    permute_156: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_52: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True);  view_270 = None
    view_271: "f32[3072]" = torch.ops.aten.view.default(sum_52, [3072]);  sum_52 = None
    permute_157: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    view_272: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_20, [8, 197, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_54: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_69);  add_75 = getitem_69 = None
    mul_208: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_19);  sub_54 = None
    mul_209: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_272, primals_100);  primals_100 = None
    mul_210: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_209, 768)
    sum_53: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_209, [2], True)
    mul_211: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_209, mul_208);  mul_209 = None
    sum_54: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_211, [2], True);  mul_211 = None
    mul_212: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_208, sum_54);  sum_54 = None
    sub_55: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_210, sum_53);  mul_210 = sum_53 = None
    sub_56: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_55, mul_212);  sub_55 = mul_212 = None
    div_18: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    mul_213: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_56);  div_18 = sub_56 = None
    mul_214: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_272, mul_208);  mul_208 = None
    sum_55: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_214, [0, 1]);  mul_214 = None
    sum_56: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_272, [0, 1]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_108: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_105, mul_213);  add_105 = mul_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_215: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_108, primals_92);  primals_92 = None
    mul_216: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_108, clone_78);  clone_78 = None
    sum_57: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1], True);  mul_216 = None
    view_273: "f32[768]" = torch.ops.aten.view.default(sum_57, [768]);  sum_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_274: "f32[1576, 768]" = torch.ops.aten.view.default(mul_215, [1576, 768]);  mul_215 = None
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    mm_22: "f32[1576, 768]" = torch.ops.aten.mm.default(view_274, permute_158);  permute_158 = None
    permute_159: "f32[768, 1576]" = torch.ops.aten.permute.default(view_274, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_159, view_175);  permute_159 = view_175 = None
    permute_160: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_274, [0], True);  view_274 = None
    view_275: "f32[768]" = torch.ops.aten.view.default(sum_58, [768]);  sum_58 = None
    permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    view_276: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_22, [8, 197, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_277: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_276, [8, 197, 12, 64]);  view_276 = None
    permute_162: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_102: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_162, memory_format = torch.contiguous_format);  permute_162 = None
    view_278: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_102, [96, 197, 64]);  clone_102 = None
    permute_163: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    bmm_32: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_163, view_278);  permute_163 = None
    permute_164: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    bmm_33: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_278, permute_164);  view_278 = permute_164 = None
    view_279: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_32, [8, 12, 197, 64]);  bmm_32 = None
    view_280: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_33, [8, 12, 197, 197]);  bmm_33 = None
    alias_14: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_217: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_280, alias_14);  view_280 = None
    sum_59: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [-1], True)
    mul_218: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_14, sum_59);  alias_14 = sum_59 = None
    sub_57: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_217, mul_218);  mul_217 = mul_218 = None
    sum_60: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_57, [0], True)
    view_281: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_57, [96, 197, 197]);  sub_57 = None
    permute_165: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    bmm_34: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_165, view_281);  permute_165 = None
    permute_166: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    bmm_35: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_281, permute_166);  view_281 = permute_166 = None
    view_282: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_34, [8, 12, 64, 197]);  bmm_34 = None
    view_283: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_35, [8, 12, 197, 64]);  bmm_35 = None
    mul_219: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_282, 0.3535533905932738);  view_282 = None
    permute_167: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_219, [0, 1, 3, 2]);  mul_219 = None
    mul_220: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_283, 0.3535533905932738);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_2: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_60, 0);  sum_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_168: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_284: "f32[38809, 12]" = torch.ops.aten.view.default(permute_168, [38809, 12]);  permute_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_4: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_2: "f32[732, 12]" = torch.ops.aten.index_put.default(full_4, [view_166], view_284, True);  full_4 = view_166 = view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_15: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_220, permute_167, view_279]);  mul_220 = permute_167 = view_279 = None
    view_285: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_15, [3, 8, 12, 197, 64]);  cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_169: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_285, [1, 3, 0, 2, 4]);  view_285 = None
    clone_103: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
    view_286: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_103, [8, 197, 2304]);  clone_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_287: "f32[1576, 2304]" = torch.ops.aten.view.default(view_286, [1576, 2304]);  view_286 = None
    permute_170: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    mm_24: "f32[1576, 768]" = torch.ops.aten.mm.default(view_287, permute_170);  permute_170 = None
    permute_171: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_287, [1, 0])
    mm_25: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_171, view_163);  permute_171 = view_163 = None
    permute_172: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_61: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[2304]" = torch.ops.aten.view.default(sum_61, [2304]);  sum_61 = None
    permute_173: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    view_289: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_24, [8, 197, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_9: "f32[768]" = torch.ops.aten.slice.Tensor(view_288, 0, 0, 768)
    slice_11: "f32[768]" = torch.ops.aten.slice.Tensor(view_288, 0, 1536, 2304);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_58: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_64);  add_71 = getitem_64 = None
    mul_221: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_18);  sub_58 = None
    mul_222: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_289, primals_93);  primals_93 = None
    mul_223: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_222, 768)
    sum_62: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True)
    mul_224: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_222, mul_221);  mul_222 = None
    sum_63: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True);  mul_224 = None
    mul_225: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_221, sum_63);  sum_63 = None
    sub_59: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_223, sum_62);  mul_223 = sum_62 = None
    sub_60: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_225);  sub_59 = mul_225 = None
    div_19: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    mul_226: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_60);  div_19 = sub_60 = None
    mul_227: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_289, mul_221);  mul_221 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_227, [0, 1]);  mul_227 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_289, [0, 1]);  view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_109: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_108, mul_226);  add_108 = mul_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_228: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_109, primals_89);  primals_89 = None
    mul_229: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_109, clone_72);  clone_72 = None
    sum_66: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1], True);  mul_229 = None
    view_290: "f32[768]" = torch.ops.aten.view.default(sum_66, [768]);  sum_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_291: "f32[1576, 768]" = torch.ops.aten.view.default(mul_228, [1576, 768]);  mul_228 = None
    permute_174: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    mm_26: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_291, permute_174);  permute_174 = None
    permute_175: "f32[768, 1576]" = torch.ops.aten.permute.default(view_291, [1, 0])
    mm_27: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_175, view_161);  permute_175 = view_161 = None
    permute_176: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
    view_292: "f32[768]" = torch.ops.aten.view.default(sum_67, [768]);  sum_67 = None
    permute_177: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
    view_293: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_26, [8, 197, 3072]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_230: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_160, 0.7071067811865476)
    erf_15: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_230);  mul_230 = None
    add_110: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_231: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_110, 0.5);  add_110 = None
    mul_232: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_160, view_160)
    mul_233: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_232, -0.5);  mul_232 = None
    exp_15: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_233);  mul_233 = None
    mul_234: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_235: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_160, mul_234);  view_160 = mul_234 = None
    add_111: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_231, mul_235);  mul_231 = mul_235 = None
    mul_236: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_293, add_111);  view_293 = add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_294: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_236, [1576, 3072]);  mul_236 = None
    permute_178: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_28: "f32[1576, 768]" = torch.ops.aten.mm.default(view_294, permute_178);  permute_178 = None
    permute_179: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_29: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_179, view_159);  permute_179 = view_159 = None
    permute_180: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_68: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
    view_295: "f32[3072]" = torch.ops.aten.view.default(sum_68, [3072]);  sum_68 = None
    permute_181: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    view_296: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_28, [8, 197, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_61: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_62);  add_67 = getitem_62 = None
    mul_237: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_17);  sub_61 = None
    mul_238: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_296, primals_90);  primals_90 = None
    mul_239: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_238, 768)
    sum_69: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_238, [2], True)
    mul_240: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_238, mul_237);  mul_238 = None
    sum_70: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_240, [2], True);  mul_240 = None
    mul_241: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_237, sum_70);  sum_70 = None
    sub_62: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_239, sum_69);  mul_239 = sum_69 = None
    sub_63: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_241);  sub_62 = mul_241 = None
    div_20: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    mul_242: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_63);  div_20 = sub_63 = None
    mul_243: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_296, mul_237);  mul_237 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_243, [0, 1]);  mul_243 = None
    sum_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_296, [0, 1]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_112: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_109, mul_242);  add_109 = mul_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_244: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_112, primals_82);  primals_82 = None
    mul_245: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_112, clone_70);  clone_70 = None
    sum_73: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_245, [0, 1], True);  mul_245 = None
    view_297: "f32[768]" = torch.ops.aten.view.default(sum_73, [768]);  sum_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_298: "f32[1576, 768]" = torch.ops.aten.view.default(mul_244, [1576, 768]);  mul_244 = None
    permute_182: "f32[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_30: "f32[1576, 768]" = torch.ops.aten.mm.default(view_298, permute_182);  permute_182 = None
    permute_183: "f32[768, 1576]" = torch.ops.aten.permute.default(view_298, [1, 0])
    mm_31: "f32[768, 768]" = torch.ops.aten.mm.default(permute_183, view_157);  permute_183 = view_157 = None
    permute_184: "f32[768, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_74: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
    view_299: "f32[768]" = torch.ops.aten.view.default(sum_74, [768]);  sum_74 = None
    permute_185: "f32[768, 768]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    view_300: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_30, [8, 197, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_301: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_300, [8, 197, 12, 64]);  view_300 = None
    permute_186: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_104: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    view_302: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_104, [96, 197, 64]);  clone_104 = None
    permute_187: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    bmm_36: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_187, view_302);  permute_187 = None
    permute_188: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_154, [0, 2, 1]);  view_154 = None
    bmm_37: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_302, permute_188);  view_302 = permute_188 = None
    view_303: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_36, [8, 12, 197, 64]);  bmm_36 = None
    view_304: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_37, [8, 12, 197, 197]);  bmm_37 = None
    alias_15: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_246: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_304, alias_15);  view_304 = None
    sum_75: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_246, [-1], True)
    mul_247: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_15, sum_75);  alias_15 = sum_75 = None
    sub_64: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_246, mul_247);  mul_246 = mul_247 = None
    sum_76: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_64, [0], True)
    view_305: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_64, [96, 197, 197]);  sub_64 = None
    permute_189: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
    bmm_38: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_189, view_305);  permute_189 = None
    permute_190: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
    bmm_39: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_305, permute_190);  view_305 = permute_190 = None
    view_306: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_38, [8, 12, 64, 197]);  bmm_38 = None
    view_307: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_39, [8, 12, 197, 64]);  bmm_39 = None
    mul_248: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_306, 0.3535533905932738);  view_306 = None
    permute_191: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_248, [0, 1, 3, 2]);  mul_248 = None
    mul_249: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_307, 0.3535533905932738);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_3: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_76, 0);  sum_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_192: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_3, [1, 2, 0]);  squeeze_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_308: "f32[38809, 12]" = torch.ops.aten.view.default(permute_192, [38809, 12]);  permute_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_5: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_3: "f32[732, 12]" = torch.ops.aten.index_put.default(full_5, [view_148], view_308, True);  full_5 = view_148 = view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_16: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_249, permute_191, view_303]);  mul_249 = permute_191 = view_303 = None
    view_309: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_16, [3, 8, 12, 197, 64]);  cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_193: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_309, [1, 3, 0, 2, 4]);  view_309 = None
    clone_105: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    view_310: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_105, [8, 197, 2304]);  clone_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_311: "f32[1576, 2304]" = torch.ops.aten.view.default(view_310, [1576, 2304]);  view_310 = None
    permute_194: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_32: "f32[1576, 768]" = torch.ops.aten.mm.default(view_311, permute_194);  permute_194 = None
    permute_195: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_311, [1, 0])
    mm_33: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_195, view_145);  permute_195 = view_145 = None
    permute_196: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_77: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_311, [0], True);  view_311 = None
    view_312: "f32[2304]" = torch.ops.aten.view.default(sum_77, [2304]);  sum_77 = None
    permute_197: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_313: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_32, [8, 197, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_12: "f32[768]" = torch.ops.aten.slice.Tensor(view_312, 0, 0, 768)
    slice_14: "f32[768]" = torch.ops.aten.slice.Tensor(view_312, 0, 1536, 2304);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_65: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_57);  add_63 = getitem_57 = None
    mul_250: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_16);  sub_65 = None
    mul_251: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_313, primals_83);  primals_83 = None
    mul_252: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_251, 768)
    sum_78: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_251, [2], True)
    mul_253: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_251, mul_250);  mul_251 = None
    sum_79: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True);  mul_253 = None
    mul_254: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_250, sum_79);  sum_79 = None
    sub_66: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_252, sum_78);  mul_252 = sum_78 = None
    sub_67: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_66, mul_254);  sub_66 = mul_254 = None
    div_21: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    mul_255: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_67);  div_21 = sub_67 = None
    mul_256: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_313, mul_250);  mul_250 = None
    sum_80: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_256, [0, 1]);  mul_256 = None
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_313, [0, 1]);  view_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_113: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_112, mul_255);  add_112 = mul_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_257: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_113, primals_79);  primals_79 = None
    mul_258: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_113, clone_64);  clone_64 = None
    sum_82: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1], True);  mul_258 = None
    view_314: "f32[768]" = torch.ops.aten.view.default(sum_82, [768]);  sum_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_315: "f32[1576, 768]" = torch.ops.aten.view.default(mul_257, [1576, 768]);  mul_257 = None
    permute_198: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_34: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_315, permute_198);  permute_198 = None
    permute_199: "f32[768, 1576]" = torch.ops.aten.permute.default(view_315, [1, 0])
    mm_35: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_199, view_143);  permute_199 = view_143 = None
    permute_200: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
    view_316: "f32[768]" = torch.ops.aten.view.default(sum_83, [768]);  sum_83 = None
    permute_201: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    view_317: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_34, [8, 197, 3072]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_259: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.7071067811865476)
    erf_16: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_259);  mul_259 = None
    add_114: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_260: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_114, 0.5);  add_114 = None
    mul_261: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, view_142)
    mul_262: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_261, -0.5);  mul_261 = None
    exp_16: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_262);  mul_262 = None
    mul_263: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_264: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, mul_263);  view_142 = mul_263 = None
    add_115: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_260, mul_264);  mul_260 = mul_264 = None
    mul_265: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_317, add_115);  view_317 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_318: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_265, [1576, 3072]);  mul_265 = None
    permute_202: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_36: "f32[1576, 768]" = torch.ops.aten.mm.default(view_318, permute_202);  permute_202 = None
    permute_203: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_318, [1, 0])
    mm_37: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_203, view_141);  permute_203 = view_141 = None
    permute_204: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_84: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
    view_319: "f32[3072]" = torch.ops.aten.view.default(sum_84, [3072]);  sum_84 = None
    permute_205: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
    view_320: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_36, [8, 197, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_68: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_55);  add_59 = getitem_55 = None
    mul_266: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_15);  sub_68 = None
    mul_267: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_320, primals_80);  primals_80 = None
    mul_268: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_267, 768)
    sum_85: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_267, [2], True)
    mul_269: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_267, mul_266);  mul_267 = None
    sum_86: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True);  mul_269 = None
    mul_270: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_266, sum_86);  sum_86 = None
    sub_69: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_268, sum_85);  mul_268 = sum_85 = None
    sub_70: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_69, mul_270);  sub_69 = mul_270 = None
    div_22: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    mul_271: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_70);  div_22 = sub_70 = None
    mul_272: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_320, mul_266);  mul_266 = None
    sum_87: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_272, [0, 1]);  mul_272 = None
    sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_320, [0, 1]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_116: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_113, mul_271);  add_113 = mul_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_273: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_116, primals_72);  primals_72 = None
    mul_274: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_116, clone_62);  clone_62 = None
    sum_89: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1], True);  mul_274 = None
    view_321: "f32[768]" = torch.ops.aten.view.default(sum_89, [768]);  sum_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_322: "f32[1576, 768]" = torch.ops.aten.view.default(mul_273, [1576, 768]);  mul_273 = None
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    mm_38: "f32[1576, 768]" = torch.ops.aten.mm.default(view_322, permute_206);  permute_206 = None
    permute_207: "f32[768, 1576]" = torch.ops.aten.permute.default(view_322, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_207, view_139);  permute_207 = view_139 = None
    permute_208: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_322, [0], True);  view_322 = None
    view_323: "f32[768]" = torch.ops.aten.view.default(sum_90, [768]);  sum_90 = None
    permute_209: "f32[768, 768]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    view_324: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_38, [8, 197, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_325: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_324, [8, 197, 12, 64]);  view_324 = None
    permute_210: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_325, [0, 2, 1, 3]);  view_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_106: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
    view_326: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_106, [96, 197, 64]);  clone_106 = None
    permute_211: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    bmm_40: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_211, view_326);  permute_211 = None
    permute_212: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    bmm_41: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_326, permute_212);  view_326 = permute_212 = None
    view_327: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_40, [8, 12, 197, 64]);  bmm_40 = None
    view_328: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_41, [8, 12, 197, 197]);  bmm_41 = None
    alias_16: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_275: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_328, alias_16);  view_328 = None
    sum_91: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_275, [-1], True)
    mul_276: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_16, sum_91);  alias_16 = sum_91 = None
    sub_71: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_275, mul_276);  mul_275 = mul_276 = None
    sum_92: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_71, [0], True)
    view_329: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_71, [96, 197, 197]);  sub_71 = None
    permute_213: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    bmm_42: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_213, view_329);  permute_213 = None
    permute_214: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    bmm_43: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_329, permute_214);  view_329 = permute_214 = None
    view_330: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_42, [8, 12, 64, 197]);  bmm_42 = None
    view_331: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_43, [8, 12, 197, 64]);  bmm_43 = None
    mul_277: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_330, 0.3535533905932738);  view_330 = None
    permute_215: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_277, [0, 1, 3, 2]);  mul_277 = None
    mul_278: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_331, 0.3535533905932738);  view_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_4: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_92, 0);  sum_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_216: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_4, [1, 2, 0]);  squeeze_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_332: "f32[38809, 12]" = torch.ops.aten.view.default(permute_216, [38809, 12]);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_6: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_4: "f32[732, 12]" = torch.ops.aten.index_put.default(full_6, [view_130], view_332, True);  full_6 = view_130 = view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_17: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_278, permute_215, view_327]);  mul_278 = permute_215 = view_327 = None
    view_333: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_17, [3, 8, 12, 197, 64]);  cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_217: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_333, [1, 3, 0, 2, 4]);  view_333 = None
    clone_107: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
    view_334: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_107, [8, 197, 2304]);  clone_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_335: "f32[1576, 2304]" = torch.ops.aten.view.default(view_334, [1576, 2304]);  view_334 = None
    permute_218: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_40: "f32[1576, 768]" = torch.ops.aten.mm.default(view_335, permute_218);  permute_218 = None
    permute_219: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_335, [1, 0])
    mm_41: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_219, view_127);  permute_219 = view_127 = None
    permute_220: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_93: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_335, [0], True);  view_335 = None
    view_336: "f32[2304]" = torch.ops.aten.view.default(sum_93, [2304]);  sum_93 = None
    permute_221: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    view_337: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_40, [8, 197, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_15: "f32[768]" = torch.ops.aten.slice.Tensor(view_336, 0, 0, 768)
    slice_17: "f32[768]" = torch.ops.aten.slice.Tensor(view_336, 0, 1536, 2304);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_72: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_50);  add_55 = getitem_50 = None
    mul_279: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_14);  sub_72 = None
    mul_280: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_337, primals_73);  primals_73 = None
    mul_281: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_280, 768)
    sum_94: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_280, [2], True)
    mul_282: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_280, mul_279);  mul_280 = None
    sum_95: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_282, [2], True);  mul_282 = None
    mul_283: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_279, sum_95);  sum_95 = None
    sub_73: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_281, sum_94);  mul_281 = sum_94 = None
    sub_74: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_73, mul_283);  sub_73 = mul_283 = None
    div_23: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    mul_284: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_74);  div_23 = sub_74 = None
    mul_285: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_337, mul_279);  mul_279 = None
    sum_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_285, [0, 1]);  mul_285 = None
    sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_337, [0, 1]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_117: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_116, mul_284);  add_116 = mul_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_286: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_117, primals_69);  primals_69 = None
    mul_287: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_117, clone_56);  clone_56 = None
    sum_98: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1], True);  mul_287 = None
    view_338: "f32[768]" = torch.ops.aten.view.default(sum_98, [768]);  sum_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_339: "f32[1576, 768]" = torch.ops.aten.view.default(mul_286, [1576, 768]);  mul_286 = None
    permute_222: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    mm_42: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_339, permute_222);  permute_222 = None
    permute_223: "f32[768, 1576]" = torch.ops.aten.permute.default(view_339, [1, 0])
    mm_43: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_223, view_125);  permute_223 = view_125 = None
    permute_224: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_99: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_339, [0], True);  view_339 = None
    view_340: "f32[768]" = torch.ops.aten.view.default(sum_99, [768]);  sum_99 = None
    permute_225: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    view_341: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_42, [8, 197, 3072]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_288: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, 0.7071067811865476)
    erf_17: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_288);  mul_288 = None
    add_118: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_289: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_118, 0.5);  add_118 = None
    mul_290: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, view_124)
    mul_291: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_290, -0.5);  mul_290 = None
    exp_17: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_291);  mul_291 = None
    mul_292: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_293: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, mul_292);  view_124 = mul_292 = None
    add_119: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_289, mul_293);  mul_289 = mul_293 = None
    mul_294: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_341, add_119);  view_341 = add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_342: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_294, [1576, 3072]);  mul_294 = None
    permute_226: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_44: "f32[1576, 768]" = torch.ops.aten.mm.default(view_342, permute_226);  permute_226 = None
    permute_227: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_342, [1, 0])
    mm_45: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_227, view_123);  permute_227 = view_123 = None
    permute_228: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_100: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_342, [0], True);  view_342 = None
    view_343: "f32[3072]" = torch.ops.aten.view.default(sum_100, [3072]);  sum_100 = None
    permute_229: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    view_344: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_44, [8, 197, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_75: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_48);  add_51 = getitem_48 = None
    mul_295: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_13);  sub_75 = None
    mul_296: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_344, primals_70);  primals_70 = None
    mul_297: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_296, 768)
    sum_101: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_296, [2], True)
    mul_298: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_296, mul_295);  mul_296 = None
    sum_102: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
    mul_299: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_295, sum_102);  sum_102 = None
    sub_76: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_297, sum_101);  mul_297 = sum_101 = None
    sub_77: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_76, mul_299);  sub_76 = mul_299 = None
    div_24: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_300: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_77);  div_24 = sub_77 = None
    mul_301: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_344, mul_295);  mul_295 = None
    sum_103: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_301, [0, 1]);  mul_301 = None
    sum_104: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_344, [0, 1]);  view_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_120: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_117, mul_300);  add_117 = mul_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_302: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_120, primals_62);  primals_62 = None
    mul_303: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_120, clone_54);  clone_54 = None
    sum_105: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_303, [0, 1], True);  mul_303 = None
    view_345: "f32[768]" = torch.ops.aten.view.default(sum_105, [768]);  sum_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_346: "f32[1576, 768]" = torch.ops.aten.view.default(mul_302, [1576, 768]);  mul_302 = None
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_46: "f32[1576, 768]" = torch.ops.aten.mm.default(view_346, permute_230);  permute_230 = None
    permute_231: "f32[768, 1576]" = torch.ops.aten.permute.default(view_346, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_231, view_121);  permute_231 = view_121 = None
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_106: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_346, [0], True);  view_346 = None
    view_347: "f32[768]" = torch.ops.aten.view.default(sum_106, [768]);  sum_106 = None
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    view_348: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_46, [8, 197, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_349: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_348, [8, 197, 12, 64]);  view_348 = None
    permute_234: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_349, [0, 2, 1, 3]);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_108: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_234, memory_format = torch.contiguous_format);  permute_234 = None
    view_350: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_108, [96, 197, 64]);  clone_108 = None
    permute_235: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
    bmm_44: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_235, view_350);  permute_235 = None
    permute_236: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_118, [0, 2, 1]);  view_118 = None
    bmm_45: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_350, permute_236);  view_350 = permute_236 = None
    view_351: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_44, [8, 12, 197, 64]);  bmm_44 = None
    view_352: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_45, [8, 12, 197, 197]);  bmm_45 = None
    alias_17: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_304: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_352, alias_17);  view_352 = None
    sum_107: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [-1], True)
    mul_305: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_17, sum_107);  alias_17 = sum_107 = None
    sub_78: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
    sum_108: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_78, [0], True)
    view_353: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_78, [96, 197, 197]);  sub_78 = None
    permute_237: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
    bmm_46: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_237, view_353);  permute_237 = None
    permute_238: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
    bmm_47: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_353, permute_238);  view_353 = permute_238 = None
    view_354: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_46, [8, 12, 64, 197]);  bmm_46 = None
    view_355: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_47, [8, 12, 197, 64]);  bmm_47 = None
    mul_306: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_354, 0.3535533905932738);  view_354 = None
    permute_239: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_306, [0, 1, 3, 2]);  mul_306 = None
    mul_307: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_355, 0.3535533905932738);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_5: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_108, 0);  sum_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_240: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_5, [1, 2, 0]);  squeeze_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_356: "f32[38809, 12]" = torch.ops.aten.view.default(permute_240, [38809, 12]);  permute_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_7: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_5: "f32[732, 12]" = torch.ops.aten.index_put.default(full_7, [view_112], view_356, True);  full_7 = view_112 = view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_18: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_307, permute_239, view_351]);  mul_307 = permute_239 = view_351 = None
    view_357: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_18, [3, 8, 12, 197, 64]);  cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_241: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_357, [1, 3, 0, 2, 4]);  view_357 = None
    clone_109: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_241, memory_format = torch.contiguous_format);  permute_241 = None
    view_358: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_109, [8, 197, 2304]);  clone_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_359: "f32[1576, 2304]" = torch.ops.aten.view.default(view_358, [1576, 2304]);  view_358 = None
    permute_242: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    mm_48: "f32[1576, 768]" = torch.ops.aten.mm.default(view_359, permute_242);  permute_242 = None
    permute_243: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_359, [1, 0])
    mm_49: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_243, view_109);  permute_243 = view_109 = None
    permute_244: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_109: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_359, [0], True);  view_359 = None
    view_360: "f32[2304]" = torch.ops.aten.view.default(sum_109, [2304]);  sum_109 = None
    permute_245: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    view_361: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_48, [8, 197, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_18: "f32[768]" = torch.ops.aten.slice.Tensor(view_360, 0, 0, 768)
    slice_20: "f32[768]" = torch.ops.aten.slice.Tensor(view_360, 0, 1536, 2304);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_79: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_43);  add_47 = getitem_43 = None
    mul_308: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_12);  sub_79 = None
    mul_309: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_361, primals_63);  primals_63 = None
    mul_310: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_309, 768)
    sum_110: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
    mul_311: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_309, mul_308);  mul_309 = None
    sum_111: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True);  mul_311 = None
    mul_312: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_308, sum_111);  sum_111 = None
    sub_80: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_310, sum_110);  mul_310 = sum_110 = None
    sub_81: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_312);  sub_80 = mul_312 = None
    div_25: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_313: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_81);  div_25 = sub_81 = None
    mul_314: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_361, mul_308);  mul_308 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_314, [0, 1]);  mul_314 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_361, [0, 1]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_121: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_120, mul_313);  add_120 = mul_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_315: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_121, primals_59);  primals_59 = None
    mul_316: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_121, clone_48);  clone_48 = None
    sum_114: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1], True);  mul_316 = None
    view_362: "f32[768]" = torch.ops.aten.view.default(sum_114, [768]);  sum_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_363: "f32[1576, 768]" = torch.ops.aten.view.default(mul_315, [1576, 768]);  mul_315 = None
    permute_246: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_50: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_363, permute_246);  permute_246 = None
    permute_247: "f32[768, 1576]" = torch.ops.aten.permute.default(view_363, [1, 0])
    mm_51: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_247, view_107);  permute_247 = view_107 = None
    permute_248: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
    view_364: "f32[768]" = torch.ops.aten.view.default(sum_115, [768]);  sum_115 = None
    permute_249: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_248, [1, 0]);  permute_248 = None
    view_365: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_50, [8, 197, 3072]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_317: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476)
    erf_18: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_317);  mul_317 = None
    add_122: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_318: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_122, 0.5);  add_122 = None
    mul_319: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, view_106)
    mul_320: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_319, -0.5);  mul_319 = None
    exp_18: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_320);  mul_320 = None
    mul_321: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_322: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, mul_321);  view_106 = mul_321 = None
    add_123: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_318, mul_322);  mul_318 = mul_322 = None
    mul_323: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_365, add_123);  view_365 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_366: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_323, [1576, 3072]);  mul_323 = None
    permute_250: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_52: "f32[1576, 768]" = torch.ops.aten.mm.default(view_366, permute_250);  permute_250 = None
    permute_251: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_366, [1, 0])
    mm_53: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_251, view_105);  permute_251 = view_105 = None
    permute_252: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_116: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_366, [0], True);  view_366 = None
    view_367: "f32[3072]" = torch.ops.aten.view.default(sum_116, [3072]);  sum_116 = None
    permute_253: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    view_368: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_52, [8, 197, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_82: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_41);  add_43 = getitem_41 = None
    mul_324: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_11);  sub_82 = None
    mul_325: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_368, primals_60);  primals_60 = None
    mul_326: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_325, 768)
    sum_117: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)
    mul_327: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_325, mul_324);  mul_325 = None
    sum_118: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
    mul_328: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_324, sum_118);  sum_118 = None
    sub_83: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_326, sum_117);  mul_326 = sum_117 = None
    sub_84: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_328);  sub_83 = mul_328 = None
    div_26: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_329: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_84);  div_26 = sub_84 = None
    mul_330: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_368, mul_324);  mul_324 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_368, [0, 1]);  view_368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_124: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_121, mul_329);  add_121 = mul_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_331: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_124, primals_52);  primals_52 = None
    mul_332: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_124, clone_46);  clone_46 = None
    sum_121: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_332, [0, 1], True);  mul_332 = None
    view_369: "f32[768]" = torch.ops.aten.view.default(sum_121, [768]);  sum_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_370: "f32[1576, 768]" = torch.ops.aten.view.default(mul_331, [1576, 768]);  mul_331 = None
    permute_254: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_54: "f32[1576, 768]" = torch.ops.aten.mm.default(view_370, permute_254);  permute_254 = None
    permute_255: "f32[768, 1576]" = torch.ops.aten.permute.default(view_370, [1, 0])
    mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_255, view_103);  permute_255 = view_103 = None
    permute_256: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_122: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_370, [0], True);  view_370 = None
    view_371: "f32[768]" = torch.ops.aten.view.default(sum_122, [768]);  sum_122 = None
    permute_257: "f32[768, 768]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    view_372: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_54, [8, 197, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_373: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_372, [8, 197, 12, 64]);  view_372 = None
    permute_258: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_110: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
    view_374: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_110, [96, 197, 64]);  clone_110 = None
    permute_259: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    bmm_48: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_259, view_374);  permute_259 = None
    permute_260: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_49: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_374, permute_260);  view_374 = permute_260 = None
    view_375: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_48, [8, 12, 197, 64]);  bmm_48 = None
    view_376: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_49, [8, 12, 197, 197]);  bmm_49 = None
    alias_18: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_333: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_376, alias_18);  view_376 = None
    sum_123: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_333, [-1], True)
    mul_334: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_18, sum_123);  alias_18 = sum_123 = None
    sub_85: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_333, mul_334);  mul_333 = mul_334 = None
    sum_124: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_85, [0], True)
    view_377: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_85, [96, 197, 197]);  sub_85 = None
    permute_261: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
    bmm_50: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_261, view_377);  permute_261 = None
    permute_262: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    bmm_51: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_377, permute_262);  view_377 = permute_262 = None
    view_378: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_50, [8, 12, 64, 197]);  bmm_50 = None
    view_379: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_51, [8, 12, 197, 64]);  bmm_51 = None
    mul_335: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_378, 0.3535533905932738);  view_378 = None
    permute_263: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_335, [0, 1, 3, 2]);  mul_335 = None
    mul_336: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_379, 0.3535533905932738);  view_379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_6: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_124, 0);  sum_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_264: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_6, [1, 2, 0]);  squeeze_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_380: "f32[38809, 12]" = torch.ops.aten.view.default(permute_264, [38809, 12]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_8: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_6: "f32[732, 12]" = torch.ops.aten.index_put.default(full_8, [view_94], view_380, True);  full_8 = view_94 = view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_19: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_336, permute_263, view_375]);  mul_336 = permute_263 = view_375 = None
    view_381: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_19, [3, 8, 12, 197, 64]);  cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_265: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_381, [1, 3, 0, 2, 4]);  view_381 = None
    clone_111: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None
    view_382: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_111, [8, 197, 2304]);  clone_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_383: "f32[1576, 2304]" = torch.ops.aten.view.default(view_382, [1576, 2304]);  view_382 = None
    permute_266: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_56: "f32[1576, 768]" = torch.ops.aten.mm.default(view_383, permute_266);  permute_266 = None
    permute_267: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_383, [1, 0])
    mm_57: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_267, view_91);  permute_267 = view_91 = None
    permute_268: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_125: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
    view_384: "f32[2304]" = torch.ops.aten.view.default(sum_125, [2304]);  sum_125 = None
    permute_269: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_385: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_56, [8, 197, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_21: "f32[768]" = torch.ops.aten.slice.Tensor(view_384, 0, 0, 768)
    slice_23: "f32[768]" = torch.ops.aten.slice.Tensor(view_384, 0, 1536, 2304);  view_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_86: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_36);  add_39 = getitem_36 = None
    mul_337: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_10);  sub_86 = None
    mul_338: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_385, primals_53);  primals_53 = None
    mul_339: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_338, 768)
    sum_126: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True)
    mul_340: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_338, mul_337);  mul_338 = None
    sum_127: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True);  mul_340 = None
    mul_341: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_337, sum_127);  sum_127 = None
    sub_87: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_339, sum_126);  mul_339 = sum_126 = None
    sub_88: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_87, mul_341);  sub_87 = mul_341 = None
    div_27: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_342: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_27, sub_88);  div_27 = sub_88 = None
    mul_343: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_385, mul_337);  mul_337 = None
    sum_128: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1]);  mul_343 = None
    sum_129: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_385, [0, 1]);  view_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_125: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_124, mul_342);  add_124 = mul_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_344: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_125, primals_49);  primals_49 = None
    mul_345: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_125, clone_40);  clone_40 = None
    sum_130: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_345, [0, 1], True);  mul_345 = None
    view_386: "f32[768]" = torch.ops.aten.view.default(sum_130, [768]);  sum_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_387: "f32[1576, 768]" = torch.ops.aten.view.default(mul_344, [1576, 768]);  mul_344 = None
    permute_270: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    mm_58: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_387, permute_270);  permute_270 = None
    permute_271: "f32[768, 1576]" = torch.ops.aten.permute.default(view_387, [1, 0])
    mm_59: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_271, view_89);  permute_271 = view_89 = None
    permute_272: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_387, [0], True);  view_387 = None
    view_388: "f32[768]" = torch.ops.aten.view.default(sum_131, [768]);  sum_131 = None
    permute_273: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
    view_389: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_58, [8, 197, 3072]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_346: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476)
    erf_19: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_346);  mul_346 = None
    add_126: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_347: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_126, 0.5);  add_126 = None
    mul_348: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, view_88)
    mul_349: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_348, -0.5);  mul_348 = None
    exp_19: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_349);  mul_349 = None
    mul_350: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_351: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, mul_350);  view_88 = mul_350 = None
    add_127: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_347, mul_351);  mul_347 = mul_351 = None
    mul_352: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_389, add_127);  view_389 = add_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_390: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_352, [1576, 3072]);  mul_352 = None
    permute_274: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    mm_60: "f32[1576, 768]" = torch.ops.aten.mm.default(view_390, permute_274);  permute_274 = None
    permute_275: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_390, [1, 0])
    mm_61: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_275, view_87);  permute_275 = view_87 = None
    permute_276: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_132: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_390, [0], True);  view_390 = None
    view_391: "f32[3072]" = torch.ops.aten.view.default(sum_132, [3072]);  sum_132 = None
    permute_277: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
    view_392: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_60, [8, 197, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_89: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_34);  add_35 = getitem_34 = None
    mul_353: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_89, rsqrt_9);  sub_89 = None
    mul_354: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_392, primals_50);  primals_50 = None
    mul_355: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_354, 768)
    sum_133: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_354, [2], True)
    mul_356: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_354, mul_353);  mul_354 = None
    sum_134: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_356, [2], True);  mul_356 = None
    mul_357: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_353, sum_134);  sum_134 = None
    sub_90: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_355, sum_133);  mul_355 = sum_133 = None
    sub_91: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_90, mul_357);  sub_90 = mul_357 = None
    div_28: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_358: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_28, sub_91);  div_28 = sub_91 = None
    mul_359: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_392, mul_353);  mul_353 = None
    sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_359, [0, 1]);  mul_359 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_392, [0, 1]);  view_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_128: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_125, mul_358);  add_125 = mul_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_360: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_128, primals_42);  primals_42 = None
    mul_361: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_128, clone_38);  clone_38 = None
    sum_137: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_361, [0, 1], True);  mul_361 = None
    view_393: "f32[768]" = torch.ops.aten.view.default(sum_137, [768]);  sum_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_394: "f32[1576, 768]" = torch.ops.aten.view.default(mul_360, [1576, 768]);  mul_360 = None
    permute_278: "f32[768, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    mm_62: "f32[1576, 768]" = torch.ops.aten.mm.default(view_394, permute_278);  permute_278 = None
    permute_279: "f32[768, 1576]" = torch.ops.aten.permute.default(view_394, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_279, view_85);  permute_279 = view_85 = None
    permute_280: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_394, [0], True);  view_394 = None
    view_395: "f32[768]" = torch.ops.aten.view.default(sum_138, [768]);  sum_138 = None
    permute_281: "f32[768, 768]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
    view_396: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_62, [8, 197, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_397: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_396, [8, 197, 12, 64]);  view_396 = None
    permute_282: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_397, [0, 2, 1, 3]);  view_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_112: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None
    view_398: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_112, [96, 197, 64]);  clone_112 = None
    permute_283: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    bmm_52: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_283, view_398);  permute_283 = None
    permute_284: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    bmm_53: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_398, permute_284);  view_398 = permute_284 = None
    view_399: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_52, [8, 12, 197, 64]);  bmm_52 = None
    view_400: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_53, [8, 12, 197, 197]);  bmm_53 = None
    alias_19: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_362: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_400, alias_19);  view_400 = None
    sum_139: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_362, [-1], True)
    mul_363: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_19, sum_139);  alias_19 = sum_139 = None
    sub_92: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_362, mul_363);  mul_362 = mul_363 = None
    sum_140: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_92, [0], True)
    view_401: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_92, [96, 197, 197]);  sub_92 = None
    permute_285: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_54: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_285, view_401);  permute_285 = None
    permute_286: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    bmm_55: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_401, permute_286);  view_401 = permute_286 = None
    view_402: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_54, [8, 12, 64, 197]);  bmm_54 = None
    view_403: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_55, [8, 12, 197, 64]);  bmm_55 = None
    mul_364: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_402, 0.3535533905932738);  view_402 = None
    permute_287: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_364, [0, 1, 3, 2]);  mul_364 = None
    mul_365: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_403, 0.3535533905932738);  view_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_7: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_140, 0);  sum_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_288: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_7, [1, 2, 0]);  squeeze_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_404: "f32[38809, 12]" = torch.ops.aten.view.default(permute_288, [38809, 12]);  permute_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_9: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_7: "f32[732, 12]" = torch.ops.aten.index_put.default(full_9, [view_76], view_404, True);  full_9 = view_76 = view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_20: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_365, permute_287, view_399]);  mul_365 = permute_287 = view_399 = None
    view_405: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_20, [3, 8, 12, 197, 64]);  cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_289: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_405, [1, 3, 0, 2, 4]);  view_405 = None
    clone_113: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    view_406: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_113, [8, 197, 2304]);  clone_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_407: "f32[1576, 2304]" = torch.ops.aten.view.default(view_406, [1576, 2304]);  view_406 = None
    permute_290: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_64: "f32[1576, 768]" = torch.ops.aten.mm.default(view_407, permute_290);  permute_290 = None
    permute_291: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_407, [1, 0])
    mm_65: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_291, view_73);  permute_291 = view_73 = None
    permute_292: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_141: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_407, [0], True);  view_407 = None
    view_408: "f32[2304]" = torch.ops.aten.view.default(sum_141, [2304]);  sum_141 = None
    permute_293: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    view_409: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_64, [8, 197, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_24: "f32[768]" = torch.ops.aten.slice.Tensor(view_408, 0, 0, 768)
    slice_26: "f32[768]" = torch.ops.aten.slice.Tensor(view_408, 0, 1536, 2304);  view_408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_93: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_29);  add_31 = getitem_29 = None
    mul_366: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_93, rsqrt_8);  sub_93 = None
    mul_367: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_409, primals_43);  primals_43 = None
    mul_368: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_367, 768)
    sum_142: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
    mul_369: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_367, mul_366);  mul_367 = None
    sum_143: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
    mul_370: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_366, sum_143);  sum_143 = None
    sub_94: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_368, sum_142);  mul_368 = sum_142 = None
    sub_95: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_94, mul_370);  sub_94 = mul_370 = None
    div_29: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_371: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_29, sub_95);  div_29 = sub_95 = None
    mul_372: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_409, mul_366);  mul_366 = None
    sum_144: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
    sum_145: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_409, [0, 1]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_129: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_128, mul_371);  add_128 = mul_371 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_373: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_129, primals_39);  primals_39 = None
    mul_374: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_129, clone_32);  clone_32 = None
    sum_146: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_374, [0, 1], True);  mul_374 = None
    view_410: "f32[768]" = torch.ops.aten.view.default(sum_146, [768]);  sum_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_411: "f32[1576, 768]" = torch.ops.aten.view.default(mul_373, [1576, 768]);  mul_373 = None
    permute_294: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_66: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_411, permute_294);  permute_294 = None
    permute_295: "f32[768, 1576]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_67: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_295, view_71);  permute_295 = view_71 = None
    permute_296: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[768]" = torch.ops.aten.view.default(sum_147, [768]);  sum_147 = None
    permute_297: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
    view_413: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_66, [8, 197, 3072]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_375: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.7071067811865476)
    erf_20: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_375);  mul_375 = None
    add_130: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_376: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_130, 0.5);  add_130 = None
    mul_377: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, view_70)
    mul_378: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_377, -0.5);  mul_377 = None
    exp_20: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_378);  mul_378 = None
    mul_379: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_380: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, mul_379);  view_70 = mul_379 = None
    add_131: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_376, mul_380);  mul_376 = mul_380 = None
    mul_381: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_413, add_131);  view_413 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_414: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_381, [1576, 3072]);  mul_381 = None
    permute_298: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_68: "f32[1576, 768]" = torch.ops.aten.mm.default(view_414, permute_298);  permute_298 = None
    permute_299: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_414, [1, 0])
    mm_69: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_299, view_69);  permute_299 = view_69 = None
    permute_300: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_148: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_414, [0], True);  view_414 = None
    view_415: "f32[3072]" = torch.ops.aten.view.default(sum_148, [3072]);  sum_148 = None
    permute_301: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
    view_416: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_68, [8, 197, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_96: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_27);  add_27 = getitem_27 = None
    mul_382: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_7);  sub_96 = None
    mul_383: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_416, primals_40);  primals_40 = None
    mul_384: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_383, 768)
    sum_149: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
    mul_385: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_383, mul_382);  mul_383 = None
    sum_150: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
    mul_386: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_382, sum_150);  sum_150 = None
    sub_97: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_384, sum_149);  mul_384 = sum_149 = None
    sub_98: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_97, mul_386);  sub_97 = mul_386 = None
    div_30: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_387: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_30, sub_98);  div_30 = sub_98 = None
    mul_388: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_416, mul_382);  mul_382 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
    sum_152: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_416, [0, 1]);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_132: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_129, mul_387);  add_129 = mul_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_389: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_132, primals_32);  primals_32 = None
    mul_390: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_132, clone_30);  clone_30 = None
    sum_153: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_390, [0, 1], True);  mul_390 = None
    view_417: "f32[768]" = torch.ops.aten.view.default(sum_153, [768]);  sum_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_418: "f32[1576, 768]" = torch.ops.aten.view.default(mul_389, [1576, 768]);  mul_389 = None
    permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_70: "f32[1576, 768]" = torch.ops.aten.mm.default(view_418, permute_302);  permute_302 = None
    permute_303: "f32[768, 1576]" = torch.ops.aten.permute.default(view_418, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_303, view_67);  permute_303 = view_67 = None
    permute_304: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_154: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_418, [0], True);  view_418 = None
    view_419: "f32[768]" = torch.ops.aten.view.default(sum_154, [768]);  sum_154 = None
    permute_305: "f32[768, 768]" = torch.ops.aten.permute.default(permute_304, [1, 0]);  permute_304 = None
    view_420: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_70, [8, 197, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_421: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_420, [8, 197, 12, 64]);  view_420 = None
    permute_306: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_421, [0, 2, 1, 3]);  view_421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_114: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_306, memory_format = torch.contiguous_format);  permute_306 = None
    view_422: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_114, [96, 197, 64]);  clone_114 = None
    permute_307: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    bmm_56: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_307, view_422);  permute_307 = None
    permute_308: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    bmm_57: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_422, permute_308);  view_422 = permute_308 = None
    view_423: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_56, [8, 12, 197, 64]);  bmm_56 = None
    view_424: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_57, [8, 12, 197, 197]);  bmm_57 = None
    alias_20: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_391: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_424, alias_20);  view_424 = None
    sum_155: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_391, [-1], True)
    mul_392: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_20, sum_155);  alias_20 = sum_155 = None
    sub_99: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_391, mul_392);  mul_391 = mul_392 = None
    sum_156: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_99, [0], True)
    view_425: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_99, [96, 197, 197]);  sub_99 = None
    permute_309: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    bmm_58: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_309, view_425);  permute_309 = None
    permute_310: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    bmm_59: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_425, permute_310);  view_425 = permute_310 = None
    view_426: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_58, [8, 12, 64, 197]);  bmm_58 = None
    view_427: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_59, [8, 12, 197, 64]);  bmm_59 = None
    mul_393: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_426, 0.3535533905932738);  view_426 = None
    permute_311: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_393, [0, 1, 3, 2]);  mul_393 = None
    mul_394: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_427, 0.3535533905932738);  view_427 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_8: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_156, 0);  sum_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_312: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_8, [1, 2, 0]);  squeeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_428: "f32[38809, 12]" = torch.ops.aten.view.default(permute_312, [38809, 12]);  permute_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_10: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_8: "f32[732, 12]" = torch.ops.aten.index_put.default(full_10, [view_58], view_428, True);  full_10 = view_58 = view_428 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_21: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_394, permute_311, view_423]);  mul_394 = permute_311 = view_423 = None
    view_429: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_21, [3, 8, 12, 197, 64]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_313: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_429, [1, 3, 0, 2, 4]);  view_429 = None
    clone_115: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_313, memory_format = torch.contiguous_format);  permute_313 = None
    view_430: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_115, [8, 197, 2304]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_431: "f32[1576, 2304]" = torch.ops.aten.view.default(view_430, [1576, 2304]);  view_430 = None
    permute_314: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_72: "f32[1576, 768]" = torch.ops.aten.mm.default(view_431, permute_314);  permute_314 = None
    permute_315: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_431, [1, 0])
    mm_73: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_315, view_55);  permute_315 = view_55 = None
    permute_316: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_157: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_431, [0], True);  view_431 = None
    view_432: "f32[2304]" = torch.ops.aten.view.default(sum_157, [2304]);  sum_157 = None
    permute_317: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
    view_433: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_72, [8, 197, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_27: "f32[768]" = torch.ops.aten.slice.Tensor(view_432, 0, 0, 768)
    slice_29: "f32[768]" = torch.ops.aten.slice.Tensor(view_432, 0, 1536, 2304);  view_432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_100: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_22);  add_23 = getitem_22 = None
    mul_395: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_6);  sub_100 = None
    mul_396: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_433, primals_33);  primals_33 = None
    mul_397: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_396, 768)
    sum_158: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_396, [2], True)
    mul_398: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_396, mul_395);  mul_396 = None
    sum_159: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_398, [2], True);  mul_398 = None
    mul_399: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_395, sum_159);  sum_159 = None
    sub_101: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_397, sum_158);  mul_397 = sum_158 = None
    sub_102: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_101, mul_399);  sub_101 = mul_399 = None
    div_31: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_400: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_31, sub_102);  div_31 = sub_102 = None
    mul_401: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_433, mul_395);  mul_395 = None
    sum_160: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_401, [0, 1]);  mul_401 = None
    sum_161: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_433, [0, 1]);  view_433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_133: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_132, mul_400);  add_132 = mul_400 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_402: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_133, primals_29);  primals_29 = None
    mul_403: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_133, clone_24);  clone_24 = None
    sum_162: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_403, [0, 1], True);  mul_403 = None
    view_434: "f32[768]" = torch.ops.aten.view.default(sum_162, [768]);  sum_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_435: "f32[1576, 768]" = torch.ops.aten.view.default(mul_402, [1576, 768]);  mul_402 = None
    permute_318: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_74: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_435, permute_318);  permute_318 = None
    permute_319: "f32[768, 1576]" = torch.ops.aten.permute.default(view_435, [1, 0])
    mm_75: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_319, view_53);  permute_319 = view_53 = None
    permute_320: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_163: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_435, [0], True);  view_435 = None
    view_436: "f32[768]" = torch.ops.aten.view.default(sum_163, [768]);  sum_163 = None
    permute_321: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
    view_437: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_74, [8, 197, 3072]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_404: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_52, 0.7071067811865476)
    erf_21: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_404);  mul_404 = None
    add_134: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_405: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_134, 0.5);  add_134 = None
    mul_406: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_52, view_52)
    mul_407: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_406, -0.5);  mul_406 = None
    exp_21: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_407);  mul_407 = None
    mul_408: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_409: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_52, mul_408);  view_52 = mul_408 = None
    add_135: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_405, mul_409);  mul_405 = mul_409 = None
    mul_410: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_437, add_135);  view_437 = add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_438: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_410, [1576, 3072]);  mul_410 = None
    permute_322: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_76: "f32[1576, 768]" = torch.ops.aten.mm.default(view_438, permute_322);  permute_322 = None
    permute_323: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_438, [1, 0])
    mm_77: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_323, view_51);  permute_323 = view_51 = None
    permute_324: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_164: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_438, [0], True);  view_438 = None
    view_439: "f32[3072]" = torch.ops.aten.view.default(sum_164, [3072]);  sum_164 = None
    permute_325: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_324, [1, 0]);  permute_324 = None
    view_440: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_76, [8, 197, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_103: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_20);  add_19 = getitem_20 = None
    mul_411: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_5);  sub_103 = None
    mul_412: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_440, primals_30);  primals_30 = None
    mul_413: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_412, 768)
    sum_165: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_412, [2], True)
    mul_414: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_412, mul_411);  mul_412 = None
    sum_166: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_414, [2], True);  mul_414 = None
    mul_415: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_411, sum_166);  sum_166 = None
    sub_104: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_413, sum_165);  mul_413 = sum_165 = None
    sub_105: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_415);  sub_104 = mul_415 = None
    div_32: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_416: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_32, sub_105);  div_32 = sub_105 = None
    mul_417: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_440, mul_411);  mul_411 = None
    sum_167: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_417, [0, 1]);  mul_417 = None
    sum_168: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_440, [0, 1]);  view_440 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_136: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_133, mul_416);  add_133 = mul_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_418: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_136, primals_22);  primals_22 = None
    mul_419: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_136, clone_22);  clone_22 = None
    sum_169: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_419, [0, 1], True);  mul_419 = None
    view_441: "f32[768]" = torch.ops.aten.view.default(sum_169, [768]);  sum_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_442: "f32[1576, 768]" = torch.ops.aten.view.default(mul_418, [1576, 768]);  mul_418 = None
    permute_326: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_78: "f32[1576, 768]" = torch.ops.aten.mm.default(view_442, permute_326);  permute_326 = None
    permute_327: "f32[768, 1576]" = torch.ops.aten.permute.default(view_442, [1, 0])
    mm_79: "f32[768, 768]" = torch.ops.aten.mm.default(permute_327, view_49);  permute_327 = view_49 = None
    permute_328: "f32[768, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_170: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
    view_443: "f32[768]" = torch.ops.aten.view.default(sum_170, [768]);  sum_170 = None
    permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(permute_328, [1, 0]);  permute_328 = None
    view_444: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_78, [8, 197, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_445: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_444, [8, 197, 12, 64]);  view_444 = None
    permute_330: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_116: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
    view_446: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_116, [96, 197, 64]);  clone_116 = None
    permute_331: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    bmm_60: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_331, view_446);  permute_331 = None
    permute_332: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
    bmm_61: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_446, permute_332);  view_446 = permute_332 = None
    view_447: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_60, [8, 12, 197, 64]);  bmm_60 = None
    view_448: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_61, [8, 12, 197, 197]);  bmm_61 = None
    alias_21: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_420: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_448, alias_21);  view_448 = None
    sum_171: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_420, [-1], True)
    mul_421: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_21, sum_171);  alias_21 = sum_171 = None
    sub_106: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_420, mul_421);  mul_420 = mul_421 = None
    sum_172: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_106, [0], True)
    view_449: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_106, [96, 197, 197]);  sub_106 = None
    permute_333: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None
    bmm_62: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_333, view_449);  permute_333 = None
    permute_334: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
    bmm_63: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_449, permute_334);  view_449 = permute_334 = None
    view_450: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_62, [8, 12, 64, 197]);  bmm_62 = None
    view_451: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_63, [8, 12, 197, 64]);  bmm_63 = None
    mul_422: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_450, 0.3535533905932738);  view_450 = None
    permute_335: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_422, [0, 1, 3, 2]);  mul_422 = None
    mul_423: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_451, 0.3535533905932738);  view_451 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_9: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_172, 0);  sum_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_336: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_9, [1, 2, 0]);  squeeze_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_452: "f32[38809, 12]" = torch.ops.aten.view.default(permute_336, [38809, 12]);  permute_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_11: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_9: "f32[732, 12]" = torch.ops.aten.index_put.default(full_11, [view_40], view_452, True);  full_11 = view_40 = view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_22: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_423, permute_335, view_447]);  mul_423 = permute_335 = view_447 = None
    view_453: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_22, [3, 8, 12, 197, 64]);  cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_337: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_453, [1, 3, 0, 2, 4]);  view_453 = None
    clone_117: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_337, memory_format = torch.contiguous_format);  permute_337 = None
    view_454: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_117, [8, 197, 2304]);  clone_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_455: "f32[1576, 2304]" = torch.ops.aten.view.default(view_454, [1576, 2304]);  view_454 = None
    permute_338: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    mm_80: "f32[1576, 768]" = torch.ops.aten.mm.default(view_455, permute_338);  permute_338 = None
    permute_339: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_455, [1, 0])
    mm_81: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_339, view_37);  permute_339 = view_37 = None
    permute_340: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_173: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_455, [0], True);  view_455 = None
    view_456: "f32[2304]" = torch.ops.aten.view.default(sum_173, [2304]);  sum_173 = None
    permute_341: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
    view_457: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_80, [8, 197, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_30: "f32[768]" = torch.ops.aten.slice.Tensor(view_456, 0, 0, 768)
    slice_32: "f32[768]" = torch.ops.aten.slice.Tensor(view_456, 0, 1536, 2304);  view_456 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_107: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_15);  add_15 = getitem_15 = None
    mul_424: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_107, rsqrt_4);  sub_107 = None
    mul_425: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_457, primals_23);  primals_23 = None
    mul_426: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_425, 768)
    sum_174: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True)
    mul_427: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_425, mul_424);  mul_425 = None
    sum_175: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_427, [2], True);  mul_427 = None
    mul_428: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_424, sum_175);  sum_175 = None
    sub_108: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_426, sum_174);  mul_426 = sum_174 = None
    sub_109: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_108, mul_428);  sub_108 = mul_428 = None
    div_33: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_429: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_33, sub_109);  div_33 = sub_109 = None
    mul_430: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_457, mul_424);  mul_424 = None
    sum_176: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_430, [0, 1]);  mul_430 = None
    sum_177: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_457, [0, 1]);  view_457 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_137: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_136, mul_429);  add_136 = mul_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_431: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_137, primals_19);  primals_19 = None
    mul_432: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_137, clone_16);  clone_16 = None
    sum_178: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_432, [0, 1], True);  mul_432 = None
    view_458: "f32[768]" = torch.ops.aten.view.default(sum_178, [768]);  sum_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_459: "f32[1576, 768]" = torch.ops.aten.view.default(mul_431, [1576, 768]);  mul_431 = None
    permute_342: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    mm_82: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_459, permute_342);  permute_342 = None
    permute_343: "f32[768, 1576]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_83: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_343, view_35);  permute_343 = view_35 = None
    permute_344: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_179: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_459, [0], True);  view_459 = None
    view_460: "f32[768]" = torch.ops.aten.view.default(sum_179, [768]);  sum_179 = None
    permute_345: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_344, [1, 0]);  permute_344 = None
    view_461: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_82, [8, 197, 3072]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_433: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.7071067811865476)
    erf_22: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_433);  mul_433 = None
    add_138: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_434: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_138, 0.5);  add_138 = None
    mul_435: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, view_34)
    mul_436: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_435, -0.5);  mul_435 = None
    exp_22: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_436);  mul_436 = None
    mul_437: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_438: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, mul_437);  view_34 = mul_437 = None
    add_139: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_434, mul_438);  mul_434 = mul_438 = None
    mul_439: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_461, add_139);  view_461 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_462: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_439, [1576, 3072]);  mul_439 = None
    permute_346: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    mm_84: "f32[1576, 768]" = torch.ops.aten.mm.default(view_462, permute_346);  permute_346 = None
    permute_347: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_462, [1, 0])
    mm_85: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_347, view_33);  permute_347 = view_33 = None
    permute_348: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_180: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
    view_463: "f32[3072]" = torch.ops.aten.view.default(sum_180, [3072]);  sum_180 = None
    permute_349: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_348, [1, 0]);  permute_348 = None
    view_464: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_84, [8, 197, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_110: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_13);  add_11 = getitem_13 = None
    mul_440: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_110, rsqrt_3);  sub_110 = None
    mul_441: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_464, primals_20);  primals_20 = None
    mul_442: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_441, 768)
    sum_181: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True)
    mul_443: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_441, mul_440);  mul_441 = None
    sum_182: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_443, [2], True);  mul_443 = None
    mul_444: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_440, sum_182);  sum_182 = None
    sub_111: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_442, sum_181);  mul_442 = sum_181 = None
    sub_112: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_444);  sub_111 = mul_444 = None
    div_34: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_445: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_34, sub_112);  div_34 = sub_112 = None
    mul_446: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_464, mul_440);  mul_440 = None
    sum_183: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_446, [0, 1]);  mul_446 = None
    sum_184: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_464, [0, 1]);  view_464 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_140: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_137, mul_445);  add_137 = mul_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_447: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_140, primals_12);  primals_12 = None
    mul_448: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_140, clone_14);  clone_14 = None
    sum_185: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_448, [0, 1], True);  mul_448 = None
    view_465: "f32[768]" = torch.ops.aten.view.default(sum_185, [768]);  sum_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_466: "f32[1576, 768]" = torch.ops.aten.view.default(mul_447, [1576, 768]);  mul_447 = None
    permute_350: "f32[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    mm_86: "f32[1576, 768]" = torch.ops.aten.mm.default(view_466, permute_350);  permute_350 = None
    permute_351: "f32[768, 1576]" = torch.ops.aten.permute.default(view_466, [1, 0])
    mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_351, view_31);  permute_351 = view_31 = None
    permute_352: "f32[768, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_466, [0], True);  view_466 = None
    view_467: "f32[768]" = torch.ops.aten.view.default(sum_186, [768]);  sum_186 = None
    permute_353: "f32[768, 768]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
    view_468: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_86, [8, 197, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_469: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_468, [8, 197, 12, 64]);  view_468 = None
    permute_354: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_118: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_354, memory_format = torch.contiguous_format);  permute_354 = None
    view_470: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_118, [96, 197, 64]);  clone_118 = None
    permute_355: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    bmm_64: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_355, view_470);  permute_355 = None
    permute_356: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    bmm_65: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_470, permute_356);  view_470 = permute_356 = None
    view_471: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_64, [8, 12, 197, 64]);  bmm_64 = None
    view_472: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_65, [8, 12, 197, 197]);  bmm_65 = None
    alias_22: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_449: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_472, alias_22);  view_472 = None
    sum_187: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_449, [-1], True)
    mul_450: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_22, sum_187);  alias_22 = sum_187 = None
    sub_113: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_449, mul_450);  mul_449 = mul_450 = None
    sum_188: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_113, [0], True)
    view_473: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_113, [96, 197, 197]);  sub_113 = None
    permute_357: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
    bmm_66: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_357, view_473);  permute_357 = None
    permute_358: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    bmm_67: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_473, permute_358);  view_473 = permute_358 = None
    view_474: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_66, [8, 12, 64, 197]);  bmm_66 = None
    view_475: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_67, [8, 12, 197, 64]);  bmm_67 = None
    mul_451: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_474, 0.3535533905932738);  view_474 = None
    permute_359: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_451, [0, 1, 3, 2]);  mul_451 = None
    mul_452: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_475, 0.3535533905932738);  view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_10: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_188, 0);  sum_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_360: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_10, [1, 2, 0]);  squeeze_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_476: "f32[38809, 12]" = torch.ops.aten.view.default(permute_360, [38809, 12]);  permute_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_12: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_10: "f32[732, 12]" = torch.ops.aten.index_put.default(full_12, [view_22], view_476, True);  full_12 = view_22 = view_476 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_23: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_452, permute_359, view_471]);  mul_452 = permute_359 = view_471 = None
    view_477: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_23, [3, 8, 12, 197, 64]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_361: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_477, [1, 3, 0, 2, 4]);  view_477 = None
    clone_119: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_361, memory_format = torch.contiguous_format);  permute_361 = None
    view_478: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_119, [8, 197, 2304]);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_479: "f32[1576, 2304]" = torch.ops.aten.view.default(view_478, [1576, 2304]);  view_478 = None
    permute_362: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_88: "f32[1576, 768]" = torch.ops.aten.mm.default(view_479, permute_362);  permute_362 = None
    permute_363: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_479, [1, 0])
    mm_89: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_363, view_19);  permute_363 = view_19 = None
    permute_364: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_189: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_479, [0], True);  view_479 = None
    view_480: "f32[2304]" = torch.ops.aten.view.default(sum_189, [2304]);  sum_189 = None
    permute_365: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_364, [1, 0]);  permute_364 = None
    view_481: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_88, [8, 197, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_33: "f32[768]" = torch.ops.aten.slice.Tensor(view_480, 0, 0, 768)
    slice_35: "f32[768]" = torch.ops.aten.slice.Tensor(view_480, 0, 1536, 2304);  view_480 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_114: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_8);  add_7 = getitem_8 = None
    mul_453: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_114, rsqrt_2);  sub_114 = None
    mul_454: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_481, primals_13);  primals_13 = None
    mul_455: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_454, 768)
    sum_190: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_454, [2], True)
    mul_456: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_454, mul_453);  mul_454 = None
    sum_191: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_456, [2], True);  mul_456 = None
    mul_457: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_453, sum_191);  sum_191 = None
    sub_115: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_455, sum_190);  mul_455 = sum_190 = None
    sub_116: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_115, mul_457);  sub_115 = mul_457 = None
    div_35: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_458: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_35, sub_116);  div_35 = sub_116 = None
    mul_459: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_481, mul_453);  mul_453 = None
    sum_192: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_459, [0, 1]);  mul_459 = None
    sum_193: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_481, [0, 1]);  view_481 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_141: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_140, mul_458);  add_140 = mul_458 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_460: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_141, primals_9);  primals_9 = None
    mul_461: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_141, clone_8);  clone_8 = None
    sum_194: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_461, [0, 1], True);  mul_461 = None
    view_482: "f32[768]" = torch.ops.aten.view.default(sum_194, [768]);  sum_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_483: "f32[1576, 768]" = torch.ops.aten.view.default(mul_460, [1576, 768]);  mul_460 = None
    permute_366: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_90: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_483, permute_366);  permute_366 = None
    permute_367: "f32[768, 1576]" = torch.ops.aten.permute.default(view_483, [1, 0])
    mm_91: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_367, view_17);  permute_367 = view_17 = None
    permute_368: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_195: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_483, [0], True);  view_483 = None
    view_484: "f32[768]" = torch.ops.aten.view.default(sum_195, [768]);  sum_195 = None
    permute_369: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_368, [1, 0]);  permute_368 = None
    view_485: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_90, [8, 197, 3072]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_462: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.7071067811865476)
    erf_23: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_462);  mul_462 = None
    add_142: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_463: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_142, 0.5);  add_142 = None
    mul_464: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_16, view_16)
    mul_465: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_464, -0.5);  mul_464 = None
    exp_23: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_465);  mul_465 = None
    mul_466: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_467: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_16, mul_466);  view_16 = mul_466 = None
    add_143: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_463, mul_467);  mul_463 = mul_467 = None
    mul_468: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_485, add_143);  view_485 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_486: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_468, [1576, 3072]);  mul_468 = None
    permute_370: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    mm_92: "f32[1576, 768]" = torch.ops.aten.mm.default(view_486, permute_370);  permute_370 = None
    permute_371: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_486, [1, 0])
    mm_93: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_371, view_15);  permute_371 = view_15 = None
    permute_372: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_196: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_486, [0], True);  view_486 = None
    view_487: "f32[3072]" = torch.ops.aten.view.default(sum_196, [3072]);  sum_196 = None
    permute_373: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
    view_488: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_92, [8, 197, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_117: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_6);  add_3 = getitem_6 = None
    mul_469: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_117, rsqrt_1);  sub_117 = None
    mul_470: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_488, primals_10);  primals_10 = None
    mul_471: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_470, 768)
    sum_197: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_470, [2], True)
    mul_472: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_470, mul_469);  mul_470 = None
    sum_198: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [2], True);  mul_472 = None
    mul_473: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_469, sum_198);  sum_198 = None
    sub_118: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_471, sum_197);  mul_471 = sum_197 = None
    sub_119: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_118, mul_473);  sub_118 = mul_473 = None
    div_36: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_474: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_36, sub_119);  div_36 = sub_119 = None
    mul_475: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_488, mul_469);  mul_469 = None
    sum_199: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_475, [0, 1]);  mul_475 = None
    sum_200: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_488, [0, 1]);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_144: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_141, mul_474);  add_141 = mul_474 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_476: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_144, primals_2);  primals_2 = None
    mul_477: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(add_144, clone_6);  clone_6 = None
    sum_201: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_477, [0, 1], True);  mul_477 = None
    view_489: "f32[768]" = torch.ops.aten.view.default(sum_201, [768]);  sum_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_490: "f32[1576, 768]" = torch.ops.aten.view.default(mul_476, [1576, 768]);  mul_476 = None
    permute_374: "f32[768, 768]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    mm_94: "f32[1576, 768]" = torch.ops.aten.mm.default(view_490, permute_374);  permute_374 = None
    permute_375: "f32[768, 1576]" = torch.ops.aten.permute.default(view_490, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_375, view_13);  permute_375 = view_13 = None
    permute_376: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_202: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
    view_491: "f32[768]" = torch.ops.aten.view.default(sum_202, [768]);  sum_202 = None
    permute_377: "f32[768, 768]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
    view_492: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_94, [8, 197, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_493: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_492, [8, 197, 12, 64]);  view_492 = None
    permute_378: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    clone_120: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(permute_378, memory_format = torch.contiguous_format);  permute_378 = None
    view_494: "f32[96, 197, 64]" = torch.ops.aten.view.default(clone_120, [96, 197, 64]);  clone_120 = None
    permute_379: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_68: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(permute_379, view_494);  permute_379 = None
    permute_380: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_69: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_494, permute_380);  view_494 = permute_380 = None
    view_495: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_68, [8, 12, 197, 64]);  bmm_68 = None
    view_496: "f32[8, 12, 197, 197]" = torch.ops.aten.view.default(bmm_69, [8, 12, 197, 197]);  bmm_69 = None
    alias_23: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_478: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(view_496, alias_23);  view_496 = None
    sum_203: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_478, [-1], True)
    mul_479: "f32[8, 12, 197, 197]" = torch.ops.aten.mul.Tensor(alias_23, sum_203);  alias_23 = sum_203 = None
    sub_120: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(mul_478, mul_479);  mul_478 = mul_479 = None
    sum_204: "f32[1, 12, 197, 197]" = torch.ops.aten.sum.dim_IntList(sub_120, [0], True)
    view_497: "f32[96, 197, 197]" = torch.ops.aten.view.default(sub_120, [96, 197, 197]);  sub_120 = None
    permute_381: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
    bmm_70: "f32[96, 64, 197]" = torch.ops.aten.bmm.default(permute_381, view_497);  permute_381 = None
    permute_382: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    bmm_71: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_497, permute_382);  view_497 = permute_382 = None
    view_498: "f32[8, 12, 64, 197]" = torch.ops.aten.view.default(bmm_70, [8, 12, 64, 197]);  bmm_70 = None
    view_499: "f32[8, 12, 197, 64]" = torch.ops.aten.view.default(bmm_71, [8, 12, 197, 64]);  bmm_71 = None
    mul_480: "f32[8, 12, 64, 197]" = torch.ops.aten.mul.Scalar(view_498, 0.3535533905932738);  view_498 = None
    permute_383: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(mul_480, [0, 1, 3, 2]);  mul_480 = None
    mul_481: "f32[8, 12, 197, 64]" = torch.ops.aten.mul.Scalar(view_499, 0.3535533905932738);  view_499 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    squeeze_11: "f32[12, 197, 197]" = torch.ops.aten.squeeze.dim(sum_204, 0);  sum_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_384: "f32[197, 197, 12]" = torch.ops.aten.permute.default(squeeze_11, [1, 2, 0]);  squeeze_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_500: "f32[38809, 12]" = torch.ops.aten.view.default(permute_384, [38809, 12]);  permute_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    full_13: "f32[732, 12]" = torch.ops.aten.full.default([732, 12], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    index_put_11: "f32[732, 12]" = torch.ops.aten.index_put.default(full_13, [view_4], view_500, True);  full_13 = view_4 = view_500 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    cat_24: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([mul_481, permute_383, view_495]);  mul_481 = permute_383 = view_495 = None
    view_501: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_24, [3, 8, 12, 197, 64]);  cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    permute_385: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_501, [1, 3, 0, 2, 4]);  view_501 = None
    clone_121: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None
    view_502: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_121, [8, 197, 2304]);  clone_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_503: "f32[1576, 2304]" = torch.ops.aten.view.default(view_502, [1576, 2304]);  view_502 = None
    permute_386: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_96: "f32[1576, 768]" = torch.ops.aten.mm.default(view_503, permute_386);  permute_386 = None
    permute_387: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_503, [1, 0])
    mm_97: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_387, view_1);  permute_387 = view_1 = None
    permute_388: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_205: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
    view_504: "f32[2304]" = torch.ops.aten.view.default(sum_205, [2304]);  sum_205 = None
    permute_389: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_388, [1, 0]);  permute_388 = None
    view_505: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_96, [8, 197, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    slice_36: "f32[768]" = torch.ops.aten.slice.Tensor(view_504, 0, 0, 768)
    slice_38: "f32[768]" = torch.ops.aten.slice.Tensor(view_504, 0, 1536, 2304);  view_504 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_121: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul_482: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_121, rsqrt);  sub_121 = None
    mul_483: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_505, primals_3);  primals_3 = None
    mul_484: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_483, 768)
    sum_206: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_483, [2], True)
    mul_485: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_483, mul_482);  mul_483 = None
    sum_207: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True);  mul_485 = None
    mul_486: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_482, sum_207);  sum_207 = None
    sub_122: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_484, sum_206);  mul_484 = sum_206 = None
    sub_123: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_122, mul_486);  sub_122 = mul_486 = None
    div_37: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_487: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_37, sub_123);  div_37 = sub_123 = None
    mul_488: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_505, mul_482);  mul_482 = None
    sum_208: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_488, [0, 1]);  mul_488 = None
    sum_209: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_505, [0, 1]);  view_505 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_145: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_144, mul_487);  add_144 = mul_487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:405, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    slice_39: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_145, 1, 0, 1)
    slice_40: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(add_145, 1, 1, 197);  add_145 = None
    sum_210: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_39, [0], True);  slice_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_390: "f32[8, 768, 196]" = torch.ops.aten.permute.default(slice_40, [0, 2, 1]);  slice_40 = None
    view_506: "f32[8, 768, 14, 14]" = torch.ops.aten.view.default(permute_390, [8, 768, 14, 14]);  permute_390 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_506, primals_224, primals_124, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_506 = primals_224 = primals_124 = None
    getitem_87: "f32[768, 3, 16, 16]" = convolution_backward[1]
    getitem_88: "f32[768]" = convolution_backward[2];  convolution_backward = None
    return pytree.tree_unflatten([addmm_48, sum_210, view_489, sum_208, sum_209, slice_36, slice_38, permute_389, index_put_11, view_482, sum_199, sum_200, view_465, sum_192, sum_193, slice_33, slice_35, permute_365, index_put_10, view_458, sum_183, sum_184, view_441, sum_176, sum_177, slice_30, slice_32, permute_341, index_put_9, view_434, sum_167, sum_168, view_417, sum_160, sum_161, slice_27, slice_29, permute_317, index_put_8, view_410, sum_151, sum_152, view_393, sum_144, sum_145, slice_24, slice_26, permute_293, index_put_7, view_386, sum_135, sum_136, view_369, sum_128, sum_129, slice_21, slice_23, permute_269, index_put_6, view_362, sum_119, sum_120, view_345, sum_112, sum_113, slice_18, slice_20, permute_245, index_put_5, view_338, sum_103, sum_104, view_321, sum_96, sum_97, slice_15, slice_17, permute_221, index_put_4, view_314, sum_87, sum_88, view_297, sum_80, sum_81, slice_12, slice_14, permute_197, index_put_3, view_290, sum_71, sum_72, view_273, sum_64, sum_65, slice_9, slice_11, permute_173, index_put_2, view_266, sum_55, sum_56, view_249, sum_48, sum_49, slice_6, slice_8, permute_149, index_put_1, view_242, sum_39, sum_40, view_225, sum_32, sum_33, slice_3, slice_5, permute_125, index_put, view_218, sum_23, sum_24, sum_16, sum_17, getitem_87, getitem_88, permute_377, view_491, permute_373, view_487, permute_369, view_484, permute_353, view_467, permute_349, view_463, permute_345, view_460, permute_329, view_443, permute_325, view_439, permute_321, view_436, permute_305, view_419, permute_301, view_415, permute_297, view_412, permute_281, view_395, permute_277, view_391, permute_273, view_388, permute_257, view_371, permute_253, view_367, permute_249, view_364, permute_233, view_347, permute_229, view_343, permute_225, view_340, permute_209, view_323, permute_205, view_319, permute_201, view_316, permute_185, view_299, permute_181, view_295, permute_177, view_292, permute_161, view_275, permute_157, view_271, permute_153, view_268, permute_137, view_251, permute_133, view_247, permute_129, view_244, permute_113, view_227, permute_109, view_223, permute_105, view_220, permute_101, view_217, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None], self._out_spec)
    