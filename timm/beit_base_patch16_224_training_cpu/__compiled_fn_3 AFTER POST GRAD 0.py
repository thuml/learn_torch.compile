from __future__ import annotations



def forward(self, primals_1: "f32[1, 1, 768]", primals_2: "f32[768]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[768]", primals_7: "f32[2304, 768]", primals_8: "f32[732, 12]", primals_9: "f32[768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[2304, 768]", primals_18: "f32[732, 12]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[768]", primals_27: "f32[2304, 768]", primals_28: "f32[732, 12]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[2304, 768]", primals_38: "f32[732, 12]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[768]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[768]", primals_46: "f32[768]", primals_47: "f32[2304, 768]", primals_48: "f32[732, 12]", primals_49: "f32[768]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[2304, 768]", primals_58: "f32[732, 12]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[2304, 768]", primals_68: "f32[732, 12]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[768]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[2304, 768]", primals_78: "f32[732, 12]", primals_79: "f32[768]", primals_80: "f32[768]", primals_81: "f32[768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[768]", primals_87: "f32[2304, 768]", primals_88: "f32[732, 12]", primals_89: "f32[768]", primals_90: "f32[768]", primals_91: "f32[768]", primals_92: "f32[768]", primals_93: "f32[768]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[2304, 768]", primals_98: "f32[732, 12]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[768]", primals_103: "f32[768]", primals_104: "f32[768]", primals_105: "f32[768]", primals_106: "f32[768]", primals_107: "f32[2304, 768]", primals_108: "f32[732, 12]", primals_109: "f32[768]", primals_110: "f32[768]", primals_111: "f32[768]", primals_112: "f32[768]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[768]", primals_116: "f32[768]", primals_117: "f32[2304, 768]", primals_118: "f32[732, 12]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[768]", primals_122: "f32[768]", primals_123: "f32[768]", primals_124: "f32[768, 3, 16, 16]", primals_125: "f32[768]", primals_126: "f32[768, 768]", primals_127: "f32[768]", primals_128: "f32[3072, 768]", primals_129: "f32[3072]", primals_130: "f32[768, 3072]", primals_131: "f32[768]", primals_132: "f32[768, 768]", primals_133: "f32[768]", primals_134: "f32[3072, 768]", primals_135: "f32[3072]", primals_136: "f32[768, 3072]", primals_137: "f32[768]", primals_138: "f32[768, 768]", primals_139: "f32[768]", primals_140: "f32[3072, 768]", primals_141: "f32[3072]", primals_142: "f32[768, 3072]", primals_143: "f32[768]", primals_144: "f32[768, 768]", primals_145: "f32[768]", primals_146: "f32[3072, 768]", primals_147: "f32[3072]", primals_148: "f32[768, 3072]", primals_149: "f32[768]", primals_150: "f32[768, 768]", primals_151: "f32[768]", primals_152: "f32[3072, 768]", primals_153: "f32[3072]", primals_154: "f32[768, 3072]", primals_155: "f32[768]", primals_156: "f32[768, 768]", primals_157: "f32[768]", primals_158: "f32[3072, 768]", primals_159: "f32[3072]", primals_160: "f32[768, 3072]", primals_161: "f32[768]", primals_162: "f32[768, 768]", primals_163: "f32[768]", primals_164: "f32[3072, 768]", primals_165: "f32[3072]", primals_166: "f32[768, 3072]", primals_167: "f32[768]", primals_168: "f32[768, 768]", primals_169: "f32[768]", primals_170: "f32[3072, 768]", primals_171: "f32[3072]", primals_172: "f32[768, 3072]", primals_173: "f32[768]", primals_174: "f32[768, 768]", primals_175: "f32[768]", primals_176: "f32[3072, 768]", primals_177: "f32[3072]", primals_178: "f32[768, 3072]", primals_179: "f32[768]", primals_180: "f32[768, 768]", primals_181: "f32[768]", primals_182: "f32[3072, 768]", primals_183: "f32[3072]", primals_184: "f32[768, 3072]", primals_185: "f32[768]", primals_186: "f32[768, 768]", primals_187: "f32[768]", primals_188: "f32[3072, 768]", primals_189: "f32[3072]", primals_190: "f32[768, 3072]", primals_191: "f32[768]", primals_192: "f32[768, 768]", primals_193: "f32[768]", primals_194: "f32[3072, 768]", primals_195: "f32[3072]", primals_196: "f32[768, 3072]", primals_197: "f32[768]", primals_198: "f32[1000, 768]", primals_199: "f32[1000]", primals_200: "f32[768]", primals_201: "i64[197, 197]", primals_202: "f32[768]", primals_203: "i64[197, 197]", primals_204: "f32[768]", primals_205: "i64[197, 197]", primals_206: "f32[768]", primals_207: "i64[197, 197]", primals_208: "f32[768]", primals_209: "i64[197, 197]", primals_210: "f32[768]", primals_211: "i64[197, 197]", primals_212: "f32[768]", primals_213: "i64[197, 197]", primals_214: "f32[768]", primals_215: "i64[197, 197]", primals_216: "f32[768]", primals_217: "i64[197, 197]", primals_218: "f32[768]", primals_219: "i64[197, 197]", primals_220: "f32[768]", primals_221: "i64[197, 197]", primals_222: "f32[768]", primals_223: "i64[197, 197]", primals_224: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(primals_224, primals_124, primals_125, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 196]" = torch.ops.aten.reshape.default(convolution, [8, 768, 196]);  convolution = None
    permute: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:405, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    expand: "f32[8, 1, 768]" = torch.ops.aten.expand.default(primals_1, [8, -1, -1]);  primals_1 = None
    cat: "f32[8, 197, 768]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 197, 1]" = var_mean[0]
    getitem_1: "f32[8, 197, 1]" = var_mean[1];  var_mean = None
    add: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    sub: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(cat, getitem_1)
    mul: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul, primals_3);  mul = None
    add_1: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_4);  mul_1 = primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_1: "f32[2304]" = torch.ops.aten.cat.default([primals_5, primals_200, primals_6]);  primals_5 = primals_200 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_1: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_1, [1576, 768]);  add_1 = None
    permute_1: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_1, view_1, permute_1);  cat_1 = None
    view_2: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(addmm, [8, 197, 2304]);  addmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_3: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(view_2, [8, 197, 3, 12, -1]);  view_2 = None
    permute_2: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
    getitem_2: "f32[8, 12, 197, 64]" = unbind[0]
    getitem_3: "f32[8, 12, 197, 64]" = unbind[1]
    getitem_4: "f32[8, 12, 197, 64]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_4: "i64[38809]" = torch.ops.aten.reshape.default(primals_201, [-1]);  primals_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_8, [view_4]);  primals_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_5: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index, [197, 197, -1]);  index = None
    
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
    view_6: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_2, [96, 197, 64]);  clone_2 = None
    expand_2: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_3, [8, 12, 64, 197]);  mul_3 = None
    clone_3: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
    view_7: "f32[96, 64, 197]" = torch.ops.aten.reshape.default(clone_3, [96, 64, 197]);  clone_3 = None
    bmm: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_6, view_7)
    view_8: "f32[8, 12, 197, 197]" = torch.ops.aten.reshape.default(bmm, [8, 12, 197, 197]);  bmm = None
    add_2: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_8, unsqueeze);  view_8 = unsqueeze = None
    amax: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_2, [-1], True)
    sub_1: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_2, amax);  add_2 = amax = None
    exp: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div)
    expand_3: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div, [8, 12, 197, 197]);  div = None
    view_9: "f32[96, 197, 197]" = torch.ops.aten.reshape.default(expand_3, [96, 197, 197]);  expand_3 = None
    expand_4: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_4, [8, 12, 197, 64]);  getitem_4 = None
    clone_4: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_10: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_4, [96, 197, 64]);  clone_4 = None
    bmm_1: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[8, 12, 197, 64]" = torch.ops.aten.reshape.default(bmm_1, [8, 12, 197, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_5: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
    clone_5: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
    view_12: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_5, [8, 197, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_13: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_12, [1576, 768]);  view_12 = None
    permute_6: "f32[768, 768]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    addmm_1: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_127, view_13, permute_6);  primals_127 = None
    view_14: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_1, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_4: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_2, view_14);  view_14 = None
    add_3: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(cat, mul_4);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_5: "f32[8, 197, 1]" = var_mean_1[0]
    getitem_6: "f32[8, 197, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_5, 1e-06);  getitem_5 = None
    rsqrt_1: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_2: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_6);  getitem_6 = None
    mul_5: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
    mul_6: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_5, primals_10)
    add_5: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_6, primals_11);  mul_6 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_15: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_5, [1576, 768]);  add_5 = None
    permute_7: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_2: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_129, view_15, permute_7);  primals_129 = None
    view_16: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_2, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_7: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.5)
    mul_8: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_16, 0.7071067811865476);  view_16 = None
    erf: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_6: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_9: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_6);  mul_7 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_17: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_9, [1576, 3072]);  mul_9 = None
    permute_8: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_3: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_131, view_17, permute_8);  primals_131 = None
    view_18: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_3, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_10: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_9, view_18);  view_18 = None
    add_7: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_3, mul_10);  add_3 = mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_7: "f32[8, 197, 1]" = var_mean_2[0]
    getitem_8: "f32[8, 197, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_7, 1e-06);  getitem_7 = None
    rsqrt_2: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_3: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_8);  getitem_8 = None
    mul_11: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = None
    mul_12: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_11, primals_13)
    add_9: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_12, primals_14);  mul_12 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_2: "f32[2304]" = torch.ops.aten.cat.default([primals_15, primals_202, primals_16]);  primals_15 = primals_202 = primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_19: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_9, [1576, 768]);  add_9 = None
    permute_9: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_4: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_2, view_19, permute_9);  cat_2 = None
    view_20: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(addmm_4, [8, 197, 2304]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_21: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(view_20, [8, 197, 3, 12, -1]);  view_20 = None
    permute_10: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_21, [2, 0, 3, 1, 4]);  view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_1 = torch.ops.aten.unbind.int(permute_10);  permute_10 = None
    getitem_9: "f32[8, 12, 197, 64]" = unbind_1[0]
    getitem_10: "f32[8, 12, 197, 64]" = unbind_1[1]
    getitem_11: "f32[8, 12, 197, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_22: "i64[38809]" = torch.ops.aten.reshape.default(primals_203, [-1]);  primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_1: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_18, [view_22]);  primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_23: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_1, [197, 197, -1]);  index_1 = None
    
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
    view_24: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_10, [96, 197, 64]);  clone_10 = None
    expand_6: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_14, [8, 12, 64, 197]);  mul_14 = None
    clone_11: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
    view_25: "f32[96, 64, 197]" = torch.ops.aten.reshape.default(clone_11, [96, 64, 197]);  clone_11 = None
    bmm_2: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_24, view_25)
    view_26: "f32[8, 12, 197, 197]" = torch.ops.aten.reshape.default(bmm_2, [8, 12, 197, 197]);  bmm_2 = None
    add_10: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_26, unsqueeze_1);  view_26 = unsqueeze_1 = None
    amax_1: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_10, [-1], True)
    sub_4: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_10, amax_1);  add_10 = amax_1 = None
    exp_1: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_1: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_1: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_1)
    expand_7: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_1, [8, 12, 197, 197]);  div_1 = None
    view_27: "f32[96, 197, 197]" = torch.ops.aten.reshape.default(expand_7, [96, 197, 197]);  expand_7 = None
    expand_8: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_11, [8, 12, 197, 64]);  getitem_11 = None
    clone_12: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_28: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_12, [96, 197, 64]);  clone_12 = None
    bmm_3: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_27, view_28)
    view_29: "f32[8, 12, 197, 64]" = torch.ops.aten.reshape.default(bmm_3, [8, 12, 197, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_13: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    clone_13: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_13, memory_format = torch.contiguous_format);  permute_13 = None
    view_30: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_13, [8, 197, 768]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_31: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_30, [1576, 768]);  view_30 = None
    permute_14: "f32[768, 768]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_5: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_133, view_31, permute_14);  primals_133 = None
    view_32: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_5, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_15: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_12, view_32);  view_32 = None
    add_11: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_7, mul_15);  add_7 = mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_11, [2], correction = 0, keepdim = True)
    getitem_12: "f32[8, 197, 1]" = var_mean_3[0]
    getitem_13: "f32[8, 197, 1]" = var_mean_3[1];  var_mean_3 = None
    add_12: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_12, 1e-06);  getitem_12 = None
    rsqrt_3: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_12);  add_12 = None
    sub_5: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_11, getitem_13);  getitem_13 = None
    mul_16: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
    mul_17: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_20)
    add_13: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_21);  mul_17 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_33: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_13, [1576, 768]);  add_13 = None
    permute_15: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_6: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_135, view_33, permute_15);  primals_135 = None
    view_34: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_6, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
    mul_19: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.7071067811865476);  view_34 = None
    erf_1: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_14: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_20: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_14);  mul_18 = add_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_35: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_20, [1576, 3072]);  mul_20 = None
    permute_16: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_7: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_137, view_35, permute_16);  primals_137 = None
    view_36: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_7, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_21: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_19, view_36);  view_36 = None
    add_15: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_11, mul_21);  add_11 = mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 197, 1]" = var_mean_4[0]
    getitem_15: "f32[8, 197, 1]" = var_mean_4[1];  var_mean_4 = None
    add_16: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_4: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
    sub_6: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_15, getitem_15);  getitem_15 = None
    mul_22: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = None
    mul_23: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_22, primals_23)
    add_17: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_23, primals_24);  mul_23 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_3: "f32[2304]" = torch.ops.aten.cat.default([primals_25, primals_204, primals_26]);  primals_25 = primals_204 = primals_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_37: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_17, [1576, 768]);  add_17 = None
    permute_17: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm_8: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_3, view_37, permute_17);  cat_3 = None
    view_38: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(addmm_8, [8, 197, 2304]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_39: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(view_38, [8, 197, 3, 12, -1]);  view_38 = None
    permute_18: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_39, [2, 0, 3, 1, 4]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_2 = torch.ops.aten.unbind.int(permute_18);  permute_18 = None
    getitem_16: "f32[8, 12, 197, 64]" = unbind_2[0]
    getitem_17: "f32[8, 12, 197, 64]" = unbind_2[1]
    getitem_18: "f32[8, 12, 197, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_40: "i64[38809]" = torch.ops.aten.reshape.default(primals_205, [-1]);  primals_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_2: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_28, [view_40]);  primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_41: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_2, [197, 197, -1]);  index_2 = None
    
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
    view_42: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_18, [96, 197, 64]);  clone_18 = None
    expand_10: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_25, [8, 12, 64, 197]);  mul_25 = None
    clone_19: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_10, memory_format = torch.contiguous_format);  expand_10 = None
    view_43: "f32[96, 64, 197]" = torch.ops.aten.reshape.default(clone_19, [96, 64, 197]);  clone_19 = None
    bmm_4: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_42, view_43)
    view_44: "f32[8, 12, 197, 197]" = torch.ops.aten.reshape.default(bmm_4, [8, 12, 197, 197]);  bmm_4 = None
    add_18: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_44, unsqueeze_2);  view_44 = unsqueeze_2 = None
    amax_2: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_18, [-1], True)
    sub_7: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_18, amax_2);  add_18 = amax_2 = None
    exp_2: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_2: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_2: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_2)
    expand_11: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_2, [8, 12, 197, 197]);  div_2 = None
    view_45: "f32[96, 197, 197]" = torch.ops.aten.reshape.default(expand_11, [96, 197, 197]);  expand_11 = None
    expand_12: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_18, [8, 12, 197, 64]);  getitem_18 = None
    clone_20: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_46: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_20, [96, 197, 64]);  clone_20 = None
    bmm_5: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_45, view_46)
    view_47: "f32[8, 12, 197, 64]" = torch.ops.aten.reshape.default(bmm_5, [8, 12, 197, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_21: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_47, [0, 2, 1, 3]);  view_47 = None
    clone_21: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_21, memory_format = torch.contiguous_format);  permute_21 = None
    view_48: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_21, [8, 197, 768]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_49: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_48, [1576, 768]);  view_48 = None
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_9: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_139, view_49, permute_22);  primals_139 = None
    view_50: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_9, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_26: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_22, view_50);  view_50 = None
    add_19: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_15, mul_26);  add_15 = mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
    getitem_19: "f32[8, 197, 1]" = var_mean_5[0]
    getitem_20: "f32[8, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    add_20: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_19, 1e-06);  getitem_19 = None
    rsqrt_5: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
    sub_8: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_19, getitem_20);  getitem_20 = None
    mul_27: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
    mul_28: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_27, primals_30)
    add_21: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_28, primals_31);  mul_28 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_51: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_21, [1576, 768]);  add_21 = None
    permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_10: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_141, view_51, permute_23);  primals_141 = None
    view_52: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_10, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_29: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_52, 0.5)
    mul_30: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_52, 0.7071067811865476);  view_52 = None
    erf_2: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_30);  mul_30 = None
    add_22: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_31: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_29, add_22);  mul_29 = add_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_53: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_31, [1576, 3072]);  mul_31 = None
    permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    addmm_11: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_143, view_53, permute_24);  primals_143 = None
    view_54: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_11, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_32: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_29, view_54);  view_54 = None
    add_23: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_19, mul_32);  add_19 = mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_21: "f32[8, 197, 1]" = var_mean_6[0]
    getitem_22: "f32[8, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    add_24: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_21, 1e-06);  getitem_21 = None
    rsqrt_6: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_9: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_22);  getitem_22 = None
    mul_33: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_6);  sub_9 = None
    mul_34: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_33, primals_33)
    add_25: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_34, primals_34);  mul_34 = primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_4: "f32[2304]" = torch.ops.aten.cat.default([primals_35, primals_206, primals_36]);  primals_35 = primals_206 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_55: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_25, [1576, 768]);  add_25 = None
    permute_25: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_12: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_4, view_55, permute_25);  cat_4 = None
    view_56: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(addmm_12, [8, 197, 2304]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_57: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(view_56, [8, 197, 3, 12, -1]);  view_56 = None
    permute_26: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_57, [2, 0, 3, 1, 4]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_3 = torch.ops.aten.unbind.int(permute_26);  permute_26 = None
    getitem_23: "f32[8, 12, 197, 64]" = unbind_3[0]
    getitem_24: "f32[8, 12, 197, 64]" = unbind_3[1]
    getitem_25: "f32[8, 12, 197, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_58: "i64[38809]" = torch.ops.aten.reshape.default(primals_207, [-1]);  primals_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_3: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_38, [view_58]);  primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_59: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_3, [197, 197, -1]);  index_3 = None
    
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
    view_60: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_26, [96, 197, 64]);  clone_26 = None
    expand_14: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_36, [8, 12, 64, 197]);  mul_36 = None
    clone_27: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_14, memory_format = torch.contiguous_format);  expand_14 = None
    view_61: "f32[96, 64, 197]" = torch.ops.aten.reshape.default(clone_27, [96, 64, 197]);  clone_27 = None
    bmm_6: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_60, view_61)
    view_62: "f32[8, 12, 197, 197]" = torch.ops.aten.reshape.default(bmm_6, [8, 12, 197, 197]);  bmm_6 = None
    add_26: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_62, unsqueeze_3);  view_62 = unsqueeze_3 = None
    amax_3: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_26, [-1], True)
    sub_10: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_26, amax_3);  add_26 = amax_3 = None
    exp_3: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_3: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_3: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_3)
    expand_15: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_3, [8, 12, 197, 197]);  div_3 = None
    view_63: "f32[96, 197, 197]" = torch.ops.aten.reshape.default(expand_15, [96, 197, 197]);  expand_15 = None
    expand_16: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_25, [8, 12, 197, 64]);  getitem_25 = None
    clone_28: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_64: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_28, [96, 197, 64]);  clone_28 = None
    bmm_7: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_63, view_64)
    view_65: "f32[8, 12, 197, 64]" = torch.ops.aten.reshape.default(bmm_7, [8, 12, 197, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_29: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_65, [0, 2, 1, 3]);  view_65 = None
    clone_29: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_66: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_29, [8, 197, 768]);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_67: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_66, [1576, 768]);  view_66 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_13: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_145, view_67, permute_30);  primals_145 = None
    view_68: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_13, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_37: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_32, view_68);  view_68 = None
    add_27: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_23, mul_37);  add_23 = mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_26: "f32[8, 197, 1]" = var_mean_7[0]
    getitem_27: "f32[8, 197, 1]" = var_mean_7[1];  var_mean_7 = None
    add_28: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-06);  getitem_26 = None
    rsqrt_7: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_11: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_27);  getitem_27 = None
    mul_38: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
    mul_39: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_38, primals_40)
    add_29: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_39, primals_41);  mul_39 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_69: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_29, [1576, 768]);  add_29 = None
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_14: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_147, view_69, permute_31);  primals_147 = None
    view_70: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_14, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_40: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
    mul_41: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.7071067811865476);  view_70 = None
    erf_3: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_41);  mul_41 = None
    add_30: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_42: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_40, add_30);  mul_40 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_71: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_42, [1576, 3072]);  mul_42 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_15: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_149, view_71, permute_32);  primals_149 = None
    view_72: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_15, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_43: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_39, view_72);  view_72 = None
    add_31: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_27, mul_43);  add_27 = mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_28: "f32[8, 197, 1]" = var_mean_8[0]
    getitem_29: "f32[8, 197, 1]" = var_mean_8[1];  var_mean_8 = None
    add_32: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-06);  getitem_28 = None
    rsqrt_8: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_12: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_29);  getitem_29 = None
    mul_44: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_8);  sub_12 = None
    mul_45: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_43)
    add_33: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_44);  mul_45 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_5: "f32[2304]" = torch.ops.aten.cat.default([primals_45, primals_208, primals_46]);  primals_45 = primals_208 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_73: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_33, [1576, 768]);  add_33 = None
    permute_33: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_16: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_5, view_73, permute_33);  cat_5 = None
    view_74: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(addmm_16, [8, 197, 2304]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_75: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(view_74, [8, 197, 3, 12, -1]);  view_74 = None
    permute_34: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_75, [2, 0, 3, 1, 4]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_4 = torch.ops.aten.unbind.int(permute_34);  permute_34 = None
    getitem_30: "f32[8, 12, 197, 64]" = unbind_4[0]
    getitem_31: "f32[8, 12, 197, 64]" = unbind_4[1]
    getitem_32: "f32[8, 12, 197, 64]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_76: "i64[38809]" = torch.ops.aten.reshape.default(primals_209, [-1]);  primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_4: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_48, [view_76]);  primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_77: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_4, [197, 197, -1]);  index_4 = None
    
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
    view_78: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_34, [96, 197, 64]);  clone_34 = None
    expand_18: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_47, [8, 12, 64, 197]);  mul_47 = None
    clone_35: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
    view_79: "f32[96, 64, 197]" = torch.ops.aten.reshape.default(clone_35, [96, 64, 197]);  clone_35 = None
    bmm_8: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[8, 12, 197, 197]" = torch.ops.aten.reshape.default(bmm_8, [8, 12, 197, 197]);  bmm_8 = None
    add_34: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_80, unsqueeze_4);  view_80 = unsqueeze_4 = None
    amax_4: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_34, [-1], True)
    sub_13: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_34, amax_4);  add_34 = amax_4 = None
    exp_4: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_4: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_4: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_4)
    expand_19: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_4, [8, 12, 197, 197]);  div_4 = None
    view_81: "f32[96, 197, 197]" = torch.ops.aten.reshape.default(expand_19, [96, 197, 197]);  expand_19 = None
    expand_20: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_32, [8, 12, 197, 64]);  getitem_32 = None
    clone_36: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_82: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_36, [96, 197, 64]);  clone_36 = None
    bmm_9: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_81, view_82)
    view_83: "f32[8, 12, 197, 64]" = torch.ops.aten.reshape.default(bmm_9, [8, 12, 197, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_37: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
    clone_37: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None
    view_84: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_37, [8, 197, 768]);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_85: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_84, [1576, 768]);  view_84 = None
    permute_38: "f32[768, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_17: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_151, view_85, permute_38);  primals_151 = None
    view_86: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_17, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_48: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_42, view_86);  view_86 = None
    add_35: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_31, mul_48);  add_31 = mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_33: "f32[8, 197, 1]" = var_mean_9[0]
    getitem_34: "f32[8, 197, 1]" = var_mean_9[1];  var_mean_9 = None
    add_36: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
    rsqrt_9: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_14: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_34);  getitem_34 = None
    mul_49: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
    mul_50: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_50)
    add_37: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_51);  mul_50 = primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_87: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_37, [1576, 768]);  add_37 = None
    permute_39: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_18: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_153, view_87, permute_39);  primals_153 = None
    view_88: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_18, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_51: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.5)
    mul_52: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476);  view_88 = None
    erf_4: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
    add_38: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_53: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_51, add_38);  mul_51 = add_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_89: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_53, [1576, 3072]);  mul_53 = None
    permute_40: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_19: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_155, view_89, permute_40);  primals_155 = None
    view_90: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_19, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_54: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_49, view_90);  view_90 = None
    add_39: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_35, mul_54);  add_35 = mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_35: "f32[8, 197, 1]" = var_mean_10[0]
    getitem_36: "f32[8, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    add_40: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_35, 1e-06);  getitem_35 = None
    rsqrt_10: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_15: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_39, getitem_36);  getitem_36 = None
    mul_55: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_10);  sub_15 = None
    mul_56: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_55, primals_53)
    add_41: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_56, primals_54);  mul_56 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_6: "f32[2304]" = torch.ops.aten.cat.default([primals_55, primals_210, primals_56]);  primals_55 = primals_210 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_91: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_41, [1576, 768]);  add_41 = None
    permute_41: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    addmm_20: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_6, view_91, permute_41);  cat_6 = None
    view_92: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(addmm_20, [8, 197, 2304]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_93: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(view_92, [8, 197, 3, 12, -1]);  view_92 = None
    permute_42: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_93, [2, 0, 3, 1, 4]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_5 = torch.ops.aten.unbind.int(permute_42);  permute_42 = None
    getitem_37: "f32[8, 12, 197, 64]" = unbind_5[0]
    getitem_38: "f32[8, 12, 197, 64]" = unbind_5[1]
    getitem_39: "f32[8, 12, 197, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_94: "i64[38809]" = torch.ops.aten.reshape.default(primals_211, [-1]);  primals_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_5: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_58, [view_94]);  primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_95: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_5, [197, 197, -1]);  index_5 = None
    
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
    view_96: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_42, [96, 197, 64]);  clone_42 = None
    expand_22: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_58, [8, 12, 64, 197]);  mul_58 = None
    clone_43: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
    view_97: "f32[96, 64, 197]" = torch.ops.aten.reshape.default(clone_43, [96, 64, 197]);  clone_43 = None
    bmm_10: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_96, view_97)
    view_98: "f32[8, 12, 197, 197]" = torch.ops.aten.reshape.default(bmm_10, [8, 12, 197, 197]);  bmm_10 = None
    add_42: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_98, unsqueeze_5);  view_98 = unsqueeze_5 = None
    amax_5: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_42, [-1], True)
    sub_16: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_42, amax_5);  add_42 = amax_5 = None
    exp_5: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_5: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_5: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_5)
    expand_23: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_5, [8, 12, 197, 197]);  div_5 = None
    view_99: "f32[96, 197, 197]" = torch.ops.aten.reshape.default(expand_23, [96, 197, 197]);  expand_23 = None
    expand_24: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_39, [8, 12, 197, 64]);  getitem_39 = None
    clone_44: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_100: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_44, [96, 197, 64]);  clone_44 = None
    bmm_11: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_99, view_100)
    view_101: "f32[8, 12, 197, 64]" = torch.ops.aten.reshape.default(bmm_11, [8, 12, 197, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_45: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_101, [0, 2, 1, 3]);  view_101 = None
    clone_45: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    view_102: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_45, [8, 197, 768]);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_103: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_102, [1576, 768]);  view_102 = None
    permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_21: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_157, view_103, permute_46);  primals_157 = None
    view_104: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_21, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_59: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_52, view_104);  view_104 = None
    add_43: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_39, mul_59);  add_39 = mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_40: "f32[8, 197, 1]" = var_mean_11[0]
    getitem_41: "f32[8, 197, 1]" = var_mean_11[1];  var_mean_11 = None
    add_44: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-06);  getitem_40 = None
    rsqrt_11: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_17: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_43, getitem_41);  getitem_41 = None
    mul_60: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
    mul_61: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_60, primals_60)
    add_45: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_61, primals_61);  mul_61 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_105: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_45, [1576, 768]);  add_45 = None
    permute_47: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    addmm_22: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_159, view_105, permute_47);  primals_159 = None
    view_106: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_22, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_62: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
    mul_63: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476);  view_106 = None
    erf_5: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_63);  mul_63 = None
    add_46: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_64: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_62, add_46);  mul_62 = add_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_107: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_64, [1576, 3072]);  mul_64 = None
    permute_48: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_23: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_161, view_107, permute_48);  primals_161 = None
    view_108: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_23, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_65: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_59, view_108);  view_108 = None
    add_47: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_43, mul_65);  add_43 = mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_47, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 197, 1]" = var_mean_12[0]
    getitem_43: "f32[8, 197, 1]" = var_mean_12[1];  var_mean_12 = None
    add_48: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_12: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_48);  add_48 = None
    sub_18: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_47, getitem_43);  getitem_43 = None
    mul_66: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_12);  sub_18 = None
    mul_67: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_63)
    add_49: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_64);  mul_67 = primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_7: "f32[2304]" = torch.ops.aten.cat.default([primals_65, primals_212, primals_66]);  primals_65 = primals_212 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_109: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_49, [1576, 768]);  add_49 = None
    permute_49: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    addmm_24: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_7, view_109, permute_49);  cat_7 = None
    view_110: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(addmm_24, [8, 197, 2304]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_111: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(view_110, [8, 197, 3, 12, -1]);  view_110 = None
    permute_50: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_6 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
    getitem_44: "f32[8, 12, 197, 64]" = unbind_6[0]
    getitem_45: "f32[8, 12, 197, 64]" = unbind_6[1]
    getitem_46: "f32[8, 12, 197, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_112: "i64[38809]" = torch.ops.aten.reshape.default(primals_213, [-1]);  primals_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_6: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_68, [view_112]);  primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_113: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_6, [197, 197, -1]);  index_6 = None
    
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
    view_114: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_50, [96, 197, 64]);  clone_50 = None
    expand_26: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_69, [8, 12, 64, 197]);  mul_69 = None
    clone_51: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_26, memory_format = torch.contiguous_format);  expand_26 = None
    view_115: "f32[96, 64, 197]" = torch.ops.aten.reshape.default(clone_51, [96, 64, 197]);  clone_51 = None
    bmm_12: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_114, view_115)
    view_116: "f32[8, 12, 197, 197]" = torch.ops.aten.reshape.default(bmm_12, [8, 12, 197, 197]);  bmm_12 = None
    add_50: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_116, unsqueeze_6);  view_116 = unsqueeze_6 = None
    amax_6: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_50, [-1], True)
    sub_19: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_50, amax_6);  add_50 = amax_6 = None
    exp_6: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_6: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_6: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_6)
    expand_27: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_6, [8, 12, 197, 197]);  div_6 = None
    view_117: "f32[96, 197, 197]" = torch.ops.aten.reshape.default(expand_27, [96, 197, 197]);  expand_27 = None
    expand_28: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_46, [8, 12, 197, 64]);  getitem_46 = None
    clone_52: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_118: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_52, [96, 197, 64]);  clone_52 = None
    bmm_13: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_117, view_118)
    view_119: "f32[8, 12, 197, 64]" = torch.ops.aten.reshape.default(bmm_13, [8, 12, 197, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_53: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None
    clone_53: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_53, memory_format = torch.contiguous_format);  permute_53 = None
    view_120: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_53, [8, 197, 768]);  clone_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_121: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_120, [1576, 768]);  view_120 = None
    permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_25: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_163, view_121, permute_54);  primals_163 = None
    view_122: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_25, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_70: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_62, view_122);  view_122 = None
    add_51: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_47, mul_70);  add_47 = mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_47: "f32[8, 197, 1]" = var_mean_13[0]
    getitem_48: "f32[8, 197, 1]" = var_mean_13[1];  var_mean_13 = None
    add_52: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_47, 1e-06);  getitem_47 = None
    rsqrt_13: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_20: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_48);  getitem_48 = None
    mul_71: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
    mul_72: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_71, primals_70)
    add_53: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_72, primals_71);  mul_72 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_123: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_53, [1576, 768]);  add_53 = None
    permute_55: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
    addmm_26: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_165, view_123, permute_55);  primals_165 = None
    view_124: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_26, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_73: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, 0.5)
    mul_74: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, 0.7071067811865476);  view_124 = None
    erf_6: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_74);  mul_74 = None
    add_54: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_75: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_73, add_54);  mul_73 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_125: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_75, [1576, 3072]);  mul_75 = None
    permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_27: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_167, view_125, permute_56);  primals_167 = None
    view_126: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_27, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_76: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_69, view_126);  view_126 = None
    add_55: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_51, mul_76);  add_51 = mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_49: "f32[8, 197, 1]" = var_mean_14[0]
    getitem_50: "f32[8, 197, 1]" = var_mean_14[1];  var_mean_14 = None
    add_56: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_49, 1e-06);  getitem_49 = None
    rsqrt_14: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_21: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_50);  getitem_50 = None
    mul_77: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_14);  sub_21 = None
    mul_78: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_73)
    add_57: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_78, primals_74);  mul_78 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_8: "f32[2304]" = torch.ops.aten.cat.default([primals_75, primals_214, primals_76]);  primals_75 = primals_214 = primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_127: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_57, [1576, 768]);  add_57 = None
    permute_57: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm_28: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_8, view_127, permute_57);  cat_8 = None
    view_128: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(addmm_28, [8, 197, 2304]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_129: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(view_128, [8, 197, 3, 12, -1]);  view_128 = None
    permute_58: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_129, [2, 0, 3, 1, 4]);  view_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_7 = torch.ops.aten.unbind.int(permute_58);  permute_58 = None
    getitem_51: "f32[8, 12, 197, 64]" = unbind_7[0]
    getitem_52: "f32[8, 12, 197, 64]" = unbind_7[1]
    getitem_53: "f32[8, 12, 197, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_130: "i64[38809]" = torch.ops.aten.reshape.default(primals_215, [-1]);  primals_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_7: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_78, [view_130]);  primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_131: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_7, [197, 197, -1]);  index_7 = None
    
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
    view_132: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_58, [96, 197, 64]);  clone_58 = None
    expand_30: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_80, [8, 12, 64, 197]);  mul_80 = None
    clone_59: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
    view_133: "f32[96, 64, 197]" = torch.ops.aten.reshape.default(clone_59, [96, 64, 197]);  clone_59 = None
    bmm_14: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_132, view_133)
    view_134: "f32[8, 12, 197, 197]" = torch.ops.aten.reshape.default(bmm_14, [8, 12, 197, 197]);  bmm_14 = None
    add_58: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_134, unsqueeze_7);  view_134 = unsqueeze_7 = None
    amax_7: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_58, [-1], True)
    sub_22: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_58, amax_7);  add_58 = amax_7 = None
    exp_7: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_7: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_7: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_7)
    expand_31: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_7, [8, 12, 197, 197]);  div_7 = None
    view_135: "f32[96, 197, 197]" = torch.ops.aten.reshape.default(expand_31, [96, 197, 197]);  expand_31 = None
    expand_32: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_53, [8, 12, 197, 64]);  getitem_53 = None
    clone_60: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_136: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_60, [96, 197, 64]);  clone_60 = None
    bmm_15: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_135, view_136)
    view_137: "f32[8, 12, 197, 64]" = torch.ops.aten.reshape.default(bmm_15, [8, 12, 197, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_61: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
    clone_61: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_61, memory_format = torch.contiguous_format);  permute_61 = None
    view_138: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_61, [8, 197, 768]);  clone_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_139: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_138, [1576, 768]);  view_138 = None
    permute_62: "f32[768, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_29: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_169, view_139, permute_62);  primals_169 = None
    view_140: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_29, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_81: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_72, view_140);  view_140 = None
    add_59: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_55, mul_81);  add_55 = mul_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_54: "f32[8, 197, 1]" = var_mean_15[0]
    getitem_55: "f32[8, 197, 1]" = var_mean_15[1];  var_mean_15 = None
    add_60: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_54, 1e-06);  getitem_54 = None
    rsqrt_15: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_23: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_55);  getitem_55 = None
    mul_82: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
    mul_83: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_82, primals_80)
    add_61: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_83, primals_81);  mul_83 = primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_141: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_61, [1576, 768]);  add_61 = None
    permute_63: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_30: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_171, view_141, permute_63);  primals_171 = None
    view_142: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_30, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_84: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
    mul_85: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.7071067811865476);  view_142 = None
    erf_7: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_85);  mul_85 = None
    add_62: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_86: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_84, add_62);  mul_84 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_143: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_86, [1576, 3072]);  mul_86 = None
    permute_64: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_31: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_173, view_143, permute_64);  primals_173 = None
    view_144: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_31, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_87: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_79, view_144);  view_144 = None
    add_63: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_59, mul_87);  add_59 = mul_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_56: "f32[8, 197, 1]" = var_mean_16[0]
    getitem_57: "f32[8, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    add_64: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-06);  getitem_56 = None
    rsqrt_16: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_24: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_57);  getitem_57 = None
    mul_88: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_16);  sub_24 = None
    mul_89: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_88, primals_83)
    add_65: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_89, primals_84);  mul_89 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_9: "f32[2304]" = torch.ops.aten.cat.default([primals_85, primals_216, primals_86]);  primals_85 = primals_216 = primals_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_145: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_65, [1576, 768]);  add_65 = None
    permute_65: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_32: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_9, view_145, permute_65);  cat_9 = None
    view_146: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(addmm_32, [8, 197, 2304]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_147: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(view_146, [8, 197, 3, 12, -1]);  view_146 = None
    permute_66: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_147, [2, 0, 3, 1, 4]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_8 = torch.ops.aten.unbind.int(permute_66);  permute_66 = None
    getitem_58: "f32[8, 12, 197, 64]" = unbind_8[0]
    getitem_59: "f32[8, 12, 197, 64]" = unbind_8[1]
    getitem_60: "f32[8, 12, 197, 64]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_148: "i64[38809]" = torch.ops.aten.reshape.default(primals_217, [-1]);  primals_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_8: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_88, [view_148]);  primals_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_149: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_8, [197, 197, -1]);  index_8 = None
    
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
    view_150: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_66, [96, 197, 64]);  clone_66 = None
    expand_34: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_91, [8, 12, 64, 197]);  mul_91 = None
    clone_67: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_34, memory_format = torch.contiguous_format);  expand_34 = None
    view_151: "f32[96, 64, 197]" = torch.ops.aten.reshape.default(clone_67, [96, 64, 197]);  clone_67 = None
    bmm_16: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_150, view_151)
    view_152: "f32[8, 12, 197, 197]" = torch.ops.aten.reshape.default(bmm_16, [8, 12, 197, 197]);  bmm_16 = None
    add_66: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_152, unsqueeze_8);  view_152 = unsqueeze_8 = None
    amax_8: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_66, [-1], True)
    sub_25: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_66, amax_8);  add_66 = amax_8 = None
    exp_8: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_8: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_8: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_8)
    expand_35: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_8, [8, 12, 197, 197]);  div_8 = None
    view_153: "f32[96, 197, 197]" = torch.ops.aten.reshape.default(expand_35, [96, 197, 197]);  expand_35 = None
    expand_36: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_60, [8, 12, 197, 64]);  getitem_60 = None
    clone_68: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_154: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_68, [96, 197, 64]);  clone_68 = None
    bmm_17: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_153, view_154)
    view_155: "f32[8, 12, 197, 64]" = torch.ops.aten.reshape.default(bmm_17, [8, 12, 197, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_69: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
    clone_69: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
    view_156: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_69, [8, 197, 768]);  clone_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_157: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_156, [1576, 768]);  view_156 = None
    permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    addmm_33: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_175, view_157, permute_70);  primals_175 = None
    view_158: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_33, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_92: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_82, view_158);  view_158 = None
    add_67: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_63, mul_92);  add_63 = mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_67, [2], correction = 0, keepdim = True)
    getitem_61: "f32[8, 197, 1]" = var_mean_17[0]
    getitem_62: "f32[8, 197, 1]" = var_mean_17[1];  var_mean_17 = None
    add_68: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_61, 1e-06);  getitem_61 = None
    rsqrt_17: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_68);  add_68 = None
    sub_26: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_67, getitem_62);  getitem_62 = None
    mul_93: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
    mul_94: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_93, primals_90)
    add_69: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_94, primals_91);  mul_94 = primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_159: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_69, [1576, 768]);  add_69 = None
    permute_71: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_34: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_177, view_159, permute_71);  primals_177 = None
    view_160: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_34, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_95: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_160, 0.5)
    mul_96: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_160, 0.7071067811865476);  view_160 = None
    erf_8: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_70: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_97: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_95, add_70);  mul_95 = add_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_161: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_97, [1576, 3072]);  mul_97 = None
    permute_72: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_35: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_179, view_161, permute_72);  primals_179 = None
    view_162: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_35, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_98: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_89, view_162);  view_162 = None
    add_71: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_67, mul_98);  add_67 = mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_71, [2], correction = 0, keepdim = True)
    getitem_63: "f32[8, 197, 1]" = var_mean_18[0]
    getitem_64: "f32[8, 197, 1]" = var_mean_18[1];  var_mean_18 = None
    add_72: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_63, 1e-06);  getitem_63 = None
    rsqrt_18: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_72);  add_72 = None
    sub_27: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_71, getitem_64);  getitem_64 = None
    mul_99: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_18);  sub_27 = None
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_99, primals_93)
    add_73: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_100, primals_94);  mul_100 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_10: "f32[2304]" = torch.ops.aten.cat.default([primals_95, primals_218, primals_96]);  primals_95 = primals_218 = primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_163: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_73, [1576, 768]);  add_73 = None
    permute_73: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_36: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_10, view_163, permute_73);  cat_10 = None
    view_164: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(addmm_36, [8, 197, 2304]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_165: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(view_164, [8, 197, 3, 12, -1]);  view_164 = None
    permute_74: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_165, [2, 0, 3, 1, 4]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_9 = torch.ops.aten.unbind.int(permute_74);  permute_74 = None
    getitem_65: "f32[8, 12, 197, 64]" = unbind_9[0]
    getitem_66: "f32[8, 12, 197, 64]" = unbind_9[1]
    getitem_67: "f32[8, 12, 197, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_166: "i64[38809]" = torch.ops.aten.reshape.default(primals_219, [-1]);  primals_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_9: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_98, [view_166]);  primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_167: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_9, [197, 197, -1]);  index_9 = None
    
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
    view_168: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_74, [96, 197, 64]);  clone_74 = None
    expand_38: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_102, [8, 12, 64, 197]);  mul_102 = None
    clone_75: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
    view_169: "f32[96, 64, 197]" = torch.ops.aten.reshape.default(clone_75, [96, 64, 197]);  clone_75 = None
    bmm_18: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_168, view_169)
    view_170: "f32[8, 12, 197, 197]" = torch.ops.aten.reshape.default(bmm_18, [8, 12, 197, 197]);  bmm_18 = None
    add_74: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_170, unsqueeze_9);  view_170 = unsqueeze_9 = None
    amax_9: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_74, [-1], True)
    sub_28: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_74, amax_9);  add_74 = amax_9 = None
    exp_9: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_9: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_9: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_9)
    expand_39: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_9, [8, 12, 197, 197]);  div_9 = None
    view_171: "f32[96, 197, 197]" = torch.ops.aten.reshape.default(expand_39, [96, 197, 197]);  expand_39 = None
    expand_40: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_67, [8, 12, 197, 64]);  getitem_67 = None
    clone_76: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_172: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_76, [96, 197, 64]);  clone_76 = None
    bmm_19: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_171, view_172)
    view_173: "f32[8, 12, 197, 64]" = torch.ops.aten.reshape.default(bmm_19, [8, 12, 197, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_77: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    clone_77: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_77, memory_format = torch.contiguous_format);  permute_77 = None
    view_174: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_77, [8, 197, 768]);  clone_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_175: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_174, [1576, 768]);  view_174 = None
    permute_78: "f32[768, 768]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    addmm_37: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_181, view_175, permute_78);  primals_181 = None
    view_176: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_37, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_103: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_92, view_176);  view_176 = None
    add_75: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_71, mul_103);  add_71 = mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_68: "f32[8, 197, 1]" = var_mean_19[0]
    getitem_69: "f32[8, 197, 1]" = var_mean_19[1];  var_mean_19 = None
    add_76: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-06);  getitem_68 = None
    rsqrt_19: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_29: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_75, getitem_69);  getitem_69 = None
    mul_104: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
    mul_105: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_104, primals_100)
    add_77: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_105, primals_101);  mul_105 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_177: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_77, [1576, 768]);  add_77 = None
    permute_79: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_38: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_183, view_177, permute_79);  primals_183 = None
    view_178: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_38, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_106: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
    mul_107: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_178, 0.7071067811865476);  view_178 = None
    erf_9: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_107);  mul_107 = None
    add_78: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_108: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_106, add_78);  mul_106 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_179: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_108, [1576, 3072]);  mul_108 = None
    permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_39: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_185, view_179, permute_80);  primals_185 = None
    view_180: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_39, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_99, view_180);  view_180 = None
    add_79: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_75, mul_109);  add_75 = mul_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_70: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_71: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_80: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-06);  getitem_70 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_30: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_71);  getitem_71 = None
    mul_110: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_20);  sub_30 = None
    mul_111: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_110, primals_103)
    add_81: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_111, primals_104);  mul_111 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_11: "f32[2304]" = torch.ops.aten.cat.default([primals_105, primals_220, primals_106]);  primals_105 = primals_220 = primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_181: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_81, [1576, 768]);  add_81 = None
    permute_81: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_40: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_11, view_181, permute_81);  cat_11 = None
    view_182: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(addmm_40, [8, 197, 2304]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_183: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(view_182, [8, 197, 3, 12, -1]);  view_182 = None
    permute_82: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_183, [2, 0, 3, 1, 4]);  view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_10 = torch.ops.aten.unbind.int(permute_82);  permute_82 = None
    getitem_72: "f32[8, 12, 197, 64]" = unbind_10[0]
    getitem_73: "f32[8, 12, 197, 64]" = unbind_10[1]
    getitem_74: "f32[8, 12, 197, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_184: "i64[38809]" = torch.ops.aten.reshape.default(primals_221, [-1]);  primals_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_10: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_108, [view_184]);  primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_185: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_10, [197, 197, -1]);  index_10 = None
    
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
    view_186: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_82, [96, 197, 64]);  clone_82 = None
    expand_42: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_113, [8, 12, 64, 197]);  mul_113 = None
    clone_83: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
    view_187: "f32[96, 64, 197]" = torch.ops.aten.reshape.default(clone_83, [96, 64, 197]);  clone_83 = None
    bmm_20: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_186, view_187)
    view_188: "f32[8, 12, 197, 197]" = torch.ops.aten.reshape.default(bmm_20, [8, 12, 197, 197]);  bmm_20 = None
    add_82: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_188, unsqueeze_10);  view_188 = unsqueeze_10 = None
    amax_10: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_82, [-1], True)
    sub_31: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_82, amax_10);  add_82 = amax_10 = None
    exp_10: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_10: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_10: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_10)
    expand_43: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_10, [8, 12, 197, 197]);  div_10 = None
    view_189: "f32[96, 197, 197]" = torch.ops.aten.reshape.default(expand_43, [96, 197, 197]);  expand_43 = None
    expand_44: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_74, [8, 12, 197, 64]);  getitem_74 = None
    clone_84: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_190: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_84, [96, 197, 64]);  clone_84 = None
    bmm_21: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_189, view_190)
    view_191: "f32[8, 12, 197, 64]" = torch.ops.aten.reshape.default(bmm_21, [8, 12, 197, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_85: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
    clone_85: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_85, memory_format = torch.contiguous_format);  permute_85 = None
    view_192: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_85, [8, 197, 768]);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_193: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_192, [1576, 768]);  view_192 = None
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    addmm_41: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_187, view_193, permute_86);  primals_187 = None
    view_194: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_41, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_114: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_102, view_194);  view_194 = None
    add_83: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_79, mul_114);  add_79 = mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_83, [2], correction = 0, keepdim = True)
    getitem_75: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_76: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_84: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-06);  getitem_75 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_32: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_83, getitem_76);  getitem_76 = None
    mul_115: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
    mul_116: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_115, primals_110)
    add_85: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_116, primals_111);  mul_116 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_195: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_85, [1576, 768]);  add_85 = None
    permute_87: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_42: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_189, view_195, permute_87);  primals_189 = None
    view_196: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_42, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_117: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.5)
    mul_118: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_196, 0.7071067811865476);  view_196 = None
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_118);  mul_118 = None
    add_86: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_119: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_117, add_86);  mul_117 = add_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_197: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_119, [1576, 3072]);  mul_119 = None
    permute_88: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_43: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_191, view_197, permute_88);  primals_191 = None
    view_198: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_43, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_120: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_109, view_198);  view_198 = None
    add_87: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_83, mul_120);  add_83 = mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_87, [2], correction = 0, keepdim = True)
    getitem_77: "f32[8, 197, 1]" = var_mean_22[0]
    getitem_78: "f32[8, 197, 1]" = var_mean_22[1];  var_mean_22 = None
    add_88: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
    rsqrt_22: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    sub_33: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_87, getitem_78);  getitem_78 = None
    mul_121: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_33, rsqrt_22);  sub_33 = None
    mul_122: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_121, primals_113)
    add_89: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_122, primals_114);  mul_122 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_12: "f32[2304]" = torch.ops.aten.cat.default([primals_115, primals_222, primals_116]);  primals_115 = primals_222 = primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_199: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_89, [1576, 768]);  add_89 = None
    permute_89: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_44: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_12, view_199, permute_89);  cat_12 = None
    view_200: "f32[8, 197, 2304]" = torch.ops.aten.reshape.default(addmm_44, [8, 197, 2304]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_201: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.reshape.default(view_200, [8, 197, 3, 12, -1]);  view_200 = None
    permute_90: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_201, [2, 0, 3, 1, 4]);  view_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_11 = torch.ops.aten.unbind.int(permute_90);  permute_90 = None
    getitem_79: "f32[8, 12, 197, 64]" = unbind_11[0]
    getitem_80: "f32[8, 12, 197, 64]" = unbind_11[1]
    getitem_81: "f32[8, 12, 197, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_202: "i64[38809]" = torch.ops.aten.reshape.default(primals_223, [-1]);  primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_11: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_118, [view_202]);  primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_203: "f32[197, 197, 12]" = torch.ops.aten.reshape.default(index_11, [197, 197, -1]);  index_11 = None
    
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
    view_204: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_90, [96, 197, 64]);  clone_90 = None
    expand_46: "f32[8, 12, 64, 197]" = torch.ops.aten.expand.default(mul_124, [8, 12, 64, 197]);  mul_124 = None
    clone_91: "f32[8, 12, 64, 197]" = torch.ops.aten.clone.default(expand_46, memory_format = torch.contiguous_format);  expand_46 = None
    view_205: "f32[96, 64, 197]" = torch.ops.aten.reshape.default(clone_91, [96, 64, 197]);  clone_91 = None
    bmm_22: "f32[96, 197, 197]" = torch.ops.aten.bmm.default(view_204, view_205)
    view_206: "f32[8, 12, 197, 197]" = torch.ops.aten.reshape.default(bmm_22, [8, 12, 197, 197]);  bmm_22 = None
    add_90: "f32[8, 12, 197, 197]" = torch.ops.aten.add.Tensor(view_206, unsqueeze_11);  view_206 = unsqueeze_11 = None
    amax_11: "f32[8, 12, 197, 1]" = torch.ops.aten.amax.default(add_90, [-1], True)
    sub_34: "f32[8, 12, 197, 197]" = torch.ops.aten.sub.Tensor(add_90, amax_11);  add_90 = amax_11 = None
    exp_11: "f32[8, 12, 197, 197]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[8, 12, 197, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_11: "f32[8, 12, 197, 197]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_11: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(div_11)
    expand_47: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(div_11, [8, 12, 197, 197]);  div_11 = None
    view_207: "f32[96, 197, 197]" = torch.ops.aten.reshape.default(expand_47, [96, 197, 197]);  expand_47 = None
    expand_48: "f32[8, 12, 197, 64]" = torch.ops.aten.expand.default(getitem_81, [8, 12, 197, 64]);  getitem_81 = None
    clone_92: "f32[8, 12, 197, 64]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
    view_208: "f32[96, 197, 64]" = torch.ops.aten.reshape.default(clone_92, [96, 197, 64]);  clone_92 = None
    bmm_23: "f32[96, 197, 64]" = torch.ops.aten.bmm.default(view_207, view_208)
    view_209: "f32[8, 12, 197, 64]" = torch.ops.aten.reshape.default(bmm_23, [8, 12, 197, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_93: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(view_209, [0, 2, 1, 3]);  view_209 = None
    clone_93: "f32[8, 197, 12, 64]" = torch.ops.aten.clone.default(permute_93, memory_format = torch.contiguous_format);  permute_93 = None
    view_210: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(clone_93, [8, 197, 768]);  clone_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_211: "f32[1576, 768]" = torch.ops.aten.reshape.default(view_210, [1576, 768]);  view_210 = None
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_45: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_193, view_211, permute_94);  primals_193 = None
    view_212: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_45, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_125: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_112, view_212);  view_212 = None
    add_91: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_87, mul_125);  add_87 = mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_91, [2], correction = 0, keepdim = True)
    getitem_82: "f32[8, 197, 1]" = var_mean_23[0]
    getitem_83: "f32[8, 197, 1]" = var_mean_23[1];  var_mean_23 = None
    add_92: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_82, 1e-06);  getitem_82 = None
    rsqrt_23: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_92);  add_92 = None
    sub_35: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_91, getitem_83);  getitem_83 = None
    mul_126: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
    mul_127: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_126, primals_120)
    add_93: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_127, primals_121);  mul_127 = primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_213: "f32[1576, 768]" = torch.ops.aten.reshape.default(add_93, [1576, 768]);  add_93 = None
    permute_95: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_46: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_195, view_213, permute_95);  primals_195 = None
    view_214: "f32[8, 197, 3072]" = torch.ops.aten.reshape.default(addmm_46, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_128: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, 0.5)
    mul_129: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, 0.7071067811865476);  view_214 = None
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
    add_94: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_130: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_128, add_94);  mul_128 = add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_215: "f32[1576, 3072]" = torch.ops.aten.reshape.default(mul_130, [1576, 3072]);  mul_130 = None
    permute_96: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_47: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_197, view_215, permute_96);  primals_197 = None
    view_216: "f32[8, 197, 768]" = torch.ops.aten.reshape.default(addmm_47, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_131: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_119, view_216);  view_216 = None
    add_95: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_91, mul_131);  add_91 = mul_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:421, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    slice_2: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(add_95, 1, 1, 9223372036854775807);  add_95 = None
    mean: "f32[8, 768]" = torch.ops.aten.mean.dim(slice_2, [1]);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(mean, [1], correction = 0, keepdim = True)
    getitem_84: "f32[8, 1]" = var_mean_24[0]
    getitem_85: "f32[8, 1]" = var_mean_24[1];  var_mean_24 = None
    add_96: "f32[8, 1]" = torch.ops.aten.add.Tensor(getitem_84, 1e-06);  getitem_84 = None
    rsqrt_24: "f32[8, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
    sub_36: "f32[8, 768]" = torch.ops.aten.sub.Tensor(mean, getitem_85);  mean = getitem_85 = None
    mul_132: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_24);  sub_36 = None
    mul_133: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_132, primals_122)
    add_97: "f32[8, 768]" = torch.ops.aten.add.Tensor(mul_133, primals_123);  mul_133 = primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:424, code: return x if pre_logits else self.head(x)
    permute_97: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_48: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_199, add_97, permute_97);  primals_199 = None
    permute_98: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_12: "f32[8, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_102: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_106: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_14: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    permute_115: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    permute_116: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    alias_12: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    permute_117: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_204, [0, 2, 1]);  view_204 = None
    permute_118: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_205, [0, 2, 1]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_122: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_15: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_126: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_130: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_16: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_134: "f32[768, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    permute_139: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    permute_140: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
    alias_13: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    permute_141: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
    permute_142: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_187, [0, 2, 1]);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_146: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_17: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_154: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_18: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    permute_163: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    permute_164: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    alias_14: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    permute_165: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
    permute_166: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_170: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_19: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_174: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_178: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_20: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_182: "f32[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    permute_187: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
    permute_188: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_154, [0, 2, 1]);  view_154 = None
    alias_15: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    permute_189: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_150, [0, 2, 1]);  view_150 = None
    permute_190: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_151, [0, 2, 1]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_194: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_21: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_198: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_202: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_22: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    permute_211: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None
    permute_212: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_136, [0, 2, 1]);  view_136 = None
    alias_16: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    permute_213: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_132, [0, 2, 1]);  view_132 = None
    permute_214: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_218: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_23: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_222: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_226: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_24: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    permute_235: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_117, [0, 2, 1]);  view_117 = None
    permute_236: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_118, [0, 2, 1]);  view_118 = None
    alias_17: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    permute_237: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
    permute_238: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_242: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_25: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_246: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_250: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_26: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_254: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    permute_259: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_99, [0, 2, 1]);  view_99 = None
    permute_260: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    alias_18: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    permute_261: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
    permute_262: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_266: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_27: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_270: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_274: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_28: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_278: "f32[768, 768]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    permute_283: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
    permute_284: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_82, [0, 2, 1]);  view_82 = None
    alias_19: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    permute_285: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    permute_286: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_290: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_29: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_294: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_298: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_30: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_302: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    permute_307: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    permute_308: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
    alias_20: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    permute_309: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_60, [0, 2, 1]);  view_60 = None
    permute_310: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_314: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_31: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_318: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_322: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_32: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_326: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    permute_331: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_45, [0, 2, 1]);  view_45 = None
    permute_332: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_46, [0, 2, 1]);  view_46 = None
    alias_21: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    permute_333: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None
    permute_334: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_338: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_33: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_342: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_346: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_34: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_350: "f32[768, 768]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    permute_355: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None
    permute_356: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_28, [0, 2, 1]);  view_28 = None
    alias_22: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    permute_357: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_24, [0, 2, 1]);  view_24 = None
    permute_358: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_25, [0, 2, 1]);  view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_362: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_35: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_366: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_370: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_36: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_374: "f32[768, 768]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    permute_379: "f32[96, 197, 197]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    permute_380: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    alias_23: "f32[8, 12, 197, 197]" = torch.ops.aten.alias.default(alias);  alias = None
    permute_381: "f32[96, 64, 197]" = torch.ops.aten.permute.default(view_6, [0, 2, 1]);  view_6 = None
    permute_382: "f32[96, 197, 64]" = torch.ops.aten.permute.default(view_7, [0, 2, 1]);  view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_386: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    return [addmm_48, primals_2, primals_3, primals_9, primals_10, primals_12, primals_13, primals_19, primals_20, primals_22, primals_23, primals_29, primals_30, primals_32, primals_33, primals_39, primals_40, primals_42, primals_43, primals_49, primals_50, primals_52, primals_53, primals_59, primals_60, primals_62, primals_63, primals_69, primals_70, primals_72, primals_73, primals_79, primals_80, primals_82, primals_83, primals_89, primals_90, primals_92, primals_93, primals_99, primals_100, primals_102, primals_103, primals_109, primals_110, primals_112, primals_113, primals_119, primals_120, primals_122, primals_124, primals_224, cat, getitem_1, rsqrt, view_1, view_4, view_13, addmm_1, mul_5, view_15, addmm_2, view_17, addmm_3, mul_11, view_19, view_22, view_31, addmm_5, mul_16, view_33, addmm_6, view_35, addmm_7, mul_22, view_37, view_40, view_49, addmm_9, mul_27, view_51, addmm_10, view_53, addmm_11, mul_33, view_55, view_58, view_67, addmm_13, mul_38, view_69, addmm_14, view_71, addmm_15, mul_44, view_73, view_76, view_85, addmm_17, mul_49, view_87, addmm_18, view_89, addmm_19, mul_55, view_91, view_94, view_103, addmm_21, mul_60, view_105, addmm_22, view_107, addmm_23, mul_66, view_109, view_112, view_121, addmm_25, mul_71, view_123, addmm_26, view_125, addmm_27, mul_77, view_127, view_130, view_139, addmm_29, mul_82, view_141, addmm_30, view_143, addmm_31, mul_88, view_145, view_148, view_157, addmm_33, mul_93, view_159, addmm_34, view_161, addmm_35, mul_99, view_163, view_166, view_175, addmm_37, mul_104, view_177, addmm_38, view_179, addmm_39, mul_110, view_181, view_184, view_193, addmm_41, mul_115, view_195, addmm_42, view_197, addmm_43, mul_121, view_199, view_202, view_211, addmm_45, mul_126, view_213, addmm_46, view_215, addmm_47, mul_132, add_97, permute_98, div_12, permute_102, permute_106, div_14, permute_110, permute_115, permute_116, alias_12, permute_117, permute_118, permute_122, div_15, permute_126, permute_130, div_16, permute_134, permute_139, permute_140, alias_13, permute_141, permute_142, permute_146, div_17, permute_150, permute_154, div_18, permute_158, permute_163, permute_164, alias_14, permute_165, permute_166, permute_170, div_19, permute_174, permute_178, div_20, permute_182, permute_187, permute_188, alias_15, permute_189, permute_190, permute_194, div_21, permute_198, permute_202, div_22, permute_206, permute_211, permute_212, alias_16, permute_213, permute_214, permute_218, div_23, permute_222, permute_226, div_24, permute_230, permute_235, permute_236, alias_17, permute_237, permute_238, permute_242, div_25, permute_246, permute_250, div_26, permute_254, permute_259, permute_260, alias_18, permute_261, permute_262, permute_266, div_27, permute_270, permute_274, div_28, permute_278, permute_283, permute_284, alias_19, permute_285, permute_286, permute_290, div_29, permute_294, permute_298, div_30, permute_302, permute_307, permute_308, alias_20, permute_309, permute_310, permute_314, div_31, permute_318, permute_322, div_32, permute_326, permute_331, permute_332, alias_21, permute_333, permute_334, permute_338, div_33, permute_342, permute_346, div_34, permute_350, permute_355, permute_356, alias_22, permute_357, permute_358, permute_362, div_35, permute_366, permute_370, div_36, permute_374, permute_379, permute_380, alias_23, permute_381, permute_382, permute_386]
    