from __future__ import annotations



def forward(self, primals_1: "f32[1, 1, 768]", primals_2: "f32[768]", primals_3: "f32[768]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[768]", primals_7: "f32[2304, 768]", primals_8: "f32[732, 12]", primals_9: "f32[768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[768]", primals_14: "f32[768]", primals_15: "f32[768]", primals_16: "f32[768]", primals_17: "f32[2304, 768]", primals_18: "f32[732, 12]", primals_19: "f32[768]", primals_20: "f32[768]", primals_21: "f32[768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[768]", primals_26: "f32[768]", primals_27: "f32[2304, 768]", primals_28: "f32[732, 12]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[768]", primals_32: "f32[768]", primals_33: "f32[768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[2304, 768]", primals_38: "f32[732, 12]", primals_39: "f32[768]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[768]", primals_43: "f32[768]", primals_44: "f32[768]", primals_45: "f32[768]", primals_46: "f32[768]", primals_47: "f32[2304, 768]", primals_48: "f32[732, 12]", primals_49: "f32[768]", primals_50: "f32[768]", primals_51: "f32[768]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[768]", primals_56: "f32[768]", primals_57: "f32[2304, 768]", primals_58: "f32[732, 12]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[768]", primals_62: "f32[768]", primals_63: "f32[768]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[2304, 768]", primals_68: "f32[732, 12]", primals_69: "f32[768]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[768]", primals_74: "f32[768]", primals_75: "f32[768]", primals_76: "f32[768]", primals_77: "f32[2304, 768]", primals_78: "f32[732, 12]", primals_79: "f32[768]", primals_80: "f32[768]", primals_81: "f32[768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[768]", primals_86: "f32[768]", primals_87: "f32[2304, 768]", primals_88: "f32[732, 12]", primals_89: "f32[768]", primals_90: "f32[768]", primals_91: "f32[768]", primals_92: "f32[768]", primals_93: "f32[768]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[2304, 768]", primals_98: "f32[732, 12]", primals_99: "f32[768]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[768]", primals_103: "f32[768]", primals_104: "f32[768]", primals_105: "f32[768]", primals_106: "f32[768]", primals_107: "f32[2304, 768]", primals_108: "f32[732, 12]", primals_109: "f32[768]", primals_110: "f32[768]", primals_111: "f32[768]", primals_112: "f32[768]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[768]", primals_116: "f32[768]", primals_117: "f32[2304, 768]", primals_118: "f32[732, 12]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[768]", primals_122: "f32[768]", primals_123: "f32[768]", primals_124: "f32[768, 3, 16, 16]", primals_125: "f32[768]", primals_126: "f32[768, 768]", primals_127: "f32[768]", primals_128: "f32[3072, 768]", primals_129: "f32[3072]", primals_130: "f32[768, 3072]", primals_131: "f32[768]", primals_132: "f32[768, 768]", primals_133: "f32[768]", primals_134: "f32[3072, 768]", primals_135: "f32[3072]", primals_136: "f32[768, 3072]", primals_137: "f32[768]", primals_138: "f32[768, 768]", primals_139: "f32[768]", primals_140: "f32[3072, 768]", primals_141: "f32[3072]", primals_142: "f32[768, 3072]", primals_143: "f32[768]", primals_144: "f32[768, 768]", primals_145: "f32[768]", primals_146: "f32[3072, 768]", primals_147: "f32[3072]", primals_148: "f32[768, 3072]", primals_149: "f32[768]", primals_150: "f32[768, 768]", primals_151: "f32[768]", primals_152: "f32[3072, 768]", primals_153: "f32[3072]", primals_154: "f32[768, 3072]", primals_155: "f32[768]", primals_156: "f32[768, 768]", primals_157: "f32[768]", primals_158: "f32[3072, 768]", primals_159: "f32[3072]", primals_160: "f32[768, 3072]", primals_161: "f32[768]", primals_162: "f32[768, 768]", primals_163: "f32[768]", primals_164: "f32[3072, 768]", primals_165: "f32[3072]", primals_166: "f32[768, 3072]", primals_167: "f32[768]", primals_168: "f32[768, 768]", primals_169: "f32[768]", primals_170: "f32[3072, 768]", primals_171: "f32[3072]", primals_172: "f32[768, 3072]", primals_173: "f32[768]", primals_174: "f32[768, 768]", primals_175: "f32[768]", primals_176: "f32[3072, 768]", primals_177: "f32[3072]", primals_178: "f32[768, 3072]", primals_179: "f32[768]", primals_180: "f32[768, 768]", primals_181: "f32[768]", primals_182: "f32[3072, 768]", primals_183: "f32[3072]", primals_184: "f32[768, 3072]", primals_185: "f32[768]", primals_186: "f32[768, 768]", primals_187: "f32[768]", primals_188: "f32[3072, 768]", primals_189: "f32[3072]", primals_190: "f32[768, 3072]", primals_191: "f32[768]", primals_192: "f32[768, 768]", primals_193: "f32[768]", primals_194: "f32[3072, 768]", primals_195: "f32[3072]", primals_196: "f32[768, 3072]", primals_197: "f32[768]", primals_198: "f32[1000, 768]", primals_199: "f32[1000]", primals_200: "f32[768]", primals_201: "i64[197, 197]", primals_202: "f32[768]", primals_203: "i64[197, 197]", primals_204: "f32[768]", primals_205: "i64[197, 197]", primals_206: "f32[768]", primals_207: "i64[197, 197]", primals_208: "f32[768]", primals_209: "i64[197, 197]", primals_210: "f32[768]", primals_211: "i64[197, 197]", primals_212: "f32[768]", primals_213: "i64[197, 197]", primals_214: "f32[768]", primals_215: "i64[197, 197]", primals_216: "f32[768]", primals_217: "i64[197, 197]", primals_218: "f32[768]", primals_219: "i64[197, 197]", primals_220: "f32[768]", primals_221: "i64[197, 197]", primals_222: "f32[768]", primals_223: "i64[197, 197]", primals_224: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(primals_224, primals_124, primals_125, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 196]" = torch.ops.aten.view.default(convolution, [8, 768, 196]);  convolution = None
    permute: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:405, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    expand: "f32[8, 1, 768]" = torch.ops.aten.expand.default(primals_1, [8, -1, -1]);  primals_1 = None
    cat: "f32[8, 197, 768]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:408, code: x = self.pos_drop(x)
    clone: "f32[8, 197, 768]" = torch.ops.aten.clone.default(cat)
    
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
    constant_pad_nd: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze, [0, 3], 0.0);  unsqueeze = None
    slice_1: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd, -1, 0, 197);  constant_pad_nd = None
    expand_1: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_1, [8, 12, 197, 197]);  slice_1 = None
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_2, getitem_3, getitem_4, expand_1, True)
    getitem_5: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention[0]
    getitem_6: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention[1]
    getitem_7: "i64[]" = _scaled_dot_product_efficient_attention[2]
    getitem_8: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
    alias: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_4: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    view_6: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_4, [8, 197, 768]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_7: "f32[1576, 768]" = torch.ops.aten.view.default(view_6, [1576, 768]);  view_6 = None
    permute_5: "f32[768, 768]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    addmm_1: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_127, view_7, permute_5);  primals_127 = None
    view_8: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_1, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_2: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_8);  view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_2: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_2, clone_2);  clone_2 = None
    add_2: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(clone, mul_2);  clone = mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_1 = torch.ops.aten.var_mean.correction(add_2, [2], correction = 0, keepdim = True)
    getitem_9: "f32[8, 197, 1]" = var_mean_1[0]
    getitem_10: "f32[8, 197, 1]" = var_mean_1[1];  var_mean_1 = None
    add_3: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-06);  getitem_9 = None
    rsqrt_1: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
    sub_1: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_2, getitem_10);  getitem_10 = None
    mul_3: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_4: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_3, primals_10)
    add_4: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_4, primals_11);  mul_4 = primals_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_9: "f32[1576, 768]" = torch.ops.aten.view.default(add_4, [1576, 768]);  add_4 = None
    permute_6: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_2: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_129, view_9, permute_6);  primals_129 = None
    view_10: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_2, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_5: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
    mul_6: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_10, 0.7071067811865476);  view_10 = None
    erf: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_6);  mul_6 = None
    add_5: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_7: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_5, add_5);  mul_5 = add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_3: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_7);  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_11: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_3, [1576, 3072]);  clone_3 = None
    permute_7: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_3: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_131, view_11, permute_7);  primals_131 = None
    view_12: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_3, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_4: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_12);  view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_8: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_9, clone_4);  clone_4 = None
    add_6: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_2, mul_8);  add_2 = mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_2 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
    getitem_11: "f32[8, 197, 1]" = var_mean_2[0]
    getitem_12: "f32[8, 197, 1]" = var_mean_2[1];  var_mean_2 = None
    add_7: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-06);  getitem_11 = None
    rsqrt_2: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    sub_2: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_6, getitem_12);  getitem_12 = None
    mul_9: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_10: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_13)
    add_8: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_14);  mul_10 = primals_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_2: "f32[2304]" = torch.ops.aten.cat.default([primals_15, primals_202, primals_16]);  primals_15 = primals_202 = primals_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_13: "f32[1576, 768]" = torch.ops.aten.view.default(add_8, [1576, 768]);  add_8 = None
    permute_8: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
    addmm_4: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_2, view_13, permute_8);  cat_2 = None
    view_14: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_4, [8, 197, 2304]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_15: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_14, [8, 197, 3, 12, -1]);  view_14 = None
    permute_9: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_15, [2, 0, 3, 1, 4]);  view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_1 = torch.ops.aten.unbind.int(permute_9);  permute_9 = None
    getitem_13: "f32[8, 12, 197, 64]" = unbind_1[0]
    getitem_14: "f32[8, 12, 197, 64]" = unbind_1[1]
    getitem_15: "f32[8, 12, 197, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_16: "i64[38809]" = torch.ops.aten.view.default(primals_203, [-1]);  primals_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_1: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_18, [view_16]);  primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_17: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_1, [197, 197, -1]);  index_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_10: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_17, [2, 0, 1]);  view_17 = None
    clone_5: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_1: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_5, 0);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_1: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_1, [0, 3], 0.0);  unsqueeze_1 = None
    slice_2: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_1, -1, 0, 197);  constant_pad_nd_1 = None
    expand_2: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_2, [8, 12, 197, 197]);  slice_2 = None
    _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_13, getitem_14, getitem_15, expand_2, True)
    getitem_16: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_1[0]
    getitem_17: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_1[1]
    getitem_18: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
    getitem_19: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
    alias_1: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_11: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
    view_18: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_11, [8, 197, 768]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_19: "f32[1576, 768]" = torch.ops.aten.view.default(view_18, [1576, 768]);  view_18 = None
    permute_12: "f32[768, 768]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    addmm_5: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_133, view_19, permute_12);  primals_133 = None
    view_20: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_5, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_6: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_11: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_12, clone_6);  clone_6 = None
    add_9: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_6, mul_11);  add_6 = mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_3 = torch.ops.aten.var_mean.correction(add_9, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 197, 1]" = var_mean_3[0]
    getitem_21: "f32[8, 197, 1]" = var_mean_3[1];  var_mean_3 = None
    add_10: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_3: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    sub_3: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_9, getitem_21);  getitem_21 = None
    mul_12: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_13: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_12, primals_20)
    add_11: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_13, primals_21);  mul_13 = primals_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_21: "f32[1576, 768]" = torch.ops.aten.view.default(add_11, [1576, 768]);  add_11 = None
    permute_13: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_6: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_135, view_21, permute_13);  primals_135 = None
    view_22: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_6, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_14: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
    mul_15: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_22, 0.7071067811865476);  view_22 = None
    erf_1: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_15);  mul_15 = None
    add_12: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_16: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_14, add_12);  mul_14 = add_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_7: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_16);  mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_23: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_7, [1576, 3072]);  clone_7 = None
    permute_14: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_7: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_137, view_23, permute_14);  primals_137 = None
    view_24: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_7, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_8: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_24);  view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_17: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_19, clone_8);  clone_8 = None
    add_13: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_9, mul_17);  add_9 = mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_4 = torch.ops.aten.var_mean.correction(add_13, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 197, 1]" = var_mean_4[0]
    getitem_23: "f32[8, 197, 1]" = var_mean_4[1];  var_mean_4 = None
    add_14: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_4: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    sub_4: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_13, getitem_23);  getitem_23 = None
    mul_18: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_19: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_18, primals_23)
    add_15: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_19, primals_24);  mul_19 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_3: "f32[2304]" = torch.ops.aten.cat.default([primals_25, primals_204, primals_26]);  primals_25 = primals_204 = primals_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_25: "f32[1576, 768]" = torch.ops.aten.view.default(add_15, [1576, 768]);  add_15 = None
    permute_15: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm_8: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_3, view_25, permute_15);  cat_3 = None
    view_26: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_8, [8, 197, 2304]);  addmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_27: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_26, [8, 197, 3, 12, -1]);  view_26 = None
    permute_16: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_27, [2, 0, 3, 1, 4]);  view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_2 = torch.ops.aten.unbind.int(permute_16);  permute_16 = None
    getitem_24: "f32[8, 12, 197, 64]" = unbind_2[0]
    getitem_25: "f32[8, 12, 197, 64]" = unbind_2[1]
    getitem_26: "f32[8, 12, 197, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_28: "i64[38809]" = torch.ops.aten.view.default(primals_205, [-1]);  primals_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_2: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_28, [view_28]);  primals_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_29: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_2, [197, 197, -1]);  index_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_17: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_29, [2, 0, 1]);  view_29 = None
    clone_9: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_17, memory_format = torch.contiguous_format);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_2: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_9, 0);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_2: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_2, [0, 3], 0.0);  unsqueeze_2 = None
    slice_3: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_2, -1, 0, 197);  constant_pad_nd_2 = None
    expand_3: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_3, [8, 12, 197, 197]);  slice_3 = None
    _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_24, getitem_25, getitem_26, expand_3, True)
    getitem_27: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_2[0]
    getitem_28: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_2[1]
    getitem_29: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
    getitem_30: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
    alias_2: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_18: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
    view_30: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_18, [8, 197, 768]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_31: "f32[1576, 768]" = torch.ops.aten.view.default(view_30, [1576, 768]);  view_30 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    addmm_9: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_139, view_31, permute_19);  primals_139 = None
    view_32: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_9, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_10: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_32);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_20: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_22, clone_10);  clone_10 = None
    add_16: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_13, mul_20);  add_13 = mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_5 = torch.ops.aten.var_mean.correction(add_16, [2], correction = 0, keepdim = True)
    getitem_31: "f32[8, 197, 1]" = var_mean_5[0]
    getitem_32: "f32[8, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    add_17: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-06);  getitem_31 = None
    rsqrt_5: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_17);  add_17 = None
    sub_5: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_16, getitem_32);  getitem_32 = None
    mul_21: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_22: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_30)
    add_18: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_22, primals_31);  mul_22 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_33: "f32[1576, 768]" = torch.ops.aten.view.default(add_18, [1576, 768]);  add_18 = None
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_10: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_141, view_33, permute_20);  primals_141 = None
    view_34: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_23: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
    mul_24: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_34, 0.7071067811865476);  view_34 = None
    erf_2: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_24);  mul_24 = None
    add_19: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_25: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_23, add_19);  mul_23 = add_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_11: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_25);  mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_35: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_11, [1576, 3072]);  clone_11 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    addmm_11: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_143, view_35, permute_21);  primals_143 = None
    view_36: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_11, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_12: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_26: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_29, clone_12);  clone_12 = None
    add_20: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_16, mul_26);  add_16 = mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_6 = torch.ops.aten.var_mean.correction(add_20, [2], correction = 0, keepdim = True)
    getitem_33: "f32[8, 197, 1]" = var_mean_6[0]
    getitem_34: "f32[8, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    add_21: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
    rsqrt_6: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    sub_6: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_20, getitem_34);  getitem_34 = None
    mul_27: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_28: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_27, primals_33)
    add_22: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_28, primals_34);  mul_28 = primals_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_4: "f32[2304]" = torch.ops.aten.cat.default([primals_35, primals_206, primals_36]);  primals_35 = primals_206 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_37: "f32[1576, 768]" = torch.ops.aten.view.default(add_22, [1576, 768]);  add_22 = None
    permute_22: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_12: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_4, view_37, permute_22);  cat_4 = None
    view_38: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_12, [8, 197, 2304]);  addmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_39: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_38, [8, 197, 3, 12, -1]);  view_38 = None
    permute_23: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_39, [2, 0, 3, 1, 4]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_3 = torch.ops.aten.unbind.int(permute_23);  permute_23 = None
    getitem_35: "f32[8, 12, 197, 64]" = unbind_3[0]
    getitem_36: "f32[8, 12, 197, 64]" = unbind_3[1]
    getitem_37: "f32[8, 12, 197, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_40: "i64[38809]" = torch.ops.aten.view.default(primals_207, [-1]);  primals_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_3: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_38, [view_40]);  primals_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_41: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_3, [197, 197, -1]);  index_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_24: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_41, [2, 0, 1]);  view_41 = None
    clone_13: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_24, memory_format = torch.contiguous_format);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_3: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_13, 0);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_3: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_3, [0, 3], 0.0);  unsqueeze_3 = None
    slice_4: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_3, -1, 0, 197);  constant_pad_nd_3 = None
    expand_4: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_4, [8, 12, 197, 197]);  slice_4 = None
    _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_35, getitem_36, getitem_37, expand_4, True)
    getitem_38: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_3[0]
    getitem_39: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_3[1]
    getitem_40: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
    getitem_41: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
    alias_3: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_25: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
    view_42: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_25, [8, 197, 768]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_43: "f32[1576, 768]" = torch.ops.aten.view.default(view_42, [1576, 768]);  view_42 = None
    permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    addmm_13: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_145, view_43, permute_26);  primals_145 = None
    view_44: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_13, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_14: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_29: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_32, clone_14);  clone_14 = None
    add_23: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_20, mul_29);  add_20 = mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_7 = torch.ops.aten.var_mean.correction(add_23, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 197, 1]" = var_mean_7[0]
    getitem_43: "f32[8, 197, 1]" = var_mean_7[1];  var_mean_7 = None
    add_24: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_7: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_24);  add_24 = None
    sub_7: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_23, getitem_43);  getitem_43 = None
    mul_30: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_31: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_40)
    add_25: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_41);  mul_31 = primals_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_45: "f32[1576, 768]" = torch.ops.aten.view.default(add_25, [1576, 768]);  add_25 = None
    permute_27: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_14: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_147, view_45, permute_27);  primals_147 = None
    view_46: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_14, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
    mul_33: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_46, 0.7071067811865476);  view_46 = None
    erf_3: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_26: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_34: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_26);  mul_32 = add_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_15: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_47: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_15, [1576, 3072]);  clone_15 = None
    permute_28: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_15: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_149, view_47, permute_28);  primals_149 = None
    view_48: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_15, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_16: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_48);  view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_35: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_39, clone_16);  clone_16 = None
    add_27: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_23, mul_35);  add_23 = mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_8 = torch.ops.aten.var_mean.correction(add_27, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 197, 1]" = var_mean_8[0]
    getitem_45: "f32[8, 197, 1]" = var_mean_8[1];  var_mean_8 = None
    add_28: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_8: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    sub_8: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_27, getitem_45);  getitem_45 = None
    mul_36: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_37: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_36, primals_43)
    add_29: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_37, primals_44);  mul_37 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_5: "f32[2304]" = torch.ops.aten.cat.default([primals_45, primals_208, primals_46]);  primals_45 = primals_208 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_49: "f32[1576, 768]" = torch.ops.aten.view.default(add_29, [1576, 768]);  add_29 = None
    permute_29: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_16: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_5, view_49, permute_29);  cat_5 = None
    view_50: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_16, [8, 197, 2304]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_51: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_50, [8, 197, 3, 12, -1]);  view_50 = None
    permute_30: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_51, [2, 0, 3, 1, 4]);  view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_4 = torch.ops.aten.unbind.int(permute_30);  permute_30 = None
    getitem_46: "f32[8, 12, 197, 64]" = unbind_4[0]
    getitem_47: "f32[8, 12, 197, 64]" = unbind_4[1]
    getitem_48: "f32[8, 12, 197, 64]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_52: "i64[38809]" = torch.ops.aten.view.default(primals_209, [-1]);  primals_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_4: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_48, [view_52]);  primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_53: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_4, [197, 197, -1]);  index_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_31: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_53, [2, 0, 1]);  view_53 = None
    clone_17: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_31, memory_format = torch.contiguous_format);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_4: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_17, 0);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_4: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_4, [0, 3], 0.0);  unsqueeze_4 = None
    slice_5: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_4, -1, 0, 197);  constant_pad_nd_4 = None
    expand_5: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_5, [8, 12, 197, 197]);  slice_5 = None
    _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_46, getitem_47, getitem_48, expand_5, True)
    getitem_49: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_4[0]
    getitem_50: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_4[1]
    getitem_51: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
    getitem_52: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
    alias_4: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_32: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
    view_54: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_32, [8, 197, 768]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_55: "f32[1576, 768]" = torch.ops.aten.view.default(view_54, [1576, 768]);  view_54 = None
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    addmm_17: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_151, view_55, permute_33);  primals_151 = None
    view_56: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_17, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_18: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_56);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_38: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_42, clone_18);  clone_18 = None
    add_30: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_27, mul_38);  add_27 = mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_9 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
    getitem_53: "f32[8, 197, 1]" = var_mean_9[0]
    getitem_54: "f32[8, 197, 1]" = var_mean_9[1];  var_mean_9 = None
    add_31: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-06);  getitem_53 = None
    rsqrt_9: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
    sub_9: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_30, getitem_54);  getitem_54 = None
    mul_39: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_40: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_39, primals_50)
    add_32: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_40, primals_51);  mul_40 = primals_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_57: "f32[1576, 768]" = torch.ops.aten.view.default(add_32, [1576, 768]);  add_32 = None
    permute_34: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_18: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_153, view_57, permute_34);  primals_153 = None
    view_58: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_18, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_41: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_42: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
    erf_4: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
    add_33: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_43: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_41, add_33);  mul_41 = add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_19: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_43);  mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_59: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_19, [1576, 3072]);  clone_19 = None
    permute_35: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_19: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_155, view_59, permute_35);  primals_155 = None
    view_60: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_19, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_20: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_44: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_49, clone_20);  clone_20 = None
    add_34: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_30, mul_44);  add_30 = mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_10 = torch.ops.aten.var_mean.correction(add_34, [2], correction = 0, keepdim = True)
    getitem_55: "f32[8, 197, 1]" = var_mean_10[0]
    getitem_56: "f32[8, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    add_35: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-06);  getitem_55 = None
    rsqrt_10: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    sub_10: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_34, getitem_56);  getitem_56 = None
    mul_45: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_46: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_45, primals_53)
    add_36: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_46, primals_54);  mul_46 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_6: "f32[2304]" = torch.ops.aten.cat.default([primals_55, primals_210, primals_56]);  primals_55 = primals_210 = primals_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_61: "f32[1576, 768]" = torch.ops.aten.view.default(add_36, [1576, 768]);  add_36 = None
    permute_36: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    addmm_20: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_6, view_61, permute_36);  cat_6 = None
    view_62: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_20, [8, 197, 2304]);  addmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_63: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_62, [8, 197, 3, 12, -1]);  view_62 = None
    permute_37: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_5 = torch.ops.aten.unbind.int(permute_37);  permute_37 = None
    getitem_57: "f32[8, 12, 197, 64]" = unbind_5[0]
    getitem_58: "f32[8, 12, 197, 64]" = unbind_5[1]
    getitem_59: "f32[8, 12, 197, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_64: "i64[38809]" = torch.ops.aten.view.default(primals_211, [-1]);  primals_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_5: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_58, [view_64]);  primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_65: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_5, [197, 197, -1]);  index_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_38: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_65, [2, 0, 1]);  view_65 = None
    clone_21: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_38, memory_format = torch.contiguous_format);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_5: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_21, 0);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_5: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_5, [0, 3], 0.0);  unsqueeze_5 = None
    slice_6: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_5, -1, 0, 197);  constant_pad_nd_5 = None
    expand_6: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_6, [8, 12, 197, 197]);  slice_6 = None
    _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_57, getitem_58, getitem_59, expand_6, True)
    getitem_60: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_5[0]
    getitem_61: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_5[1]
    getitem_62: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
    getitem_63: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
    alias_5: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_39: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
    view_66: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_39, [8, 197, 768]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_67: "f32[1576, 768]" = torch.ops.aten.view.default(view_66, [1576, 768]);  view_66 = None
    permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    addmm_21: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_157, view_67, permute_40);  primals_157 = None
    view_68: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_21, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_22: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_68);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_47: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_52, clone_22);  clone_22 = None
    add_37: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_34, mul_47);  add_34 = mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_11 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 197, 1]" = var_mean_11[0]
    getitem_65: "f32[8, 197, 1]" = var_mean_11[1];  var_mean_11 = None
    add_38: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_11: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
    sub_11: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_37, getitem_65);  getitem_65 = None
    mul_48: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_49: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_48, primals_60)
    add_39: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_49, primals_61);  mul_49 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_69: "f32[1576, 768]" = torch.ops.aten.view.default(add_39, [1576, 768]);  add_39 = None
    permute_41: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    addmm_22: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_159, view_69, permute_41);  primals_159 = None
    view_70: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_50: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
    mul_51: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_70, 0.7071067811865476);  view_70 = None
    erf_5: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
    add_40: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_52: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_50, add_40);  mul_50 = add_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_52);  mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_71: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_23, [1576, 3072]);  clone_23 = None
    permute_42: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    addmm_23: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_161, view_71, permute_42);  primals_161 = None
    view_72: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_23, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_72);  view_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_53: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_59, clone_24);  clone_24 = None
    add_41: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_37, mul_53);  add_37 = mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_12 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 197, 1]" = var_mean_12[0]
    getitem_67: "f32[8, 197, 1]" = var_mean_12[1];  var_mean_12 = None
    add_42: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_12: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    sub_12: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_41, getitem_67);  getitem_67 = None
    mul_54: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_55: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_54, primals_63)
    add_43: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_55, primals_64);  mul_55 = primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_7: "f32[2304]" = torch.ops.aten.cat.default([primals_65, primals_212, primals_66]);  primals_65 = primals_212 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_73: "f32[1576, 768]" = torch.ops.aten.view.default(add_43, [1576, 768]);  add_43 = None
    permute_43: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    addmm_24: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_7, view_73, permute_43);  cat_7 = None
    view_74: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_24, [8, 197, 2304]);  addmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_75: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_74, [8, 197, 3, 12, -1]);  view_74 = None
    permute_44: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_75, [2, 0, 3, 1, 4]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_6 = torch.ops.aten.unbind.int(permute_44);  permute_44 = None
    getitem_68: "f32[8, 12, 197, 64]" = unbind_6[0]
    getitem_69: "f32[8, 12, 197, 64]" = unbind_6[1]
    getitem_70: "f32[8, 12, 197, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_76: "i64[38809]" = torch.ops.aten.view.default(primals_213, [-1]);  primals_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_6: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_68, [view_76]);  primals_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_77: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_6, [197, 197, -1]);  index_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_45: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_77, [2, 0, 1]);  view_77 = None
    clone_25: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_45, memory_format = torch.contiguous_format);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_6: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_25, 0);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_6: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_6, [0, 3], 0.0);  unsqueeze_6 = None
    slice_7: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_6, -1, 0, 197);  constant_pad_nd_6 = None
    expand_7: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_7, [8, 12, 197, 197]);  slice_7 = None
    _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_68, getitem_69, getitem_70, expand_7, True)
    getitem_71: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_6[0]
    getitem_72: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_6[1]
    getitem_73: "i64[]" = _scaled_dot_product_efficient_attention_6[2]
    getitem_74: "i64[]" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None
    alias_6: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_46: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
    view_78: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_46, [8, 197, 768]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_79: "f32[1576, 768]" = torch.ops.aten.view.default(view_78, [1576, 768]);  view_78 = None
    permute_47: "f32[768, 768]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    addmm_25: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_163, view_79, permute_47);  primals_163 = None
    view_80: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_25, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_26: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_56: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_62, clone_26);  clone_26 = None
    add_44: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_41, mul_56);  add_41 = mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_13 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
    getitem_75: "f32[8, 197, 1]" = var_mean_13[0]
    getitem_76: "f32[8, 197, 1]" = var_mean_13[1];  var_mean_13 = None
    add_45: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-06);  getitem_75 = None
    rsqrt_13: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
    sub_13: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_44, getitem_76);  getitem_76 = None
    mul_57: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_58: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_57, primals_70)
    add_46: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_58, primals_71);  mul_58 = primals_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_81: "f32[1576, 768]" = torch.ops.aten.view.default(add_46, [1576, 768]);  add_46 = None
    permute_48: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
    addmm_26: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_165, view_81, permute_48);  primals_165 = None
    view_82: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_26, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_59: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_82, 0.5)
    mul_60: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_82, 0.7071067811865476);  view_82 = None
    erf_6: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_47: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_61: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_59, add_47);  mul_59 = add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_27: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_61);  mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_83: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_27, [1576, 3072]);  clone_27 = None
    permute_49: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    addmm_27: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_167, view_83, permute_49);  primals_167 = None
    view_84: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_27, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_28: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_84);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_62: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_69, clone_28);  clone_28 = None
    add_48: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_44, mul_62);  add_44 = mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_14 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
    getitem_77: "f32[8, 197, 1]" = var_mean_14[0]
    getitem_78: "f32[8, 197, 1]" = var_mean_14[1];  var_mean_14 = None
    add_49: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
    rsqrt_14: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    sub_14: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_48, getitem_78);  getitem_78 = None
    mul_63: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_64: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_63, primals_73)
    add_50: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_64, primals_74);  mul_64 = primals_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_8: "f32[2304]" = torch.ops.aten.cat.default([primals_75, primals_214, primals_76]);  primals_75 = primals_214 = primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_85: "f32[1576, 768]" = torch.ops.aten.view.default(add_50, [1576, 768]);  add_50 = None
    permute_50: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm_28: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_8, view_85, permute_50);  cat_8 = None
    view_86: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_28, [8, 197, 2304]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_87: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_86, [8, 197, 3, 12, -1]);  view_86 = None
    permute_51: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_87, [2, 0, 3, 1, 4]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_7 = torch.ops.aten.unbind.int(permute_51);  permute_51 = None
    getitem_79: "f32[8, 12, 197, 64]" = unbind_7[0]
    getitem_80: "f32[8, 12, 197, 64]" = unbind_7[1]
    getitem_81: "f32[8, 12, 197, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_88: "i64[38809]" = torch.ops.aten.view.default(primals_215, [-1]);  primals_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_7: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_78, [view_88]);  primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_89: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_7, [197, 197, -1]);  index_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_52: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_89, [2, 0, 1]);  view_89 = None
    clone_29: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_52, memory_format = torch.contiguous_format);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_7: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_29, 0);  clone_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_7: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_7, [0, 3], 0.0);  unsqueeze_7 = None
    slice_8: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_7, -1, 0, 197);  constant_pad_nd_7 = None
    expand_8: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_8, [8, 12, 197, 197]);  slice_8 = None
    _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_79, getitem_80, getitem_81, expand_8, True)
    getitem_82: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_7[0]
    getitem_83: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_7[1]
    getitem_84: "i64[]" = _scaled_dot_product_efficient_attention_7[2]
    getitem_85: "i64[]" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None
    alias_7: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_82)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_53: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
    view_90: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_53, [8, 197, 768]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_91: "f32[1576, 768]" = torch.ops.aten.view.default(view_90, [1576, 768]);  view_90 = None
    permute_54: "f32[768, 768]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    addmm_29: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_169, view_91, permute_54);  primals_169 = None
    view_92: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_29, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_30: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_92);  view_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_65: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_72, clone_30);  clone_30 = None
    add_51: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_48, mul_65);  add_48 = mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_15 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 197, 1]" = var_mean_15[0]
    getitem_87: "f32[8, 197, 1]" = var_mean_15[1];  var_mean_15 = None
    add_52: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
    rsqrt_15: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
    sub_15: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_51, getitem_87);  getitem_87 = None
    mul_66: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_67: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_66, primals_80)
    add_53: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_67, primals_81);  mul_67 = primals_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_93: "f32[1576, 768]" = torch.ops.aten.view.default(add_53, [1576, 768]);  add_53 = None
    permute_55: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    addmm_30: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_171, view_93, permute_55);  primals_171 = None
    view_94: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_30, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_68: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
    mul_69: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_94, 0.7071067811865476);  view_94 = None
    erf_7: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_54: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_70: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_68, add_54);  mul_68 = add_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_31: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_70);  mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_95: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_31, [1576, 3072]);  clone_31 = None
    permute_56: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    addmm_31: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_173, view_95, permute_56);  primals_173 = None
    view_96: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_31, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_32: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_71: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_79, clone_32);  clone_32 = None
    add_55: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_51, mul_71);  add_51 = mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_16 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 197, 1]" = var_mean_16[0]
    getitem_89: "f32[8, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    add_56: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_16: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    sub_16: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_55, getitem_89);  getitem_89 = None
    mul_72: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_73: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_83)
    add_57: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_84);  mul_73 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_9: "f32[2304]" = torch.ops.aten.cat.default([primals_85, primals_216, primals_86]);  primals_85 = primals_216 = primals_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_97: "f32[1576, 768]" = torch.ops.aten.view.default(add_57, [1576, 768]);  add_57 = None
    permute_57: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_32: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_9, view_97, permute_57);  cat_9 = None
    view_98: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_32, [8, 197, 2304]);  addmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_99: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_98, [8, 197, 3, 12, -1]);  view_98 = None
    permute_58: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_99, [2, 0, 3, 1, 4]);  view_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_8 = torch.ops.aten.unbind.int(permute_58);  permute_58 = None
    getitem_90: "f32[8, 12, 197, 64]" = unbind_8[0]
    getitem_91: "f32[8, 12, 197, 64]" = unbind_8[1]
    getitem_92: "f32[8, 12, 197, 64]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_100: "i64[38809]" = torch.ops.aten.view.default(primals_217, [-1]);  primals_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_8: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_88, [view_100]);  primals_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_101: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_8, [197, 197, -1]);  index_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_59: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_101, [2, 0, 1]);  view_101 = None
    clone_33: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_59, memory_format = torch.contiguous_format);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_8: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_33, 0);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_8: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_8, [0, 3], 0.0);  unsqueeze_8 = None
    slice_9: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_8, -1, 0, 197);  constant_pad_nd_8 = None
    expand_9: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_9, [8, 12, 197, 197]);  slice_9 = None
    _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_90, getitem_91, getitem_92, expand_9, True)
    getitem_93: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_8[0]
    getitem_94: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_8[1]
    getitem_95: "i64[]" = _scaled_dot_product_efficient_attention_8[2]
    getitem_96: "i64[]" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None
    alias_8: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_93)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_60: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
    view_102: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_60, [8, 197, 768]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_103: "f32[1576, 768]" = torch.ops.aten.view.default(view_102, [1576, 768]);  view_102 = None
    permute_61: "f32[768, 768]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    addmm_33: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_175, view_103, permute_61);  primals_175 = None
    view_104: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_33, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_34: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_74: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_82, clone_34);  clone_34 = None
    add_58: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_55, mul_74);  add_55 = mul_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_17 = torch.ops.aten.var_mean.correction(add_58, [2], correction = 0, keepdim = True)
    getitem_97: "f32[8, 197, 1]" = var_mean_17[0]
    getitem_98: "f32[8, 197, 1]" = var_mean_17[1];  var_mean_17 = None
    add_59: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-06);  getitem_97 = None
    rsqrt_17: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_59);  add_59 = None
    sub_17: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_58, getitem_98);  getitem_98 = None
    mul_75: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_76: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_75, primals_90)
    add_60: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_76, primals_91);  mul_76 = primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_105: "f32[1576, 768]" = torch.ops.aten.view.default(add_60, [1576, 768]);  add_60 = None
    permute_62: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    addmm_34: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_177, view_105, permute_62);  primals_177 = None
    view_106: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_77: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
    mul_78: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476);  view_106 = None
    erf_8: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_61: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_79: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_77, add_61);  mul_77 = add_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_35: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_79);  mul_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_107: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_35, [1576, 3072]);  clone_35 = None
    permute_63: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    addmm_35: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_179, view_107, permute_63);  primals_179 = None
    view_108: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_35, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_36: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_80: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_89, clone_36);  clone_36 = None
    add_62: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_58, mul_80);  add_58 = mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_18 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
    getitem_99: "f32[8, 197, 1]" = var_mean_18[0]
    getitem_100: "f32[8, 197, 1]" = var_mean_18[1];  var_mean_18 = None
    add_63: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-06);  getitem_99 = None
    rsqrt_18: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
    sub_18: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_62, getitem_100);  getitem_100 = None
    mul_81: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_82: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_81, primals_93)
    add_64: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_82, primals_94);  mul_82 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_10: "f32[2304]" = torch.ops.aten.cat.default([primals_95, primals_218, primals_96]);  primals_95 = primals_218 = primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_109: "f32[1576, 768]" = torch.ops.aten.view.default(add_64, [1576, 768]);  add_64 = None
    permute_64: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_36: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_10, view_109, permute_64);  cat_10 = None
    view_110: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_36, [8, 197, 2304]);  addmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_111: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_110, [8, 197, 3, 12, -1]);  view_110 = None
    permute_65: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_9 = torch.ops.aten.unbind.int(permute_65);  permute_65 = None
    getitem_101: "f32[8, 12, 197, 64]" = unbind_9[0]
    getitem_102: "f32[8, 12, 197, 64]" = unbind_9[1]
    getitem_103: "f32[8, 12, 197, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_112: "i64[38809]" = torch.ops.aten.view.default(primals_219, [-1]);  primals_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_9: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_98, [view_112]);  primals_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_113: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_9, [197, 197, -1]);  index_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_66: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_113, [2, 0, 1]);  view_113 = None
    clone_37: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_66, memory_format = torch.contiguous_format);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_9: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_37, 0);  clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_9: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_9, [0, 3], 0.0);  unsqueeze_9 = None
    slice_10: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_9, -1, 0, 197);  constant_pad_nd_9 = None
    expand_10: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_10, [8, 12, 197, 197]);  slice_10 = None
    _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_101, getitem_102, getitem_103, expand_10, True)
    getitem_104: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_9[0]
    getitem_105: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_9[1]
    getitem_106: "i64[]" = _scaled_dot_product_efficient_attention_9[2]
    getitem_107: "i64[]" = _scaled_dot_product_efficient_attention_9[3];  _scaled_dot_product_efficient_attention_9 = None
    alias_9: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_67: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
    view_114: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_67, [8, 197, 768]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_115: "f32[1576, 768]" = torch.ops.aten.view.default(view_114, [1576, 768]);  view_114 = None
    permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    addmm_37: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_181, view_115, permute_68);  primals_181 = None
    view_116: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_37, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_38: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_116);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_83: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_92, clone_38);  clone_38 = None
    add_65: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_62, mul_83);  add_62 = mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_19 = torch.ops.aten.var_mean.correction(add_65, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 197, 1]" = var_mean_19[0]
    getitem_109: "f32[8, 197, 1]" = var_mean_19[1];  var_mean_19 = None
    add_66: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_19: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_19: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_65, getitem_109);  getitem_109 = None
    mul_84: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_85: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_84, primals_100)
    add_67: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_85, primals_101);  mul_85 = primals_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[1576, 768]" = torch.ops.aten.view.default(add_67, [1576, 768]);  add_67 = None
    permute_69: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    addmm_38: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_183, view_117, permute_69);  primals_183 = None
    view_118: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_38, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_86: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_87: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
    erf_9: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_87);  mul_87 = None
    add_68: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_88: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_86, add_68);  mul_86 = add_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_39: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_88);  mul_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_119: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_39, [1576, 3072]);  clone_39 = None
    permute_70: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    addmm_39: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_185, view_119, permute_70);  primals_185 = None
    view_120: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_39, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_40: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_89: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_99, clone_40);  clone_40 = None
    add_69: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_65, mul_89);  add_65 = mul_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_20 = torch.ops.aten.var_mean.correction(add_69, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_111: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_70: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_70);  add_70 = None
    sub_20: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_69, getitem_111);  getitem_111 = None
    mul_90: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_91: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_90, primals_103)
    add_71: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_91, primals_104);  mul_91 = primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_11: "f32[2304]" = torch.ops.aten.cat.default([primals_105, primals_220, primals_106]);  primals_105 = primals_220 = primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_121: "f32[1576, 768]" = torch.ops.aten.view.default(add_71, [1576, 768]);  add_71 = None
    permute_71: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_40: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_11, view_121, permute_71);  cat_11 = None
    view_122: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_40, [8, 197, 2304]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_123: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_122, [8, 197, 3, 12, -1]);  view_122 = None
    permute_72: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_123, [2, 0, 3, 1, 4]);  view_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_10 = torch.ops.aten.unbind.int(permute_72);  permute_72 = None
    getitem_112: "f32[8, 12, 197, 64]" = unbind_10[0]
    getitem_113: "f32[8, 12, 197, 64]" = unbind_10[1]
    getitem_114: "f32[8, 12, 197, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_124: "i64[38809]" = torch.ops.aten.view.default(primals_221, [-1]);  primals_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_10: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_108, [view_124]);  primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_125: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_10, [197, 197, -1]);  index_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_73: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_125, [2, 0, 1]);  view_125 = None
    clone_41: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_10: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_41, 0);  clone_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_10: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_10, [0, 3], 0.0);  unsqueeze_10 = None
    slice_11: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_10, -1, 0, 197);  constant_pad_nd_10 = None
    expand_11: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_11, [8, 12, 197, 197]);  slice_11 = None
    _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_112, getitem_113, getitem_114, expand_11, True)
    getitem_115: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_10[0]
    getitem_116: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_10[1]
    getitem_117: "i64[]" = _scaled_dot_product_efficient_attention_10[2]
    getitem_118: "i64[]" = _scaled_dot_product_efficient_attention_10[3];  _scaled_dot_product_efficient_attention_10 = None
    alias_10: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_74: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
    view_126: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_74, [8, 197, 768]);  permute_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_127: "f32[1576, 768]" = torch.ops.aten.view.default(view_126, [1576, 768]);  view_126 = None
    permute_75: "f32[768, 768]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    addmm_41: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_187, view_127, permute_75);  primals_187 = None
    view_128: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_41, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_42: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_128);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_92: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_102, clone_42);  clone_42 = None
    add_72: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_69, mul_92);  add_69 = mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_21 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_119: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_120: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_73: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-06);  getitem_119 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_21: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_72, getitem_120);  getitem_120 = None
    mul_93: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_94: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_93, primals_110)
    add_74: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_94, primals_111);  mul_94 = primals_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_129: "f32[1576, 768]" = torch.ops.aten.view.default(add_74, [1576, 768]);  add_74 = None
    permute_76: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    addmm_42: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_189, view_129, permute_76);  primals_189 = None
    view_130: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_42, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_95: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_130, 0.5)
    mul_96: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_130, 0.7071067811865476);  view_130 = None
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_96);  mul_96 = None
    add_75: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_97: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_95, add_75);  mul_95 = add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_43: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_97);  mul_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_131: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_43, [1576, 3072]);  clone_43 = None
    permute_77: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    addmm_43: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_191, view_131, permute_77);  primals_191 = None
    view_132: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_43, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_44: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_132);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_98: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_109, clone_44);  clone_44 = None
    add_76: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_72, mul_98);  add_72 = mul_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_22 = torch.ops.aten.var_mean.correction(add_76, [2], correction = 0, keepdim = True)
    getitem_121: "f32[8, 197, 1]" = var_mean_22[0]
    getitem_122: "f32[8, 197, 1]" = var_mean_22[1];  var_mean_22 = None
    add_77: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-06);  getitem_121 = None
    rsqrt_22: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_77);  add_77 = None
    sub_22: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_76, getitem_122);  getitem_122 = None
    mul_99: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_99, primals_113)
    add_78: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_100, primals_114);  mul_100 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    cat_12: "f32[2304]" = torch.ops.aten.cat.default([primals_115, primals_222, primals_116]);  primals_115 = primals_222 = primals_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    view_133: "f32[1576, 768]" = torch.ops.aten.view.default(add_78, [1576, 768]);  add_78 = None
    permute_78: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_44: "f32[1576, 2304]" = torch.ops.aten.addmm.default(cat_12, view_133, permute_78);  cat_12 = None
    view_134: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_44, [8, 197, 2304]);  addmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    view_135: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_134, [8, 197, 3, 12, -1]);  view_134 = None
    permute_79: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_135, [2, 0, 3, 1, 4]);  view_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_11 = torch.ops.aten.unbind.int(permute_79);  permute_79 = None
    getitem_123: "f32[8, 12, 197, 64]" = unbind_11[0]
    getitem_124: "f32[8, 12, 197, 64]" = unbind_11[1]
    getitem_125: "f32[8, 12, 197, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_136: "i64[38809]" = torch.ops.aten.view.default(primals_223, [-1]);  primals_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    index_11: "f32[38809, 12]" = torch.ops.aten.index.Tensor(primals_118, [view_136]);  primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    view_137: "f32[197, 197, 12]" = torch.ops.aten.view.default(index_11, [197, 197, -1]);  index_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_80: "f32[12, 197, 197]" = torch.ops.aten.permute.default(view_137, [2, 0, 1]);  view_137 = None
    clone_45: "f32[12, 197, 197]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    unsqueeze_11: "f32[1, 12, 197, 197]" = torch.ops.aten.unsqueeze.default(clone_45, 0);  clone_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    constant_pad_nd_11: "f32[1, 12, 197, 200]" = torch.ops.aten.constant_pad_nd.default(unsqueeze_11, [0, 3], 0.0);  unsqueeze_11 = None
    slice_12: "f32[1, 12, 197, 197]" = torch.ops.aten.slice.Tensor(constant_pad_nd_11, -1, 0, 197);  constant_pad_nd_11 = None
    expand_12: "f32[8, 12, 197, 197]" = torch.ops.aten.expand.default(slice_12, [8, 12, 197, 197]);  slice_12 = None
    _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_123, getitem_124, getitem_125, expand_12, True)
    getitem_126: "f32[8, 12, 197, 64]" = _scaled_dot_product_efficient_attention_11[0]
    getitem_127: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_11[1]
    getitem_128: "i64[]" = _scaled_dot_product_efficient_attention_11[2]
    getitem_129: "i64[]" = _scaled_dot_product_efficient_attention_11[3];  _scaled_dot_product_efficient_attention_11 = None
    alias_11: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_126)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_81: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
    view_138: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_81, [8, 197, 768]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    view_139: "f32[1576, 768]" = torch.ops.aten.view.default(view_138, [1576, 768]);  view_138 = None
    permute_82: "f32[768, 768]" = torch.ops.aten.permute.default(primals_192, [1, 0]);  primals_192 = None
    addmm_45: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_193, view_139, permute_82);  primals_193 = None
    view_140: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_45, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    clone_46: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_140);  view_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_112, clone_46);  clone_46 = None
    add_79: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_76, mul_101);  add_76 = mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_23 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_130: "f32[8, 197, 1]" = var_mean_23[0]
    getitem_131: "f32[8, 197, 1]" = var_mean_23[1];  var_mean_23 = None
    add_80: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-06);  getitem_130 = None
    rsqrt_23: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_23: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_79, getitem_131);  getitem_131 = None
    mul_102: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_103: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_102, primals_120)
    add_81: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_103, primals_121);  mul_103 = primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_141: "f32[1576, 768]" = torch.ops.aten.view.default(add_81, [1576, 768]);  add_81 = None
    permute_83: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_194, [1, 0]);  primals_194 = None
    addmm_46: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_195, view_141, permute_83);  primals_195 = None
    view_142: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_104: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
    mul_105: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_142, 0.7071067811865476);  view_142 = None
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_105);  mul_105 = None
    add_82: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_106: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_104, add_82);  mul_104 = add_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_47: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_106);  mul_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_143: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_47, [1576, 3072]);  clone_47 = None
    permute_84: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_196, [1, 0]);  primals_196 = None
    addmm_47: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_197, view_143, permute_84);  primals_197 = None
    view_144: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_47, [8, 197, 768])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_48: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_144);  view_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_107: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(primals_119, clone_48);  clone_48 = None
    add_83: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_79, mul_107);  add_79 = mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:421, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    slice_13: "f32[8, 197, 768]" = torch.ops.aten.slice.Tensor(add_83, 0, 0, 9223372036854775807);  add_83 = None
    slice_14: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(slice_13, 1, 1, 9223372036854775807);  slice_13 = None
    mean: "f32[8, 768]" = torch.ops.aten.mean.dim(slice_14, [1]);  slice_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    var_mean_24 = torch.ops.aten.var_mean.correction(mean, [1], correction = 0, keepdim = True)
    getitem_132: "f32[8, 1]" = var_mean_24[0]
    getitem_133: "f32[8, 1]" = var_mean_24[1];  var_mean_24 = None
    add_84: "f32[8, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_24: "f32[8, 1]" = torch.ops.aten.rsqrt.default(add_84);  add_84 = None
    sub_24: "f32[8, 768]" = torch.ops.aten.sub.Tensor(mean, getitem_133);  mean = getitem_133 = None
    mul_108: "f32[8, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_109: "f32[8, 768]" = torch.ops.aten.mul.Tensor(mul_108, primals_122)
    add_85: "f32[8, 768]" = torch.ops.aten.add.Tensor(mul_109, primals_123);  mul_109 = primals_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:423, code: x = self.head_drop(x)
    clone_49: "f32[8, 768]" = torch.ops.aten.clone.default(add_85);  add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:424, code: return x if pre_logits else self.head(x)
    permute_85: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_198, [1, 0]);  primals_198 = None
    addmm_48: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_199, clone_49, permute_85);  primals_199 = None
    permute_86: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div: "f32[8, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_90: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_94: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_2: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_98: "f32[768, 768]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    alias_12: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_105: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_3: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_109: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_113: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_4: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_117: "f32[768, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    alias_13: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_124: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_5: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_128: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_132: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_6: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_136: "f32[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    alias_14: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_143: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_7: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_147: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_151: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_8: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_155: "f32[768, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    alias_15: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_162: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_9: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_166: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_170: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_10: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_174: "f32[768, 768]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    alias_16: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_181: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_11: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_185: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_189: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_12: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_193: "f32[768, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    alias_17: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_200: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_13: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_204: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_208: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_14: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_212: "f32[768, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    alias_18: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_219: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_15: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_223: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_227: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_16: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_231: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    alias_19: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_238: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_17: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_242: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_246: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_18: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_250: "f32[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    alias_20: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_257: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_19: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_261: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_265: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_20: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    alias_21: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_276: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_21: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_280: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_284: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_22: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_288: "f32[768, 768]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    alias_22: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_295: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_23: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_299: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_303: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    div_24: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    permute_307: "f32[768, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    alias_23: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    permute_314: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    return [addmm_48, primals_2, primals_3, primals_9, primals_10, primals_12, primals_13, primals_19, primals_20, primals_22, primals_23, primals_29, primals_30, primals_32, primals_33, primals_39, primals_40, primals_42, primals_43, primals_49, primals_50, primals_52, primals_53, primals_59, primals_60, primals_62, primals_63, primals_69, primals_70, primals_72, primals_73, primals_79, primals_80, primals_82, primals_83, primals_89, primals_90, primals_92, primals_93, primals_99, primals_100, primals_102, primals_103, primals_109, primals_110, primals_112, primals_113, primals_119, primals_120, primals_122, primals_124, primals_224, cat, getitem_1, rsqrt, view_1, getitem_2, getitem_3, getitem_4, view_4, expand_1, getitem_6, getitem_7, getitem_8, view_7, addmm_1, mul_3, view_9, addmm_2, view_11, addmm_3, mul_9, view_13, getitem_13, getitem_14, getitem_15, view_16, expand_2, getitem_17, getitem_18, getitem_19, view_19, addmm_5, mul_12, view_21, addmm_6, view_23, addmm_7, mul_18, view_25, getitem_24, getitem_25, getitem_26, view_28, expand_3, getitem_28, getitem_29, getitem_30, view_31, addmm_9, mul_21, view_33, addmm_10, view_35, addmm_11, mul_27, view_37, getitem_35, getitem_36, getitem_37, view_40, expand_4, getitem_39, getitem_40, getitem_41, view_43, addmm_13, mul_30, view_45, addmm_14, view_47, addmm_15, mul_36, view_49, getitem_46, getitem_47, getitem_48, view_52, expand_5, getitem_50, getitem_51, getitem_52, view_55, addmm_17, mul_39, view_57, addmm_18, view_59, addmm_19, mul_45, view_61, getitem_57, getitem_58, getitem_59, view_64, expand_6, getitem_61, getitem_62, getitem_63, view_67, addmm_21, mul_48, view_69, addmm_22, view_71, addmm_23, mul_54, view_73, getitem_68, getitem_69, getitem_70, view_76, expand_7, getitem_72, getitem_73, getitem_74, view_79, addmm_25, mul_57, view_81, addmm_26, view_83, addmm_27, mul_63, view_85, getitem_79, getitem_80, getitem_81, view_88, expand_8, getitem_83, getitem_84, getitem_85, view_91, addmm_29, mul_66, view_93, addmm_30, view_95, addmm_31, mul_72, view_97, getitem_90, getitem_91, getitem_92, view_100, expand_9, getitem_94, getitem_95, getitem_96, view_103, addmm_33, mul_75, view_105, addmm_34, view_107, addmm_35, mul_81, view_109, getitem_101, getitem_102, getitem_103, view_112, expand_10, getitem_105, getitem_106, getitem_107, view_115, addmm_37, mul_84, view_117, addmm_38, view_119, addmm_39, mul_90, view_121, getitem_112, getitem_113, getitem_114, view_124, expand_11, getitem_116, getitem_117, getitem_118, view_127, addmm_41, mul_93, view_129, addmm_42, view_131, addmm_43, mul_99, view_133, getitem_123, getitem_124, getitem_125, view_136, expand_12, getitem_127, getitem_128, getitem_129, view_139, addmm_45, mul_102, view_141, addmm_46, view_143, addmm_47, mul_108, clone_49, permute_86, div, permute_90, permute_94, div_2, permute_98, alias_12, permute_105, div_3, permute_109, permute_113, div_4, permute_117, alias_13, permute_124, div_5, permute_128, permute_132, div_6, permute_136, alias_14, permute_143, div_7, permute_147, permute_151, div_8, permute_155, alias_15, permute_162, div_9, permute_166, permute_170, div_10, permute_174, alias_16, permute_181, div_11, permute_185, permute_189, div_12, permute_193, alias_17, permute_200, div_13, permute_204, permute_208, div_14, permute_212, alias_18, permute_219, div_15, permute_223, permute_227, div_16, permute_231, alias_19, permute_238, div_17, permute_242, permute_246, div_18, permute_250, alias_20, permute_257, div_19, permute_261, permute_265, div_20, permute_269, alias_21, permute_276, div_21, permute_280, permute_284, div_22, permute_288, alias_22, permute_295, div_23, permute_299, permute_303, div_24, permute_307, alias_23, permute_314]
    