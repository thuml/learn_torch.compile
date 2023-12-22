from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1, 198, 768]"; primals_2: "f32[1, 1, 768]"; primals_3: "f32[1, 1, 768]"; primals_4: "f32[768, 3, 16, 16]"; primals_5: "f32[768]"; primals_6: "f32[768]"; primals_7: "f32[768]"; primals_8: "f32[2304, 768]"; primals_9: "f32[2304]"; primals_10: "f32[768, 768]"; primals_11: "f32[768]"; primals_12: "f32[768]"; primals_13: "f32[768]"; primals_14: "f32[3072, 768]"; primals_15: "f32[3072]"; primals_16: "f32[768, 3072]"; primals_17: "f32[768]"; primals_18: "f32[768]"; primals_19: "f32[768]"; primals_20: "f32[2304, 768]"; primals_21: "f32[2304]"; primals_22: "f32[768, 768]"; primals_23: "f32[768]"; primals_24: "f32[768]"; primals_25: "f32[768]"; primals_26: "f32[3072, 768]"; primals_27: "f32[3072]"; primals_28: "f32[768, 3072]"; primals_29: "f32[768]"; primals_30: "f32[768]"; primals_31: "f32[768]"; primals_32: "f32[2304, 768]"; primals_33: "f32[2304]"; primals_34: "f32[768, 768]"; primals_35: "f32[768]"; primals_36: "f32[768]"; primals_37: "f32[768]"; primals_38: "f32[3072, 768]"; primals_39: "f32[3072]"; primals_40: "f32[768, 3072]"; primals_41: "f32[768]"; primals_42: "f32[768]"; primals_43: "f32[768]"; primals_44: "f32[2304, 768]"; primals_45: "f32[2304]"; primals_46: "f32[768, 768]"; primals_47: "f32[768]"; primals_48: "f32[768]"; primals_49: "f32[768]"; primals_50: "f32[3072, 768]"; primals_51: "f32[3072]"; primals_52: "f32[768, 3072]"; primals_53: "f32[768]"; primals_54: "f32[768]"; primals_55: "f32[768]"; primals_56: "f32[2304, 768]"; primals_57: "f32[2304]"; primals_58: "f32[768, 768]"; primals_59: "f32[768]"; primals_60: "f32[768]"; primals_61: "f32[768]"; primals_62: "f32[3072, 768]"; primals_63: "f32[3072]"; primals_64: "f32[768, 3072]"; primals_65: "f32[768]"; primals_66: "f32[768]"; primals_67: "f32[768]"; primals_68: "f32[2304, 768]"; primals_69: "f32[2304]"; primals_70: "f32[768, 768]"; primals_71: "f32[768]"; primals_72: "f32[768]"; primals_73: "f32[768]"; primals_74: "f32[3072, 768]"; primals_75: "f32[3072]"; primals_76: "f32[768, 3072]"; primals_77: "f32[768]"; primals_78: "f32[768]"; primals_79: "f32[768]"; primals_80: "f32[2304, 768]"; primals_81: "f32[2304]"; primals_82: "f32[768, 768]"; primals_83: "f32[768]"; primals_84: "f32[768]"; primals_85: "f32[768]"; primals_86: "f32[3072, 768]"; primals_87: "f32[3072]"; primals_88: "f32[768, 3072]"; primals_89: "f32[768]"; primals_90: "f32[768]"; primals_91: "f32[768]"; primals_92: "f32[2304, 768]"; primals_93: "f32[2304]"; primals_94: "f32[768, 768]"; primals_95: "f32[768]"; primals_96: "f32[768]"; primals_97: "f32[768]"; primals_98: "f32[3072, 768]"; primals_99: "f32[3072]"; primals_100: "f32[768, 3072]"; primals_101: "f32[768]"; primals_102: "f32[768]"; primals_103: "f32[768]"; primals_104: "f32[2304, 768]"; primals_105: "f32[2304]"; primals_106: "f32[768, 768]"; primals_107: "f32[768]"; primals_108: "f32[768]"; primals_109: "f32[768]"; primals_110: "f32[3072, 768]"; primals_111: "f32[3072]"; primals_112: "f32[768, 3072]"; primals_113: "f32[768]"; primals_114: "f32[768]"; primals_115: "f32[768]"; primals_116: "f32[2304, 768]"; primals_117: "f32[2304]"; primals_118: "f32[768, 768]"; primals_119: "f32[768]"; primals_120: "f32[768]"; primals_121: "f32[768]"; primals_122: "f32[3072, 768]"; primals_123: "f32[3072]"; primals_124: "f32[768, 3072]"; primals_125: "f32[768]"; primals_126: "f32[768]"; primals_127: "f32[768]"; primals_128: "f32[2304, 768]"; primals_129: "f32[2304]"; primals_130: "f32[768, 768]"; primals_131: "f32[768]"; primals_132: "f32[768]"; primals_133: "f32[768]"; primals_134: "f32[3072, 768]"; primals_135: "f32[3072]"; primals_136: "f32[768, 3072]"; primals_137: "f32[768]"; primals_138: "f32[768]"; primals_139: "f32[768]"; primals_140: "f32[2304, 768]"; primals_141: "f32[2304]"; primals_142: "f32[768, 768]"; primals_143: "f32[768]"; primals_144: "f32[768]"; primals_145: "f32[768]"; primals_146: "f32[3072, 768]"; primals_147: "f32[3072]"; primals_148: "f32[768, 3072]"; primals_149: "f32[768]"; primals_150: "f32[768]"; primals_151: "f32[768]"; primals_152: "f32[1000, 768]"; primals_153: "f32[1000]"; primals_154: "f32[1000, 768]"; primals_155: "f32[1000]"; primals_156: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(primals_156, primals_4, primals_5, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 196]" = torch.ops.aten.view.default(convolution, [8, 768, 196]);  convolution = None
    permute: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:100, code: self.cls_token.expand(x.shape[0], -1, -1),
    expand: "f32[8, 1, 768]" = torch.ops.aten.expand.default(primals_2, [8, -1, -1]);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:101, code: self.dist_token.expand(x.shape[0], -1, -1),
    expand_1: "f32[8, 1, 768]" = torch.ops.aten.expand.default(primals_3, [8, -1, -1]);  primals_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:99, code: x = torch.cat((
    cat: "f32[8, 198, 768]" = torch.ops.aten.cat.default([expand, expand_1, permute], 1);  expand = expand_1 = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:104, code: x = x + pos_embed
    add: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(cat, primals_1);  cat = primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:105, code: return self.pos_drop(x)
    clone: "f32[8, 198, 768]" = torch.ops.aten.clone.default(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 198, 1]" = var_mean[0]
    getitem_1: "f32[8, 198, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(clone, getitem_1)
    mul: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul, primals_6);  mul = None
    add_2: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_7);  mul_1 = primals_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_1: "f32[1584, 768]" = torch.ops.aten.view.default(add_2, [1584, 768]);  add_2 = None
    permute_1: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
    addmm: "f32[1584, 2304]" = torch.ops.aten.addmm.default(primals_9, view_1, permute_1);  primals_9 = None
    view_2: "f32[8, 198, 2304]" = torch.ops.aten.view.default(addmm, [8, 198, 2304]);  addmm = None
    view_3: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.view.default(view_2, [8, 198, 3, 12, 64]);  view_2 = None
    permute_2: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
    getitem_2: "f32[8, 12, 198, 64]" = unbind[0]
    getitem_3: "f32[8, 12, 198, 64]" = unbind[1]
    getitem_4: "f32[8, 12, 198, 64]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_2, getitem_3, getitem_4, None, True)
    getitem_5: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention[0]
    getitem_6: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention[1]
    getitem_7: "i64[]" = _scaled_dot_product_efficient_attention[2]
    getitem_8: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
    alias: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(getitem_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_3: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    view_4: "f32[8, 198, 768]" = torch.ops.aten.view.default(permute_3, [8, 198, 768]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_5: "f32[1584, 768]" = torch.ops.aten.view.default(view_4, [1584, 768]);  view_4 = None
    permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
    addmm_1: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_11, view_5, permute_4);  primals_11 = None
    view_6: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_1, [8, 198, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_1: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_6);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_3: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(clone, clone_1);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_9: "f32[8, 198, 1]" = var_mean_1[0]
    getitem_10: "f32[8, 198, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-06);  getitem_9 = None
    rsqrt_1: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_1: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_10)
    mul_2: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_3: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_12);  mul_2 = None
    add_5: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_13);  mul_3 = primals_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_7: "f32[1584, 768]" = torch.ops.aten.view.default(add_5, [1584, 768]);  add_5 = None
    permute_5: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
    addmm_2: "f32[1584, 3072]" = torch.ops.aten.addmm.default(primals_15, view_7, permute_5);  primals_15 = None
    view_8: "f32[8, 198, 3072]" = torch.ops.aten.view.default(addmm_2, [8, 198, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_4: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_8, 0.5)
    mul_5: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476)
    erf: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_6: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_6: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_2: "f32[8, 198, 3072]" = torch.ops.aten.clone.default(mul_6);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_9: "f32[1584, 3072]" = torch.ops.aten.view.default(clone_2, [1584, 3072]);  clone_2 = None
    permute_6: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
    addmm_3: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_17, view_9, permute_6);  primals_17 = None
    view_10: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_3, [8, 198, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_3: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_10);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_7: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_3, clone_3);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_11: "f32[8, 198, 1]" = var_mean_2[0]
    getitem_12: "f32[8, 198, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-06);  getitem_11 = None
    rsqrt_2: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_2: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_12)
    mul_7: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_8: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_7, primals_18);  mul_7 = None
    add_9: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_8, primals_19);  mul_8 = primals_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_11: "f32[1584, 768]" = torch.ops.aten.view.default(add_9, [1584, 768]);  add_9 = None
    permute_7: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
    addmm_4: "f32[1584, 2304]" = torch.ops.aten.addmm.default(primals_21, view_11, permute_7);  primals_21 = None
    view_12: "f32[8, 198, 2304]" = torch.ops.aten.view.default(addmm_4, [8, 198, 2304]);  addmm_4 = None
    view_13: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.view.default(view_12, [8, 198, 3, 12, 64]);  view_12 = None
    permute_8: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_13, [2, 0, 3, 1, 4]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_8);  permute_8 = None
    getitem_13: "f32[8, 12, 198, 64]" = unbind_1[0]
    getitem_14: "f32[8, 12, 198, 64]" = unbind_1[1]
    getitem_15: "f32[8, 12, 198, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_13, getitem_14, getitem_15, None, True)
    getitem_16: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_1[0]
    getitem_17: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_1[1]
    getitem_18: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
    getitem_19: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
    alias_1: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(getitem_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_9: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
    view_14: "f32[8, 198, 768]" = torch.ops.aten.view.default(permute_9, [8, 198, 768]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_15: "f32[1584, 768]" = torch.ops.aten.view.default(view_14, [1584, 768]);  view_14 = None
    permute_10: "f32[768, 768]" = torch.ops.aten.permute.default(primals_22, [1, 0]);  primals_22 = None
    addmm_5: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_23, view_15, permute_10);  primals_23 = None
    view_16: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_5, [8, 198, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_4: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_16);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_10: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_7, clone_4);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 198, 1]" = var_mean_3[0]
    getitem_21: "f32[8, 198, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_3: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_3: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_21)
    mul_9: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_10: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_24);  mul_9 = None
    add_12: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_25);  mul_10 = primals_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_17: "f32[1584, 768]" = torch.ops.aten.view.default(add_12, [1584, 768]);  add_12 = None
    permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
    addmm_6: "f32[1584, 3072]" = torch.ops.aten.addmm.default(primals_27, view_17, permute_11);  primals_27 = None
    view_18: "f32[8, 198, 3072]" = torch.ops.aten.view.default(addmm_6, [8, 198, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_11: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_12: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf_1: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_13: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_13: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_5: "f32[8, 198, 3072]" = torch.ops.aten.clone.default(mul_13);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_19: "f32[1584, 3072]" = torch.ops.aten.view.default(clone_5, [1584, 3072]);  clone_5 = None
    permute_12: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
    addmm_7: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_29, view_19, permute_12);  primals_29 = None
    view_20: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_7, [8, 198, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_6: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_14: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_10, clone_6);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 198, 1]" = var_mean_4[0]
    getitem_23: "f32[8, 198, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_4: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_4: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_23)
    mul_14: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_15: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_30);  mul_14 = None
    add_16: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_15, primals_31);  mul_15 = primals_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_21: "f32[1584, 768]" = torch.ops.aten.view.default(add_16, [1584, 768]);  add_16 = None
    permute_13: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
    addmm_8: "f32[1584, 2304]" = torch.ops.aten.addmm.default(primals_33, view_21, permute_13);  primals_33 = None
    view_22: "f32[8, 198, 2304]" = torch.ops.aten.view.default(addmm_8, [8, 198, 2304]);  addmm_8 = None
    view_23: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.view.default(view_22, [8, 198, 3, 12, 64]);  view_22 = None
    permute_14: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_23, [2, 0, 3, 1, 4]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_14);  permute_14 = None
    getitem_24: "f32[8, 12, 198, 64]" = unbind_2[0]
    getitem_25: "f32[8, 12, 198, 64]" = unbind_2[1]
    getitem_26: "f32[8, 12, 198, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_24, getitem_25, getitem_26, None, True)
    getitem_27: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_2[0]
    getitem_28: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_2[1]
    getitem_29: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
    getitem_30: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
    alias_2: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(getitem_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_15: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
    view_24: "f32[8, 198, 768]" = torch.ops.aten.view.default(permute_15, [8, 198, 768]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_25: "f32[1584, 768]" = torch.ops.aten.view.default(view_24, [1584, 768]);  view_24 = None
    permute_16: "f32[768, 768]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
    addmm_9: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_35, view_25, permute_16);  primals_35 = None
    view_26: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_9, [8, 198, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_7: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_26);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_17: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_14, clone_7);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_31: "f32[8, 198, 1]" = var_mean_5[0]
    getitem_32: "f32[8, 198, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-06);  getitem_31 = None
    rsqrt_5: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_5: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_32)
    mul_16: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_17: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_36);  mul_16 = None
    add_19: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_37);  mul_17 = primals_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_27: "f32[1584, 768]" = torch.ops.aten.view.default(add_19, [1584, 768]);  add_19 = None
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
    addmm_10: "f32[1584, 3072]" = torch.ops.aten.addmm.default(primals_39, view_27, permute_17);  primals_39 = None
    view_28: "f32[8, 198, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 198, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.5)
    mul_19: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476)
    erf_2: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_20: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_20: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_8: "f32[8, 198, 3072]" = torch.ops.aten.clone.default(mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_29: "f32[1584, 3072]" = torch.ops.aten.view.default(clone_8, [1584, 3072]);  clone_8 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_40, [1, 0]);  primals_40 = None
    addmm_11: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_41, view_29, permute_18);  primals_41 = None
    view_30: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_11, [8, 198, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_9: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_30);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_21: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_17, clone_9);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_33: "f32[8, 198, 1]" = var_mean_6[0]
    getitem_34: "f32[8, 198, 1]" = var_mean_6[1];  var_mean_6 = None
    add_22: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
    rsqrt_6: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_6: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_34)
    mul_21: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_22: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_42);  mul_21 = None
    add_23: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_22, primals_43);  mul_22 = primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_31: "f32[1584, 768]" = torch.ops.aten.view.default(add_23, [1584, 768]);  add_23 = None
    permute_19: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    addmm_12: "f32[1584, 2304]" = torch.ops.aten.addmm.default(primals_45, view_31, permute_19);  primals_45 = None
    view_32: "f32[8, 198, 2304]" = torch.ops.aten.view.default(addmm_12, [8, 198, 2304]);  addmm_12 = None
    view_33: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.view.default(view_32, [8, 198, 3, 12, 64]);  view_32 = None
    permute_20: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_33, [2, 0, 3, 1, 4]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_20);  permute_20 = None
    getitem_35: "f32[8, 12, 198, 64]" = unbind_3[0]
    getitem_36: "f32[8, 12, 198, 64]" = unbind_3[1]
    getitem_37: "f32[8, 12, 198, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_35, getitem_36, getitem_37, None, True)
    getitem_38: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_3[0]
    getitem_39: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_3[1]
    getitem_40: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
    getitem_41: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
    alias_3: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(getitem_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_21: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
    view_34: "f32[8, 198, 768]" = torch.ops.aten.view.default(permute_21, [8, 198, 768]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_35: "f32[1584, 768]" = torch.ops.aten.view.default(view_34, [1584, 768]);  view_34 = None
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    addmm_13: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_47, view_35, permute_22);  primals_47 = None
    view_36: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_13, [8, 198, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_10: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_24: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_21, clone_10);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 198, 1]" = var_mean_7[0]
    getitem_43: "f32[8, 198, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_7: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_7: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_43)
    mul_23: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_24: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_23, primals_48);  mul_23 = None
    add_26: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_24, primals_49);  mul_24 = primals_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_37: "f32[1584, 768]" = torch.ops.aten.view.default(add_26, [1584, 768]);  add_26 = None
    permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    addmm_14: "f32[1584, 3072]" = torch.ops.aten.addmm.default(primals_51, view_37, permute_23);  primals_51 = None
    view_38: "f32[8, 198, 3072]" = torch.ops.aten.view.default(addmm_14, [8, 198, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_25: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_26: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_3: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
    add_27: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_27: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_11: "f32[8, 198, 3072]" = torch.ops.aten.clone.default(mul_27);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_39: "f32[1584, 3072]" = torch.ops.aten.view.default(clone_11, [1584, 3072]);  clone_11 = None
    permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    addmm_15: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_53, view_39, permute_24);  primals_53 = None
    view_40: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_15, [8, 198, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_12: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_28: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_24, clone_12);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 198, 1]" = var_mean_8[0]
    getitem_45: "f32[8, 198, 1]" = var_mean_8[1];  var_mean_8 = None
    add_29: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_8: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_8: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_45)
    mul_28: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_29: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_54);  mul_28 = None
    add_30: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_29, primals_55);  mul_29 = primals_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_41: "f32[1584, 768]" = torch.ops.aten.view.default(add_30, [1584, 768]);  add_30 = None
    permute_25: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    addmm_16: "f32[1584, 2304]" = torch.ops.aten.addmm.default(primals_57, view_41, permute_25);  primals_57 = None
    view_42: "f32[8, 198, 2304]" = torch.ops.aten.view.default(addmm_16, [8, 198, 2304]);  addmm_16 = None
    view_43: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.view.default(view_42, [8, 198, 3, 12, 64]);  view_42 = None
    permute_26: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_43, [2, 0, 3, 1, 4]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_26);  permute_26 = None
    getitem_46: "f32[8, 12, 198, 64]" = unbind_4[0]
    getitem_47: "f32[8, 12, 198, 64]" = unbind_4[1]
    getitem_48: "f32[8, 12, 198, 64]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_46, getitem_47, getitem_48, None, True)
    getitem_49: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_4[0]
    getitem_50: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_4[1]
    getitem_51: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
    getitem_52: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
    alias_4: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(getitem_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_27: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
    view_44: "f32[8, 198, 768]" = torch.ops.aten.view.default(permute_27, [8, 198, 768]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_45: "f32[1584, 768]" = torch.ops.aten.view.default(view_44, [1584, 768]);  view_44 = None
    permute_28: "f32[768, 768]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    addmm_17: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_59, view_45, permute_28);  primals_59 = None
    view_46: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_17, [8, 198, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_13: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_46);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_31: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_28, clone_13);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_53: "f32[8, 198, 1]" = var_mean_9[0]
    getitem_54: "f32[8, 198, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-06);  getitem_53 = None
    rsqrt_9: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_9: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_54)
    mul_30: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_31: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_60);  mul_30 = None
    add_33: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_61);  mul_31 = primals_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_47: "f32[1584, 768]" = torch.ops.aten.view.default(add_33, [1584, 768]);  add_33 = None
    permute_29: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    addmm_18: "f32[1584, 3072]" = torch.ops.aten.addmm.default(primals_63, view_47, permute_29);  primals_63 = None
    view_48: "f32[8, 198, 3072]" = torch.ops.aten.view.default(addmm_18, [8, 198, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_48, 0.5)
    mul_33: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476)
    erf_4: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_34: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_34: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_14: "f32[8, 198, 3072]" = torch.ops.aten.clone.default(mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_49: "f32[1584, 3072]" = torch.ops.aten.view.default(clone_14, [1584, 3072]);  clone_14 = None
    permute_30: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    addmm_19: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_65, view_49, permute_30);  primals_65 = None
    view_50: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_19, [8, 198, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_15: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_50);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_35: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_31, clone_15);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_55: "f32[8, 198, 1]" = var_mean_10[0]
    getitem_56: "f32[8, 198, 1]" = var_mean_10[1];  var_mean_10 = None
    add_36: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-06);  getitem_55 = None
    rsqrt_10: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_10: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_56)
    mul_35: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_36: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_66);  mul_35 = None
    add_37: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_36, primals_67);  mul_36 = primals_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_51: "f32[1584, 768]" = torch.ops.aten.view.default(add_37, [1584, 768]);  add_37 = None
    permute_31: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    addmm_20: "f32[1584, 2304]" = torch.ops.aten.addmm.default(primals_69, view_51, permute_31);  primals_69 = None
    view_52: "f32[8, 198, 2304]" = torch.ops.aten.view.default(addmm_20, [8, 198, 2304]);  addmm_20 = None
    view_53: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.view.default(view_52, [8, 198, 3, 12, 64]);  view_52 = None
    permute_32: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_53, [2, 0, 3, 1, 4]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_32);  permute_32 = None
    getitem_57: "f32[8, 12, 198, 64]" = unbind_5[0]
    getitem_58: "f32[8, 12, 198, 64]" = unbind_5[1]
    getitem_59: "f32[8, 12, 198, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_57, getitem_58, getitem_59, None, True)
    getitem_60: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_5[0]
    getitem_61: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_5[1]
    getitem_62: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
    getitem_63: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
    alias_5: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(getitem_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_33: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
    view_54: "f32[8, 198, 768]" = torch.ops.aten.view.default(permute_33, [8, 198, 768]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_55: "f32[1584, 768]" = torch.ops.aten.view.default(view_54, [1584, 768]);  view_54 = None
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    addmm_21: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_71, view_55, permute_34);  primals_71 = None
    view_56: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_21, [8, 198, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_16: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_56);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_38: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_35, clone_16);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 198, 1]" = var_mean_11[0]
    getitem_65: "f32[8, 198, 1]" = var_mean_11[1];  var_mean_11 = None
    add_39: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_11: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_11: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_65)
    mul_37: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_38: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_72);  mul_37 = None
    add_40: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_73);  mul_38 = primals_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_57: "f32[1584, 768]" = torch.ops.aten.view.default(add_40, [1584, 768]);  add_40 = None
    permute_35: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    addmm_22: "f32[1584, 3072]" = torch.ops.aten.addmm.default(primals_75, view_57, permute_35);  primals_75 = None
    view_58: "f32[8, 198, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 198, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_39: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_40: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
    erf_5: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_41: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_41: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_17: "f32[8, 198, 3072]" = torch.ops.aten.clone.default(mul_41);  mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_59: "f32[1584, 3072]" = torch.ops.aten.view.default(clone_17, [1584, 3072]);  clone_17 = None
    permute_36: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    addmm_23: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_77, view_59, permute_36);  primals_77 = None
    view_60: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_23, [8, 198, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_18: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_42: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_38, clone_18);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 198, 1]" = var_mean_12[0]
    getitem_67: "f32[8, 198, 1]" = var_mean_12[1];  var_mean_12 = None
    add_43: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_12: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_12: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_67)
    mul_42: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_43: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_78);  mul_42 = None
    add_44: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_79);  mul_43 = primals_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_61: "f32[1584, 768]" = torch.ops.aten.view.default(add_44, [1584, 768]);  add_44 = None
    permute_37: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    addmm_24: "f32[1584, 2304]" = torch.ops.aten.addmm.default(primals_81, view_61, permute_37);  primals_81 = None
    view_62: "f32[8, 198, 2304]" = torch.ops.aten.view.default(addmm_24, [8, 198, 2304]);  addmm_24 = None
    view_63: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.view.default(view_62, [8, 198, 3, 12, 64]);  view_62 = None
    permute_38: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_38);  permute_38 = None
    getitem_68: "f32[8, 12, 198, 64]" = unbind_6[0]
    getitem_69: "f32[8, 12, 198, 64]" = unbind_6[1]
    getitem_70: "f32[8, 12, 198, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_68, getitem_69, getitem_70, None, True)
    getitem_71: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_6[0]
    getitem_72: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_6[1]
    getitem_73: "i64[]" = _scaled_dot_product_efficient_attention_6[2]
    getitem_74: "i64[]" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None
    alias_6: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(getitem_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_39: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
    view_64: "f32[8, 198, 768]" = torch.ops.aten.view.default(permute_39, [8, 198, 768]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_65: "f32[1584, 768]" = torch.ops.aten.view.default(view_64, [1584, 768]);  view_64 = None
    permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    addmm_25: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_83, view_65, permute_40);  primals_83 = None
    view_66: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_25, [8, 198, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_19: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_66);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_45: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_42, clone_19);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_75: "f32[8, 198, 1]" = var_mean_13[0]
    getitem_76: "f32[8, 198, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-06);  getitem_75 = None
    rsqrt_13: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_13: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_76)
    mul_44: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_45: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_84);  mul_44 = None
    add_47: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_85);  mul_45 = primals_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_67: "f32[1584, 768]" = torch.ops.aten.view.default(add_47, [1584, 768]);  add_47 = None
    permute_41: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    addmm_26: "f32[1584, 3072]" = torch.ops.aten.addmm.default(primals_87, view_67, permute_41);  primals_87 = None
    view_68: "f32[8, 198, 3072]" = torch.ops.aten.view.default(addmm_26, [8, 198, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_68, 0.5)
    mul_47: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476)
    erf_6: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_48: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_48: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_48);  mul_46 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_20: "f32[8, 198, 3072]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_69: "f32[1584, 3072]" = torch.ops.aten.view.default(clone_20, [1584, 3072]);  clone_20 = None
    permute_42: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    addmm_27: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_89, view_69, permute_42);  primals_89 = None
    view_70: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_27, [8, 198, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_21: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_70);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_49: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_45, clone_21);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_77: "f32[8, 198, 1]" = var_mean_14[0]
    getitem_78: "f32[8, 198, 1]" = var_mean_14[1];  var_mean_14 = None
    add_50: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
    rsqrt_14: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_14: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_78)
    mul_49: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_50: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_90);  mul_49 = None
    add_51: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_91);  mul_50 = primals_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_71: "f32[1584, 768]" = torch.ops.aten.view.default(add_51, [1584, 768]);  add_51 = None
    permute_43: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    addmm_28: "f32[1584, 2304]" = torch.ops.aten.addmm.default(primals_93, view_71, permute_43);  primals_93 = None
    view_72: "f32[8, 198, 2304]" = torch.ops.aten.view.default(addmm_28, [8, 198, 2304]);  addmm_28 = None
    view_73: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.view.default(view_72, [8, 198, 3, 12, 64]);  view_72 = None
    permute_44: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_73, [2, 0, 3, 1, 4]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_44);  permute_44 = None
    getitem_79: "f32[8, 12, 198, 64]" = unbind_7[0]
    getitem_80: "f32[8, 12, 198, 64]" = unbind_7[1]
    getitem_81: "f32[8, 12, 198, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_79, getitem_80, getitem_81, None, True)
    getitem_82: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_7[0]
    getitem_83: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_7[1]
    getitem_84: "i64[]" = _scaled_dot_product_efficient_attention_7[2]
    getitem_85: "i64[]" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None
    alias_7: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(getitem_82)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_45: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
    view_74: "f32[8, 198, 768]" = torch.ops.aten.view.default(permute_45, [8, 198, 768]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_75: "f32[1584, 768]" = torch.ops.aten.view.default(view_74, [1584, 768]);  view_74 = None
    permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    addmm_29: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_95, view_75, permute_46);  primals_95 = None
    view_76: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_29, [8, 198, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_22: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_76);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_52: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_49, clone_22);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 198, 1]" = var_mean_15[0]
    getitem_87: "f32[8, 198, 1]" = var_mean_15[1];  var_mean_15 = None
    add_53: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
    rsqrt_15: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_15: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_87)
    mul_51: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_52: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_51, primals_96);  mul_51 = None
    add_54: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_52, primals_97);  mul_52 = primals_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_77: "f32[1584, 768]" = torch.ops.aten.view.default(add_54, [1584, 768]);  add_54 = None
    permute_47: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    addmm_30: "f32[1584, 3072]" = torch.ops.aten.addmm.default(primals_99, view_77, permute_47);  primals_99 = None
    view_78: "f32[8, 198, 3072]" = torch.ops.aten.view.default(addmm_30, [8, 198, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_53: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_54: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
    erf_7: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_55: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_55: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_55);  mul_53 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 198, 3072]" = torch.ops.aten.clone.default(mul_55);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_79: "f32[1584, 3072]" = torch.ops.aten.view.default(clone_23, [1584, 3072]);  clone_23 = None
    permute_48: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    addmm_31: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_101, view_79, permute_48);  primals_101 = None
    view_80: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_31, [8, 198, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_56: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_52, clone_24);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 198, 1]" = var_mean_16[0]
    getitem_89: "f32[8, 198, 1]" = var_mean_16[1];  var_mean_16 = None
    add_57: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_16: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_16: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_89)
    mul_56: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_57: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_56, primals_102);  mul_56 = None
    add_58: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_57, primals_103);  mul_57 = primals_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_81: "f32[1584, 768]" = torch.ops.aten.view.default(add_58, [1584, 768]);  add_58 = None
    permute_49: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
    addmm_32: "f32[1584, 2304]" = torch.ops.aten.addmm.default(primals_105, view_81, permute_49);  primals_105 = None
    view_82: "f32[8, 198, 2304]" = torch.ops.aten.view.default(addmm_32, [8, 198, 2304]);  addmm_32 = None
    view_83: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.view.default(view_82, [8, 198, 3, 12, 64]);  view_82 = None
    permute_50: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_83, [2, 0, 3, 1, 4]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
    getitem_90: "f32[8, 12, 198, 64]" = unbind_8[0]
    getitem_91: "f32[8, 12, 198, 64]" = unbind_8[1]
    getitem_92: "f32[8, 12, 198, 64]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_90, getitem_91, getitem_92, None, True)
    getitem_93: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_8[0]
    getitem_94: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_8[1]
    getitem_95: "i64[]" = _scaled_dot_product_efficient_attention_8[2]
    getitem_96: "i64[]" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None
    alias_8: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(getitem_93)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_51: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
    view_84: "f32[8, 198, 768]" = torch.ops.aten.view.default(permute_51, [8, 198, 768]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_85: "f32[1584, 768]" = torch.ops.aten.view.default(view_84, [1584, 768]);  view_84 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    addmm_33: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_107, view_85, permute_52);  primals_107 = None
    view_86: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_33, [8, 198, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_25: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_86);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_59: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_56, clone_25);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_97: "f32[8, 198, 1]" = var_mean_17[0]
    getitem_98: "f32[8, 198, 1]" = var_mean_17[1];  var_mean_17 = None
    add_60: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-06);  getitem_97 = None
    rsqrt_17: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_17: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_98)
    mul_58: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_59: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_108);  mul_58 = None
    add_61: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_109);  mul_59 = primals_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_87: "f32[1584, 768]" = torch.ops.aten.view.default(add_61, [1584, 768]);  add_61 = None
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    addmm_34: "f32[1584, 3072]" = torch.ops.aten.addmm.default(primals_111, view_87, permute_53);  primals_111 = None
    view_88: "f32[8, 198, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 198, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_60: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.5)
    mul_61: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476)
    erf_8: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_62: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_62: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_62);  mul_60 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_26: "f32[8, 198, 3072]" = torch.ops.aten.clone.default(mul_62);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_89: "f32[1584, 3072]" = torch.ops.aten.view.default(clone_26, [1584, 3072]);  clone_26 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    addmm_35: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_113, view_89, permute_54);  primals_113 = None
    view_90: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_35, [8, 198, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_27: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_90);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_63: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_59, clone_27);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_99: "f32[8, 198, 1]" = var_mean_18[0]
    getitem_100: "f32[8, 198, 1]" = var_mean_18[1];  var_mean_18 = None
    add_64: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-06);  getitem_99 = None
    rsqrt_18: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_18: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_100)
    mul_63: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_64: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_63, primals_114);  mul_63 = None
    add_65: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_64, primals_115);  mul_64 = primals_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_91: "f32[1584, 768]" = torch.ops.aten.view.default(add_65, [1584, 768]);  add_65 = None
    permute_55: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    addmm_36: "f32[1584, 2304]" = torch.ops.aten.addmm.default(primals_117, view_91, permute_55);  primals_117 = None
    view_92: "f32[8, 198, 2304]" = torch.ops.aten.view.default(addmm_36, [8, 198, 2304]);  addmm_36 = None
    view_93: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.view.default(view_92, [8, 198, 3, 12, 64]);  view_92 = None
    permute_56: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_93, [2, 0, 3, 1, 4]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_56);  permute_56 = None
    getitem_101: "f32[8, 12, 198, 64]" = unbind_9[0]
    getitem_102: "f32[8, 12, 198, 64]" = unbind_9[1]
    getitem_103: "f32[8, 12, 198, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_101, getitem_102, getitem_103, None, True)
    getitem_104: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_9[0]
    getitem_105: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_9[1]
    getitem_106: "i64[]" = _scaled_dot_product_efficient_attention_9[2]
    getitem_107: "i64[]" = _scaled_dot_product_efficient_attention_9[3];  _scaled_dot_product_efficient_attention_9 = None
    alias_9: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(getitem_104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_57: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
    view_94: "f32[8, 198, 768]" = torch.ops.aten.view.default(permute_57, [8, 198, 768]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_95: "f32[1584, 768]" = torch.ops.aten.view.default(view_94, [1584, 768]);  view_94 = None
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    addmm_37: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_119, view_95, permute_58);  primals_119 = None
    view_96: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_37, [8, 198, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_28: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_66: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_63, clone_28);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 198, 1]" = var_mean_19[0]
    getitem_109: "f32[8, 198, 1]" = var_mean_19[1];  var_mean_19 = None
    add_67: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_19: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_19: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_109)
    mul_65: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_66: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_120);  mul_65 = None
    add_68: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_121);  mul_66 = primals_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_97: "f32[1584, 768]" = torch.ops.aten.view.default(add_68, [1584, 768]);  add_68 = None
    permute_59: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    addmm_38: "f32[1584, 3072]" = torch.ops.aten.addmm.default(primals_123, view_97, permute_59);  primals_123 = None
    view_98: "f32[8, 198, 3072]" = torch.ops.aten.view.default(addmm_38, [8, 198, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_68: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
    erf_9: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_69: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_69: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_69);  mul_67 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_29: "f32[8, 198, 3072]" = torch.ops.aten.clone.default(mul_69);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_99: "f32[1584, 3072]" = torch.ops.aten.view.default(clone_29, [1584, 3072]);  clone_29 = None
    permute_60: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    addmm_39: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_125, view_99, permute_60);  primals_125 = None
    view_100: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_39, [8, 198, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_30: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_70: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_66, clone_30);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 198, 1]" = var_mean_20[0]
    getitem_111: "f32[8, 198, 1]" = var_mean_20[1];  var_mean_20 = None
    add_71: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_20: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_20: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_111)
    mul_70: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_71: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_70, primals_126);  mul_70 = None
    add_72: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_71, primals_127);  mul_71 = primals_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_101: "f32[1584, 768]" = torch.ops.aten.view.default(add_72, [1584, 768]);  add_72 = None
    permute_61: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    addmm_40: "f32[1584, 2304]" = torch.ops.aten.addmm.default(primals_129, view_101, permute_61);  primals_129 = None
    view_102: "f32[8, 198, 2304]" = torch.ops.aten.view.default(addmm_40, [8, 198, 2304]);  addmm_40 = None
    view_103: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.view.default(view_102, [8, 198, 3, 12, 64]);  view_102 = None
    permute_62: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_103, [2, 0, 3, 1, 4]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_62);  permute_62 = None
    getitem_112: "f32[8, 12, 198, 64]" = unbind_10[0]
    getitem_113: "f32[8, 12, 198, 64]" = unbind_10[1]
    getitem_114: "f32[8, 12, 198, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_112, getitem_113, getitem_114, None, True)
    getitem_115: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_10[0]
    getitem_116: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_10[1]
    getitem_117: "i64[]" = _scaled_dot_product_efficient_attention_10[2]
    getitem_118: "i64[]" = _scaled_dot_product_efficient_attention_10[3];  _scaled_dot_product_efficient_attention_10 = None
    alias_10: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(getitem_115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_63: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
    view_104: "f32[8, 198, 768]" = torch.ops.aten.view.default(permute_63, [8, 198, 768]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_105: "f32[1584, 768]" = torch.ops.aten.view.default(view_104, [1584, 768]);  view_104 = None
    permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    addmm_41: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_131, view_105, permute_64);  primals_131 = None
    view_106: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_41, [8, 198, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_31: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_106);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_73: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_70, clone_31);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_119: "f32[8, 198, 1]" = var_mean_21[0]
    getitem_120: "f32[8, 198, 1]" = var_mean_21[1];  var_mean_21 = None
    add_74: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-06);  getitem_119 = None
    rsqrt_21: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_21: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_120)
    mul_72: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_73: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_132);  mul_72 = None
    add_75: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_133);  mul_73 = primals_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_107: "f32[1584, 768]" = torch.ops.aten.view.default(add_75, [1584, 768]);  add_75 = None
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    addmm_42: "f32[1584, 3072]" = torch.ops.aten.addmm.default(primals_135, view_107, permute_65);  primals_135 = None
    view_108: "f32[8, 198, 3072]" = torch.ops.aten.view.default(addmm_42, [8, 198, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_74: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_108, 0.5)
    mul_75: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476)
    erf_10: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_76: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_76: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_74, add_76);  mul_74 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_32: "f32[8, 198, 3072]" = torch.ops.aten.clone.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_109: "f32[1584, 3072]" = torch.ops.aten.view.default(clone_32, [1584, 3072]);  clone_32 = None
    permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    addmm_43: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_137, view_109, permute_66);  primals_137 = None
    view_110: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_43, [8, 198, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_33: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_110);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_77: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_73, clone_33);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_121: "f32[8, 198, 1]" = var_mean_22[0]
    getitem_122: "f32[8, 198, 1]" = var_mean_22[1];  var_mean_22 = None
    add_78: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-06);  getitem_121 = None
    rsqrt_22: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_22: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_122)
    mul_77: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_78: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_138);  mul_77 = None
    add_79: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_78, primals_139);  mul_78 = primals_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_111: "f32[1584, 768]" = torch.ops.aten.view.default(add_79, [1584, 768]);  add_79 = None
    permute_67: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    addmm_44: "f32[1584, 2304]" = torch.ops.aten.addmm.default(primals_141, view_111, permute_67);  primals_141 = None
    view_112: "f32[8, 198, 2304]" = torch.ops.aten.view.default(addmm_44, [8, 198, 2304]);  addmm_44 = None
    view_113: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.view.default(view_112, [8, 198, 3, 12, 64]);  view_112 = None
    permute_68: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.permute.default(view_113, [2, 0, 3, 1, 4]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_68);  permute_68 = None
    getitem_123: "f32[8, 12, 198, 64]" = unbind_11[0]
    getitem_124: "f32[8, 12, 198, 64]" = unbind_11[1]
    getitem_125: "f32[8, 12, 198, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_123, getitem_124, getitem_125, None, True)
    getitem_126: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_11[0]
    getitem_127: "f32[8, 12, 224]" = _scaled_dot_product_efficient_attention_11[1]
    getitem_128: "i64[]" = _scaled_dot_product_efficient_attention_11[2]
    getitem_129: "i64[]" = _scaled_dot_product_efficient_attention_11[3];  _scaled_dot_product_efficient_attention_11 = None
    alias_11: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(getitem_126)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_69: "f32[8, 198, 12, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
    view_114: "f32[8, 198, 768]" = torch.ops.aten.view.default(permute_69, [8, 198, 768]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_115: "f32[1584, 768]" = torch.ops.aten.view.default(view_114, [1584, 768]);  view_114 = None
    permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    addmm_45: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_143, view_115, permute_70);  primals_143 = None
    view_116: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_45, [8, 198, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_34: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_116);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_80: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_77, clone_34);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_130: "f32[8, 198, 1]" = var_mean_23[0]
    getitem_131: "f32[8, 198, 1]" = var_mean_23[1];  var_mean_23 = None
    add_81: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-06);  getitem_130 = None
    rsqrt_23: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_23: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_131)
    mul_79: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_80: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_79, primals_144);  mul_79 = None
    add_82: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_80, primals_145);  mul_80 = primals_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[1584, 768]" = torch.ops.aten.view.default(add_82, [1584, 768]);  add_82 = None
    permute_71: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    addmm_46: "f32[1584, 3072]" = torch.ops.aten.addmm.default(primals_147, view_117, permute_71);  primals_147 = None
    view_118: "f32[8, 198, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 198, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_81: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_82: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
    erf_11: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
    add_83: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_83: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_81, add_83);  mul_81 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_35: "f32[8, 198, 3072]" = torch.ops.aten.clone.default(mul_83);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_119: "f32[1584, 3072]" = torch.ops.aten.view.default(clone_35, [1584, 3072]);  clone_35 = None
    permute_72: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    addmm_47: "f32[1584, 768]" = torch.ops.aten.addmm.default(primals_149, view_119, permute_72);  primals_149 = None
    view_120: "f32[8, 198, 768]" = torch.ops.aten.view.default(addmm_47, [8, 198, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_36: "f32[8, 198, 768]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_84: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_80, clone_36);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:641, code: x = self.norm(x)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 198, 1]" = var_mean_24[0]
    getitem_133: "f32[8, 198, 1]" = var_mean_24[1];  var_mean_24 = None
    add_85: "f32[8, 198, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_24: "f32[8, 198, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_24: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_133)
    mul_84: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_85: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_84, primals_150);  mul_84 = None
    add_86: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_85, primals_151);  mul_85 = primals_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:108, code: x, x_dist = x[:, 0], x[:, 1]
    slice_1: "f32[8, 198, 768]" = torch.ops.aten.slice.Tensor(add_86, 0, 0, 9223372036854775807)
    select: "f32[8, 768]" = torch.ops.aten.select.int(slice_1, 1, 0);  slice_1 = None
    slice_2: "f32[8, 198, 768]" = torch.ops.aten.slice.Tensor(add_86, 0, 0, 9223372036854775807);  add_86 = None
    select_1: "f32[8, 768]" = torch.ops.aten.select.int(slice_2, 1, 1);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:111, code: x = self.head(x)
    permute_73: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    addmm_48: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_153, select, permute_73);  primals_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:112, code: x_dist = self.head_dist(x_dist)
    permute_74: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    addmm_49: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_155, select_1, permute_74);  primals_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:118, code: return (x + x_dist) / 2
    add_87: "f32[8, 1000]" = torch.ops.aten.add.Tensor(addmm_48, addmm_49);  addmm_48 = addmm_49 = None
    div: "f32[8, 1000]" = torch.ops.aten.div.Tensor(add_87, 2);  add_87 = None
    div_1: "f32[8, 1000]" = torch.ops.aten.div.Tensor(tangents_1, 2);  tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:112, code: x_dist = self.head_dist(x_dist)
    permute_75: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    mm: "f32[8, 768]" = torch.ops.aten.mm.default(div_1, permute_75);  permute_75 = None
    permute_76: "f32[1000, 8]" = torch.ops.aten.permute.default(div_1, [1, 0])
    mm_1: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_76, select_1);  permute_76 = select_1 = None
    permute_77: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(div_1, [0], True)
    view_121: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_78: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:111, code: x = self.head(x)
    permute_79: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    mm_2: "f32[8, 768]" = torch.ops.aten.mm.default(div_1, permute_79);  permute_79 = None
    permute_80: "f32[1000, 8]" = torch.ops.aten.permute.default(div_1, [1, 0])
    mm_3: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_80, select);  permute_80 = select = None
    permute_81: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_2: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(div_1, [0], True);  div_1 = None
    view_122: "f32[1000]" = torch.ops.aten.view.default(sum_2, [1000]);  sum_2 = None
    permute_82: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:108, code: x, x_dist = x[:, 0], x[:, 1]
    full: "f32[8, 198, 768]" = torch.ops.aten.full.default([8, 198, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter: "f32[8, 198, 768]" = torch.ops.aten.select_scatter.default(full, mm, 1, 1);  full = mm = None
    full_1: "f32[8, 198, 768]" = torch.ops.aten.full.default([8, 198, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter: "f32[8, 198, 768]" = torch.ops.aten.slice_scatter.default(full_1, select_scatter, 0, 0, 9223372036854775807);  full_1 = select_scatter = None
    full_2: "f32[8, 198, 768]" = torch.ops.aten.full.default([8, 198, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_1: "f32[8, 198, 768]" = torch.ops.aten.select_scatter.default(full_2, mm_2, 1, 0);  full_2 = mm_2 = None
    full_3: "f32[8, 198, 768]" = torch.ops.aten.full.default([8, 198, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_1: "f32[8, 198, 768]" = torch.ops.aten.slice_scatter.default(full_3, select_scatter_1, 0, 0, 9223372036854775807);  full_3 = select_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:108, code: x, x_dist = x[:, 0], x[:, 1]
    add_88: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(slice_scatter, slice_scatter_1);  slice_scatter = slice_scatter_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:641, code: x = self.norm(x)
    sub_25: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_133);  add_84 = getitem_133 = None
    mul_86: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = None
    mul_87: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(add_88, primals_150);  primals_150 = None
    mul_88: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_87, 768)
    sum_3: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_87, [2], True)
    mul_89: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_87, mul_86);  mul_87 = None
    sum_4: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [2], True);  mul_89 = None
    mul_90: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_86, sum_4);  sum_4 = None
    sub_26: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_88, sum_3);  mul_88 = sum_3 = None
    sub_27: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_26, mul_90);  sub_26 = mul_90 = None
    div_2: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    mul_91: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_27);  div_2 = sub_27 = None
    mul_92: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(add_88, mul_86);  mul_86 = None
    sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_92, [0, 1]);  mul_92 = None
    sum_6: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_88, [0, 1]);  add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_123: "f32[1584, 768]" = torch.ops.aten.view.default(mul_91, [1584, 768])
    permute_83: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    mm_4: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_123, permute_83);  permute_83 = None
    permute_84: "f32[768, 1584]" = torch.ops.aten.permute.default(view_123, [1, 0])
    mm_5: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_84, view_119);  permute_84 = view_119 = None
    permute_85: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_7: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_123, [0], True);  view_123 = None
    view_124: "f32[768]" = torch.ops.aten.view.default(sum_7, [768]);  sum_7 = None
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    view_125: "f32[8, 198, 3072]" = torch.ops.aten.view.default(mm_4, [8, 198, 3072]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_93: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
    erf_12: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_93);  mul_93 = None
    add_89: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_94: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_89, 0.5);  add_89 = None
    mul_95: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_118, view_118)
    mul_96: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_95, -0.5);  mul_95 = None
    exp: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_96);  mul_96 = None
    mul_97: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_98: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_118, mul_97);  view_118 = mul_97 = None
    add_90: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_94, mul_98);  mul_94 = mul_98 = None
    mul_99: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_125, add_90);  view_125 = add_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_126: "f32[1584, 3072]" = torch.ops.aten.view.default(mul_99, [1584, 3072]);  mul_99 = None
    permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_6: "f32[1584, 768]" = torch.ops.aten.mm.default(view_126, permute_87);  permute_87 = None
    permute_88: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_126, [1, 0])
    mm_7: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_88, view_117);  permute_88 = view_117 = None
    permute_89: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_8: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_126, [0], True);  view_126 = None
    view_127: "f32[3072]" = torch.ops.aten.view.default(sum_8, [3072]);  sum_8 = None
    permute_90: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    view_128: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_6, [8, 198, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_28: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_131);  add_80 = getitem_131 = None
    mul_100: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_23);  sub_28 = None
    mul_101: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_128, primals_144);  primals_144 = None
    mul_102: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_101, 768)
    sum_9: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [2], True)
    mul_103: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_101, mul_100);  mul_101 = None
    sum_10: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_103, [2], True);  mul_103 = None
    mul_104: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_100, sum_10);  sum_10 = None
    sub_29: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_102, sum_9);  mul_102 = sum_9 = None
    sub_30: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_29, mul_104);  sub_29 = mul_104 = None
    div_3: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    mul_105: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_30);  div_3 = sub_30 = None
    mul_106: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_128, mul_100);  mul_100 = None
    sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_106, [0, 1]);  mul_106 = None
    sum_12: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_128, [0, 1]);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_91: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(mul_91, mul_105);  mul_91 = mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_129: "f32[1584, 768]" = torch.ops.aten.view.default(add_91, [1584, 768])
    permute_91: "f32[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_8: "f32[1584, 768]" = torch.ops.aten.mm.default(view_129, permute_91);  permute_91 = None
    permute_92: "f32[768, 1584]" = torch.ops.aten.permute.default(view_129, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_92, view_115);  permute_92 = view_115 = None
    permute_93: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_13: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_129, [0], True);  view_129 = None
    view_130: "f32[768]" = torch.ops.aten.view.default(sum_13, [768]);  sum_13 = None
    permute_94: "f32[768, 768]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    view_131: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_8, [8, 198, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_132: "f32[8, 198, 12, 64]" = torch.ops.aten.view.default(view_131, [8, 198, 12, 64]);  view_131 = None
    permute_95: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_12: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_95, getitem_123, getitem_124, getitem_125, None, alias_12, getitem_127, getitem_128, getitem_129, 0.0, [True, True, True, False]);  permute_95 = getitem_123 = getitem_124 = getitem_125 = alias_12 = getitem_127 = getitem_128 = getitem_129 = None
    getitem_134: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward[0]
    getitem_135: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward[1]
    getitem_136: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_1: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_134, getitem_135, getitem_136]);  getitem_134 = getitem_135 = getitem_136 = None
    view_133: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.view.default(cat_1, [3, 8, 12, 198, 64]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_96: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_133, [1, 3, 0, 2, 4]);  view_133 = None
    clone_37: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_96, memory_format = torch.contiguous_format);  permute_96 = None
    view_134: "f32[8, 198, 2304]" = torch.ops.aten.view.default(clone_37, [8, 198, 2304]);  clone_37 = None
    view_135: "f32[1584, 2304]" = torch.ops.aten.view.default(view_134, [1584, 2304]);  view_134 = None
    permute_97: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_10: "f32[1584, 768]" = torch.ops.aten.mm.default(view_135, permute_97);  permute_97 = None
    permute_98: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_135, [1, 0])
    mm_11: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_98, view_111);  permute_98 = view_111 = None
    permute_99: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_14: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_135, [0], True);  view_135 = None
    view_136: "f32[2304]" = torch.ops.aten.view.default(sum_14, [2304]);  sum_14 = None
    permute_100: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    view_137: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_10, [8, 198, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_31: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_122);  add_77 = getitem_122 = None
    mul_107: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_22);  sub_31 = None
    mul_108: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_137, primals_138);  primals_138 = None
    mul_109: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_108, 768)
    sum_15: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True)
    mul_110: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_108, mul_107);  mul_108 = None
    sum_16: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True);  mul_110 = None
    mul_111: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_107, sum_16);  sum_16 = None
    sub_32: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_109, sum_15);  mul_109 = sum_15 = None
    sub_33: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_111);  sub_32 = mul_111 = None
    div_4: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    mul_112: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_33);  div_4 = sub_33 = None
    mul_113: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_137, mul_107);  mul_107 = None
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1]);  mul_113 = None
    sum_18: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_137, [0, 1]);  view_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_92: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_91, mul_112);  add_91 = mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_138: "f32[1584, 768]" = torch.ops.aten.view.default(add_92, [1584, 768])
    permute_101: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_12: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_138, permute_101);  permute_101 = None
    permute_102: "f32[768, 1584]" = torch.ops.aten.permute.default(view_138, [1, 0])
    mm_13: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_102, view_109);  permute_102 = view_109 = None
    permute_103: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_19: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_138, [0], True);  view_138 = None
    view_139: "f32[768]" = torch.ops.aten.view.default(sum_19, [768]);  sum_19 = None
    permute_104: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    view_140: "f32[8, 198, 3072]" = torch.ops.aten.view.default(mm_12, [8, 198, 3072]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_114: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476)
    erf_13: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
    add_93: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_115: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_93, 0.5);  add_93 = None
    mul_116: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_108, view_108)
    mul_117: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_116, -0.5);  mul_116 = None
    exp_1: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_117);  mul_117 = None
    mul_118: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_119: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_108, mul_118);  view_108 = mul_118 = None
    add_94: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_115, mul_119);  mul_115 = mul_119 = None
    mul_120: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_140, add_94);  view_140 = add_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_141: "f32[1584, 3072]" = torch.ops.aten.view.default(mul_120, [1584, 3072]);  mul_120 = None
    permute_105: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_14: "f32[1584, 768]" = torch.ops.aten.mm.default(view_141, permute_105);  permute_105 = None
    permute_106: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_141, [1, 0])
    mm_15: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_106, view_107);  permute_106 = view_107 = None
    permute_107: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_20: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_141, [0], True);  view_141 = None
    view_142: "f32[3072]" = torch.ops.aten.view.default(sum_20, [3072]);  sum_20 = None
    permute_108: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    view_143: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_14, [8, 198, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_34: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_120);  add_73 = getitem_120 = None
    mul_121: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_21);  sub_34 = None
    mul_122: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_143, primals_132);  primals_132 = None
    mul_123: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_122, 768)
    sum_21: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
    mul_124: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_122, mul_121);  mul_122 = None
    sum_22: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True);  mul_124 = None
    mul_125: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_121, sum_22);  sum_22 = None
    sub_35: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_123, sum_21);  mul_123 = sum_21 = None
    sub_36: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_35, mul_125);  sub_35 = mul_125 = None
    div_5: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    mul_126: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_36);  div_5 = sub_36 = None
    mul_127: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_143, mul_121);  mul_121 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1]);  mul_127 = None
    sum_24: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_143, [0, 1]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_95: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_92, mul_126);  add_92 = mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_144: "f32[1584, 768]" = torch.ops.aten.view.default(add_95, [1584, 768])
    permute_109: "f32[768, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_16: "f32[1584, 768]" = torch.ops.aten.mm.default(view_144, permute_109);  permute_109 = None
    permute_110: "f32[768, 1584]" = torch.ops.aten.permute.default(view_144, [1, 0])
    mm_17: "f32[768, 768]" = torch.ops.aten.mm.default(permute_110, view_105);  permute_110 = view_105 = None
    permute_111: "f32[768, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_25: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_144, [0], True);  view_144 = None
    view_145: "f32[768]" = torch.ops.aten.view.default(sum_25, [768]);  sum_25 = None
    permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    view_146: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_16, [8, 198, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_147: "f32[8, 198, 12, 64]" = torch.ops.aten.view.default(view_146, [8, 198, 12, 64]);  view_146 = None
    permute_113: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_13: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_113, getitem_112, getitem_113, getitem_114, None, alias_13, getitem_116, getitem_117, getitem_118, 0.0, [True, True, True, False]);  permute_113 = getitem_112 = getitem_113 = getitem_114 = alias_13 = getitem_116 = getitem_117 = getitem_118 = None
    getitem_138: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_1[0]
    getitem_139: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_1[1]
    getitem_140: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_2: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_138, getitem_139, getitem_140]);  getitem_138 = getitem_139 = getitem_140 = None
    view_148: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.view.default(cat_2, [3, 8, 12, 198, 64]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_114: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_148, [1, 3, 0, 2, 4]);  view_148 = None
    clone_38: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_149: "f32[8, 198, 2304]" = torch.ops.aten.view.default(clone_38, [8, 198, 2304]);  clone_38 = None
    view_150: "f32[1584, 2304]" = torch.ops.aten.view.default(view_149, [1584, 2304]);  view_149 = None
    permute_115: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    mm_18: "f32[1584, 768]" = torch.ops.aten.mm.default(view_150, permute_115);  permute_115 = None
    permute_116: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_150, [1, 0])
    mm_19: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_116, view_101);  permute_116 = view_101 = None
    permute_117: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_26: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_150, [0], True);  view_150 = None
    view_151: "f32[2304]" = torch.ops.aten.view.default(sum_26, [2304]);  sum_26 = None
    permute_118: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    view_152: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_18, [8, 198, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_37: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_111);  add_70 = getitem_111 = None
    mul_128: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_20);  sub_37 = None
    mul_129: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_152, primals_126);  primals_126 = None
    mul_130: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_129, 768)
    sum_27: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True)
    mul_131: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_129, mul_128);  mul_129 = None
    sum_28: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True);  mul_131 = None
    mul_132: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_128, sum_28);  sum_28 = None
    sub_38: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_130, sum_27);  mul_130 = sum_27 = None
    sub_39: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_132);  sub_38 = mul_132 = None
    div_6: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    mul_133: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_39);  div_6 = sub_39 = None
    mul_134: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_152, mul_128);  mul_128 = None
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1]);  mul_134 = None
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_152, [0, 1]);  view_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_96: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_95, mul_133);  add_95 = mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_153: "f32[1584, 768]" = torch.ops.aten.view.default(add_96, [1584, 768])
    permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    mm_20: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_153, permute_119);  permute_119 = None
    permute_120: "f32[768, 1584]" = torch.ops.aten.permute.default(view_153, [1, 0])
    mm_21: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_120, view_99);  permute_120 = view_99 = None
    permute_121: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_31: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_153, [0], True);  view_153 = None
    view_154: "f32[768]" = torch.ops.aten.view.default(sum_31, [768]);  sum_31 = None
    permute_122: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    view_155: "f32[8, 198, 3072]" = torch.ops.aten.view.default(mm_20, [8, 198, 3072]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_135: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
    erf_14: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_135);  mul_135 = None
    add_97: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_136: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_97, 0.5);  add_97 = None
    mul_137: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_98, view_98)
    mul_138: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_137, -0.5);  mul_137 = None
    exp_2: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_138);  mul_138 = None
    mul_139: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_140: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_98, mul_139);  view_98 = mul_139 = None
    add_98: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_136, mul_140);  mul_136 = mul_140 = None
    mul_141: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_155, add_98);  view_155 = add_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_156: "f32[1584, 3072]" = torch.ops.aten.view.default(mul_141, [1584, 3072]);  mul_141 = None
    permute_123: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_22: "f32[1584, 768]" = torch.ops.aten.mm.default(view_156, permute_123);  permute_123 = None
    permute_124: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_156, [1, 0])
    mm_23: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_124, view_97);  permute_124 = view_97 = None
    permute_125: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_32: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_156, [0], True);  view_156 = None
    view_157: "f32[3072]" = torch.ops.aten.view.default(sum_32, [3072]);  sum_32 = None
    permute_126: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    view_158: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_22, [8, 198, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_40: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_109);  add_66 = getitem_109 = None
    mul_142: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_19);  sub_40 = None
    mul_143: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_158, primals_120);  primals_120 = None
    mul_144: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_143, 768)
    sum_33: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_143, [2], True)
    mul_145: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_143, mul_142);  mul_143 = None
    sum_34: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True);  mul_145 = None
    mul_146: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_142, sum_34);  sum_34 = None
    sub_41: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_144, sum_33);  mul_144 = sum_33 = None
    sub_42: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_146);  sub_41 = mul_146 = None
    div_7: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    mul_147: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_42);  div_7 = sub_42 = None
    mul_148: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_158, mul_142);  mul_142 = None
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1]);  mul_148 = None
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_158, [0, 1]);  view_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_99: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_96, mul_147);  add_96 = mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_159: "f32[1584, 768]" = torch.ops.aten.view.default(add_99, [1584, 768])
    permute_127: "f32[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_24: "f32[1584, 768]" = torch.ops.aten.mm.default(view_159, permute_127);  permute_127 = None
    permute_128: "f32[768, 1584]" = torch.ops.aten.permute.default(view_159, [1, 0])
    mm_25: "f32[768, 768]" = torch.ops.aten.mm.default(permute_128, view_95);  permute_128 = view_95 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_159, [0], True);  view_159 = None
    view_160: "f32[768]" = torch.ops.aten.view.default(sum_37, [768]);  sum_37 = None
    permute_130: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    view_161: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_24, [8, 198, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_162: "f32[8, 198, 12, 64]" = torch.ops.aten.view.default(view_161, [8, 198, 12, 64]);  view_161 = None
    permute_131: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_14: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_131, getitem_101, getitem_102, getitem_103, None, alias_14, getitem_105, getitem_106, getitem_107, 0.0, [True, True, True, False]);  permute_131 = getitem_101 = getitem_102 = getitem_103 = alias_14 = getitem_105 = getitem_106 = getitem_107 = None
    getitem_142: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_2[0]
    getitem_143: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_2[1]
    getitem_144: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_3: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_142, getitem_143, getitem_144]);  getitem_142 = getitem_143 = getitem_144 = None
    view_163: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.view.default(cat_3, [3, 8, 12, 198, 64]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_132: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_163, [1, 3, 0, 2, 4]);  view_163 = None
    clone_39: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None
    view_164: "f32[8, 198, 2304]" = torch.ops.aten.view.default(clone_39, [8, 198, 2304]);  clone_39 = None
    view_165: "f32[1584, 2304]" = torch.ops.aten.view.default(view_164, [1584, 2304]);  view_164 = None
    permute_133: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_26: "f32[1584, 768]" = torch.ops.aten.mm.default(view_165, permute_133);  permute_133 = None
    permute_134: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_165, [1, 0])
    mm_27: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_134, view_91);  permute_134 = view_91 = None
    permute_135: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_38: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_165, [0], True);  view_165 = None
    view_166: "f32[2304]" = torch.ops.aten.view.default(sum_38, [2304]);  sum_38 = None
    permute_136: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
    view_167: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_26, [8, 198, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_43: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_100);  add_63 = getitem_100 = None
    mul_149: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_18);  sub_43 = None
    mul_150: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_167, primals_114);  primals_114 = None
    mul_151: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_150, 768)
    sum_39: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True)
    mul_152: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_150, mul_149);  mul_150 = None
    sum_40: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
    mul_153: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_149, sum_40);  sum_40 = None
    sub_44: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_151, sum_39);  mul_151 = sum_39 = None
    sub_45: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_153);  sub_44 = mul_153 = None
    div_8: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    mul_154: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_45);  div_8 = sub_45 = None
    mul_155: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_167, mul_149);  mul_149 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
    sum_42: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_167, [0, 1]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_100: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_99, mul_154);  add_99 = mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_168: "f32[1584, 768]" = torch.ops.aten.view.default(add_100, [1584, 768])
    permute_137: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_28: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_168, permute_137);  permute_137 = None
    permute_138: "f32[768, 1584]" = torch.ops.aten.permute.default(view_168, [1, 0])
    mm_29: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_138, view_89);  permute_138 = view_89 = None
    permute_139: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_168, [0], True);  view_168 = None
    view_169: "f32[768]" = torch.ops.aten.view.default(sum_43, [768]);  sum_43 = None
    permute_140: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    view_170: "f32[8, 198, 3072]" = torch.ops.aten.view.default(mm_28, [8, 198, 3072]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_156: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476)
    erf_15: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_156);  mul_156 = None
    add_101: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_157: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_101, 0.5);  add_101 = None
    mul_158: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_88, view_88)
    mul_159: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_158, -0.5);  mul_158 = None
    exp_3: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_159);  mul_159 = None
    mul_160: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_161: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_88, mul_160);  view_88 = mul_160 = None
    add_102: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_157, mul_161);  mul_157 = mul_161 = None
    mul_162: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_170, add_102);  view_170 = add_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_171: "f32[1584, 3072]" = torch.ops.aten.view.default(mul_162, [1584, 3072]);  mul_162 = None
    permute_141: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_30: "f32[1584, 768]" = torch.ops.aten.mm.default(view_171, permute_141);  permute_141 = None
    permute_142: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_171, [1, 0])
    mm_31: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_142, view_87);  permute_142 = view_87 = None
    permute_143: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_44: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_171, [0], True);  view_171 = None
    view_172: "f32[3072]" = torch.ops.aten.view.default(sum_44, [3072]);  sum_44 = None
    permute_144: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
    view_173: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_30, [8, 198, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_46: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_98);  add_59 = getitem_98 = None
    mul_163: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_17);  sub_46 = None
    mul_164: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_173, primals_108);  primals_108 = None
    mul_165: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_164, 768)
    sum_45: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
    mul_166: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_164, mul_163);  mul_164 = None
    sum_46: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
    mul_167: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_163, sum_46);  sum_46 = None
    sub_47: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_165, sum_45);  mul_165 = sum_45 = None
    sub_48: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_167);  sub_47 = mul_167 = None
    div_9: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    mul_168: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_48);  div_9 = sub_48 = None
    mul_169: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_173, mul_163);  mul_163 = None
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
    sum_48: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_173, [0, 1]);  view_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_103: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_100, mul_168);  add_100 = mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_174: "f32[1584, 768]" = torch.ops.aten.view.default(add_103, [1584, 768])
    permute_145: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_32: "f32[1584, 768]" = torch.ops.aten.mm.default(view_174, permute_145);  permute_145 = None
    permute_146: "f32[768, 1584]" = torch.ops.aten.permute.default(view_174, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_146, view_85);  permute_146 = view_85 = None
    permute_147: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_49: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_174, [0], True);  view_174 = None
    view_175: "f32[768]" = torch.ops.aten.view.default(sum_49, [768]);  sum_49 = None
    permute_148: "f32[768, 768]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
    view_176: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_32, [8, 198, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_177: "f32[8, 198, 12, 64]" = torch.ops.aten.view.default(view_176, [8, 198, 12, 64]);  view_176 = None
    permute_149: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_177, [0, 2, 1, 3]);  view_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_15: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_149, getitem_90, getitem_91, getitem_92, None, alias_15, getitem_94, getitem_95, getitem_96, 0.0, [True, True, True, False]);  permute_149 = getitem_90 = getitem_91 = getitem_92 = alias_15 = getitem_94 = getitem_95 = getitem_96 = None
    getitem_146: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_3[0]
    getitem_147: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_3[1]
    getitem_148: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_4: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_146, getitem_147, getitem_148]);  getitem_146 = getitem_147 = getitem_148 = None
    view_178: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.view.default(cat_4, [3, 8, 12, 198, 64]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_150: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_178, [1, 3, 0, 2, 4]);  view_178 = None
    clone_40: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    view_179: "f32[8, 198, 2304]" = torch.ops.aten.view.default(clone_40, [8, 198, 2304]);  clone_40 = None
    view_180: "f32[1584, 2304]" = torch.ops.aten.view.default(view_179, [1584, 2304]);  view_179 = None
    permute_151: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    mm_34: "f32[1584, 768]" = torch.ops.aten.mm.default(view_180, permute_151);  permute_151 = None
    permute_152: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_180, [1, 0])
    mm_35: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_152, view_81);  permute_152 = view_81 = None
    permute_153: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_50: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_180, [0], True);  view_180 = None
    view_181: "f32[2304]" = torch.ops.aten.view.default(sum_50, [2304]);  sum_50 = None
    permute_154: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_182: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_34, [8, 198, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_49: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_89);  add_56 = getitem_89 = None
    mul_170: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_16);  sub_49 = None
    mul_171: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_182, primals_102);  primals_102 = None
    mul_172: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_171, 768)
    sum_51: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True)
    mul_173: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_171, mul_170);  mul_171 = None
    sum_52: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
    mul_174: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_170, sum_52);  sum_52 = None
    sub_50: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_172, sum_51);  mul_172 = sum_51 = None
    sub_51: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_174);  sub_50 = mul_174 = None
    div_10: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    mul_175: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_51);  div_10 = sub_51 = None
    mul_176: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_182, mul_170);  mul_170 = None
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 1]);  mul_176 = None
    sum_54: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_182, [0, 1]);  view_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_104: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_103, mul_175);  add_103 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_183: "f32[1584, 768]" = torch.ops.aten.view.default(add_104, [1584, 768])
    permute_155: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_36: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_183, permute_155);  permute_155 = None
    permute_156: "f32[768, 1584]" = torch.ops.aten.permute.default(view_183, [1, 0])
    mm_37: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_156, view_79);  permute_156 = view_79 = None
    permute_157: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_55: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_183, [0], True);  view_183 = None
    view_184: "f32[768]" = torch.ops.aten.view.default(sum_55, [768]);  sum_55 = None
    permute_158: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    view_185: "f32[8, 198, 3072]" = torch.ops.aten.view.default(mm_36, [8, 198, 3072]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_177: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
    erf_16: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_177);  mul_177 = None
    add_105: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_178: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_105, 0.5);  add_105 = None
    mul_179: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_78, view_78)
    mul_180: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_179, -0.5);  mul_179 = None
    exp_4: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_180);  mul_180 = None
    mul_181: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_182: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_78, mul_181);  view_78 = mul_181 = None
    add_106: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_178, mul_182);  mul_178 = mul_182 = None
    mul_183: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_185, add_106);  view_185 = add_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_186: "f32[1584, 3072]" = torch.ops.aten.view.default(mul_183, [1584, 3072]);  mul_183 = None
    permute_159: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_38: "f32[1584, 768]" = torch.ops.aten.mm.default(view_186, permute_159);  permute_159 = None
    permute_160: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_186, [1, 0])
    mm_39: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_160, view_77);  permute_160 = view_77 = None
    permute_161: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_56: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_186, [0], True);  view_186 = None
    view_187: "f32[3072]" = torch.ops.aten.view.default(sum_56, [3072]);  sum_56 = None
    permute_162: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    view_188: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_38, [8, 198, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_52: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_87);  add_52 = getitem_87 = None
    mul_184: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_15);  sub_52 = None
    mul_185: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_188, primals_96);  primals_96 = None
    mul_186: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_185, 768)
    sum_57: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True)
    mul_187: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_185, mul_184);  mul_185 = None
    sum_58: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True);  mul_187 = None
    mul_188: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_184, sum_58);  sum_58 = None
    sub_53: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_186, sum_57);  mul_186 = sum_57 = None
    sub_54: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_53, mul_188);  sub_53 = mul_188 = None
    div_11: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    mul_189: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_54);  div_11 = sub_54 = None
    mul_190: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_188, mul_184);  mul_184 = None
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_190, [0, 1]);  mul_190 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_188, [0, 1]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_107: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_104, mul_189);  add_104 = mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_189: "f32[1584, 768]" = torch.ops.aten.view.default(add_107, [1584, 768])
    permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_40: "f32[1584, 768]" = torch.ops.aten.mm.default(view_189, permute_163);  permute_163 = None
    permute_164: "f32[768, 1584]" = torch.ops.aten.permute.default(view_189, [1, 0])
    mm_41: "f32[768, 768]" = torch.ops.aten.mm.default(permute_164, view_75);  permute_164 = view_75 = None
    permute_165: "f32[768, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_61: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True);  view_189 = None
    view_190: "f32[768]" = torch.ops.aten.view.default(sum_61, [768]);  sum_61 = None
    permute_166: "f32[768, 768]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    view_191: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_40, [8, 198, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_192: "f32[8, 198, 12, 64]" = torch.ops.aten.view.default(view_191, [8, 198, 12, 64]);  view_191 = None
    permute_167: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_16: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_167, getitem_79, getitem_80, getitem_81, None, alias_16, getitem_83, getitem_84, getitem_85, 0.0, [True, True, True, False]);  permute_167 = getitem_79 = getitem_80 = getitem_81 = alias_16 = getitem_83 = getitem_84 = getitem_85 = None
    getitem_150: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_4[0]
    getitem_151: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_4[1]
    getitem_152: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_5: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_150, getitem_151, getitem_152]);  getitem_150 = getitem_151 = getitem_152 = None
    view_193: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.view.default(cat_5, [3, 8, 12, 198, 64]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_168: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_193, [1, 3, 0, 2, 4]);  view_193 = None
    clone_41: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    view_194: "f32[8, 198, 2304]" = torch.ops.aten.view.default(clone_41, [8, 198, 2304]);  clone_41 = None
    view_195: "f32[1584, 2304]" = torch.ops.aten.view.default(view_194, [1584, 2304]);  view_194 = None
    permute_169: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_42: "f32[1584, 768]" = torch.ops.aten.mm.default(view_195, permute_169);  permute_169 = None
    permute_170: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_195, [1, 0])
    mm_43: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_170, view_71);  permute_170 = view_71 = None
    permute_171: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_62: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_195, [0], True);  view_195 = None
    view_196: "f32[2304]" = torch.ops.aten.view.default(sum_62, [2304]);  sum_62 = None
    permute_172: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
    view_197: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_42, [8, 198, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_55: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_78);  add_49 = getitem_78 = None
    mul_191: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_14);  sub_55 = None
    mul_192: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_197, primals_90);  primals_90 = None
    mul_193: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_192, 768)
    sum_63: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
    mul_194: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_192, mul_191);  mul_192 = None
    sum_64: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
    mul_195: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_191, sum_64);  sum_64 = None
    sub_56: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_193, sum_63);  mul_193 = sum_63 = None
    sub_57: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_195);  sub_56 = mul_195 = None
    div_12: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    mul_196: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_57);  div_12 = sub_57 = None
    mul_197: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_197, mul_191);  mul_191 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_197, [0, 1]);  view_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_108: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_107, mul_196);  add_107 = mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_198: "f32[1584, 768]" = torch.ops.aten.view.default(add_108, [1584, 768])
    permute_173: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_44: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_198, permute_173);  permute_173 = None
    permute_174: "f32[768, 1584]" = torch.ops.aten.permute.default(view_198, [1, 0])
    mm_45: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_174, view_69);  permute_174 = view_69 = None
    permute_175: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_198, [0], True);  view_198 = None
    view_199: "f32[768]" = torch.ops.aten.view.default(sum_67, [768]);  sum_67 = None
    permute_176: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    view_200: "f32[8, 198, 3072]" = torch.ops.aten.view.default(mm_44, [8, 198, 3072]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_198: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476)
    erf_17: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_198);  mul_198 = None
    add_109: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_199: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_109, 0.5);  add_109 = None
    mul_200: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_68, view_68)
    mul_201: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_200, -0.5);  mul_200 = None
    exp_5: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_201);  mul_201 = None
    mul_202: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_203: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_68, mul_202);  view_68 = mul_202 = None
    add_110: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_199, mul_203);  mul_199 = mul_203 = None
    mul_204: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_200, add_110);  view_200 = add_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_201: "f32[1584, 3072]" = torch.ops.aten.view.default(mul_204, [1584, 3072]);  mul_204 = None
    permute_177: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_46: "f32[1584, 768]" = torch.ops.aten.mm.default(view_201, permute_177);  permute_177 = None
    permute_178: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_201, [1, 0])
    mm_47: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_178, view_67);  permute_178 = view_67 = None
    permute_179: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_68: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_201, [0], True);  view_201 = None
    view_202: "f32[3072]" = torch.ops.aten.view.default(sum_68, [3072]);  sum_68 = None
    permute_180: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    view_203: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_46, [8, 198, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_58: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_76);  add_45 = getitem_76 = None
    mul_205: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_13);  sub_58 = None
    mul_206: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_203, primals_84);  primals_84 = None
    mul_207: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_206, 768)
    sum_69: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
    mul_208: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_206, mul_205);  mul_206 = None
    sum_70: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
    mul_209: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_205, sum_70);  sum_70 = None
    sub_59: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_207, sum_69);  mul_207 = sum_69 = None
    sub_60: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_209);  sub_59 = mul_209 = None
    div_13: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_210: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_60);  div_13 = sub_60 = None
    mul_211: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_203, mul_205);  mul_205 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
    sum_72: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_203, [0, 1]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_111: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_108, mul_210);  add_108 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_204: "f32[1584, 768]" = torch.ops.aten.view.default(add_111, [1584, 768])
    permute_181: "f32[768, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    mm_48: "f32[1584, 768]" = torch.ops.aten.mm.default(view_204, permute_181);  permute_181 = None
    permute_182: "f32[768, 1584]" = torch.ops.aten.permute.default(view_204, [1, 0])
    mm_49: "f32[768, 768]" = torch.ops.aten.mm.default(permute_182, view_65);  permute_182 = view_65 = None
    permute_183: "f32[768, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_204, [0], True);  view_204 = None
    view_205: "f32[768]" = torch.ops.aten.view.default(sum_73, [768]);  sum_73 = None
    permute_184: "f32[768, 768]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
    view_206: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_48, [8, 198, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_207: "f32[8, 198, 12, 64]" = torch.ops.aten.view.default(view_206, [8, 198, 12, 64]);  view_206 = None
    permute_185: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_207, [0, 2, 1, 3]);  view_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_17: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_185, getitem_68, getitem_69, getitem_70, None, alias_17, getitem_72, getitem_73, getitem_74, 0.0, [True, True, True, False]);  permute_185 = getitem_68 = getitem_69 = getitem_70 = alias_17 = getitem_72 = getitem_73 = getitem_74 = None
    getitem_154: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_5[0]
    getitem_155: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_5[1]
    getitem_156: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_6: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_154, getitem_155, getitem_156]);  getitem_154 = getitem_155 = getitem_156 = None
    view_208: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.view.default(cat_6, [3, 8, 12, 198, 64]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_186: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_208, [1, 3, 0, 2, 4]);  view_208 = None
    clone_42: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_186, memory_format = torch.contiguous_format);  permute_186 = None
    view_209: "f32[8, 198, 2304]" = torch.ops.aten.view.default(clone_42, [8, 198, 2304]);  clone_42 = None
    view_210: "f32[1584, 2304]" = torch.ops.aten.view.default(view_209, [1584, 2304]);  view_209 = None
    permute_187: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    mm_50: "f32[1584, 768]" = torch.ops.aten.mm.default(view_210, permute_187);  permute_187 = None
    permute_188: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_210, [1, 0])
    mm_51: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_188, view_61);  permute_188 = view_61 = None
    permute_189: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_74: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_210, [0], True);  view_210 = None
    view_211: "f32[2304]" = torch.ops.aten.view.default(sum_74, [2304]);  sum_74 = None
    permute_190: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
    view_212: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_50, [8, 198, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_61: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_67);  add_42 = getitem_67 = None
    mul_212: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_12);  sub_61 = None
    mul_213: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_212, primals_78);  primals_78 = None
    mul_214: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
    sum_75: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_212);  mul_213 = None
    sum_76: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_212, sum_76);  sum_76 = None
    sub_62: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_75);  mul_214 = sum_75 = None
    sub_63: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_216);  sub_62 = mul_216 = None
    div_14: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_217: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_63);  div_14 = sub_63 = None
    mul_218: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_212, mul_212);  mul_212 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_78: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_212, [0, 1]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_112: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_111, mul_217);  add_111 = mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_213: "f32[1584, 768]" = torch.ops.aten.view.default(add_112, [1584, 768])
    permute_191: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_52: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_213, permute_191);  permute_191 = None
    permute_192: "f32[768, 1584]" = torch.ops.aten.permute.default(view_213, [1, 0])
    mm_53: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_192, view_59);  permute_192 = view_59 = None
    permute_193: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_213, [0], True);  view_213 = None
    view_214: "f32[768]" = torch.ops.aten.view.default(sum_79, [768]);  sum_79 = None
    permute_194: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    view_215: "f32[8, 198, 3072]" = torch.ops.aten.view.default(mm_52, [8, 198, 3072]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_219: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
    erf_18: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_219);  mul_219 = None
    add_113: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_220: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_113, 0.5);  add_113 = None
    mul_221: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_58, view_58)
    mul_222: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_221, -0.5);  mul_221 = None
    exp_6: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_222);  mul_222 = None
    mul_223: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_224: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_58, mul_223);  view_58 = mul_223 = None
    add_114: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_220, mul_224);  mul_220 = mul_224 = None
    mul_225: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_215, add_114);  view_215 = add_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_216: "f32[1584, 3072]" = torch.ops.aten.view.default(mul_225, [1584, 3072]);  mul_225 = None
    permute_195: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_54: "f32[1584, 768]" = torch.ops.aten.mm.default(view_216, permute_195);  permute_195 = None
    permute_196: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_216, [1, 0])
    mm_55: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_196, view_57);  permute_196 = view_57 = None
    permute_197: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_80: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_216, [0], True);  view_216 = None
    view_217: "f32[3072]" = torch.ops.aten.view.default(sum_80, [3072]);  sum_80 = None
    permute_198: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
    view_218: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_54, [8, 198, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_64: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_65);  add_38 = getitem_65 = None
    mul_226: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_11);  sub_64 = None
    mul_227: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_218, primals_72);  primals_72 = None
    mul_228: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_227, 768)
    sum_81: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
    mul_229: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_227, mul_226);  mul_227 = None
    sum_82: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
    mul_230: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_226, sum_82);  sum_82 = None
    sub_65: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_228, sum_81);  mul_228 = sum_81 = None
    sub_66: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_230);  sub_65 = mul_230 = None
    div_15: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_231: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_66);  div_15 = sub_66 = None
    mul_232: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_218, mul_226);  mul_226 = None
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
    sum_84: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_218, [0, 1]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_115: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_112, mul_231);  add_112 = mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_219: "f32[1584, 768]" = torch.ops.aten.view.default(add_115, [1584, 768])
    permute_199: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_56: "f32[1584, 768]" = torch.ops.aten.mm.default(view_219, permute_199);  permute_199 = None
    permute_200: "f32[768, 1584]" = torch.ops.aten.permute.default(view_219, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_200, view_55);  permute_200 = view_55 = None
    permute_201: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_85: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_219, [0], True);  view_219 = None
    view_220: "f32[768]" = torch.ops.aten.view.default(sum_85, [768]);  sum_85 = None
    permute_202: "f32[768, 768]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    view_221: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_56, [8, 198, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_222: "f32[8, 198, 12, 64]" = torch.ops.aten.view.default(view_221, [8, 198, 12, 64]);  view_221 = None
    permute_203: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_18: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_203, getitem_57, getitem_58, getitem_59, None, alias_18, getitem_61, getitem_62, getitem_63, 0.0, [True, True, True, False]);  permute_203 = getitem_57 = getitem_58 = getitem_59 = alias_18 = getitem_61 = getitem_62 = getitem_63 = None
    getitem_158: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_6[0]
    getitem_159: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_6[1]
    getitem_160: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_7: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_158, getitem_159, getitem_160]);  getitem_158 = getitem_159 = getitem_160 = None
    view_223: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.view.default(cat_7, [3, 8, 12, 198, 64]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_204: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_223, [1, 3, 0, 2, 4]);  view_223 = None
    clone_43: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
    view_224: "f32[8, 198, 2304]" = torch.ops.aten.view.default(clone_43, [8, 198, 2304]);  clone_43 = None
    view_225: "f32[1584, 2304]" = torch.ops.aten.view.default(view_224, [1584, 2304]);  view_224 = None
    permute_205: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_58: "f32[1584, 768]" = torch.ops.aten.mm.default(view_225, permute_205);  permute_205 = None
    permute_206: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_225, [1, 0])
    mm_59: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_206, view_51);  permute_206 = view_51 = None
    permute_207: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_86: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_225, [0], True);  view_225 = None
    view_226: "f32[2304]" = torch.ops.aten.view.default(sum_86, [2304]);  sum_86 = None
    permute_208: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    view_227: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_58, [8, 198, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_67: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_56);  add_35 = getitem_56 = None
    mul_233: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_10);  sub_67 = None
    mul_234: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_227, primals_66);  primals_66 = None
    mul_235: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_234, 768)
    sum_87: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True)
    mul_236: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_234, mul_233);  mul_234 = None
    sum_88: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True);  mul_236 = None
    mul_237: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_233, sum_88);  sum_88 = None
    sub_68: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_235, sum_87);  mul_235 = sum_87 = None
    sub_69: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_237);  sub_68 = mul_237 = None
    div_16: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_238: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_69);  div_16 = sub_69 = None
    mul_239: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_227, mul_233);  mul_233 = None
    sum_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1]);  mul_239 = None
    sum_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_227, [0, 1]);  view_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_116: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_115, mul_238);  add_115 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_228: "f32[1584, 768]" = torch.ops.aten.view.default(add_116, [1584, 768])
    permute_209: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_60: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_228, permute_209);  permute_209 = None
    permute_210: "f32[768, 1584]" = torch.ops.aten.permute.default(view_228, [1, 0])
    mm_61: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_210, view_49);  permute_210 = view_49 = None
    permute_211: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_91: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_228, [0], True);  view_228 = None
    view_229: "f32[768]" = torch.ops.aten.view.default(sum_91, [768]);  sum_91 = None
    permute_212: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    view_230: "f32[8, 198, 3072]" = torch.ops.aten.view.default(mm_60, [8, 198, 3072]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_240: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476)
    erf_19: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_240);  mul_240 = None
    add_117: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_241: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_117, 0.5);  add_117 = None
    mul_242: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_48, view_48)
    mul_243: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_242, -0.5);  mul_242 = None
    exp_7: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_243);  mul_243 = None
    mul_244: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_245: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_48, mul_244);  view_48 = mul_244 = None
    add_118: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_241, mul_245);  mul_241 = mul_245 = None
    mul_246: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_230, add_118);  view_230 = add_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_231: "f32[1584, 3072]" = torch.ops.aten.view.default(mul_246, [1584, 3072]);  mul_246 = None
    permute_213: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    mm_62: "f32[1584, 768]" = torch.ops.aten.mm.default(view_231, permute_213);  permute_213 = None
    permute_214: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_231, [1, 0])
    mm_63: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_214, view_47);  permute_214 = view_47 = None
    permute_215: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_92: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_231, [0], True);  view_231 = None
    view_232: "f32[3072]" = torch.ops.aten.view.default(sum_92, [3072]);  sum_92 = None
    permute_216: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    view_233: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_62, [8, 198, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_70: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_54);  add_31 = getitem_54 = None
    mul_247: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_9);  sub_70 = None
    mul_248: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_233, primals_60);  primals_60 = None
    mul_249: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_248, 768)
    sum_93: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [2], True)
    mul_250: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_248, mul_247);  mul_248 = None
    sum_94: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True);  mul_250 = None
    mul_251: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_247, sum_94);  sum_94 = None
    sub_71: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_249, sum_93);  mul_249 = sum_93 = None
    sub_72: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_251);  sub_71 = mul_251 = None
    div_17: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_252: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_72);  div_17 = sub_72 = None
    mul_253: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_233, mul_247);  mul_247 = None
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1]);  mul_253 = None
    sum_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_233, [0, 1]);  view_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_119: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_116, mul_252);  add_116 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_234: "f32[1584, 768]" = torch.ops.aten.view.default(add_119, [1584, 768])
    permute_217: "f32[768, 768]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    mm_64: "f32[1584, 768]" = torch.ops.aten.mm.default(view_234, permute_217);  permute_217 = None
    permute_218: "f32[768, 1584]" = torch.ops.aten.permute.default(view_234, [1, 0])
    mm_65: "f32[768, 768]" = torch.ops.aten.mm.default(permute_218, view_45);  permute_218 = view_45 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_234, [0], True);  view_234 = None
    view_235: "f32[768]" = torch.ops.aten.view.default(sum_97, [768]);  sum_97 = None
    permute_220: "f32[768, 768]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    view_236: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_64, [8, 198, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_237: "f32[8, 198, 12, 64]" = torch.ops.aten.view.default(view_236, [8, 198, 12, 64]);  view_236 = None
    permute_221: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_237, [0, 2, 1, 3]);  view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_19: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_221, getitem_46, getitem_47, getitem_48, None, alias_19, getitem_50, getitem_51, getitem_52, 0.0, [True, True, True, False]);  permute_221 = getitem_46 = getitem_47 = getitem_48 = alias_19 = getitem_50 = getitem_51 = getitem_52 = None
    getitem_162: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_7[0]
    getitem_163: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_7[1]
    getitem_164: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_8: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_162, getitem_163, getitem_164]);  getitem_162 = getitem_163 = getitem_164 = None
    view_238: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.view.default(cat_8, [3, 8, 12, 198, 64]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_222: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_238, [1, 3, 0, 2, 4]);  view_238 = None
    clone_44: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_222, memory_format = torch.contiguous_format);  permute_222 = None
    view_239: "f32[8, 198, 2304]" = torch.ops.aten.view.default(clone_44, [8, 198, 2304]);  clone_44 = None
    view_240: "f32[1584, 2304]" = torch.ops.aten.view.default(view_239, [1584, 2304]);  view_239 = None
    permute_223: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_66: "f32[1584, 768]" = torch.ops.aten.mm.default(view_240, permute_223);  permute_223 = None
    permute_224: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_240, [1, 0])
    mm_67: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_224, view_41);  permute_224 = view_41 = None
    permute_225: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_98: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_240, [0], True);  view_240 = None
    view_241: "f32[2304]" = torch.ops.aten.view.default(sum_98, [2304]);  sum_98 = None
    permute_226: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
    view_242: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_66, [8, 198, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_73: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_45);  add_28 = getitem_45 = None
    mul_254: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_8);  sub_73 = None
    mul_255: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_242, primals_54);  primals_54 = None
    mul_256: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_255, 768)
    sum_99: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True)
    mul_257: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_255, mul_254);  mul_255 = None
    sum_100: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
    mul_258: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_254, sum_100);  sum_100 = None
    sub_74: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_256, sum_99);  mul_256 = sum_99 = None
    sub_75: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_258);  sub_74 = mul_258 = None
    div_18: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_259: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_75);  div_18 = sub_75 = None
    mul_260: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_242, mul_254);  mul_254 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1]);  mul_260 = None
    sum_102: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_242, [0, 1]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_120: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_119, mul_259);  add_119 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_243: "f32[1584, 768]" = torch.ops.aten.view.default(add_120, [1584, 768])
    permute_227: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_68: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_243, permute_227);  permute_227 = None
    permute_228: "f32[768, 1584]" = torch.ops.aten.permute.default(view_243, [1, 0])
    mm_69: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_228, view_39);  permute_228 = view_39 = None
    permute_229: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_243, [0], True);  view_243 = None
    view_244: "f32[768]" = torch.ops.aten.view.default(sum_103, [768]);  sum_103 = None
    permute_230: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    view_245: "f32[8, 198, 3072]" = torch.ops.aten.view.default(mm_68, [8, 198, 3072]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_261: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_20: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_261);  mul_261 = None
    add_121: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_262: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_121, 0.5);  add_121 = None
    mul_263: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_38, view_38)
    mul_264: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_263, -0.5);  mul_263 = None
    exp_8: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_264);  mul_264 = None
    mul_265: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_266: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_38, mul_265);  view_38 = mul_265 = None
    add_122: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_262, mul_266);  mul_262 = mul_266 = None
    mul_267: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_245, add_122);  view_245 = add_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_246: "f32[1584, 3072]" = torch.ops.aten.view.default(mul_267, [1584, 3072]);  mul_267 = None
    permute_231: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_70: "f32[1584, 768]" = torch.ops.aten.mm.default(view_246, permute_231);  permute_231 = None
    permute_232: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_246, [1, 0])
    mm_71: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_232, view_37);  permute_232 = view_37 = None
    permute_233: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_104: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
    view_247: "f32[3072]" = torch.ops.aten.view.default(sum_104, [3072]);  sum_104 = None
    permute_234: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    view_248: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_70, [8, 198, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_76: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_43);  add_24 = getitem_43 = None
    mul_268: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_7);  sub_76 = None
    mul_269: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_248, primals_48);  primals_48 = None
    mul_270: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_269, 768)
    sum_105: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
    mul_271: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_269, mul_268);  mul_269 = None
    sum_106: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
    mul_272: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_268, sum_106);  sum_106 = None
    sub_77: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_270, sum_105);  mul_270 = sum_105 = None
    sub_78: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_272);  sub_77 = mul_272 = None
    div_19: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_273: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_78);  div_19 = sub_78 = None
    mul_274: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_248, mul_268);  mul_268 = None
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
    sum_108: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_248, [0, 1]);  view_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_123: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_120, mul_273);  add_120 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_249: "f32[1584, 768]" = torch.ops.aten.view.default(add_123, [1584, 768])
    permute_235: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_72: "f32[1584, 768]" = torch.ops.aten.mm.default(view_249, permute_235);  permute_235 = None
    permute_236: "f32[768, 1584]" = torch.ops.aten.permute.default(view_249, [1, 0])
    mm_73: "f32[768, 768]" = torch.ops.aten.mm.default(permute_236, view_35);  permute_236 = view_35 = None
    permute_237: "f32[768, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_109: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
    view_250: "f32[768]" = torch.ops.aten.view.default(sum_109, [768]);  sum_109 = None
    permute_238: "f32[768, 768]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
    view_251: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_72, [8, 198, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_252: "f32[8, 198, 12, 64]" = torch.ops.aten.view.default(view_251, [8, 198, 12, 64]);  view_251 = None
    permute_239: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1, 3]);  view_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_20: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_239, getitem_35, getitem_36, getitem_37, None, alias_20, getitem_39, getitem_40, getitem_41, 0.0, [True, True, True, False]);  permute_239 = getitem_35 = getitem_36 = getitem_37 = alias_20 = getitem_39 = getitem_40 = getitem_41 = None
    getitem_166: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_8[0]
    getitem_167: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_8[1]
    getitem_168: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_9: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_166, getitem_167, getitem_168]);  getitem_166 = getitem_167 = getitem_168 = None
    view_253: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.view.default(cat_9, [3, 8, 12, 198, 64]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_240: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_253, [1, 3, 0, 2, 4]);  view_253 = None
    clone_45: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
    view_254: "f32[8, 198, 2304]" = torch.ops.aten.view.default(clone_45, [8, 198, 2304]);  clone_45 = None
    view_255: "f32[1584, 2304]" = torch.ops.aten.view.default(view_254, [1584, 2304]);  view_254 = None
    permute_241: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_74: "f32[1584, 768]" = torch.ops.aten.mm.default(view_255, permute_241);  permute_241 = None
    permute_242: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_255, [1, 0])
    mm_75: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_242, view_31);  permute_242 = view_31 = None
    permute_243: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_110: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_255, [0], True);  view_255 = None
    view_256: "f32[2304]" = torch.ops.aten.view.default(sum_110, [2304]);  sum_110 = None
    permute_244: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    view_257: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_74, [8, 198, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_79: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_34);  add_21 = getitem_34 = None
    mul_275: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_6);  sub_79 = None
    mul_276: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_257, primals_42);  primals_42 = None
    mul_277: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_276, 768)
    sum_111: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True)
    mul_278: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_276, mul_275);  mul_276 = None
    sum_112: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True);  mul_278 = None
    mul_279: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_275, sum_112);  sum_112 = None
    sub_80: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_277, sum_111);  mul_277 = sum_111 = None
    sub_81: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_279);  sub_80 = mul_279 = None
    div_20: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_280: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_81);  div_20 = sub_81 = None
    mul_281: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_257, mul_275);  mul_275 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 1]);  mul_281 = None
    sum_114: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_257, [0, 1]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_124: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_123, mul_280);  add_123 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_258: "f32[1584, 768]" = torch.ops.aten.view.default(add_124, [1584, 768])
    permute_245: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_76: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_258, permute_245);  permute_245 = None
    permute_246: "f32[768, 1584]" = torch.ops.aten.permute.default(view_258, [1, 0])
    mm_77: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_246, view_29);  permute_246 = view_29 = None
    permute_247: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_258, [0], True);  view_258 = None
    view_259: "f32[768]" = torch.ops.aten.view.default(sum_115, [768]);  sum_115 = None
    permute_248: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    view_260: "f32[8, 198, 3072]" = torch.ops.aten.view.default(mm_76, [8, 198, 3072]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_282: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476)
    erf_21: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_282);  mul_282 = None
    add_125: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_283: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_125, 0.5);  add_125 = None
    mul_284: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_28, view_28)
    mul_285: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_284, -0.5);  mul_284 = None
    exp_9: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_285);  mul_285 = None
    mul_286: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_287: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_28, mul_286);  view_28 = mul_286 = None
    add_126: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_283, mul_287);  mul_283 = mul_287 = None
    mul_288: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_260, add_126);  view_260 = add_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_261: "f32[1584, 3072]" = torch.ops.aten.view.default(mul_288, [1584, 3072]);  mul_288 = None
    permute_249: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    mm_78: "f32[1584, 768]" = torch.ops.aten.mm.default(view_261, permute_249);  permute_249 = None
    permute_250: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_261, [1, 0])
    mm_79: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_250, view_27);  permute_250 = view_27 = None
    permute_251: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_116: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_261, [0], True);  view_261 = None
    view_262: "f32[3072]" = torch.ops.aten.view.default(sum_116, [3072]);  sum_116 = None
    permute_252: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
    view_263: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_78, [8, 198, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_82: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_32);  add_17 = getitem_32 = None
    mul_289: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_5);  sub_82 = None
    mul_290: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_263, primals_36);  primals_36 = None
    mul_291: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_290, 768)
    sum_117: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True)
    mul_292: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_290, mul_289);  mul_290 = None
    sum_118: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
    mul_293: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_289, sum_118);  sum_118 = None
    sub_83: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_291, sum_117);  mul_291 = sum_117 = None
    sub_84: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_293);  sub_83 = mul_293 = None
    div_21: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_294: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_84);  div_21 = sub_84 = None
    mul_295: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_263, mul_289);  mul_289 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_263, [0, 1]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_127: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_124, mul_294);  add_124 = mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_264: "f32[1584, 768]" = torch.ops.aten.view.default(add_127, [1584, 768])
    permute_253: "f32[768, 768]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    mm_80: "f32[1584, 768]" = torch.ops.aten.mm.default(view_264, permute_253);  permute_253 = None
    permute_254: "f32[768, 1584]" = torch.ops.aten.permute.default(view_264, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_254, view_25);  permute_254 = view_25 = None
    permute_255: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_121: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
    view_265: "f32[768]" = torch.ops.aten.view.default(sum_121, [768]);  sum_121 = None
    permute_256: "f32[768, 768]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
    view_266: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_80, [8, 198, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_267: "f32[8, 198, 12, 64]" = torch.ops.aten.view.default(view_266, [8, 198, 12, 64]);  view_266 = None
    permute_257: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_21: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_257, getitem_24, getitem_25, getitem_26, None, alias_21, getitem_28, getitem_29, getitem_30, 0.0, [True, True, True, False]);  permute_257 = getitem_24 = getitem_25 = getitem_26 = alias_21 = getitem_28 = getitem_29 = getitem_30 = None
    getitem_170: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_9[0]
    getitem_171: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_9[1]
    getitem_172: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_9[2];  _scaled_dot_product_efficient_attention_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_10: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_170, getitem_171, getitem_172]);  getitem_170 = getitem_171 = getitem_172 = None
    view_268: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.view.default(cat_10, [3, 8, 12, 198, 64]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_258: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_268, [1, 3, 0, 2, 4]);  view_268 = None
    clone_46: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_258, memory_format = torch.contiguous_format);  permute_258 = None
    view_269: "f32[8, 198, 2304]" = torch.ops.aten.view.default(clone_46, [8, 198, 2304]);  clone_46 = None
    view_270: "f32[1584, 2304]" = torch.ops.aten.view.default(view_269, [1584, 2304]);  view_269 = None
    permute_259: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_82: "f32[1584, 768]" = torch.ops.aten.mm.default(view_270, permute_259);  permute_259 = None
    permute_260: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_270, [1, 0])
    mm_83: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_260, view_21);  permute_260 = view_21 = None
    permute_261: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_122: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_270, [0], True);  view_270 = None
    view_271: "f32[2304]" = torch.ops.aten.view.default(sum_122, [2304]);  sum_122 = None
    permute_262: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
    view_272: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_82, [8, 198, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_85: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_23);  add_14 = getitem_23 = None
    mul_296: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_4);  sub_85 = None
    mul_297: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_272, primals_30);  primals_30 = None
    mul_298: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_297, 768)
    sum_123: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
    mul_299: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_297, mul_296);  mul_297 = None
    sum_124: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
    mul_300: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_296, sum_124);  sum_124 = None
    sub_86: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_298, sum_123);  mul_298 = sum_123 = None
    sub_87: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_86, mul_300);  sub_86 = mul_300 = None
    div_22: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_301: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_87);  div_22 = sub_87 = None
    mul_302: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_272, mul_296);  mul_296 = None
    sum_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
    sum_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_272, [0, 1]);  view_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_128: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_127, mul_301);  add_127 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_273: "f32[1584, 768]" = torch.ops.aten.view.default(add_128, [1584, 768])
    permute_263: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_84: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_273, permute_263);  permute_263 = None
    permute_264: "f32[768, 1584]" = torch.ops.aten.permute.default(view_273, [1, 0])
    mm_85: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_264, view_19);  permute_264 = view_19 = None
    permute_265: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
    view_274: "f32[768]" = torch.ops.aten.view.default(sum_127, [768]);  sum_127 = None
    permute_266: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    view_275: "f32[8, 198, 3072]" = torch.ops.aten.view.default(mm_84, [8, 198, 3072]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_303: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf_22: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_303);  mul_303 = None
    add_129: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_304: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_129, 0.5);  add_129 = None
    mul_305: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_18, view_18)
    mul_306: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_305, -0.5);  mul_305 = None
    exp_10: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_306);  mul_306 = None
    mul_307: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_308: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_18, mul_307);  view_18 = mul_307 = None
    add_130: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_304, mul_308);  mul_304 = mul_308 = None
    mul_309: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_275, add_130);  view_275 = add_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_276: "f32[1584, 3072]" = torch.ops.aten.view.default(mul_309, [1584, 3072]);  mul_309 = None
    permute_267: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_86: "f32[1584, 768]" = torch.ops.aten.mm.default(view_276, permute_267);  permute_267 = None
    permute_268: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_87: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_268, view_17);  permute_268 = view_17 = None
    permute_269: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_128: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
    view_277: "f32[3072]" = torch.ops.aten.view.default(sum_128, [3072]);  sum_128 = None
    permute_270: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_269, [1, 0]);  permute_269 = None
    view_278: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_86, [8, 198, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_88: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_21);  add_10 = getitem_21 = None
    mul_310: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_3);  sub_88 = None
    mul_311: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_278, primals_24);  primals_24 = None
    mul_312: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_311, 768)
    sum_129: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
    mul_313: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_311, mul_310);  mul_311 = None
    sum_130: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
    mul_314: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_310, sum_130);  sum_130 = None
    sub_89: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_312, sum_129);  mul_312 = sum_129 = None
    sub_90: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_89, mul_314);  sub_89 = mul_314 = None
    div_23: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_315: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_90);  div_23 = sub_90 = None
    mul_316: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_278, mul_310);  mul_310 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
    sum_132: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_131: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_128, mul_315);  add_128 = mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_279: "f32[1584, 768]" = torch.ops.aten.view.default(add_131, [1584, 768])
    permute_271: "f32[768, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_88: "f32[1584, 768]" = torch.ops.aten.mm.default(view_279, permute_271);  permute_271 = None
    permute_272: "f32[768, 1584]" = torch.ops.aten.permute.default(view_279, [1, 0])
    mm_89: "f32[768, 768]" = torch.ops.aten.mm.default(permute_272, view_15);  permute_272 = view_15 = None
    permute_273: "f32[768, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_133: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
    view_280: "f32[768]" = torch.ops.aten.view.default(sum_133, [768]);  sum_133 = None
    permute_274: "f32[768, 768]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
    view_281: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_88, [8, 198, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_282: "f32[8, 198, 12, 64]" = torch.ops.aten.view.default(view_281, [8, 198, 12, 64]);  view_281 = None
    permute_275: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_22: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_275, getitem_13, getitem_14, getitem_15, None, alias_22, getitem_17, getitem_18, getitem_19, 0.0, [True, True, True, False]);  permute_275 = getitem_13 = getitem_14 = getitem_15 = alias_22 = getitem_17 = getitem_18 = getitem_19 = None
    getitem_174: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_10[0]
    getitem_175: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_10[1]
    getitem_176: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_10[2];  _scaled_dot_product_efficient_attention_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_11: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_174, getitem_175, getitem_176]);  getitem_174 = getitem_175 = getitem_176 = None
    view_283: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.view.default(cat_11, [3, 8, 12, 198, 64]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_276: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_283, [1, 3, 0, 2, 4]);  view_283 = None
    clone_47: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
    view_284: "f32[8, 198, 2304]" = torch.ops.aten.view.default(clone_47, [8, 198, 2304]);  clone_47 = None
    view_285: "f32[1584, 2304]" = torch.ops.aten.view.default(view_284, [1584, 2304]);  view_284 = None
    permute_277: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    mm_90: "f32[1584, 768]" = torch.ops.aten.mm.default(view_285, permute_277);  permute_277 = None
    permute_278: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_285, [1, 0])
    mm_91: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_278, view_11);  permute_278 = view_11 = None
    permute_279: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_134: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_285, [0], True);  view_285 = None
    view_286: "f32[2304]" = torch.ops.aten.view.default(sum_134, [2304]);  sum_134 = None
    permute_280: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    view_287: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_90, [8, 198, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_91: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_12);  add_7 = getitem_12 = None
    mul_317: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_2);  sub_91 = None
    mul_318: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_287, primals_18);  primals_18 = None
    mul_319: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_318, 768)
    sum_135: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
    mul_320: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_318, mul_317);  mul_318 = None
    sum_136: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
    mul_321: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_317, sum_136);  sum_136 = None
    sub_92: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_319, sum_135);  mul_319 = sum_135 = None
    sub_93: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_321);  sub_92 = mul_321 = None
    div_24: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_322: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_93);  div_24 = sub_93 = None
    mul_323: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_287, mul_317);  mul_317 = None
    sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
    sum_138: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_287, [0, 1]);  view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_132: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_131, mul_322);  add_131 = mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_288: "f32[1584, 768]" = torch.ops.aten.view.default(add_132, [1584, 768])
    permute_281: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    mm_92: "f32[1584, 3072]" = torch.ops.aten.mm.default(view_288, permute_281);  permute_281 = None
    permute_282: "f32[768, 1584]" = torch.ops.aten.permute.default(view_288, [1, 0])
    mm_93: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_282, view_9);  permute_282 = view_9 = None
    permute_283: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_139: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_288, [0], True);  view_288 = None
    view_289: "f32[768]" = torch.ops.aten.view.default(sum_139, [768]);  sum_139 = None
    permute_284: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None
    view_290: "f32[8, 198, 3072]" = torch.ops.aten.view.default(mm_92, [8, 198, 3072]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_324: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476)
    erf_23: "f32[8, 198, 3072]" = torch.ops.aten.erf.default(mul_324);  mul_324 = None
    add_133: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_325: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(add_133, 0.5);  add_133 = None
    mul_326: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_8, view_8)
    mul_327: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(mul_326, -0.5);  mul_326 = None
    exp_11: "f32[8, 198, 3072]" = torch.ops.aten.exp.default(mul_327);  mul_327 = None
    mul_328: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_329: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_8, mul_328);  view_8 = mul_328 = None
    add_134: "f32[8, 198, 3072]" = torch.ops.aten.add.Tensor(mul_325, mul_329);  mul_325 = mul_329 = None
    mul_330: "f32[8, 198, 3072]" = torch.ops.aten.mul.Tensor(view_290, add_134);  view_290 = add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_291: "f32[1584, 3072]" = torch.ops.aten.view.default(mul_330, [1584, 3072]);  mul_330 = None
    permute_285: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm_94: "f32[1584, 768]" = torch.ops.aten.mm.default(view_291, permute_285);  permute_285 = None
    permute_286: "f32[3072, 1584]" = torch.ops.aten.permute.default(view_291, [1, 0])
    mm_95: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_286, view_7);  permute_286 = view_7 = None
    permute_287: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_140: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
    view_292: "f32[3072]" = torch.ops.aten.view.default(sum_140, [3072]);  sum_140 = None
    permute_288: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
    view_293: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_94, [8, 198, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_94: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_10);  add_3 = getitem_10 = None
    mul_331: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_1);  sub_94 = None
    mul_332: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_293, primals_12);  primals_12 = None
    mul_333: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_332, 768)
    sum_141: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True)
    mul_334: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_332, mul_331);  mul_332 = None
    sum_142: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True);  mul_334 = None
    mul_335: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_331, sum_142);  sum_142 = None
    sub_95: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_333, sum_141);  mul_333 = sum_141 = None
    sub_96: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_335);  sub_95 = mul_335 = None
    div_25: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_336: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_25, sub_96);  div_25 = sub_96 = None
    mul_337: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_293, mul_331);  mul_331 = None
    sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 1]);  mul_337 = None
    sum_144: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_293, [0, 1]);  view_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_135: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_132, mul_336);  add_132 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_294: "f32[1584, 768]" = torch.ops.aten.view.default(add_135, [1584, 768])
    permute_289: "f32[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    mm_96: "f32[1584, 768]" = torch.ops.aten.mm.default(view_294, permute_289);  permute_289 = None
    permute_290: "f32[768, 1584]" = torch.ops.aten.permute.default(view_294, [1, 0])
    mm_97: "f32[768, 768]" = torch.ops.aten.mm.default(permute_290, view_5);  permute_290 = view_5 = None
    permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_145: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_294, [0], True);  view_294 = None
    view_295: "f32[768]" = torch.ops.aten.view.default(sum_145, [768]);  sum_145 = None
    permute_292: "f32[768, 768]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
    view_296: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_96, [8, 198, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_297: "f32[8, 198, 12, 64]" = torch.ops.aten.view.default(view_296, [8, 198, 12, 64]);  view_296 = None
    permute_293: "f32[8, 12, 198, 64]" = torch.ops.aten.permute.default(view_297, [0, 2, 1, 3]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_23: "f32[8, 12, 198, 64]" = torch.ops.aten.alias.default(alias);  alias = None
    _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_293, getitem_2, getitem_3, getitem_4, None, alias_23, getitem_6, getitem_7, getitem_8, 0.0, [True, True, True, False]);  permute_293 = getitem_2 = getitem_3 = getitem_4 = alias_23 = getitem_6 = getitem_7 = getitem_8 = None
    getitem_178: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_11[0]
    getitem_179: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_11[1]
    getitem_180: "f32[8, 12, 198, 64]" = _scaled_dot_product_efficient_attention_backward_11[2];  _scaled_dot_product_efficient_attention_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_12: "f32[24, 12, 198, 64]" = torch.ops.aten.cat.default([getitem_178, getitem_179, getitem_180]);  getitem_178 = getitem_179 = getitem_180 = None
    view_298: "f32[3, 8, 12, 198, 64]" = torch.ops.aten.view.default(cat_12, [3, 8, 12, 198, 64]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_294: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.permute.default(view_298, [1, 3, 0, 2, 4]);  view_298 = None
    clone_48: "f32[8, 198, 3, 12, 64]" = torch.ops.aten.clone.default(permute_294, memory_format = torch.contiguous_format);  permute_294 = None
    view_299: "f32[8, 198, 2304]" = torch.ops.aten.view.default(clone_48, [8, 198, 2304]);  clone_48 = None
    view_300: "f32[1584, 2304]" = torch.ops.aten.view.default(view_299, [1584, 2304]);  view_299 = None
    permute_295: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_98: "f32[1584, 768]" = torch.ops.aten.mm.default(view_300, permute_295);  permute_295 = None
    permute_296: "f32[2304, 1584]" = torch.ops.aten.permute.default(view_300, [1, 0])
    mm_99: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_296, view_1);  permute_296 = view_1 = None
    permute_297: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_146: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_300, [0], True);  view_300 = None
    view_301: "f32[2304]" = torch.ops.aten.view.default(sum_146, [2304]);  sum_146 = None
    permute_298: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
    view_302: "f32[8, 198, 768]" = torch.ops.aten.view.default(mm_98, [8, 198, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_97: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul_338: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt);  sub_97 = None
    mul_339: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_302, primals_6);  primals_6 = None
    mul_340: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_339, 768)
    sum_147: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True)
    mul_341: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_339, mul_338);  mul_339 = None
    sum_148: "f32[8, 198, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
    mul_342: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(mul_338, sum_148);  sum_148 = None
    sub_98: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(mul_340, sum_147);  mul_340 = sum_147 = None
    sub_99: "f32[8, 198, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_342);  sub_98 = mul_342 = None
    div_26: "f32[8, 198, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_343: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(div_26, sub_99);  div_26 = sub_99 = None
    mul_344: "f32[8, 198, 768]" = torch.ops.aten.mul.Tensor(view_302, mul_338);  mul_338 = None
    sum_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 1]);  mul_344 = None
    sum_150: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_302, [0, 1]);  view_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_136: "f32[8, 198, 768]" = torch.ops.aten.add.Tensor(add_135, mul_343);  add_135 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:104, code: x = x + pos_embed
    sum_151: "f32[1, 198, 768]" = torch.ops.aten.sum.dim_IntList(add_136, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:99, code: x = torch.cat((
    slice_3: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_136, 1, 0, 1)
    slice_4: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_136, 1, 1, 2)
    slice_5: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(add_136, 1, 2, 198);  add_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:101, code: self.dist_token.expand(x.shape[0], -1, -1),
    sum_152: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_4, [0], True);  slice_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:100, code: self.cls_token.expand(x.shape[0], -1, -1),
    sum_153: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_3, [0], True);  slice_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_299: "f32[8, 768, 196]" = torch.ops.aten.permute.default(slice_5, [0, 2, 1]);  slice_5 = None
    view_303: "f32[8, 768, 14, 14]" = torch.ops.aten.view.default(permute_299, [8, 768, 14, 14]);  permute_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    sum_154: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_303, [0, 2, 3])
    convolution_backward = torch.ops.aten.convolution_backward.default(view_303, primals_156, primals_4, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_303 = primals_156 = primals_4 = None
    getitem_183: "f32[768, 3, 16, 16]" = convolution_backward[1];  convolution_backward = None
    return pytree.tree_unflatten([div, sum_151, sum_153, sum_152, getitem_183, sum_154, sum_149, sum_150, permute_298, view_301, permute_292, view_295, sum_143, sum_144, permute_288, view_292, permute_284, view_289, sum_137, sum_138, permute_280, view_286, permute_274, view_280, sum_131, sum_132, permute_270, view_277, permute_266, view_274, sum_125, sum_126, permute_262, view_271, permute_256, view_265, sum_119, sum_120, permute_252, view_262, permute_248, view_259, sum_113, sum_114, permute_244, view_256, permute_238, view_250, sum_107, sum_108, permute_234, view_247, permute_230, view_244, sum_101, sum_102, permute_226, view_241, permute_220, view_235, sum_95, sum_96, permute_216, view_232, permute_212, view_229, sum_89, sum_90, permute_208, view_226, permute_202, view_220, sum_83, sum_84, permute_198, view_217, permute_194, view_214, sum_77, sum_78, permute_190, view_211, permute_184, view_205, sum_71, sum_72, permute_180, view_202, permute_176, view_199, sum_65, sum_66, permute_172, view_196, permute_166, view_190, sum_59, sum_60, permute_162, view_187, permute_158, view_184, sum_53, sum_54, permute_154, view_181, permute_148, view_175, sum_47, sum_48, permute_144, view_172, permute_140, view_169, sum_41, sum_42, permute_136, view_166, permute_130, view_160, sum_35, sum_36, permute_126, view_157, permute_122, view_154, sum_29, sum_30, permute_118, view_151, permute_112, view_145, sum_23, sum_24, permute_108, view_142, permute_104, view_139, sum_17, sum_18, permute_100, view_136, permute_94, view_130, sum_11, sum_12, permute_90, view_127, permute_86, view_124, sum_5, sum_6, permute_82, view_122, permute_78, view_121, None], self._out_spec)
    