from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[1, 197, 768]"; primals_2: "f32[1, 1, 768]"; primals_3: "f32[768, 3, 16, 16]"; primals_4: "f32[768]"; primals_5: "f32[768]"; primals_6: "f32[768]"; primals_7: "f32[2304, 768]"; primals_8: "f32[2304]"; primals_9: "f32[768, 768]"; primals_10: "f32[768]"; primals_11: "f32[768]"; primals_12: "f32[768]"; primals_13: "f32[3072, 768]"; primals_14: "f32[3072]"; primals_15: "f32[768, 3072]"; primals_16: "f32[768]"; primals_17: "f32[768]"; primals_18: "f32[768]"; primals_19: "f32[2304, 768]"; primals_20: "f32[2304]"; primals_21: "f32[768, 768]"; primals_22: "f32[768]"; primals_23: "f32[768]"; primals_24: "f32[768]"; primals_25: "f32[3072, 768]"; primals_26: "f32[3072]"; primals_27: "f32[768, 3072]"; primals_28: "f32[768]"; primals_29: "f32[768]"; primals_30: "f32[768]"; primals_31: "f32[2304, 768]"; primals_32: "f32[2304]"; primals_33: "f32[768, 768]"; primals_34: "f32[768]"; primals_35: "f32[768]"; primals_36: "f32[768]"; primals_37: "f32[3072, 768]"; primals_38: "f32[3072]"; primals_39: "f32[768, 3072]"; primals_40: "f32[768]"; primals_41: "f32[768]"; primals_42: "f32[768]"; primals_43: "f32[2304, 768]"; primals_44: "f32[2304]"; primals_45: "f32[768, 768]"; primals_46: "f32[768]"; primals_47: "f32[768]"; primals_48: "f32[768]"; primals_49: "f32[3072, 768]"; primals_50: "f32[3072]"; primals_51: "f32[768, 3072]"; primals_52: "f32[768]"; primals_53: "f32[768]"; primals_54: "f32[768]"; primals_55: "f32[2304, 768]"; primals_56: "f32[2304]"; primals_57: "f32[768, 768]"; primals_58: "f32[768]"; primals_59: "f32[768]"; primals_60: "f32[768]"; primals_61: "f32[3072, 768]"; primals_62: "f32[3072]"; primals_63: "f32[768, 3072]"; primals_64: "f32[768]"; primals_65: "f32[768]"; primals_66: "f32[768]"; primals_67: "f32[2304, 768]"; primals_68: "f32[2304]"; primals_69: "f32[768, 768]"; primals_70: "f32[768]"; primals_71: "f32[768]"; primals_72: "f32[768]"; primals_73: "f32[3072, 768]"; primals_74: "f32[3072]"; primals_75: "f32[768, 3072]"; primals_76: "f32[768]"; primals_77: "f32[768]"; primals_78: "f32[768]"; primals_79: "f32[2304, 768]"; primals_80: "f32[2304]"; primals_81: "f32[768, 768]"; primals_82: "f32[768]"; primals_83: "f32[768]"; primals_84: "f32[768]"; primals_85: "f32[3072, 768]"; primals_86: "f32[3072]"; primals_87: "f32[768, 3072]"; primals_88: "f32[768]"; primals_89: "f32[768]"; primals_90: "f32[768]"; primals_91: "f32[2304, 768]"; primals_92: "f32[2304]"; primals_93: "f32[768, 768]"; primals_94: "f32[768]"; primals_95: "f32[768]"; primals_96: "f32[768]"; primals_97: "f32[3072, 768]"; primals_98: "f32[3072]"; primals_99: "f32[768, 3072]"; primals_100: "f32[768]"; primals_101: "f32[768]"; primals_102: "f32[768]"; primals_103: "f32[2304, 768]"; primals_104: "f32[2304]"; primals_105: "f32[768, 768]"; primals_106: "f32[768]"; primals_107: "f32[768]"; primals_108: "f32[768]"; primals_109: "f32[3072, 768]"; primals_110: "f32[3072]"; primals_111: "f32[768, 3072]"; primals_112: "f32[768]"; primals_113: "f32[768]"; primals_114: "f32[768]"; primals_115: "f32[2304, 768]"; primals_116: "f32[2304]"; primals_117: "f32[768, 768]"; primals_118: "f32[768]"; primals_119: "f32[768]"; primals_120: "f32[768]"; primals_121: "f32[3072, 768]"; primals_122: "f32[3072]"; primals_123: "f32[768, 3072]"; primals_124: "f32[768]"; primals_125: "f32[768]"; primals_126: "f32[768]"; primals_127: "f32[2304, 768]"; primals_128: "f32[2304]"; primals_129: "f32[768, 768]"; primals_130: "f32[768]"; primals_131: "f32[768]"; primals_132: "f32[768]"; primals_133: "f32[3072, 768]"; primals_134: "f32[3072]"; primals_135: "f32[768, 3072]"; primals_136: "f32[768]"; primals_137: "f32[768]"; primals_138: "f32[768]"; primals_139: "f32[2304, 768]"; primals_140: "f32[2304]"; primals_141: "f32[768, 768]"; primals_142: "f32[768]"; primals_143: "f32[768]"; primals_144: "f32[768]"; primals_145: "f32[3072, 768]"; primals_146: "f32[3072]"; primals_147: "f32[768, 3072]"; primals_148: "f32[768]"; primals_149: "f32[768]"; primals_150: "f32[768]"; primals_151: "f32[1000, 768]"; primals_152: "f32[1000]"; primals_153: "f32[8, 3, 224, 224]"; tangents_1: "f32[8, 1000]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution: "f32[8, 768, 14, 14]" = torch.ops.aten.convolution.default(primals_153, primals_3, primals_4, [16, 16], [0, 0], [1, 1], False, [0, 0], 1);  primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    view: "f32[8, 768, 196]" = torch.ops.aten.view.default(convolution, [8, 768, 196]);  convolution = None
    permute: "f32[8, 196, 768]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:579, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    expand: "f32[8, 1, 768]" = torch.ops.aten.expand.default(primals_2, [8, -1, -1]);  primals_2 = None
    cat: "f32[8, 197, 768]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:580, code: x = x + pos_embed
    add: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(cat, primals_1);  cat = primals_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:581, code: return self.pos_drop(x)
    clone: "f32[8, 197, 768]" = torch.ops.aten.clone.default(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean = torch.ops.aten.var_mean.correction(clone, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 197, 1]" = var_mean[0]
    getitem_1: "f32[8, 197, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(clone, getitem_1)
    mul: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul, primals_5);  mul = None
    add_2: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_1: "f32[1576, 768]" = torch.ops.aten.view.default(add_2, [1576, 768]);  add_2 = None
    permute_1: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm: "f32[1576, 2304]" = torch.ops.aten.addmm.default(primals_8, view_1, permute_1);  primals_8 = None
    view_2: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm, [8, 197, 2304]);  addmm = None
    view_3: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_2, [8, 197, 3, 12, 64]);  view_2 = None
    permute_2: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
    getitem_2: "f32[8, 12, 197, 64]" = unbind[0]
    getitem_3: "f32[8, 12, 197, 64]" = unbind[1]
    getitem_4: "f32[8, 12, 197, 64]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_2, getitem_3, getitem_4)
    getitem_5: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention[0]
    getitem_6: "f32[8, 12, 197]" = _scaled_dot_product_flash_attention[1]
    getitem_7: "i32[]" = _scaled_dot_product_flash_attention[2]
    getitem_8: "i32[]" = _scaled_dot_product_flash_attention[3]
    getitem_11: "i64[]" = _scaled_dot_product_flash_attention[6]
    getitem_12: "i64[]" = _scaled_dot_product_flash_attention[7];  _scaled_dot_product_flash_attention = None
    alias: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_3: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    view_4: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_3, [8, 197, 768]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_5: "f32[1576, 768]" = torch.ops.aten.view.default(view_4, [1576, 768]);  view_4 = None
    permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
    addmm_1: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_10, view_5, permute_4);  primals_10 = None
    view_6: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_1, [8, 197, 768]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_1: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_6);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_3: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(clone, clone_1);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 197, 1]" = var_mean_1[0]
    getitem_15: "f32[8, 197, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_1: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_1: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_15)
    mul_2: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_3: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_11);  mul_2 = None
    add_5: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_12);  mul_3 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_7: "f32[1576, 768]" = torch.ops.aten.view.default(add_5, [1576, 768]);  add_5 = None
    permute_5: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_2: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_14, view_7, permute_5);  primals_14 = None
    view_8: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_2, [8, 197, 3072]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_4: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_8, 0.5)
    mul_5: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476)
    erf: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_6: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_6: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_2: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_6);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_9: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_2, [1576, 3072]);  clone_2 = None
    permute_6: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_3: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_16, view_9, permute_6);  primals_16 = None
    view_10: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_3, [8, 197, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_3: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_10);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_7: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_3, clone_3);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 197, 1]" = var_mean_2[0]
    getitem_17: "f32[8, 197, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_2: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_2: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_17)
    mul_7: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_8: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_7, primals_17);  mul_7 = None
    add_9: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_8, primals_18);  mul_8 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_11: "f32[1576, 768]" = torch.ops.aten.view.default(add_9, [1576, 768]);  add_9 = None
    permute_7: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    addmm_4: "f32[1576, 2304]" = torch.ops.aten.addmm.default(primals_20, view_11, permute_7);  primals_20 = None
    view_12: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_4, [8, 197, 2304]);  addmm_4 = None
    view_13: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_12, [8, 197, 3, 12, 64]);  view_12 = None
    permute_8: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_13, [2, 0, 3, 1, 4]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_8);  permute_8 = None
    getitem_18: "f32[8, 12, 197, 64]" = unbind_1[0]
    getitem_19: "f32[8, 12, 197, 64]" = unbind_1[1]
    getitem_20: "f32[8, 12, 197, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_1 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_18, getitem_19, getitem_20)
    getitem_21: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_1[0]
    getitem_22: "f32[8, 12, 197]" = _scaled_dot_product_flash_attention_1[1]
    getitem_23: "i32[]" = _scaled_dot_product_flash_attention_1[2]
    getitem_24: "i32[]" = _scaled_dot_product_flash_attention_1[3]
    getitem_27: "i64[]" = _scaled_dot_product_flash_attention_1[6]
    getitem_28: "i64[]" = _scaled_dot_product_flash_attention_1[7];  _scaled_dot_product_flash_attention_1 = None
    alias_1: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_9: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_21, [0, 2, 1, 3]);  getitem_21 = None
    view_14: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_9, [8, 197, 768]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_15: "f32[1576, 768]" = torch.ops.aten.view.default(view_14, [1576, 768]);  view_14 = None
    permute_10: "f32[768, 768]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_5: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_22, view_15, permute_10);  primals_22 = None
    view_16: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_5, [8, 197, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_4: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_16);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_10: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_7, clone_4);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 197, 1]" = var_mean_3[0]
    getitem_31: "f32[8, 197, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_3: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_3: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_31)
    mul_9: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_10: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_23);  mul_9 = None
    add_12: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_24);  mul_10 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_17: "f32[1576, 768]" = torch.ops.aten.view.default(add_12, [1576, 768]);  add_12 = None
    permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_6: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_26, view_17, permute_11);  primals_26 = None
    view_18: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_6, [8, 197, 3072]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_11: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_12: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf_1: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_13: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_13: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_5: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_13);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_19: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_5, [1576, 3072]);  clone_5 = None
    permute_12: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm_7: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_28, view_19, permute_12);  primals_28 = None
    view_20: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_7, [8, 197, 768]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_6: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_14: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_10, clone_6);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 197, 1]" = var_mean_4[0]
    getitem_33: "f32[8, 197, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_4: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_4: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_33)
    mul_14: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_15: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_29);  mul_14 = None
    add_16: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_15, primals_30);  mul_15 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_21: "f32[1576, 768]" = torch.ops.aten.view.default(add_16, [1576, 768]);  add_16 = None
    permute_13: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    addmm_8: "f32[1576, 2304]" = torch.ops.aten.addmm.default(primals_32, view_21, permute_13);  primals_32 = None
    view_22: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_8, [8, 197, 2304]);  addmm_8 = None
    view_23: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_22, [8, 197, 3, 12, 64]);  view_22 = None
    permute_14: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_23, [2, 0, 3, 1, 4]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_14);  permute_14 = None
    getitem_34: "f32[8, 12, 197, 64]" = unbind_2[0]
    getitem_35: "f32[8, 12, 197, 64]" = unbind_2[1]
    getitem_36: "f32[8, 12, 197, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_2 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_34, getitem_35, getitem_36)
    getitem_37: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_2[0]
    getitem_38: "f32[8, 12, 197]" = _scaled_dot_product_flash_attention_2[1]
    getitem_39: "i32[]" = _scaled_dot_product_flash_attention_2[2]
    getitem_40: "i32[]" = _scaled_dot_product_flash_attention_2[3]
    getitem_43: "i64[]" = _scaled_dot_product_flash_attention_2[6]
    getitem_44: "i64[]" = _scaled_dot_product_flash_attention_2[7];  _scaled_dot_product_flash_attention_2 = None
    alias_2: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_15: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_37, [0, 2, 1, 3]);  getitem_37 = None
    view_24: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_15, [8, 197, 768]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_25: "f32[1576, 768]" = torch.ops.aten.view.default(view_24, [1576, 768]);  view_24 = None
    permute_16: "f32[768, 768]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
    addmm_9: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_34, view_25, permute_16);  primals_34 = None
    view_26: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_9, [8, 197, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_7: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_26);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_17: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_14, clone_7);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 197, 1]" = var_mean_5[0]
    getitem_47: "f32[8, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_5: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_5: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_47)
    mul_16: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_17: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_35);  mul_16 = None
    add_19: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_36);  mul_17 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_27: "f32[1576, 768]" = torch.ops.aten.view.default(add_19, [1576, 768]);  add_19 = None
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_10: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_38, view_27, permute_17);  primals_38 = None
    view_28: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 197, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.5)
    mul_19: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476)
    erf_2: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_20: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_20: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_8: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_29: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_8, [1576, 3072]);  clone_8 = None
    permute_18: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    addmm_11: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_40, view_29, permute_18);  primals_40 = None
    view_30: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_11, [8, 197, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_9: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_30);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_21: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_17, clone_9);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 197, 1]" = var_mean_6[0]
    getitem_49: "f32[8, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    add_22: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_6: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_6: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_49)
    mul_21: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_22: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_41);  mul_21 = None
    add_23: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_22, primals_42);  mul_22 = primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_31: "f32[1576, 768]" = torch.ops.aten.view.default(add_23, [1576, 768]);  add_23 = None
    permute_19: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    addmm_12: "f32[1576, 2304]" = torch.ops.aten.addmm.default(primals_44, view_31, permute_19);  primals_44 = None
    view_32: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_12, [8, 197, 2304]);  addmm_12 = None
    view_33: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_32, [8, 197, 3, 12, 64]);  view_32 = None
    permute_20: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_33, [2, 0, 3, 1, 4]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_20);  permute_20 = None
    getitem_50: "f32[8, 12, 197, 64]" = unbind_3[0]
    getitem_51: "f32[8, 12, 197, 64]" = unbind_3[1]
    getitem_52: "f32[8, 12, 197, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_3 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_50, getitem_51, getitem_52)
    getitem_53: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_3[0]
    getitem_54: "f32[8, 12, 197]" = _scaled_dot_product_flash_attention_3[1]
    getitem_55: "i32[]" = _scaled_dot_product_flash_attention_3[2]
    getitem_56: "i32[]" = _scaled_dot_product_flash_attention_3[3]
    getitem_59: "i64[]" = _scaled_dot_product_flash_attention_3[6]
    getitem_60: "i64[]" = _scaled_dot_product_flash_attention_3[7];  _scaled_dot_product_flash_attention_3 = None
    alias_3: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_21: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_53, [0, 2, 1, 3]);  getitem_53 = None
    view_34: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_21, [8, 197, 768]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_35: "f32[1576, 768]" = torch.ops.aten.view.default(view_34, [1576, 768]);  view_34 = None
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    addmm_13: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_46, view_35, permute_22);  primals_46 = None
    view_36: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_13, [8, 197, 768]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_10: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_36);  view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_24: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_21, clone_10);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_62: "f32[8, 197, 1]" = var_mean_7[0]
    getitem_63: "f32[8, 197, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
    rsqrt_7: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_7: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_63)
    mul_23: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_24: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_23, primals_47);  mul_23 = None
    add_26: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_24, primals_48);  mul_24 = primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_37: "f32[1576, 768]" = torch.ops.aten.view.default(add_26, [1576, 768]);  add_26 = None
    permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    addmm_14: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_50, view_37, permute_23);  primals_50 = None
    view_38: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_14, [8, 197, 3072]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_25: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_26: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_3: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
    add_27: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_27: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_25, add_27);  mul_25 = add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_11: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_27);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_39: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_11, [1576, 3072]);  clone_11 = None
    permute_24: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    addmm_15: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_52, view_39, permute_24);  primals_52 = None
    view_40: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_15, [8, 197, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_12: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_28: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_24, clone_12);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 197, 1]" = var_mean_8[0]
    getitem_65: "f32[8, 197, 1]" = var_mean_8[1];  var_mean_8 = None
    add_29: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_8: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_8: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_65)
    mul_28: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_29: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_53);  mul_28 = None
    add_30: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_29, primals_54);  mul_29 = primals_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_41: "f32[1576, 768]" = torch.ops.aten.view.default(add_30, [1576, 768]);  add_30 = None
    permute_25: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_16: "f32[1576, 2304]" = torch.ops.aten.addmm.default(primals_56, view_41, permute_25);  primals_56 = None
    view_42: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_16, [8, 197, 2304]);  addmm_16 = None
    view_43: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_42, [8, 197, 3, 12, 64]);  view_42 = None
    permute_26: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_43, [2, 0, 3, 1, 4]);  view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_26);  permute_26 = None
    getitem_66: "f32[8, 12, 197, 64]" = unbind_4[0]
    getitem_67: "f32[8, 12, 197, 64]" = unbind_4[1]
    getitem_68: "f32[8, 12, 197, 64]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_4 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_66, getitem_67, getitem_68)
    getitem_69: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_4[0]
    getitem_70: "f32[8, 12, 197]" = _scaled_dot_product_flash_attention_4[1]
    getitem_71: "i32[]" = _scaled_dot_product_flash_attention_4[2]
    getitem_72: "i32[]" = _scaled_dot_product_flash_attention_4[3]
    getitem_75: "i64[]" = _scaled_dot_product_flash_attention_4[6]
    getitem_76: "i64[]" = _scaled_dot_product_flash_attention_4[7];  _scaled_dot_product_flash_attention_4 = None
    alias_4: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_69)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_27: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_69, [0, 2, 1, 3]);  getitem_69 = None
    view_44: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_27, [8, 197, 768]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_45: "f32[1576, 768]" = torch.ops.aten.view.default(view_44, [1576, 768]);  view_44 = None
    permute_28: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    addmm_17: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_58, view_45, permute_28);  primals_58 = None
    view_46: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_17, [8, 197, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_13: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_46);  view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_31: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_28, clone_13);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_78: "f32[8, 197, 1]" = var_mean_9[0]
    getitem_79: "f32[8, 197, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_9: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_9: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_79)
    mul_30: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_31: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_59);  mul_30 = None
    add_33: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_60);  mul_31 = primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_47: "f32[1576, 768]" = torch.ops.aten.view.default(add_33, [1576, 768]);  add_33 = None
    permute_29: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    addmm_18: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_62, view_47, permute_29);  primals_62 = None
    view_48: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_18, [8, 197, 3072]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_48, 0.5)
    mul_33: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476)
    erf_4: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_34: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_34: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_34);  mul_32 = add_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_14: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_49: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_14, [1576, 3072]);  clone_14 = None
    permute_30: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    addmm_19: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_64, view_49, permute_30);  primals_64 = None
    view_50: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_19, [8, 197, 768]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_15: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_50);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_35: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_31, clone_15);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_80: "f32[8, 197, 1]" = var_mean_10[0]
    getitem_81: "f32[8, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    add_36: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
    rsqrt_10: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_10: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_81)
    mul_35: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_36: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_65);  mul_35 = None
    add_37: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_36, primals_66);  mul_36 = primals_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_51: "f32[1576, 768]" = torch.ops.aten.view.default(add_37, [1576, 768]);  add_37 = None
    permute_31: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    addmm_20: "f32[1576, 2304]" = torch.ops.aten.addmm.default(primals_68, view_51, permute_31);  primals_68 = None
    view_52: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_20, [8, 197, 2304]);  addmm_20 = None
    view_53: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_52, [8, 197, 3, 12, 64]);  view_52 = None
    permute_32: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_53, [2, 0, 3, 1, 4]);  view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_32);  permute_32 = None
    getitem_82: "f32[8, 12, 197, 64]" = unbind_5[0]
    getitem_83: "f32[8, 12, 197, 64]" = unbind_5[1]
    getitem_84: "f32[8, 12, 197, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_5 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_82, getitem_83, getitem_84)
    getitem_85: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_5[0]
    getitem_86: "f32[8, 12, 197]" = _scaled_dot_product_flash_attention_5[1]
    getitem_87: "i32[]" = _scaled_dot_product_flash_attention_5[2]
    getitem_88: "i32[]" = _scaled_dot_product_flash_attention_5[3]
    getitem_91: "i64[]" = _scaled_dot_product_flash_attention_5[6]
    getitem_92: "i64[]" = _scaled_dot_product_flash_attention_5[7];  _scaled_dot_product_flash_attention_5 = None
    alias_5: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_85)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_33: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_85, [0, 2, 1, 3]);  getitem_85 = None
    view_54: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_33, [8, 197, 768]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_55: "f32[1576, 768]" = torch.ops.aten.view.default(view_54, [1576, 768]);  view_54 = None
    permute_34: "f32[768, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    addmm_21: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_70, view_55, permute_34);  primals_70 = None
    view_56: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_21, [8, 197, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_16: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_56);  view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_38: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_35, clone_16);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_94: "f32[8, 197, 1]" = var_mean_11[0]
    getitem_95: "f32[8, 197, 1]" = var_mean_11[1];  var_mean_11 = None
    add_39: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
    rsqrt_11: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_11: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_95)
    mul_37: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_38: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_71);  mul_37 = None
    add_40: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_72);  mul_38 = primals_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_57: "f32[1576, 768]" = torch.ops.aten.view.default(add_40, [1576, 768]);  add_40 = None
    permute_35: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_22: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_74, view_57, permute_35);  primals_74 = None
    view_58: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 197, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_39: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_40: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
    erf_5: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_41: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_41: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_39, add_41);  mul_39 = add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_17: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_41);  mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_59: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_17, [1576, 3072]);  clone_17 = None
    permute_36: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    addmm_23: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_76, view_59, permute_36);  primals_76 = None
    view_60: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_23, [8, 197, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_18: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_42: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_38, clone_18);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_96: "f32[8, 197, 1]" = var_mean_12[0]
    getitem_97: "f32[8, 197, 1]" = var_mean_12[1];  var_mean_12 = None
    add_43: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
    rsqrt_12: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_12: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_97)
    mul_42: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_43: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_77);  mul_42 = None
    add_44: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_43, primals_78);  mul_43 = primals_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_61: "f32[1576, 768]" = torch.ops.aten.view.default(add_44, [1576, 768]);  add_44 = None
    permute_37: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    addmm_24: "f32[1576, 2304]" = torch.ops.aten.addmm.default(primals_80, view_61, permute_37);  primals_80 = None
    view_62: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_24, [8, 197, 2304]);  addmm_24 = None
    view_63: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_62, [8, 197, 3, 12, 64]);  view_62 = None
    permute_38: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_63, [2, 0, 3, 1, 4]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_38);  permute_38 = None
    getitem_98: "f32[8, 12, 197, 64]" = unbind_6[0]
    getitem_99: "f32[8, 12, 197, 64]" = unbind_6[1]
    getitem_100: "f32[8, 12, 197, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_6 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_98, getitem_99, getitem_100)
    getitem_101: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_6[0]
    getitem_102: "f32[8, 12, 197]" = _scaled_dot_product_flash_attention_6[1]
    getitem_103: "i32[]" = _scaled_dot_product_flash_attention_6[2]
    getitem_104: "i32[]" = _scaled_dot_product_flash_attention_6[3]
    getitem_107: "i64[]" = _scaled_dot_product_flash_attention_6[6]
    getitem_108: "i64[]" = _scaled_dot_product_flash_attention_6[7];  _scaled_dot_product_flash_attention_6 = None
    alias_6: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_101)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_39: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_101, [0, 2, 1, 3]);  getitem_101 = None
    view_64: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_39, [8, 197, 768]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_65: "f32[1576, 768]" = torch.ops.aten.view.default(view_64, [1576, 768]);  view_64 = None
    permute_40: "f32[768, 768]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_25: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_82, view_65, permute_40);  primals_82 = None
    view_66: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_25, [8, 197, 768]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_19: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_66);  view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_45: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_42, clone_19);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 197, 1]" = var_mean_13[0]
    getitem_111: "f32[8, 197, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_13: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_13: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_111)
    mul_44: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_45: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_83);  mul_44 = None
    add_47: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_84);  mul_45 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_67: "f32[1576, 768]" = torch.ops.aten.view.default(add_47, [1576, 768]);  add_47 = None
    permute_41: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_26: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_86, view_67, permute_41);  primals_86 = None
    view_68: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_26, [8, 197, 3072]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_68, 0.5)
    mul_47: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476)
    erf_6: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_48: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_48: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_46, add_48);  mul_46 = add_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_20: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_69: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_20, [1576, 3072]);  clone_20 = None
    permute_42: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_27: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_88, view_69, permute_42);  primals_88 = None
    view_70: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_27, [8, 197, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_21: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_70);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_49: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_45, clone_21);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 197, 1]" = var_mean_14[0]
    getitem_113: "f32[8, 197, 1]" = var_mean_14[1];  var_mean_14 = None
    add_50: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
    rsqrt_14: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_14: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_113)
    mul_49: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_50: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_89);  mul_49 = None
    add_51: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_50, primals_90);  mul_50 = primals_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_71: "f32[1576, 768]" = torch.ops.aten.view.default(add_51, [1576, 768]);  add_51 = None
    permute_43: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_28: "f32[1576, 2304]" = torch.ops.aten.addmm.default(primals_92, view_71, permute_43);  primals_92 = None
    view_72: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_28, [8, 197, 2304]);  addmm_28 = None
    view_73: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_72, [8, 197, 3, 12, 64]);  view_72 = None
    permute_44: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_73, [2, 0, 3, 1, 4]);  view_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_44);  permute_44 = None
    getitem_114: "f32[8, 12, 197, 64]" = unbind_7[0]
    getitem_115: "f32[8, 12, 197, 64]" = unbind_7[1]
    getitem_116: "f32[8, 12, 197, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_7 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_114, getitem_115, getitem_116)
    getitem_117: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_7[0]
    getitem_118: "f32[8, 12, 197]" = _scaled_dot_product_flash_attention_7[1]
    getitem_119: "i32[]" = _scaled_dot_product_flash_attention_7[2]
    getitem_120: "i32[]" = _scaled_dot_product_flash_attention_7[3]
    getitem_123: "i64[]" = _scaled_dot_product_flash_attention_7[6]
    getitem_124: "i64[]" = _scaled_dot_product_flash_attention_7[7];  _scaled_dot_product_flash_attention_7 = None
    alias_7: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_117)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_45: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_117, [0, 2, 1, 3]);  getitem_117 = None
    view_74: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_45, [8, 197, 768]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_75: "f32[1576, 768]" = torch.ops.aten.view.default(view_74, [1576, 768]);  view_74 = None
    permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_29: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_94, view_75, permute_46);  primals_94 = None
    view_76: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_29, [8, 197, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_22: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_76);  view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_52: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_49, clone_22);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_126: "f32[8, 197, 1]" = var_mean_15[0]
    getitem_127: "f32[8, 197, 1]" = var_mean_15[1];  var_mean_15 = None
    add_53: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
    rsqrt_15: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_15: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_127)
    mul_51: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_52: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_51, primals_95);  mul_51 = None
    add_54: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_52, primals_96);  mul_52 = primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_77: "f32[1576, 768]" = torch.ops.aten.view.default(add_54, [1576, 768]);  add_54 = None
    permute_47: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_30: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_98, view_77, permute_47);  primals_98 = None
    view_78: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_30, [8, 197, 3072]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_53: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_54: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
    erf_7: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_55: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_55: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_53, add_55);  mul_53 = add_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_55);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_79: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_23, [1576, 3072]);  clone_23 = None
    permute_48: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_31: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_100, view_79, permute_48);  primals_100 = None
    view_80: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_31, [8, 197, 768]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_56: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_52, clone_24);  clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_128: "f32[8, 197, 1]" = var_mean_16[0]
    getitem_129: "f32[8, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    add_57: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
    rsqrt_16: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_16: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_129)
    mul_56: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_57: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_56, primals_101);  mul_56 = None
    add_58: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_57, primals_102);  mul_57 = primals_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_81: "f32[1576, 768]" = torch.ops.aten.view.default(add_58, [1576, 768]);  add_58 = None
    permute_49: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_32: "f32[1576, 2304]" = torch.ops.aten.addmm.default(primals_104, view_81, permute_49);  primals_104 = None
    view_82: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_32, [8, 197, 2304]);  addmm_32 = None
    view_83: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_82, [8, 197, 3, 12, 64]);  view_82 = None
    permute_50: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_83, [2, 0, 3, 1, 4]);  view_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_50);  permute_50 = None
    getitem_130: "f32[8, 12, 197, 64]" = unbind_8[0]
    getitem_131: "f32[8, 12, 197, 64]" = unbind_8[1]
    getitem_132: "f32[8, 12, 197, 64]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_8 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_130, getitem_131, getitem_132)
    getitem_133: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_8[0]
    getitem_134: "f32[8, 12, 197]" = _scaled_dot_product_flash_attention_8[1]
    getitem_135: "i32[]" = _scaled_dot_product_flash_attention_8[2]
    getitem_136: "i32[]" = _scaled_dot_product_flash_attention_8[3]
    getitem_139: "i64[]" = _scaled_dot_product_flash_attention_8[6]
    getitem_140: "i64[]" = _scaled_dot_product_flash_attention_8[7];  _scaled_dot_product_flash_attention_8 = None
    alias_8: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_133)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_51: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None
    view_84: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_51, [8, 197, 768]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_85: "f32[1576, 768]" = torch.ops.aten.view.default(view_84, [1576, 768]);  view_84 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_33: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_106, view_85, permute_52);  primals_106 = None
    view_86: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_33, [8, 197, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_25: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_86);  view_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_59: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_56, clone_25);  clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_142: "f32[8, 197, 1]" = var_mean_17[0]
    getitem_143: "f32[8, 197, 1]" = var_mean_17[1];  var_mean_17 = None
    add_60: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
    rsqrt_17: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_17: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_143)
    mul_58: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_59: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_107);  mul_58 = None
    add_61: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_108);  mul_59 = primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_87: "f32[1576, 768]" = torch.ops.aten.view.default(add_61, [1576, 768]);  add_61 = None
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    addmm_34: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_110, view_87, permute_53);  primals_110 = None
    view_88: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 197, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_60: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.5)
    mul_61: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476)
    erf_8: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_62: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_62: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_60, add_62);  mul_60 = add_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_26: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_62);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_89: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_26, [1576, 3072]);  clone_26 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_35: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_112, view_89, permute_54);  primals_112 = None
    view_90: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_35, [8, 197, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_27: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_90);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_63: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_59, clone_27);  clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 197, 1]" = var_mean_18[0]
    getitem_145: "f32[8, 197, 1]" = var_mean_18[1];  var_mean_18 = None
    add_64: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_18: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_18: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_145)
    mul_63: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_64: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_63, primals_113);  mul_63 = None
    add_65: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_64, primals_114);  mul_64 = primals_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_91: "f32[1576, 768]" = torch.ops.aten.view.default(add_65, [1576, 768]);  add_65 = None
    permute_55: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_36: "f32[1576, 2304]" = torch.ops.aten.addmm.default(primals_116, view_91, permute_55);  primals_116 = None
    view_92: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_36, [8, 197, 2304]);  addmm_36 = None
    view_93: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_92, [8, 197, 3, 12, 64]);  view_92 = None
    permute_56: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_93, [2, 0, 3, 1, 4]);  view_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_56);  permute_56 = None
    getitem_146: "f32[8, 12, 197, 64]" = unbind_9[0]
    getitem_147: "f32[8, 12, 197, 64]" = unbind_9[1]
    getitem_148: "f32[8, 12, 197, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_9 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_146, getitem_147, getitem_148)
    getitem_149: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_9[0]
    getitem_150: "f32[8, 12, 197]" = _scaled_dot_product_flash_attention_9[1]
    getitem_151: "i32[]" = _scaled_dot_product_flash_attention_9[2]
    getitem_152: "i32[]" = _scaled_dot_product_flash_attention_9[3]
    getitem_155: "i64[]" = _scaled_dot_product_flash_attention_9[6]
    getitem_156: "i64[]" = _scaled_dot_product_flash_attention_9[7];  _scaled_dot_product_flash_attention_9 = None
    alias_9: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_149)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_57: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_149, [0, 2, 1, 3]);  getitem_149 = None
    view_94: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_57, [8, 197, 768]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_95: "f32[1576, 768]" = torch.ops.aten.view.default(view_94, [1576, 768]);  view_94 = None
    permute_58: "f32[768, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_37: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_118, view_95, permute_58);  primals_118 = None
    view_96: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_37, [8, 197, 768]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_28: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_96);  view_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_66: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_63, clone_28);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_158: "f32[8, 197, 1]" = var_mean_19[0]
    getitem_159: "f32[8, 197, 1]" = var_mean_19[1];  var_mean_19 = None
    add_67: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
    rsqrt_19: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_19: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_159)
    mul_65: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_66: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_119);  mul_65 = None
    add_68: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_120);  mul_66 = primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_97: "f32[1576, 768]" = torch.ops.aten.view.default(add_68, [1576, 768]);  add_68 = None
    permute_59: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_38: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_122, view_97, permute_59);  primals_122 = None
    view_98: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_38, [8, 197, 3072]);  addmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_68: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
    erf_9: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_69: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_69: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_67, add_69);  mul_67 = add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_29: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_69);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_99: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_29, [1576, 3072]);  clone_29 = None
    permute_60: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    addmm_39: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_124, view_99, permute_60);  primals_124 = None
    view_100: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_39, [8, 197, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_30: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_100);  view_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_70: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_66, clone_30);  clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_161: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_71: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_20: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_161)
    mul_70: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_71: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_70, primals_125);  mul_70 = None
    add_72: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_71, primals_126);  mul_71 = primals_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_101: "f32[1576, 768]" = torch.ops.aten.view.default(add_72, [1576, 768]);  add_72 = None
    permute_61: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    addmm_40: "f32[1576, 2304]" = torch.ops.aten.addmm.default(primals_128, view_101, permute_61);  primals_128 = None
    view_102: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_40, [8, 197, 2304]);  addmm_40 = None
    view_103: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_102, [8, 197, 3, 12, 64]);  view_102 = None
    permute_62: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_103, [2, 0, 3, 1, 4]);  view_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_62);  permute_62 = None
    getitem_162: "f32[8, 12, 197, 64]" = unbind_10[0]
    getitem_163: "f32[8, 12, 197, 64]" = unbind_10[1]
    getitem_164: "f32[8, 12, 197, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_10 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_162, getitem_163, getitem_164)
    getitem_165: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_10[0]
    getitem_166: "f32[8, 12, 197]" = _scaled_dot_product_flash_attention_10[1]
    getitem_167: "i32[]" = _scaled_dot_product_flash_attention_10[2]
    getitem_168: "i32[]" = _scaled_dot_product_flash_attention_10[3]
    getitem_171: "i64[]" = _scaled_dot_product_flash_attention_10[6]
    getitem_172: "i64[]" = _scaled_dot_product_flash_attention_10[7];  _scaled_dot_product_flash_attention_10 = None
    alias_10: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_165)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_63: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_165, [0, 2, 1, 3]);  getitem_165 = None
    view_104: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_63, [8, 197, 768]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_105: "f32[1576, 768]" = torch.ops.aten.view.default(view_104, [1576, 768]);  view_104 = None
    permute_64: "f32[768, 768]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_41: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_130, view_105, permute_64);  primals_130 = None
    view_106: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_41, [8, 197, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_31: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_106);  view_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_73: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_70, clone_31);  clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_174: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_175: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_74: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_21: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_175)
    mul_72: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_73: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_131);  mul_72 = None
    add_75: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_132);  mul_73 = primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_107: "f32[1576, 768]" = torch.ops.aten.view.default(add_75, [1576, 768]);  add_75 = None
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    addmm_42: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_134, view_107, permute_65);  primals_134 = None
    view_108: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_42, [8, 197, 3072]);  addmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_74: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_108, 0.5)
    mul_75: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476)
    erf_10: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_76: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_76: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_74, add_76);  mul_74 = add_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_32: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_109: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_32, [1576, 3072]);  clone_32 = None
    permute_66: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_43: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_136, view_109, permute_66);  primals_136 = None
    view_110: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_43, [8, 197, 768]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_33: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_110);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_77: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_73, clone_33);  clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_176: "f32[8, 197, 1]" = var_mean_22[0]
    getitem_177: "f32[8, 197, 1]" = var_mean_22[1];  var_mean_22 = None
    add_78: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
    rsqrt_22: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_22: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_177)
    mul_77: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_78: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_137);  mul_77 = None
    add_79: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_78, primals_138);  mul_78 = primals_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_111: "f32[1576, 768]" = torch.ops.aten.view.default(add_79, [1576, 768]);  add_79 = None
    permute_67: "f32[768, 2304]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    addmm_44: "f32[1576, 2304]" = torch.ops.aten.addmm.default(primals_140, view_111, permute_67);  primals_140 = None
    view_112: "f32[8, 197, 2304]" = torch.ops.aten.view.default(addmm_44, [8, 197, 2304]);  addmm_44 = None
    view_113: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.view.default(view_112, [8, 197, 3, 12, 64]);  view_112 = None
    permute_68: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.permute.default(view_113, [2, 0, 3, 1, 4]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_68);  permute_68 = None
    getitem_178: "f32[8, 12, 197, 64]" = unbind_11[0]
    getitem_179: "f32[8, 12, 197, 64]" = unbind_11[1]
    getitem_180: "f32[8, 12, 197, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_flash_attention_11 = torch.ops.aten._scaled_dot_product_flash_attention.default(getitem_178, getitem_179, getitem_180)
    getitem_181: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_11[0]
    getitem_182: "f32[8, 12, 197]" = _scaled_dot_product_flash_attention_11[1]
    getitem_183: "i32[]" = _scaled_dot_product_flash_attention_11[2]
    getitem_184: "i32[]" = _scaled_dot_product_flash_attention_11[3]
    getitem_187: "i64[]" = _scaled_dot_product_flash_attention_11[6]
    getitem_188: "i64[]" = _scaled_dot_product_flash_attention_11[7];  _scaled_dot_product_flash_attention_11 = None
    alias_11: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(getitem_181)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_69: "f32[8, 197, 12, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None
    view_114: "f32[8, 197, 768]" = torch.ops.aten.view.default(permute_69, [8, 197, 768]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_115: "f32[1576, 768]" = torch.ops.aten.view.default(view_114, [1576, 768]);  view_114 = None
    permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_45: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_142, view_115, permute_70);  primals_142 = None
    view_116: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_45, [8, 197, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_34: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_116);  view_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_80: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_77, clone_34);  clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_190: "f32[8, 197, 1]" = var_mean_23[0]
    getitem_191: "f32[8, 197, 1]" = var_mean_23[1];  var_mean_23 = None
    add_81: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-06);  getitem_190 = None
    rsqrt_23: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_23: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_191)
    mul_79: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_80: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_79, primals_143);  mul_79 = None
    add_82: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_80, primals_144);  mul_80 = primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[1576, 768]" = torch.ops.aten.view.default(add_82, [1576, 768]);  add_82 = None
    permute_71: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_46: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_146, view_117, permute_71);  primals_146 = None
    view_118: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 197, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_81: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_82: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
    erf_11: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
    add_83: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_83: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_81, add_83);  mul_81 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_35: "f32[8, 197, 3072]" = torch.ops.aten.clone.default(mul_83);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_119: "f32[1576, 3072]" = torch.ops.aten.view.default(clone_35, [1576, 3072]);  clone_35 = None
    permute_72: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm_47: "f32[1576, 768]" = torch.ops.aten.addmm.default(primals_148, view_119, permute_72);  primals_148 = None
    view_120: "f32[8, 197, 768]" = torch.ops.aten.view.default(addmm_47, [8, 197, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_36: "f32[8, 197, 768]" = torch.ops.aten.clone.default(view_120);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_84: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_80, clone_36);  clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:641, code: x = self.norm(x)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_192: "f32[8, 197, 1]" = var_mean_24[0]
    getitem_193: "f32[8, 197, 1]" = var_mean_24[1];  var_mean_24 = None
    add_85: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
    rsqrt_24: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_24: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_193)
    mul_84: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_85: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_84, primals_149);  mul_84 = None
    add_86: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_85, primals_150);  mul_85 = primals_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:646, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    slice_1: "f32[8, 197, 768]" = torch.ops.aten.slice.Tensor(add_86, 0, 0, 9223372036854775807);  add_86 = None
    select: "f32[8, 768]" = torch.ops.aten.select.int(slice_1, 1, 0);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:648, code: x = self.head_drop(x)
    clone_37: "f32[8, 768]" = torch.ops.aten.clone.default(select);  select = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:649, code: return x if pre_logits else self.head(x)
    permute_73: "f32[768, 1000]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_48: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_152, clone_37, permute_73);  primals_152 = None
    permute_74: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    mm: "f32[8, 768]" = torch.ops.aten.mm.default(tangents_1, permute_74);  permute_74 = None
    permute_75: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 768]" = torch.ops.aten.mm.default(permute_75, clone_37);  permute_75 = clone_37 = None
    permute_76: "f32[768, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_1: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_121: "f32[1000]" = torch.ops.aten.view.default(sum_1, [1000]);  sum_1 = None
    permute_77: "f32[1000, 768]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:646, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    full: "f32[8, 197, 768]" = torch.ops.aten.full.default([8, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter: "f32[8, 197, 768]" = torch.ops.aten.select_scatter.default(full, mm, 1, 0);  full = mm = None
    full_1: "f32[8, 197, 768]" = torch.ops.aten.full.default([8, 197, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter: "f32[8, 197, 768]" = torch.ops.aten.slice_scatter.default(full_1, select_scatter, 0, 0, 9223372036854775807);  full_1 = select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:641, code: x = self.norm(x)
    sub_25: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_193);  add_84 = getitem_193 = None
    mul_86: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_24);  sub_25 = None
    mul_87: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter, primals_149);  primals_149 = None
    mul_88: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_87, 768)
    sum_2: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_87, [2], True)
    mul_89: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_87, mul_86);  mul_87 = None
    sum_3: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_89, [2], True);  mul_89 = None
    mul_90: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_86, sum_3);  sum_3 = None
    sub_26: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_88, sum_2);  mul_88 = sum_2 = None
    sub_27: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_26, mul_90);  sub_26 = mul_90 = None
    div: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    mul_91: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div, sub_27);  div = sub_27 = None
    mul_92: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(slice_scatter, mul_86);  mul_86 = None
    sum_4: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_92, [0, 1]);  mul_92 = None
    sum_5: "f32[768]" = torch.ops.aten.sum.dim_IntList(slice_scatter, [0, 1]);  slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_122: "f32[1576, 768]" = torch.ops.aten.view.default(mul_91, [1576, 768])
    permute_78: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    mm_2: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_122, permute_78);  permute_78 = None
    permute_79: "f32[768, 1576]" = torch.ops.aten.permute.default(view_122, [1, 0])
    mm_3: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_79, view_119);  permute_79 = view_119 = None
    permute_80: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_6: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_122, [0], True);  view_122 = None
    view_123: "f32[768]" = torch.ops.aten.view.default(sum_6, [768]);  sum_6 = None
    permute_81: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
    view_124: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_2, [8, 197, 3072]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_93: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476)
    erf_12: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_93);  mul_93 = None
    add_87: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_94: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_87, 0.5);  add_87 = None
    mul_95: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, view_118)
    mul_96: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_95, -0.5);  mul_95 = None
    exp: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_96);  mul_96 = None
    mul_97: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp, 0.3989422804014327);  exp = None
    mul_98: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, mul_97);  view_118 = mul_97 = None
    add_88: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_94, mul_98);  mul_94 = mul_98 = None
    mul_99: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_124, add_88);  view_124 = add_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_125: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_99, [1576, 3072]);  mul_99 = None
    permute_82: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    mm_4: "f32[1576, 768]" = torch.ops.aten.mm.default(view_125, permute_82);  permute_82 = None
    permute_83: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_125, [1, 0])
    mm_5: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_83, view_117);  permute_83 = view_117 = None
    permute_84: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_7: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_125, [0], True);  view_125 = None
    view_126: "f32[3072]" = torch.ops.aten.view.default(sum_7, [3072]);  sum_7 = None
    permute_85: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    view_127: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_4, [8, 197, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_28: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_191);  add_80 = getitem_191 = None
    mul_100: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_23);  sub_28 = None
    mul_101: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, primals_143);  primals_143 = None
    mul_102: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_101, 768)
    sum_8: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_101, [2], True)
    mul_103: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_101, mul_100);  mul_101 = None
    sum_9: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_103, [2], True);  mul_103 = None
    mul_104: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_100, sum_9);  sum_9 = None
    sub_29: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_102, sum_8);  mul_102 = sum_8 = None
    sub_30: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_29, mul_104);  sub_29 = mul_104 = None
    div_1: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    mul_105: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_1, sub_30);  div_1 = sub_30 = None
    mul_106: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_127, mul_100);  mul_100 = None
    sum_10: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_106, [0, 1]);  mul_106 = None
    sum_11: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_127, [0, 1]);  view_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_89: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_91, mul_105);  mul_91 = mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_128: "f32[1576, 768]" = torch.ops.aten.view.default(add_89, [1576, 768])
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_6: "f32[1576, 768]" = torch.ops.aten.mm.default(view_128, permute_86);  permute_86 = None
    permute_87: "f32[768, 1576]" = torch.ops.aten.permute.default(view_128, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_87, view_115);  permute_87 = view_115 = None
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_12: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_128, [0], True);  view_128 = None
    view_129: "f32[768]" = torch.ops.aten.view.default(sum_12, [768]);  sum_12 = None
    permute_89: "f32[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    view_130: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_6, [8, 197, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_131: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_130, [8, 197, 12, 64]);  view_130 = None
    permute_90: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_12: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    _scaled_dot_product_flash_attention_backward = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_90, getitem_178, getitem_179, getitem_180, alias_12, getitem_182, getitem_183, getitem_184, 0, 0, 0.0, False, getitem_187, getitem_188);  permute_90 = getitem_178 = getitem_179 = getitem_180 = alias_12 = getitem_182 = getitem_183 = getitem_184 = getitem_187 = getitem_188 = None
    getitem_194: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward[0]
    getitem_195: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward[1]
    getitem_196: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward[2];  _scaled_dot_product_flash_attention_backward = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_1: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_194, getitem_195, getitem_196]);  getitem_194 = getitem_195 = getitem_196 = None
    view_132: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_1, [3, 8, 12, 197, 64]);  cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_91: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_132, [1, 3, 0, 2, 4]);  view_132 = None
    clone_38: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_91, memory_format = torch.contiguous_format);  permute_91 = None
    view_133: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_38, [8, 197, 2304]);  clone_38 = None
    view_134: "f32[1576, 2304]" = torch.ops.aten.view.default(view_133, [1576, 2304]);  view_133 = None
    permute_92: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    mm_8: "f32[1576, 768]" = torch.ops.aten.mm.default(view_134, permute_92);  permute_92 = None
    permute_93: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_134, [1, 0])
    mm_9: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_93, view_111);  permute_93 = view_111 = None
    permute_94: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_13: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_134, [0], True);  view_134 = None
    view_135: "f32[2304]" = torch.ops.aten.view.default(sum_13, [2304]);  sum_13 = None
    permute_95: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    view_136: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_8, [8, 197, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_31: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_177);  add_77 = getitem_177 = None
    mul_107: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_22);  sub_31 = None
    mul_108: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_136, primals_137);  primals_137 = None
    mul_109: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_108, 768)
    sum_14: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_108, [2], True)
    mul_110: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_108, mul_107);  mul_108 = None
    sum_15: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_110, [2], True);  mul_110 = None
    mul_111: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_107, sum_15);  sum_15 = None
    sub_32: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_109, sum_14);  mul_109 = sum_14 = None
    sub_33: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_32, mul_111);  sub_32 = mul_111 = None
    div_2: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    mul_112: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_2, sub_33);  div_2 = sub_33 = None
    mul_113: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_136, mul_107);  mul_107 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_113, [0, 1]);  mul_113 = None
    sum_17: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_136, [0, 1]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_90: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_89, mul_112);  add_89 = mul_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_137: "f32[1576, 768]" = torch.ops.aten.view.default(add_90, [1576, 768])
    permute_96: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_10: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_137, permute_96);  permute_96 = None
    permute_97: "f32[768, 1576]" = torch.ops.aten.permute.default(view_137, [1, 0])
    mm_11: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_97, view_109);  permute_97 = view_109 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_18: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_137, [0], True);  view_137 = None
    view_138: "f32[768]" = torch.ops.aten.view.default(sum_18, [768]);  sum_18 = None
    permute_99: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    view_139: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_10, [8, 197, 3072]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_114: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476)
    erf_13: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_114);  mul_114 = None
    add_91: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_115: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_91, 0.5);  add_91 = None
    mul_116: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_108, view_108)
    mul_117: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_116, -0.5);  mul_116 = None
    exp_1: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_117);  mul_117 = None
    mul_118: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_1, 0.3989422804014327);  exp_1 = None
    mul_119: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_108, mul_118);  view_108 = mul_118 = None
    add_92: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_115, mul_119);  mul_115 = mul_119 = None
    mul_120: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_139, add_92);  view_139 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_140: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_120, [1576, 3072]);  mul_120 = None
    permute_100: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_12: "f32[1576, 768]" = torch.ops.aten.mm.default(view_140, permute_100);  permute_100 = None
    permute_101: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_140, [1, 0])
    mm_13: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_101, view_107);  permute_101 = view_107 = None
    permute_102: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_19: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_140, [0], True);  view_140 = None
    view_141: "f32[3072]" = torch.ops.aten.view.default(sum_19, [3072]);  sum_19 = None
    permute_103: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
    view_142: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_12, [8, 197, 768]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_34: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_175);  add_73 = getitem_175 = None
    mul_121: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_21);  sub_34 = None
    mul_122: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_142, primals_131);  primals_131 = None
    mul_123: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_122, 768)
    sum_20: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_122, [2], True)
    mul_124: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_122, mul_121);  mul_122 = None
    sum_21: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_124, [2], True);  mul_124 = None
    mul_125: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_121, sum_21);  sum_21 = None
    sub_35: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_123, sum_20);  mul_123 = sum_20 = None
    sub_36: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_35, mul_125);  sub_35 = mul_125 = None
    div_3: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    mul_126: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_3, sub_36);  div_3 = sub_36 = None
    mul_127: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_142, mul_121);  mul_121 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_127, [0, 1]);  mul_127 = None
    sum_23: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_142, [0, 1]);  view_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_93: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_90, mul_126);  add_90 = mul_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_143: "f32[1576, 768]" = torch.ops.aten.view.default(add_93, [1576, 768])
    permute_104: "f32[768, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_14: "f32[1576, 768]" = torch.ops.aten.mm.default(view_143, permute_104);  permute_104 = None
    permute_105: "f32[768, 1576]" = torch.ops.aten.permute.default(view_143, [1, 0])
    mm_15: "f32[768, 768]" = torch.ops.aten.mm.default(permute_105, view_105);  permute_105 = view_105 = None
    permute_106: "f32[768, 768]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_24: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_143, [0], True);  view_143 = None
    view_144: "f32[768]" = torch.ops.aten.view.default(sum_24, [768]);  sum_24 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    view_145: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_14, [8, 197, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_146: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_145, [8, 197, 12, 64]);  view_145 = None
    permute_108: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_13: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    _scaled_dot_product_flash_attention_backward_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_108, getitem_162, getitem_163, getitem_164, alias_13, getitem_166, getitem_167, getitem_168, 0, 0, 0.0, False, getitem_171, getitem_172);  permute_108 = getitem_162 = getitem_163 = getitem_164 = alias_13 = getitem_166 = getitem_167 = getitem_168 = getitem_171 = getitem_172 = None
    getitem_197: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_1[0]
    getitem_198: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_1[1]
    getitem_199: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_1[2];  _scaled_dot_product_flash_attention_backward_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_2: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_197, getitem_198, getitem_199]);  getitem_197 = getitem_198 = getitem_199 = None
    view_147: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_2, [3, 8, 12, 197, 64]);  cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_109: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_147, [1, 3, 0, 2, 4]);  view_147 = None
    clone_39: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
    view_148: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_39, [8, 197, 2304]);  clone_39 = None
    view_149: "f32[1576, 2304]" = torch.ops.aten.view.default(view_148, [1576, 2304]);  view_148 = None
    permute_110: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    mm_16: "f32[1576, 768]" = torch.ops.aten.mm.default(view_149, permute_110);  permute_110 = None
    permute_111: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_149, [1, 0])
    mm_17: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_111, view_101);  permute_111 = view_101 = None
    permute_112: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_25: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_149, [0], True);  view_149 = None
    view_150: "f32[2304]" = torch.ops.aten.view.default(sum_25, [2304]);  sum_25 = None
    permute_113: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    view_151: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_16, [8, 197, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_37: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_161);  add_70 = getitem_161 = None
    mul_128: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_20);  sub_37 = None
    mul_129: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_151, primals_125);  primals_125 = None
    mul_130: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_129, 768)
    sum_26: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_129, [2], True)
    mul_131: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_129, mul_128);  mul_129 = None
    sum_27: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_131, [2], True);  mul_131 = None
    mul_132: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_128, sum_27);  sum_27 = None
    sub_38: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_130, sum_26);  mul_130 = sum_26 = None
    sub_39: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_38, mul_132);  sub_38 = mul_132 = None
    div_4: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    mul_133: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_4, sub_39);  div_4 = sub_39 = None
    mul_134: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_151, mul_128);  mul_128 = None
    sum_28: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_134, [0, 1]);  mul_134 = None
    sum_29: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_151, [0, 1]);  view_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_94: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_93, mul_133);  add_93 = mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_152: "f32[1576, 768]" = torch.ops.aten.view.default(add_94, [1576, 768])
    permute_114: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    mm_18: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_152, permute_114);  permute_114 = None
    permute_115: "f32[768, 1576]" = torch.ops.aten.permute.default(view_152, [1, 0])
    mm_19: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_115, view_99);  permute_115 = view_99 = None
    permute_116: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_30: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_152, [0], True);  view_152 = None
    view_153: "f32[768]" = torch.ops.aten.view.default(sum_30, [768]);  sum_30 = None
    permute_117: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    view_154: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_18, [8, 197, 3072]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_135: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476)
    erf_14: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_135);  mul_135 = None
    add_95: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_136: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_95, 0.5);  add_95 = None
    mul_137: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_98, view_98)
    mul_138: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_137, -0.5);  mul_137 = None
    exp_2: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_138);  mul_138 = None
    mul_139: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_2, 0.3989422804014327);  exp_2 = None
    mul_140: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_98, mul_139);  view_98 = mul_139 = None
    add_96: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_136, mul_140);  mul_136 = mul_140 = None
    mul_141: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_154, add_96);  view_154 = add_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_155: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_141, [1576, 3072]);  mul_141 = None
    permute_118: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_20: "f32[1576, 768]" = torch.ops.aten.mm.default(view_155, permute_118);  permute_118 = None
    permute_119: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_155, [1, 0])
    mm_21: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_119, view_97);  permute_119 = view_97 = None
    permute_120: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_31: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_155, [0], True);  view_155 = None
    view_156: "f32[3072]" = torch.ops.aten.view.default(sum_31, [3072]);  sum_31 = None
    permute_121: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    view_157: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_20, [8, 197, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_40: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_159);  add_66 = getitem_159 = None
    mul_142: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_19);  sub_40 = None
    mul_143: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_157, primals_119);  primals_119 = None
    mul_144: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_143, 768)
    sum_32: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_143, [2], True)
    mul_145: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_143, mul_142);  mul_143 = None
    sum_33: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_145, [2], True);  mul_145 = None
    mul_146: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_142, sum_33);  sum_33 = None
    sub_41: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_144, sum_32);  mul_144 = sum_32 = None
    sub_42: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_41, mul_146);  sub_41 = mul_146 = None
    div_5: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    mul_147: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_5, sub_42);  div_5 = sub_42 = None
    mul_148: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_157, mul_142);  mul_142 = None
    sum_34: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_148, [0, 1]);  mul_148 = None
    sum_35: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_157, [0, 1]);  view_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_97: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_94, mul_147);  add_94 = mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_158: "f32[1576, 768]" = torch.ops.aten.view.default(add_97, [1576, 768])
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    mm_22: "f32[1576, 768]" = torch.ops.aten.mm.default(view_158, permute_122);  permute_122 = None
    permute_123: "f32[768, 1576]" = torch.ops.aten.permute.default(view_158, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_123, view_95);  permute_123 = view_95 = None
    permute_124: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_158, [0], True);  view_158 = None
    view_159: "f32[768]" = torch.ops.aten.view.default(sum_36, [768]);  sum_36 = None
    permute_125: "f32[768, 768]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    view_160: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_22, [8, 197, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_161: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_160, [8, 197, 12, 64]);  view_160 = None
    permute_126: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_161, [0, 2, 1, 3]);  view_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_14: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    _scaled_dot_product_flash_attention_backward_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_126, getitem_146, getitem_147, getitem_148, alias_14, getitem_150, getitem_151, getitem_152, 0, 0, 0.0, False, getitem_155, getitem_156);  permute_126 = getitem_146 = getitem_147 = getitem_148 = alias_14 = getitem_150 = getitem_151 = getitem_152 = getitem_155 = getitem_156 = None
    getitem_200: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_2[0]
    getitem_201: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_2[1]
    getitem_202: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_2[2];  _scaled_dot_product_flash_attention_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_3: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_200, getitem_201, getitem_202]);  getitem_200 = getitem_201 = getitem_202 = None
    view_162: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_3, [3, 8, 12, 197, 64]);  cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_127: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_162, [1, 3, 0, 2, 4]);  view_162 = None
    clone_40: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_127, memory_format = torch.contiguous_format);  permute_127 = None
    view_163: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_40, [8, 197, 2304]);  clone_40 = None
    view_164: "f32[1576, 2304]" = torch.ops.aten.view.default(view_163, [1576, 2304]);  view_163 = None
    permute_128: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_24: "f32[1576, 768]" = torch.ops.aten.mm.default(view_164, permute_128);  permute_128 = None
    permute_129: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_164, [1, 0])
    mm_25: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_129, view_91);  permute_129 = view_91 = None
    permute_130: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_37: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_164, [0], True);  view_164 = None
    view_165: "f32[2304]" = torch.ops.aten.view.default(sum_37, [2304]);  sum_37 = None
    permute_131: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    view_166: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_24, [8, 197, 768]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_43: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_145);  add_63 = getitem_145 = None
    mul_149: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_18);  sub_43 = None
    mul_150: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_166, primals_113);  primals_113 = None
    mul_151: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_150, 768)
    sum_38: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_150, [2], True)
    mul_152: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_150, mul_149);  mul_150 = None
    sum_39: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [2], True);  mul_152 = None
    mul_153: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_149, sum_39);  sum_39 = None
    sub_44: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_151, sum_38);  mul_151 = sum_38 = None
    sub_45: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_44, mul_153);  sub_44 = mul_153 = None
    div_6: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    mul_154: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_6, sub_45);  div_6 = sub_45 = None
    mul_155: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_166, mul_149);  mul_149 = None
    sum_40: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_155, [0, 1]);  mul_155 = None
    sum_41: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_166, [0, 1]);  view_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_98: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_97, mul_154);  add_97 = mul_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_167: "f32[1576, 768]" = torch.ops.aten.view.default(add_98, [1576, 768])
    permute_132: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_26: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_167, permute_132);  permute_132 = None
    permute_133: "f32[768, 1576]" = torch.ops.aten.permute.default(view_167, [1, 0])
    mm_27: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_133, view_89);  permute_133 = view_89 = None
    permute_134: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_167, [0], True);  view_167 = None
    view_168: "f32[768]" = torch.ops.aten.view.default(sum_42, [768]);  sum_42 = None
    permute_135: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    view_169: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_26, [8, 197, 3072]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_156: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476)
    erf_15: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_156);  mul_156 = None
    add_99: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_157: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_99, 0.5);  add_99 = None
    mul_158: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, view_88)
    mul_159: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_158, -0.5);  mul_158 = None
    exp_3: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_159);  mul_159 = None
    mul_160: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_3, 0.3989422804014327);  exp_3 = None
    mul_161: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, mul_160);  view_88 = mul_160 = None
    add_100: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_157, mul_161);  mul_157 = mul_161 = None
    mul_162: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_169, add_100);  view_169 = add_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_170: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_162, [1576, 3072]);  mul_162 = None
    permute_136: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_28: "f32[1576, 768]" = torch.ops.aten.mm.default(view_170, permute_136);  permute_136 = None
    permute_137: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_170, [1, 0])
    mm_29: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_137, view_87);  permute_137 = view_87 = None
    permute_138: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_43: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_170, [0], True);  view_170 = None
    view_171: "f32[3072]" = torch.ops.aten.view.default(sum_43, [3072]);  sum_43 = None
    permute_139: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    view_172: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_28, [8, 197, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_46: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_143);  add_59 = getitem_143 = None
    mul_163: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_17);  sub_46 = None
    mul_164: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_172, primals_107);  primals_107 = None
    mul_165: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_164, 768)
    sum_44: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_164, [2], True)
    mul_166: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_164, mul_163);  mul_164 = None
    sum_45: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_166, [2], True);  mul_166 = None
    mul_167: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_163, sum_45);  sum_45 = None
    sub_47: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_165, sum_44);  mul_165 = sum_44 = None
    sub_48: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_47, mul_167);  sub_47 = mul_167 = None
    div_7: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    mul_168: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_7, sub_48);  div_7 = sub_48 = None
    mul_169: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_172, mul_163);  mul_163 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_169, [0, 1]);  mul_169 = None
    sum_47: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_172, [0, 1]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_101: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_98, mul_168);  add_98 = mul_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_173: "f32[1576, 768]" = torch.ops.aten.view.default(add_101, [1576, 768])
    permute_140: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_30: "f32[1576, 768]" = torch.ops.aten.mm.default(view_173, permute_140);  permute_140 = None
    permute_141: "f32[768, 1576]" = torch.ops.aten.permute.default(view_173, [1, 0])
    mm_31: "f32[768, 768]" = torch.ops.aten.mm.default(permute_141, view_85);  permute_141 = view_85 = None
    permute_142: "f32[768, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_48: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_173, [0], True);  view_173 = None
    view_174: "f32[768]" = torch.ops.aten.view.default(sum_48, [768]);  sum_48 = None
    permute_143: "f32[768, 768]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    view_175: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_30, [8, 197, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_176: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_175, [8, 197, 12, 64]);  view_175 = None
    permute_144: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_15: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    _scaled_dot_product_flash_attention_backward_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_144, getitem_130, getitem_131, getitem_132, alias_15, getitem_134, getitem_135, getitem_136, 0, 0, 0.0, False, getitem_139, getitem_140);  permute_144 = getitem_130 = getitem_131 = getitem_132 = alias_15 = getitem_134 = getitem_135 = getitem_136 = getitem_139 = getitem_140 = None
    getitem_203: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_3[0]
    getitem_204: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_3[1]
    getitem_205: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_3[2];  _scaled_dot_product_flash_attention_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_4: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_203, getitem_204, getitem_205]);  getitem_203 = getitem_204 = getitem_205 = None
    view_177: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_4, [3, 8, 12, 197, 64]);  cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_145: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_177, [1, 3, 0, 2, 4]);  view_177 = None
    clone_41: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_145, memory_format = torch.contiguous_format);  permute_145 = None
    view_178: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_41, [8, 197, 2304]);  clone_41 = None
    view_179: "f32[1576, 2304]" = torch.ops.aten.view.default(view_178, [1576, 2304]);  view_178 = None
    permute_146: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    mm_32: "f32[1576, 768]" = torch.ops.aten.mm.default(view_179, permute_146);  permute_146 = None
    permute_147: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_179, [1, 0])
    mm_33: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_147, view_81);  permute_147 = view_81 = None
    permute_148: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_49: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_179, [0], True);  view_179 = None
    view_180: "f32[2304]" = torch.ops.aten.view.default(sum_49, [2304]);  sum_49 = None
    permute_149: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    view_181: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_32, [8, 197, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_49: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_129);  add_56 = getitem_129 = None
    mul_170: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_16);  sub_49 = None
    mul_171: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_181, primals_101);  primals_101 = None
    mul_172: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_171, 768)
    sum_50: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True)
    mul_173: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_171, mul_170);  mul_171 = None
    sum_51: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
    mul_174: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_170, sum_51);  sum_51 = None
    sub_50: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_172, sum_50);  mul_172 = sum_50 = None
    sub_51: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_50, mul_174);  sub_50 = mul_174 = None
    div_8: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    mul_175: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_8, sub_51);  div_8 = sub_51 = None
    mul_176: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_181, mul_170);  mul_170 = None
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 1]);  mul_176 = None
    sum_53: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_181, [0, 1]);  view_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_102: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_101, mul_175);  add_101 = mul_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_182: "f32[1576, 768]" = torch.ops.aten.view.default(add_102, [1576, 768])
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_34: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_182, permute_150);  permute_150 = None
    permute_151: "f32[768, 1576]" = torch.ops.aten.permute.default(view_182, [1, 0])
    mm_35: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_151, view_79);  permute_151 = view_79 = None
    permute_152: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_54: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_182, [0], True);  view_182 = None
    view_183: "f32[768]" = torch.ops.aten.view.default(sum_54, [768]);  sum_54 = None
    permute_153: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
    view_184: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_34, [8, 197, 3072]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_177: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476)
    erf_16: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_177);  mul_177 = None
    add_103: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_178: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_103, 0.5);  add_103 = None
    mul_179: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_78, view_78)
    mul_180: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_179, -0.5);  mul_179 = None
    exp_4: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_180);  mul_180 = None
    mul_181: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_4, 0.3989422804014327);  exp_4 = None
    mul_182: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_78, mul_181);  view_78 = mul_181 = None
    add_104: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_178, mul_182);  mul_178 = mul_182 = None
    mul_183: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_184, add_104);  view_184 = add_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_185: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_183, [1576, 3072]);  mul_183 = None
    permute_154: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    mm_36: "f32[1576, 768]" = torch.ops.aten.mm.default(view_185, permute_154);  permute_154 = None
    permute_155: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_185, [1, 0])
    mm_37: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_155, view_77);  permute_155 = view_77 = None
    permute_156: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_55: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_185, [0], True);  view_185 = None
    view_186: "f32[3072]" = torch.ops.aten.view.default(sum_55, [3072]);  sum_55 = None
    permute_157: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
    view_187: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_36, [8, 197, 768]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_52: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_127);  add_52 = getitem_127 = None
    mul_184: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_15);  sub_52 = None
    mul_185: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_187, primals_95);  primals_95 = None
    mul_186: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_185, 768)
    sum_56: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_185, [2], True)
    mul_187: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_185, mul_184);  mul_185 = None
    sum_57: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_187, [2], True);  mul_187 = None
    mul_188: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_184, sum_57);  sum_57 = None
    sub_53: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_186, sum_56);  mul_186 = sum_56 = None
    sub_54: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_53, mul_188);  sub_53 = mul_188 = None
    div_9: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    mul_189: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_9, sub_54);  div_9 = sub_54 = None
    mul_190: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_187, mul_184);  mul_184 = None
    sum_58: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_190, [0, 1]);  mul_190 = None
    sum_59: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_187, [0, 1]);  view_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_105: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_102, mul_189);  add_102 = mul_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_188: "f32[1576, 768]" = torch.ops.aten.view.default(add_105, [1576, 768])
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_38: "f32[1576, 768]" = torch.ops.aten.mm.default(view_188, permute_158);  permute_158 = None
    permute_159: "f32[768, 1576]" = torch.ops.aten.permute.default(view_188, [1, 0])
    mm_39: "f32[768, 768]" = torch.ops.aten.mm.default(permute_159, view_75);  permute_159 = view_75 = None
    permute_160: "f32[768, 768]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_60: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_188, [0], True);  view_188 = None
    view_189: "f32[768]" = torch.ops.aten.view.default(sum_60, [768]);  sum_60 = None
    permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    view_190: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_38, [8, 197, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_191: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_190, [8, 197, 12, 64]);  view_190 = None
    permute_162: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_16: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    _scaled_dot_product_flash_attention_backward_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_162, getitem_114, getitem_115, getitem_116, alias_16, getitem_118, getitem_119, getitem_120, 0, 0, 0.0, False, getitem_123, getitem_124);  permute_162 = getitem_114 = getitem_115 = getitem_116 = alias_16 = getitem_118 = getitem_119 = getitem_120 = getitem_123 = getitem_124 = None
    getitem_206: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_4[0]
    getitem_207: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_4[1]
    getitem_208: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_4[2];  _scaled_dot_product_flash_attention_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_5: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_206, getitem_207, getitem_208]);  getitem_206 = getitem_207 = getitem_208 = None
    view_192: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_5, [3, 8, 12, 197, 64]);  cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_163: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_192, [1, 3, 0, 2, 4]);  view_192 = None
    clone_42: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_163, memory_format = torch.contiguous_format);  permute_163 = None
    view_193: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_42, [8, 197, 2304]);  clone_42 = None
    view_194: "f32[1576, 2304]" = torch.ops.aten.view.default(view_193, [1576, 2304]);  view_193 = None
    permute_164: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_40: "f32[1576, 768]" = torch.ops.aten.mm.default(view_194, permute_164);  permute_164 = None
    permute_165: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_194, [1, 0])
    mm_41: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_165, view_71);  permute_165 = view_71 = None
    permute_166: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_61: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_194, [0], True);  view_194 = None
    view_195: "f32[2304]" = torch.ops.aten.view.default(sum_61, [2304]);  sum_61 = None
    permute_167: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_166, [1, 0]);  permute_166 = None
    view_196: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_40, [8, 197, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_55: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_113);  add_49 = getitem_113 = None
    mul_191: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_14);  sub_55 = None
    mul_192: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_196, primals_89);  primals_89 = None
    mul_193: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_192, 768)
    sum_62: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_192, [2], True)
    mul_194: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_192, mul_191);  mul_192 = None
    sum_63: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True);  mul_194 = None
    mul_195: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_191, sum_63);  sum_63 = None
    sub_56: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_193, sum_62);  mul_193 = sum_62 = None
    sub_57: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_56, mul_195);  sub_56 = mul_195 = None
    div_10: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    mul_196: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_10, sub_57);  div_10 = sub_57 = None
    mul_197: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_196, mul_191);  mul_191 = None
    sum_64: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_197, [0, 1]);  mul_197 = None
    sum_65: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_196, [0, 1]);  view_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_106: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_105, mul_196);  add_105 = mul_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_197: "f32[1576, 768]" = torch.ops.aten.view.default(add_106, [1576, 768])
    permute_168: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_42: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_197, permute_168);  permute_168 = None
    permute_169: "f32[768, 1576]" = torch.ops.aten.permute.default(view_197, [1, 0])
    mm_43: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_169, view_69);  permute_169 = view_69 = None
    permute_170: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
    view_198: "f32[768]" = torch.ops.aten.view.default(sum_66, [768]);  sum_66 = None
    permute_171: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    view_199: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_42, [8, 197, 3072]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_198: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476)
    erf_17: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_198);  mul_198 = None
    add_107: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_199: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_107, 0.5);  add_107 = None
    mul_200: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_68, view_68)
    mul_201: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_200, -0.5);  mul_200 = None
    exp_5: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_201);  mul_201 = None
    mul_202: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_5, 0.3989422804014327);  exp_5 = None
    mul_203: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_68, mul_202);  view_68 = mul_202 = None
    add_108: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_199, mul_203);  mul_199 = mul_203 = None
    mul_204: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_199, add_108);  view_199 = add_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_200: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_204, [1576, 3072]);  mul_204 = None
    permute_172: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_44: "f32[1576, 768]" = torch.ops.aten.mm.default(view_200, permute_172);  permute_172 = None
    permute_173: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_200, [1, 0])
    mm_45: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_173, view_67);  permute_173 = view_67 = None
    permute_174: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_67: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_200, [0], True);  view_200 = None
    view_201: "f32[3072]" = torch.ops.aten.view.default(sum_67, [3072]);  sum_67 = None
    permute_175: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    view_202: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_44, [8, 197, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_58: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_111);  add_45 = getitem_111 = None
    mul_205: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_13);  sub_58 = None
    mul_206: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_202, primals_83);  primals_83 = None
    mul_207: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_206, 768)
    sum_68: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_206, [2], True)
    mul_208: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_206, mul_205);  mul_206 = None
    sum_69: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_208, [2], True);  mul_208 = None
    mul_209: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_205, sum_69);  sum_69 = None
    sub_59: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_207, sum_68);  mul_207 = sum_68 = None
    sub_60: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_59, mul_209);  sub_59 = mul_209 = None
    div_11: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    mul_210: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_11, sub_60);  div_11 = sub_60 = None
    mul_211: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_202, mul_205);  mul_205 = None
    sum_70: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_211, [0, 1]);  mul_211 = None
    sum_71: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_202, [0, 1]);  view_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_109: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_106, mul_210);  add_106 = mul_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_203: "f32[1576, 768]" = torch.ops.aten.view.default(add_109, [1576, 768])
    permute_176: "f32[768, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    mm_46: "f32[1576, 768]" = torch.ops.aten.mm.default(view_203, permute_176);  permute_176 = None
    permute_177: "f32[768, 1576]" = torch.ops.aten.permute.default(view_203, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_177, view_65);  permute_177 = view_65 = None
    permute_178: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_203, [0], True);  view_203 = None
    view_204: "f32[768]" = torch.ops.aten.view.default(sum_72, [768]);  sum_72 = None
    permute_179: "f32[768, 768]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    view_205: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_46, [8, 197, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_206: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_205, [8, 197, 12, 64]);  view_205 = None
    permute_180: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_17: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    _scaled_dot_product_flash_attention_backward_5 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_180, getitem_98, getitem_99, getitem_100, alias_17, getitem_102, getitem_103, getitem_104, 0, 0, 0.0, False, getitem_107, getitem_108);  permute_180 = getitem_98 = getitem_99 = getitem_100 = alias_17 = getitem_102 = getitem_103 = getitem_104 = getitem_107 = getitem_108 = None
    getitem_209: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_5[0]
    getitem_210: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_5[1]
    getitem_211: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_5[2];  _scaled_dot_product_flash_attention_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_6: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_209, getitem_210, getitem_211]);  getitem_209 = getitem_210 = getitem_211 = None
    view_207: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_6, [3, 8, 12, 197, 64]);  cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_181: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_207, [1, 3, 0, 2, 4]);  view_207 = None
    clone_43: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_181, memory_format = torch.contiguous_format);  permute_181 = None
    view_208: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_43, [8, 197, 2304]);  clone_43 = None
    view_209: "f32[1576, 2304]" = torch.ops.aten.view.default(view_208, [1576, 2304]);  view_208 = None
    permute_182: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    mm_48: "f32[1576, 768]" = torch.ops.aten.mm.default(view_209, permute_182);  permute_182 = None
    permute_183: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_209, [1, 0])
    mm_49: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_183, view_61);  permute_183 = view_61 = None
    permute_184: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_73: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_209, [0], True);  view_209 = None
    view_210: "f32[2304]" = torch.ops.aten.view.default(sum_73, [2304]);  sum_73 = None
    permute_185: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    view_211: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_48, [8, 197, 768]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_61: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_97);  add_42 = getitem_97 = None
    mul_212: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_12);  sub_61 = None
    mul_213: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, primals_77);  primals_77 = None
    mul_214: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
    sum_74: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_212);  mul_213 = None
    sum_75: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_212, sum_75);  sum_75 = None
    sub_62: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_74);  mul_214 = sum_74 = None
    sub_63: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_62, mul_216);  sub_62 = mul_216 = None
    div_12: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    mul_217: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_12, sub_63);  div_12 = sub_63 = None
    mul_218: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_211, mul_212);  mul_212 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_77: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_211, [0, 1]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_110: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_109, mul_217);  add_109 = mul_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_212: "f32[1576, 768]" = torch.ops.aten.view.default(add_110, [1576, 768])
    permute_186: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    mm_50: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_212, permute_186);  permute_186 = None
    permute_187: "f32[768, 1576]" = torch.ops.aten.permute.default(view_212, [1, 0])
    mm_51: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_187, view_59);  permute_187 = view_59 = None
    permute_188: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_78: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_212, [0], True);  view_212 = None
    view_213: "f32[768]" = torch.ops.aten.view.default(sum_78, [768]);  sum_78 = None
    permute_189: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_188, [1, 0]);  permute_188 = None
    view_214: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_50, [8, 197, 3072]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_219: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476)
    erf_18: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_219);  mul_219 = None
    add_111: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_220: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_111, 0.5);  add_111 = None
    mul_221: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, view_58)
    mul_222: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_221, -0.5);  mul_221 = None
    exp_6: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_222);  mul_222 = None
    mul_223: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_6, 0.3989422804014327);  exp_6 = None
    mul_224: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, mul_223);  view_58 = mul_223 = None
    add_112: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_220, mul_224);  mul_220 = mul_224 = None
    mul_225: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_214, add_112);  view_214 = add_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_215: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_225, [1576, 3072]);  mul_225 = None
    permute_190: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_52: "f32[1576, 768]" = torch.ops.aten.mm.default(view_215, permute_190);  permute_190 = None
    permute_191: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_215, [1, 0])
    mm_53: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_191, view_57);  permute_191 = view_57 = None
    permute_192: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_79: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_215, [0], True);  view_215 = None
    view_216: "f32[3072]" = torch.ops.aten.view.default(sum_79, [3072]);  sum_79 = None
    permute_193: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
    view_217: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_52, [8, 197, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_64: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_95);  add_38 = getitem_95 = None
    mul_226: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_11);  sub_64 = None
    mul_227: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_217, primals_71);  primals_71 = None
    mul_228: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_227, 768)
    sum_80: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
    mul_229: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_227, mul_226);  mul_227 = None
    sum_81: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
    mul_230: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_226, sum_81);  sum_81 = None
    sub_65: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_228, sum_80);  mul_228 = sum_80 = None
    sub_66: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_65, mul_230);  sub_65 = mul_230 = None
    div_13: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    mul_231: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_13, sub_66);  div_13 = sub_66 = None
    mul_232: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_217, mul_226);  mul_226 = None
    sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
    sum_83: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_217, [0, 1]);  view_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_113: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_110, mul_231);  add_110 = mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_218: "f32[1576, 768]" = torch.ops.aten.view.default(add_113, [1576, 768])
    permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    mm_54: "f32[1576, 768]" = torch.ops.aten.mm.default(view_218, permute_194);  permute_194 = None
    permute_195: "f32[768, 1576]" = torch.ops.aten.permute.default(view_218, [1, 0])
    mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_195, view_55);  permute_195 = view_55 = None
    permute_196: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_84: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_218, [0], True);  view_218 = None
    view_219: "f32[768]" = torch.ops.aten.view.default(sum_84, [768]);  sum_84 = None
    permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_220: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_54, [8, 197, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_221: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_220, [8, 197, 12, 64]);  view_220 = None
    permute_198: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_18: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    _scaled_dot_product_flash_attention_backward_6 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_198, getitem_82, getitem_83, getitem_84, alias_18, getitem_86, getitem_87, getitem_88, 0, 0, 0.0, False, getitem_91, getitem_92);  permute_198 = getitem_82 = getitem_83 = getitem_84 = alias_18 = getitem_86 = getitem_87 = getitem_88 = getitem_91 = getitem_92 = None
    getitem_212: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_6[0]
    getitem_213: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_6[1]
    getitem_214: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_6[2];  _scaled_dot_product_flash_attention_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_7: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_212, getitem_213, getitem_214]);  getitem_212 = getitem_213 = getitem_214 = None
    view_222: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_7, [3, 8, 12, 197, 64]);  cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_199: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_222, [1, 3, 0, 2, 4]);  view_222 = None
    clone_44: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_199, memory_format = torch.contiguous_format);  permute_199 = None
    view_223: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_44, [8, 197, 2304]);  clone_44 = None
    view_224: "f32[1576, 2304]" = torch.ops.aten.view.default(view_223, [1576, 2304]);  view_223 = None
    permute_200: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_56: "f32[1576, 768]" = torch.ops.aten.mm.default(view_224, permute_200);  permute_200 = None
    permute_201: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_224, [1, 0])
    mm_57: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_201, view_51);  permute_201 = view_51 = None
    permute_202: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_85: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[2304]" = torch.ops.aten.view.default(sum_85, [2304]);  sum_85 = None
    permute_203: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    view_226: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_56, [8, 197, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_67: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_81);  add_35 = getitem_81 = None
    mul_233: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_10);  sub_67 = None
    mul_234: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_226, primals_65);  primals_65 = None
    mul_235: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_234, 768)
    sum_86: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_234, [2], True)
    mul_236: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_234, mul_233);  mul_234 = None
    sum_87: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_236, [2], True);  mul_236 = None
    mul_237: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_233, sum_87);  sum_87 = None
    sub_68: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_235, sum_86);  mul_235 = sum_86 = None
    sub_69: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_68, mul_237);  sub_68 = mul_237 = None
    div_14: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    mul_238: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_14, sub_69);  div_14 = sub_69 = None
    mul_239: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_226, mul_233);  mul_233 = None
    sum_88: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_239, [0, 1]);  mul_239 = None
    sum_89: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_226, [0, 1]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_114: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_113, mul_238);  add_113 = mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_227: "f32[1576, 768]" = torch.ops.aten.view.default(add_114, [1576, 768])
    permute_204: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_58: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_227, permute_204);  permute_204 = None
    permute_205: "f32[768, 1576]" = torch.ops.aten.permute.default(view_227, [1, 0])
    mm_59: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_205, view_49);  permute_205 = view_49 = None
    permute_206: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_90: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
    view_228: "f32[768]" = torch.ops.aten.view.default(sum_90, [768]);  sum_90 = None
    permute_207: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
    view_229: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_58, [8, 197, 3072]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_240: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476)
    erf_19: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_240);  mul_240 = None
    add_115: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_241: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_115, 0.5);  add_115 = None
    mul_242: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_48, view_48)
    mul_243: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_242, -0.5);  mul_242 = None
    exp_7: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_243);  mul_243 = None
    mul_244: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_7, 0.3989422804014327);  exp_7 = None
    mul_245: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_48, mul_244);  view_48 = mul_244 = None
    add_116: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_241, mul_245);  mul_241 = mul_245 = None
    mul_246: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_229, add_116);  view_229 = add_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_230: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_246, [1576, 3072]);  mul_246 = None
    permute_208: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    mm_60: "f32[1576, 768]" = torch.ops.aten.mm.default(view_230, permute_208);  permute_208 = None
    permute_209: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_230, [1, 0])
    mm_61: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_209, view_47);  permute_209 = view_47 = None
    permute_210: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_91: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_230, [0], True);  view_230 = None
    view_231: "f32[3072]" = torch.ops.aten.view.default(sum_91, [3072]);  sum_91 = None
    permute_211: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_210, [1, 0]);  permute_210 = None
    view_232: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_60, [8, 197, 768]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_70: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_79);  add_31 = getitem_79 = None
    mul_247: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_9);  sub_70 = None
    mul_248: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_232, primals_59);  primals_59 = None
    mul_249: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_248, 768)
    sum_92: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_248, [2], True)
    mul_250: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_248, mul_247);  mul_248 = None
    sum_93: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True);  mul_250 = None
    mul_251: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_247, sum_93);  sum_93 = None
    sub_71: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_249, sum_92);  mul_249 = sum_92 = None
    sub_72: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_71, mul_251);  sub_71 = mul_251 = None
    div_15: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    mul_252: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_15, sub_72);  div_15 = sub_72 = None
    mul_253: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_232, mul_247);  mul_247 = None
    sum_94: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_253, [0, 1]);  mul_253 = None
    sum_95: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_232, [0, 1]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_117: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_114, mul_252);  add_114 = mul_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_233: "f32[1576, 768]" = torch.ops.aten.view.default(add_117, [1576, 768])
    permute_212: "f32[768, 768]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    mm_62: "f32[1576, 768]" = torch.ops.aten.mm.default(view_233, permute_212);  permute_212 = None
    permute_213: "f32[768, 1576]" = torch.ops.aten.permute.default(view_233, [1, 0])
    mm_63: "f32[768, 768]" = torch.ops.aten.mm.default(permute_213, view_45);  permute_213 = view_45 = None
    permute_214: "f32[768, 768]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_233, [0], True);  view_233 = None
    view_234: "f32[768]" = torch.ops.aten.view.default(sum_96, [768]);  sum_96 = None
    permute_215: "f32[768, 768]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    view_235: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_62, [8, 197, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_236: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_235, [8, 197, 12, 64]);  view_235 = None
    permute_216: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_236, [0, 2, 1, 3]);  view_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_19: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    _scaled_dot_product_flash_attention_backward_7 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_216, getitem_66, getitem_67, getitem_68, alias_19, getitem_70, getitem_71, getitem_72, 0, 0, 0.0, False, getitem_75, getitem_76);  permute_216 = getitem_66 = getitem_67 = getitem_68 = alias_19 = getitem_70 = getitem_71 = getitem_72 = getitem_75 = getitem_76 = None
    getitem_215: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_7[0]
    getitem_216: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_7[1]
    getitem_217: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_7[2];  _scaled_dot_product_flash_attention_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_8: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_215, getitem_216, getitem_217]);  getitem_215 = getitem_216 = getitem_217 = None
    view_237: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_8, [3, 8, 12, 197, 64]);  cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_217: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_237, [1, 3, 0, 2, 4]);  view_237 = None
    clone_45: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_217, memory_format = torch.contiguous_format);  permute_217 = None
    view_238: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_45, [8, 197, 2304]);  clone_45 = None
    view_239: "f32[1576, 2304]" = torch.ops.aten.view.default(view_238, [1576, 2304]);  view_238 = None
    permute_218: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    mm_64: "f32[1576, 768]" = torch.ops.aten.mm.default(view_239, permute_218);  permute_218 = None
    permute_219: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_239, [1, 0])
    mm_65: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_219, view_41);  permute_219 = view_41 = None
    permute_220: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_97: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_239, [0], True);  view_239 = None
    view_240: "f32[2304]" = torch.ops.aten.view.default(sum_97, [2304]);  sum_97 = None
    permute_221: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    view_241: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_64, [8, 197, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_73: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_65);  add_28 = getitem_65 = None
    mul_254: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_73, rsqrt_8);  sub_73 = None
    mul_255: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_241, primals_53);  primals_53 = None
    mul_256: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_255, 768)
    sum_98: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True)
    mul_257: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_255, mul_254);  mul_255 = None
    sum_99: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_257, [2], True);  mul_257 = None
    mul_258: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_254, sum_99);  sum_99 = None
    sub_74: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_256, sum_98);  mul_256 = sum_98 = None
    sub_75: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_74, mul_258);  sub_74 = mul_258 = None
    div_16: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    mul_259: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_16, sub_75);  div_16 = sub_75 = None
    mul_260: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_241, mul_254);  mul_254 = None
    sum_100: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_260, [0, 1]);  mul_260 = None
    sum_101: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_241, [0, 1]);  view_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_118: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_117, mul_259);  add_117 = mul_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_242: "f32[1576, 768]" = torch.ops.aten.view.default(add_118, [1576, 768])
    permute_222: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_66: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_242, permute_222);  permute_222 = None
    permute_223: "f32[768, 1576]" = torch.ops.aten.permute.default(view_242, [1, 0])
    mm_67: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_223, view_39);  permute_223 = view_39 = None
    permute_224: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_242, [0], True);  view_242 = None
    view_243: "f32[768]" = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
    permute_225: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    view_244: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_66, [8, 197, 3072]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_261: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_20: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_261);  mul_261 = None
    add_119: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_262: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_119, 0.5);  add_119 = None
    mul_263: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_38, view_38)
    mul_264: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_263, -0.5);  mul_263 = None
    exp_8: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_264);  mul_264 = None
    mul_265: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_266: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_38, mul_265);  view_38 = mul_265 = None
    add_120: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_262, mul_266);  mul_262 = mul_266 = None
    mul_267: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_244, add_120);  view_244 = add_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_245: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_267, [1576, 3072]);  mul_267 = None
    permute_226: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    mm_68: "f32[1576, 768]" = torch.ops.aten.mm.default(view_245, permute_226);  permute_226 = None
    permute_227: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_245, [1, 0])
    mm_69: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_227, view_37);  permute_227 = view_37 = None
    permute_228: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_103: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_245, [0], True);  view_245 = None
    view_246: "f32[3072]" = torch.ops.aten.view.default(sum_103, [3072]);  sum_103 = None
    permute_229: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    view_247: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_68, [8, 197, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_76: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_63);  add_24 = getitem_63 = None
    mul_268: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_7);  sub_76 = None
    mul_269: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_247, primals_47);  primals_47 = None
    mul_270: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_269, 768)
    sum_104: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
    mul_271: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_269, mul_268);  mul_269 = None
    sum_105: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
    mul_272: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_268, sum_105);  sum_105 = None
    sub_77: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_270, sum_104);  mul_270 = sum_104 = None
    sub_78: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_77, mul_272);  sub_77 = mul_272 = None
    div_17: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    mul_273: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_17, sub_78);  div_17 = sub_78 = None
    mul_274: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_247, mul_268);  mul_268 = None
    sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
    sum_107: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_247, [0, 1]);  view_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_121: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_118, mul_273);  add_118 = mul_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_248: "f32[1576, 768]" = torch.ops.aten.view.default(add_121, [1576, 768])
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_70: "f32[1576, 768]" = torch.ops.aten.mm.default(view_248, permute_230);  permute_230 = None
    permute_231: "f32[768, 1576]" = torch.ops.aten.permute.default(view_248, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_231, view_35);  permute_231 = view_35 = None
    permute_232: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_108: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_248, [0], True);  view_248 = None
    view_249: "f32[768]" = torch.ops.aten.view.default(sum_108, [768]);  sum_108 = None
    permute_233: "f32[768, 768]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    view_250: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_70, [8, 197, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_251: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_250, [8, 197, 12, 64]);  view_250 = None
    permute_234: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_251, [0, 2, 1, 3]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_20: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    _scaled_dot_product_flash_attention_backward_8 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_234, getitem_50, getitem_51, getitem_52, alias_20, getitem_54, getitem_55, getitem_56, 0, 0, 0.0, False, getitem_59, getitem_60);  permute_234 = getitem_50 = getitem_51 = getitem_52 = alias_20 = getitem_54 = getitem_55 = getitem_56 = getitem_59 = getitem_60 = None
    getitem_218: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_8[0]
    getitem_219: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_8[1]
    getitem_220: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_8[2];  _scaled_dot_product_flash_attention_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_9: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_218, getitem_219, getitem_220]);  getitem_218 = getitem_219 = getitem_220 = None
    view_252: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_9, [3, 8, 12, 197, 64]);  cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_235: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_252, [1, 3, 0, 2, 4]);  view_252 = None
    clone_46: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_235, memory_format = torch.contiguous_format);  permute_235 = None
    view_253: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_46, [8, 197, 2304]);  clone_46 = None
    view_254: "f32[1576, 2304]" = torch.ops.aten.view.default(view_253, [1576, 2304]);  view_253 = None
    permute_236: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_72: "f32[1576, 768]" = torch.ops.aten.mm.default(view_254, permute_236);  permute_236 = None
    permute_237: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_254, [1, 0])
    mm_73: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_237, view_31);  permute_237 = view_31 = None
    permute_238: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_109: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_254, [0], True);  view_254 = None
    view_255: "f32[2304]" = torch.ops.aten.view.default(sum_109, [2304]);  sum_109 = None
    permute_239: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
    view_256: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_72, [8, 197, 768]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_79: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_49);  add_21 = getitem_49 = None
    mul_275: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_6);  sub_79 = None
    mul_276: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_256, primals_41);  primals_41 = None
    mul_277: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_276, 768)
    sum_110: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True)
    mul_278: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_276, mul_275);  mul_276 = None
    sum_111: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_278, [2], True);  mul_278 = None
    mul_279: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_275, sum_111);  sum_111 = None
    sub_80: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_277, sum_110);  mul_277 = sum_110 = None
    sub_81: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_80, mul_279);  sub_80 = mul_279 = None
    div_18: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    mul_280: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_18, sub_81);  div_18 = sub_81 = None
    mul_281: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_256, mul_275);  mul_275 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_281, [0, 1]);  mul_281 = None
    sum_113: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_256, [0, 1]);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_122: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_121, mul_280);  add_121 = mul_280 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_257: "f32[1576, 768]" = torch.ops.aten.view.default(add_122, [1576, 768])
    permute_240: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    mm_74: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_257, permute_240);  permute_240 = None
    permute_241: "f32[768, 1576]" = torch.ops.aten.permute.default(view_257, [1, 0])
    mm_75: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_241, view_29);  permute_241 = view_29 = None
    permute_242: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_114: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_257, [0], True);  view_257 = None
    view_258: "f32[768]" = torch.ops.aten.view.default(sum_114, [768]);  sum_114 = None
    permute_243: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_259: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_74, [8, 197, 3072]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_282: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476)
    erf_21: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_282);  mul_282 = None
    add_123: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_283: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_123, 0.5);  add_123 = None
    mul_284: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_28, view_28)
    mul_285: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_284, -0.5);  mul_284 = None
    exp_9: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_285);  mul_285 = None
    mul_286: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_287: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_28, mul_286);  view_28 = mul_286 = None
    add_124: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_283, mul_287);  mul_283 = mul_287 = None
    mul_288: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_259, add_124);  view_259 = add_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_260: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_288, [1576, 3072]);  mul_288 = None
    permute_244: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    mm_76: "f32[1576, 768]" = torch.ops.aten.mm.default(view_260, permute_244);  permute_244 = None
    permute_245: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_260, [1, 0])
    mm_77: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_245, view_27);  permute_245 = view_27 = None
    permute_246: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_115: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_260, [0], True);  view_260 = None
    view_261: "f32[3072]" = torch.ops.aten.view.default(sum_115, [3072]);  sum_115 = None
    permute_247: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_262: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_76, [8, 197, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_82: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_47);  add_17 = getitem_47 = None
    mul_289: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_5);  sub_82 = None
    mul_290: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_262, primals_35);  primals_35 = None
    mul_291: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_290, 768)
    sum_116: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True)
    mul_292: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_290, mul_289);  mul_290 = None
    sum_117: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_292, [2], True);  mul_292 = None
    mul_293: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_289, sum_117);  sum_117 = None
    sub_83: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_291, sum_116);  mul_291 = sum_116 = None
    sub_84: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_83, mul_293);  sub_83 = mul_293 = None
    div_19: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    mul_294: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_19, sub_84);  div_19 = sub_84 = None
    mul_295: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_262, mul_289);  mul_289 = None
    sum_118: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_295, [0, 1]);  mul_295 = None
    sum_119: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_262, [0, 1]);  view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_125: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_122, mul_294);  add_122 = mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_263: "f32[1576, 768]" = torch.ops.aten.view.default(add_125, [1576, 768])
    permute_248: "f32[768, 768]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    mm_78: "f32[1576, 768]" = torch.ops.aten.mm.default(view_263, permute_248);  permute_248 = None
    permute_249: "f32[768, 1576]" = torch.ops.aten.permute.default(view_263, [1, 0])
    mm_79: "f32[768, 768]" = torch.ops.aten.mm.default(permute_249, view_25);  permute_249 = view_25 = None
    permute_250: "f32[768, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_120: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_263, [0], True);  view_263 = None
    view_264: "f32[768]" = torch.ops.aten.view.default(sum_120, [768]);  sum_120 = None
    permute_251: "f32[768, 768]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    view_265: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_78, [8, 197, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_266: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_265, [8, 197, 12, 64]);  view_265 = None
    permute_252: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_21: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    _scaled_dot_product_flash_attention_backward_9 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_252, getitem_34, getitem_35, getitem_36, alias_21, getitem_38, getitem_39, getitem_40, 0, 0, 0.0, False, getitem_43, getitem_44);  permute_252 = getitem_34 = getitem_35 = getitem_36 = alias_21 = getitem_38 = getitem_39 = getitem_40 = getitem_43 = getitem_44 = None
    getitem_221: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_9[0]
    getitem_222: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_9[1]
    getitem_223: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_9[2];  _scaled_dot_product_flash_attention_backward_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_10: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_221, getitem_222, getitem_223]);  getitem_221 = getitem_222 = getitem_223 = None
    view_267: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_10, [3, 8, 12, 197, 64]);  cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_253: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_267, [1, 3, 0, 2, 4]);  view_267 = None
    clone_47: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_253, memory_format = torch.contiguous_format);  permute_253 = None
    view_268: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_47, [8, 197, 2304]);  clone_47 = None
    view_269: "f32[1576, 2304]" = torch.ops.aten.view.default(view_268, [1576, 2304]);  view_268 = None
    permute_254: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_80: "f32[1576, 768]" = torch.ops.aten.mm.default(view_269, permute_254);  permute_254 = None
    permute_255: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_269, [1, 0])
    mm_81: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_255, view_21);  permute_255 = view_21 = None
    permute_256: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_121: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_269, [0], True);  view_269 = None
    view_270: "f32[2304]" = torch.ops.aten.view.default(sum_121, [2304]);  sum_121 = None
    permute_257: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    view_271: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_80, [8, 197, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_85: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_33);  add_14 = getitem_33 = None
    mul_296: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_85, rsqrt_4);  sub_85 = None
    mul_297: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_271, primals_29);  primals_29 = None
    mul_298: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_297, 768)
    sum_122: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [2], True)
    mul_299: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_297, mul_296);  mul_297 = None
    sum_123: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_299, [2], True);  mul_299 = None
    mul_300: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_296, sum_123);  sum_123 = None
    sub_86: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_298, sum_122);  mul_298 = sum_122 = None
    sub_87: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_86, mul_300);  sub_86 = mul_300 = None
    div_20: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    mul_301: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_20, sub_87);  div_20 = sub_87 = None
    mul_302: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_271, mul_296);  mul_296 = None
    sum_124: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_302, [0, 1]);  mul_302 = None
    sum_125: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_271, [0, 1]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_126: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_125, mul_301);  add_125 = mul_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_272: "f32[1576, 768]" = torch.ops.aten.view.default(add_126, [1576, 768])
    permute_258: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    mm_82: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_272, permute_258);  permute_258 = None
    permute_259: "f32[768, 1576]" = torch.ops.aten.permute.default(view_272, [1, 0])
    mm_83: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_259, view_19);  permute_259 = view_19 = None
    permute_260: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
    view_273: "f32[768]" = torch.ops.aten.view.default(sum_126, [768]);  sum_126 = None
    permute_261: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
    view_274: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_82, [8, 197, 3072]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_303: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf_22: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_303);  mul_303 = None
    add_127: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_304: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_127, 0.5);  add_127 = None
    mul_305: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_18, view_18)
    mul_306: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_305, -0.5);  mul_305 = None
    exp_10: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_306);  mul_306 = None
    mul_307: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_308: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_18, mul_307);  view_18 = mul_307 = None
    add_128: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_304, mul_308);  mul_304 = mul_308 = None
    mul_309: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_274, add_128);  view_274 = add_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_275: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_309, [1576, 3072]);  mul_309 = None
    permute_262: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_84: "f32[1576, 768]" = torch.ops.aten.mm.default(view_275, permute_262);  permute_262 = None
    permute_263: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_275, [1, 0])
    mm_85: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_263, view_17);  permute_263 = view_17 = None
    permute_264: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_127: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_275, [0], True);  view_275 = None
    view_276: "f32[3072]" = torch.ops.aten.view.default(sum_127, [3072]);  sum_127 = None
    permute_265: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    view_277: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_84, [8, 197, 768]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_88: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_31);  add_10 = getitem_31 = None
    mul_310: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_3);  sub_88 = None
    mul_311: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_277, primals_23);  primals_23 = None
    mul_312: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_311, 768)
    sum_128: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_311, [2], True)
    mul_313: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_311, mul_310);  mul_311 = None
    sum_129: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_313, [2], True);  mul_313 = None
    mul_314: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_310, sum_129);  sum_129 = None
    sub_89: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_312, sum_128);  mul_312 = sum_128 = None
    sub_90: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_89, mul_314);  sub_89 = mul_314 = None
    div_21: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    mul_315: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_21, sub_90);  div_21 = sub_90 = None
    mul_316: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_277, mul_310);  mul_310 = None
    sum_130: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_316, [0, 1]);  mul_316 = None
    sum_131: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_277, [0, 1]);  view_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_129: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_126, mul_315);  add_126 = mul_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_278: "f32[1576, 768]" = torch.ops.aten.view.default(add_129, [1576, 768])
    permute_266: "f32[768, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_86: "f32[1576, 768]" = torch.ops.aten.mm.default(view_278, permute_266);  permute_266 = None
    permute_267: "f32[768, 1576]" = torch.ops.aten.permute.default(view_278, [1, 0])
    mm_87: "f32[768, 768]" = torch.ops.aten.mm.default(permute_267, view_15);  permute_267 = view_15 = None
    permute_268: "f32[768, 768]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_278, [0], True);  view_278 = None
    view_279: "f32[768]" = torch.ops.aten.view.default(sum_132, [768]);  sum_132 = None
    permute_269: "f32[768, 768]" = torch.ops.aten.permute.default(permute_268, [1, 0]);  permute_268 = None
    view_280: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_86, [8, 197, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_281: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_280, [8, 197, 12, 64]);  view_280 = None
    permute_270: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_22: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    _scaled_dot_product_flash_attention_backward_10 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_270, getitem_18, getitem_19, getitem_20, alias_22, getitem_22, getitem_23, getitem_24, 0, 0, 0.0, False, getitem_27, getitem_28);  permute_270 = getitem_18 = getitem_19 = getitem_20 = alias_22 = getitem_22 = getitem_23 = getitem_24 = getitem_27 = getitem_28 = None
    getitem_224: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_10[0]
    getitem_225: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_10[1]
    getitem_226: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_10[2];  _scaled_dot_product_flash_attention_backward_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_11: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_224, getitem_225, getitem_226]);  getitem_224 = getitem_225 = getitem_226 = None
    view_282: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_11, [3, 8, 12, 197, 64]);  cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_271: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_282, [1, 3, 0, 2, 4]);  view_282 = None
    clone_48: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_271, memory_format = torch.contiguous_format);  permute_271 = None
    view_283: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_48, [8, 197, 2304]);  clone_48 = None
    view_284: "f32[1576, 2304]" = torch.ops.aten.view.default(view_283, [1576, 2304]);  view_283 = None
    permute_272: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    mm_88: "f32[1576, 768]" = torch.ops.aten.mm.default(view_284, permute_272);  permute_272 = None
    permute_273: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_284, [1, 0])
    mm_89: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_273, view_11);  permute_273 = view_11 = None
    permute_274: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_133: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_284, [0], True);  view_284 = None
    view_285: "f32[2304]" = torch.ops.aten.view.default(sum_133, [2304]);  sum_133 = None
    permute_275: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    view_286: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_88, [8, 197, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_91: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_17);  add_7 = getitem_17 = None
    mul_317: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_2);  sub_91 = None
    mul_318: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_286, primals_17);  primals_17 = None
    mul_319: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_318, 768)
    sum_134: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True)
    mul_320: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_318, mul_317);  mul_318 = None
    sum_135: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_320, [2], True);  mul_320 = None
    mul_321: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_317, sum_135);  sum_135 = None
    sub_92: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_319, sum_134);  mul_319 = sum_134 = None
    sub_93: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_92, mul_321);  sub_92 = mul_321 = None
    div_22: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    mul_322: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_22, sub_93);  div_22 = sub_93 = None
    mul_323: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_286, mul_317);  mul_317 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_323, [0, 1]);  mul_323 = None
    sum_137: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_286, [0, 1]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_130: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_129, mul_322);  add_129 = mul_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_287: "f32[1576, 768]" = torch.ops.aten.view.default(add_130, [1576, 768])
    permute_276: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    mm_90: "f32[1576, 3072]" = torch.ops.aten.mm.default(view_287, permute_276);  permute_276 = None
    permute_277: "f32[768, 1576]" = torch.ops.aten.permute.default(view_287, [1, 0])
    mm_91: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_277, view_9);  permute_277 = view_9 = None
    permute_278: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_138: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[768]" = torch.ops.aten.view.default(sum_138, [768]);  sum_138 = None
    permute_279: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
    view_289: "f32[8, 197, 3072]" = torch.ops.aten.view.default(mm_90, [8, 197, 3072]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_324: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476)
    erf_23: "f32[8, 197, 3072]" = torch.ops.aten.erf.default(mul_324);  mul_324 = None
    add_131: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_325: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(add_131, 0.5);  add_131 = None
    mul_326: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_8, view_8)
    mul_327: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(mul_326, -0.5);  mul_326 = None
    exp_11: "f32[8, 197, 3072]" = torch.ops.aten.exp.default(mul_327);  mul_327 = None
    mul_328: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_329: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_8, mul_328);  view_8 = mul_328 = None
    add_132: "f32[8, 197, 3072]" = torch.ops.aten.add.Tensor(mul_325, mul_329);  mul_325 = mul_329 = None
    mul_330: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_289, add_132);  view_289 = add_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_290: "f32[1576, 3072]" = torch.ops.aten.view.default(mul_330, [1576, 3072]);  mul_330 = None
    permute_280: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    mm_92: "f32[1576, 768]" = torch.ops.aten.mm.default(view_290, permute_280);  permute_280 = None
    permute_281: "f32[3072, 1576]" = torch.ops.aten.permute.default(view_290, [1, 0])
    mm_93: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_281, view_7);  permute_281 = view_7 = None
    permute_282: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_139: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_290, [0], True);  view_290 = None
    view_291: "f32[3072]" = torch.ops.aten.view.default(sum_139, [3072]);  sum_139 = None
    permute_283: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
    view_292: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_92, [8, 197, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    sub_94: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_15);  add_3 = getitem_15 = None
    mul_331: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_1);  sub_94 = None
    mul_332: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_292, primals_11);  primals_11 = None
    mul_333: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_332, 768)
    sum_140: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_332, [2], True)
    mul_334: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_332, mul_331);  mul_332 = None
    sum_141: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_334, [2], True);  mul_334 = None
    mul_335: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_331, sum_141);  sum_141 = None
    sub_95: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_333, sum_140);  mul_333 = sum_140 = None
    sub_96: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_95, mul_335);  sub_95 = mul_335 = None
    div_23: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    mul_336: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_23, sub_96);  div_23 = sub_96 = None
    mul_337: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_292, mul_331);  mul_331 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_337, [0, 1]);  mul_337 = None
    sum_143: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_292, [0, 1]);  view_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_133: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_130, mul_336);  add_130 = mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_293: "f32[1576, 768]" = torch.ops.aten.view.default(add_133, [1576, 768])
    permute_284: "f32[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    mm_94: "f32[1576, 768]" = torch.ops.aten.mm.default(view_293, permute_284);  permute_284 = None
    permute_285: "f32[768, 1576]" = torch.ops.aten.permute.default(view_293, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_285, view_5);  permute_285 = view_5 = None
    permute_286: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_144: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
    view_294: "f32[768]" = torch.ops.aten.view.default(sum_144, [768]);  sum_144 = None
    permute_287: "f32[768, 768]" = torch.ops.aten.permute.default(permute_286, [1, 0]);  permute_286 = None
    view_295: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_94, [8, 197, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    view_296: "f32[8, 197, 12, 64]" = torch.ops.aten.view.default(view_295, [8, 197, 12, 64]);  view_295 = None
    permute_288: "f32[8, 12, 197, 64]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_23: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias);  alias = None
    _scaled_dot_product_flash_attention_backward_11 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(permute_288, getitem_2, getitem_3, getitem_4, alias_23, getitem_6, getitem_7, getitem_8, 0, 0, 0.0, False, getitem_11, getitem_12);  permute_288 = getitem_2 = getitem_3 = getitem_4 = alias_23 = getitem_6 = getitem_7 = getitem_8 = getitem_11 = getitem_12 = None
    getitem_227: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_11[0]
    getitem_228: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_11[1]
    getitem_229: "f32[8, 12, 197, 64]" = _scaled_dot_product_flash_attention_backward_11[2];  _scaled_dot_product_flash_attention_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    cat_12: "f32[24, 12, 197, 64]" = torch.ops.aten.cat.default([getitem_227, getitem_228, getitem_229]);  getitem_227 = getitem_228 = getitem_229 = None
    view_297: "f32[3, 8, 12, 197, 64]" = torch.ops.aten.view.default(cat_12, [3, 8, 12, 197, 64]);  cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_289: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.permute.default(view_297, [1, 3, 0, 2, 4]);  view_297 = None
    clone_49: "f32[8, 197, 3, 12, 64]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    view_298: "f32[8, 197, 2304]" = torch.ops.aten.view.default(clone_49, [8, 197, 2304]);  clone_49 = None
    view_299: "f32[1576, 2304]" = torch.ops.aten.view.default(view_298, [1576, 2304]);  view_298 = None
    permute_290: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    mm_96: "f32[1576, 768]" = torch.ops.aten.mm.default(view_299, permute_290);  permute_290 = None
    permute_291: "f32[2304, 1576]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_97: "f32[2304, 768]" = torch.ops.aten.mm.default(permute_291, view_1);  permute_291 = view_1 = None
    permute_292: "f32[768, 2304]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_145: "f32[1, 2304]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[2304]" = torch.ops.aten.view.default(sum_145, [2304]);  sum_145 = None
    permute_293: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    view_301: "f32[8, 197, 768]" = torch.ops.aten.view.default(mm_96, [8, 197, 768]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    sub_97: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  clone = getitem_1 = None
    mul_338: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_97, rsqrt);  sub_97 = None
    mul_339: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_301, primals_5);  primals_5 = None
    mul_340: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_339, 768)
    sum_146: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True)
    mul_341: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_339, mul_338);  mul_339 = None
    sum_147: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
    mul_342: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_338, sum_147);  sum_147 = None
    sub_98: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(mul_340, sum_146);  mul_340 = sum_146 = None
    sub_99: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(sub_98, mul_342);  sub_98 = mul_342 = None
    div_24: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    mul_343: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(div_24, sub_99);  div_24 = sub_99 = None
    mul_344: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(view_301, mul_338);  mul_338 = None
    sum_148: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 1]);  mul_344 = None
    sum_149: "f32[768]" = torch.ops.aten.sum.dim_IntList(view_301, [0, 1]);  view_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_134: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_133, mul_343);  add_133 = mul_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:580, code: x = x + pos_embed
    sum_150: "f32[1, 197, 768]" = torch.ops.aten.sum.dim_IntList(add_134, [0], True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:579, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    slice_2: "f32[8, 1, 768]" = torch.ops.aten.slice.Tensor(add_134, 1, 0, 1)
    slice_3: "f32[8, 196, 768]" = torch.ops.aten.slice.Tensor(add_134, 1, 1, 197);  add_134 = None
    sum_151: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(slice_2, [0], True);  slice_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_294: "f32[8, 768, 196]" = torch.ops.aten.permute.default(slice_3, [0, 2, 1]);  slice_3 = None
    view_302: "f32[8, 768, 14, 14]" = torch.ops.aten.view.default(permute_294, [8, 768, 14, 14]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward = torch.ops.aten.convolution_backward.default(view_302, primals_153, primals_3, [768], [16, 16], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_302 = primals_153 = primals_3 = None
    getitem_231: "f32[768, 3, 16, 16]" = convolution_backward[1]
    getitem_232: "f32[768]" = convolution_backward[2];  convolution_backward = None
    return pytree.tree_unflatten([addmm_48, sum_150, sum_151, getitem_231, getitem_232, sum_148, sum_149, permute_293, view_300, permute_287, view_294, sum_142, sum_143, permute_283, view_291, permute_279, view_288, sum_136, sum_137, permute_275, view_285, permute_269, view_279, sum_130, sum_131, permute_265, view_276, permute_261, view_273, sum_124, sum_125, permute_257, view_270, permute_251, view_264, sum_118, sum_119, permute_247, view_261, permute_243, view_258, sum_112, sum_113, permute_239, view_255, permute_233, view_249, sum_106, sum_107, permute_229, view_246, permute_225, view_243, sum_100, sum_101, permute_221, view_240, permute_215, view_234, sum_94, sum_95, permute_211, view_231, permute_207, view_228, sum_88, sum_89, permute_203, view_225, permute_197, view_219, sum_82, sum_83, permute_193, view_216, permute_189, view_213, sum_76, sum_77, permute_185, view_210, permute_179, view_204, sum_70, sum_71, permute_175, view_201, permute_171, view_198, sum_64, sum_65, permute_167, view_195, permute_161, view_189, sum_58, sum_59, permute_157, view_186, permute_153, view_183, sum_52, sum_53, permute_149, view_180, permute_143, view_174, sum_46, sum_47, permute_139, view_171, permute_135, view_168, sum_40, sum_41, permute_131, view_165, permute_125, view_159, sum_34, sum_35, permute_121, view_156, permute_117, view_153, sum_28, sum_29, permute_113, view_150, permute_107, view_144, sum_22, sum_23, permute_103, view_141, permute_99, view_138, sum_16, sum_17, permute_95, view_135, permute_89, view_129, sum_10, sum_11, permute_85, view_126, permute_81, view_123, sum_4, sum_5, permute_77, view_121, None], self._out_spec)
    