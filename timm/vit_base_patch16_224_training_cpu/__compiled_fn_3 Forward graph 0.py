from __future__ import annotations



def forward(self, primals_1: "f32[1, 197, 768]", primals_2: "f32[1, 1, 768]", primals_3: "f32[768, 3, 16, 16]", primals_4: "f32[768]", primals_5: "f32[768]", primals_6: "f32[768]", primals_7: "f32[2304, 768]", primals_8: "f32[2304]", primals_9: "f32[768, 768]", primals_10: "f32[768]", primals_11: "f32[768]", primals_12: "f32[768]", primals_13: "f32[3072, 768]", primals_14: "f32[3072]", primals_15: "f32[768, 3072]", primals_16: "f32[768]", primals_17: "f32[768]", primals_18: "f32[768]", primals_19: "f32[2304, 768]", primals_20: "f32[2304]", primals_21: "f32[768, 768]", primals_22: "f32[768]", primals_23: "f32[768]", primals_24: "f32[768]", primals_25: "f32[3072, 768]", primals_26: "f32[3072]", primals_27: "f32[768, 3072]", primals_28: "f32[768]", primals_29: "f32[768]", primals_30: "f32[768]", primals_31: "f32[2304, 768]", primals_32: "f32[2304]", primals_33: "f32[768, 768]", primals_34: "f32[768]", primals_35: "f32[768]", primals_36: "f32[768]", primals_37: "f32[3072, 768]", primals_38: "f32[3072]", primals_39: "f32[768, 3072]", primals_40: "f32[768]", primals_41: "f32[768]", primals_42: "f32[768]", primals_43: "f32[2304, 768]", primals_44: "f32[2304]", primals_45: "f32[768, 768]", primals_46: "f32[768]", primals_47: "f32[768]", primals_48: "f32[768]", primals_49: "f32[3072, 768]", primals_50: "f32[3072]", primals_51: "f32[768, 3072]", primals_52: "f32[768]", primals_53: "f32[768]", primals_54: "f32[768]", primals_55: "f32[2304, 768]", primals_56: "f32[2304]", primals_57: "f32[768, 768]", primals_58: "f32[768]", primals_59: "f32[768]", primals_60: "f32[768]", primals_61: "f32[3072, 768]", primals_62: "f32[3072]", primals_63: "f32[768, 3072]", primals_64: "f32[768]", primals_65: "f32[768]", primals_66: "f32[768]", primals_67: "f32[2304, 768]", primals_68: "f32[2304]", primals_69: "f32[768, 768]", primals_70: "f32[768]", primals_71: "f32[768]", primals_72: "f32[768]", primals_73: "f32[3072, 768]", primals_74: "f32[3072]", primals_75: "f32[768, 3072]", primals_76: "f32[768]", primals_77: "f32[768]", primals_78: "f32[768]", primals_79: "f32[2304, 768]", primals_80: "f32[2304]", primals_81: "f32[768, 768]", primals_82: "f32[768]", primals_83: "f32[768]", primals_84: "f32[768]", primals_85: "f32[3072, 768]", primals_86: "f32[3072]", primals_87: "f32[768, 3072]", primals_88: "f32[768]", primals_89: "f32[768]", primals_90: "f32[768]", primals_91: "f32[2304, 768]", primals_92: "f32[2304]", primals_93: "f32[768, 768]", primals_94: "f32[768]", primals_95: "f32[768]", primals_96: "f32[768]", primals_97: "f32[3072, 768]", primals_98: "f32[3072]", primals_99: "f32[768, 3072]", primals_100: "f32[768]", primals_101: "f32[768]", primals_102: "f32[768]", primals_103: "f32[2304, 768]", primals_104: "f32[2304]", primals_105: "f32[768, 768]", primals_106: "f32[768]", primals_107: "f32[768]", primals_108: "f32[768]", primals_109: "f32[3072, 768]", primals_110: "f32[3072]", primals_111: "f32[768, 3072]", primals_112: "f32[768]", primals_113: "f32[768]", primals_114: "f32[768]", primals_115: "f32[2304, 768]", primals_116: "f32[2304]", primals_117: "f32[768, 768]", primals_118: "f32[768]", primals_119: "f32[768]", primals_120: "f32[768]", primals_121: "f32[3072, 768]", primals_122: "f32[3072]", primals_123: "f32[768, 3072]", primals_124: "f32[768]", primals_125: "f32[768]", primals_126: "f32[768]", primals_127: "f32[2304, 768]", primals_128: "f32[2304]", primals_129: "f32[768, 768]", primals_130: "f32[768]", primals_131: "f32[768]", primals_132: "f32[768]", primals_133: "f32[3072, 768]", primals_134: "f32[3072]", primals_135: "f32[768, 3072]", primals_136: "f32[768]", primals_137: "f32[768]", primals_138: "f32[768]", primals_139: "f32[2304, 768]", primals_140: "f32[2304]", primals_141: "f32[768, 768]", primals_142: "f32[768]", primals_143: "f32[768]", primals_144: "f32[768]", primals_145: "f32[3072, 768]", primals_146: "f32[3072]", primals_147: "f32[768, 3072]", primals_148: "f32[768]", primals_149: "f32[768]", primals_150: "f32[768]", primals_151: "f32[1000, 768]", primals_152: "f32[1000]", primals_153: "f32[8, 3, 224, 224]"):
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
    sub: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(clone, getitem_1);  getitem_1 = None
    mul: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul, primals_5)
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
    add_3: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(clone, clone_1);  clone = clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_14: "f32[8, 197, 1]" = var_mean_1[0]
    getitem_15: "f32[8, 197, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_14, 1e-06);  getitem_14 = None
    rsqrt_1: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_1: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_3, getitem_15);  getitem_15 = None
    mul_2: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_3: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_2, primals_11)
    add_5: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_3, primals_12);  mul_3 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_7: "f32[1576, 768]" = torch.ops.aten.view.default(add_5, [1576, 768]);  add_5 = None
    permute_5: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_2: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_14, view_7, permute_5);  primals_14 = None
    view_8: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_2, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_4: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_8, 0.5)
    mul_5: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476);  view_8 = None
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
    add_7: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_3, clone_3);  add_3 = clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_16: "f32[8, 197, 1]" = var_mean_2[0]
    getitem_17: "f32[8, 197, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-06);  getitem_16 = None
    rsqrt_2: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_2: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_7, getitem_17);  getitem_17 = None
    mul_7: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_8: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_7, primals_17)
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
    add_10: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_7, clone_4);  add_7 = clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_30: "f32[8, 197, 1]" = var_mean_3[0]
    getitem_31: "f32[8, 197, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-06);  getitem_30 = None
    rsqrt_3: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_3: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_10, getitem_31);  getitem_31 = None
    mul_9: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_10: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_9, primals_23)
    add_12: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_10, primals_24);  mul_10 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_17: "f32[1576, 768]" = torch.ops.aten.view.default(add_12, [1576, 768]);  add_12 = None
    permute_11: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_6: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_26, view_17, permute_11);  primals_26 = None
    view_18: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_6, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_11: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_12: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
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
    add_14: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_10, clone_6);  add_10 = clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_32: "f32[8, 197, 1]" = var_mean_4[0]
    getitem_33: "f32[8, 197, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_32, 1e-06);  getitem_32 = None
    rsqrt_4: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_4: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_14, getitem_33);  getitem_33 = None
    mul_14: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_15: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_14, primals_29)
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
    add_17: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_14, clone_7);  add_14 = clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_46: "f32[8, 197, 1]" = var_mean_5[0]
    getitem_47: "f32[8, 197, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-06);  getitem_46 = None
    rsqrt_5: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_5: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_17, getitem_47);  getitem_47 = None
    mul_16: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_17: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_16, primals_35)
    add_19: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_17, primals_36);  mul_17 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_27: "f32[1576, 768]" = torch.ops.aten.view.default(add_19, [1576, 768]);  add_19 = None
    permute_17: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_10: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_38, view_27, permute_17);  primals_38 = None
    view_28: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_10, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.5)
    mul_19: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476);  view_28 = None
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
    add_21: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_17, clone_9);  add_17 = clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_6 = torch.ops.aten.var_mean.correction(add_21, [2], correction = 0, keepdim = True)
    getitem_48: "f32[8, 197, 1]" = var_mean_6[0]
    getitem_49: "f32[8, 197, 1]" = var_mean_6[1];  var_mean_6 = None
    add_22: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-06);  getitem_48 = None
    rsqrt_6: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_22);  add_22 = None
    sub_6: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_21, getitem_49);  getitem_49 = None
    mul_21: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_22: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_21, primals_41)
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
    add_24: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_21, clone_10);  add_21 = clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_7 = torch.ops.aten.var_mean.correction(add_24, [2], correction = 0, keepdim = True)
    getitem_62: "f32[8, 197, 1]" = var_mean_7[0]
    getitem_63: "f32[8, 197, 1]" = var_mean_7[1];  var_mean_7 = None
    add_25: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_62, 1e-06);  getitem_62 = None
    rsqrt_7: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    sub_7: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_24, getitem_63);  getitem_63 = None
    mul_23: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_24: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_23, primals_47)
    add_26: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_24, primals_48);  mul_24 = primals_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_37: "f32[1576, 768]" = torch.ops.aten.view.default(add_26, [1576, 768]);  add_26 = None
    permute_23: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    addmm_14: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_50, view_37, permute_23);  primals_50 = None
    view_38: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_14, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_25: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.5)
    mul_26: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476);  view_38 = None
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
    add_28: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_24, clone_12);  add_24 = clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_8 = torch.ops.aten.var_mean.correction(add_28, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 197, 1]" = var_mean_8[0]
    getitem_65: "f32[8, 197, 1]" = var_mean_8[1];  var_mean_8 = None
    add_29: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_8: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_29);  add_29 = None
    sub_8: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_28, getitem_65);  getitem_65 = None
    mul_28: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_29: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_28, primals_53)
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
    add_31: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_28, clone_13);  add_28 = clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_9 = torch.ops.aten.var_mean.correction(add_31, [2], correction = 0, keepdim = True)
    getitem_78: "f32[8, 197, 1]" = var_mean_9[0]
    getitem_79: "f32[8, 197, 1]" = var_mean_9[1];  var_mean_9 = None
    add_32: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-06);  getitem_78 = None
    rsqrt_9: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    sub_9: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_31, getitem_79);  getitem_79 = None
    mul_30: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_31: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_30, primals_59)
    add_33: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_31, primals_60);  mul_31 = primals_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_47: "f32[1576, 768]" = torch.ops.aten.view.default(add_33, [1576, 768]);  add_33 = None
    permute_29: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    addmm_18: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_62, view_47, permute_29);  primals_62 = None
    view_48: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_18, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_48, 0.5)
    mul_33: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_48, 0.7071067811865476);  view_48 = None
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
    add_35: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_31, clone_15);  add_31 = clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_10 = torch.ops.aten.var_mean.correction(add_35, [2], correction = 0, keepdim = True)
    getitem_80: "f32[8, 197, 1]" = var_mean_10[0]
    getitem_81: "f32[8, 197, 1]" = var_mean_10[1];  var_mean_10 = None
    add_36: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-06);  getitem_80 = None
    rsqrt_10: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_36);  add_36 = None
    sub_10: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_35, getitem_81);  getitem_81 = None
    mul_35: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_36: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_35, primals_65)
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
    add_38: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_35, clone_16);  add_35 = clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_11 = torch.ops.aten.var_mean.correction(add_38, [2], correction = 0, keepdim = True)
    getitem_94: "f32[8, 197, 1]" = var_mean_11[0]
    getitem_95: "f32[8, 197, 1]" = var_mean_11[1];  var_mean_11 = None
    add_39: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_94, 1e-06);  getitem_94 = None
    rsqrt_11: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    sub_11: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_38, getitem_95);  getitem_95 = None
    mul_37: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_38: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_37, primals_71)
    add_40: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_38, primals_72);  mul_38 = primals_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_57: "f32[1576, 768]" = torch.ops.aten.view.default(add_40, [1576, 768]);  add_40 = None
    permute_35: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_22: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_74, view_57, permute_35);  primals_74 = None
    view_58: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_22, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_39: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
    mul_40: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_58, 0.7071067811865476);  view_58 = None
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
    add_42: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_38, clone_18);  add_38 = clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_12 = torch.ops.aten.var_mean.correction(add_42, [2], correction = 0, keepdim = True)
    getitem_96: "f32[8, 197, 1]" = var_mean_12[0]
    getitem_97: "f32[8, 197, 1]" = var_mean_12[1];  var_mean_12 = None
    add_43: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-06);  getitem_96 = None
    rsqrt_12: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_43);  add_43 = None
    sub_12: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_42, getitem_97);  getitem_97 = None
    mul_42: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_43: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_42, primals_77)
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
    add_45: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_42, clone_19);  add_42 = clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_13 = torch.ops.aten.var_mean.correction(add_45, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 197, 1]" = var_mean_13[0]
    getitem_111: "f32[8, 197, 1]" = var_mean_13[1];  var_mean_13 = None
    add_46: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_13: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    sub_13: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_45, getitem_111);  getitem_111 = None
    mul_44: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_45: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_44, primals_83)
    add_47: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_45, primals_84);  mul_45 = primals_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_67: "f32[1576, 768]" = torch.ops.aten.view.default(add_47, [1576, 768]);  add_47 = None
    permute_41: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_26: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_86, view_67, permute_41);  primals_86 = None
    view_68: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_26, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_68, 0.5)
    mul_47: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_68, 0.7071067811865476);  view_68 = None
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
    add_49: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_45, clone_21);  add_45 = clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_14 = torch.ops.aten.var_mean.correction(add_49, [2], correction = 0, keepdim = True)
    getitem_112: "f32[8, 197, 1]" = var_mean_14[0]
    getitem_113: "f32[8, 197, 1]" = var_mean_14[1];  var_mean_14 = None
    add_50: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_112, 1e-06);  getitem_112 = None
    rsqrt_14: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_50);  add_50 = None
    sub_14: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_49, getitem_113);  getitem_113 = None
    mul_49: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_50: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_49, primals_89)
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
    add_52: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_49, clone_22);  add_49 = clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_15 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
    getitem_126: "f32[8, 197, 1]" = var_mean_15[0]
    getitem_127: "f32[8, 197, 1]" = var_mean_15[1];  var_mean_15 = None
    add_53: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-06);  getitem_126 = None
    rsqrt_15: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    sub_15: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_52, getitem_127);  getitem_127 = None
    mul_51: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_52: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_51, primals_95)
    add_54: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_52, primals_96);  mul_52 = primals_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_77: "f32[1576, 768]" = torch.ops.aten.view.default(add_54, [1576, 768]);  add_54 = None
    permute_47: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_30: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_98, view_77, permute_47);  primals_98 = None
    view_78: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_30, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_53: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.5)
    mul_54: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_78, 0.7071067811865476);  view_78 = None
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
    add_56: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_52, clone_24);  add_52 = clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_16 = torch.ops.aten.var_mean.correction(add_56, [2], correction = 0, keepdim = True)
    getitem_128: "f32[8, 197, 1]" = var_mean_16[0]
    getitem_129: "f32[8, 197, 1]" = var_mean_16[1];  var_mean_16 = None
    add_57: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-06);  getitem_128 = None
    rsqrt_16: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_57);  add_57 = None
    sub_16: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_56, getitem_129);  getitem_129 = None
    mul_56: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_57: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_56, primals_101)
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
    add_59: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_56, clone_25);  add_56 = clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_17 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
    getitem_142: "f32[8, 197, 1]" = var_mean_17[0]
    getitem_143: "f32[8, 197, 1]" = var_mean_17[1];  var_mean_17 = None
    add_60: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_142, 1e-06);  getitem_142 = None
    rsqrt_17: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    sub_17: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_59, getitem_143);  getitem_143 = None
    mul_58: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_59: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_58, primals_107)
    add_61: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_59, primals_108);  mul_59 = primals_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_87: "f32[1576, 768]" = torch.ops.aten.view.default(add_61, [1576, 768]);  add_61 = None
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    addmm_34: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_110, view_87, permute_53);  primals_110 = None
    view_88: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_34, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_60: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.5)
    mul_61: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_88, 0.7071067811865476);  view_88 = None
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
    add_63: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_59, clone_27);  add_59 = clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_18 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
    getitem_144: "f32[8, 197, 1]" = var_mean_18[0]
    getitem_145: "f32[8, 197, 1]" = var_mean_18[1];  var_mean_18 = None
    add_64: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_144, 1e-06);  getitem_144 = None
    rsqrt_18: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
    sub_18: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_63, getitem_145);  getitem_145 = None
    mul_63: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_64: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_63, primals_113)
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
    add_66: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_63, clone_28);  add_63 = clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
    getitem_158: "f32[8, 197, 1]" = var_mean_19[0]
    getitem_159: "f32[8, 197, 1]" = var_mean_19[1];  var_mean_19 = None
    add_67: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-06);  getitem_158 = None
    rsqrt_19: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    sub_19: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_66, getitem_159);  getitem_159 = None
    mul_65: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_66: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_65, primals_119)
    add_68: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_66, primals_120);  mul_66 = primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_97: "f32[1576, 768]" = torch.ops.aten.view.default(add_68, [1576, 768]);  add_68 = None
    permute_59: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_38: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_122, view_97, permute_59);  primals_122 = None
    view_98: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_38, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.5)
    mul_68: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_98, 0.7071067811865476);  view_98 = None
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
    add_70: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_66, clone_30);  add_66 = clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_20 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
    getitem_160: "f32[8, 197, 1]" = var_mean_20[0]
    getitem_161: "f32[8, 197, 1]" = var_mean_20[1];  var_mean_20 = None
    add_71: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-06);  getitem_160 = None
    rsqrt_20: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    sub_20: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_70, getitem_161);  getitem_161 = None
    mul_70: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_71: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_70, primals_125)
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
    add_73: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_70, clone_31);  add_70 = clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_21 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
    getitem_174: "f32[8, 197, 1]" = var_mean_21[0]
    getitem_175: "f32[8, 197, 1]" = var_mean_21[1];  var_mean_21 = None
    add_74: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_174, 1e-06);  getitem_174 = None
    rsqrt_21: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
    sub_21: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_73, getitem_175);  getitem_175 = None
    mul_72: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_73: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_72, primals_131)
    add_75: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_73, primals_132);  mul_73 = primals_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_107: "f32[1576, 768]" = torch.ops.aten.view.default(add_75, [1576, 768]);  add_75 = None
    permute_65: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    addmm_42: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_134, view_107, permute_65);  primals_134 = None
    view_108: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_42, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_74: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_108, 0.5)
    mul_75: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_108, 0.7071067811865476);  view_108 = None
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
    add_77: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_73, clone_33);  add_73 = clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_22 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
    getitem_176: "f32[8, 197, 1]" = var_mean_22[0]
    getitem_177: "f32[8, 197, 1]" = var_mean_22[1];  var_mean_22 = None
    add_78: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-06);  getitem_176 = None
    rsqrt_22: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    sub_22: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_77, getitem_177);  getitem_177 = None
    mul_77: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_78: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_77, primals_137)
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
    add_80: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_77, clone_34);  add_77 = clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_23 = torch.ops.aten.var_mean.correction(add_80, [2], correction = 0, keepdim = True)
    getitem_190: "f32[8, 197, 1]" = var_mean_23[0]
    getitem_191: "f32[8, 197, 1]" = var_mean_23[1];  var_mean_23 = None
    add_81: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-06);  getitem_190 = None
    rsqrt_23: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    sub_23: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_80, getitem_191);  getitem_191 = None
    mul_79: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_80: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_79, primals_143)
    add_82: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(mul_80, primals_144);  mul_80 = primals_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_117: "f32[1576, 768]" = torch.ops.aten.view.default(add_82, [1576, 768]);  add_82 = None
    permute_71: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_46: "f32[1576, 3072]" = torch.ops.aten.addmm.default(primals_146, view_117, permute_71);  primals_146 = None
    view_118: "f32[8, 197, 3072]" = torch.ops.aten.view.default(addmm_46, [8, 197, 3072])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_81: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
    mul_82: "f32[8, 197, 3072]" = torch.ops.aten.mul.Tensor(view_118, 0.7071067811865476);  view_118 = None
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
    add_84: "f32[8, 197, 768]" = torch.ops.aten.add.Tensor(add_80, clone_36);  add_80 = clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:641, code: x = self.norm(x)
    var_mean_24 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
    getitem_192: "f32[8, 197, 1]" = var_mean_24[0]
    getitem_193: "f32[8, 197, 1]" = var_mean_24[1];  var_mean_24 = None
    add_85: "f32[8, 197, 1]" = torch.ops.aten.add.Tensor(getitem_192, 1e-06);  getitem_192 = None
    rsqrt_24: "f32[8, 197, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    sub_24: "f32[8, 197, 768]" = torch.ops.aten.sub.Tensor(add_84, getitem_193);  add_84 = getitem_193 = None
    mul_84: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_85: "f32[8, 197, 768]" = torch.ops.aten.mul.Tensor(mul_84, primals_149)
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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:641, code: x = self.norm(x)
    div: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 768);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_78: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_82: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_1: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 768);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_86: "f32[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_12: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_92: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_2: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 768);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_96: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_100: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_3: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 768);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_104: "f32[768, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_13: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_110: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_4: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 768);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_114: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_118: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_5: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 768);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_122: "f32[768, 768]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_14: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_128: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_6: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 768);  rsqrt_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_132: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_136: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_7: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 768);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_140: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_15: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_146: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_8: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 768);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_150: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_154: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_9: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 768);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_16: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_164: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_10: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 768);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_168: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_172: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_11: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 768);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_176: "f32[768, 768]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_17: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_182: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_12: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 768);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_186: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_190: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_13: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 768);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_18: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_200: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_14: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 768);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_204: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_208: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_15: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 768);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_212: "f32[768, 768]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_19: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_218: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_16: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 768);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_222: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_226: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_17: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 768);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_20: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_236: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_18: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 768);  rsqrt_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_240: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_244: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_19: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 768);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_248: "f32[768, 768]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_21: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_254: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_20: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 768);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_258: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_262: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_21: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 768);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_266: "f32[768, 768]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_22: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_272: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_22: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 768);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_276: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_280: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_23: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_284: "f32[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_23: "f32[8, 12, 197, 64]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_290: "f32[2304, 768]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_24: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
    return [addmm_48, primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_47, primals_53, primals_59, primals_65, primals_71, primals_77, primals_83, primals_89, primals_95, primals_101, primals_107, primals_113, primals_119, primals_125, primals_131, primals_137, primals_143, primals_149, primals_153, mul, view_1, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, getitem_11, getitem_12, view_5, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_18, getitem_19, getitem_20, getitem_22, getitem_23, getitem_24, getitem_27, getitem_28, view_15, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_34, getitem_35, getitem_36, getitem_38, getitem_39, getitem_40, getitem_43, getitem_44, view_25, mul_16, view_27, addmm_10, view_29, mul_21, view_31, getitem_50, getitem_51, getitem_52, getitem_54, getitem_55, getitem_56, getitem_59, getitem_60, view_35, mul_23, view_37, addmm_14, view_39, mul_28, view_41, getitem_66, getitem_67, getitem_68, getitem_70, getitem_71, getitem_72, getitem_75, getitem_76, view_45, mul_30, view_47, addmm_18, view_49, mul_35, view_51, getitem_82, getitem_83, getitem_84, getitem_86, getitem_87, getitem_88, getitem_91, getitem_92, view_55, mul_37, view_57, addmm_22, view_59, mul_42, view_61, getitem_98, getitem_99, getitem_100, getitem_102, getitem_103, getitem_104, getitem_107, getitem_108, view_65, mul_44, view_67, addmm_26, view_69, mul_49, view_71, getitem_114, getitem_115, getitem_116, getitem_118, getitem_119, getitem_120, getitem_123, getitem_124, view_75, mul_51, view_77, addmm_30, view_79, mul_56, view_81, getitem_130, getitem_131, getitem_132, getitem_134, getitem_135, getitem_136, getitem_139, getitem_140, view_85, mul_58, view_87, addmm_34, view_89, mul_63, view_91, getitem_146, getitem_147, getitem_148, getitem_150, getitem_151, getitem_152, getitem_155, getitem_156, view_95, mul_65, view_97, addmm_38, view_99, mul_70, view_101, getitem_162, getitem_163, getitem_164, getitem_166, getitem_167, getitem_168, getitem_171, getitem_172, view_105, mul_72, view_107, addmm_42, view_109, mul_77, view_111, getitem_178, getitem_179, getitem_180, getitem_182, getitem_183, getitem_184, getitem_187, getitem_188, view_115, mul_79, view_117, addmm_46, view_119, mul_84, clone_37, permute_74, div, permute_78, permute_82, div_1, permute_86, alias_12, permute_92, div_2, permute_96, permute_100, div_3, permute_104, alias_13, permute_110, div_4, permute_114, permute_118, div_5, permute_122, alias_14, permute_128, div_6, permute_132, permute_136, div_7, permute_140, alias_15, permute_146, div_8, permute_150, permute_154, div_9, permute_158, alias_16, permute_164, div_10, permute_168, permute_172, div_11, permute_176, alias_17, permute_182, div_12, permute_186, permute_190, div_13, permute_194, alias_18, permute_200, div_14, permute_204, permute_208, div_15, permute_212, alias_19, permute_218, div_16, permute_222, permute_226, div_17, permute_230, alias_20, permute_236, div_18, permute_240, permute_244, div_19, permute_248, alias_21, permute_254, div_20, permute_258, permute_262, div_21, permute_266, alias_22, permute_272, div_22, permute_276, permute_280, div_23, permute_284, alias_23, permute_290, div_24]
    