from __future__ import annotations



def forward(self, primals_1: "f32[1, 256, 31, 31]", primals_2: "f32[1, 1, 256]", primals_3: "f32[256, 3, 14, 14]", primals_4: "f32[256]", primals_5: "f32[256]", primals_6: "f32[256]", primals_7: "f32[768, 256]", primals_8: "f32[768]", primals_9: "f32[256, 256]", primals_10: "f32[256]", primals_11: "f32[256]", primals_12: "f32[256]", primals_13: "f32[1024, 256]", primals_14: "f32[1024]", primals_15: "f32[256, 1024]", primals_16: "f32[256]", primals_17: "f32[256]", primals_18: "f32[256]", primals_19: "f32[768, 256]", primals_20: "f32[768]", primals_21: "f32[256, 256]", primals_22: "f32[256]", primals_23: "f32[256]", primals_24: "f32[256]", primals_25: "f32[1024, 256]", primals_26: "f32[1024]", primals_27: "f32[256, 1024]", primals_28: "f32[256]", primals_29: "f32[256]", primals_30: "f32[256]", primals_31: "f32[768, 256]", primals_32: "f32[768]", primals_33: "f32[256, 256]", primals_34: "f32[256]", primals_35: "f32[256]", primals_36: "f32[256]", primals_37: "f32[1024, 256]", primals_38: "f32[1024]", primals_39: "f32[256, 1024]", primals_40: "f32[256]", primals_41: "f32[512, 1, 3, 3]", primals_42: "f32[512]", primals_43: "f32[512, 256]", primals_44: "f32[512]", primals_45: "f32[512]", primals_46: "f32[512]", primals_47: "f32[1536, 512]", primals_48: "f32[1536]", primals_49: "f32[512, 512]", primals_50: "f32[512]", primals_51: "f32[512]", primals_52: "f32[512]", primals_53: "f32[2048, 512]", primals_54: "f32[2048]", primals_55: "f32[512, 2048]", primals_56: "f32[512]", primals_57: "f32[512]", primals_58: "f32[512]", primals_59: "f32[1536, 512]", primals_60: "f32[1536]", primals_61: "f32[512, 512]", primals_62: "f32[512]", primals_63: "f32[512]", primals_64: "f32[512]", primals_65: "f32[2048, 512]", primals_66: "f32[2048]", primals_67: "f32[512, 2048]", primals_68: "f32[512]", primals_69: "f32[512]", primals_70: "f32[512]", primals_71: "f32[1536, 512]", primals_72: "f32[1536]", primals_73: "f32[512, 512]", primals_74: "f32[512]", primals_75: "f32[512]", primals_76: "f32[512]", primals_77: "f32[2048, 512]", primals_78: "f32[2048]", primals_79: "f32[512, 2048]", primals_80: "f32[512]", primals_81: "f32[512]", primals_82: "f32[512]", primals_83: "f32[1536, 512]", primals_84: "f32[1536]", primals_85: "f32[512, 512]", primals_86: "f32[512]", primals_87: "f32[512]", primals_88: "f32[512]", primals_89: "f32[2048, 512]", primals_90: "f32[2048]", primals_91: "f32[512, 2048]", primals_92: "f32[512]", primals_93: "f32[512]", primals_94: "f32[512]", primals_95: "f32[1536, 512]", primals_96: "f32[1536]", primals_97: "f32[512, 512]", primals_98: "f32[512]", primals_99: "f32[512]", primals_100: "f32[512]", primals_101: "f32[2048, 512]", primals_102: "f32[2048]", primals_103: "f32[512, 2048]", primals_104: "f32[512]", primals_105: "f32[512]", primals_106: "f32[512]", primals_107: "f32[1536, 512]", primals_108: "f32[1536]", primals_109: "f32[512, 512]", primals_110: "f32[512]", primals_111: "f32[512]", primals_112: "f32[512]", primals_113: "f32[2048, 512]", primals_114: "f32[2048]", primals_115: "f32[512, 2048]", primals_116: "f32[512]", primals_117: "f32[1024, 1, 3, 3]", primals_118: "f32[1024]", primals_119: "f32[1024, 512]", primals_120: "f32[1024]", primals_121: "f32[1024]", primals_122: "f32[1024]", primals_123: "f32[3072, 1024]", primals_124: "f32[3072]", primals_125: "f32[1024, 1024]", primals_126: "f32[1024]", primals_127: "f32[1024]", primals_128: "f32[1024]", primals_129: "f32[4096, 1024]", primals_130: "f32[4096]", primals_131: "f32[1024, 4096]", primals_132: "f32[1024]", primals_133: "f32[1024]", primals_134: "f32[1024]", primals_135: "f32[3072, 1024]", primals_136: "f32[3072]", primals_137: "f32[1024, 1024]", primals_138: "f32[1024]", primals_139: "f32[1024]", primals_140: "f32[1024]", primals_141: "f32[4096, 1024]", primals_142: "f32[4096]", primals_143: "f32[1024, 4096]", primals_144: "f32[1024]", primals_145: "f32[1024]", primals_146: "f32[1024]", primals_147: "f32[3072, 1024]", primals_148: "f32[3072]", primals_149: "f32[1024, 1024]", primals_150: "f32[1024]", primals_151: "f32[1024]", primals_152: "f32[1024]", primals_153: "f32[4096, 1024]", primals_154: "f32[4096]", primals_155: "f32[1024, 4096]", primals_156: "f32[1024]", primals_157: "f32[1024]", primals_158: "f32[1024]", primals_159: "f32[3072, 1024]", primals_160: "f32[3072]", primals_161: "f32[1024, 1024]", primals_162: "f32[1024]", primals_163: "f32[1024]", primals_164: "f32[1024]", primals_165: "f32[4096, 1024]", primals_166: "f32[4096]", primals_167: "f32[1024, 4096]", primals_168: "f32[1024]", primals_169: "f32[1024]", primals_170: "f32[1024]", primals_171: "f32[1000, 1024]", primals_172: "f32[1000]", primals_173: "f32[8, 3, 224, 224]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:138, code: x = self.conv(x)
    convolution: "f32[8, 256, 31, 31]" = torch.ops.aten.convolution.default(primals_173, primals_3, primals_4, [7, 7], [0, 0], [1, 1], False, [0, 0], 1);  primals_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:257, code: x = self.pos_drop(x + self.pos_embed)
    add: "f32[8, 256, 31, 31]" = torch.ops.aten.add.Tensor(convolution, primals_1);  convolution = primals_1 = None
    clone: "f32[8, 256, 31, 31]" = torch.ops.aten.clone.default(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:258, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    expand: "f32[8, 1, 256]" = torch.ops.aten.expand.default(primals_2, [8, -1, -1]);  primals_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:81, code: x = x.flatten(2).transpose(1, 2)
    view: "f32[8, 256, 961]" = torch.ops.aten.view.default(clone, [8, 256, 961]);  clone = None
    permute: "f32[8, 961, 256]" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:82, code: x = torch.cat((cls_tokens, x), dim=1)
    cat: "f32[8, 962, 256]" = torch.ops.aten.cat.default([expand, permute], 1);  expand = permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean = torch.ops.aten.var_mean.correction(cat, [2], correction = 0, keepdim = True)
    getitem: "f32[8, 962, 1]" = var_mean[0]
    getitem_1: "f32[8, 962, 1]" = var_mean[1];  var_mean = None
    add_1: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-06);  getitem = None
    rsqrt: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    sub: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(cat, getitem_1)
    mul: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
    mul_1: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul, primals_5);  mul = None
    add_2: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_1: "f32[7696, 256]" = torch.ops.aten.view.default(add_2, [7696, 256]);  add_2 = None
    permute_1: "f32[256, 768]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
    addmm: "f32[7696, 768]" = torch.ops.aten.addmm.default(primals_8, view_1, permute_1);  primals_8 = None
    view_2: "f32[8, 962, 768]" = torch.ops.aten.view.default(addmm, [8, 962, 768]);  addmm = None
    view_3: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.view.default(view_2, [8, 962, 3, 4, 64]);  view_2 = None
    permute_2: "f32[3, 8, 4, 962, 64]" = torch.ops.aten.permute.default(view_3, [2, 0, 3, 1, 4]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = torch.ops.aten.unbind.int(permute_2);  permute_2 = None
    getitem_2: "f32[8, 4, 962, 64]" = unbind[0]
    getitem_3: "f32[8, 4, 962, 64]" = unbind[1]
    getitem_4: "f32[8, 4, 962, 64]" = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_2, getitem_3, getitem_4, None, True)
    getitem_5: "f32[8, 4, 962, 64]" = _scaled_dot_product_efficient_attention[0]
    getitem_6: "f32[8, 4, 992]" = _scaled_dot_product_efficient_attention[1]
    getitem_7: "i64[]" = _scaled_dot_product_efficient_attention[2]
    getitem_8: "i64[]" = _scaled_dot_product_efficient_attention[3];  _scaled_dot_product_efficient_attention = None
    alias: "f32[8, 4, 962, 64]" = torch.ops.aten.alias.default(getitem_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_3: "f32[8, 962, 4, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3]);  getitem_5 = None
    view_4: "f32[8, 962, 256]" = torch.ops.aten.view.default(permute_3, [8, 962, 256]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_5: "f32[7696, 256]" = torch.ops.aten.view.default(view_4, [7696, 256]);  view_4 = None
    permute_4: "f32[256, 256]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
    addmm_1: "f32[7696, 256]" = torch.ops.aten.addmm.default(primals_10, view_5, permute_4);  primals_10 = None
    view_6: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_1, [8, 962, 256]);  addmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_1: "f32[8, 962, 256]" = torch.ops.aten.clone.default(view_6);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_3: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(cat, clone_1);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_1 = torch.ops.aten.var_mean.correction(add_3, [2], correction = 0, keepdim = True)
    getitem_9: "f32[8, 962, 1]" = var_mean_1[0]
    getitem_10: "f32[8, 962, 1]" = var_mean_1[1];  var_mean_1 = None
    add_4: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_9, 1e-06);  getitem_9 = None
    rsqrt_1: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    sub_1: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_3, getitem_10);  getitem_10 = None
    mul_2: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    mul_3: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_2, primals_11)
    add_5: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_3, primals_12);  mul_3 = primals_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_7: "f32[7696, 256]" = torch.ops.aten.view.default(add_5, [7696, 256]);  add_5 = None
    permute_5: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
    addmm_2: "f32[7696, 1024]" = torch.ops.aten.addmm.default(primals_14, view_7, permute_5);  primals_14 = None
    view_8: "f32[8, 962, 1024]" = torch.ops.aten.view.default(addmm_2, [8, 962, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_4: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_8, 0.5)
    mul_5: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_8, 0.7071067811865476);  view_8 = None
    erf: "f32[8, 962, 1024]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
    add_6: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_6: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(mul_4, add_6);  mul_4 = add_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_2: "f32[8, 962, 1024]" = torch.ops.aten.clone.default(mul_6);  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_9: "f32[7696, 1024]" = torch.ops.aten.view.default(clone_2, [7696, 1024]);  clone_2 = None
    permute_6: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
    addmm_3: "f32[7696, 256]" = torch.ops.aten.addmm.default(primals_16, view_9, permute_6);  primals_16 = None
    view_10: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_3, [8, 962, 256]);  addmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_3: "f32[8, 962, 256]" = torch.ops.aten.clone.default(view_10);  view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_7: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_3, clone_3);  add_3 = clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_2 = torch.ops.aten.var_mean.correction(add_7, [2], correction = 0, keepdim = True)
    getitem_11: "f32[8, 962, 1]" = var_mean_2[0]
    getitem_12: "f32[8, 962, 1]" = var_mean_2[1];  var_mean_2 = None
    add_8: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_11, 1e-06);  getitem_11 = None
    rsqrt_2: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_8);  add_8 = None
    sub_2: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_7, getitem_12);  getitem_12 = None
    mul_7: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_2);  sub_2 = None
    mul_8: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_7, primals_17)
    add_9: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_8, primals_18);  mul_8 = primals_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_11: "f32[7696, 256]" = torch.ops.aten.view.default(add_9, [7696, 256]);  add_9 = None
    permute_7: "f32[256, 768]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
    addmm_4: "f32[7696, 768]" = torch.ops.aten.addmm.default(primals_20, view_11, permute_7);  primals_20 = None
    view_12: "f32[8, 962, 768]" = torch.ops.aten.view.default(addmm_4, [8, 962, 768]);  addmm_4 = None
    view_13: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.view.default(view_12, [8, 962, 3, 4, 64]);  view_12 = None
    permute_8: "f32[3, 8, 4, 962, 64]" = torch.ops.aten.permute.default(view_13, [2, 0, 3, 1, 4]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = torch.ops.aten.unbind.int(permute_8);  permute_8 = None
    getitem_13: "f32[8, 4, 962, 64]" = unbind_1[0]
    getitem_14: "f32[8, 4, 962, 64]" = unbind_1[1]
    getitem_15: "f32[8, 4, 962, 64]" = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_13, getitem_14, getitem_15, None, True)
    getitem_16: "f32[8, 4, 962, 64]" = _scaled_dot_product_efficient_attention_1[0]
    getitem_17: "f32[8, 4, 992]" = _scaled_dot_product_efficient_attention_1[1]
    getitem_18: "i64[]" = _scaled_dot_product_efficient_attention_1[2]
    getitem_19: "i64[]" = _scaled_dot_product_efficient_attention_1[3];  _scaled_dot_product_efficient_attention_1 = None
    alias_1: "f32[8, 4, 962, 64]" = torch.ops.aten.alias.default(getitem_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_9: "f32[8, 962, 4, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3]);  getitem_16 = None
    view_14: "f32[8, 962, 256]" = torch.ops.aten.view.default(permute_9, [8, 962, 256]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_15: "f32[7696, 256]" = torch.ops.aten.view.default(view_14, [7696, 256]);  view_14 = None
    permute_10: "f32[256, 256]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
    addmm_5: "f32[7696, 256]" = torch.ops.aten.addmm.default(primals_22, view_15, permute_10);  primals_22 = None
    view_16: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_5, [8, 962, 256]);  addmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_4: "f32[8, 962, 256]" = torch.ops.aten.clone.default(view_16);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_10: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_7, clone_4);  add_7 = clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_3 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
    getitem_20: "f32[8, 962, 1]" = var_mean_3[0]
    getitem_21: "f32[8, 962, 1]" = var_mean_3[1];  var_mean_3 = None
    add_11: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-06);  getitem_20 = None
    rsqrt_3: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    sub_3: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_10, getitem_21);  getitem_21 = None
    mul_9: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_3);  sub_3 = None
    mul_10: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_9, primals_23)
    add_12: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_10, primals_24);  mul_10 = primals_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_17: "f32[7696, 256]" = torch.ops.aten.view.default(add_12, [7696, 256]);  add_12 = None
    permute_11: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
    addmm_6: "f32[7696, 1024]" = torch.ops.aten.addmm.default(primals_26, view_17, permute_11);  primals_26 = None
    view_18: "f32[8, 962, 1024]" = torch.ops.aten.view.default(addmm_6, [8, 962, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_11: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_18, 0.5)
    mul_12: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476);  view_18 = None
    erf_1: "f32[8, 962, 1024]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_13: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_13: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(mul_11, add_13);  mul_11 = add_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_5: "f32[8, 962, 1024]" = torch.ops.aten.clone.default(mul_13);  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_19: "f32[7696, 1024]" = torch.ops.aten.view.default(clone_5, [7696, 1024]);  clone_5 = None
    permute_12: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
    addmm_7: "f32[7696, 256]" = torch.ops.aten.addmm.default(primals_28, view_19, permute_12);  primals_28 = None
    view_20: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_7, [8, 962, 256]);  addmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_6: "f32[8, 962, 256]" = torch.ops.aten.clone.default(view_20);  view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_14: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_10, clone_6);  add_10 = clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_4 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
    getitem_22: "f32[8, 962, 1]" = var_mean_4[0]
    getitem_23: "f32[8, 962, 1]" = var_mean_4[1];  var_mean_4 = None
    add_15: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_22, 1e-06);  getitem_22 = None
    rsqrt_4: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
    sub_4: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_14, getitem_23);  getitem_23 = None
    mul_14: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_4);  sub_4 = None
    mul_15: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_14, primals_29)
    add_16: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_15, primals_30);  mul_15 = primals_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_21: "f32[7696, 256]" = torch.ops.aten.view.default(add_16, [7696, 256]);  add_16 = None
    permute_13: "f32[256, 768]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
    addmm_8: "f32[7696, 768]" = torch.ops.aten.addmm.default(primals_32, view_21, permute_13);  primals_32 = None
    view_22: "f32[8, 962, 768]" = torch.ops.aten.view.default(addmm_8, [8, 962, 768]);  addmm_8 = None
    view_23: "f32[8, 962, 3, 4, 64]" = torch.ops.aten.view.default(view_22, [8, 962, 3, 4, 64]);  view_22 = None
    permute_14: "f32[3, 8, 4, 962, 64]" = torch.ops.aten.permute.default(view_23, [2, 0, 3, 1, 4]);  view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = torch.ops.aten.unbind.int(permute_14);  permute_14 = None
    getitem_24: "f32[8, 4, 962, 64]" = unbind_2[0]
    getitem_25: "f32[8, 4, 962, 64]" = unbind_2[1]
    getitem_26: "f32[8, 4, 962, 64]" = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_24, getitem_25, getitem_26, None, True)
    getitem_27: "f32[8, 4, 962, 64]" = _scaled_dot_product_efficient_attention_2[0]
    getitem_28: "f32[8, 4, 992]" = _scaled_dot_product_efficient_attention_2[1]
    getitem_29: "i64[]" = _scaled_dot_product_efficient_attention_2[2]
    getitem_30: "i64[]" = _scaled_dot_product_efficient_attention_2[3];  _scaled_dot_product_efficient_attention_2 = None
    alias_2: "f32[8, 4, 962, 64]" = torch.ops.aten.alias.default(getitem_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_15: "f32[8, 962, 4, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3]);  getitem_27 = None
    view_24: "f32[8, 962, 256]" = torch.ops.aten.view.default(permute_15, [8, 962, 256]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_25: "f32[7696, 256]" = torch.ops.aten.view.default(view_24, [7696, 256]);  view_24 = None
    permute_16: "f32[256, 256]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
    addmm_9: "f32[7696, 256]" = torch.ops.aten.addmm.default(primals_34, view_25, permute_16);  primals_34 = None
    view_26: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_9, [8, 962, 256]);  addmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_7: "f32[8, 962, 256]" = torch.ops.aten.clone.default(view_26);  view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_17: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_14, clone_7);  add_14 = clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_5 = torch.ops.aten.var_mean.correction(add_17, [2], correction = 0, keepdim = True)
    getitem_31: "f32[8, 962, 1]" = var_mean_5[0]
    getitem_32: "f32[8, 962, 1]" = var_mean_5[1];  var_mean_5 = None
    add_18: "f32[8, 962, 1]" = torch.ops.aten.add.Tensor(getitem_31, 1e-06);  getitem_31 = None
    rsqrt_5: "f32[8, 962, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    sub_5: "f32[8, 962, 256]" = torch.ops.aten.sub.Tensor(add_17, getitem_32);  getitem_32 = None
    mul_16: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_5);  sub_5 = None
    mul_17: "f32[8, 962, 256]" = torch.ops.aten.mul.Tensor(mul_16, primals_35)
    add_19: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(mul_17, primals_36);  mul_17 = primals_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_27: "f32[7696, 256]" = torch.ops.aten.view.default(add_19, [7696, 256]);  add_19 = None
    permute_17: "f32[256, 1024]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
    addmm_10: "f32[7696, 1024]" = torch.ops.aten.addmm.default(primals_38, view_27, permute_17);  primals_38 = None
    view_28: "f32[8, 962, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 962, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_28, 0.5)
    mul_19: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(view_28, 0.7071067811865476);  view_28 = None
    erf_2: "f32[8, 962, 1024]" = torch.ops.aten.erf.default(mul_19);  mul_19 = None
    add_20: "f32[8, 962, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_20: "f32[8, 962, 1024]" = torch.ops.aten.mul.Tensor(mul_18, add_20);  mul_18 = add_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_8: "f32[8, 962, 1024]" = torch.ops.aten.clone.default(mul_20);  mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_29: "f32[7696, 1024]" = torch.ops.aten.view.default(clone_8, [7696, 1024]);  clone_8 = None
    permute_18: "f32[1024, 256]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
    addmm_11: "f32[7696, 256]" = torch.ops.aten.addmm.default(primals_40, view_29, permute_18);  primals_40 = None
    view_30: "f32[8, 962, 256]" = torch.ops.aten.view.default(addmm_11, [8, 962, 256]);  addmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_9: "f32[8, 962, 256]" = torch.ops.aten.clone.default(view_30);  view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_21: "f32[8, 962, 256]" = torch.ops.aten.add.Tensor(add_17, clone_9);  add_17 = clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    slice_1: "f32[8, 962, 256]" = torch.ops.aten.slice.Tensor(add_21, 0, 0, 9223372036854775807);  add_21 = None
    slice_2: "f32[8, 1, 256]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:88, code: x = x[:, token_length:]
    slice_4: "f32[8, 961, 256]" = torch.ops.aten.slice.Tensor(slice_1, 1, 1, 9223372036854775807);  slice_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:89, code: x = x.transpose(1, 2).reshape(B, C, H, W)
    permute_19: "f32[8, 256, 961]" = torch.ops.aten.permute.default(slice_4, [0, 2, 1]);  slice_4 = None
    view_31: "f32[8, 256, 31, 31]" = torch.ops.aten.view.default(permute_19, [8, 256, 31, 31]);  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:110, code: x = self.conv(x)
    convolution_1: "f32[8, 512, 16, 16]" = torch.ops.aten.convolution.default(view_31, primals_41, primals_42, [2, 2], [1, 1], [1, 1], False, [0, 0], 256);  primals_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:111, code: cls_token = self.fc(cls_token)
    permute_20: "f32[256, 512]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
    view_32: "f32[8, 256]" = torch.ops.aten.view.default(slice_2, [8, 256]);  slice_2 = None
    mm: "f32[8, 512]" = torch.ops.aten.mm.default(view_32, permute_20)
    view_33: "f32[8, 1, 512]" = torch.ops.aten.view.default(mm, [8, 1, 512]);  mm = None
    add_22: "f32[8, 1, 512]" = torch.ops.aten.add.Tensor(view_33, primals_44);  view_33 = primals_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:81, code: x = x.flatten(2).transpose(1, 2)
    view_34: "f32[8, 512, 256]" = torch.ops.aten.view.default(convolution_1, [8, 512, 256]);  convolution_1 = None
    permute_21: "f32[8, 256, 512]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:82, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_1: "f32[8, 257, 512]" = torch.ops.aten.cat.default([add_22, permute_21], 1);  add_22 = permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_6 = torch.ops.aten.var_mean.correction(cat_1, [2], correction = 0, keepdim = True)
    getitem_33: "f32[8, 257, 1]" = var_mean_6[0]
    getitem_34: "f32[8, 257, 1]" = var_mean_6[1];  var_mean_6 = None
    add_23: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_33, 1e-06);  getitem_33 = None
    rsqrt_6: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_23);  add_23 = None
    sub_6: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(cat_1, getitem_34)
    mul_21: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_6);  sub_6 = None
    mul_22: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_21, primals_45);  mul_21 = None
    add_24: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_22, primals_46);  mul_22 = primals_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_35: "f32[2056, 512]" = torch.ops.aten.view.default(add_24, [2056, 512]);  add_24 = None
    permute_22: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
    addmm_12: "f32[2056, 1536]" = torch.ops.aten.addmm.default(primals_48, view_35, permute_22);  primals_48 = None
    view_36: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_12, [8, 257, 1536]);  addmm_12 = None
    view_37: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_36, [8, 257, 3, 8, 64]);  view_36 = None
    permute_23: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_37, [2, 0, 3, 1, 4]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = torch.ops.aten.unbind.int(permute_23);  permute_23 = None
    getitem_35: "f32[8, 8, 257, 64]" = unbind_3[0]
    getitem_36: "f32[8, 8, 257, 64]" = unbind_3[1]
    getitem_37: "f32[8, 8, 257, 64]" = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_35, getitem_36, getitem_37, None, True)
    getitem_38: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_3[0]
    getitem_39: "f32[8, 8, 288]" = _scaled_dot_product_efficient_attention_3[1]
    getitem_40: "i64[]" = _scaled_dot_product_efficient_attention_3[2]
    getitem_41: "i64[]" = _scaled_dot_product_efficient_attention_3[3];  _scaled_dot_product_efficient_attention_3 = None
    alias_3: "f32[8, 8, 257, 64]" = torch.ops.aten.alias.default(getitem_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_24: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3]);  getitem_38 = None
    view_38: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_24, [8, 257, 512]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_39: "f32[2056, 512]" = torch.ops.aten.view.default(view_38, [2056, 512]);  view_38 = None
    permute_25: "f32[512, 512]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    addmm_13: "f32[2056, 512]" = torch.ops.aten.addmm.default(primals_50, view_39, permute_25);  primals_50 = None
    view_40: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_13, [8, 257, 512]);  addmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_10: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_40);  view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_25: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(cat_1, clone_10);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_7 = torch.ops.aten.var_mean.correction(add_25, [2], correction = 0, keepdim = True)
    getitem_42: "f32[8, 257, 1]" = var_mean_7[0]
    getitem_43: "f32[8, 257, 1]" = var_mean_7[1];  var_mean_7 = None
    add_26: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_42, 1e-06);  getitem_42 = None
    rsqrt_7: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_26);  add_26 = None
    sub_7: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_25, getitem_43);  getitem_43 = None
    mul_23: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_7);  sub_7 = None
    mul_24: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_23, primals_51)
    add_27: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_24, primals_52);  mul_24 = primals_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_41: "f32[2056, 512]" = torch.ops.aten.view.default(add_27, [2056, 512]);  add_27 = None
    permute_26: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    addmm_14: "f32[2056, 2048]" = torch.ops.aten.addmm.default(primals_54, view_41, permute_26);  primals_54 = None
    view_42: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_14, [8, 257, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_25: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_42, 0.5)
    mul_26: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_42, 0.7071067811865476);  view_42 = None
    erf_3: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_26);  mul_26 = None
    add_28: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_27: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_25, add_28);  mul_25 = add_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_11: "f32[8, 257, 2048]" = torch.ops.aten.clone.default(mul_27);  mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_43: "f32[2056, 2048]" = torch.ops.aten.view.default(clone_11, [2056, 2048]);  clone_11 = None
    permute_27: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_15: "f32[2056, 512]" = torch.ops.aten.addmm.default(primals_56, view_43, permute_27);  primals_56 = None
    view_44: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_15, [8, 257, 512]);  addmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_12: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_44);  view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_29: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_25, clone_12);  add_25 = clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_8 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
    getitem_44: "f32[8, 257, 1]" = var_mean_8[0]
    getitem_45: "f32[8, 257, 1]" = var_mean_8[1];  var_mean_8 = None
    add_30: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_44, 1e-06);  getitem_44 = None
    rsqrt_8: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
    sub_8: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_29, getitem_45);  getitem_45 = None
    mul_28: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_8);  sub_8 = None
    mul_29: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_28, primals_57)
    add_31: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_29, primals_58);  mul_29 = primals_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_45: "f32[2056, 512]" = torch.ops.aten.view.default(add_31, [2056, 512]);  add_31 = None
    permute_28: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    addmm_16: "f32[2056, 1536]" = torch.ops.aten.addmm.default(primals_60, view_45, permute_28);  primals_60 = None
    view_46: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_16, [8, 257, 1536]);  addmm_16 = None
    view_47: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_46, [8, 257, 3, 8, 64]);  view_46 = None
    permute_29: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_47, [2, 0, 3, 1, 4]);  view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = torch.ops.aten.unbind.int(permute_29);  permute_29 = None
    getitem_46: "f32[8, 8, 257, 64]" = unbind_4[0]
    getitem_47: "f32[8, 8, 257, 64]" = unbind_4[1]
    getitem_48: "f32[8, 8, 257, 64]" = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_46, getitem_47, getitem_48, None, True)
    getitem_49: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_4[0]
    getitem_50: "f32[8, 8, 288]" = _scaled_dot_product_efficient_attention_4[1]
    getitem_51: "i64[]" = _scaled_dot_product_efficient_attention_4[2]
    getitem_52: "i64[]" = _scaled_dot_product_efficient_attention_4[3];  _scaled_dot_product_efficient_attention_4 = None
    alias_4: "f32[8, 8, 257, 64]" = torch.ops.aten.alias.default(getitem_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_30: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3]);  getitem_49 = None
    view_48: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_30, [8, 257, 512]);  permute_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_49: "f32[2056, 512]" = torch.ops.aten.view.default(view_48, [2056, 512]);  view_48 = None
    permute_31: "f32[512, 512]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    addmm_17: "f32[2056, 512]" = torch.ops.aten.addmm.default(primals_62, view_49, permute_31);  primals_62 = None
    view_50: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_17, [8, 257, 512]);  addmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_13: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_50);  view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_32: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_29, clone_13);  add_29 = clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_9 = torch.ops.aten.var_mean.correction(add_32, [2], correction = 0, keepdim = True)
    getitem_53: "f32[8, 257, 1]" = var_mean_9[0]
    getitem_54: "f32[8, 257, 1]" = var_mean_9[1];  var_mean_9 = None
    add_33: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_53, 1e-06);  getitem_53 = None
    rsqrt_9: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_33);  add_33 = None
    sub_9: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_32, getitem_54);  getitem_54 = None
    mul_30: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_9, rsqrt_9);  sub_9 = None
    mul_31: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_30, primals_63)
    add_34: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_31, primals_64);  mul_31 = primals_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_51: "f32[2056, 512]" = torch.ops.aten.view.default(add_34, [2056, 512]);  add_34 = None
    permute_32: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    addmm_18: "f32[2056, 2048]" = torch.ops.aten.addmm.default(primals_66, view_51, permute_32);  primals_66 = None
    view_52: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_18, [8, 257, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_32: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_52, 0.5)
    mul_33: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_52, 0.7071067811865476);  view_52 = None
    erf_4: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_35: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_34: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_32, add_35);  mul_32 = add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_14: "f32[8, 257, 2048]" = torch.ops.aten.clone.default(mul_34);  mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_53: "f32[2056, 2048]" = torch.ops.aten.view.default(clone_14, [2056, 2048]);  clone_14 = None
    permute_33: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    addmm_19: "f32[2056, 512]" = torch.ops.aten.addmm.default(primals_68, view_53, permute_33);  primals_68 = None
    view_54: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_19, [8, 257, 512]);  addmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_15: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_54);  view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_36: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_32, clone_15);  add_32 = clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_10 = torch.ops.aten.var_mean.correction(add_36, [2], correction = 0, keepdim = True)
    getitem_55: "f32[8, 257, 1]" = var_mean_10[0]
    getitem_56: "f32[8, 257, 1]" = var_mean_10[1];  var_mean_10 = None
    add_37: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_55, 1e-06);  getitem_55 = None
    rsqrt_10: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_37);  add_37 = None
    sub_10: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_36, getitem_56);  getitem_56 = None
    mul_35: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_10);  sub_10 = None
    mul_36: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_35, primals_69)
    add_38: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_36, primals_70);  mul_36 = primals_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_55: "f32[2056, 512]" = torch.ops.aten.view.default(add_38, [2056, 512]);  add_38 = None
    permute_34: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_20: "f32[2056, 1536]" = torch.ops.aten.addmm.default(primals_72, view_55, permute_34);  primals_72 = None
    view_56: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_20, [8, 257, 1536]);  addmm_20 = None
    view_57: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_56, [8, 257, 3, 8, 64]);  view_56 = None
    permute_35: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_57, [2, 0, 3, 1, 4]);  view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = torch.ops.aten.unbind.int(permute_35);  permute_35 = None
    getitem_57: "f32[8, 8, 257, 64]" = unbind_5[0]
    getitem_58: "f32[8, 8, 257, 64]" = unbind_5[1]
    getitem_59: "f32[8, 8, 257, 64]" = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_57, getitem_58, getitem_59, None, True)
    getitem_60: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_5[0]
    getitem_61: "f32[8, 8, 288]" = _scaled_dot_product_efficient_attention_5[1]
    getitem_62: "i64[]" = _scaled_dot_product_efficient_attention_5[2]
    getitem_63: "i64[]" = _scaled_dot_product_efficient_attention_5[3];  _scaled_dot_product_efficient_attention_5 = None
    alias_5: "f32[8, 8, 257, 64]" = torch.ops.aten.alias.default(getitem_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_36: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3]);  getitem_60 = None
    view_58: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_36, [8, 257, 512]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_59: "f32[2056, 512]" = torch.ops.aten.view.default(view_58, [2056, 512]);  view_58 = None
    permute_37: "f32[512, 512]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_21: "f32[2056, 512]" = torch.ops.aten.addmm.default(primals_74, view_59, permute_37);  primals_74 = None
    view_60: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_21, [8, 257, 512]);  addmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_16: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_60);  view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_39: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_36, clone_16);  add_36 = clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_11 = torch.ops.aten.var_mean.correction(add_39, [2], correction = 0, keepdim = True)
    getitem_64: "f32[8, 257, 1]" = var_mean_11[0]
    getitem_65: "f32[8, 257, 1]" = var_mean_11[1];  var_mean_11 = None
    add_40: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_64, 1e-06);  getitem_64 = None
    rsqrt_11: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_40);  add_40 = None
    sub_11: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_39, getitem_65);  getitem_65 = None
    mul_37: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_11);  sub_11 = None
    mul_38: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_37, primals_75)
    add_41: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_38, primals_76);  mul_38 = primals_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_61: "f32[2056, 512]" = torch.ops.aten.view.default(add_41, [2056, 512]);  add_41 = None
    permute_38: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm_22: "f32[2056, 2048]" = torch.ops.aten.addmm.default(primals_78, view_61, permute_38);  primals_78 = None
    view_62: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_22, [8, 257, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_39: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_62, 0.5)
    mul_40: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_62, 0.7071067811865476);  view_62 = None
    erf_5: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_40);  mul_40 = None
    add_42: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_41: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_39, add_42);  mul_39 = add_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_17: "f32[8, 257, 2048]" = torch.ops.aten.clone.default(mul_41);  mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_63: "f32[2056, 2048]" = torch.ops.aten.view.default(clone_17, [2056, 2048]);  clone_17 = None
    permute_39: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    addmm_23: "f32[2056, 512]" = torch.ops.aten.addmm.default(primals_80, view_63, permute_39);  primals_80 = None
    view_64: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_23, [8, 257, 512]);  addmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_18: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_64);  view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_43: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_39, clone_18);  add_39 = clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_12 = torch.ops.aten.var_mean.correction(add_43, [2], correction = 0, keepdim = True)
    getitem_66: "f32[8, 257, 1]" = var_mean_12[0]
    getitem_67: "f32[8, 257, 1]" = var_mean_12[1];  var_mean_12 = None
    add_44: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-06);  getitem_66 = None
    rsqrt_12: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_44);  add_44 = None
    sub_12: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_43, getitem_67);  getitem_67 = None
    mul_42: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_12);  sub_12 = None
    mul_43: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_42, primals_81)
    add_45: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_43, primals_82);  mul_43 = primals_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_65: "f32[2056, 512]" = torch.ops.aten.view.default(add_45, [2056, 512]);  add_45 = None
    permute_40: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    addmm_24: "f32[2056, 1536]" = torch.ops.aten.addmm.default(primals_84, view_65, permute_40);  primals_84 = None
    view_66: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_24, [8, 257, 1536]);  addmm_24 = None
    view_67: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_66, [8, 257, 3, 8, 64]);  view_66 = None
    permute_41: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_67, [2, 0, 3, 1, 4]);  view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = torch.ops.aten.unbind.int(permute_41);  permute_41 = None
    getitem_68: "f32[8, 8, 257, 64]" = unbind_6[0]
    getitem_69: "f32[8, 8, 257, 64]" = unbind_6[1]
    getitem_70: "f32[8, 8, 257, 64]" = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_68, getitem_69, getitem_70, None, True)
    getitem_71: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_6[0]
    getitem_72: "f32[8, 8, 288]" = _scaled_dot_product_efficient_attention_6[1]
    getitem_73: "i64[]" = _scaled_dot_product_efficient_attention_6[2]
    getitem_74: "i64[]" = _scaled_dot_product_efficient_attention_6[3];  _scaled_dot_product_efficient_attention_6 = None
    alias_6: "f32[8, 8, 257, 64]" = torch.ops.aten.alias.default(getitem_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_42: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3]);  getitem_71 = None
    view_68: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_42, [8, 257, 512]);  permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_69: "f32[2056, 512]" = torch.ops.aten.view.default(view_68, [2056, 512]);  view_68 = None
    permute_43: "f32[512, 512]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_25: "f32[2056, 512]" = torch.ops.aten.addmm.default(primals_86, view_69, permute_43);  primals_86 = None
    view_70: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_25, [8, 257, 512]);  addmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_19: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_70);  view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_46: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_43, clone_19);  add_43 = clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_13 = torch.ops.aten.var_mean.correction(add_46, [2], correction = 0, keepdim = True)
    getitem_75: "f32[8, 257, 1]" = var_mean_13[0]
    getitem_76: "f32[8, 257, 1]" = var_mean_13[1];  var_mean_13 = None
    add_47: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_75, 1e-06);  getitem_75 = None
    rsqrt_13: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_47);  add_47 = None
    sub_13: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_46, getitem_76);  getitem_76 = None
    mul_44: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_13);  sub_13 = None
    mul_45: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_44, primals_87)
    add_48: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_45, primals_88);  mul_45 = primals_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_71: "f32[2056, 512]" = torch.ops.aten.view.default(add_48, [2056, 512]);  add_48 = None
    permute_44: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    addmm_26: "f32[2056, 2048]" = torch.ops.aten.addmm.default(primals_90, view_71, permute_44);  primals_90 = None
    view_72: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 257, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_46: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_72, 0.5)
    mul_47: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_72, 0.7071067811865476);  view_72 = None
    erf_6: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_47);  mul_47 = None
    add_49: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_48: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_46, add_49);  mul_46 = add_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_20: "f32[8, 257, 2048]" = torch.ops.aten.clone.default(mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_73: "f32[2056, 2048]" = torch.ops.aten.view.default(clone_20, [2056, 2048]);  clone_20 = None
    permute_45: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_27: "f32[2056, 512]" = torch.ops.aten.addmm.default(primals_92, view_73, permute_45);  primals_92 = None
    view_74: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_27, [8, 257, 512]);  addmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_21: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_74);  view_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_50: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_46, clone_21);  add_46 = clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_14 = torch.ops.aten.var_mean.correction(add_50, [2], correction = 0, keepdim = True)
    getitem_77: "f32[8, 257, 1]" = var_mean_14[0]
    getitem_78: "f32[8, 257, 1]" = var_mean_14[1];  var_mean_14 = None
    add_51: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_77, 1e-06);  getitem_77 = None
    rsqrt_14: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_51);  add_51 = None
    sub_14: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_50, getitem_78);  getitem_78 = None
    mul_49: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_14);  sub_14 = None
    mul_50: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_49, primals_93)
    add_52: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_50, primals_94);  mul_50 = primals_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_75: "f32[2056, 512]" = torch.ops.aten.view.default(add_52, [2056, 512]);  add_52 = None
    permute_46: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_28: "f32[2056, 1536]" = torch.ops.aten.addmm.default(primals_96, view_75, permute_46);  primals_96 = None
    view_76: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_28, [8, 257, 1536]);  addmm_28 = None
    view_77: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_76, [8, 257, 3, 8, 64]);  view_76 = None
    permute_47: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_77, [2, 0, 3, 1, 4]);  view_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = torch.ops.aten.unbind.int(permute_47);  permute_47 = None
    getitem_79: "f32[8, 8, 257, 64]" = unbind_7[0]
    getitem_80: "f32[8, 8, 257, 64]" = unbind_7[1]
    getitem_81: "f32[8, 8, 257, 64]" = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_79, getitem_80, getitem_81, None, True)
    getitem_82: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_7[0]
    getitem_83: "f32[8, 8, 288]" = _scaled_dot_product_efficient_attention_7[1]
    getitem_84: "i64[]" = _scaled_dot_product_efficient_attention_7[2]
    getitem_85: "i64[]" = _scaled_dot_product_efficient_attention_7[3];  _scaled_dot_product_efficient_attention_7 = None
    alias_7: "f32[8, 8, 257, 64]" = torch.ops.aten.alias.default(getitem_82)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_48: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None
    view_78: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_48, [8, 257, 512]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_79: "f32[2056, 512]" = torch.ops.aten.view.default(view_78, [2056, 512]);  view_78 = None
    permute_49: "f32[512, 512]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_29: "f32[2056, 512]" = torch.ops.aten.addmm.default(primals_98, view_79, permute_49);  primals_98 = None
    view_80: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_29, [8, 257, 512]);  addmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_22: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_80);  view_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_53: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_50, clone_22);  add_50 = clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_15 = torch.ops.aten.var_mean.correction(add_53, [2], correction = 0, keepdim = True)
    getitem_86: "f32[8, 257, 1]" = var_mean_15[0]
    getitem_87: "f32[8, 257, 1]" = var_mean_15[1];  var_mean_15 = None
    add_54: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-06);  getitem_86 = None
    rsqrt_15: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_54);  add_54 = None
    sub_15: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_53, getitem_87);  getitem_87 = None
    mul_51: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_15);  sub_15 = None
    mul_52: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_51, primals_99)
    add_55: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_52, primals_100);  mul_52 = primals_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_81: "f32[2056, 512]" = torch.ops.aten.view.default(add_55, [2056, 512]);  add_55 = None
    permute_50: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_30: "f32[2056, 2048]" = torch.ops.aten.addmm.default(primals_102, view_81, permute_50);  primals_102 = None
    view_82: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 257, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_53: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_82, 0.5)
    mul_54: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_82, 0.7071067811865476);  view_82 = None
    erf_7: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_54);  mul_54 = None
    add_56: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_55: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_53, add_56);  mul_53 = add_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_23: "f32[8, 257, 2048]" = torch.ops.aten.clone.default(mul_55);  mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_83: "f32[2056, 2048]" = torch.ops.aten.view.default(clone_23, [2056, 2048]);  clone_23 = None
    permute_51: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_31: "f32[2056, 512]" = torch.ops.aten.addmm.default(primals_104, view_83, permute_51);  primals_104 = None
    view_84: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_31, [8, 257, 512]);  addmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_24: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_84);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_57: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_53, clone_24);  add_53 = clone_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_16 = torch.ops.aten.var_mean.correction(add_57, [2], correction = 0, keepdim = True)
    getitem_88: "f32[8, 257, 1]" = var_mean_16[0]
    getitem_89: "f32[8, 257, 1]" = var_mean_16[1];  var_mean_16 = None
    add_58: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-06);  getitem_88 = None
    rsqrt_16: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_58);  add_58 = None
    sub_16: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_57, getitem_89);  getitem_89 = None
    mul_56: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_16);  sub_16 = None
    mul_57: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_56, primals_105)
    add_59: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_57, primals_106);  mul_57 = primals_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_85: "f32[2056, 512]" = torch.ops.aten.view.default(add_59, [2056, 512]);  add_59 = None
    permute_52: "f32[512, 1536]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_32: "f32[2056, 1536]" = torch.ops.aten.addmm.default(primals_108, view_85, permute_52);  primals_108 = None
    view_86: "f32[8, 257, 1536]" = torch.ops.aten.view.default(addmm_32, [8, 257, 1536]);  addmm_32 = None
    view_87: "f32[8, 257, 3, 8, 64]" = torch.ops.aten.view.default(view_86, [8, 257, 3, 8, 64]);  view_86 = None
    permute_53: "f32[3, 8, 8, 257, 64]" = torch.ops.aten.permute.default(view_87, [2, 0, 3, 1, 4]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = torch.ops.aten.unbind.int(permute_53);  permute_53 = None
    getitem_90: "f32[8, 8, 257, 64]" = unbind_8[0]
    getitem_91: "f32[8, 8, 257, 64]" = unbind_8[1]
    getitem_92: "f32[8, 8, 257, 64]" = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_90, getitem_91, getitem_92, None, True)
    getitem_93: "f32[8, 8, 257, 64]" = _scaled_dot_product_efficient_attention_8[0]
    getitem_94: "f32[8, 8, 288]" = _scaled_dot_product_efficient_attention_8[1]
    getitem_95: "i64[]" = _scaled_dot_product_efficient_attention_8[2]
    getitem_96: "i64[]" = _scaled_dot_product_efficient_attention_8[3];  _scaled_dot_product_efficient_attention_8 = None
    alias_8: "f32[8, 8, 257, 64]" = torch.ops.aten.alias.default(getitem_93)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_54: "f32[8, 257, 8, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3]);  getitem_93 = None
    view_88: "f32[8, 257, 512]" = torch.ops.aten.view.default(permute_54, [8, 257, 512]);  permute_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_89: "f32[2056, 512]" = torch.ops.aten.view.default(view_88, [2056, 512]);  view_88 = None
    permute_55: "f32[512, 512]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    addmm_33: "f32[2056, 512]" = torch.ops.aten.addmm.default(primals_110, view_89, permute_55);  primals_110 = None
    view_90: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_33, [8, 257, 512]);  addmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_25: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_90);  view_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_60: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_57, clone_25);  add_57 = clone_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_17 = torch.ops.aten.var_mean.correction(add_60, [2], correction = 0, keepdim = True)
    getitem_97: "f32[8, 257, 1]" = var_mean_17[0]
    getitem_98: "f32[8, 257, 1]" = var_mean_17[1];  var_mean_17 = None
    add_61: "f32[8, 257, 1]" = torch.ops.aten.add.Tensor(getitem_97, 1e-06);  getitem_97 = None
    rsqrt_17: "f32[8, 257, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    sub_17: "f32[8, 257, 512]" = torch.ops.aten.sub.Tensor(add_60, getitem_98);  getitem_98 = None
    mul_58: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_17);  sub_17 = None
    mul_59: "f32[8, 257, 512]" = torch.ops.aten.mul.Tensor(mul_58, primals_111)
    add_62: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(mul_59, primals_112);  mul_59 = primals_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_91: "f32[2056, 512]" = torch.ops.aten.view.default(add_62, [2056, 512]);  add_62 = None
    permute_56: "f32[512, 2048]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    addmm_34: "f32[2056, 2048]" = torch.ops.aten.addmm.default(primals_114, view_91, permute_56);  primals_114 = None
    view_92: "f32[8, 257, 2048]" = torch.ops.aten.view.default(addmm_34, [8, 257, 2048])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_60: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_92, 0.5)
    mul_61: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(view_92, 0.7071067811865476);  view_92 = None
    erf_8: "f32[8, 257, 2048]" = torch.ops.aten.erf.default(mul_61);  mul_61 = None
    add_63: "f32[8, 257, 2048]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_62: "f32[8, 257, 2048]" = torch.ops.aten.mul.Tensor(mul_60, add_63);  mul_60 = add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_26: "f32[8, 257, 2048]" = torch.ops.aten.clone.default(mul_62);  mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_93: "f32[2056, 2048]" = torch.ops.aten.view.default(clone_26, [2056, 2048]);  clone_26 = None
    permute_57: "f32[2048, 512]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_35: "f32[2056, 512]" = torch.ops.aten.addmm.default(primals_116, view_93, permute_57);  primals_116 = None
    view_94: "f32[8, 257, 512]" = torch.ops.aten.view.default(addmm_35, [8, 257, 512]);  addmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_27: "f32[8, 257, 512]" = torch.ops.aten.clone.default(view_94);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_64: "f32[8, 257, 512]" = torch.ops.aten.add.Tensor(add_60, clone_27);  add_60 = clone_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    slice_5: "f32[8, 257, 512]" = torch.ops.aten.slice.Tensor(add_64, 0, 0, 9223372036854775807);  add_64 = None
    slice_6: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(slice_5, 1, 0, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:88, code: x = x[:, token_length:]
    slice_8: "f32[8, 256, 512]" = torch.ops.aten.slice.Tensor(slice_5, 1, 1, 9223372036854775807);  slice_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:89, code: x = x.transpose(1, 2).reshape(B, C, H, W)
    permute_58: "f32[8, 512, 256]" = torch.ops.aten.permute.default(slice_8, [0, 2, 1]);  slice_8 = None
    view_95: "f32[8, 512, 16, 16]" = torch.ops.aten.view.default(permute_58, [8, 512, 16, 16]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:110, code: x = self.conv(x)
    convolution_2: "f32[8, 1024, 8, 8]" = torch.ops.aten.convolution.default(view_95, primals_117, primals_118, [2, 2], [1, 1], [1, 1], False, [0, 0], 512);  primals_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:111, code: cls_token = self.fc(cls_token)
    permute_59: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    view_96: "f32[8, 512]" = torch.ops.aten.view.default(slice_6, [8, 512]);  slice_6 = None
    mm_1: "f32[8, 1024]" = torch.ops.aten.mm.default(view_96, permute_59)
    view_97: "f32[8, 1, 1024]" = torch.ops.aten.view.default(mm_1, [8, 1, 1024]);  mm_1 = None
    add_65: "f32[8, 1, 1024]" = torch.ops.aten.add.Tensor(view_97, primals_120);  view_97 = primals_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:81, code: x = x.flatten(2).transpose(1, 2)
    view_98: "f32[8, 1024, 64]" = torch.ops.aten.view.default(convolution_2, [8, 1024, 64]);  convolution_2 = None
    permute_60: "f32[8, 64, 1024]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:82, code: x = torch.cat((cls_tokens, x), dim=1)
    cat_2: "f32[8, 65, 1024]" = torch.ops.aten.cat.default([add_65, permute_60], 1);  add_65 = permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_18 = torch.ops.aten.var_mean.correction(cat_2, [2], correction = 0, keepdim = True)
    getitem_99: "f32[8, 65, 1]" = var_mean_18[0]
    getitem_100: "f32[8, 65, 1]" = var_mean_18[1];  var_mean_18 = None
    add_66: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_99, 1e-06);  getitem_99 = None
    rsqrt_18: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_66);  add_66 = None
    sub_18: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(cat_2, getitem_100)
    mul_63: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_18);  sub_18 = None
    mul_64: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_63, primals_121);  mul_63 = None
    add_67: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_64, primals_122);  mul_64 = primals_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_99: "f32[520, 1024]" = torch.ops.aten.view.default(add_67, [520, 1024]);  add_67 = None
    permute_61: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    addmm_36: "f32[520, 3072]" = torch.ops.aten.addmm.default(primals_124, view_99, permute_61);  primals_124 = None
    view_100: "f32[8, 65, 3072]" = torch.ops.aten.view.default(addmm_36, [8, 65, 3072]);  addmm_36 = None
    view_101: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.view.default(view_100, [8, 65, 3, 16, 64]);  view_100 = None
    permute_62: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.permute.default(view_101, [2, 0, 3, 1, 4]);  view_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_9 = torch.ops.aten.unbind.int(permute_62);  permute_62 = None
    getitem_101: "f32[8, 16, 65, 64]" = unbind_9[0]
    getitem_102: "f32[8, 16, 65, 64]" = unbind_9[1]
    getitem_103: "f32[8, 16, 65, 64]" = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_101, getitem_102, getitem_103, None, True)
    getitem_104: "f32[8, 16, 65, 64]" = _scaled_dot_product_efficient_attention_9[0]
    getitem_105: "f32[8, 16, 96]" = _scaled_dot_product_efficient_attention_9[1]
    getitem_106: "i64[]" = _scaled_dot_product_efficient_attention_9[2]
    getitem_107: "i64[]" = _scaled_dot_product_efficient_attention_9[3];  _scaled_dot_product_efficient_attention_9 = None
    alias_9: "f32[8, 16, 65, 64]" = torch.ops.aten.alias.default(getitem_104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_63: "f32[8, 65, 16, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None
    view_102: "f32[8, 65, 1024]" = torch.ops.aten.view.default(permute_63, [8, 65, 1024]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_103: "f32[520, 1024]" = torch.ops.aten.view.default(view_102, [520, 1024]);  view_102 = None
    permute_64: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_37: "f32[520, 1024]" = torch.ops.aten.addmm.default(primals_126, view_103, permute_64);  primals_126 = None
    view_104: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_37, [8, 65, 1024]);  addmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_28: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_104);  view_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_68: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(cat_2, clone_28);  clone_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_19 = torch.ops.aten.var_mean.correction(add_68, [2], correction = 0, keepdim = True)
    getitem_108: "f32[8, 65, 1]" = var_mean_19[0]
    getitem_109: "f32[8, 65, 1]" = var_mean_19[1];  var_mean_19 = None
    add_69: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-06);  getitem_108 = None
    rsqrt_19: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_69);  add_69 = None
    sub_19: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_68, getitem_109);  getitem_109 = None
    mul_65: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_19);  sub_19 = None
    mul_66: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_65, primals_127)
    add_70: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_66, primals_128);  mul_66 = primals_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_105: "f32[520, 1024]" = torch.ops.aten.view.default(add_70, [520, 1024]);  add_70 = None
    permute_65: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_38: "f32[520, 4096]" = torch.ops.aten.addmm.default(primals_130, view_105, permute_65);  primals_130 = None
    view_106: "f32[8, 65, 4096]" = torch.ops.aten.view.default(addmm_38, [8, 65, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_67: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
    mul_68: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_106, 0.7071067811865476);  view_106 = None
    erf_9: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_71: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_69: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_67, add_71);  mul_67 = add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_29: "f32[8, 65, 4096]" = torch.ops.aten.clone.default(mul_69);  mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_107: "f32[520, 4096]" = torch.ops.aten.view.default(clone_29, [520, 4096]);  clone_29 = None
    permute_66: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_39: "f32[520, 1024]" = torch.ops.aten.addmm.default(primals_132, view_107, permute_66);  primals_132 = None
    view_108: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_39, [8, 65, 1024]);  addmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_30: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_108);  view_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_72: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_68, clone_30);  add_68 = clone_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_20 = torch.ops.aten.var_mean.correction(add_72, [2], correction = 0, keepdim = True)
    getitem_110: "f32[8, 65, 1]" = var_mean_20[0]
    getitem_111: "f32[8, 65, 1]" = var_mean_20[1];  var_mean_20 = None
    add_73: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-06);  getitem_110 = None
    rsqrt_20: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_73);  add_73 = None
    sub_20: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_72, getitem_111);  getitem_111 = None
    mul_70: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_20);  sub_20 = None
    mul_71: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_70, primals_133)
    add_74: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_71, primals_134);  mul_71 = primals_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_109: "f32[520, 1024]" = torch.ops.aten.view.default(add_74, [520, 1024]);  add_74 = None
    permute_67: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_40: "f32[520, 3072]" = torch.ops.aten.addmm.default(primals_136, view_109, permute_67);  primals_136 = None
    view_110: "f32[8, 65, 3072]" = torch.ops.aten.view.default(addmm_40, [8, 65, 3072]);  addmm_40 = None
    view_111: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.view.default(view_110, [8, 65, 3, 16, 64]);  view_110 = None
    permute_68: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.permute.default(view_111, [2, 0, 3, 1, 4]);  view_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_10 = torch.ops.aten.unbind.int(permute_68);  permute_68 = None
    getitem_112: "f32[8, 16, 65, 64]" = unbind_10[0]
    getitem_113: "f32[8, 16, 65, 64]" = unbind_10[1]
    getitem_114: "f32[8, 16, 65, 64]" = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_112, getitem_113, getitem_114, None, True)
    getitem_115: "f32[8, 16, 65, 64]" = _scaled_dot_product_efficient_attention_10[0]
    getitem_116: "f32[8, 16, 96]" = _scaled_dot_product_efficient_attention_10[1]
    getitem_117: "i64[]" = _scaled_dot_product_efficient_attention_10[2]
    getitem_118: "i64[]" = _scaled_dot_product_efficient_attention_10[3];  _scaled_dot_product_efficient_attention_10 = None
    alias_10: "f32[8, 16, 65, 64]" = torch.ops.aten.alias.default(getitem_115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_69: "f32[8, 65, 16, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3]);  getitem_115 = None
    view_112: "f32[8, 65, 1024]" = torch.ops.aten.view.default(permute_69, [8, 65, 1024]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_113: "f32[520, 1024]" = torch.ops.aten.view.default(view_112, [520, 1024]);  view_112 = None
    permute_70: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_41: "f32[520, 1024]" = torch.ops.aten.addmm.default(primals_138, view_113, permute_70);  primals_138 = None
    view_114: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_41, [8, 65, 1024]);  addmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_31: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_114);  view_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_75: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_72, clone_31);  add_72 = clone_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_21 = torch.ops.aten.var_mean.correction(add_75, [2], correction = 0, keepdim = True)
    getitem_119: "f32[8, 65, 1]" = var_mean_21[0]
    getitem_120: "f32[8, 65, 1]" = var_mean_21[1];  var_mean_21 = None
    add_76: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_119, 1e-06);  getitem_119 = None
    rsqrt_21: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_76);  add_76 = None
    sub_21: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_75, getitem_120);  getitem_120 = None
    mul_72: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_21, rsqrt_21);  sub_21 = None
    mul_73: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_72, primals_139)
    add_77: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_73, primals_140);  mul_73 = primals_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_115: "f32[520, 1024]" = torch.ops.aten.view.default(add_77, [520, 1024]);  add_77 = None
    permute_71: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_42: "f32[520, 4096]" = torch.ops.aten.addmm.default(primals_142, view_115, permute_71);  primals_142 = None
    view_116: "f32[8, 65, 4096]" = torch.ops.aten.view.default(addmm_42, [8, 65, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_74: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_116, 0.5)
    mul_75: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_116, 0.7071067811865476);  view_116 = None
    erf_10: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
    add_78: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_76: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_74, add_78);  mul_74 = add_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_32: "f32[8, 65, 4096]" = torch.ops.aten.clone.default(mul_76);  mul_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_117: "f32[520, 4096]" = torch.ops.aten.view.default(clone_32, [520, 4096]);  clone_32 = None
    permute_72: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_43: "f32[520, 1024]" = torch.ops.aten.addmm.default(primals_144, view_117, permute_72);  primals_144 = None
    view_118: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_43, [8, 65, 1024]);  addmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_33: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_118);  view_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_79: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_75, clone_33);  add_75 = clone_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_22 = torch.ops.aten.var_mean.correction(add_79, [2], correction = 0, keepdim = True)
    getitem_121: "f32[8, 65, 1]" = var_mean_22[0]
    getitem_122: "f32[8, 65, 1]" = var_mean_22[1];  var_mean_22 = None
    add_80: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_121, 1e-06);  getitem_121 = None
    rsqrt_22: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_80);  add_80 = None
    sub_22: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_79, getitem_122);  getitem_122 = None
    mul_77: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_22);  sub_22 = None
    mul_78: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_77, primals_145)
    add_81: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_78, primals_146);  mul_78 = primals_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_119: "f32[520, 1024]" = torch.ops.aten.view.default(add_81, [520, 1024]);  add_81 = None
    permute_73: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm_44: "f32[520, 3072]" = torch.ops.aten.addmm.default(primals_148, view_119, permute_73);  primals_148 = None
    view_120: "f32[8, 65, 3072]" = torch.ops.aten.view.default(addmm_44, [8, 65, 3072]);  addmm_44 = None
    view_121: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.view.default(view_120, [8, 65, 3, 16, 64]);  view_120 = None
    permute_74: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.permute.default(view_121, [2, 0, 3, 1, 4]);  view_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_11 = torch.ops.aten.unbind.int(permute_74);  permute_74 = None
    getitem_123: "f32[8, 16, 65, 64]" = unbind_11[0]
    getitem_124: "f32[8, 16, 65, 64]" = unbind_11[1]
    getitem_125: "f32[8, 16, 65, 64]" = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_123, getitem_124, getitem_125, None, True)
    getitem_126: "f32[8, 16, 65, 64]" = _scaled_dot_product_efficient_attention_11[0]
    getitem_127: "f32[8, 16, 96]" = _scaled_dot_product_efficient_attention_11[1]
    getitem_128: "i64[]" = _scaled_dot_product_efficient_attention_11[2]
    getitem_129: "i64[]" = _scaled_dot_product_efficient_attention_11[3];  _scaled_dot_product_efficient_attention_11 = None
    alias_11: "f32[8, 16, 65, 64]" = torch.ops.aten.alias.default(getitem_126)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_75: "f32[8, 65, 16, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None
    view_122: "f32[8, 65, 1024]" = torch.ops.aten.view.default(permute_75, [8, 65, 1024]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_123: "f32[520, 1024]" = torch.ops.aten.view.default(view_122, [520, 1024]);  view_122 = None
    permute_76: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    addmm_45: "f32[520, 1024]" = torch.ops.aten.addmm.default(primals_150, view_123, permute_76);  primals_150 = None
    view_124: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_45, [8, 65, 1024]);  addmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_34: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_124);  view_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_82: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_79, clone_34);  add_79 = clone_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_23 = torch.ops.aten.var_mean.correction(add_82, [2], correction = 0, keepdim = True)
    getitem_130: "f32[8, 65, 1]" = var_mean_23[0]
    getitem_131: "f32[8, 65, 1]" = var_mean_23[1];  var_mean_23 = None
    add_83: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-06);  getitem_130 = None
    rsqrt_23: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_83);  add_83 = None
    sub_23: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_82, getitem_131);  getitem_131 = None
    mul_79: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_23);  sub_23 = None
    mul_80: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_79, primals_151)
    add_84: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_80, primals_152);  mul_80 = primals_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_125: "f32[520, 1024]" = torch.ops.aten.view.default(add_84, [520, 1024]);  add_84 = None
    permute_77: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    addmm_46: "f32[520, 4096]" = torch.ops.aten.addmm.default(primals_154, view_125, permute_77);  primals_154 = None
    view_126: "f32[8, 65, 4096]" = torch.ops.aten.view.default(addmm_46, [8, 65, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_81: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_126, 0.5)
    mul_82: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_126, 0.7071067811865476);  view_126 = None
    erf_11: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_82);  mul_82 = None
    add_85: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_83: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_81, add_85);  mul_81 = add_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_35: "f32[8, 65, 4096]" = torch.ops.aten.clone.default(mul_83);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_127: "f32[520, 4096]" = torch.ops.aten.view.default(clone_35, [520, 4096]);  clone_35 = None
    permute_78: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_47: "f32[520, 1024]" = torch.ops.aten.addmm.default(primals_156, view_127, permute_78);  primals_156 = None
    view_128: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_47, [8, 65, 1024]);  addmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_36: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_128);  view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_86: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_82, clone_36);  add_82 = clone_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    var_mean_24 = torch.ops.aten.var_mean.correction(add_86, [2], correction = 0, keepdim = True)
    getitem_132: "f32[8, 65, 1]" = var_mean_24[0]
    getitem_133: "f32[8, 65, 1]" = var_mean_24[1];  var_mean_24 = None
    add_87: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_132, 1e-06);  getitem_132 = None
    rsqrt_24: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_87);  add_87 = None
    sub_24: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_86, getitem_133);  getitem_133 = None
    mul_84: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_24);  sub_24 = None
    mul_85: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_84, primals_157)
    add_88: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_85, primals_158);  mul_85 = primals_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    view_129: "f32[520, 1024]" = torch.ops.aten.view.default(add_88, [520, 1024]);  add_88 = None
    permute_79: "f32[1024, 3072]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    addmm_48: "f32[520, 3072]" = torch.ops.aten.addmm.default(primals_160, view_129, permute_79);  primals_160 = None
    view_130: "f32[8, 65, 3072]" = torch.ops.aten.view.default(addmm_48, [8, 65, 3072]);  addmm_48 = None
    view_131: "f32[8, 65, 3, 16, 64]" = torch.ops.aten.view.default(view_130, [8, 65, 3, 16, 64]);  view_130 = None
    permute_80: "f32[3, 8, 16, 65, 64]" = torch.ops.aten.permute.default(view_131, [2, 0, 3, 1, 4]);  view_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_12 = torch.ops.aten.unbind.int(permute_80);  permute_80 = None
    getitem_134: "f32[8, 16, 65, 64]" = unbind_12[0]
    getitem_135: "f32[8, 16, 65, 64]" = unbind_12[1]
    getitem_136: "f32[8, 16, 65, 64]" = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(getitem_134, getitem_135, getitem_136, None, True)
    getitem_137: "f32[8, 16, 65, 64]" = _scaled_dot_product_efficient_attention_12[0]
    getitem_138: "f32[8, 16, 96]" = _scaled_dot_product_efficient_attention_12[1]
    getitem_139: "i64[]" = _scaled_dot_product_efficient_attention_12[2]
    getitem_140: "i64[]" = _scaled_dot_product_efficient_attention_12[3];  _scaled_dot_product_efficient_attention_12 = None
    alias_12: "f32[8, 16, 65, 64]" = torch.ops.aten.alias.default(getitem_137)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    permute_81: "f32[8, 65, 16, 64]" = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3]);  getitem_137 = None
    view_132: "f32[8, 65, 1024]" = torch.ops.aten.view.default(permute_81, [8, 65, 1024]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    view_133: "f32[520, 1024]" = torch.ops.aten.view.default(view_132, [520, 1024]);  view_132 = None
    permute_82: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_49: "f32[520, 1024]" = torch.ops.aten.addmm.default(primals_162, view_133, permute_82);  primals_162 = None
    view_134: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_49, [8, 65, 1024]);  addmm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    clone_37: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_134);  view_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    add_89: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_86, clone_37);  add_86 = clone_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    var_mean_25 = torch.ops.aten.var_mean.correction(add_89, [2], correction = 0, keepdim = True)
    getitem_141: "f32[8, 65, 1]" = var_mean_25[0]
    getitem_142: "f32[8, 65, 1]" = var_mean_25[1];  var_mean_25 = None
    add_90: "f32[8, 65, 1]" = torch.ops.aten.add.Tensor(getitem_141, 1e-06);  getitem_141 = None
    rsqrt_25: "f32[8, 65, 1]" = torch.ops.aten.rsqrt.default(add_90);  add_90 = None
    sub_25: "f32[8, 65, 1024]" = torch.ops.aten.sub.Tensor(add_89, getitem_142);  getitem_142 = None
    mul_86: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_25);  sub_25 = None
    mul_87: "f32[8, 65, 1024]" = torch.ops.aten.mul.Tensor(mul_86, primals_163)
    add_91: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(mul_87, primals_164);  mul_87 = primals_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_135: "f32[520, 1024]" = torch.ops.aten.view.default(add_91, [520, 1024]);  add_91 = None
    permute_83: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    addmm_50: "f32[520, 4096]" = torch.ops.aten.addmm.default(primals_166, view_135, permute_83);  primals_166 = None
    view_136: "f32[8, 65, 4096]" = torch.ops.aten.view.default(addmm_50, [8, 65, 4096])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_88: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_136, 0.5)
    mul_89: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(view_136, 0.7071067811865476);  view_136 = None
    erf_12: "f32[8, 65, 4096]" = torch.ops.aten.erf.default(mul_89);  mul_89 = None
    add_92: "f32[8, 65, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_90: "f32[8, 65, 4096]" = torch.ops.aten.mul.Tensor(mul_88, add_92);  mul_88 = add_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    clone_38: "f32[8, 65, 4096]" = torch.ops.aten.clone.default(mul_90);  mul_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_137: "f32[520, 4096]" = torch.ops.aten.view.default(clone_38, [520, 4096]);  clone_38 = None
    permute_84: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
    addmm_51: "f32[520, 1024]" = torch.ops.aten.addmm.default(primals_168, view_137, permute_84);  primals_168 = None
    view_138: "f32[8, 65, 1024]" = torch.ops.aten.view.default(addmm_51, [8, 65, 1024]);  addmm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    clone_39: "f32[8, 65, 1024]" = torch.ops.aten.clone.default(view_138);  view_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    add_93: "f32[8, 65, 1024]" = torch.ops.aten.add.Tensor(add_89, clone_39);  add_89 = clone_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    slice_9: "f32[8, 65, 1024]" = torch.ops.aten.slice.Tensor(add_93, 0, 0, 9223372036854775807);  add_93 = None
    slice_10: "f32[8, 1, 1024]" = torch.ops.aten.slice.Tensor(slice_9, 1, 0, 1);  slice_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:260, code: cls_tokens = self.norm(cls_tokens)
    clone_40: "f32[8, 1, 1024]" = torch.ops.aten.clone.default(slice_10, memory_format = torch.contiguous_format);  slice_10 = None
    var_mean_26 = torch.ops.aten.var_mean.correction(clone_40, [2], correction = 0, keepdim = True)
    getitem_143: "f32[8, 1, 1]" = var_mean_26[0]
    getitem_144: "f32[8, 1, 1]" = var_mean_26[1];  var_mean_26 = None
    add_94: "f32[8, 1, 1]" = torch.ops.aten.add.Tensor(getitem_143, 1e-06);  getitem_143 = None
    rsqrt_26: "f32[8, 1, 1]" = torch.ops.aten.rsqrt.default(add_94);  add_94 = None
    sub_26: "f32[8, 1, 1024]" = torch.ops.aten.sub.Tensor(clone_40, getitem_144);  clone_40 = getitem_144 = None
    mul_91: "f32[8, 1, 1024]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_26);  sub_26 = None
    mul_92: "f32[8, 1, 1024]" = torch.ops.aten.mul.Tensor(mul_91, primals_169)
    add_95: "f32[8, 1, 1024]" = torch.ops.aten.add.Tensor(mul_92, primals_170);  mul_92 = primals_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:280, code: x = x[:, 0]
    slice_13: "f32[8, 1, 1024]" = torch.ops.aten.slice.Tensor(add_95, 0, 0, 9223372036854775807);  add_95 = None
    select: "f32[8, 1024]" = torch.ops.aten.select.int(slice_13, 1, 0);  slice_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:281, code: x = self.head_drop(x)
    clone_41: "f32[8, 1024]" = torch.ops.aten.clone.default(select);  select = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:283, code: x = self.head(x)
    permute_86: "f32[1024, 1000]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm_52: "f32[8, 1000]" = torch.ops.aten.addmm.default(primals_172, clone_41, permute_86);  primals_172 = None
    permute_87: "f32[1000, 1024]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:260, code: cls_tokens = self.norm(cls_tokens)
    div: "f32[8, 1, 1]" = torch.ops.aten.div.Tensor(rsqrt_26, 1024);  rsqrt_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_91: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_95: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_1: "f32[8, 65, 1]" = torch.ops.aten.div.Tensor(rsqrt_25, 1024);  rsqrt_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_13: "f32[8, 16, 65, 64]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_105: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_2: "f32[8, 65, 1]" = torch.ops.aten.div.Tensor(rsqrt_24, 1024);  rsqrt_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_109: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_113: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_3: "f32[8, 65, 1]" = torch.ops.aten.div.Tensor(rsqrt_23, 1024);  rsqrt_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_117: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_14: "f32[8, 16, 65, 64]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_123: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_4: "f32[8, 65, 1]" = torch.ops.aten.div.Tensor(rsqrt_22, 1024);  rsqrt_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_127: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_5: "f32[8, 65, 1]" = torch.ops.aten.div.Tensor(rsqrt_21, 1024);  rsqrt_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_135: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_15: "f32[8, 16, 65, 64]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_141: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_6: "f32[8, 65, 1]" = torch.ops.aten.div.Tensor(rsqrt_20, 1024);  rsqrt_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_145: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_149: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_7: "f32[8, 65, 1]" = torch.ops.aten.div.Tensor(rsqrt_19, 1024);  rsqrt_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_153: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_16: "f32[8, 16, 65, 64]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_159: "f32[3072, 1024]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:111, code: cls_token = self.fc(cls_token)
    permute_166: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_169: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_173: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_9: "f32[8, 257, 1]" = torch.ops.aten.div.Tensor(rsqrt_17, 512);  rsqrt_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_177: "f32[512, 512]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_17: "f32[8, 8, 257, 64]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_183: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_10: "f32[8, 257, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 512);  rsqrt_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_187: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_191: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_50, [1, 0]);  permute_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_11: "f32[8, 257, 1]" = torch.ops.aten.div.Tensor(rsqrt_15, 512);  rsqrt_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_195: "f32[512, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_18: "f32[8, 8, 257, 64]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_201: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_12: "f32[8, 257, 1]" = torch.ops.aten.div.Tensor(rsqrt_14, 512);  rsqrt_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_205: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_209: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_13: "f32[8, 257, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 512);  rsqrt_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_213: "f32[512, 512]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_19: "f32[8, 8, 257, 64]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_219: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_14: "f32[8, 257, 1]" = torch.ops.aten.div.Tensor(rsqrt_12, 512);  rsqrt_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_223: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_227: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_15: "f32[8, 257, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 512);  rsqrt_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_231: "f32[512, 512]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_20: "f32[8, 8, 257, 64]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_237: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_16: "f32[8, 257, 1]" = torch.ops.aten.div.Tensor(rsqrt_10, 512);  rsqrt_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_241: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_245: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_17: "f32[8, 257, 1]" = torch.ops.aten.div.Tensor(rsqrt_9, 512);  rsqrt_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_249: "f32[512, 512]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_21: "f32[8, 8, 257, 64]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_255: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_28, [1, 0]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_18: "f32[8, 257, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 512);  rsqrt_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_259: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_263: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_19: "f32[8, 257, 1]" = torch.ops.aten.div.Tensor(rsqrt_7, 512);  rsqrt_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_267: "f32[512, 512]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_22: "f32[8, 8, 257, 64]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_273: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:111, code: cls_token = self.fc(cls_token)
    permute_280: "f32[512, 256]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_283: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_287: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_21: "f32[8, 962, 1]" = torch.ops.aten.div.Tensor(rsqrt_5, 256);  rsqrt_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_291: "f32[256, 256]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_23: "f32[8, 4, 962, 64]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_297: "f32[768, 256]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_22: "f32[8, 962, 1]" = torch.ops.aten.div.Tensor(rsqrt_4, 256);  rsqrt_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_301: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_305: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_23: "f32[8, 962, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 256);  rsqrt_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_309: "f32[256, 256]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_24: "f32[8, 4, 962, 64]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_315: "f32[768, 256]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    div_24: "f32[8, 962, 1]" = torch.ops.aten.div.Tensor(rsqrt_2, 256);  rsqrt_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    permute_319: "f32[256, 1024]" = torch.ops.aten.permute.default(permute_6, [1, 0]);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    permute_323: "f32[1024, 256]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    div_25: "f32[8, 962, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 256);  rsqrt_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    permute_327: "f32[256, 256]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    alias_25: "f32[8, 4, 962, 64]" = torch.ops.aten.alias.default(alias);  alias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    permute_333: "f32[768, 256]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
    return [addmm_52, primals_3, primals_5, primals_11, primals_17, primals_23, primals_29, primals_35, primals_41, primals_45, primals_51, primals_57, primals_63, primals_69, primals_75, primals_81, primals_87, primals_93, primals_99, primals_105, primals_111, primals_117, primals_121, primals_127, primals_133, primals_139, primals_145, primals_151, primals_157, primals_163, primals_169, primals_173, cat, getitem_1, rsqrt, view_1, getitem_2, getitem_3, getitem_4, getitem_6, getitem_7, getitem_8, view_5, mul_2, view_7, addmm_2, view_9, mul_7, view_11, getitem_13, getitem_14, getitem_15, getitem_17, getitem_18, getitem_19, view_15, mul_9, view_17, addmm_6, view_19, mul_14, view_21, getitem_24, getitem_25, getitem_26, getitem_28, getitem_29, getitem_30, view_25, mul_16, view_27, addmm_10, view_29, view_31, view_32, cat_1, getitem_34, rsqrt_6, view_35, getitem_35, getitem_36, getitem_37, getitem_39, getitem_40, getitem_41, view_39, mul_23, view_41, addmm_14, view_43, mul_28, view_45, getitem_46, getitem_47, getitem_48, getitem_50, getitem_51, getitem_52, view_49, mul_30, view_51, addmm_18, view_53, mul_35, view_55, getitem_57, getitem_58, getitem_59, getitem_61, getitem_62, getitem_63, view_59, mul_37, view_61, addmm_22, view_63, mul_42, view_65, getitem_68, getitem_69, getitem_70, getitem_72, getitem_73, getitem_74, view_69, mul_44, view_71, addmm_26, view_73, mul_49, view_75, getitem_79, getitem_80, getitem_81, getitem_83, getitem_84, getitem_85, view_79, mul_51, view_81, addmm_30, view_83, mul_56, view_85, getitem_90, getitem_91, getitem_92, getitem_94, getitem_95, getitem_96, view_89, mul_58, view_91, addmm_34, view_93, view_95, view_96, cat_2, getitem_100, rsqrt_18, view_99, getitem_101, getitem_102, getitem_103, getitem_105, getitem_106, getitem_107, view_103, mul_65, view_105, addmm_38, view_107, mul_70, view_109, getitem_112, getitem_113, getitem_114, getitem_116, getitem_117, getitem_118, view_113, mul_72, view_115, addmm_42, view_117, mul_77, view_119, getitem_123, getitem_124, getitem_125, getitem_127, getitem_128, getitem_129, view_123, mul_79, view_125, addmm_46, view_127, mul_84, view_129, getitem_134, getitem_135, getitem_136, getitem_138, getitem_139, getitem_140, view_133, mul_86, view_135, addmm_50, view_137, mul_91, clone_41, permute_87, div, permute_91, permute_95, div_1, permute_99, alias_13, permute_105, div_2, permute_109, permute_113, div_3, permute_117, alias_14, permute_123, div_4, permute_127, permute_131, div_5, permute_135, alias_15, permute_141, div_6, permute_145, permute_149, div_7, permute_153, alias_16, permute_159, permute_166, permute_169, permute_173, div_9, permute_177, alias_17, permute_183, div_10, permute_187, permute_191, div_11, permute_195, alias_18, permute_201, div_12, permute_205, permute_209, div_13, permute_213, alias_19, permute_219, div_14, permute_223, permute_227, div_15, permute_231, alias_20, permute_237, div_16, permute_241, permute_245, div_17, permute_249, alias_21, permute_255, div_18, permute_259, permute_263, div_19, permute_267, alias_22, permute_273, permute_280, permute_283, permute_287, div_21, permute_291, alias_23, permute_297, div_22, permute_301, permute_305, div_23, permute_309, alias_24, permute_315, div_24, permute_319, permute_323, div_25, permute_327, alias_25, permute_333]
    