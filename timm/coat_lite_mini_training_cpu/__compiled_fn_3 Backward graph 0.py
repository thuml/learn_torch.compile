from __future__ import annotations



def forward(self, primals_2: "f32[64]", primals_4: "f32[64]", primals_6: "f32[64]", primals_8: "f32[64]", primals_11: "f32[128]", primals_13: "f32[128]", primals_15: "f32[128]", primals_17: "f32[128]", primals_20: "f32[320]", primals_22: "f32[320]", primals_24: "f32[320]", primals_26: "f32[320]", primals_29: "f32[512]", primals_31: "f32[512]", primals_33: "f32[512]", primals_35: "f32[512]", primals_37: "f32[512]", primals_39: "f32[64, 3, 4, 4]", primals_41: "f32[64]", primals_43: "f32[64, 1, 3, 3]", primals_47: "f32[16, 1, 3, 3]", primals_49: "f32[24, 1, 5, 5]", primals_51: "f32[24, 1, 7, 7]", primals_67: "f32[128, 64, 2, 2]", primals_69: "f32[128]", primals_71: "f32[128, 1, 3, 3]", primals_75: "f32[32, 1, 3, 3]", primals_77: "f32[48, 1, 5, 5]", primals_79: "f32[48, 1, 7, 7]", primals_95: "f32[320, 128, 2, 2]", primals_97: "f32[320]", primals_99: "f32[320, 1, 3, 3]", primals_103: "f32[80, 1, 3, 3]", primals_105: "f32[120, 1, 5, 5]", primals_107: "f32[120, 1, 7, 7]", primals_123: "f32[512, 320, 2, 2]", primals_125: "f32[512]", primals_127: "f32[512, 1, 3, 3]", primals_131: "f32[128, 1, 3, 3]", primals_133: "f32[192, 1, 5, 5]", primals_135: "f32[192, 1, 7, 7]", primals_153: "f32[8, 3, 224, 224]", mul: "f32[8, 3136, 64]", view_1: "f32[8, 64, 56, 56]", cat_1: "f32[8, 3137, 64]", getitem_3: "f32[8, 3137, 1]", rsqrt_1: "f32[8, 3137, 1]", view_3: "f32[25096, 64]", slice_8: "f32[8, 8, 3136, 8]", getitem_7: "f32[8, 16, 56, 56]", getitem_8: "f32[8, 24, 56, 56]", getitem_9: "f32[8, 24, 56, 56]", cat_2: "f32[8, 64, 56, 56]", view_15: "f32[25096, 64]", mul_6: "f32[8, 3137, 64]", view_17: "f32[25096, 64]", addmm_2: "f32[25096, 512]", view_19: "f32[25096, 512]", view_21: "f32[8, 64, 56, 56]", cat_3: "f32[8, 3137, 64]", getitem_13: "f32[8, 3137, 1]", rsqrt_3: "f32[8, 3137, 1]", view_23: "f32[25096, 64]", slice_20: "f32[8, 8, 3136, 8]", getitem_17: "f32[8, 16, 56, 56]", getitem_18: "f32[8, 24, 56, 56]", getitem_19: "f32[8, 24, 56, 56]", cat_4: "f32[8, 64, 56, 56]", view_35: "f32[25096, 64]", mul_15: "f32[8, 3137, 64]", view_37: "f32[25096, 64]", addmm_6: "f32[25096, 512]", view_39: "f32[25096, 512]", clone_15: "f32[8, 64, 56, 56]", mul_20: "f32[8, 784, 128]", view_43: "f32[8, 128, 28, 28]", cat_6: "f32[8, 785, 128]", getitem_25: "f32[8, 785, 1]", rsqrt_6: "f32[8, 785, 1]", view_45: "f32[6280, 128]", slice_35: "f32[8, 8, 784, 16]", getitem_29: "f32[8, 32, 28, 28]", getitem_30: "f32[8, 48, 28, 28]", getitem_31: "f32[8, 48, 28, 28]", cat_7: "f32[8, 128, 28, 28]", view_57: "f32[6280, 128]", mul_26: "f32[8, 785, 128]", view_59: "f32[6280, 128]", addmm_10: "f32[6280, 1024]", view_61: "f32[6280, 1024]", view_63: "f32[8, 128, 28, 28]", cat_8: "f32[8, 785, 128]", getitem_35: "f32[8, 785, 1]", rsqrt_8: "f32[8, 785, 1]", view_65: "f32[6280, 128]", slice_47: "f32[8, 8, 784, 16]", getitem_39: "f32[8, 32, 28, 28]", getitem_40: "f32[8, 48, 28, 28]", getitem_41: "f32[8, 48, 28, 28]", cat_9: "f32[8, 128, 28, 28]", view_77: "f32[6280, 128]", mul_35: "f32[8, 785, 128]", view_79: "f32[6280, 128]", addmm_14: "f32[6280, 1024]", view_81: "f32[6280, 1024]", clone_31: "f32[8, 128, 28, 28]", mul_40: "f32[8, 196, 320]", view_85: "f32[8, 320, 14, 14]", cat_11: "f32[8, 197, 320]", getitem_47: "f32[8, 197, 1]", rsqrt_11: "f32[8, 197, 1]", view_87: "f32[1576, 320]", slice_62: "f32[8, 8, 196, 40]", getitem_51: "f32[8, 80, 14, 14]", getitem_52: "f32[8, 120, 14, 14]", getitem_53: "f32[8, 120, 14, 14]", cat_12: "f32[8, 320, 14, 14]", view_99: "f32[1576, 320]", mul_46: "f32[8, 197, 320]", view_101: "f32[1576, 320]", addmm_18: "f32[1576, 1280]", view_103: "f32[1576, 1280]", view_105: "f32[8, 320, 14, 14]", cat_13: "f32[8, 197, 320]", getitem_57: "f32[8, 197, 1]", rsqrt_13: "f32[8, 197, 1]", view_107: "f32[1576, 320]", slice_74: "f32[8, 8, 196, 40]", getitem_61: "f32[8, 80, 14, 14]", getitem_62: "f32[8, 120, 14, 14]", getitem_63: "f32[8, 120, 14, 14]", cat_14: "f32[8, 320, 14, 14]", view_119: "f32[1576, 320]", mul_55: "f32[8, 197, 320]", view_121: "f32[1576, 320]", addmm_22: "f32[1576, 1280]", view_123: "f32[1576, 1280]", clone_47: "f32[8, 320, 14, 14]", mul_60: "f32[8, 49, 512]", view_127: "f32[8, 512, 7, 7]", cat_16: "f32[8, 50, 512]", getitem_69: "f32[8, 50, 1]", rsqrt_16: "f32[8, 50, 1]", view_129: "f32[400, 512]", slice_89: "f32[8, 8, 49, 64]", getitem_73: "f32[8, 128, 7, 7]", getitem_74: "f32[8, 192, 7, 7]", getitem_75: "f32[8, 192, 7, 7]", cat_17: "f32[8, 512, 7, 7]", view_141: "f32[400, 512]", mul_66: "f32[8, 50, 512]", view_143: "f32[400, 512]", addmm_26: "f32[400, 2048]", view_145: "f32[400, 2048]", view_147: "f32[8, 512, 7, 7]", cat_18: "f32[8, 50, 512]", getitem_79: "f32[8, 50, 1]", rsqrt_18: "f32[8, 50, 1]", view_149: "f32[400, 512]", slice_101: "f32[8, 8, 49, 64]", getitem_83: "f32[8, 128, 7, 7]", getitem_84: "f32[8, 192, 7, 7]", getitem_85: "f32[8, 192, 7, 7]", cat_19: "f32[8, 512, 7, 7]", view_161: "f32[400, 512]", mul_75: "f32[8, 50, 512]", view_163: "f32[400, 512]", addmm_30: "f32[400, 2048]", view_165: "f32[400, 2048]", mul_80: "f32[8, 50, 512]", clone_64: "f32[8, 512]", permute_97: "f32[1000, 512]", div_8: "f32[8, 50, 1]", permute_101: "f32[512, 2048]", permute_105: "f32[2048, 512]", div_9: "f32[8, 50, 1]", permute_109: "f32[512, 512]", permute_116: "f32[64, 64, 50]", permute_117: "f32[64, 64, 64]", permute_118: "f32[64, 50, 64]", permute_119: "f32[64, 64, 50]", alias_8: "f32[8, 8, 50, 64]", permute_122: "f32[1536, 512]", permute_128: "f32[512, 2048]", permute_132: "f32[2048, 512]", div_11: "f32[8, 50, 1]", permute_136: "f32[512, 512]", permute_143: "f32[64, 64, 50]", permute_144: "f32[64, 64, 64]", permute_145: "f32[64, 50, 64]", permute_146: "f32[64, 64, 50]", alias_9: "f32[8, 8, 50, 64]", permute_149: "f32[1536, 512]", div_13: "f32[8, 49, 1]", permute_157: "f32[320, 1280]", permute_161: "f32[1280, 320]", div_14: "f32[8, 197, 1]", permute_165: "f32[320, 320]", permute_172: "f32[64, 40, 197]", permute_173: "f32[64, 40, 40]", permute_174: "f32[64, 197, 40]", permute_175: "f32[64, 40, 197]", alias_10: "f32[8, 8, 197, 40]", permute_178: "f32[960, 320]", permute_184: "f32[320, 1280]", permute_188: "f32[1280, 320]", div_16: "f32[8, 197, 1]", permute_192: "f32[320, 320]", permute_199: "f32[64, 40, 197]", permute_200: "f32[64, 40, 40]", permute_201: "f32[64, 197, 40]", permute_202: "f32[64, 40, 197]", alias_11: "f32[8, 8, 197, 40]", permute_205: "f32[960, 320]", div_18: "f32[8, 196, 1]", permute_213: "f32[128, 1024]", permute_217: "f32[1024, 128]", div_19: "f32[8, 785, 1]", permute_221: "f32[128, 128]", permute_228: "f32[64, 16, 785]", permute_229: "f32[64, 16, 16]", permute_230: "f32[64, 785, 16]", permute_231: "f32[64, 16, 785]", alias_12: "f32[8, 8, 785, 16]", permute_234: "f32[384, 128]", permute_240: "f32[128, 1024]", permute_244: "f32[1024, 128]", div_21: "f32[8, 785, 1]", permute_248: "f32[128, 128]", permute_255: "f32[64, 16, 785]", permute_256: "f32[64, 16, 16]", permute_257: "f32[64, 785, 16]", permute_258: "f32[64, 16, 785]", alias_13: "f32[8, 8, 785, 16]", permute_261: "f32[384, 128]", div_23: "f32[8, 784, 1]", permute_269: "f32[64, 512]", permute_273: "f32[512, 64]", div_24: "f32[8, 3137, 1]", permute_277: "f32[64, 64]", permute_284: "f32[64, 8, 3137]", permute_285: "f32[64, 8, 8]", permute_286: "f32[64, 3137, 8]", permute_287: "f32[64, 8, 3137]", alias_14: "f32[8, 8, 3137, 8]", permute_290: "f32[192, 64]", permute_296: "f32[64, 512]", permute_300: "f32[512, 64]", div_26: "f32[8, 3137, 1]", permute_304: "f32[64, 64]", permute_311: "f32[64, 8, 3137]", permute_312: "f32[64, 8, 8]", permute_313: "f32[64, 3137, 8]", permute_314: "f32[64, 8, 3137]", alias_15: "f32[8, 8, 3137, 8]", permute_317: "f32[192, 64]", div_28: "f32[8, 3136, 1]", tangents_1: "f32[8, 1000]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_1: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(cat_1, getitem_3);  cat_1 = getitem_3 = None
    mul_2: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_13: "f32[8, 8, 8, 3136]" = torch.ops.aten.view.default(cat_2, [8, 8, 8, 3136]);  cat_2 = None
    permute_7: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_13, [0, 1, 3, 2]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_18: "f32[8, 3137, 512]" = torch.ops.aten.view.default(addmm_2, [8, 3137, 512]);  addmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_9: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_18, 0.7071067811865476)
    erf: "f32[8, 3137, 512]" = torch.ops.aten.erf.default(mul_9);  mul_9 = None
    add_9: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_4: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(cat_3, getitem_13);  cat_3 = getitem_13 = None
    mul_11: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_3);  sub_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_33: "f32[8, 8, 8, 3136]" = torch.ops.aten.view.default(cat_4, [8, 8, 8, 3136]);  cat_4 = None
    permute_18: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_33, [0, 1, 3, 2]);  view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_38: "f32[8, 3137, 512]" = torch.ops.aten.view.default(addmm_6, [8, 3137, 512]);  addmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_18: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_38, 0.7071067811865476)
    erf_1: "f32[8, 3137, 512]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_18: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_8: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(cat_6, getitem_25);  cat_6 = getitem_25 = None
    mul_22: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_55: "f32[8, 8, 16, 784]" = torch.ops.aten.view.default(cat_7, [8, 8, 16, 784]);  cat_7 = None
    permute_31: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_55, [0, 1, 3, 2]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_60: "f32[8, 785, 1024]" = torch.ops.aten.view.default(addmm_10, [8, 785, 1024]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_29: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_60, 0.7071067811865476)
    erf_2: "f32[8, 785, 1024]" = torch.ops.aten.erf.default(mul_29);  mul_29 = None
    add_29: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_11: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(cat_8, getitem_35);  cat_8 = getitem_35 = None
    mul_31: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_75: "f32[8, 8, 16, 784]" = torch.ops.aten.view.default(cat_9, [8, 8, 16, 784]);  cat_9 = None
    permute_42: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_75, [0, 1, 3, 2]);  view_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_80: "f32[8, 785, 1024]" = torch.ops.aten.view.default(addmm_14, [8, 785, 1024]);  addmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_38: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_80, 0.7071067811865476)
    erf_3: "f32[8, 785, 1024]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_38: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_15: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(cat_11, getitem_47);  cat_11 = getitem_47 = None
    mul_42: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_97: "f32[8, 8, 40, 196]" = torch.ops.aten.view.default(cat_12, [8, 8, 40, 196]);  cat_12 = None
    permute_55: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_97, [0, 1, 3, 2]);  view_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_102: "f32[8, 197, 1280]" = torch.ops.aten.view.default(addmm_18, [8, 197, 1280]);  addmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_49: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_102, 0.7071067811865476)
    erf_4: "f32[8, 197, 1280]" = torch.ops.aten.erf.default(mul_49);  mul_49 = None
    add_49: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_18: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(cat_13, getitem_57);  cat_13 = getitem_57 = None
    mul_51: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_117: "f32[8, 8, 40, 196]" = torch.ops.aten.view.default(cat_14, [8, 8, 40, 196]);  cat_14 = None
    permute_66: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_117, [0, 1, 3, 2]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_122: "f32[8, 197, 1280]" = torch.ops.aten.view.default(addmm_22, [8, 197, 1280]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_58: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_122, 0.7071067811865476)
    erf_5: "f32[8, 197, 1280]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_58: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_22: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(cat_16, getitem_69);  cat_16 = getitem_69 = None
    mul_62: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_16);  sub_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_139: "f32[8, 8, 64, 49]" = torch.ops.aten.view.default(cat_17, [8, 8, 64, 49]);  cat_17 = None
    permute_79: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_139, [0, 1, 3, 2]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_144: "f32[8, 50, 2048]" = torch.ops.aten.view.default(addmm_26, [8, 50, 2048]);  addmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_69: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_144, 0.7071067811865476)
    erf_6: "f32[8, 50, 2048]" = torch.ops.aten.erf.default(mul_69);  mul_69 = None
    add_69: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    sub_25: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(cat_18, getitem_79);  cat_18 = getitem_79 = None
    mul_71: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_18);  sub_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    view_159: "f32[8, 8, 64, 49]" = torch.ops.aten.view.default(cat_19, [8, 8, 64, 49]);  cat_19 = None
    permute_90: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_159, [0, 1, 3, 2]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_164: "f32[8, 50, 2048]" = torch.ops.aten.view.default(addmm_30, [8, 50, 2048]);  addmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_78: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_164, 0.7071067811865476)
    erf_7: "f32[8, 50, 2048]" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
    add_78: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:662, code: return x if pre_logits else self.head(x)
    mm: "f32[8, 512]" = torch.ops.aten.mm.default(tangents_1, permute_97);  permute_97 = None
    permute_98: "f32[1000, 8]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
    mm_1: "f32[1000, 512]" = torch.ops.aten.mm.default(permute_98, clone_64);  permute_98 = clone_64 = None
    permute_99: "f32[512, 1000]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_9: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
    view_168: "f32[1000]" = torch.ops.aten.view.default(sum_9, [1000]);  sum_9 = None
    permute_100: "f32[1000, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:660, code: x = x_feat[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x_feat[:, 0]
    full_default: "f32[8, 50, 512]" = torch.ops.aten.full.default([8, 50, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    select_scatter: "f32[8, 50, 512]" = torch.ops.aten.select_scatter.default(full_default, mm, 1, 0);  mm = None
    slice_scatter: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_default, select_scatter, 0, 0, 9223372036854775807);  select_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_83: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(slice_scatter, primals_37);  primals_37 = None
    mul_84: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_83, 512)
    sum_10: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_83, [2], True)
    mul_85: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_83, mul_80);  mul_83 = None
    sum_11: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_85, [2], True);  mul_85 = None
    mul_86: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_80, sum_11);  sum_11 = None
    sub_30: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(mul_84, sum_10);  mul_84 = sum_10 = None
    sub_31: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(sub_30, mul_86);  sub_30 = mul_86 = None
    mul_87: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(div_8, sub_31);  div_8 = sub_31 = None
    mul_88: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(slice_scatter, mul_80);  mul_80 = None
    sum_12: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_88, [0, 1]);  mul_88 = None
    sum_13: "f32[512]" = torch.ops.aten.sum.dim_IntList(slice_scatter, [0, 1]);  slice_scatter = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_169: "f32[400, 512]" = torch.ops.aten.view.default(mul_87, [400, 512])
    mm_2: "f32[400, 2048]" = torch.ops.aten.mm.default(view_169, permute_101);  permute_101 = None
    permute_102: "f32[512, 400]" = torch.ops.aten.permute.default(view_169, [1, 0])
    mm_3: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_102, view_165);  permute_102 = view_165 = None
    permute_103: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_14: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_169, [0], True);  view_169 = None
    view_170: "f32[512]" = torch.ops.aten.view.default(sum_14, [512]);  sum_14 = None
    permute_104: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    view_171: "f32[8, 50, 2048]" = torch.ops.aten.view.default(mm_2, [8, 50, 2048]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_90: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(add_78, 0.5);  add_78 = None
    mul_91: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_164, view_164)
    mul_92: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(mul_91, -0.5);  mul_91 = None
    exp_8: "f32[8, 50, 2048]" = torch.ops.aten.exp.default(mul_92);  mul_92 = None
    mul_93: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(exp_8, 0.3989422804014327);  exp_8 = None
    mul_94: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_164, mul_93);  view_164 = mul_93 = None
    add_83: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(mul_90, mul_94);  mul_90 = mul_94 = None
    mul_95: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_171, add_83);  view_171 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_172: "f32[400, 2048]" = torch.ops.aten.view.default(mul_95, [400, 2048]);  mul_95 = None
    mm_4: "f32[400, 512]" = torch.ops.aten.mm.default(view_172, permute_105);  permute_105 = None
    permute_106: "f32[2048, 400]" = torch.ops.aten.permute.default(view_172, [1, 0])
    mm_5: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_106, view_163);  permute_106 = view_163 = None
    permute_107: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_15: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_172, [0], True);  view_172 = None
    view_173: "f32[2048]" = torch.ops.aten.view.default(sum_15, [2048]);  sum_15 = None
    permute_108: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    view_174: "f32[8, 50, 512]" = torch.ops.aten.view.default(mm_4, [8, 50, 512]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_97: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_174, primals_35);  primals_35 = None
    mul_98: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_97, 512)
    sum_16: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_97, [2], True)
    mul_99: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_97, mul_75);  mul_97 = None
    sum_17: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_99, [2], True);  mul_99 = None
    mul_100: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_75, sum_17);  sum_17 = None
    sub_33: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(mul_98, sum_16);  mul_98 = sum_16 = None
    sub_34: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(sub_33, mul_100);  sub_33 = mul_100 = None
    mul_101: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(div_9, sub_34);  div_9 = sub_34 = None
    mul_102: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_174, mul_75);  mul_75 = None
    sum_18: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_102, [0, 1]);  mul_102 = None
    sum_19: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_174, [0, 1]);  view_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_84: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(mul_87, mul_101);  mul_87 = mul_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_175: "f32[400, 512]" = torch.ops.aten.view.default(add_84, [400, 512])
    mm_6: "f32[400, 512]" = torch.ops.aten.mm.default(view_175, permute_109);  permute_109 = None
    permute_110: "f32[512, 400]" = torch.ops.aten.permute.default(view_175, [1, 0])
    mm_7: "f32[512, 512]" = torch.ops.aten.mm.default(permute_110, view_161);  permute_110 = view_161 = None
    permute_111: "f32[512, 512]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_20: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_175, [0], True);  view_175 = None
    view_176: "f32[512]" = torch.ops.aten.view.default(sum_20, [512]);  sum_20 = None
    permute_112: "f32[512, 512]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    view_177: "f32[8, 50, 512]" = torch.ops.aten.view.default(mm_6, [8, 50, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_178: "f32[8, 50, 8, 64]" = torch.ops.aten.view.default(view_177, [8, 50, 8, 64]);  view_177 = None
    permute_113: "f32[8, 8, 50, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_103: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(permute_113, 0.125)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_8: "f32[8, 8, 49, 64]" = torch.ops.aten.constant_pad_nd.default(permute_113, [0, 0, -1, 0, 0, 0]);  permute_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_104: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(constant_pad_nd_8, slice_101);  slice_101 = None
    mul_105: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(constant_pad_nd_8, permute_90);  constant_pad_nd_8 = permute_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_114: "f32[8, 8, 64, 49]" = torch.ops.aten.permute.default(mul_104, [0, 1, 3, 2]);  mul_104 = None
    view_179: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_114, [8, 512, 7, 7]);  permute_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_110: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(view_179, 1, 0, 128)
    slice_111: "f32[8, 192, 7, 7]" = torch.ops.aten.slice.Tensor(view_179, 1, 128, 320)
    slice_112: "f32[8, 192, 7, 7]" = torch.ops.aten.slice.Tensor(view_179, 1, 320, 512);  view_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward = torch.ops.aten.convolution_backward.default(slice_112, getitem_85, primals_135, [192], [1, 1], [3, 3], [1, 1], False, [0, 0], 192, [True, True, True]);  slice_112 = getitem_85 = None
    getitem_90: "f32[8, 192, 7, 7]" = convolution_backward[0]
    getitem_91: "f32[192, 1, 7, 7]" = convolution_backward[1]
    getitem_92: "f32[192]" = convolution_backward[2];  convolution_backward = None
    convolution_backward_1 = torch.ops.aten.convolution_backward.default(slice_111, getitem_84, primals_133, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 192, [True, True, True]);  slice_111 = getitem_84 = None
    getitem_93: "f32[8, 192, 7, 7]" = convolution_backward_1[0]
    getitem_94: "f32[192, 1, 5, 5]" = convolution_backward_1[1]
    getitem_95: "f32[192]" = convolution_backward_1[2];  convolution_backward_1 = None
    convolution_backward_2 = torch.ops.aten.convolution_backward.default(slice_110, getitem_83, primals_131, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, True]);  slice_110 = getitem_83 = None
    getitem_96: "f32[8, 128, 7, 7]" = convolution_backward_2[0]
    getitem_97: "f32[128, 1, 3, 3]" = convolution_backward_2[1]
    getitem_98: "f32[128]" = convolution_backward_2[2];  convolution_backward_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_20: "f32[8, 512, 7, 7]" = torch.ops.aten.cat.default([getitem_96, getitem_93, getitem_90], 1);  getitem_96 = getitem_93 = getitem_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_180: "f32[8, 8, 64, 49]" = torch.ops.aten.view.default(cat_20, [8, 8, 64, 49]);  cat_20 = None
    permute_115: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_180, [0, 1, 3, 2]);  view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_default_2: "f32[8, 8, 49, 64]" = torch.ops.aten.full.default([8, 8, 49, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_1: "f32[8, 8, 49, 64]" = torch.ops.aten.slice_scatter.default(full_default_2, permute_115, 3, 0, 9223372036854775807);  permute_115 = None
    full_default_3: "f32[8, 8, 50, 64]" = torch.ops.aten.full.default([8, 8, 50, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_2: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_1, 2, 1, 9223372036854775807);  slice_scatter_1 = None
    slice_scatter_3: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_2, 1, 0, 9223372036854775807);  slice_scatter_2 = None
    slice_scatter_4: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_3, 0, 0, 9223372036854775807);  slice_scatter_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_scatter_5: "f32[8, 8, 49, 64]" = torch.ops.aten.slice_scatter.default(full_default_2, mul_105, 3, 0, 9223372036854775807);  mul_105 = None
    slice_scatter_6: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_5, 2, 1, 9223372036854775807);  slice_scatter_5 = None
    slice_scatter_7: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_6, 1, 0, 9223372036854775807);  slice_scatter_6 = None
    slice_scatter_8: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_7, 0, 0, 9223372036854775807);  slice_scatter_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_65: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(mul_103, memory_format = torch.contiguous_format);  mul_103 = None
    view_181: "f32[64, 50, 64]" = torch.ops.aten.view.default(clone_65, [64, 50, 64]);  clone_65 = None
    bmm_16: "f32[64, 64, 64]" = torch.ops.aten.bmm.default(permute_116, view_181);  permute_116 = None
    bmm_17: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(view_181, permute_117);  view_181 = permute_117 = None
    view_182: "f32[8, 8, 64, 64]" = torch.ops.aten.view.default(bmm_16, [8, 8, 64, 64]);  bmm_16 = None
    view_183: "f32[8, 8, 50, 64]" = torch.ops.aten.view.default(bmm_17, [8, 8, 50, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_85: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(slice_scatter_8, view_183);  slice_scatter_8 = view_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_184: "f32[64, 64, 64]" = torch.ops.aten.view.default(view_182, [64, 64, 64]);  view_182 = None
    bmm_18: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(permute_118, view_184);  permute_118 = None
    bmm_19: "f32[64, 64, 50]" = torch.ops.aten.bmm.default(view_184, permute_119);  view_184 = permute_119 = None
    view_185: "f32[8, 8, 50, 64]" = torch.ops.aten.view.default(bmm_18, [8, 8, 50, 64]);  bmm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_86: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(slice_scatter_4, view_185);  slice_scatter_4 = view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_186: "f32[8, 8, 64, 50]" = torch.ops.aten.view.default(bmm_19, [8, 8, 64, 50]);  bmm_19 = None
    permute_120: "f32[8, 8, 50, 64]" = torch.ops.aten.permute.default(view_186, [0, 1, 3, 2]);  view_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    mul_106: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(permute_120, alias_8);  permute_120 = None
    sum_21: "f32[8, 8, 1, 64]" = torch.ops.aten.sum.dim_IntList(mul_106, [2], True)
    mul_107: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(alias_8, sum_21);  alias_8 = sum_21 = None
    sub_35: "f32[8, 8, 50, 64]" = torch.ops.aten.sub.Tensor(mul_106, mul_107);  mul_106 = mul_107 = None
    clone_66: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(sub_35, memory_format = torch.contiguous_format);  sub_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_21: "f32[24, 8, 50, 64]" = torch.ops.aten.cat.default([add_85, clone_66, add_86]);  add_85 = clone_66 = add_86 = None
    view_187: "f32[3, 8, 8, 50, 64]" = torch.ops.aten.view.default(cat_21, [3, 8, 8, 50, 64]);  cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_121: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.permute.default(view_187, [1, 3, 0, 2, 4]);  view_187 = None
    clone_67: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.clone.default(permute_121, memory_format = torch.contiguous_format);  permute_121 = None
    view_188: "f32[8, 50, 1536]" = torch.ops.aten.view.default(clone_67, [8, 50, 1536]);  clone_67 = None
    view_189: "f32[400, 1536]" = torch.ops.aten.view.default(view_188, [400, 1536]);  view_188 = None
    mm_8: "f32[400, 512]" = torch.ops.aten.mm.default(view_189, permute_122);  permute_122 = None
    permute_123: "f32[1536, 400]" = torch.ops.aten.permute.default(view_189, [1, 0])
    mm_9: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_123, view_149);  permute_123 = view_149 = None
    permute_124: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_22: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_189, [0], True);  view_189 = None
    view_190: "f32[1536]" = torch.ops.aten.view.default(sum_22, [1536]);  sum_22 = None
    permute_125: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_124, [1, 0]);  permute_124 = None
    view_191: "f32[8, 50, 512]" = torch.ops.aten.view.default(mm_8, [8, 50, 512]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_109: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_191, primals_33);  primals_33 = None
    mul_110: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_109, 512)
    sum_23: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True)
    mul_111: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_109, mul_71);  mul_109 = None
    sum_24: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_111, [2], True);  mul_111 = None
    mul_112: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_71, sum_24);  sum_24 = None
    sub_37: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(mul_110, sum_23);  mul_110 = sum_23 = None
    sub_38: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(sub_37, mul_112);  sub_37 = mul_112 = None
    div_10: "f32[8, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_18, 512);  rsqrt_18 = None
    mul_113: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(div_10, sub_38);  div_10 = sub_38 = None
    mul_114: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_191, mul_71);  mul_71 = None
    sum_25: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_114, [0, 1]);  mul_114 = None
    sum_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_191, [0, 1]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_87: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_84, mul_113);  add_84 = mul_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_113: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(add_87, 1, 0, 1)
    slice_114: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(add_87, 1, 1, 50);  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_126: "f32[8, 512, 49]" = torch.ops.aten.permute.default(slice_114, [0, 2, 1]);  slice_114 = None
    view_192: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_126, [8, 512, 7, 7]);  permute_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_3 = torch.ops.aten.convolution_backward.default(view_192, view_147, primals_127, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 512, [True, True, True]);  view_147 = None
    getitem_99: "f32[8, 512, 7, 7]" = convolution_backward_3[0]
    getitem_100: "f32[512, 1, 3, 3]" = convolution_backward_3[1]
    getitem_101: "f32[512]" = convolution_backward_3[2];  convolution_backward_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_88: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(view_192, getitem_99);  view_192 = getitem_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_193: "f32[8, 512, 49]" = torch.ops.aten.view.default(add_88, [8, 512, 49]);  add_88 = None
    permute_127: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_scatter_9: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_default, permute_127, 1, 1, 9223372036854775807);  permute_127 = None
    slice_scatter_10: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_9, 0, 0, 9223372036854775807);  slice_scatter_9 = None
    slice_scatter_11: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_default, slice_113, 1, 0, 1);  slice_113 = None
    slice_scatter_12: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_11, 0, 0, 9223372036854775807);  slice_scatter_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_89: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(slice_scatter_10, slice_scatter_12);  slice_scatter_10 = slice_scatter_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_194: "f32[400, 512]" = torch.ops.aten.view.default(add_89, [400, 512])
    mm_10: "f32[400, 2048]" = torch.ops.aten.mm.default(view_194, permute_128);  permute_128 = None
    permute_129: "f32[512, 400]" = torch.ops.aten.permute.default(view_194, [1, 0])
    mm_11: "f32[512, 2048]" = torch.ops.aten.mm.default(permute_129, view_145);  permute_129 = view_145 = None
    permute_130: "f32[2048, 512]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_27: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_194, [0], True);  view_194 = None
    view_195: "f32[512]" = torch.ops.aten.view.default(sum_27, [512]);  sum_27 = None
    permute_131: "f32[512, 2048]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    view_196: "f32[8, 50, 2048]" = torch.ops.aten.view.default(mm_10, [8, 50, 2048]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_116: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(add_69, 0.5);  add_69 = None
    mul_117: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_144, view_144)
    mul_118: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(mul_117, -0.5);  mul_117 = None
    exp_9: "f32[8, 50, 2048]" = torch.ops.aten.exp.default(mul_118);  mul_118 = None
    mul_119: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(exp_9, 0.3989422804014327);  exp_9 = None
    mul_120: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_144, mul_119);  view_144 = mul_119 = None
    add_91: "f32[8, 50, 2048]" = torch.ops.aten.add.Tensor(mul_116, mul_120);  mul_116 = mul_120 = None
    mul_121: "f32[8, 50, 2048]" = torch.ops.aten.mul.Tensor(view_196, add_91);  view_196 = add_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_197: "f32[400, 2048]" = torch.ops.aten.view.default(mul_121, [400, 2048]);  mul_121 = None
    mm_12: "f32[400, 512]" = torch.ops.aten.mm.default(view_197, permute_132);  permute_132 = None
    permute_133: "f32[2048, 400]" = torch.ops.aten.permute.default(view_197, [1, 0])
    mm_13: "f32[2048, 512]" = torch.ops.aten.mm.default(permute_133, view_143);  permute_133 = view_143 = None
    permute_134: "f32[512, 2048]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_28: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_197, [0], True);  view_197 = None
    view_198: "f32[2048]" = torch.ops.aten.view.default(sum_28, [2048]);  sum_28 = None
    permute_135: "f32[2048, 512]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    view_199: "f32[8, 50, 512]" = torch.ops.aten.view.default(mm_12, [8, 50, 512]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_123: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_199, primals_31);  primals_31 = None
    mul_124: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_123, 512)
    sum_29: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [2], True)
    mul_125: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_123, mul_66);  mul_123 = None
    sum_30: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_125, [2], True);  mul_125 = None
    mul_126: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_66, sum_30);  sum_30 = None
    sub_40: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(mul_124, sum_29);  mul_124 = sum_29 = None
    sub_41: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(sub_40, mul_126);  sub_40 = mul_126 = None
    mul_127: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(div_11, sub_41);  div_11 = sub_41 = None
    mul_128: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_199, mul_66);  mul_66 = None
    sum_31: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_128, [0, 1]);  mul_128 = None
    sum_32: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_199, [0, 1]);  view_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_92: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_89, mul_127);  add_89 = mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_200: "f32[400, 512]" = torch.ops.aten.view.default(add_92, [400, 512])
    mm_14: "f32[400, 512]" = torch.ops.aten.mm.default(view_200, permute_136);  permute_136 = None
    permute_137: "f32[512, 400]" = torch.ops.aten.permute.default(view_200, [1, 0])
    mm_15: "f32[512, 512]" = torch.ops.aten.mm.default(permute_137, view_141);  permute_137 = view_141 = None
    permute_138: "f32[512, 512]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_33: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_200, [0], True);  view_200 = None
    view_201: "f32[512]" = torch.ops.aten.view.default(sum_33, [512]);  sum_33 = None
    permute_139: "f32[512, 512]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    view_202: "f32[8, 50, 512]" = torch.ops.aten.view.default(mm_14, [8, 50, 512]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_203: "f32[8, 50, 8, 64]" = torch.ops.aten.view.default(view_202, [8, 50, 8, 64]);  view_202 = None
    permute_140: "f32[8, 8, 50, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_129: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(permute_140, 0.125)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_9: "f32[8, 8, 49, 64]" = torch.ops.aten.constant_pad_nd.default(permute_140, [0, 0, -1, 0, 0, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_130: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(constant_pad_nd_9, slice_89);  slice_89 = None
    mul_131: "f32[8, 8, 49, 64]" = torch.ops.aten.mul.Tensor(constant_pad_nd_9, permute_79);  constant_pad_nd_9 = permute_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_141: "f32[8, 8, 64, 49]" = torch.ops.aten.permute.default(mul_130, [0, 1, 3, 2]);  mul_130 = None
    view_204: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_141, [8, 512, 7, 7]);  permute_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_115: "f32[8, 128, 7, 7]" = torch.ops.aten.slice.Tensor(view_204, 1, 0, 128)
    slice_116: "f32[8, 192, 7, 7]" = torch.ops.aten.slice.Tensor(view_204, 1, 128, 320)
    slice_117: "f32[8, 192, 7, 7]" = torch.ops.aten.slice.Tensor(view_204, 1, 320, 512);  view_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_4 = torch.ops.aten.convolution_backward.default(slice_117, getitem_75, primals_135, [192], [1, 1], [3, 3], [1, 1], False, [0, 0], 192, [True, True, True]);  slice_117 = getitem_75 = primals_135 = None
    getitem_102: "f32[8, 192, 7, 7]" = convolution_backward_4[0]
    getitem_103: "f32[192, 1, 7, 7]" = convolution_backward_4[1]
    getitem_104: "f32[192]" = convolution_backward_4[2];  convolution_backward_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_93: "f32[192, 1, 7, 7]" = torch.ops.aten.add.Tensor(getitem_91, getitem_103);  getitem_91 = getitem_103 = None
    add_94: "f32[192]" = torch.ops.aten.add.Tensor(getitem_92, getitem_104);  getitem_92 = getitem_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_5 = torch.ops.aten.convolution_backward.default(slice_116, getitem_74, primals_133, [192], [1, 1], [2, 2], [1, 1], False, [0, 0], 192, [True, True, True]);  slice_116 = getitem_74 = primals_133 = None
    getitem_105: "f32[8, 192, 7, 7]" = convolution_backward_5[0]
    getitem_106: "f32[192, 1, 5, 5]" = convolution_backward_5[1]
    getitem_107: "f32[192]" = convolution_backward_5[2];  convolution_backward_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_95: "f32[192, 1, 5, 5]" = torch.ops.aten.add.Tensor(getitem_94, getitem_106);  getitem_94 = getitem_106 = None
    add_96: "f32[192]" = torch.ops.aten.add.Tensor(getitem_95, getitem_107);  getitem_95 = getitem_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_6 = torch.ops.aten.convolution_backward.default(slice_115, getitem_73, primals_131, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, True]);  slice_115 = getitem_73 = primals_131 = None
    getitem_108: "f32[8, 128, 7, 7]" = convolution_backward_6[0]
    getitem_109: "f32[128, 1, 3, 3]" = convolution_backward_6[1]
    getitem_110: "f32[128]" = convolution_backward_6[2];  convolution_backward_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_97: "f32[128, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_97, getitem_109);  getitem_97 = getitem_109 = None
    add_98: "f32[128]" = torch.ops.aten.add.Tensor(getitem_98, getitem_110);  getitem_98 = getitem_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_22: "f32[8, 512, 7, 7]" = torch.ops.aten.cat.default([getitem_108, getitem_105, getitem_102], 1);  getitem_108 = getitem_105 = getitem_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_205: "f32[8, 8, 64, 49]" = torch.ops.aten.view.default(cat_22, [8, 8, 64, 49]);  cat_22 = None
    permute_142: "f32[8, 8, 49, 64]" = torch.ops.aten.permute.default(view_205, [0, 1, 3, 2]);  view_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_scatter_13: "f32[8, 8, 49, 64]" = torch.ops.aten.slice_scatter.default(full_default_2, permute_142, 3, 0, 9223372036854775807);  permute_142 = None
    slice_scatter_14: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_13, 2, 1, 9223372036854775807);  slice_scatter_13 = None
    slice_scatter_15: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_14, 1, 0, 9223372036854775807);  slice_scatter_14 = None
    slice_scatter_16: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_15, 0, 0, 9223372036854775807);  slice_scatter_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_scatter_17: "f32[8, 8, 49, 64]" = torch.ops.aten.slice_scatter.default(full_default_2, mul_131, 3, 0, 9223372036854775807);  full_default_2 = mul_131 = None
    slice_scatter_18: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_17, 2, 1, 9223372036854775807);  slice_scatter_17 = None
    slice_scatter_19: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_18, 1, 0, 9223372036854775807);  slice_scatter_18 = None
    slice_scatter_20: "f32[8, 8, 50, 64]" = torch.ops.aten.slice_scatter.default(full_default_3, slice_scatter_19, 0, 0, 9223372036854775807);  full_default_3 = slice_scatter_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_68: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(mul_129, memory_format = torch.contiguous_format);  mul_129 = None
    view_206: "f32[64, 50, 64]" = torch.ops.aten.view.default(clone_68, [64, 50, 64]);  clone_68 = None
    bmm_20: "f32[64, 64, 64]" = torch.ops.aten.bmm.default(permute_143, view_206);  permute_143 = None
    bmm_21: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(view_206, permute_144);  view_206 = permute_144 = None
    view_207: "f32[8, 8, 64, 64]" = torch.ops.aten.view.default(bmm_20, [8, 8, 64, 64]);  bmm_20 = None
    view_208: "f32[8, 8, 50, 64]" = torch.ops.aten.view.default(bmm_21, [8, 8, 50, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_99: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(slice_scatter_20, view_208);  slice_scatter_20 = view_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_209: "f32[64, 64, 64]" = torch.ops.aten.view.default(view_207, [64, 64, 64]);  view_207 = None
    bmm_22: "f32[64, 50, 64]" = torch.ops.aten.bmm.default(permute_145, view_209);  permute_145 = None
    bmm_23: "f32[64, 64, 50]" = torch.ops.aten.bmm.default(view_209, permute_146);  view_209 = permute_146 = None
    view_210: "f32[8, 8, 50, 64]" = torch.ops.aten.view.default(bmm_22, [8, 8, 50, 64]);  bmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_100: "f32[8, 8, 50, 64]" = torch.ops.aten.add.Tensor(slice_scatter_16, view_210);  slice_scatter_16 = view_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_211: "f32[8, 8, 64, 50]" = torch.ops.aten.view.default(bmm_23, [8, 8, 64, 50]);  bmm_23 = None
    permute_147: "f32[8, 8, 50, 64]" = torch.ops.aten.permute.default(view_211, [0, 1, 3, 2]);  view_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    mul_132: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(permute_147, alias_9);  permute_147 = None
    sum_34: "f32[8, 8, 1, 64]" = torch.ops.aten.sum.dim_IntList(mul_132, [2], True)
    mul_133: "f32[8, 8, 50, 64]" = torch.ops.aten.mul.Tensor(alias_9, sum_34);  alias_9 = sum_34 = None
    sub_42: "f32[8, 8, 50, 64]" = torch.ops.aten.sub.Tensor(mul_132, mul_133);  mul_132 = mul_133 = None
    clone_69: "f32[8, 8, 50, 64]" = torch.ops.aten.clone.default(sub_42, memory_format = torch.contiguous_format);  sub_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_23: "f32[24, 8, 50, 64]" = torch.ops.aten.cat.default([add_99, clone_69, add_100]);  add_99 = clone_69 = add_100 = None
    view_212: "f32[3, 8, 8, 50, 64]" = torch.ops.aten.view.default(cat_23, [3, 8, 8, 50, 64]);  cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_148: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.permute.default(view_212, [1, 3, 0, 2, 4]);  view_212 = None
    clone_70: "f32[8, 50, 3, 8, 64]" = torch.ops.aten.clone.default(permute_148, memory_format = torch.contiguous_format);  permute_148 = None
    view_213: "f32[8, 50, 1536]" = torch.ops.aten.view.default(clone_70, [8, 50, 1536]);  clone_70 = None
    view_214: "f32[400, 1536]" = torch.ops.aten.view.default(view_213, [400, 1536]);  view_213 = None
    mm_16: "f32[400, 512]" = torch.ops.aten.mm.default(view_214, permute_149);  permute_149 = None
    permute_150: "f32[1536, 400]" = torch.ops.aten.permute.default(view_214, [1, 0])
    mm_17: "f32[1536, 512]" = torch.ops.aten.mm.default(permute_150, view_129);  permute_150 = view_129 = None
    permute_151: "f32[512, 1536]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_35: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_214, [0], True);  view_214 = None
    view_215: "f32[1536]" = torch.ops.aten.view.default(sum_35, [1536]);  sum_35 = None
    permute_152: "f32[1536, 512]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    view_216: "f32[8, 50, 512]" = torch.ops.aten.view.default(mm_16, [8, 50, 512]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_135: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_216, primals_29);  primals_29 = None
    mul_136: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_135, 512)
    sum_36: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_135, [2], True)
    mul_137: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_135, mul_62);  mul_135 = None
    sum_37: "f32[8, 50, 1]" = torch.ops.aten.sum.dim_IntList(mul_137, [2], True);  mul_137 = None
    mul_138: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(mul_62, sum_37);  sum_37 = None
    sub_44: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(mul_136, sum_36);  mul_136 = sum_36 = None
    sub_45: "f32[8, 50, 512]" = torch.ops.aten.sub.Tensor(sub_44, mul_138);  sub_44 = mul_138 = None
    div_12: "f32[8, 50, 1]" = torch.ops.aten.div.Tensor(rsqrt_16, 512);  rsqrt_16 = None
    mul_139: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(div_12, sub_45);  div_12 = sub_45 = None
    mul_140: "f32[8, 50, 512]" = torch.ops.aten.mul.Tensor(view_216, mul_62);  mul_62 = None
    sum_38: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_140, [0, 1]);  mul_140 = None
    sum_39: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_216, [0, 1]);  view_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_101: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(add_92, mul_139);  add_92 = mul_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_118: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(add_101, 1, 0, 1)
    slice_119: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(add_101, 1, 1, 50);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_153: "f32[8, 512, 49]" = torch.ops.aten.permute.default(slice_119, [0, 2, 1]);  slice_119 = None
    view_217: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_153, [8, 512, 7, 7]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_7 = torch.ops.aten.convolution_backward.default(view_217, view_127, primals_127, [512], [1, 1], [1, 1], [1, 1], False, [0, 0], 512, [True, True, True]);  view_127 = primals_127 = None
    getitem_111: "f32[8, 512, 7, 7]" = convolution_backward_7[0]
    getitem_112: "f32[512, 1, 3, 3]" = convolution_backward_7[1]
    getitem_113: "f32[512]" = convolution_backward_7[2];  convolution_backward_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_102: "f32[8, 512, 7, 7]" = torch.ops.aten.add.Tensor(view_217, getitem_111);  view_217 = getitem_111 = None
    add_103: "f32[512, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_100, getitem_112);  getitem_100 = getitem_112 = None
    add_104: "f32[512]" = torch.ops.aten.add.Tensor(getitem_101, getitem_113);  getitem_101 = getitem_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_218: "f32[8, 512, 49]" = torch.ops.aten.view.default(add_102, [8, 512, 49]);  add_102 = None
    permute_154: "f32[8, 49, 512]" = torch.ops.aten.permute.default(view_218, [0, 2, 1]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_scatter_21: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_default, permute_154, 1, 1, 9223372036854775807);  permute_154 = None
    slice_scatter_22: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_21, 0, 0, 9223372036854775807);  slice_scatter_21 = None
    slice_scatter_23: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_default, slice_118, 1, 0, 1);  slice_118 = None
    slice_scatter_24: "f32[8, 50, 512]" = torch.ops.aten.slice_scatter.default(full_default, slice_scatter_23, 0, 0, 9223372036854775807);  full_default = slice_scatter_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_105: "f32[8, 50, 512]" = torch.ops.aten.add.Tensor(slice_scatter_22, slice_scatter_24);  slice_scatter_22 = slice_scatter_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_120: "f32[8, 1, 512]" = torch.ops.aten.slice.Tensor(add_105, 1, 0, 1)
    slice_121: "f32[8, 49, 512]" = torch.ops.aten.slice.Tensor(add_105, 1, 1, 50);  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    sum_40: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(slice_120, [0], True);  slice_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_71: "f32[8, 49, 512]" = torch.ops.aten.clone.default(slice_121, memory_format = torch.contiguous_format);  slice_121 = None
    mul_142: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(clone_71, primals_125);  primals_125 = None
    mul_143: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_142, 512)
    sum_41: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_142, [2], True)
    mul_144: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_142, mul_60);  mul_142 = None
    sum_42: "f32[8, 49, 1]" = torch.ops.aten.sum.dim_IntList(mul_144, [2], True);  mul_144 = None
    mul_145: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(mul_60, sum_42);  sum_42 = None
    sub_47: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(mul_143, sum_41);  mul_143 = sum_41 = None
    sub_48: "f32[8, 49, 512]" = torch.ops.aten.sub.Tensor(sub_47, mul_145);  sub_47 = mul_145 = None
    mul_146: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(div_13, sub_48);  div_13 = sub_48 = None
    mul_147: "f32[8, 49, 512]" = torch.ops.aten.mul.Tensor(clone_71, mul_60);  mul_60 = None
    sum_43: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_147, [0, 1]);  mul_147 = None
    sum_44: "f32[512]" = torch.ops.aten.sum.dim_IntList(clone_71, [0, 1]);  clone_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_155: "f32[8, 512, 49]" = torch.ops.aten.permute.default(mul_146, [0, 2, 1]);  mul_146 = None
    view_219: "f32[8, 512, 7, 7]" = torch.ops.aten.view.default(permute_155, [8, 512, 7, 7]);  permute_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_8 = torch.ops.aten.convolution_backward.default(view_219, clone_47, primals_123, [512], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_219 = clone_47 = primals_123 = None
    getitem_114: "f32[8, 320, 14, 14]" = convolution_backward_8[0]
    getitem_115: "f32[512, 320, 2, 2]" = convolution_backward_8[1]
    getitem_116: "f32[512]" = convolution_backward_8[2];  convolution_backward_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:595, code: x3_nocls = remove_cls(x3).reshape(B, H3, W3, -1).permute(0, 3, 1, 2).contiguous()
    permute_156: "f32[8, 14, 14, 320]" = torch.ops.aten.permute.default(getitem_114, [0, 2, 3, 1]);  getitem_114 = None
    view_220: "f32[8, 196, 320]" = torch.ops.aten.view.default(permute_156, [8, 196, 320]);  permute_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    full_default_26: "f32[8, 196, 320]" = torch.ops.aten.full.default([8, 196, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_25: "f32[8, 196, 320]" = torch.ops.aten.slice_scatter.default(full_default_26, view_220, 2, 0, 9223372036854775807);  full_default_26 = view_220 = None
    full_default_27: "f32[8, 197, 320]" = torch.ops.aten.full.default([8, 197, 320], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_26: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_default_27, slice_scatter_25, 1, 1, 9223372036854775807);  slice_scatter_25 = None
    slice_scatter_27: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_default_27, slice_scatter_26, 0, 0, 9223372036854775807);  slice_scatter_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_221: "f32[1576, 320]" = torch.ops.aten.view.default(slice_scatter_27, [1576, 320])
    mm_18: "f32[1576, 1280]" = torch.ops.aten.mm.default(view_221, permute_157);  permute_157 = None
    permute_158: "f32[320, 1576]" = torch.ops.aten.permute.default(view_221, [1, 0])
    mm_19: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_158, view_123);  permute_158 = view_123 = None
    permute_159: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_45: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_221, [0], True);  view_221 = None
    view_222: "f32[320]" = torch.ops.aten.view.default(sum_45, [320]);  sum_45 = None
    permute_160: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    view_223: "f32[8, 197, 1280]" = torch.ops.aten.view.default(mm_18, [8, 197, 1280]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_149: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(add_58, 0.5);  add_58 = None
    mul_150: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_122, view_122)
    mul_151: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(mul_150, -0.5);  mul_150 = None
    exp_10: "f32[8, 197, 1280]" = torch.ops.aten.exp.default(mul_151);  mul_151 = None
    mul_152: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(exp_10, 0.3989422804014327);  exp_10 = None
    mul_153: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_122, mul_152);  view_122 = mul_152 = None
    add_107: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(mul_149, mul_153);  mul_149 = mul_153 = None
    mul_154: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_223, add_107);  view_223 = add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_224: "f32[1576, 1280]" = torch.ops.aten.view.default(mul_154, [1576, 1280]);  mul_154 = None
    mm_20: "f32[1576, 320]" = torch.ops.aten.mm.default(view_224, permute_161);  permute_161 = None
    permute_162: "f32[1280, 1576]" = torch.ops.aten.permute.default(view_224, [1, 0])
    mm_21: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_162, view_121);  permute_162 = view_121 = None
    permute_163: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_46: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_224, [0], True);  view_224 = None
    view_225: "f32[1280]" = torch.ops.aten.view.default(sum_46, [1280]);  sum_46 = None
    permute_164: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    view_226: "f32[8, 197, 320]" = torch.ops.aten.view.default(mm_20, [8, 197, 320]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_156: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_226, primals_26);  primals_26 = None
    mul_157: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_156, 320)
    sum_47: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_156, [2], True)
    mul_158: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_156, mul_55);  mul_156 = None
    sum_48: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_158, [2], True);  mul_158 = None
    mul_159: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_55, sum_48);  sum_48 = None
    sub_50: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(mul_157, sum_47);  mul_157 = sum_47 = None
    sub_51: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(sub_50, mul_159);  sub_50 = mul_159 = None
    mul_160: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(div_14, sub_51);  div_14 = sub_51 = None
    mul_161: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_226, mul_55);  mul_55 = None
    sum_49: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_161, [0, 1]);  mul_161 = None
    sum_50: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_226, [0, 1]);  view_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_108: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(slice_scatter_27, mul_160);  slice_scatter_27 = mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_227: "f32[1576, 320]" = torch.ops.aten.view.default(add_108, [1576, 320])
    mm_22: "f32[1576, 320]" = torch.ops.aten.mm.default(view_227, permute_165);  permute_165 = None
    permute_166: "f32[320, 1576]" = torch.ops.aten.permute.default(view_227, [1, 0])
    mm_23: "f32[320, 320]" = torch.ops.aten.mm.default(permute_166, view_119);  permute_166 = view_119 = None
    permute_167: "f32[320, 320]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_51: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_227, [0], True);  view_227 = None
    view_228: "f32[320]" = torch.ops.aten.view.default(sum_51, [320]);  sum_51 = None
    permute_168: "f32[320, 320]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    view_229: "f32[8, 197, 320]" = torch.ops.aten.view.default(mm_22, [8, 197, 320]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_230: "f32[8, 197, 8, 40]" = torch.ops.aten.view.default(view_229, [8, 197, 8, 40]);  view_229 = None
    permute_169: "f32[8, 8, 197, 40]" = torch.ops.aten.permute.default(view_230, [0, 2, 1, 3]);  view_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_162: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(permute_169, 0.15811388300841897)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_10: "f32[8, 8, 196, 40]" = torch.ops.aten.constant_pad_nd.default(permute_169, [0, 0, -1, 0, 0, 0]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_163: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(constant_pad_nd_10, slice_74);  slice_74 = None
    mul_164: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(constant_pad_nd_10, permute_66);  constant_pad_nd_10 = permute_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_170: "f32[8, 8, 40, 196]" = torch.ops.aten.permute.default(mul_163, [0, 1, 3, 2]);  mul_163 = None
    view_231: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_170, [8, 320, 14, 14]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_122: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(view_231, 1, 0, 80)
    slice_123: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(view_231, 1, 80, 200)
    slice_124: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(view_231, 1, 200, 320);  view_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_9 = torch.ops.aten.convolution_backward.default(slice_124, getitem_63, primals_107, [120], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, True]);  slice_124 = getitem_63 = None
    getitem_117: "f32[8, 120, 14, 14]" = convolution_backward_9[0]
    getitem_118: "f32[120, 1, 7, 7]" = convolution_backward_9[1]
    getitem_119: "f32[120]" = convolution_backward_9[2];  convolution_backward_9 = None
    convolution_backward_10 = torch.ops.aten.convolution_backward.default(slice_123, getitem_62, primals_105, [120], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, True]);  slice_123 = getitem_62 = None
    getitem_120: "f32[8, 120, 14, 14]" = convolution_backward_10[0]
    getitem_121: "f32[120, 1, 5, 5]" = convolution_backward_10[1]
    getitem_122: "f32[120]" = convolution_backward_10[2];  convolution_backward_10 = None
    convolution_backward_11 = torch.ops.aten.convolution_backward.default(slice_122, getitem_61, primals_103, [80], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, True]);  slice_122 = getitem_61 = None
    getitem_123: "f32[8, 80, 14, 14]" = convolution_backward_11[0]
    getitem_124: "f32[80, 1, 3, 3]" = convolution_backward_11[1]
    getitem_125: "f32[80]" = convolution_backward_11[2];  convolution_backward_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_24: "f32[8, 320, 14, 14]" = torch.ops.aten.cat.default([getitem_123, getitem_120, getitem_117], 1);  getitem_123 = getitem_120 = getitem_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_232: "f32[8, 8, 40, 196]" = torch.ops.aten.view.default(cat_24, [8, 8, 40, 196]);  cat_24 = None
    permute_171: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_232, [0, 1, 3, 2]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_default_29: "f32[8, 8, 196, 40]" = torch.ops.aten.full.default([8, 8, 196, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_28: "f32[8, 8, 196, 40]" = torch.ops.aten.slice_scatter.default(full_default_29, permute_171, 3, 0, 9223372036854775807);  permute_171 = None
    full_default_30: "f32[8, 8, 197, 40]" = torch.ops.aten.full.default([8, 8, 197, 40], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_29: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter_28, 2, 1, 9223372036854775807);  slice_scatter_28 = None
    slice_scatter_30: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter_29, 1, 0, 9223372036854775807);  slice_scatter_29 = None
    slice_scatter_31: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter_30, 0, 0, 9223372036854775807);  slice_scatter_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_scatter_32: "f32[8, 8, 196, 40]" = torch.ops.aten.slice_scatter.default(full_default_29, mul_164, 3, 0, 9223372036854775807);  mul_164 = None
    slice_scatter_33: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter_32, 2, 1, 9223372036854775807);  slice_scatter_32 = None
    slice_scatter_34: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter_33, 1, 0, 9223372036854775807);  slice_scatter_33 = None
    slice_scatter_35: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter_34, 0, 0, 9223372036854775807);  slice_scatter_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_73: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(mul_162, memory_format = torch.contiguous_format);  mul_162 = None
    view_233: "f32[64, 197, 40]" = torch.ops.aten.view.default(clone_73, [64, 197, 40]);  clone_73 = None
    bmm_24: "f32[64, 40, 40]" = torch.ops.aten.bmm.default(permute_172, view_233);  permute_172 = None
    bmm_25: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(view_233, permute_173);  view_233 = permute_173 = None
    view_234: "f32[8, 8, 40, 40]" = torch.ops.aten.view.default(bmm_24, [8, 8, 40, 40]);  bmm_24 = None
    view_235: "f32[8, 8, 197, 40]" = torch.ops.aten.view.default(bmm_25, [8, 8, 197, 40]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_109: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(slice_scatter_35, view_235);  slice_scatter_35 = view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_236: "f32[64, 40, 40]" = torch.ops.aten.view.default(view_234, [64, 40, 40]);  view_234 = None
    bmm_26: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(permute_174, view_236);  permute_174 = None
    bmm_27: "f32[64, 40, 197]" = torch.ops.aten.bmm.default(view_236, permute_175);  view_236 = permute_175 = None
    view_237: "f32[8, 8, 197, 40]" = torch.ops.aten.view.default(bmm_26, [8, 8, 197, 40]);  bmm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_110: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(slice_scatter_31, view_237);  slice_scatter_31 = view_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_238: "f32[8, 8, 40, 197]" = torch.ops.aten.view.default(bmm_27, [8, 8, 40, 197]);  bmm_27 = None
    permute_176: "f32[8, 8, 197, 40]" = torch.ops.aten.permute.default(view_238, [0, 1, 3, 2]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    mul_165: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(permute_176, alias_10);  permute_176 = None
    sum_52: "f32[8, 8, 1, 40]" = torch.ops.aten.sum.dim_IntList(mul_165, [2], True)
    mul_166: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(alias_10, sum_52);  alias_10 = sum_52 = None
    sub_52: "f32[8, 8, 197, 40]" = torch.ops.aten.sub.Tensor(mul_165, mul_166);  mul_165 = mul_166 = None
    clone_74: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(sub_52, memory_format = torch.contiguous_format);  sub_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_25: "f32[24, 8, 197, 40]" = torch.ops.aten.cat.default([add_109, clone_74, add_110]);  add_109 = clone_74 = add_110 = None
    view_239: "f32[3, 8, 8, 197, 40]" = torch.ops.aten.view.default(cat_25, [3, 8, 8, 197, 40]);  cat_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_177: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.permute.default(view_239, [1, 3, 0, 2, 4]);  view_239 = None
    clone_75: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    view_240: "f32[8, 197, 960]" = torch.ops.aten.view.default(clone_75, [8, 197, 960]);  clone_75 = None
    view_241: "f32[1576, 960]" = torch.ops.aten.view.default(view_240, [1576, 960]);  view_240 = None
    mm_24: "f32[1576, 320]" = torch.ops.aten.mm.default(view_241, permute_178);  permute_178 = None
    permute_179: "f32[960, 1576]" = torch.ops.aten.permute.default(view_241, [1, 0])
    mm_25: "f32[960, 320]" = torch.ops.aten.mm.default(permute_179, view_107);  permute_179 = view_107 = None
    permute_180: "f32[320, 960]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_53: "f32[1, 960]" = torch.ops.aten.sum.dim_IntList(view_241, [0], True);  view_241 = None
    view_242: "f32[960]" = torch.ops.aten.view.default(sum_53, [960]);  sum_53 = None
    permute_181: "f32[960, 320]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    view_243: "f32[8, 197, 320]" = torch.ops.aten.view.default(mm_24, [8, 197, 320]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_168: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_243, primals_24);  primals_24 = None
    mul_169: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_168, 320)
    sum_54: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_168, [2], True)
    mul_170: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_168, mul_51);  mul_168 = None
    sum_55: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_170, [2], True);  mul_170 = None
    mul_171: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_51, sum_55);  sum_55 = None
    sub_54: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(mul_169, sum_54);  mul_169 = sum_54 = None
    sub_55: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(sub_54, mul_171);  sub_54 = mul_171 = None
    div_15: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_13, 320);  rsqrt_13 = None
    mul_172: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(div_15, sub_55);  div_15 = sub_55 = None
    mul_173: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_243, mul_51);  mul_51 = None
    sum_56: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_173, [0, 1]);  mul_173 = None
    sum_57: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_243, [0, 1]);  view_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_111: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_108, mul_172);  add_108 = mul_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_125: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(add_111, 1, 0, 1)
    slice_126: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(add_111, 1, 1, 197);  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_182: "f32[8, 320, 196]" = torch.ops.aten.permute.default(slice_126, [0, 2, 1]);  slice_126 = None
    view_244: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_182, [8, 320, 14, 14]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_12 = torch.ops.aten.convolution_backward.default(view_244, view_105, primals_99, [320], [1, 1], [1, 1], [1, 1], False, [0, 0], 320, [True, True, True]);  view_105 = None
    getitem_126: "f32[8, 320, 14, 14]" = convolution_backward_12[0]
    getitem_127: "f32[320, 1, 3, 3]" = convolution_backward_12[1]
    getitem_128: "f32[320]" = convolution_backward_12[2];  convolution_backward_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_112: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(view_244, getitem_126);  view_244 = getitem_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_245: "f32[8, 320, 196]" = torch.ops.aten.view.default(add_112, [8, 320, 196]);  add_112 = None
    permute_183: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_scatter_36: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_default_27, permute_183, 1, 1, 9223372036854775807);  permute_183 = None
    slice_scatter_37: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_default_27, slice_scatter_36, 0, 0, 9223372036854775807);  slice_scatter_36 = None
    slice_scatter_38: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_default_27, slice_125, 1, 0, 1);  slice_125 = None
    slice_scatter_39: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_default_27, slice_scatter_38, 0, 0, 9223372036854775807);  slice_scatter_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_113: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(slice_scatter_37, slice_scatter_39);  slice_scatter_37 = slice_scatter_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_246: "f32[1576, 320]" = torch.ops.aten.view.default(add_113, [1576, 320])
    mm_26: "f32[1576, 1280]" = torch.ops.aten.mm.default(view_246, permute_184);  permute_184 = None
    permute_185: "f32[320, 1576]" = torch.ops.aten.permute.default(view_246, [1, 0])
    mm_27: "f32[320, 1280]" = torch.ops.aten.mm.default(permute_185, view_103);  permute_185 = view_103 = None
    permute_186: "f32[1280, 320]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_58: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_246, [0], True);  view_246 = None
    view_247: "f32[320]" = torch.ops.aten.view.default(sum_58, [320]);  sum_58 = None
    permute_187: "f32[320, 1280]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    view_248: "f32[8, 197, 1280]" = torch.ops.aten.view.default(mm_26, [8, 197, 1280]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_175: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(add_49, 0.5);  add_49 = None
    mul_176: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_102, view_102)
    mul_177: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(mul_176, -0.5);  mul_176 = None
    exp_11: "f32[8, 197, 1280]" = torch.ops.aten.exp.default(mul_177);  mul_177 = None
    mul_178: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(exp_11, 0.3989422804014327);  exp_11 = None
    mul_179: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_102, mul_178);  view_102 = mul_178 = None
    add_115: "f32[8, 197, 1280]" = torch.ops.aten.add.Tensor(mul_175, mul_179);  mul_175 = mul_179 = None
    mul_180: "f32[8, 197, 1280]" = torch.ops.aten.mul.Tensor(view_248, add_115);  view_248 = add_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_249: "f32[1576, 1280]" = torch.ops.aten.view.default(mul_180, [1576, 1280]);  mul_180 = None
    mm_28: "f32[1576, 320]" = torch.ops.aten.mm.default(view_249, permute_188);  permute_188 = None
    permute_189: "f32[1280, 1576]" = torch.ops.aten.permute.default(view_249, [1, 0])
    mm_29: "f32[1280, 320]" = torch.ops.aten.mm.default(permute_189, view_101);  permute_189 = view_101 = None
    permute_190: "f32[320, 1280]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_59: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_249, [0], True);  view_249 = None
    view_250: "f32[1280]" = torch.ops.aten.view.default(sum_59, [1280]);  sum_59 = None
    permute_191: "f32[1280, 320]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    view_251: "f32[8, 197, 320]" = torch.ops.aten.view.default(mm_28, [8, 197, 320]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_182: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_251, primals_22);  primals_22 = None
    mul_183: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_182, 320)
    sum_60: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_182, [2], True)
    mul_184: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_182, mul_46);  mul_182 = None
    sum_61: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_184, [2], True);  mul_184 = None
    mul_185: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_46, sum_61);  sum_61 = None
    sub_57: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(mul_183, sum_60);  mul_183 = sum_60 = None
    sub_58: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(sub_57, mul_185);  sub_57 = mul_185 = None
    mul_186: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(div_16, sub_58);  div_16 = sub_58 = None
    mul_187: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_251, mul_46);  mul_46 = None
    sum_62: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_187, [0, 1]);  mul_187 = None
    sum_63: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_251, [0, 1]);  view_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_116: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_113, mul_186);  add_113 = mul_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_252: "f32[1576, 320]" = torch.ops.aten.view.default(add_116, [1576, 320])
    mm_30: "f32[1576, 320]" = torch.ops.aten.mm.default(view_252, permute_192);  permute_192 = None
    permute_193: "f32[320, 1576]" = torch.ops.aten.permute.default(view_252, [1, 0])
    mm_31: "f32[320, 320]" = torch.ops.aten.mm.default(permute_193, view_99);  permute_193 = view_99 = None
    permute_194: "f32[320, 320]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_64: "f32[1, 320]" = torch.ops.aten.sum.dim_IntList(view_252, [0], True);  view_252 = None
    view_253: "f32[320]" = torch.ops.aten.view.default(sum_64, [320]);  sum_64 = None
    permute_195: "f32[320, 320]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
    view_254: "f32[8, 197, 320]" = torch.ops.aten.view.default(mm_30, [8, 197, 320]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_255: "f32[8, 197, 8, 40]" = torch.ops.aten.view.default(view_254, [8, 197, 8, 40]);  view_254 = None
    permute_196: "f32[8, 8, 197, 40]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_188: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(permute_196, 0.15811388300841897)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_11: "f32[8, 8, 196, 40]" = torch.ops.aten.constant_pad_nd.default(permute_196, [0, 0, -1, 0, 0, 0]);  permute_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_189: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(constant_pad_nd_11, slice_62);  slice_62 = None
    mul_190: "f32[8, 8, 196, 40]" = torch.ops.aten.mul.Tensor(constant_pad_nd_11, permute_55);  constant_pad_nd_11 = permute_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_197: "f32[8, 8, 40, 196]" = torch.ops.aten.permute.default(mul_189, [0, 1, 3, 2]);  mul_189 = None
    view_256: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_197, [8, 320, 14, 14]);  permute_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_127: "f32[8, 80, 14, 14]" = torch.ops.aten.slice.Tensor(view_256, 1, 0, 80)
    slice_128: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(view_256, 1, 80, 200)
    slice_129: "f32[8, 120, 14, 14]" = torch.ops.aten.slice.Tensor(view_256, 1, 200, 320);  view_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_13 = torch.ops.aten.convolution_backward.default(slice_129, getitem_53, primals_107, [120], [1, 1], [3, 3], [1, 1], False, [0, 0], 120, [True, True, True]);  slice_129 = getitem_53 = primals_107 = None
    getitem_129: "f32[8, 120, 14, 14]" = convolution_backward_13[0]
    getitem_130: "f32[120, 1, 7, 7]" = convolution_backward_13[1]
    getitem_131: "f32[120]" = convolution_backward_13[2];  convolution_backward_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_117: "f32[120, 1, 7, 7]" = torch.ops.aten.add.Tensor(getitem_118, getitem_130);  getitem_118 = getitem_130 = None
    add_118: "f32[120]" = torch.ops.aten.add.Tensor(getitem_119, getitem_131);  getitem_119 = getitem_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_14 = torch.ops.aten.convolution_backward.default(slice_128, getitem_52, primals_105, [120], [1, 1], [2, 2], [1, 1], False, [0, 0], 120, [True, True, True]);  slice_128 = getitem_52 = primals_105 = None
    getitem_132: "f32[8, 120, 14, 14]" = convolution_backward_14[0]
    getitem_133: "f32[120, 1, 5, 5]" = convolution_backward_14[1]
    getitem_134: "f32[120]" = convolution_backward_14[2];  convolution_backward_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_119: "f32[120, 1, 5, 5]" = torch.ops.aten.add.Tensor(getitem_121, getitem_133);  getitem_121 = getitem_133 = None
    add_120: "f32[120]" = torch.ops.aten.add.Tensor(getitem_122, getitem_134);  getitem_122 = getitem_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_15 = torch.ops.aten.convolution_backward.default(slice_127, getitem_51, primals_103, [80], [1, 1], [1, 1], [1, 1], False, [0, 0], 80, [True, True, True]);  slice_127 = getitem_51 = primals_103 = None
    getitem_135: "f32[8, 80, 14, 14]" = convolution_backward_15[0]
    getitem_136: "f32[80, 1, 3, 3]" = convolution_backward_15[1]
    getitem_137: "f32[80]" = convolution_backward_15[2];  convolution_backward_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_121: "f32[80, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_124, getitem_136);  getitem_124 = getitem_136 = None
    add_122: "f32[80]" = torch.ops.aten.add.Tensor(getitem_125, getitem_137);  getitem_125 = getitem_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_26: "f32[8, 320, 14, 14]" = torch.ops.aten.cat.default([getitem_135, getitem_132, getitem_129], 1);  getitem_135 = getitem_132 = getitem_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_257: "f32[8, 8, 40, 196]" = torch.ops.aten.view.default(cat_26, [8, 8, 40, 196]);  cat_26 = None
    permute_198: "f32[8, 8, 196, 40]" = torch.ops.aten.permute.default(view_257, [0, 1, 3, 2]);  view_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_scatter_40: "f32[8, 8, 196, 40]" = torch.ops.aten.slice_scatter.default(full_default_29, permute_198, 3, 0, 9223372036854775807);  permute_198 = None
    slice_scatter_41: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter_40, 2, 1, 9223372036854775807);  slice_scatter_40 = None
    slice_scatter_42: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter_41, 1, 0, 9223372036854775807);  slice_scatter_41 = None
    slice_scatter_43: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter_42, 0, 0, 9223372036854775807);  slice_scatter_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_scatter_44: "f32[8, 8, 196, 40]" = torch.ops.aten.slice_scatter.default(full_default_29, mul_190, 3, 0, 9223372036854775807);  full_default_29 = mul_190 = None
    slice_scatter_45: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter_44, 2, 1, 9223372036854775807);  slice_scatter_44 = None
    slice_scatter_46: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter_45, 1, 0, 9223372036854775807);  slice_scatter_45 = None
    slice_scatter_47: "f32[8, 8, 197, 40]" = torch.ops.aten.slice_scatter.default(full_default_30, slice_scatter_46, 0, 0, 9223372036854775807);  full_default_30 = slice_scatter_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_76: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(mul_188, memory_format = torch.contiguous_format);  mul_188 = None
    view_258: "f32[64, 197, 40]" = torch.ops.aten.view.default(clone_76, [64, 197, 40]);  clone_76 = None
    bmm_28: "f32[64, 40, 40]" = torch.ops.aten.bmm.default(permute_199, view_258);  permute_199 = None
    bmm_29: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(view_258, permute_200);  view_258 = permute_200 = None
    view_259: "f32[8, 8, 40, 40]" = torch.ops.aten.view.default(bmm_28, [8, 8, 40, 40]);  bmm_28 = None
    view_260: "f32[8, 8, 197, 40]" = torch.ops.aten.view.default(bmm_29, [8, 8, 197, 40]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_123: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(slice_scatter_47, view_260);  slice_scatter_47 = view_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_261: "f32[64, 40, 40]" = torch.ops.aten.view.default(view_259, [64, 40, 40]);  view_259 = None
    bmm_30: "f32[64, 197, 40]" = torch.ops.aten.bmm.default(permute_201, view_261);  permute_201 = None
    bmm_31: "f32[64, 40, 197]" = torch.ops.aten.bmm.default(view_261, permute_202);  view_261 = permute_202 = None
    view_262: "f32[8, 8, 197, 40]" = torch.ops.aten.view.default(bmm_30, [8, 8, 197, 40]);  bmm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_124: "f32[8, 8, 197, 40]" = torch.ops.aten.add.Tensor(slice_scatter_43, view_262);  slice_scatter_43 = view_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_263: "f32[8, 8, 40, 197]" = torch.ops.aten.view.default(bmm_31, [8, 8, 40, 197]);  bmm_31 = None
    permute_203: "f32[8, 8, 197, 40]" = torch.ops.aten.permute.default(view_263, [0, 1, 3, 2]);  view_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    mul_191: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(permute_203, alias_11);  permute_203 = None
    sum_65: "f32[8, 8, 1, 40]" = torch.ops.aten.sum.dim_IntList(mul_191, [2], True)
    mul_192: "f32[8, 8, 197, 40]" = torch.ops.aten.mul.Tensor(alias_11, sum_65);  alias_11 = sum_65 = None
    sub_59: "f32[8, 8, 197, 40]" = torch.ops.aten.sub.Tensor(mul_191, mul_192);  mul_191 = mul_192 = None
    clone_77: "f32[8, 8, 197, 40]" = torch.ops.aten.clone.default(sub_59, memory_format = torch.contiguous_format);  sub_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_27: "f32[24, 8, 197, 40]" = torch.ops.aten.cat.default([add_123, clone_77, add_124]);  add_123 = clone_77 = add_124 = None
    view_264: "f32[3, 8, 8, 197, 40]" = torch.ops.aten.view.default(cat_27, [3, 8, 8, 197, 40]);  cat_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_204: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.permute.default(view_264, [1, 3, 0, 2, 4]);  view_264 = None
    clone_78: "f32[8, 197, 3, 8, 40]" = torch.ops.aten.clone.default(permute_204, memory_format = torch.contiguous_format);  permute_204 = None
    view_265: "f32[8, 197, 960]" = torch.ops.aten.view.default(clone_78, [8, 197, 960]);  clone_78 = None
    view_266: "f32[1576, 960]" = torch.ops.aten.view.default(view_265, [1576, 960]);  view_265 = None
    mm_32: "f32[1576, 320]" = torch.ops.aten.mm.default(view_266, permute_205);  permute_205 = None
    permute_206: "f32[960, 1576]" = torch.ops.aten.permute.default(view_266, [1, 0])
    mm_33: "f32[960, 320]" = torch.ops.aten.mm.default(permute_206, view_87);  permute_206 = view_87 = None
    permute_207: "f32[320, 960]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_66: "f32[1, 960]" = torch.ops.aten.sum.dim_IntList(view_266, [0], True);  view_266 = None
    view_267: "f32[960]" = torch.ops.aten.view.default(sum_66, [960]);  sum_66 = None
    permute_208: "f32[960, 320]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    view_268: "f32[8, 197, 320]" = torch.ops.aten.view.default(mm_32, [8, 197, 320]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_194: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_268, primals_20);  primals_20 = None
    mul_195: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_194, 320)
    sum_67: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_194, [2], True)
    mul_196: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_194, mul_42);  mul_194 = None
    sum_68: "f32[8, 197, 1]" = torch.ops.aten.sum.dim_IntList(mul_196, [2], True);  mul_196 = None
    mul_197: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(mul_42, sum_68);  sum_68 = None
    sub_61: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(mul_195, sum_67);  mul_195 = sum_67 = None
    sub_62: "f32[8, 197, 320]" = torch.ops.aten.sub.Tensor(sub_61, mul_197);  sub_61 = mul_197 = None
    div_17: "f32[8, 197, 1]" = torch.ops.aten.div.Tensor(rsqrt_11, 320);  rsqrt_11 = None
    mul_198: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(div_17, sub_62);  div_17 = sub_62 = None
    mul_199: "f32[8, 197, 320]" = torch.ops.aten.mul.Tensor(view_268, mul_42);  mul_42 = None
    sum_69: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_199, [0, 1]);  mul_199 = None
    sum_70: "f32[320]" = torch.ops.aten.sum.dim_IntList(view_268, [0, 1]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_125: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(add_116, mul_198);  add_116 = mul_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_130: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(add_125, 1, 0, 1)
    slice_131: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(add_125, 1, 1, 197);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_209: "f32[8, 320, 196]" = torch.ops.aten.permute.default(slice_131, [0, 2, 1]);  slice_131 = None
    view_269: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_209, [8, 320, 14, 14]);  permute_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_16 = torch.ops.aten.convolution_backward.default(view_269, view_85, primals_99, [320], [1, 1], [1, 1], [1, 1], False, [0, 0], 320, [True, True, True]);  view_85 = primals_99 = None
    getitem_138: "f32[8, 320, 14, 14]" = convolution_backward_16[0]
    getitem_139: "f32[320, 1, 3, 3]" = convolution_backward_16[1]
    getitem_140: "f32[320]" = convolution_backward_16[2];  convolution_backward_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_126: "f32[8, 320, 14, 14]" = torch.ops.aten.add.Tensor(view_269, getitem_138);  view_269 = getitem_138 = None
    add_127: "f32[320, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_127, getitem_139);  getitem_127 = getitem_139 = None
    add_128: "f32[320]" = torch.ops.aten.add.Tensor(getitem_128, getitem_140);  getitem_128 = getitem_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_270: "f32[8, 320, 196]" = torch.ops.aten.view.default(add_126, [8, 320, 196]);  add_126 = None
    permute_210: "f32[8, 196, 320]" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_scatter_48: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_default_27, permute_210, 1, 1, 9223372036854775807);  permute_210 = None
    slice_scatter_49: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_default_27, slice_scatter_48, 0, 0, 9223372036854775807);  slice_scatter_48 = None
    slice_scatter_50: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_default_27, slice_130, 1, 0, 1);  slice_130 = None
    slice_scatter_51: "f32[8, 197, 320]" = torch.ops.aten.slice_scatter.default(full_default_27, slice_scatter_50, 0, 0, 9223372036854775807);  full_default_27 = slice_scatter_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_129: "f32[8, 197, 320]" = torch.ops.aten.add.Tensor(slice_scatter_49, slice_scatter_51);  slice_scatter_49 = slice_scatter_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_132: "f32[8, 1, 320]" = torch.ops.aten.slice.Tensor(add_129, 1, 0, 1)
    slice_133: "f32[8, 196, 320]" = torch.ops.aten.slice.Tensor(add_129, 1, 1, 197);  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    sum_71: "f32[1, 1, 320]" = torch.ops.aten.sum.dim_IntList(slice_132, [0], True);  slice_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_79: "f32[8, 196, 320]" = torch.ops.aten.clone.default(slice_133, memory_format = torch.contiguous_format);  slice_133 = None
    mul_201: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_79, primals_97);  primals_97 = None
    mul_202: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_201, 320)
    sum_72: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [2], True)
    mul_203: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_201, mul_40);  mul_201 = None
    sum_73: "f32[8, 196, 1]" = torch.ops.aten.sum.dim_IntList(mul_203, [2], True);  mul_203 = None
    mul_204: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(mul_40, sum_73);  sum_73 = None
    sub_64: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(mul_202, sum_72);  mul_202 = sum_72 = None
    sub_65: "f32[8, 196, 320]" = torch.ops.aten.sub.Tensor(sub_64, mul_204);  sub_64 = mul_204 = None
    mul_205: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(div_18, sub_65);  div_18 = sub_65 = None
    mul_206: "f32[8, 196, 320]" = torch.ops.aten.mul.Tensor(clone_79, mul_40);  mul_40 = None
    sum_74: "f32[320]" = torch.ops.aten.sum.dim_IntList(mul_206, [0, 1]);  mul_206 = None
    sum_75: "f32[320]" = torch.ops.aten.sum.dim_IntList(clone_79, [0, 1]);  clone_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_211: "f32[8, 320, 196]" = torch.ops.aten.permute.default(mul_205, [0, 2, 1]);  mul_205 = None
    view_271: "f32[8, 320, 14, 14]" = torch.ops.aten.view.default(permute_211, [8, 320, 14, 14]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_17 = torch.ops.aten.convolution_backward.default(view_271, clone_31, primals_95, [320], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_271 = clone_31 = primals_95 = None
    getitem_141: "f32[8, 128, 28, 28]" = convolution_backward_17[0]
    getitem_142: "f32[320, 128, 2, 2]" = convolution_backward_17[1]
    getitem_143: "f32[320]" = convolution_backward_17[2];  convolution_backward_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:587, code: x2_nocls = remove_cls(x2).reshape(B, H2, W2, -1).permute(0, 3, 1, 2).contiguous()
    permute_212: "f32[8, 28, 28, 128]" = torch.ops.aten.permute.default(getitem_141, [0, 2, 3, 1]);  getitem_141 = None
    view_272: "f32[8, 784, 128]" = torch.ops.aten.view.default(permute_212, [8, 784, 128]);  permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    full_default_53: "f32[8, 784, 128]" = torch.ops.aten.full.default([8, 784, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_52: "f32[8, 784, 128]" = torch.ops.aten.slice_scatter.default(full_default_53, view_272, 2, 0, 9223372036854775807);  full_default_53 = view_272 = None
    full_default_54: "f32[8, 785, 128]" = torch.ops.aten.full.default([8, 785, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_53: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_default_54, slice_scatter_52, 1, 1, 9223372036854775807);  slice_scatter_52 = None
    slice_scatter_54: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_default_54, slice_scatter_53, 0, 0, 9223372036854775807);  slice_scatter_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_273: "f32[6280, 128]" = torch.ops.aten.view.default(slice_scatter_54, [6280, 128])
    mm_34: "f32[6280, 1024]" = torch.ops.aten.mm.default(view_273, permute_213);  permute_213 = None
    permute_214: "f32[128, 6280]" = torch.ops.aten.permute.default(view_273, [1, 0])
    mm_35: "f32[128, 1024]" = torch.ops.aten.mm.default(permute_214, view_81);  permute_214 = view_81 = None
    permute_215: "f32[1024, 128]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_76: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_273, [0], True);  view_273 = None
    view_274: "f32[128]" = torch.ops.aten.view.default(sum_76, [128]);  sum_76 = None
    permute_216: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
    view_275: "f32[8, 785, 1024]" = torch.ops.aten.view.default(mm_34, [8, 785, 1024]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_208: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(add_38, 0.5);  add_38 = None
    mul_209: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_80, view_80)
    mul_210: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(mul_209, -0.5);  mul_209 = None
    exp_12: "f32[8, 785, 1024]" = torch.ops.aten.exp.default(mul_210);  mul_210 = None
    mul_211: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_212: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_80, mul_211);  view_80 = mul_211 = None
    add_131: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(mul_208, mul_212);  mul_208 = mul_212 = None
    mul_213: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_275, add_131);  view_275 = add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_276: "f32[6280, 1024]" = torch.ops.aten.view.default(mul_213, [6280, 1024]);  mul_213 = None
    mm_36: "f32[6280, 128]" = torch.ops.aten.mm.default(view_276, permute_217);  permute_217 = None
    permute_218: "f32[1024, 6280]" = torch.ops.aten.permute.default(view_276, [1, 0])
    mm_37: "f32[1024, 128]" = torch.ops.aten.mm.default(permute_218, view_79);  permute_218 = view_79 = None
    permute_219: "f32[128, 1024]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_77: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_276, [0], True);  view_276 = None
    view_277: "f32[1024]" = torch.ops.aten.view.default(sum_77, [1024]);  sum_77 = None
    permute_220: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    view_278: "f32[8, 785, 128]" = torch.ops.aten.view.default(mm_36, [8, 785, 128]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_215: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_278, primals_17);  primals_17 = None
    mul_216: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_215, 128)
    sum_78: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True)
    mul_217: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_215, mul_35);  mul_215 = None
    sum_79: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_217, [2], True);  mul_217 = None
    mul_218: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_35, sum_79);  sum_79 = None
    sub_67: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(mul_216, sum_78);  mul_216 = sum_78 = None
    sub_68: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(sub_67, mul_218);  sub_67 = mul_218 = None
    mul_219: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(div_19, sub_68);  div_19 = sub_68 = None
    mul_220: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_278, mul_35);  mul_35 = None
    sum_80: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_220, [0, 1]);  mul_220 = None
    sum_81: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_278, [0, 1]);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_132: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(slice_scatter_54, mul_219);  slice_scatter_54 = mul_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_279: "f32[6280, 128]" = torch.ops.aten.view.default(add_132, [6280, 128])
    mm_38: "f32[6280, 128]" = torch.ops.aten.mm.default(view_279, permute_221);  permute_221 = None
    permute_222: "f32[128, 6280]" = torch.ops.aten.permute.default(view_279, [1, 0])
    mm_39: "f32[128, 128]" = torch.ops.aten.mm.default(permute_222, view_77);  permute_222 = view_77 = None
    permute_223: "f32[128, 128]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_82: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_279, [0], True);  view_279 = None
    view_280: "f32[128]" = torch.ops.aten.view.default(sum_82, [128]);  sum_82 = None
    permute_224: "f32[128, 128]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    view_281: "f32[8, 785, 128]" = torch.ops.aten.view.default(mm_38, [8, 785, 128]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_282: "f32[8, 785, 8, 16]" = torch.ops.aten.view.default(view_281, [8, 785, 8, 16]);  view_281 = None
    permute_225: "f32[8, 8, 785, 16]" = torch.ops.aten.permute.default(view_282, [0, 2, 1, 3]);  view_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_221: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(permute_225, 0.25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_12: "f32[8, 8, 784, 16]" = torch.ops.aten.constant_pad_nd.default(permute_225, [0, 0, -1, 0, 0, 0]);  permute_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_222: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(constant_pad_nd_12, slice_47);  slice_47 = None
    mul_223: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(constant_pad_nd_12, permute_42);  constant_pad_nd_12 = permute_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_226: "f32[8, 8, 16, 784]" = torch.ops.aten.permute.default(mul_222, [0, 1, 3, 2]);  mul_222 = None
    view_283: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_226, [8, 128, 28, 28]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_134: "f32[8, 32, 28, 28]" = torch.ops.aten.slice.Tensor(view_283, 1, 0, 32)
    slice_135: "f32[8, 48, 28, 28]" = torch.ops.aten.slice.Tensor(view_283, 1, 32, 80)
    slice_136: "f32[8, 48, 28, 28]" = torch.ops.aten.slice.Tensor(view_283, 1, 80, 128);  view_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_18 = torch.ops.aten.convolution_backward.default(slice_136, getitem_41, primals_79, [48], [1, 1], [3, 3], [1, 1], False, [0, 0], 48, [True, True, True]);  slice_136 = getitem_41 = None
    getitem_144: "f32[8, 48, 28, 28]" = convolution_backward_18[0]
    getitem_145: "f32[48, 1, 7, 7]" = convolution_backward_18[1]
    getitem_146: "f32[48]" = convolution_backward_18[2];  convolution_backward_18 = None
    convolution_backward_19 = torch.ops.aten.convolution_backward.default(slice_135, getitem_40, primals_77, [48], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, True]);  slice_135 = getitem_40 = None
    getitem_147: "f32[8, 48, 28, 28]" = convolution_backward_19[0]
    getitem_148: "f32[48, 1, 5, 5]" = convolution_backward_19[1]
    getitem_149: "f32[48]" = convolution_backward_19[2];  convolution_backward_19 = None
    convolution_backward_20 = torch.ops.aten.convolution_backward.default(slice_134, getitem_39, primals_75, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, True]);  slice_134 = getitem_39 = None
    getitem_150: "f32[8, 32, 28, 28]" = convolution_backward_20[0]
    getitem_151: "f32[32, 1, 3, 3]" = convolution_backward_20[1]
    getitem_152: "f32[32]" = convolution_backward_20[2];  convolution_backward_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_28: "f32[8, 128, 28, 28]" = torch.ops.aten.cat.default([getitem_150, getitem_147, getitem_144], 1);  getitem_150 = getitem_147 = getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_284: "f32[8, 8, 16, 784]" = torch.ops.aten.view.default(cat_28, [8, 8, 16, 784]);  cat_28 = None
    permute_227: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_284, [0, 1, 3, 2]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_default_56: "f32[8, 8, 784, 16]" = torch.ops.aten.full.default([8, 8, 784, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_55: "f32[8, 8, 784, 16]" = torch.ops.aten.slice_scatter.default(full_default_56, permute_227, 3, 0, 9223372036854775807);  permute_227 = None
    full_default_57: "f32[8, 8, 785, 16]" = torch.ops.aten.full.default([8, 8, 785, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_56: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_55, 2, 1, 9223372036854775807);  slice_scatter_55 = None
    slice_scatter_57: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_56, 1, 0, 9223372036854775807);  slice_scatter_56 = None
    slice_scatter_58: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_57, 0, 0, 9223372036854775807);  slice_scatter_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_scatter_59: "f32[8, 8, 784, 16]" = torch.ops.aten.slice_scatter.default(full_default_56, mul_223, 3, 0, 9223372036854775807);  mul_223 = None
    slice_scatter_60: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_59, 2, 1, 9223372036854775807);  slice_scatter_59 = None
    slice_scatter_61: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_60, 1, 0, 9223372036854775807);  slice_scatter_60 = None
    slice_scatter_62: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_61, 0, 0, 9223372036854775807);  slice_scatter_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_81: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(mul_221, memory_format = torch.contiguous_format);  mul_221 = None
    view_285: "f32[64, 785, 16]" = torch.ops.aten.view.default(clone_81, [64, 785, 16]);  clone_81 = None
    bmm_32: "f32[64, 16, 16]" = torch.ops.aten.bmm.default(permute_228, view_285);  permute_228 = None
    bmm_33: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(view_285, permute_229);  view_285 = permute_229 = None
    view_286: "f32[8, 8, 16, 16]" = torch.ops.aten.view.default(bmm_32, [8, 8, 16, 16]);  bmm_32 = None
    view_287: "f32[8, 8, 785, 16]" = torch.ops.aten.view.default(bmm_33, [8, 8, 785, 16]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_133: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(slice_scatter_62, view_287);  slice_scatter_62 = view_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_288: "f32[64, 16, 16]" = torch.ops.aten.view.default(view_286, [64, 16, 16]);  view_286 = None
    bmm_34: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(permute_230, view_288);  permute_230 = None
    bmm_35: "f32[64, 16, 785]" = torch.ops.aten.bmm.default(view_288, permute_231);  view_288 = permute_231 = None
    view_289: "f32[8, 8, 785, 16]" = torch.ops.aten.view.default(bmm_34, [8, 8, 785, 16]);  bmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_134: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(slice_scatter_58, view_289);  slice_scatter_58 = view_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_290: "f32[8, 8, 16, 785]" = torch.ops.aten.view.default(bmm_35, [8, 8, 16, 785]);  bmm_35 = None
    permute_232: "f32[8, 8, 785, 16]" = torch.ops.aten.permute.default(view_290, [0, 1, 3, 2]);  view_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    mul_224: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(permute_232, alias_12);  permute_232 = None
    sum_83: "f32[8, 8, 1, 16]" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
    mul_225: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(alias_12, sum_83);  alias_12 = sum_83 = None
    sub_69: "f32[8, 8, 785, 16]" = torch.ops.aten.sub.Tensor(mul_224, mul_225);  mul_224 = mul_225 = None
    clone_82: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(sub_69, memory_format = torch.contiguous_format);  sub_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_29: "f32[24, 8, 785, 16]" = torch.ops.aten.cat.default([add_133, clone_82, add_134]);  add_133 = clone_82 = add_134 = None
    view_291: "f32[3, 8, 8, 785, 16]" = torch.ops.aten.view.default(cat_29, [3, 8, 8, 785, 16]);  cat_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_233: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.permute.default(view_291, [1, 3, 0, 2, 4]);  view_291 = None
    clone_83: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.clone.default(permute_233, memory_format = torch.contiguous_format);  permute_233 = None
    view_292: "f32[8, 785, 384]" = torch.ops.aten.view.default(clone_83, [8, 785, 384]);  clone_83 = None
    view_293: "f32[6280, 384]" = torch.ops.aten.view.default(view_292, [6280, 384]);  view_292 = None
    mm_40: "f32[6280, 128]" = torch.ops.aten.mm.default(view_293, permute_234);  permute_234 = None
    permute_235: "f32[384, 6280]" = torch.ops.aten.permute.default(view_293, [1, 0])
    mm_41: "f32[384, 128]" = torch.ops.aten.mm.default(permute_235, view_65);  permute_235 = view_65 = None
    permute_236: "f32[128, 384]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_84: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_293, [0], True);  view_293 = None
    view_294: "f32[384]" = torch.ops.aten.view.default(sum_84, [384]);  sum_84 = None
    permute_237: "f32[384, 128]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
    view_295: "f32[8, 785, 128]" = torch.ops.aten.view.default(mm_40, [8, 785, 128]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_227: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_295, primals_15);  primals_15 = None
    mul_228: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_227, 128)
    sum_85: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_227, [2], True)
    mul_229: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_227, mul_31);  mul_227 = None
    sum_86: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True);  mul_229 = None
    mul_230: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_31, sum_86);  sum_86 = None
    sub_71: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(mul_228, sum_85);  mul_228 = sum_85 = None
    sub_72: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(sub_71, mul_230);  sub_71 = mul_230 = None
    div_20: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_8, 128);  rsqrt_8 = None
    mul_231: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(div_20, sub_72);  div_20 = sub_72 = None
    mul_232: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_295, mul_31);  mul_31 = None
    sum_87: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_232, [0, 1]);  mul_232 = None
    sum_88: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_295, [0, 1]);  view_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_135: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_132, mul_231);  add_132 = mul_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_137: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_135, 1, 0, 1)
    slice_138: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(add_135, 1, 1, 785);  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_238: "f32[8, 128, 784]" = torch.ops.aten.permute.default(slice_138, [0, 2, 1]);  slice_138 = None
    view_296: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_238, [8, 128, 28, 28]);  permute_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_21 = torch.ops.aten.convolution_backward.default(view_296, view_63, primals_71, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, True]);  view_63 = None
    getitem_153: "f32[8, 128, 28, 28]" = convolution_backward_21[0]
    getitem_154: "f32[128, 1, 3, 3]" = convolution_backward_21[1]
    getitem_155: "f32[128]" = convolution_backward_21[2];  convolution_backward_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_136: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(view_296, getitem_153);  view_296 = getitem_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_297: "f32[8, 128, 784]" = torch.ops.aten.view.default(add_136, [8, 128, 784]);  add_136 = None
    permute_239: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_scatter_63: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_default_54, permute_239, 1, 1, 9223372036854775807);  permute_239 = None
    slice_scatter_64: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_default_54, slice_scatter_63, 0, 0, 9223372036854775807);  slice_scatter_63 = None
    slice_scatter_65: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_default_54, slice_137, 1, 0, 1);  slice_137 = None
    slice_scatter_66: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_default_54, slice_scatter_65, 0, 0, 9223372036854775807);  slice_scatter_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_137: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(slice_scatter_64, slice_scatter_66);  slice_scatter_64 = slice_scatter_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_298: "f32[6280, 128]" = torch.ops.aten.view.default(add_137, [6280, 128])
    mm_42: "f32[6280, 1024]" = torch.ops.aten.mm.default(view_298, permute_240);  permute_240 = None
    permute_241: "f32[128, 6280]" = torch.ops.aten.permute.default(view_298, [1, 0])
    mm_43: "f32[128, 1024]" = torch.ops.aten.mm.default(permute_241, view_61);  permute_241 = view_61 = None
    permute_242: "f32[1024, 128]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_89: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_298, [0], True);  view_298 = None
    view_299: "f32[128]" = torch.ops.aten.view.default(sum_89, [128]);  sum_89 = None
    permute_243: "f32[128, 1024]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    view_300: "f32[8, 785, 1024]" = torch.ops.aten.view.default(mm_42, [8, 785, 1024]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_234: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(add_29, 0.5);  add_29 = None
    mul_235: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_60, view_60)
    mul_236: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(mul_235, -0.5);  mul_235 = None
    exp_13: "f32[8, 785, 1024]" = torch.ops.aten.exp.default(mul_236);  mul_236 = None
    mul_237: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_238: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_60, mul_237);  view_60 = mul_237 = None
    add_139: "f32[8, 785, 1024]" = torch.ops.aten.add.Tensor(mul_234, mul_238);  mul_234 = mul_238 = None
    mul_239: "f32[8, 785, 1024]" = torch.ops.aten.mul.Tensor(view_300, add_139);  view_300 = add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_301: "f32[6280, 1024]" = torch.ops.aten.view.default(mul_239, [6280, 1024]);  mul_239 = None
    mm_44: "f32[6280, 128]" = torch.ops.aten.mm.default(view_301, permute_244);  permute_244 = None
    permute_245: "f32[1024, 6280]" = torch.ops.aten.permute.default(view_301, [1, 0])
    mm_45: "f32[1024, 128]" = torch.ops.aten.mm.default(permute_245, view_59);  permute_245 = view_59 = None
    permute_246: "f32[128, 1024]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_90: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_301, [0], True);  view_301 = None
    view_302: "f32[1024]" = torch.ops.aten.view.default(sum_90, [1024]);  sum_90 = None
    permute_247: "f32[1024, 128]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
    view_303: "f32[8, 785, 128]" = torch.ops.aten.view.default(mm_44, [8, 785, 128]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_241: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_303, primals_13);  primals_13 = None
    mul_242: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_241, 128)
    sum_91: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [2], True)
    mul_243: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_241, mul_26);  mul_241 = None
    sum_92: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
    mul_244: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_26, sum_92);  sum_92 = None
    sub_74: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(mul_242, sum_91);  mul_242 = sum_91 = None
    sub_75: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(sub_74, mul_244);  sub_74 = mul_244 = None
    mul_245: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(div_21, sub_75);  div_21 = sub_75 = None
    mul_246: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_303, mul_26);  mul_26 = None
    sum_93: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_246, [0, 1]);  mul_246 = None
    sum_94: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_303, [0, 1]);  view_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_140: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_137, mul_245);  add_137 = mul_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_304: "f32[6280, 128]" = torch.ops.aten.view.default(add_140, [6280, 128])
    mm_46: "f32[6280, 128]" = torch.ops.aten.mm.default(view_304, permute_248);  permute_248 = None
    permute_249: "f32[128, 6280]" = torch.ops.aten.permute.default(view_304, [1, 0])
    mm_47: "f32[128, 128]" = torch.ops.aten.mm.default(permute_249, view_57);  permute_249 = view_57 = None
    permute_250: "f32[128, 128]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_95: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
    view_305: "f32[128]" = torch.ops.aten.view.default(sum_95, [128]);  sum_95 = None
    permute_251: "f32[128, 128]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
    view_306: "f32[8, 785, 128]" = torch.ops.aten.view.default(mm_46, [8, 785, 128]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_307: "f32[8, 785, 8, 16]" = torch.ops.aten.view.default(view_306, [8, 785, 8, 16]);  view_306 = None
    permute_252: "f32[8, 8, 785, 16]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_247: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(permute_252, 0.25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_13: "f32[8, 8, 784, 16]" = torch.ops.aten.constant_pad_nd.default(permute_252, [0, 0, -1, 0, 0, 0]);  permute_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_248: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(constant_pad_nd_13, slice_35);  slice_35 = None
    mul_249: "f32[8, 8, 784, 16]" = torch.ops.aten.mul.Tensor(constant_pad_nd_13, permute_31);  constant_pad_nd_13 = permute_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_253: "f32[8, 8, 16, 784]" = torch.ops.aten.permute.default(mul_248, [0, 1, 3, 2]);  mul_248 = None
    view_308: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_253, [8, 128, 28, 28]);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_139: "f32[8, 32, 28, 28]" = torch.ops.aten.slice.Tensor(view_308, 1, 0, 32)
    slice_140: "f32[8, 48, 28, 28]" = torch.ops.aten.slice.Tensor(view_308, 1, 32, 80)
    slice_141: "f32[8, 48, 28, 28]" = torch.ops.aten.slice.Tensor(view_308, 1, 80, 128);  view_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_22 = torch.ops.aten.convolution_backward.default(slice_141, getitem_31, primals_79, [48], [1, 1], [3, 3], [1, 1], False, [0, 0], 48, [True, True, True]);  slice_141 = getitem_31 = primals_79 = None
    getitem_156: "f32[8, 48, 28, 28]" = convolution_backward_22[0]
    getitem_157: "f32[48, 1, 7, 7]" = convolution_backward_22[1]
    getitem_158: "f32[48]" = convolution_backward_22[2];  convolution_backward_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_141: "f32[48, 1, 7, 7]" = torch.ops.aten.add.Tensor(getitem_145, getitem_157);  getitem_145 = getitem_157 = None
    add_142: "f32[48]" = torch.ops.aten.add.Tensor(getitem_146, getitem_158);  getitem_146 = getitem_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_23 = torch.ops.aten.convolution_backward.default(slice_140, getitem_30, primals_77, [48], [1, 1], [2, 2], [1, 1], False, [0, 0], 48, [True, True, True]);  slice_140 = getitem_30 = primals_77 = None
    getitem_159: "f32[8, 48, 28, 28]" = convolution_backward_23[0]
    getitem_160: "f32[48, 1, 5, 5]" = convolution_backward_23[1]
    getitem_161: "f32[48]" = convolution_backward_23[2];  convolution_backward_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_143: "f32[48, 1, 5, 5]" = torch.ops.aten.add.Tensor(getitem_148, getitem_160);  getitem_148 = getitem_160 = None
    add_144: "f32[48]" = torch.ops.aten.add.Tensor(getitem_149, getitem_161);  getitem_149 = getitem_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_24 = torch.ops.aten.convolution_backward.default(slice_139, getitem_29, primals_75, [32], [1, 1], [1, 1], [1, 1], False, [0, 0], 32, [True, True, True]);  slice_139 = getitem_29 = primals_75 = None
    getitem_162: "f32[8, 32, 28, 28]" = convolution_backward_24[0]
    getitem_163: "f32[32, 1, 3, 3]" = convolution_backward_24[1]
    getitem_164: "f32[32]" = convolution_backward_24[2];  convolution_backward_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_145: "f32[32, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_151, getitem_163);  getitem_151 = getitem_163 = None
    add_146: "f32[32]" = torch.ops.aten.add.Tensor(getitem_152, getitem_164);  getitem_152 = getitem_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_30: "f32[8, 128, 28, 28]" = torch.ops.aten.cat.default([getitem_162, getitem_159, getitem_156], 1);  getitem_162 = getitem_159 = getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_309: "f32[8, 8, 16, 784]" = torch.ops.aten.view.default(cat_30, [8, 8, 16, 784]);  cat_30 = None
    permute_254: "f32[8, 8, 784, 16]" = torch.ops.aten.permute.default(view_309, [0, 1, 3, 2]);  view_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_scatter_67: "f32[8, 8, 784, 16]" = torch.ops.aten.slice_scatter.default(full_default_56, permute_254, 3, 0, 9223372036854775807);  permute_254 = None
    slice_scatter_68: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_67, 2, 1, 9223372036854775807);  slice_scatter_67 = None
    slice_scatter_69: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_68, 1, 0, 9223372036854775807);  slice_scatter_68 = None
    slice_scatter_70: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_69, 0, 0, 9223372036854775807);  slice_scatter_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_scatter_71: "f32[8, 8, 784, 16]" = torch.ops.aten.slice_scatter.default(full_default_56, mul_249, 3, 0, 9223372036854775807);  full_default_56 = mul_249 = None
    slice_scatter_72: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_71, 2, 1, 9223372036854775807);  slice_scatter_71 = None
    slice_scatter_73: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_72, 1, 0, 9223372036854775807);  slice_scatter_72 = None
    slice_scatter_74: "f32[8, 8, 785, 16]" = torch.ops.aten.slice_scatter.default(full_default_57, slice_scatter_73, 0, 0, 9223372036854775807);  full_default_57 = slice_scatter_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_84: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(mul_247, memory_format = torch.contiguous_format);  mul_247 = None
    view_310: "f32[64, 785, 16]" = torch.ops.aten.view.default(clone_84, [64, 785, 16]);  clone_84 = None
    bmm_36: "f32[64, 16, 16]" = torch.ops.aten.bmm.default(permute_255, view_310);  permute_255 = None
    bmm_37: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(view_310, permute_256);  view_310 = permute_256 = None
    view_311: "f32[8, 8, 16, 16]" = torch.ops.aten.view.default(bmm_36, [8, 8, 16, 16]);  bmm_36 = None
    view_312: "f32[8, 8, 785, 16]" = torch.ops.aten.view.default(bmm_37, [8, 8, 785, 16]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_147: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(slice_scatter_74, view_312);  slice_scatter_74 = view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_313: "f32[64, 16, 16]" = torch.ops.aten.view.default(view_311, [64, 16, 16]);  view_311 = None
    bmm_38: "f32[64, 785, 16]" = torch.ops.aten.bmm.default(permute_257, view_313);  permute_257 = None
    bmm_39: "f32[64, 16, 785]" = torch.ops.aten.bmm.default(view_313, permute_258);  view_313 = permute_258 = None
    view_314: "f32[8, 8, 785, 16]" = torch.ops.aten.view.default(bmm_38, [8, 8, 785, 16]);  bmm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_148: "f32[8, 8, 785, 16]" = torch.ops.aten.add.Tensor(slice_scatter_70, view_314);  slice_scatter_70 = view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_315: "f32[8, 8, 16, 785]" = torch.ops.aten.view.default(bmm_39, [8, 8, 16, 785]);  bmm_39 = None
    permute_259: "f32[8, 8, 785, 16]" = torch.ops.aten.permute.default(view_315, [0, 1, 3, 2]);  view_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    mul_250: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(permute_259, alias_13);  permute_259 = None
    sum_96: "f32[8, 8, 1, 16]" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True)
    mul_251: "f32[8, 8, 785, 16]" = torch.ops.aten.mul.Tensor(alias_13, sum_96);  alias_13 = sum_96 = None
    sub_76: "f32[8, 8, 785, 16]" = torch.ops.aten.sub.Tensor(mul_250, mul_251);  mul_250 = mul_251 = None
    clone_85: "f32[8, 8, 785, 16]" = torch.ops.aten.clone.default(sub_76, memory_format = torch.contiguous_format);  sub_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_31: "f32[24, 8, 785, 16]" = torch.ops.aten.cat.default([add_147, clone_85, add_148]);  add_147 = clone_85 = add_148 = None
    view_316: "f32[3, 8, 8, 785, 16]" = torch.ops.aten.view.default(cat_31, [3, 8, 8, 785, 16]);  cat_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_260: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.permute.default(view_316, [1, 3, 0, 2, 4]);  view_316 = None
    clone_86: "f32[8, 785, 3, 8, 16]" = torch.ops.aten.clone.default(permute_260, memory_format = torch.contiguous_format);  permute_260 = None
    view_317: "f32[8, 785, 384]" = torch.ops.aten.view.default(clone_86, [8, 785, 384]);  clone_86 = None
    view_318: "f32[6280, 384]" = torch.ops.aten.view.default(view_317, [6280, 384]);  view_317 = None
    mm_48: "f32[6280, 128]" = torch.ops.aten.mm.default(view_318, permute_261);  permute_261 = None
    permute_262: "f32[384, 6280]" = torch.ops.aten.permute.default(view_318, [1, 0])
    mm_49: "f32[384, 128]" = torch.ops.aten.mm.default(permute_262, view_45);  permute_262 = view_45 = None
    permute_263: "f32[128, 384]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_97: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_318, [0], True);  view_318 = None
    view_319: "f32[384]" = torch.ops.aten.view.default(sum_97, [384]);  sum_97 = None
    permute_264: "f32[384, 128]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    view_320: "f32[8, 785, 128]" = torch.ops.aten.view.default(mm_48, [8, 785, 128]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_253: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_320, primals_11);  primals_11 = None
    mul_254: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_253, 128)
    sum_98: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_253, [2], True)
    mul_255: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_253, mul_22);  mul_253 = None
    sum_99: "f32[8, 785, 1]" = torch.ops.aten.sum.dim_IntList(mul_255, [2], True);  mul_255 = None
    mul_256: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(mul_22, sum_99);  sum_99 = None
    sub_78: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(mul_254, sum_98);  mul_254 = sum_98 = None
    sub_79: "f32[8, 785, 128]" = torch.ops.aten.sub.Tensor(sub_78, mul_256);  sub_78 = mul_256 = None
    div_22: "f32[8, 785, 1]" = torch.ops.aten.div.Tensor(rsqrt_6, 128);  rsqrt_6 = None
    mul_257: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(div_22, sub_79);  div_22 = sub_79 = None
    mul_258: "f32[8, 785, 128]" = torch.ops.aten.mul.Tensor(view_320, mul_22);  mul_22 = None
    sum_100: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1]);  mul_258 = None
    sum_101: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_320, [0, 1]);  view_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_149: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(add_140, mul_257);  add_140 = mul_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_142: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_149, 1, 0, 1)
    slice_143: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(add_149, 1, 1, 785);  add_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_265: "f32[8, 128, 784]" = torch.ops.aten.permute.default(slice_143, [0, 2, 1]);  slice_143 = None
    view_321: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_265, [8, 128, 28, 28]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_25 = torch.ops.aten.convolution_backward.default(view_321, view_43, primals_71, [128], [1, 1], [1, 1], [1, 1], False, [0, 0], 128, [True, True, True]);  view_43 = primals_71 = None
    getitem_165: "f32[8, 128, 28, 28]" = convolution_backward_25[0]
    getitem_166: "f32[128, 1, 3, 3]" = convolution_backward_25[1]
    getitem_167: "f32[128]" = convolution_backward_25[2];  convolution_backward_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_150: "f32[8, 128, 28, 28]" = torch.ops.aten.add.Tensor(view_321, getitem_165);  view_321 = getitem_165 = None
    add_151: "f32[128, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_154, getitem_166);  getitem_154 = getitem_166 = None
    add_152: "f32[128]" = torch.ops.aten.add.Tensor(getitem_155, getitem_167);  getitem_155 = getitem_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_322: "f32[8, 128, 784]" = torch.ops.aten.view.default(add_150, [8, 128, 784]);  add_150 = None
    permute_266: "f32[8, 784, 128]" = torch.ops.aten.permute.default(view_322, [0, 2, 1]);  view_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_scatter_75: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_default_54, permute_266, 1, 1, 9223372036854775807);  permute_266 = None
    slice_scatter_76: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_default_54, slice_scatter_75, 0, 0, 9223372036854775807);  slice_scatter_75 = None
    slice_scatter_77: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_default_54, slice_142, 1, 0, 1);  slice_142 = None
    slice_scatter_78: "f32[8, 785, 128]" = torch.ops.aten.slice_scatter.default(full_default_54, slice_scatter_77, 0, 0, 9223372036854775807);  full_default_54 = slice_scatter_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_153: "f32[8, 785, 128]" = torch.ops.aten.add.Tensor(slice_scatter_76, slice_scatter_78);  slice_scatter_76 = slice_scatter_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_144: "f32[8, 1, 128]" = torch.ops.aten.slice.Tensor(add_153, 1, 0, 1)
    slice_145: "f32[8, 784, 128]" = torch.ops.aten.slice.Tensor(add_153, 1, 1, 785);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    sum_102: "f32[1, 1, 128]" = torch.ops.aten.sum.dim_IntList(slice_144, [0], True);  slice_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_87: "f32[8, 784, 128]" = torch.ops.aten.clone.default(slice_145, memory_format = torch.contiguous_format);  slice_145 = None
    mul_260: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_87, primals_69);  primals_69 = None
    mul_261: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_260, 128)
    sum_103: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_260, [2], True)
    mul_262: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_260, mul_20);  mul_260 = None
    sum_104: "f32[8, 784, 1]" = torch.ops.aten.sum.dim_IntList(mul_262, [2], True);  mul_262 = None
    mul_263: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(mul_20, sum_104);  sum_104 = None
    sub_81: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(mul_261, sum_103);  mul_261 = sum_103 = None
    sub_82: "f32[8, 784, 128]" = torch.ops.aten.sub.Tensor(sub_81, mul_263);  sub_81 = mul_263 = None
    mul_264: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(div_23, sub_82);  div_23 = sub_82 = None
    mul_265: "f32[8, 784, 128]" = torch.ops.aten.mul.Tensor(clone_87, mul_20);  mul_20 = None
    sum_105: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_265, [0, 1]);  mul_265 = None
    sum_106: "f32[128]" = torch.ops.aten.sum.dim_IntList(clone_87, [0, 1]);  clone_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_267: "f32[8, 128, 784]" = torch.ops.aten.permute.default(mul_264, [0, 2, 1]);  mul_264 = None
    view_323: "f32[8, 128, 28, 28]" = torch.ops.aten.view.default(permute_267, [8, 128, 28, 28]);  permute_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_26 = torch.ops.aten.convolution_backward.default(view_323, clone_15, primals_67, [128], [2, 2], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]);  view_323 = clone_15 = primals_67 = None
    getitem_168: "f32[8, 64, 56, 56]" = convolution_backward_26[0]
    getitem_169: "f32[128, 64, 2, 2]" = convolution_backward_26[1]
    getitem_170: "f32[128]" = convolution_backward_26[2];  convolution_backward_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:579, code: x1_nocls = remove_cls(x1).reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()
    permute_268: "f32[8, 56, 56, 64]" = torch.ops.aten.permute.default(getitem_168, [0, 2, 3, 1]);  getitem_168 = None
    view_324: "f32[8, 3136, 64]" = torch.ops.aten.view.default(permute_268, [8, 3136, 64]);  permute_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:684, code: return x[:, 1:, :]
    full_default_80: "f32[8, 3136, 64]" = torch.ops.aten.full.default([8, 3136, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_79: "f32[8, 3136, 64]" = torch.ops.aten.slice_scatter.default(full_default_80, view_324, 2, 0, 9223372036854775807);  full_default_80 = view_324 = None
    full_default_81: "f32[8, 3137, 64]" = torch.ops.aten.full.default([8, 3137, 64], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_80: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_default_81, slice_scatter_79, 1, 1, 9223372036854775807);  slice_scatter_79 = None
    slice_scatter_81: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_default_81, slice_scatter_80, 0, 0, 9223372036854775807);  slice_scatter_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_325: "f32[25096, 64]" = torch.ops.aten.view.default(slice_scatter_81, [25096, 64])
    mm_50: "f32[25096, 512]" = torch.ops.aten.mm.default(view_325, permute_269);  permute_269 = None
    permute_270: "f32[64, 25096]" = torch.ops.aten.permute.default(view_325, [1, 0])
    mm_51: "f32[64, 512]" = torch.ops.aten.mm.default(permute_270, view_39);  permute_270 = view_39 = None
    permute_271: "f32[512, 64]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_107: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_325, [0], True);  view_325 = None
    view_326: "f32[64]" = torch.ops.aten.view.default(sum_107, [64]);  sum_107 = None
    permute_272: "f32[64, 512]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
    view_327: "f32[8, 3137, 512]" = torch.ops.aten.view.default(mm_50, [8, 3137, 512]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_267: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
    mul_268: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_38, view_38)
    mul_269: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(mul_268, -0.5);  mul_268 = None
    exp_14: "f32[8, 3137, 512]" = torch.ops.aten.exp.default(mul_269);  mul_269 = None
    mul_270: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_271: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_38, mul_270);  view_38 = mul_270 = None
    add_155: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(mul_267, mul_271);  mul_267 = mul_271 = None
    mul_272: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_327, add_155);  view_327 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_328: "f32[25096, 512]" = torch.ops.aten.view.default(mul_272, [25096, 512]);  mul_272 = None
    mm_52: "f32[25096, 64]" = torch.ops.aten.mm.default(view_328, permute_273);  permute_273 = None
    permute_274: "f32[512, 25096]" = torch.ops.aten.permute.default(view_328, [1, 0])
    mm_53: "f32[512, 64]" = torch.ops.aten.mm.default(permute_274, view_37);  permute_274 = view_37 = None
    permute_275: "f32[64, 512]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_108: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
    view_329: "f32[512]" = torch.ops.aten.view.default(sum_108, [512]);  sum_108 = None
    permute_276: "f32[512, 64]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
    view_330: "f32[8, 3137, 64]" = torch.ops.aten.view.default(mm_52, [8, 3137, 64]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_274: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_330, primals_8);  primals_8 = None
    mul_275: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_274, 64)
    sum_109: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_274, [2], True)
    mul_276: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_274, mul_15);  mul_274 = None
    sum_110: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_276, [2], True);  mul_276 = None
    mul_277: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_15, sum_110);  sum_110 = None
    sub_84: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(mul_275, sum_109);  mul_275 = sum_109 = None
    sub_85: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(sub_84, mul_277);  sub_84 = mul_277 = None
    mul_278: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(div_24, sub_85);  div_24 = sub_85 = None
    mul_279: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_330, mul_15);  mul_15 = None
    sum_111: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_279, [0, 1]);  mul_279 = None
    sum_112: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_330, [0, 1]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_156: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(slice_scatter_81, mul_278);  slice_scatter_81 = mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_331: "f32[25096, 64]" = torch.ops.aten.view.default(add_156, [25096, 64])
    mm_54: "f32[25096, 64]" = torch.ops.aten.mm.default(view_331, permute_277);  permute_277 = None
    permute_278: "f32[64, 25096]" = torch.ops.aten.permute.default(view_331, [1, 0])
    mm_55: "f32[64, 64]" = torch.ops.aten.mm.default(permute_278, view_35);  permute_278 = view_35 = None
    permute_279: "f32[64, 64]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_113: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[64]" = torch.ops.aten.view.default(sum_113, [64]);  sum_113 = None
    permute_280: "f32[64, 64]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
    view_333: "f32[8, 3137, 64]" = torch.ops.aten.view.default(mm_54, [8, 3137, 64]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_334: "f32[8, 3137, 8, 8]" = torch.ops.aten.view.default(view_333, [8, 3137, 8, 8]);  view_333 = None
    permute_281: "f32[8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_334, [0, 2, 1, 3]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_280: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(permute_281, 0.3535533905932738)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_14: "f32[8, 8, 3136, 8]" = torch.ops.aten.constant_pad_nd.default(permute_281, [0, 0, -1, 0, 0, 0]);  permute_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_281: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(constant_pad_nd_14, slice_20);  slice_20 = None
    mul_282: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(constant_pad_nd_14, permute_18);  constant_pad_nd_14 = permute_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_282: "f32[8, 8, 8, 3136]" = torch.ops.aten.permute.default(mul_281, [0, 1, 3, 2]);  mul_281 = None
    view_335: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_282, [8, 64, 56, 56]);  permute_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_146: "f32[8, 16, 56, 56]" = torch.ops.aten.slice.Tensor(view_335, 1, 0, 16)
    slice_147: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(view_335, 1, 16, 40)
    slice_148: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(view_335, 1, 40, 64);  view_335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_27 = torch.ops.aten.convolution_backward.default(slice_148, getitem_19, primals_51, [24], [1, 1], [3, 3], [1, 1], False, [0, 0], 24, [True, True, True]);  slice_148 = getitem_19 = None
    getitem_171: "f32[8, 24, 56, 56]" = convolution_backward_27[0]
    getitem_172: "f32[24, 1, 7, 7]" = convolution_backward_27[1]
    getitem_173: "f32[24]" = convolution_backward_27[2];  convolution_backward_27 = None
    convolution_backward_28 = torch.ops.aten.convolution_backward.default(slice_147, getitem_18, primals_49, [24], [1, 1], [2, 2], [1, 1], False, [0, 0], 24, [True, True, True]);  slice_147 = getitem_18 = None
    getitem_174: "f32[8, 24, 56, 56]" = convolution_backward_28[0]
    getitem_175: "f32[24, 1, 5, 5]" = convolution_backward_28[1]
    getitem_176: "f32[24]" = convolution_backward_28[2];  convolution_backward_28 = None
    convolution_backward_29 = torch.ops.aten.convolution_backward.default(slice_146, getitem_17, primals_47, [16], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, True]);  slice_146 = getitem_17 = None
    getitem_177: "f32[8, 16, 56, 56]" = convolution_backward_29[0]
    getitem_178: "f32[16, 1, 3, 3]" = convolution_backward_29[1]
    getitem_179: "f32[16]" = convolution_backward_29[2];  convolution_backward_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_32: "f32[8, 64, 56, 56]" = torch.ops.aten.cat.default([getitem_177, getitem_174, getitem_171], 1);  getitem_177 = getitem_174 = getitem_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_336: "f32[8, 8, 8, 3136]" = torch.ops.aten.view.default(cat_32, [8, 8, 8, 3136]);  cat_32 = None
    permute_283: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_336, [0, 1, 3, 2]);  view_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    full_default_83: "f32[8, 8, 3136, 8]" = torch.ops.aten.full.default([8, 8, 3136, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_82: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice_scatter.default(full_default_83, permute_283, 3, 0, 9223372036854775807);  permute_283 = None
    full_default_84: "f32[8, 8, 3137, 8]" = torch.ops.aten.full.default([8, 8, 3137, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    slice_scatter_83: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_default_84, slice_scatter_82, 2, 1, 9223372036854775807);  slice_scatter_82 = None
    slice_scatter_84: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_default_84, slice_scatter_83, 1, 0, 9223372036854775807);  slice_scatter_83 = None
    slice_scatter_85: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_default_84, slice_scatter_84, 0, 0, 9223372036854775807);  slice_scatter_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_scatter_86: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice_scatter.default(full_default_83, mul_282, 3, 0, 9223372036854775807);  mul_282 = None
    slice_scatter_87: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_default_84, slice_scatter_86, 2, 1, 9223372036854775807);  slice_scatter_86 = None
    slice_scatter_88: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_default_84, slice_scatter_87, 1, 0, 9223372036854775807);  slice_scatter_87 = None
    slice_scatter_89: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_default_84, slice_scatter_88, 0, 0, 9223372036854775807);  slice_scatter_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_89: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(mul_280, memory_format = torch.contiguous_format);  mul_280 = None
    view_337: "f32[64, 3137, 8]" = torch.ops.aten.view.default(clone_89, [64, 3137, 8]);  clone_89 = None
    bmm_40: "f32[64, 8, 8]" = torch.ops.aten.bmm.default(permute_284, view_337);  permute_284 = None
    bmm_41: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(view_337, permute_285);  view_337 = permute_285 = None
    view_338: "f32[8, 8, 8, 8]" = torch.ops.aten.view.default(bmm_40, [8, 8, 8, 8]);  bmm_40 = None
    view_339: "f32[8, 8, 3137, 8]" = torch.ops.aten.view.default(bmm_41, [8, 8, 3137, 8]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_157: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(slice_scatter_89, view_339);  slice_scatter_89 = view_339 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_340: "f32[64, 8, 8]" = torch.ops.aten.view.default(view_338, [64, 8, 8]);  view_338 = None
    bmm_42: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(permute_286, view_340);  permute_286 = None
    bmm_43: "f32[64, 8, 3137]" = torch.ops.aten.bmm.default(view_340, permute_287);  view_340 = permute_287 = None
    view_341: "f32[8, 8, 3137, 8]" = torch.ops.aten.view.default(bmm_42, [8, 8, 3137, 8]);  bmm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_158: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(slice_scatter_85, view_341);  slice_scatter_85 = view_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_342: "f32[8, 8, 8, 3137]" = torch.ops.aten.view.default(bmm_43, [8, 8, 8, 3137]);  bmm_43 = None
    permute_288: "f32[8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_342, [0, 1, 3, 2]);  view_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    mul_283: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(permute_288, alias_14);  permute_288 = None
    sum_114: "f32[8, 8, 1, 8]" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True)
    mul_284: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(alias_14, sum_114);  alias_14 = sum_114 = None
    sub_86: "f32[8, 8, 3137, 8]" = torch.ops.aten.sub.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
    clone_90: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(sub_86, memory_format = torch.contiguous_format);  sub_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_33: "f32[24, 8, 3137, 8]" = torch.ops.aten.cat.default([add_157, clone_90, add_158]);  add_157 = clone_90 = add_158 = None
    view_343: "f32[3, 8, 8, 3137, 8]" = torch.ops.aten.view.default(cat_33, [3, 8, 8, 3137, 8]);  cat_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_289: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.permute.default(view_343, [1, 3, 0, 2, 4]);  view_343 = None
    clone_91: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
    view_344: "f32[8, 3137, 192]" = torch.ops.aten.view.default(clone_91, [8, 3137, 192]);  clone_91 = None
    view_345: "f32[25096, 192]" = torch.ops.aten.view.default(view_344, [25096, 192]);  view_344 = None
    mm_56: "f32[25096, 64]" = torch.ops.aten.mm.default(view_345, permute_290);  permute_290 = None
    permute_291: "f32[192, 25096]" = torch.ops.aten.permute.default(view_345, [1, 0])
    mm_57: "f32[192, 64]" = torch.ops.aten.mm.default(permute_291, view_23);  permute_291 = view_23 = None
    permute_292: "f32[64, 192]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_115: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_345, [0], True);  view_345 = None
    view_346: "f32[192]" = torch.ops.aten.view.default(sum_115, [192]);  sum_115 = None
    permute_293: "f32[192, 64]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
    view_347: "f32[8, 3137, 64]" = torch.ops.aten.view.default(mm_56, [8, 3137, 64]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_286: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_347, primals_6);  primals_6 = None
    mul_287: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_286, 64)
    sum_116: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_286, [2], True)
    mul_288: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_286, mul_11);  mul_286 = None
    sum_117: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True);  mul_288 = None
    mul_289: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_11, sum_117);  sum_117 = None
    sub_88: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(mul_287, sum_116);  mul_287 = sum_116 = None
    sub_89: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(sub_88, mul_289);  sub_88 = mul_289 = None
    div_25: "f32[8, 3137, 1]" = torch.ops.aten.div.Tensor(rsqrt_3, 64);  rsqrt_3 = None
    mul_290: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(div_25, sub_89);  div_25 = sub_89 = None
    mul_291: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_347, mul_11);  mul_11 = None
    sum_118: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_291, [0, 1]);  mul_291 = None
    sum_119: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_347, [0, 1]);  view_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_159: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_156, mul_290);  add_156 = mul_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_149: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(add_159, 1, 0, 1)
    slice_150: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(add_159, 1, 1, 3137);  add_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_294: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(slice_150, [0, 2, 1]);  slice_150 = None
    view_348: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_294, [8, 64, 56, 56]);  permute_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_30 = torch.ops.aten.convolution_backward.default(view_348, view_21, primals_43, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, True]);  view_21 = None
    getitem_180: "f32[8, 64, 56, 56]" = convolution_backward_30[0]
    getitem_181: "f32[64, 1, 3, 3]" = convolution_backward_30[1]
    getitem_182: "f32[64]" = convolution_backward_30[2];  convolution_backward_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_160: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(view_348, getitem_180);  view_348 = getitem_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_349: "f32[8, 64, 3136]" = torch.ops.aten.view.default(add_160, [8, 64, 3136]);  add_160 = None
    permute_295: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_scatter_90: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_default_81, permute_295, 1, 1, 9223372036854775807);  permute_295 = None
    slice_scatter_91: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_default_81, slice_scatter_90, 0, 0, 9223372036854775807);  slice_scatter_90 = None
    slice_scatter_92: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_default_81, slice_149, 1, 0, 1);  slice_149 = None
    slice_scatter_93: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_default_81, slice_scatter_92, 0, 0, 9223372036854775807);  slice_scatter_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_161: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(slice_scatter_91, slice_scatter_93);  slice_scatter_91 = slice_scatter_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    view_350: "f32[25096, 64]" = torch.ops.aten.view.default(add_161, [25096, 64])
    mm_58: "f32[25096, 512]" = torch.ops.aten.mm.default(view_350, permute_296);  permute_296 = None
    permute_297: "f32[64, 25096]" = torch.ops.aten.permute.default(view_350, [1, 0])
    mm_59: "f32[64, 512]" = torch.ops.aten.mm.default(permute_297, view_19);  permute_297 = view_19 = None
    permute_298: "f32[512, 64]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_120: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_350, [0], True);  view_350 = None
    view_351: "f32[64]" = torch.ops.aten.view.default(sum_120, [64]);  sum_120 = None
    permute_299: "f32[64, 512]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
    view_352: "f32[8, 3137, 512]" = torch.ops.aten.view.default(mm_58, [8, 3137, 512]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    mul_293: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(add_9, 0.5);  add_9 = None
    mul_294: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_18, view_18)
    mul_295: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(mul_294, -0.5);  mul_294 = None
    exp_15: "f32[8, 3137, 512]" = torch.ops.aten.exp.default(mul_295);  mul_295 = None
    mul_296: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_297: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_18, mul_296);  view_18 = mul_296 = None
    add_163: "f32[8, 3137, 512]" = torch.ops.aten.add.Tensor(mul_293, mul_297);  mul_293 = mul_297 = None
    mul_298: "f32[8, 3137, 512]" = torch.ops.aten.mul.Tensor(view_352, add_163);  view_352 = add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    view_353: "f32[25096, 512]" = torch.ops.aten.view.default(mul_298, [25096, 512]);  mul_298 = None
    mm_60: "f32[25096, 64]" = torch.ops.aten.mm.default(view_353, permute_300);  permute_300 = None
    permute_301: "f32[512, 25096]" = torch.ops.aten.permute.default(view_353, [1, 0])
    mm_61: "f32[512, 64]" = torch.ops.aten.mm.default(permute_301, view_17);  permute_301 = view_17 = None
    permute_302: "f32[64, 512]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_121: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_353, [0], True);  view_353 = None
    view_354: "f32[512]" = torch.ops.aten.view.default(sum_121, [512]);  sum_121 = None
    permute_303: "f32[512, 64]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
    view_355: "f32[8, 3137, 64]" = torch.ops.aten.view.default(mm_60, [8, 3137, 64]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_300: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_355, primals_4);  primals_4 = None
    mul_301: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_300, 64)
    sum_122: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True)
    mul_302: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_300, mul_6);  mul_300 = None
    sum_123: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True);  mul_302 = None
    mul_303: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_6, sum_123);  sum_123 = None
    sub_91: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(mul_301, sum_122);  mul_301 = sum_122 = None
    sub_92: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(sub_91, mul_303);  sub_91 = mul_303 = None
    mul_304: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(div_26, sub_92);  div_26 = sub_92 = None
    mul_305: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_355, mul_6);  mul_6 = None
    sum_124: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 1]);  mul_305 = None
    sum_125: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_355, [0, 1]);  view_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_164: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_161, mul_304);  add_161 = mul_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:135, code: x = self.proj(x)
    view_356: "f32[25096, 64]" = torch.ops.aten.view.default(add_164, [25096, 64])
    mm_62: "f32[25096, 64]" = torch.ops.aten.mm.default(view_356, permute_304);  permute_304 = None
    permute_305: "f32[64, 25096]" = torch.ops.aten.permute.default(view_356, [1, 0])
    mm_63: "f32[64, 64]" = torch.ops.aten.mm.default(permute_305, view_15);  permute_305 = view_15 = None
    permute_306: "f32[64, 64]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_126: "f32[1, 64]" = torch.ops.aten.sum.dim_IntList(view_356, [0], True);  view_356 = None
    view_357: "f32[64]" = torch.ops.aten.view.default(sum_126, [64]);  sum_126 = None
    permute_307: "f32[64, 64]" = torch.ops.aten.permute.default(permute_306, [1, 0]);  permute_306 = None
    view_358: "f32[8, 3137, 64]" = torch.ops.aten.view.default(mm_62, [8, 3137, 64]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:132, code: x = x.transpose(1, 2).reshape(B, N, C)  # [B, h, N, Ch] -> [B, N, h, Ch] -> [B, N, C]
    view_359: "f32[8, 3137, 8, 8]" = torch.ops.aten.view.default(view_358, [8, 3137, 8, 8]);  view_358 = None
    permute_308: "f32[8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_359, [0, 2, 1, 3]);  view_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:131, code: x = self.scale * factor_att + crpe
    mul_306: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(permute_308, 0.3535533905932738)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:87, code: EV_hat = F.pad(EV_hat, (0, 0, 1, 0, 0, 0))  # [B, h, N, Ch].
    constant_pad_nd_15: "f32[8, 8, 3136, 8]" = torch.ops.aten.constant_pad_nd.default(permute_308, [0, 0, -1, 0, 0, 0]);  permute_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:86, code: EV_hat = q_img * conv_v_img
    mul_307: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(constant_pad_nd_15, slice_8);  slice_8 = None
    mul_308: "f32[8, 8, 3136, 8]" = torch.ops.aten.mul.Tensor(constant_pad_nd_15, permute_7);  constant_pad_nd_15 = permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:84, code: conv_v_img = conv_v_img.reshape(B, num_heads, C, H * W).transpose(-1, -2)
    permute_309: "f32[8, 8, 8, 3136]" = torch.ops.aten.permute.default(mul_307, [0, 1, 3, 2]);  mul_307 = None
    view_360: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_309, [8, 64, 56, 56]);  permute_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:83, code: conv_v_img = torch.cat(conv_v_img_list, dim=1)
    slice_151: "f32[8, 16, 56, 56]" = torch.ops.aten.slice.Tensor(view_360, 1, 0, 16)
    slice_152: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(view_360, 1, 16, 40)
    slice_153: "f32[8, 24, 56, 56]" = torch.ops.aten.slice.Tensor(view_360, 1, 40, 64);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_31 = torch.ops.aten.convolution_backward.default(slice_153, getitem_9, primals_51, [24], [1, 1], [3, 3], [1, 1], False, [0, 0], 24, [True, True, True]);  slice_153 = getitem_9 = primals_51 = None
    getitem_183: "f32[8, 24, 56, 56]" = convolution_backward_31[0]
    getitem_184: "f32[24, 1, 7, 7]" = convolution_backward_31[1]
    getitem_185: "f32[24]" = convolution_backward_31[2];  convolution_backward_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_165: "f32[24, 1, 7, 7]" = torch.ops.aten.add.Tensor(getitem_172, getitem_184);  getitem_172 = getitem_184 = None
    add_166: "f32[24]" = torch.ops.aten.add.Tensor(getitem_173, getitem_185);  getitem_173 = getitem_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_32 = torch.ops.aten.convolution_backward.default(slice_152, getitem_8, primals_49, [24], [1, 1], [2, 2], [1, 1], False, [0, 0], 24, [True, True, True]);  slice_152 = getitem_8 = primals_49 = None
    getitem_186: "f32[8, 24, 56, 56]" = convolution_backward_32[0]
    getitem_187: "f32[24, 1, 5, 5]" = convolution_backward_32[1]
    getitem_188: "f32[24]" = convolution_backward_32[2];  convolution_backward_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_167: "f32[24, 1, 5, 5]" = torch.ops.aten.add.Tensor(getitem_175, getitem_187);  getitem_175 = getitem_187 = None
    add_168: "f32[24]" = torch.ops.aten.add.Tensor(getitem_176, getitem_188);  getitem_176 = getitem_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    convolution_backward_33 = torch.ops.aten.convolution_backward.default(slice_151, getitem_7, primals_47, [16], [1, 1], [1, 1], [1, 1], False, [0, 0], 16, [True, True, True]);  slice_151 = getitem_7 = primals_47 = None
    getitem_189: "f32[8, 16, 56, 56]" = convolution_backward_33[0]
    getitem_190: "f32[16, 1, 3, 3]" = convolution_backward_33[1]
    getitem_191: "f32[16]" = convolution_backward_33[2];  convolution_backward_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:82, code: conv_v_img_list.append(conv(v_img_list[i]))
    add_169: "f32[16, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_178, getitem_190);  getitem_178 = getitem_190 = None
    add_170: "f32[16]" = torch.ops.aten.add.Tensor(getitem_179, getitem_191);  getitem_179 = getitem_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:79, code: v_img_list = torch.split(v_img, self.channel_splits, dim=1)  # Split according to channels
    cat_34: "f32[8, 64, 56, 56]" = torch.ops.aten.cat.default([getitem_189, getitem_186, getitem_183], 1);  getitem_189 = getitem_186 = getitem_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:78, code: v_img = v_img.transpose(-1, -2).reshape(B, num_heads * C, H, W)
    view_361: "f32[8, 8, 8, 3136]" = torch.ops.aten.view.default(cat_34, [8, 8, 8, 3136]);  cat_34 = None
    permute_310: "f32[8, 8, 3136, 8]" = torch.ops.aten.permute.default(view_361, [0, 1, 3, 2]);  view_361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:76, code: v_img = v[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_scatter_94: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice_scatter.default(full_default_83, permute_310, 3, 0, 9223372036854775807);  permute_310 = None
    slice_scatter_95: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_default_84, slice_scatter_94, 2, 1, 9223372036854775807);  slice_scatter_94 = None
    slice_scatter_96: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_default_84, slice_scatter_95, 1, 0, 9223372036854775807);  slice_scatter_95 = None
    slice_scatter_97: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_default_84, slice_scatter_96, 0, 0, 9223372036854775807);  slice_scatter_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:75, code: q_img = q[:, :, 1:, :]  # [B, h, H*W, Ch]
    slice_scatter_98: "f32[8, 8, 3136, 8]" = torch.ops.aten.slice_scatter.default(full_default_83, mul_308, 3, 0, 9223372036854775807);  full_default_83 = mul_308 = None
    slice_scatter_99: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_default_84, slice_scatter_98, 2, 1, 9223372036854775807);  slice_scatter_98 = None
    slice_scatter_100: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_default_84, slice_scatter_99, 1, 0, 9223372036854775807);  slice_scatter_99 = None
    slice_scatter_101: "f32[8, 8, 3137, 8]" = torch.ops.aten.slice_scatter.default(full_default_84, slice_scatter_100, 0, 0, 9223372036854775807);  full_default_84 = slice_scatter_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    clone_92: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(mul_306, memory_format = torch.contiguous_format);  mul_306 = None
    view_362: "f32[64, 3137, 8]" = torch.ops.aten.view.default(clone_92, [64, 3137, 8]);  clone_92 = None
    bmm_44: "f32[64, 8, 8]" = torch.ops.aten.bmm.default(permute_311, view_362);  permute_311 = None
    bmm_45: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(view_362, permute_312);  view_362 = permute_312 = None
    view_363: "f32[8, 8, 8, 8]" = torch.ops.aten.view.default(bmm_44, [8, 8, 8, 8]);  bmm_44 = None
    view_364: "f32[8, 8, 3137, 8]" = torch.ops.aten.view.default(bmm_45, [8, 8, 3137, 8]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:125, code: factor_att = q @ factor_att
    add_171: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(slice_scatter_101, view_364);  slice_scatter_101 = view_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_365: "f32[64, 8, 8]" = torch.ops.aten.view.default(view_363, [64, 8, 8]);  view_363 = None
    bmm_46: "f32[64, 3137, 8]" = torch.ops.aten.bmm.default(permute_313, view_365);  permute_313 = None
    bmm_47: "f32[64, 8, 3137]" = torch.ops.aten.bmm.default(view_365, permute_314);  view_365 = permute_314 = None
    view_366: "f32[8, 8, 3137, 8]" = torch.ops.aten.view.default(bmm_46, [8, 8, 3137, 8]);  bmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    add_172: "f32[8, 8, 3137, 8]" = torch.ops.aten.add.Tensor(slice_scatter_97, view_366);  slice_scatter_97 = view_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:124, code: factor_att = k_softmax.transpose(-1, -2) @ v
    view_367: "f32[8, 8, 8, 3137]" = torch.ops.aten.view.default(bmm_47, [8, 8, 8, 3137]);  bmm_47 = None
    permute_315: "f32[8, 8, 3137, 8]" = torch.ops.aten.permute.default(view_367, [0, 1, 3, 2]);  view_367 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:123, code: k_softmax = k.softmax(dim=2)
    mul_309: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(permute_315, alias_15);  permute_315 = None
    sum_127: "f32[8, 8, 1, 8]" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True)
    mul_310: "f32[8, 8, 3137, 8]" = torch.ops.aten.mul.Tensor(alias_15, sum_127);  alias_15 = sum_127 = None
    sub_93: "f32[8, 8, 3137, 8]" = torch.ops.aten.sub.Tensor(mul_309, mul_310);  mul_309 = mul_310 = None
    clone_93: "f32[8, 8, 3137, 8]" = torch.ops.aten.clone.default(sub_93, memory_format = torch.contiguous_format);  sub_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:120, code: q, k, v = qkv.unbind(0)  # [B, h, N, Ch]
    cat_35: "f32[24, 8, 3137, 8]" = torch.ops.aten.cat.default([add_171, clone_93, add_172]);  add_171 = clone_93 = add_172 = None
    view_368: "f32[3, 8, 8, 3137, 8]" = torch.ops.aten.view.default(cat_35, [3, 8, 8, 3137, 8]);  cat_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:119, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    permute_316: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.permute.default(view_368, [1, 3, 0, 2, 4]);  view_368 = None
    clone_94: "f32[8, 3137, 3, 8, 8]" = torch.ops.aten.clone.default(permute_316, memory_format = torch.contiguous_format);  permute_316 = None
    view_369: "f32[8, 3137, 192]" = torch.ops.aten.view.default(clone_94, [8, 3137, 192]);  clone_94 = None
    view_370: "f32[25096, 192]" = torch.ops.aten.view.default(view_369, [25096, 192]);  view_369 = None
    mm_64: "f32[25096, 64]" = torch.ops.aten.mm.default(view_370, permute_317);  permute_317 = None
    permute_318: "f32[192, 25096]" = torch.ops.aten.permute.default(view_370, [1, 0])
    mm_65: "f32[192, 64]" = torch.ops.aten.mm.default(permute_318, view_3);  permute_318 = view_3 = None
    permute_319: "f32[64, 192]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_128: "f32[1, 192]" = torch.ops.aten.sum.dim_IntList(view_370, [0], True);  view_370 = None
    view_371: "f32[192]" = torch.ops.aten.view.default(sum_128, [192]);  sum_128 = None
    permute_320: "f32[192, 64]" = torch.ops.aten.permute.default(permute_319, [1, 0]);  permute_319 = None
    view_372: "f32[8, 3137, 64]" = torch.ops.aten.view.default(mm_64, [8, 3137, 64]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    mul_312: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_372, primals_2);  primals_2 = None
    mul_313: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_312, 64)
    sum_129: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_312, [2], True)
    mul_314: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_312, mul_2);  mul_312 = None
    sum_130: "f32[8, 3137, 1]" = torch.ops.aten.sum.dim_IntList(mul_314, [2], True);  mul_314 = None
    mul_315: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(mul_2, sum_130);  sum_130 = None
    sub_95: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(mul_313, sum_129);  mul_313 = sum_129 = None
    sub_96: "f32[8, 3137, 64]" = torch.ops.aten.sub.Tensor(sub_95, mul_315);  sub_95 = mul_315 = None
    div_27: "f32[8, 3137, 1]" = torch.ops.aten.div.Tensor(rsqrt_1, 64);  rsqrt_1 = None
    mul_316: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(div_27, sub_96);  div_27 = sub_96 = None
    mul_317: "f32[8, 3137, 64]" = torch.ops.aten.mul.Tensor(view_372, mul_2);  mul_2 = None
    sum_131: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_317, [0, 1]);  mul_317 = None
    sum_132: "f32[64]" = torch.ops.aten.sum.dim_IntList(view_372, [0, 1]);  view_372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    add_173: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(add_164, mul_316);  add_164 = mul_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:163, code: x = torch.cat((cls_token, x), dim=1)
    slice_154: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(add_173, 1, 0, 1)
    slice_155: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(add_173, 1, 1, 3137);  add_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:160, code: x = x.flatten(2).transpose(1, 2)
    permute_321: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(slice_155, [0, 2, 1]);  slice_155 = None
    view_373: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_321, [8, 64, 56, 56]);  permute_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    convolution_backward_34 = torch.ops.aten.convolution_backward.default(view_373, view_1, primals_43, [64], [1, 1], [1, 1], [1, 1], False, [0, 0], 64, [True, True, True]);  view_1 = primals_43 = None
    getitem_192: "f32[8, 64, 56, 56]" = convolution_backward_34[0]
    getitem_193: "f32[64, 1, 3, 3]" = convolution_backward_34[1]
    getitem_194: "f32[64]" = convolution_backward_34[2];  convolution_backward_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:159, code: x = self.proj(feat) + feat
    add_174: "f32[8, 64, 56, 56]" = torch.ops.aten.add.Tensor(view_373, getitem_192);  view_373 = getitem_192 = None
    add_175: "f32[64, 1, 3, 3]" = torch.ops.aten.add.Tensor(getitem_181, getitem_193);  getitem_181 = getitem_193 = None
    add_176: "f32[64]" = torch.ops.aten.add.Tensor(getitem_182, getitem_194);  getitem_182 = getitem_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:158, code: feat = img_tokens.transpose(1, 2).view(B, C, H, W)
    view_374: "f32[8, 64, 3136]" = torch.ops.aten.view.default(add_174, [8, 64, 3136]);  add_174 = None
    permute_322: "f32[8, 3136, 64]" = torch.ops.aten.permute.default(view_374, [0, 2, 1]);  view_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    slice_scatter_102: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_default_81, permute_322, 1, 1, 9223372036854775807);  permute_322 = None
    slice_scatter_103: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_default_81, slice_scatter_102, 0, 0, 9223372036854775807);  slice_scatter_102 = None
    slice_scatter_104: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_default_81, slice_154, 1, 0, 1);  slice_154 = None
    slice_scatter_105: "f32[8, 3137, 64]" = torch.ops.aten.slice_scatter.default(full_default_81, slice_scatter_104, 0, 0, 9223372036854775807);  full_default_81 = slice_scatter_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:155, code: cls_token, img_tokens = x[:, :1], x[:, 1:]  # [B, 1, C], [B, H*W, C]
    add_177: "f32[8, 3137, 64]" = torch.ops.aten.add.Tensor(slice_scatter_103, slice_scatter_105);  slice_scatter_103 = slice_scatter_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:678, code: x = torch.cat((cls_tokens, x), dim=1)
    slice_156: "f32[8, 1, 64]" = torch.ops.aten.slice.Tensor(add_177, 1, 0, 1)
    slice_157: "f32[8, 3136, 64]" = torch.ops.aten.slice.Tensor(add_177, 1, 1, 3137);  add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/coat.py:677, code: cls_tokens = cls_token.expand(x.shape[0], -1, -1)
    sum_133: "f32[1, 1, 64]" = torch.ops.aten.sum.dim_IntList(slice_156, [0], True);  slice_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    clone_95: "f32[8, 3136, 64]" = torch.ops.aten.clone.default(slice_157, memory_format = torch.contiguous_format);  slice_157 = None
    mul_319: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_95, primals_41);  primals_41 = None
    mul_320: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_319, 64)
    sum_134: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True)
    mul_321: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul_319, mul);  mul_319 = None
    sum_135: "f32[8, 3136, 1]" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True);  mul_321 = None
    mul_322: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(mul, sum_135);  sum_135 = None
    sub_98: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(mul_320, sum_134);  mul_320 = sum_134 = None
    sub_99: "f32[8, 3136, 64]" = torch.ops.aten.sub.Tensor(sub_98, mul_322);  sub_98 = mul_322 = None
    mul_323: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(div_28, sub_99);  div_28 = sub_99 = None
    mul_324: "f32[8, 3136, 64]" = torch.ops.aten.mul.Tensor(clone_95, mul);  mul = None
    sum_136: "f32[64]" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1]);  mul_324 = None
    sum_137: "f32[64]" = torch.ops.aten.sum.dim_IntList(clone_95, [0, 1]);  clone_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    permute_323: "f32[8, 64, 3136]" = torch.ops.aten.permute.default(mul_323, [0, 2, 1]);  mul_323 = None
    view_375: "f32[8, 64, 56, 56]" = torch.ops.aten.view.default(permute_323, [8, 64, 56, 56]);  permute_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    convolution_backward_35 = torch.ops.aten.convolution_backward.default(view_375, primals_153, primals_39, [64], [4, 4], [0, 0], [1, 1], False, [0, 0], 1, [False, True, True]);  view_375 = primals_153 = primals_39 = None
    getitem_196: "f32[64, 3, 4, 4]" = convolution_backward_35[1]
    getitem_197: "f32[64]" = convolution_backward_35[2];  convolution_backward_35 = None
    return [sum_133, sum_131, sum_132, sum_124, sum_125, sum_118, sum_119, sum_111, sum_112, sum_102, sum_100, sum_101, sum_93, sum_94, sum_87, sum_88, sum_80, sum_81, sum_71, sum_69, sum_70, sum_62, sum_63, sum_56, sum_57, sum_49, sum_50, sum_40, sum_38, sum_39, sum_31, sum_32, sum_25, sum_26, sum_18, sum_19, sum_12, sum_13, getitem_196, getitem_197, sum_136, sum_137, add_175, add_176, permute_320, view_371, add_169, add_170, add_167, add_168, add_165, add_166, permute_307, view_357, permute_303, view_354, permute_299, view_351, permute_293, view_346, permute_280, view_332, permute_276, view_329, permute_272, view_326, getitem_169, getitem_170, sum_105, sum_106, add_151, add_152, permute_264, view_319, add_145, add_146, add_143, add_144, add_141, add_142, permute_251, view_305, permute_247, view_302, permute_243, view_299, permute_237, view_294, permute_224, view_280, permute_220, view_277, permute_216, view_274, getitem_142, getitem_143, sum_74, sum_75, add_127, add_128, permute_208, view_267, add_121, add_122, add_119, add_120, add_117, add_118, permute_195, view_253, permute_191, view_250, permute_187, view_247, permute_181, view_242, permute_168, view_228, permute_164, view_225, permute_160, view_222, getitem_115, getitem_116, sum_43, sum_44, add_103, add_104, permute_152, view_215, add_97, add_98, add_95, add_96, add_93, add_94, permute_139, view_201, permute_135, view_198, permute_131, view_195, permute_125, view_190, permute_112, view_176, permute_108, view_173, permute_104, view_170, permute_100, view_168, None]
    