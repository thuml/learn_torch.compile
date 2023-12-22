from __future__ import annotations



def forward(self, primals_9: "f32[768]", primals_15: "f32[768]", primals_25: "f32[768]", primals_31: "f32[768]", primals_41: "f32[768]", primals_47: "f32[768]", primals_57: "f32[768]", primals_63: "f32[768]", primals_73: "f32[768]", primals_79: "f32[768]", primals_89: "f32[768]", primals_95: "f32[768]", primals_105: "f32[768]", primals_111: "f32[768]", primals_121: "f32[768]", primals_127: "f32[768]", primals_137: "f32[768]", primals_143: "f32[768]", primals_153: "f32[768]", primals_159: "f32[768]", primals_169: "f32[768]", primals_175: "f32[768]", primals_185: "f32[768]", primals_191: "f32[768]", view: "f32[1024, 768]", slice_64: "f32[1, 256, 1, 257]", rev_1: "f32[1, 256, 1, 257]", unsqueeze_16: "b8[1, 1024, 1, 1]", getitem_1: "b8[1, 1024, 12, 513]", view_69: "f32[1024, 768]", getitem_3: "b8[1, 1024, 768]", mul_1: "f32[1, 1024, 768]", view_71: "f32[1024, 768]", addmm_4: "f32[1024, 3072]", view_73: "f32[1024, 3072]", getitem_7: "b8[1, 1024, 768]", mul_6: "f32[1, 1024, 768]", view_75: "f32[1024, 768]", getitem_11: "b8[1, 1024, 12, 513]", view_144: "f32[1024, 768]", getitem_13: "b8[1, 1024, 768]", mul_9: "f32[1, 1024, 768]", view_146: "f32[1024, 768]", addmm_10: "f32[1024, 3072]", view_148: "f32[1024, 3072]", getitem_17: "b8[1, 1024, 768]", mul_14: "f32[1, 1024, 768]", view_150: "f32[1024, 768]", getitem_21: "b8[1, 1024, 12, 513]", view_219: "f32[1024, 768]", getitem_23: "b8[1, 1024, 768]", mul_17: "f32[1, 1024, 768]", view_221: "f32[1024, 768]", addmm_16: "f32[1024, 3072]", view_223: "f32[1024, 3072]", getitem_27: "b8[1, 1024, 768]", mul_22: "f32[1, 1024, 768]", view_225: "f32[1024, 768]", getitem_31: "b8[1, 1024, 12, 513]", view_294: "f32[1024, 768]", getitem_33: "b8[1, 1024, 768]", mul_25: "f32[1, 1024, 768]", view_296: "f32[1024, 768]", addmm_22: "f32[1024, 3072]", view_298: "f32[1024, 3072]", getitem_37: "b8[1, 1024, 768]", mul_30: "f32[1, 1024, 768]", view_300: "f32[1024, 768]", getitem_41: "b8[1, 1024, 12, 513]", view_369: "f32[1024, 768]", getitem_43: "b8[1, 1024, 768]", mul_33: "f32[1, 1024, 768]", view_371: "f32[1024, 768]", addmm_28: "f32[1024, 3072]", view_373: "f32[1024, 3072]", getitem_47: "b8[1, 1024, 768]", mul_38: "f32[1, 1024, 768]", view_375: "f32[1024, 768]", getitem_51: "b8[1, 1024, 12, 513]", view_444: "f32[1024, 768]", getitem_53: "b8[1, 1024, 768]", mul_41: "f32[1, 1024, 768]", view_446: "f32[1024, 768]", addmm_34: "f32[1024, 3072]", view_448: "f32[1024, 3072]", getitem_57: "b8[1, 1024, 768]", mul_46: "f32[1, 1024, 768]", view_450: "f32[1024, 768]", getitem_61: "b8[1, 1024, 12, 513]", view_519: "f32[1024, 768]", getitem_63: "b8[1, 1024, 768]", mul_49: "f32[1, 1024, 768]", view_521: "f32[1024, 768]", addmm_40: "f32[1024, 3072]", view_523: "f32[1024, 3072]", getitem_67: "b8[1, 1024, 768]", mul_54: "f32[1, 1024, 768]", view_525: "f32[1024, 768]", getitem_71: "b8[1, 1024, 12, 513]", view_594: "f32[1024, 768]", getitem_73: "b8[1, 1024, 768]", mul_57: "f32[1, 1024, 768]", view_596: "f32[1024, 768]", addmm_46: "f32[1024, 3072]", view_598: "f32[1024, 3072]", getitem_77: "b8[1, 1024, 768]", mul_62: "f32[1, 1024, 768]", view_600: "f32[1024, 768]", getitem_81: "b8[1, 1024, 12, 513]", view_669: "f32[1024, 768]", getitem_83: "b8[1, 1024, 768]", mul_65: "f32[1, 1024, 768]", view_671: "f32[1024, 768]", addmm_52: "f32[1024, 3072]", view_673: "f32[1024, 3072]", getitem_87: "b8[1, 1024, 768]", mul_70: "f32[1, 1024, 768]", view_675: "f32[1024, 768]", getitem_91: "b8[1, 1024, 12, 513]", view_744: "f32[1024, 768]", getitem_93: "b8[1, 1024, 768]", mul_73: "f32[1, 1024, 768]", view_746: "f32[1024, 768]", addmm_58: "f32[1024, 3072]", view_748: "f32[1024, 3072]", getitem_97: "b8[1, 1024, 768]", mul_78: "f32[1, 1024, 768]", view_750: "f32[1024, 768]", getitem_101: "b8[1, 1024, 12, 513]", view_819: "f32[1024, 768]", getitem_103: "b8[1, 1024, 768]", mul_81: "f32[1, 1024, 768]", view_821: "f32[1024, 768]", addmm_64: "f32[1024, 3072]", view_823: "f32[1024, 3072]", getitem_107: "b8[1, 1024, 768]", mul_86: "f32[1, 1024, 768]", view_825: "f32[1024, 768]", getitem_111: "b8[1, 1024, 12, 513]", view_894: "f32[1024, 768]", getitem_113: "b8[1, 1024, 768]", mul_89: "f32[1, 1024, 768]", view_896: "f32[1024, 768]", addmm_70: "f32[1024, 3072]", view_898: "f32[1024, 3072]", getitem_117: "b8[1, 1024, 768]", mul_94: "f32[1, 1024, 768]", div_120: "f32[1, 1024, 1]", permute_756: "f32[768, 3072]", permute_760: "f32[3072, 768]", div_121: "f32[1, 1024, 1]", permute_764: "f32[768, 768]", permute_772: "f32[48, 768, 256]", permute_773: "f32[48, 64, 768]", alias_12: "f32[1, 1024, 12, 513]", permute_783: "f32[36, 64, 512]", permute_784: "f32[36, 512, 64]", permute_795: "f32[768, 768]", permute_799: "f32[768, 768]", permute_808: "f32[768, 768]", div_123: "f32[1, 1024, 1]", permute_814: "f32[768, 3072]", permute_818: "f32[3072, 768]", div_124: "f32[1, 1024, 1]", permute_822: "f32[768, 768]", permute_830: "f32[48, 768, 256]", permute_831: "f32[48, 64, 768]", alias_13: "f32[1, 1024, 12, 513]", permute_841: "f32[36, 64, 512]", permute_842: "f32[36, 512, 64]", permute_853: "f32[768, 768]", permute_857: "f32[768, 768]", permute_866: "f32[768, 768]", div_126: "f32[1, 1024, 1]", permute_872: "f32[768, 3072]", permute_876: "f32[3072, 768]", div_127: "f32[1, 1024, 1]", permute_880: "f32[768, 768]", permute_888: "f32[48, 768, 256]", permute_889: "f32[48, 64, 768]", alias_14: "f32[1, 1024, 12, 513]", permute_899: "f32[36, 64, 512]", permute_900: "f32[36, 512, 64]", permute_911: "f32[768, 768]", permute_915: "f32[768, 768]", permute_924: "f32[768, 768]", div_129: "f32[1, 1024, 1]", permute_930: "f32[768, 3072]", permute_934: "f32[3072, 768]", div_130: "f32[1, 1024, 1]", permute_938: "f32[768, 768]", permute_946: "f32[48, 768, 256]", permute_947: "f32[48, 64, 768]", alias_15: "f32[1, 1024, 12, 513]", permute_957: "f32[36, 64, 512]", permute_958: "f32[36, 512, 64]", permute_969: "f32[768, 768]", permute_973: "f32[768, 768]", permute_982: "f32[768, 768]", div_132: "f32[1, 1024, 1]", permute_988: "f32[768, 3072]", permute_992: "f32[3072, 768]", div_133: "f32[1, 1024, 1]", permute_996: "f32[768, 768]", permute_1004: "f32[48, 768, 256]", permute_1005: "f32[48, 64, 768]", alias_16: "f32[1, 1024, 12, 513]", permute_1015: "f32[36, 64, 512]", permute_1016: "f32[36, 512, 64]", permute_1027: "f32[768, 768]", permute_1031: "f32[768, 768]", permute_1040: "f32[768, 768]", div_135: "f32[1, 1024, 1]", permute_1046: "f32[768, 3072]", permute_1050: "f32[3072, 768]", div_136: "f32[1, 1024, 1]", permute_1054: "f32[768, 768]", permute_1062: "f32[48, 768, 256]", permute_1063: "f32[48, 64, 768]", alias_17: "f32[1, 1024, 12, 513]", permute_1073: "f32[36, 64, 512]", permute_1074: "f32[36, 512, 64]", permute_1085: "f32[768, 768]", permute_1089: "f32[768, 768]", permute_1098: "f32[768, 768]", div_138: "f32[1, 1024, 1]", permute_1104: "f32[768, 3072]", permute_1108: "f32[3072, 768]", div_139: "f32[1, 1024, 1]", permute_1112: "f32[768, 768]", permute_1120: "f32[48, 768, 256]", permute_1121: "f32[48, 64, 768]", alias_18: "f32[1, 1024, 12, 513]", permute_1131: "f32[36, 64, 512]", permute_1132: "f32[36, 512, 64]", permute_1143: "f32[768, 768]", permute_1147: "f32[768, 768]", permute_1156: "f32[768, 768]", div_141: "f32[1, 1024, 1]", permute_1162: "f32[768, 3072]", permute_1166: "f32[3072, 768]", div_142: "f32[1, 1024, 1]", permute_1170: "f32[768, 768]", permute_1178: "f32[48, 768, 256]", permute_1179: "f32[48, 64, 768]", alias_19: "f32[1, 1024, 12, 513]", permute_1189: "f32[36, 64, 512]", permute_1190: "f32[36, 512, 64]", permute_1201: "f32[768, 768]", permute_1205: "f32[768, 768]", permute_1214: "f32[768, 768]", div_144: "f32[1, 1024, 1]", permute_1220: "f32[768, 3072]", permute_1224: "f32[3072, 768]", div_145: "f32[1, 1024, 1]", permute_1228: "f32[768, 768]", permute_1236: "f32[48, 768, 256]", permute_1237: "f32[48, 64, 768]", alias_20: "f32[1, 1024, 12, 513]", permute_1247: "f32[36, 64, 512]", permute_1248: "f32[36, 512, 64]", permute_1259: "f32[768, 768]", permute_1263: "f32[768, 768]", permute_1272: "f32[768, 768]", div_147: "f32[1, 1024, 1]", permute_1278: "f32[768, 3072]", permute_1282: "f32[3072, 768]", div_148: "f32[1, 1024, 1]", permute_1286: "f32[768, 768]", permute_1294: "f32[48, 768, 256]", permute_1295: "f32[48, 64, 768]", alias_21: "f32[1, 1024, 12, 513]", permute_1305: "f32[36, 64, 512]", permute_1306: "f32[36, 512, 64]", permute_1317: "f32[768, 768]", permute_1321: "f32[768, 768]", permute_1330: "f32[768, 768]", div_150: "f32[1, 1024, 1]", permute_1336: "f32[768, 3072]", permute_1340: "f32[3072, 768]", div_151: "f32[1, 1024, 1]", permute_1344: "f32[768, 768]", permute_1352: "f32[48, 768, 256]", permute_1353: "f32[48, 64, 768]", alias_22: "f32[1, 1024, 12, 513]", permute_1363: "f32[36, 64, 512]", permute_1364: "f32[36, 512, 64]", permute_1375: "f32[768, 768]", permute_1379: "f32[768, 768]", permute_1388: "f32[768, 768]", div_153: "f32[1, 1024, 1]", permute_1394: "f32[768, 3072]", permute_1398: "f32[3072, 768]", div_154: "f32[1, 1024, 1]", permute_1402: "f32[768, 768]", permute_1410: "f32[48, 768, 256]", permute_1411: "f32[48, 64, 768]", alias_23: "f32[1, 1024, 12, 513]", permute_1421: "f32[36, 64, 512]", permute_1422: "f32[36, 512, 64]", permute_1433: "f32[768, 768]", permute_1437: "f32[768, 768]", permute_1446: "f32[768, 768]", tangents_1: "f32[1, 1024, 768]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:804, code: beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
    full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:808, code: beginning_mask = beginning_mask.expand(beginning_input.size())
    expand: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(slice_64, [1, 256, 12, 257]);  slice_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    convert_element_type: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand, torch.bool);  expand = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:813, code: ending_mask = ending_mask.expand(ending_input.size())
    expand_1: "f32[1, 256, 12, 257]" = torch.ops.aten.expand.default(rev_1, [1, 256, 12, 257]);  rev_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    convert_element_type_1: "b8[1, 256, 12, 257]" = torch.ops.prims.convert_element_type.default(expand_1, torch.bool);  expand_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_72: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_4, [1, 1024, 3072]);  addmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_4: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_72, 0.7071067811865476)
    erf: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_4);  mul_4 = None
    add_7: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_147: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_10, [1, 1024, 3072]);  addmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_12: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_147, 0.7071067811865476)
    erf_1: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
    add_18: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_222: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_16, [1, 1024, 3072]);  addmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_20: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_222, 0.7071067811865476)
    erf_2: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_20);  mul_20 = None
    add_29: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_297: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_22, [1, 1024, 3072]);  addmm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_28: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_297, 0.7071067811865476)
    erf_3: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_40: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_372: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_28, [1, 1024, 3072]);  addmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_36: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_372, 0.7071067811865476)
    erf_4: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_36);  mul_36 = None
    add_51: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_447: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_34, [1, 1024, 3072]);  addmm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_44: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_447, 0.7071067811865476)
    erf_5: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_44);  mul_44 = None
    add_62: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_522: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_40, [1, 1024, 3072]);  addmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_52: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_522, 0.7071067811865476)
    erf_6: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_52);  mul_52 = None
    add_73: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_597: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_46, [1, 1024, 3072]);  addmm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_60: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_597, 0.7071067811865476)
    erf_7: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_84: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_672: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_52, [1, 1024, 3072]);  addmm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_68: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_672, 0.7071067811865476)
    erf_8: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_68);  mul_68 = None
    add_95: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_747: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_58, [1, 1024, 3072]);  addmm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_76: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_747, 0.7071067811865476)
    erf_9: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_76);  mul_76 = None
    add_106: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_822: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_64, [1, 1024, 3072]);  addmm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_84: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_822, 0.7071067811865476)
    erf_10: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_84);  mul_84 = None
    add_117: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_897: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(addmm_70, [1, 1024, 3072]);  addmm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_92: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_897, 0.7071067811865476)
    erf_11: "f32[1, 1024, 3072]" = torch.ops.aten.erf.default(mul_92);  mul_92 = None
    add_128: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1348, code: hidden_states = hidden_states[:, : hidden_states.shape[1] - padding_len]
    full_default_120: "f32[1, 1024, 768]" = torch.ops.aten.full.default([1, 1024, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_528: "f32[1, 1024, 768]" = torch.ops.aten.slice_scatter.default(full_default_120, tangents_1, 0, 0, 9223372036854775807);  full_default_120 = tangents_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_97: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(slice_scatter_528, primals_191);  primals_191 = None
    mul_98: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_97, 768)
    sum_13: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_97, [2], True)
    mul_99: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_97, mul_94);  mul_97 = None
    sum_14: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_99, [2], True);  mul_99 = None
    mul_100: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_94, sum_14);  sum_14 = None
    sub_97: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_98, sum_13);  mul_98 = sum_13 = None
    sub_98: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_97, mul_100);  sub_97 = mul_100 = None
    mul_101: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_120, sub_98);  div_120 = sub_98 = None
    mul_102: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(slice_scatter_528, mul_94);  mul_94 = None
    sum_15: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_102, [0, 1]);  mul_102 = None
    sum_16: "f32[768]" = torch.ops.aten.sum.dim_IntList(slice_scatter_528, [0, 1]);  slice_scatter_528 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_60: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_117, torch.float32);  getitem_117 = None
    mul_103: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
    mul_104: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_101, mul_103);  mul_103 = None
    clone_60: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_104, memory_format = torch.contiguous_format);  mul_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_900: "f32[1024, 768]" = torch.ops.aten.view.default(clone_60, [1024, 768]);  clone_60 = None
    mm: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_900, permute_756);  permute_756 = None
    permute_757: "f32[768, 1024]" = torch.ops.aten.permute.default(view_900, [1, 0])
    mm_1: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_757, view_898);  permute_757 = view_898 = None
    permute_758: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_17: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_900, [0], True);  view_900 = None
    view_901: "f32[768]" = torch.ops.aten.view.default(sum_17, [768]);  sum_17 = None
    permute_759: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_758, [1, 0]);  permute_758 = None
    view_902: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm, [1, 1024, 3072]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_106: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_128, 0.5);  add_128 = None
    mul_107: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_897, view_897)
    mul_108: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_107, -0.5);  mul_107 = None
    exp_12: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_108);  mul_108 = None
    mul_109: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_110: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_897, mul_109);  view_897 = mul_109 = None
    add_133: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_106, mul_110);  mul_106 = mul_110 = None
    mul_111: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_902, add_133);  view_902 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_903: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_111, [1024, 3072]);  mul_111 = None
    mm_2: "f32[1024, 768]" = torch.ops.aten.mm.default(view_903, permute_760);  permute_760 = None
    permute_761: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_903, [1, 0])
    mm_3: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_761, view_896);  permute_761 = view_896 = None
    permute_762: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_18: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_903, [0], True);  view_903 = None
    view_904: "f32[3072]" = torch.ops.aten.view.default(sum_18, [3072]);  sum_18 = None
    permute_763: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_762, [1, 0]);  permute_762 = None
    view_905: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_2, [1, 1024, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    add_134: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_101, view_905);  mul_101 = view_905 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_113: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_134, primals_185);  primals_185 = None
    mul_114: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_113, 768)
    sum_19: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_113, [2], True)
    mul_115: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_113, mul_89);  mul_113 = None
    sum_20: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_115, [2], True);  mul_115 = None
    mul_116: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_89, sum_20);  sum_20 = None
    sub_100: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_114, sum_19);  mul_114 = sum_19 = None
    sub_101: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_100, mul_116);  sub_100 = mul_116 = None
    mul_117: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_121, sub_101);  div_121 = sub_101 = None
    mul_118: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_134, mul_89);  mul_89 = None
    sum_21: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_118, [0, 1]);  mul_118 = None
    sum_22: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_134, [0, 1]);  add_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_61: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_113, torch.float32);  getitem_113 = None
    mul_119: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
    mul_120: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_117, mul_119);  mul_119 = None
    clone_61: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_120, memory_format = torch.contiguous_format);  mul_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_906: "f32[1024, 768]" = torch.ops.aten.view.default(clone_61, [1024, 768]);  clone_61 = None
    mm_4: "f32[1024, 768]" = torch.ops.aten.mm.default(view_906, permute_764);  permute_764 = None
    permute_765: "f32[768, 1024]" = torch.ops.aten.permute.default(view_906, [1, 0])
    mm_5: "f32[768, 768]" = torch.ops.aten.mm.default(permute_765, view_894);  permute_765 = view_894 = None
    permute_766: "f32[768, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_906, [0], True);  view_906 = None
    view_907: "f32[768]" = torch.ops.aten.view.default(sum_23, [768]);  sum_23 = None
    permute_767: "f32[768, 768]" = torch.ops.aten.permute.default(permute_766, [1, 0]);  permute_766 = None
    view_908: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_4, [1, 1024, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_768: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(view_908, [1, 0, 2]);  view_908 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    view_909: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(permute_768, [1024, 1, 12, 64]);  permute_768 = None
    permute_769: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_909, [1, 0, 2, 3]);  view_909 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    permute_770: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_769, [0, 2, 1, 3]);  permute_769 = None
    view_910: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_770, [12, 4, 256, 64]);  permute_770 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    view_911: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_910, [12, 4, 256, 64, 1]);  view_910 = None
    permute_771: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_911, [0, 1, 2, 4, 3]);  view_911 = None
    clone_62: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.clone.default(permute_771, memory_format = torch.contiguous_format);  permute_771 = None
    view_912: "f32[48, 256, 64]" = torch.ops.aten.view.default(clone_62, [48, 256, 64]);  clone_62 = None
    bmm_24: "f32[48, 768, 64]" = torch.ops.aten.bmm.default(permute_772, view_912);  permute_772 = None
    bmm_25: "f32[48, 256, 768]" = torch.ops.aten.bmm.default(view_912, permute_773);  view_912 = permute_773 = None
    view_913: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_24, [12, 4, 768, 64, 1]);  bmm_24 = None
    permute_774: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(view_913, [0, 1, 4, 3, 2]);  view_913 = None
    view_914: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_25, [12, 4, 256, 768, 1]);  bmm_25 = None
    permute_775: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(view_914, [0, 1, 2, 4, 3]);  view_914 = None
    permute_776: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_774, [0, 1, 4, 3, 2]);  permute_774 = None
    squeeze: "f32[12, 4, 768, 64]" = torch.ops.aten.squeeze.dim(permute_776, 4);  permute_776 = None
    permute_777: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_775, [0, 1, 2, 4, 3]);  permute_775 = None
    squeeze_1: "f32[12, 4, 256, 768]" = torch.ops.aten.squeeze.dim(permute_777, 4);  permute_777 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    full_default_121: "f32[12, 4, 256, 769]" = torch.ops.aten.full.default([12, 4, 256, 769], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_529: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, squeeze_1, 3, 0, -1);  squeeze_1 = None
    slice_scatter_530: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_529, 2, 0, 9223372036854775807);  slice_scatter_529 = None
    slice_scatter_531: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_530, 1, 0, 9223372036854775807);  slice_scatter_530 = None
    slice_scatter_532: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_531, 0, 0, 9223372036854775807);  slice_scatter_531 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_915: "f32[12, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_532, [12, 4, 196864]);  slice_scatter_532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    full_default_125: "f32[12, 4, 197120]" = torch.ops.aten.full.default([12, 4, 197120], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_533: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, view_915, 2, 0, -256);  view_915 = None
    slice_scatter_534: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_533, 1, 0, 9223372036854775807);  slice_scatter_533 = None
    slice_scatter_535: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_534, 0, 0, 9223372036854775807);  slice_scatter_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_916: "f32[12, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_535, [12, 4, 256, 770]);  slice_scatter_535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_48: "f32[12, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_916, [0, -257]);  view_916 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    full_default_128: "f32[1179648]" = torch.ops.aten.full.default([1179648], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_48: "i64[1179648]" = torch.ops.prims.iota.default(1179648, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    as_strided_72: "i64[12, 4, 768, 64]" = torch.ops.aten.as_strided.default(iota_48, [12, 4, 768, 64], [98304, 16384, 64, 1], 0);  iota_48 = None
    view_917: "f32[2359296]" = torch.ops.aten.view.default(squeeze, [-1]);  squeeze = None
    clone_63: "i64[12, 4, 768, 64]" = torch.ops.aten.clone.default(as_strided_72, memory_format = torch.contiguous_format);  as_strided_72 = None
    view_918: "i64[2359296]" = torch.ops.aten.view.default(clone_63, [2359296]);  clone_63 = None
    index_put: "f32[1179648]" = torch.ops.aten.index_put.default(full_default_128, [view_918], view_917, True);  view_917 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    as_strided_74: "f32[12, 1536, 64]" = torch.ops.aten.as_strided.default(index_put, [12, 1536, 64], [98304, 64, 1], 0);  index_put = None
    constant_pad_nd_49: "f32[12, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_74, [0, 0, -256, -256]);  as_strided_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    view_919: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_49, [1, 12, 1024, 64]);  constant_pad_nd_49 = None
    permute_778: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_919, [0, 2, 1, 3]);  view_919 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    view_920: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_48, [1, 12, 1024, 513]);  constant_pad_nd_48 = None
    permute_779: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_920, [0, 2, 1, 3]);  view_920 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    permute_780: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_778, [1, 0, 2, 3]);  permute_778 = None
    clone_64: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_780, memory_format = torch.contiguous_format);  permute_780 = None
    view_921: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_64, [1024, 1, 768]);  clone_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    convert_element_type_62: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(getitem_111, torch.float32);  getitem_111 = None
    mul_121: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
    mul_122: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_779, mul_121);  permute_779 = mul_121 = None
    clone_65: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_122, memory_format = torch.contiguous_format);  mul_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_96: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, clone_65);  clone_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    mul_123: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_96, alias_12);  where_96 = None
    sum_24: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_123, [-1], True)
    mul_124: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(alias_12, sum_24);  alias_12 = sum_24 = None
    sub_102: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(mul_123, mul_124);  mul_123 = mul_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    squeeze_2: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(sub_102, 0);  sub_102 = None
    full_117: "f32[6303744]" = torch.ops.aten.full.default([6303744], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    as_strided_75: "f32[1024, 12, 513]" = torch.ops.aten.as_strided.default(full_117, [1024, 12, 513], [513, 525312, 1], 0)
    copy_144: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_2);  squeeze_2 = None
    as_strided_scatter: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_144, [1024, 12, 513], [513, 525312, 1], 0);  copy_144 = None
    as_strided_78: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter = None
    new_empty_strided: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_78, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_145: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided, as_strided_78);  new_empty_strided = as_strided_78 = None
    as_strided_80: "f32[1, 1024, 12, 513]" = torch.ops.aten.as_strided.default(copy_145, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0)
    clone_66: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(as_strided_80, memory_format = torch.contiguous_format)
    copy_146: "f32[1, 1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_80, clone_66);  as_strided_80 = clone_66 = None
    as_strided_scatter_1: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_145, copy_146, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0);  copy_145 = copy_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    new_empty_strided_1: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_1, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_147: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_1, as_strided_scatter_1);  new_empty_strided_1 = as_strided_scatter_1 = None
    as_strided_83: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_147, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240)
    clone_67: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_83, memory_format = torch.contiguous_format)
    full_default_130: "f32[1, 256, 12, 257]" = torch.ops.aten.full.default([1, 256, 12, 257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    copy_148: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_83, full_default_130);  as_strided_83 = None
    as_strided_scatter_2: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_147, copy_148, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240);  copy_147 = copy_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_97: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_67);  clone_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    full_default_132: "f32[1, 256, 12, 513]" = torch.ops.aten.full.default([1, 256, 12, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_536: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_97, 3, -257, 9223372036854775807);  where_97 = None
    slice_scatter_537: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_536, 2, 0, 9223372036854775807);  slice_scatter_536 = None
    full_default_134: "f32[1, 1024, 12, 513]" = torch.ops.aten.full.default([1, 1024, 12, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_538: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_537, 1, -256, 9223372036854775807);  slice_scatter_537 = None
    slice_scatter_539: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_538, 0, 0, 9223372036854775807);  slice_scatter_538 = None
    squeeze_3: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(slice_scatter_539, 0);  slice_scatter_539 = None
    copy_149: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_3);  squeeze_3 = None
    as_strided_scatter_3: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_149, [1024, 12, 513], [513, 525312, 1], 0);  copy_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    as_strided_88: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_3, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_3 = None
    add_135: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_2, as_strided_88);  as_strided_scatter_2 = as_strided_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    new_empty_strided_2: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_135, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_150: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_2, add_135);  new_empty_strided_2 = add_135 = None
    as_strided_90: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_150, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0)
    clone_68: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_90, memory_format = torch.contiguous_format)
    copy_151: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_90, full_default_130);  as_strided_90 = None
    as_strided_scatter_4: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_150, copy_151, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0);  copy_150 = copy_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_98: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_68);  clone_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    slice_scatter_540: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_98, 3, 0, 257);  where_98 = None
    slice_scatter_541: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_540, 2, 0, 9223372036854775807);  slice_scatter_540 = None
    slice_scatter_542: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_541, 1, 0, 256);  slice_scatter_541 = None
    slice_scatter_543: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_542, 0, 0, 9223372036854775807);  slice_scatter_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    permute_781: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_543, [0, 2, 1, 3]);  slice_scatter_543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_922: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_781, [12, 4, 256, 513]);  permute_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    add_136: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_4, view_922);  as_strided_scatter_4 = view_922 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_3: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_136, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_152: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_3, add_136);  new_empty_strided_3 = add_136 = None
    as_strided_93: "f32[12, 255, 255]" = torch.ops.aten.as_strided.default(copy_152, [12, 255, 255], [525312, 513, 1], 514)
    clone_69: "f32[12, 255, 255]" = torch.ops.aten.clone.default(as_strided_93, memory_format = torch.contiguous_format)
    full_default_142: "f32[12, 255, 255]" = torch.ops.aten.full.default([12, 255, 255], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    copy_153: "f32[12, 255, 255]" = torch.ops.aten.copy.default(as_strided_93, full_default_142);  as_strided_93 = None
    as_strided_scatter_5: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_152, copy_153, [12, 255, 255], [525312, 513, 1], 514);  copy_152 = copy_153 = None
    full_default_143: "f32[12, 255, 513]" = torch.ops.aten.full.default([12, 255, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_544: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_143, clone_69, 2, -255, 9223372036854775807);  clone_69 = None
    full_default_144: "f32[12, 512, 513]" = torch.ops.aten.full.default([12, 512, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_545: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_544, 1, 0, 255);  slice_scatter_544 = None
    full_default_145: "f32[12, 3, 512, 513]" = torch.ops.aten.full.default([12, 3, 512, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    select_scatter_48: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_545, 1, 0);  slice_scatter_545 = None
    slice_scatter_546: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_48, 0, 0, 9223372036854775807);  select_scatter_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_4: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_5, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_154: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_4, as_strided_scatter_5);  new_empty_strided_4 = as_strided_scatter_5 = None
    as_strided_96: "f32[12, 3, 256, 256]" = torch.ops.aten.as_strided.default(copy_154, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328)
    clone_70: "f32[12, 3, 256, 256]" = torch.ops.aten.clone.default(as_strided_96, memory_format = torch.contiguous_format)
    full_default_147: "f32[12, 3, 256, 256]" = torch.ops.aten.full.default([12, 3, 256, 256], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    copy_155: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(as_strided_96, full_default_147);  as_strided_96 = None
    as_strided_scatter_6: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_154, copy_155, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328);  copy_154 = copy_155 = None
    full_default_148: "f32[12, 3, 256, 513]" = torch.ops.aten.full.default([12, 3, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_547: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_70, 3, 257, 9223372036854775807);  clone_70 = None
    slice_scatter_548: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_547, 2, -257, -1);  slice_scatter_547 = None
    slice_scatter_549: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_548, 1, 0, 9223372036854775807);  slice_scatter_548 = None
    slice_scatter_550: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_549, 0, 0, 9223372036854775807);  slice_scatter_549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    add_137: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(slice_scatter_546, slice_scatter_550);  slice_scatter_546 = slice_scatter_550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_5: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_6, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_156: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_5, as_strided_scatter_6);  new_empty_strided_5 = as_strided_scatter_6 = None
    as_strided_99: "f32[12, 256, 257]" = torch.ops.aten.as_strided.default(copy_156, [12, 256, 257], [525312, 513, 1], 394240)
    clone_71: "f32[12, 256, 257]" = torch.ops.aten.clone.default(as_strided_99, memory_format = torch.contiguous_format)
    full_default_152: "f32[12, 256, 257]" = torch.ops.aten.full.default([12, 256, 257], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    copy_157: "f32[12, 256, 257]" = torch.ops.aten.copy.default(as_strided_99, full_default_152);  as_strided_99 = None
    as_strided_scatter_7: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_156, copy_157, [12, 256, 257], [525312, 513, 1], 394240);  copy_156 = copy_157 = None
    full_default_153: "f32[12, 256, 513]" = torch.ops.aten.full.default([12, 256, 513], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_scatter_551: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_153, clone_71, 2, 0, 257);  clone_71 = None
    slice_scatter_552: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_551, 1, 256, 9223372036854775807);  slice_scatter_551 = None
    select_scatter_49: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_552, 1, -1);  slice_scatter_552 = None
    slice_scatter_553: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_49, 0, 0, 9223372036854775807);  select_scatter_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_138: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_137, slice_scatter_553);  add_137 = slice_scatter_553 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_6: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_7, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_158: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_6, as_strided_scatter_7);  new_empty_strided_6 = as_strided_scatter_7 = None
    as_strided_102: "f32[12, 3, 256, 257]" = torch.ops.aten.as_strided.default(copy_158, [12, 3, 256, 257], [525312, 131328, 513, 1], 256);  copy_158 = None
    clone_72: "f32[12, 3, 256, 257]" = torch.ops.aten.clone.default(as_strided_102, memory_format = torch.contiguous_format);  as_strided_102 = None
    slice_scatter_554: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_72, 3, 0, 257);  clone_72 = None
    slice_scatter_555: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_554, 2, 0, 256);  slice_scatter_554 = None
    slice_scatter_556: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_555, 1, 0, 9223372036854775807);  slice_scatter_555 = None
    slice_scatter_557: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_556, 0, 0, 9223372036854775807);  slice_scatter_556 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_139: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_138, slice_scatter_557);  add_138 = slice_scatter_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_923: "f32[12, 3, 513, 512]" = torch.ops.aten.view.default(add_139, [12, 3, 513, 512]);  add_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_50: "f32[12, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_923, [0, 0, 0, -1]);  view_923 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    view_924: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_50, [12, 3, 512, 512, 1]);  constant_pad_nd_50 = None
    permute_782: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_924, [0, 1, 2, 4, 3]);  view_924 = None
    view_925: "f32[36, 512, 512]" = torch.ops.aten.view.default(permute_782, [36, 512, 512]);  permute_782 = None
    bmm_26: "f32[36, 64, 512]" = torch.ops.aten.bmm.default(permute_783, view_925);  permute_783 = None
    bmm_27: "f32[36, 512, 64]" = torch.ops.aten.bmm.default(view_925, permute_784);  view_925 = permute_784 = None
    view_926: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_26, [12, 3, 64, 512, 1]);  bmm_26 = None
    permute_785: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_926, [0, 1, 4, 3, 2]);  view_926 = None
    view_927: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_27, [12, 3, 512, 64, 1]);  bmm_27 = None
    permute_786: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(view_927, [0, 1, 2, 4, 3]);  view_927 = None
    permute_787: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_785, [0, 1, 3, 4, 2]);  permute_785 = None
    squeeze_4: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_787, 4);  permute_787 = None
    permute_788: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_786, [0, 1, 2, 4, 3]);  permute_786 = None
    squeeze_5: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_788, 4);  permute_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    full_default_161: "f32[786432]" = torch.ops.aten.full.default([786432], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    iota_49: "i64[786432]" = torch.ops.prims.iota.default(786432, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
    as_strided_103: "i64[12, 3, 512, 64]" = torch.ops.aten.as_strided.default(iota_49, [12, 3, 512, 64], [64, 196608, 768, 1], 0);  iota_49 = None
    clone_73: "f32[12, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_4, memory_format = torch.contiguous_format);  squeeze_4 = None
    view_928: "f32[1179648]" = torch.ops.aten.view.default(clone_73, [1179648]);  clone_73 = None
    clone_74: "i64[12, 3, 512, 64]" = torch.ops.aten.clone.default(as_strided_103, memory_format = torch.contiguous_format);  as_strided_103 = None
    view_929: "i64[1179648]" = torch.ops.aten.view.default(clone_74, [1179648]);  clone_74 = None
    index_put_1: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_928, True);  view_928 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    view_931: "f32[1179648]" = torch.ops.aten.view.default(squeeze_5, [-1]);  squeeze_5 = None
    index_put_2: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_931, True);  view_931 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    as_strided_107: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_2, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_2 = None
    view_938: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_107, [12, 1024, 64]);  as_strided_107 = None
    view_939: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_938, [1, 12, 1024, 64]);  view_938 = None
    permute_793: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_939, [0, 2, 1, 3]);  view_939 = None
    permute_794: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_793, [1, 0, 2, 3]);  permute_793 = None
    view_940: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_794, [1024, 1, 768]);  permute_794 = None
    squeeze_7: "f32[1024, 768]" = torch.ops.aten.squeeze.dim(view_940, 1);  view_940 = None
    as_strided_108: "f32[1024, 768]" = torch.ops.aten.as_strided.default(full_default_161, [1024, 768], [768, 1], 0)
    copy_159: "f32[1024, 768]" = torch.ops.aten.copy.default(as_strided_108, squeeze_7);  squeeze_7 = None
    as_strided_scatter_8: "f32[786432]" = torch.ops.aten.as_strided_scatter.default(full_default_161, copy_159, [1024, 768], [768, 1], 0);  copy_159 = None
    as_strided_111: "f32[1024, 768]" = torch.ops.aten.as_strided.default(as_strided_scatter_8, [1024, 768], [768, 1], 0);  as_strided_scatter_8 = None
    new_empty_strided_7: "f32[1024, 768]" = torch.ops.aten.new_empty_strided.default(as_strided_111, [1024, 768], [768, 1])
    copy_160: "f32[1024, 768]" = torch.ops.aten.copy.default(new_empty_strided_7, as_strided_111);  new_empty_strided_7 = as_strided_111 = None
    as_strided_113: "f32[1024, 1, 768]" = torch.ops.aten.as_strided.default(copy_160, [1024, 1, 768], [768, 768, 1], 0)
    clone_76: "f32[1024, 1, 768]" = torch.ops.aten.clone.default(as_strided_113, memory_format = torch.contiguous_format)
    div_122: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(clone_76, 8.0);  clone_76 = None
    copy_161: "f32[1024, 1, 768]" = torch.ops.aten.copy.default(as_strided_113, div_122);  as_strided_113 = div_122 = None
    as_strided_scatter_9: "f32[1024, 768]" = torch.ops.aten.as_strided_scatter.default(copy_160, copy_161, [1024, 1, 768], [768, 768, 1], 0);  copy_160 = copy_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_941: "f32[1024, 768]" = torch.ops.aten.view.default(view_921, [1024, 768]);  view_921 = None
    mm_6: "f32[1024, 768]" = torch.ops.aten.mm.default(view_941, permute_795);  permute_795 = None
    permute_796: "f32[768, 1024]" = torch.ops.aten.permute.default(view_941, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_796, view_825);  permute_796 = None
    permute_797: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_25: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_941, [0], True);  view_941 = None
    view_942: "f32[768]" = torch.ops.aten.view.default(sum_25, [768]);  sum_25 = None
    permute_798: "f32[768, 768]" = torch.ops.aten.permute.default(permute_797, [1, 0]);  permute_797 = None
    view_943: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_6, [1024, 1, 768]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    as_strided_115: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_1, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_1 = None
    view_945: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_115, [12, 1024, 64]);  as_strided_115 = None
    view_946: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_945, [1, 12, 1024, 64]);  view_945 = None
    permute_800: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_946, [0, 2, 1, 3]);  view_946 = None
    permute_801: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_800, [1, 0, 2, 3]);  permute_800 = None
    view_947: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_801, [1024, 1, 768]);  permute_801 = None
    view_948: "f32[1024, 768]" = torch.ops.aten.view.default(view_947, [1024, 768]);  view_947 = None
    mm_8: "f32[1024, 768]" = torch.ops.aten.mm.default(view_948, permute_799);  permute_799 = None
    permute_805: "f32[768, 1024]" = torch.ops.aten.permute.default(view_948, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_805, view_825);  permute_805 = None
    permute_806: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_26: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_948, [0], True);  view_948 = None
    view_953: "f32[768]" = torch.ops.aten.view.default(sum_26, [768]);  sum_26 = None
    permute_807: "f32[768, 768]" = torch.ops.aten.permute.default(permute_806, [1, 0]);  permute_806 = None
    view_954: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_8, [1024, 1, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    add_140: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(view_943, view_954);  view_943 = view_954 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    mm_10: "f32[1024, 768]" = torch.ops.aten.mm.default(as_strided_scatter_9, permute_808);  permute_808 = None
    permute_810: "f32[768, 1024]" = torch.ops.aten.permute.default(as_strided_scatter_9, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_810, view_825);  permute_810 = view_825 = None
    permute_811: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_27: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(as_strided_scatter_9, [0], True);  as_strided_scatter_9 = None
    view_955: "f32[768]" = torch.ops.aten.view.default(sum_27, [768]);  sum_27 = None
    permute_812: "f32[768, 768]" = torch.ops.aten.permute.default(permute_811, [1, 0]);  permute_811 = None
    view_956: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_10, [1024, 1, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    add_141: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(add_140, view_956);  add_140 = view_956 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_813: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(add_141, [1, 0, 2]);  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    add_142: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_117, permute_813);  mul_117 = permute_813 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_126: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_142, primals_175);  primals_175 = None
    mul_127: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_126, 768)
    sum_28: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_126, [2], True)
    mul_128: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_126, mul_86);  mul_126 = None
    sum_29: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_128, [2], True);  mul_128 = None
    mul_129: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_86, sum_29);  sum_29 = None
    sub_104: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_127, sum_28);  mul_127 = sum_28 = None
    sub_105: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_104, mul_129);  sub_104 = mul_129 = None
    mul_130: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_123, sub_105);  div_123 = sub_105 = None
    mul_131: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_142, mul_86);  mul_86 = None
    sum_30: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_131, [0, 1]);  mul_131 = None
    sum_31: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 1]);  add_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_63: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_107, torch.float32);  getitem_107 = None
    mul_132: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
    mul_133: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_130, mul_132);  mul_132 = None
    clone_77: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_133, memory_format = torch.contiguous_format);  mul_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_957: "f32[1024, 768]" = torch.ops.aten.view.default(clone_77, [1024, 768]);  clone_77 = None
    mm_12: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_957, permute_814);  permute_814 = None
    permute_815: "f32[768, 1024]" = torch.ops.aten.permute.default(view_957, [1, 0])
    mm_13: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_815, view_823);  permute_815 = view_823 = None
    permute_816: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_32: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_957, [0], True);  view_957 = None
    view_958: "f32[768]" = torch.ops.aten.view.default(sum_32, [768]);  sum_32 = None
    permute_817: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_816, [1, 0]);  permute_816 = None
    view_959: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_12, [1, 1024, 3072]);  mm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_135: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_117, 0.5);  add_117 = None
    mul_136: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_822, view_822)
    mul_137: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_136, -0.5);  mul_136 = None
    exp_13: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_137);  mul_137 = None
    mul_138: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_139: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_822, mul_138);  view_822 = mul_138 = None
    add_144: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_135, mul_139);  mul_135 = mul_139 = None
    mul_140: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_959, add_144);  view_959 = add_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_960: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_140, [1024, 3072]);  mul_140 = None
    mm_14: "f32[1024, 768]" = torch.ops.aten.mm.default(view_960, permute_818);  permute_818 = None
    permute_819: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_960, [1, 0])
    mm_15: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_819, view_821);  permute_819 = view_821 = None
    permute_820: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_33: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_960, [0], True);  view_960 = None
    view_961: "f32[3072]" = torch.ops.aten.view.default(sum_33, [3072]);  sum_33 = None
    permute_821: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_820, [1, 0]);  permute_820 = None
    view_962: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_14, [1, 1024, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    add_145: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_130, view_962);  mul_130 = view_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_142: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_145, primals_169);  primals_169 = None
    mul_143: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_142, 768)
    sum_34: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_142, [2], True)
    mul_144: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_142, mul_81);  mul_142 = None
    sum_35: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_144, [2], True);  mul_144 = None
    mul_145: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_81, sum_35);  sum_35 = None
    sub_107: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_143, sum_34);  mul_143 = sum_34 = None
    sub_108: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_107, mul_145);  sub_107 = mul_145 = None
    mul_146: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_124, sub_108);  div_124 = sub_108 = None
    mul_147: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_145, mul_81);  mul_81 = None
    sum_36: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_147, [0, 1]);  mul_147 = None
    sum_37: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_145, [0, 1]);  add_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_64: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_103, torch.float32);  getitem_103 = None
    mul_148: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
    mul_149: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_146, mul_148);  mul_148 = None
    clone_78: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_149, memory_format = torch.contiguous_format);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_963: "f32[1024, 768]" = torch.ops.aten.view.default(clone_78, [1024, 768]);  clone_78 = None
    mm_16: "f32[1024, 768]" = torch.ops.aten.mm.default(view_963, permute_822);  permute_822 = None
    permute_823: "f32[768, 1024]" = torch.ops.aten.permute.default(view_963, [1, 0])
    mm_17: "f32[768, 768]" = torch.ops.aten.mm.default(permute_823, view_819);  permute_823 = view_819 = None
    permute_824: "f32[768, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_963, [0], True);  view_963 = None
    view_964: "f32[768]" = torch.ops.aten.view.default(sum_38, [768]);  sum_38 = None
    permute_825: "f32[768, 768]" = torch.ops.aten.permute.default(permute_824, [1, 0]);  permute_824 = None
    view_965: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_16, [1, 1024, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_826: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(view_965, [1, 0, 2]);  view_965 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    view_966: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(permute_826, [1024, 1, 12, 64]);  permute_826 = None
    permute_827: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_966, [1, 0, 2, 3]);  view_966 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    permute_828: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_827, [0, 2, 1, 3]);  permute_827 = None
    view_967: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_828, [12, 4, 256, 64]);  permute_828 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    view_968: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_967, [12, 4, 256, 64, 1]);  view_967 = None
    permute_829: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_968, [0, 1, 2, 4, 3]);  view_968 = None
    clone_79: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.clone.default(permute_829, memory_format = torch.contiguous_format);  permute_829 = None
    view_969: "f32[48, 256, 64]" = torch.ops.aten.view.default(clone_79, [48, 256, 64]);  clone_79 = None
    bmm_28: "f32[48, 768, 64]" = torch.ops.aten.bmm.default(permute_830, view_969);  permute_830 = None
    bmm_29: "f32[48, 256, 768]" = torch.ops.aten.bmm.default(view_969, permute_831);  view_969 = permute_831 = None
    view_970: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_28, [12, 4, 768, 64, 1]);  bmm_28 = None
    permute_832: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(view_970, [0, 1, 4, 3, 2]);  view_970 = None
    view_971: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_29, [12, 4, 256, 768, 1]);  bmm_29 = None
    permute_833: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(view_971, [0, 1, 2, 4, 3]);  view_971 = None
    permute_834: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_832, [0, 1, 4, 3, 2]);  permute_832 = None
    squeeze_8: "f32[12, 4, 768, 64]" = torch.ops.aten.squeeze.dim(permute_834, 4);  permute_834 = None
    permute_835: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_833, [0, 1, 2, 4, 3]);  permute_833 = None
    squeeze_9: "f32[12, 4, 256, 768]" = torch.ops.aten.squeeze.dim(permute_835, 4);  permute_835 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_scatter_558: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, squeeze_9, 3, 0, -1);  squeeze_9 = None
    slice_scatter_559: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_558, 2, 0, 9223372036854775807);  slice_scatter_558 = None
    slice_scatter_560: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_559, 1, 0, 9223372036854775807);  slice_scatter_559 = None
    slice_scatter_561: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_560, 0, 0, 9223372036854775807);  slice_scatter_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_972: "f32[12, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_561, [12, 4, 196864]);  slice_scatter_561 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_scatter_562: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, view_972, 2, 0, -256);  view_972 = None
    slice_scatter_563: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_562, 1, 0, 9223372036854775807);  slice_scatter_562 = None
    slice_scatter_564: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_563, 0, 0, 9223372036854775807);  slice_scatter_563 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_973: "f32[12, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_564, [12, 4, 256, 770]);  slice_scatter_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_51: "f32[12, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_973, [0, -257]);  view_973 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    view_974: "f32[2359296]" = torch.ops.aten.view.default(squeeze_8, [-1]);  squeeze_8 = None
    index_put_3: "f32[1179648]" = torch.ops.aten.index_put.default(full_default_128, [view_918], view_974, True);  view_974 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    as_strided_119: "f32[12, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_3, [12, 1536, 64], [98304, 64, 1], 0);  index_put_3 = None
    constant_pad_nd_52: "f32[12, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_119, [0, 0, -256, -256]);  as_strided_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    view_976: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_52, [1, 12, 1024, 64]);  constant_pad_nd_52 = None
    permute_836: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_976, [0, 2, 1, 3]);  view_976 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    view_977: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_51, [1, 12, 1024, 513]);  constant_pad_nd_51 = None
    permute_837: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_977, [0, 2, 1, 3]);  view_977 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    permute_838: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_836, [1, 0, 2, 3]);  permute_836 = None
    clone_81: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
    view_978: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_81, [1024, 1, 768]);  clone_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    convert_element_type_65: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(getitem_101, torch.float32);  getitem_101 = None
    mul_150: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
    mul_151: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_837, mul_150);  permute_837 = mul_150 = None
    clone_82: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_151, memory_format = torch.contiguous_format);  mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_99: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, clone_82);  clone_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    mul_152: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_99, alias_13);  where_99 = None
    sum_39: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_152, [-1], True)
    mul_153: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(alias_13, sum_39);  alias_13 = sum_39 = None
    sub_109: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(mul_152, mul_153);  mul_152 = mul_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    squeeze_10: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(sub_109, 0);  sub_109 = None
    copy_162: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_10);  squeeze_10 = None
    as_strided_scatter_10: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_162, [1024, 12, 513], [513, 525312, 1], 0);  copy_162 = None
    as_strided_123: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_10, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_10 = None
    new_empty_strided_8: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_123, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_163: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_8, as_strided_123);  new_empty_strided_8 = as_strided_123 = None
    as_strided_125: "f32[1, 1024, 12, 513]" = torch.ops.aten.as_strided.default(copy_163, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0)
    clone_83: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(as_strided_125, memory_format = torch.contiguous_format)
    copy_164: "f32[1, 1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_125, clone_83);  as_strided_125 = clone_83 = None
    as_strided_scatter_11: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_163, copy_164, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0);  copy_163 = copy_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    new_empty_strided_9: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_11, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_165: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_9, as_strided_scatter_11);  new_empty_strided_9 = as_strided_scatter_11 = None
    as_strided_128: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_165, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240)
    clone_84: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_128, memory_format = torch.contiguous_format)
    copy_166: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_128, full_default_130);  as_strided_128 = None
    as_strided_scatter_12: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_165, copy_166, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240);  copy_165 = copy_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_100: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_84);  clone_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    slice_scatter_565: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_100, 3, -257, 9223372036854775807);  where_100 = None
    slice_scatter_566: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_565, 2, 0, 9223372036854775807);  slice_scatter_565 = None
    slice_scatter_567: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_566, 1, -256, 9223372036854775807);  slice_scatter_566 = None
    slice_scatter_568: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_567, 0, 0, 9223372036854775807);  slice_scatter_567 = None
    squeeze_11: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(slice_scatter_568, 0);  slice_scatter_568 = None
    copy_167: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_11);  squeeze_11 = None
    as_strided_scatter_13: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_167, [1024, 12, 513], [513, 525312, 1], 0);  copy_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    as_strided_133: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_13, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_13 = None
    add_146: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_12, as_strided_133);  as_strided_scatter_12 = as_strided_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    new_empty_strided_10: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_146, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_168: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_10, add_146);  new_empty_strided_10 = add_146 = None
    as_strided_135: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_168, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0)
    clone_85: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_135, memory_format = torch.contiguous_format)
    copy_169: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_135, full_default_130);  as_strided_135 = None
    as_strided_scatter_14: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_168, copy_169, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0);  copy_168 = copy_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_101: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_85);  clone_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    slice_scatter_569: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_101, 3, 0, 257);  where_101 = None
    slice_scatter_570: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_569, 2, 0, 9223372036854775807);  slice_scatter_569 = None
    slice_scatter_571: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_570, 1, 0, 256);  slice_scatter_570 = None
    slice_scatter_572: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_571, 0, 0, 9223372036854775807);  slice_scatter_571 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    permute_839: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_572, [0, 2, 1, 3]);  slice_scatter_572 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_979: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_839, [12, 4, 256, 513]);  permute_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    add_147: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_14, view_979);  as_strided_scatter_14 = view_979 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_11: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_147, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_170: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_11, add_147);  new_empty_strided_11 = add_147 = None
    as_strided_138: "f32[12, 255, 255]" = torch.ops.aten.as_strided.default(copy_170, [12, 255, 255], [525312, 513, 1], 514)
    clone_86: "f32[12, 255, 255]" = torch.ops.aten.clone.default(as_strided_138, memory_format = torch.contiguous_format)
    copy_171: "f32[12, 255, 255]" = torch.ops.aten.copy.default(as_strided_138, full_default_142);  as_strided_138 = None
    as_strided_scatter_15: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_170, copy_171, [12, 255, 255], [525312, 513, 1], 514);  copy_170 = copy_171 = None
    slice_scatter_573: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_143, clone_86, 2, -255, 9223372036854775807);  clone_86 = None
    slice_scatter_574: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_573, 1, 0, 255);  slice_scatter_573 = None
    select_scatter_50: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_574, 1, 0);  slice_scatter_574 = None
    slice_scatter_575: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_50, 0, 0, 9223372036854775807);  select_scatter_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_12: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_15, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_172: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_12, as_strided_scatter_15);  new_empty_strided_12 = as_strided_scatter_15 = None
    as_strided_141: "f32[12, 3, 256, 256]" = torch.ops.aten.as_strided.default(copy_172, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328)
    clone_87: "f32[12, 3, 256, 256]" = torch.ops.aten.clone.default(as_strided_141, memory_format = torch.contiguous_format)
    copy_173: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(as_strided_141, full_default_147);  as_strided_141 = None
    as_strided_scatter_16: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_172, copy_173, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328);  copy_172 = copy_173 = None
    slice_scatter_576: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_87, 3, 257, 9223372036854775807);  clone_87 = None
    slice_scatter_577: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_576, 2, -257, -1);  slice_scatter_576 = None
    slice_scatter_578: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_577, 1, 0, 9223372036854775807);  slice_scatter_577 = None
    slice_scatter_579: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_578, 0, 0, 9223372036854775807);  slice_scatter_578 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    add_148: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(slice_scatter_575, slice_scatter_579);  slice_scatter_575 = slice_scatter_579 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_13: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_16, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_174: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_13, as_strided_scatter_16);  new_empty_strided_13 = as_strided_scatter_16 = None
    as_strided_144: "f32[12, 256, 257]" = torch.ops.aten.as_strided.default(copy_174, [12, 256, 257], [525312, 513, 1], 394240)
    clone_88: "f32[12, 256, 257]" = torch.ops.aten.clone.default(as_strided_144, memory_format = torch.contiguous_format)
    copy_175: "f32[12, 256, 257]" = torch.ops.aten.copy.default(as_strided_144, full_default_152);  as_strided_144 = None
    as_strided_scatter_17: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_174, copy_175, [12, 256, 257], [525312, 513, 1], 394240);  copy_174 = copy_175 = None
    slice_scatter_580: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_153, clone_88, 2, 0, 257);  clone_88 = None
    slice_scatter_581: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_580, 1, 256, 9223372036854775807);  slice_scatter_580 = None
    select_scatter_51: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_581, 1, -1);  slice_scatter_581 = None
    slice_scatter_582: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_51, 0, 0, 9223372036854775807);  select_scatter_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_149: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_148, slice_scatter_582);  add_148 = slice_scatter_582 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_14: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_17, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_176: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_14, as_strided_scatter_17);  new_empty_strided_14 = as_strided_scatter_17 = None
    as_strided_147: "f32[12, 3, 256, 257]" = torch.ops.aten.as_strided.default(copy_176, [12, 3, 256, 257], [525312, 131328, 513, 1], 256);  copy_176 = None
    clone_89: "f32[12, 3, 256, 257]" = torch.ops.aten.clone.default(as_strided_147, memory_format = torch.contiguous_format);  as_strided_147 = None
    slice_scatter_583: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_89, 3, 0, 257);  clone_89 = None
    slice_scatter_584: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_583, 2, 0, 256);  slice_scatter_583 = None
    slice_scatter_585: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_584, 1, 0, 9223372036854775807);  slice_scatter_584 = None
    slice_scatter_586: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_585, 0, 0, 9223372036854775807);  slice_scatter_585 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_150: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_149, slice_scatter_586);  add_149 = slice_scatter_586 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_980: "f32[12, 3, 513, 512]" = torch.ops.aten.view.default(add_150, [12, 3, 513, 512]);  add_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_53: "f32[12, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_980, [0, 0, 0, -1]);  view_980 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    view_981: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_53, [12, 3, 512, 512, 1]);  constant_pad_nd_53 = None
    permute_840: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_981, [0, 1, 2, 4, 3]);  view_981 = None
    view_982: "f32[36, 512, 512]" = torch.ops.aten.view.default(permute_840, [36, 512, 512]);  permute_840 = None
    bmm_30: "f32[36, 64, 512]" = torch.ops.aten.bmm.default(permute_841, view_982);  permute_841 = None
    bmm_31: "f32[36, 512, 64]" = torch.ops.aten.bmm.default(view_982, permute_842);  view_982 = permute_842 = None
    view_983: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_30, [12, 3, 64, 512, 1]);  bmm_30 = None
    permute_843: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_983, [0, 1, 4, 3, 2]);  view_983 = None
    view_984: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_31, [12, 3, 512, 64, 1]);  bmm_31 = None
    permute_844: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(view_984, [0, 1, 2, 4, 3]);  view_984 = None
    permute_845: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_843, [0, 1, 3, 4, 2]);  permute_843 = None
    squeeze_12: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_845, 4);  permute_845 = None
    permute_846: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_844, [0, 1, 2, 4, 3]);  permute_844 = None
    squeeze_13: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_846, 4);  permute_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    clone_90: "f32[12, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_12, memory_format = torch.contiguous_format);  squeeze_12 = None
    view_985: "f32[1179648]" = torch.ops.aten.view.default(clone_90, [1179648]);  clone_90 = None
    index_put_4: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_985, True);  view_985 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    view_988: "f32[1179648]" = torch.ops.aten.view.default(squeeze_13, [-1]);  squeeze_13 = None
    index_put_5: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_988, True);  view_988 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    as_strided_152: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_5, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_5 = None
    view_995: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_152, [12, 1024, 64]);  as_strided_152 = None
    view_996: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_995, [1, 12, 1024, 64]);  view_995 = None
    permute_851: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_996, [0, 2, 1, 3]);  view_996 = None
    permute_852: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_851, [1, 0, 2, 3]);  permute_851 = None
    view_997: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_852, [1024, 1, 768]);  permute_852 = None
    squeeze_15: "f32[1024, 768]" = torch.ops.aten.squeeze.dim(view_997, 1);  view_997 = None
    copy_177: "f32[1024, 768]" = torch.ops.aten.copy.default(as_strided_108, squeeze_15);  squeeze_15 = None
    as_strided_scatter_18: "f32[786432]" = torch.ops.aten.as_strided_scatter.default(full_default_161, copy_177, [1024, 768], [768, 1], 0);  copy_177 = None
    as_strided_156: "f32[1024, 768]" = torch.ops.aten.as_strided.default(as_strided_scatter_18, [1024, 768], [768, 1], 0);  as_strided_scatter_18 = None
    new_empty_strided_15: "f32[1024, 768]" = torch.ops.aten.new_empty_strided.default(as_strided_156, [1024, 768], [768, 1])
    copy_178: "f32[1024, 768]" = torch.ops.aten.copy.default(new_empty_strided_15, as_strided_156);  new_empty_strided_15 = as_strided_156 = None
    as_strided_158: "f32[1024, 1, 768]" = torch.ops.aten.as_strided.default(copy_178, [1024, 1, 768], [768, 768, 1], 0)
    clone_93: "f32[1024, 1, 768]" = torch.ops.aten.clone.default(as_strided_158, memory_format = torch.contiguous_format)
    div_125: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(clone_93, 8.0);  clone_93 = None
    copy_179: "f32[1024, 1, 768]" = torch.ops.aten.copy.default(as_strided_158, div_125);  as_strided_158 = div_125 = None
    as_strided_scatter_19: "f32[1024, 768]" = torch.ops.aten.as_strided_scatter.default(copy_178, copy_179, [1024, 1, 768], [768, 768, 1], 0);  copy_178 = copy_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_998: "f32[1024, 768]" = torch.ops.aten.view.default(view_978, [1024, 768]);  view_978 = None
    mm_18: "f32[1024, 768]" = torch.ops.aten.mm.default(view_998, permute_853);  permute_853 = None
    permute_854: "f32[768, 1024]" = torch.ops.aten.permute.default(view_998, [1, 0])
    mm_19: "f32[768, 768]" = torch.ops.aten.mm.default(permute_854, view_750);  permute_854 = None
    permute_855: "f32[768, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_40: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_998, [0], True);  view_998 = None
    view_999: "f32[768]" = torch.ops.aten.view.default(sum_40, [768]);  sum_40 = None
    permute_856: "f32[768, 768]" = torch.ops.aten.permute.default(permute_855, [1, 0]);  permute_855 = None
    view_1000: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_18, [1024, 1, 768]);  mm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    as_strided_160: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_4, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_4 = None
    view_1002: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_160, [12, 1024, 64]);  as_strided_160 = None
    view_1003: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1002, [1, 12, 1024, 64]);  view_1002 = None
    permute_858: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1003, [0, 2, 1, 3]);  view_1003 = None
    permute_859: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_858, [1, 0, 2, 3]);  permute_858 = None
    view_1004: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_859, [1024, 1, 768]);  permute_859 = None
    view_1005: "f32[1024, 768]" = torch.ops.aten.view.default(view_1004, [1024, 768]);  view_1004 = None
    mm_20: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1005, permute_857);  permute_857 = None
    permute_863: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1005, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_863, view_750);  permute_863 = None
    permute_864: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_41: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1005, [0], True);  view_1005 = None
    view_1010: "f32[768]" = torch.ops.aten.view.default(sum_41, [768]);  sum_41 = None
    permute_865: "f32[768, 768]" = torch.ops.aten.permute.default(permute_864, [1, 0]);  permute_864 = None
    view_1011: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_20, [1024, 1, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    add_151: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(view_1000, view_1011);  view_1000 = view_1011 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    mm_22: "f32[1024, 768]" = torch.ops.aten.mm.default(as_strided_scatter_19, permute_866);  permute_866 = None
    permute_868: "f32[768, 1024]" = torch.ops.aten.permute.default(as_strided_scatter_19, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_868, view_750);  permute_868 = view_750 = None
    permute_869: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_42: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(as_strided_scatter_19, [0], True);  as_strided_scatter_19 = None
    view_1012: "f32[768]" = torch.ops.aten.view.default(sum_42, [768]);  sum_42 = None
    permute_870: "f32[768, 768]" = torch.ops.aten.permute.default(permute_869, [1, 0]);  permute_869 = None
    view_1013: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_22, [1024, 1, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    add_152: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(add_151, view_1013);  add_151 = view_1013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_871: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(add_152, [1, 0, 2]);  add_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    add_153: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_146, permute_871);  mul_146 = permute_871 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_155: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_153, primals_159);  primals_159 = None
    mul_156: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_155, 768)
    sum_43: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_155, [2], True)
    mul_157: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_155, mul_78);  mul_155 = None
    sum_44: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [2], True);  mul_157 = None
    mul_158: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_78, sum_44);  sum_44 = None
    sub_111: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_156, sum_43);  mul_156 = sum_43 = None
    sub_112: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_111, mul_158);  sub_111 = mul_158 = None
    mul_159: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_126, sub_112);  div_126 = sub_112 = None
    mul_160: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_153, mul_78);  mul_78 = None
    sum_45: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_160, [0, 1]);  mul_160 = None
    sum_46: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_153, [0, 1]);  add_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_66: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_97, torch.float32);  getitem_97 = None
    mul_161: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
    mul_162: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_159, mul_161);  mul_161 = None
    clone_94: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_162, memory_format = torch.contiguous_format);  mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_1014: "f32[1024, 768]" = torch.ops.aten.view.default(clone_94, [1024, 768]);  clone_94 = None
    mm_24: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_1014, permute_872);  permute_872 = None
    permute_873: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1014, [1, 0])
    mm_25: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_873, view_748);  permute_873 = view_748 = None
    permute_874: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_47: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
    view_1015: "f32[768]" = torch.ops.aten.view.default(sum_47, [768]);  sum_47 = None
    permute_875: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_874, [1, 0]);  permute_874 = None
    view_1016: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_24, [1, 1024, 3072]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_164: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
    mul_165: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_747, view_747)
    mul_166: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_165, -0.5);  mul_165 = None
    exp_14: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_166);  mul_166 = None
    mul_167: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_168: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_747, mul_167);  view_747 = mul_167 = None
    add_155: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_164, mul_168);  mul_164 = mul_168 = None
    mul_169: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1016, add_155);  view_1016 = add_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_1017: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_169, [1024, 3072]);  mul_169 = None
    mm_26: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1017, permute_876);  permute_876 = None
    permute_877: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_1017, [1, 0])
    mm_27: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_877, view_746);  permute_877 = view_746 = None
    permute_878: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_48: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1017, [0], True);  view_1017 = None
    view_1018: "f32[3072]" = torch.ops.aten.view.default(sum_48, [3072]);  sum_48 = None
    permute_879: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_878, [1, 0]);  permute_878 = None
    view_1019: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_26, [1, 1024, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    add_156: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_159, view_1019);  mul_159 = view_1019 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_171: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_156, primals_153);  primals_153 = None
    mul_172: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_171, 768)
    sum_49: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_171, [2], True)
    mul_173: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_171, mul_73);  mul_171 = None
    sum_50: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_173, [2], True);  mul_173 = None
    mul_174: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_73, sum_50);  sum_50 = None
    sub_114: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_172, sum_49);  mul_172 = sum_49 = None
    sub_115: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_114, mul_174);  sub_114 = mul_174 = None
    mul_175: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_127, sub_115);  div_127 = sub_115 = None
    mul_176: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_156, mul_73);  mul_73 = None
    sum_51: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_176, [0, 1]);  mul_176 = None
    sum_52: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_156, [0, 1]);  add_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_67: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_93, torch.float32);  getitem_93 = None
    mul_177: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
    mul_178: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_175, mul_177);  mul_177 = None
    clone_95: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_178, memory_format = torch.contiguous_format);  mul_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_1020: "f32[1024, 768]" = torch.ops.aten.view.default(clone_95, [1024, 768]);  clone_95 = None
    mm_28: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1020, permute_880);  permute_880 = None
    permute_881: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1020, [1, 0])
    mm_29: "f32[768, 768]" = torch.ops.aten.mm.default(permute_881, view_744);  permute_881 = view_744 = None
    permute_882: "f32[768, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1020, [0], True);  view_1020 = None
    view_1021: "f32[768]" = torch.ops.aten.view.default(sum_53, [768]);  sum_53 = None
    permute_883: "f32[768, 768]" = torch.ops.aten.permute.default(permute_882, [1, 0]);  permute_882 = None
    view_1022: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_28, [1, 1024, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_884: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(view_1022, [1, 0, 2]);  view_1022 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    view_1023: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(permute_884, [1024, 1, 12, 64]);  permute_884 = None
    permute_885: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1023, [1, 0, 2, 3]);  view_1023 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    permute_886: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_885, [0, 2, 1, 3]);  permute_885 = None
    view_1024: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_886, [12, 4, 256, 64]);  permute_886 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    view_1025: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_1024, [12, 4, 256, 64, 1]);  view_1024 = None
    permute_887: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1025, [0, 1, 2, 4, 3]);  view_1025 = None
    clone_96: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.clone.default(permute_887, memory_format = torch.contiguous_format);  permute_887 = None
    view_1026: "f32[48, 256, 64]" = torch.ops.aten.view.default(clone_96, [48, 256, 64]);  clone_96 = None
    bmm_32: "f32[48, 768, 64]" = torch.ops.aten.bmm.default(permute_888, view_1026);  permute_888 = None
    bmm_33: "f32[48, 256, 768]" = torch.ops.aten.bmm.default(view_1026, permute_889);  view_1026 = permute_889 = None
    view_1027: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_32, [12, 4, 768, 64, 1]);  bmm_32 = None
    permute_890: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(view_1027, [0, 1, 4, 3, 2]);  view_1027 = None
    view_1028: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_33, [12, 4, 256, 768, 1]);  bmm_33 = None
    permute_891: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(view_1028, [0, 1, 2, 4, 3]);  view_1028 = None
    permute_892: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_890, [0, 1, 4, 3, 2]);  permute_890 = None
    squeeze_16: "f32[12, 4, 768, 64]" = torch.ops.aten.squeeze.dim(permute_892, 4);  permute_892 = None
    permute_893: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_891, [0, 1, 2, 4, 3]);  permute_891 = None
    squeeze_17: "f32[12, 4, 256, 768]" = torch.ops.aten.squeeze.dim(permute_893, 4);  permute_893 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_scatter_587: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, squeeze_17, 3, 0, -1);  squeeze_17 = None
    slice_scatter_588: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_587, 2, 0, 9223372036854775807);  slice_scatter_587 = None
    slice_scatter_589: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_588, 1, 0, 9223372036854775807);  slice_scatter_588 = None
    slice_scatter_590: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_589, 0, 0, 9223372036854775807);  slice_scatter_589 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1029: "f32[12, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_590, [12, 4, 196864]);  slice_scatter_590 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_scatter_591: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, view_1029, 2, 0, -256);  view_1029 = None
    slice_scatter_592: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_591, 1, 0, 9223372036854775807);  slice_scatter_591 = None
    slice_scatter_593: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_592, 0, 0, 9223372036854775807);  slice_scatter_592 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1030: "f32[12, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_593, [12, 4, 256, 770]);  slice_scatter_593 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_54: "f32[12, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1030, [0, -257]);  view_1030 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    view_1031: "f32[2359296]" = torch.ops.aten.view.default(squeeze_16, [-1]);  squeeze_16 = None
    index_put_6: "f32[1179648]" = torch.ops.aten.index_put.default(full_default_128, [view_918], view_1031, True);  view_1031 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    as_strided_164: "f32[12, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_6, [12, 1536, 64], [98304, 64, 1], 0);  index_put_6 = None
    constant_pad_nd_55: "f32[12, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_164, [0, 0, -256, -256]);  as_strided_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    view_1033: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_55, [1, 12, 1024, 64]);  constant_pad_nd_55 = None
    permute_894: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1033, [0, 2, 1, 3]);  view_1033 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    view_1034: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_54, [1, 12, 1024, 513]);  constant_pad_nd_54 = None
    permute_895: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1034, [0, 2, 1, 3]);  view_1034 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    permute_896: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_894, [1, 0, 2, 3]);  permute_894 = None
    clone_98: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_896, memory_format = torch.contiguous_format);  permute_896 = None
    view_1035: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_98, [1024, 1, 768]);  clone_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    convert_element_type_68: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(getitem_91, torch.float32);  getitem_91 = None
    mul_179: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
    mul_180: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_895, mul_179);  permute_895 = mul_179 = None
    clone_99: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_180, memory_format = torch.contiguous_format);  mul_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_102: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, clone_99);  clone_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    mul_181: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_102, alias_14);  where_102 = None
    sum_54: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_181, [-1], True)
    mul_182: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(alias_14, sum_54);  alias_14 = sum_54 = None
    sub_116: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(mul_181, mul_182);  mul_181 = mul_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    squeeze_18: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(sub_116, 0);  sub_116 = None
    copy_180: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_18);  squeeze_18 = None
    as_strided_scatter_20: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_180, [1024, 12, 513], [513, 525312, 1], 0);  copy_180 = None
    as_strided_168: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_20, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_20 = None
    new_empty_strided_16: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_168, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_181: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_16, as_strided_168);  new_empty_strided_16 = as_strided_168 = None
    as_strided_170: "f32[1, 1024, 12, 513]" = torch.ops.aten.as_strided.default(copy_181, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0)
    clone_100: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(as_strided_170, memory_format = torch.contiguous_format)
    copy_182: "f32[1, 1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_170, clone_100);  as_strided_170 = clone_100 = None
    as_strided_scatter_21: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_181, copy_182, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0);  copy_181 = copy_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    new_empty_strided_17: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_21, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_183: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_17, as_strided_scatter_21);  new_empty_strided_17 = as_strided_scatter_21 = None
    as_strided_173: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_183, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240)
    clone_101: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_173, memory_format = torch.contiguous_format)
    copy_184: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_173, full_default_130);  as_strided_173 = None
    as_strided_scatter_22: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_183, copy_184, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240);  copy_183 = copy_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_103: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_101);  clone_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    slice_scatter_594: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_103, 3, -257, 9223372036854775807);  where_103 = None
    slice_scatter_595: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_594, 2, 0, 9223372036854775807);  slice_scatter_594 = None
    slice_scatter_596: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_595, 1, -256, 9223372036854775807);  slice_scatter_595 = None
    slice_scatter_597: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_596, 0, 0, 9223372036854775807);  slice_scatter_596 = None
    squeeze_19: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(slice_scatter_597, 0);  slice_scatter_597 = None
    copy_185: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_19);  squeeze_19 = None
    as_strided_scatter_23: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_185, [1024, 12, 513], [513, 525312, 1], 0);  copy_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    as_strided_178: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_23, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_23 = None
    add_157: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_22, as_strided_178);  as_strided_scatter_22 = as_strided_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    new_empty_strided_18: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_157, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_186: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_18, add_157);  new_empty_strided_18 = add_157 = None
    as_strided_180: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_186, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0)
    clone_102: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_180, memory_format = torch.contiguous_format)
    copy_187: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_180, full_default_130);  as_strided_180 = None
    as_strided_scatter_24: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_186, copy_187, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0);  copy_186 = copy_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_104: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_102);  clone_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    slice_scatter_598: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_104, 3, 0, 257);  where_104 = None
    slice_scatter_599: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_598, 2, 0, 9223372036854775807);  slice_scatter_598 = None
    slice_scatter_600: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_599, 1, 0, 256);  slice_scatter_599 = None
    slice_scatter_601: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_600, 0, 0, 9223372036854775807);  slice_scatter_600 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    permute_897: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_601, [0, 2, 1, 3]);  slice_scatter_601 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_1036: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_897, [12, 4, 256, 513]);  permute_897 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    add_158: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_24, view_1036);  as_strided_scatter_24 = view_1036 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_19: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_158, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_188: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_19, add_158);  new_empty_strided_19 = add_158 = None
    as_strided_183: "f32[12, 255, 255]" = torch.ops.aten.as_strided.default(copy_188, [12, 255, 255], [525312, 513, 1], 514)
    clone_103: "f32[12, 255, 255]" = torch.ops.aten.clone.default(as_strided_183, memory_format = torch.contiguous_format)
    copy_189: "f32[12, 255, 255]" = torch.ops.aten.copy.default(as_strided_183, full_default_142);  as_strided_183 = None
    as_strided_scatter_25: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_188, copy_189, [12, 255, 255], [525312, 513, 1], 514);  copy_188 = copy_189 = None
    slice_scatter_602: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_143, clone_103, 2, -255, 9223372036854775807);  clone_103 = None
    slice_scatter_603: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_602, 1, 0, 255);  slice_scatter_602 = None
    select_scatter_52: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_603, 1, 0);  slice_scatter_603 = None
    slice_scatter_604: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_52, 0, 0, 9223372036854775807);  select_scatter_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_20: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_25, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_190: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_20, as_strided_scatter_25);  new_empty_strided_20 = as_strided_scatter_25 = None
    as_strided_186: "f32[12, 3, 256, 256]" = torch.ops.aten.as_strided.default(copy_190, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328)
    clone_104: "f32[12, 3, 256, 256]" = torch.ops.aten.clone.default(as_strided_186, memory_format = torch.contiguous_format)
    copy_191: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(as_strided_186, full_default_147);  as_strided_186 = None
    as_strided_scatter_26: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_190, copy_191, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328);  copy_190 = copy_191 = None
    slice_scatter_605: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_104, 3, 257, 9223372036854775807);  clone_104 = None
    slice_scatter_606: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_605, 2, -257, -1);  slice_scatter_605 = None
    slice_scatter_607: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_606, 1, 0, 9223372036854775807);  slice_scatter_606 = None
    slice_scatter_608: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_607, 0, 0, 9223372036854775807);  slice_scatter_607 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    add_159: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(slice_scatter_604, slice_scatter_608);  slice_scatter_604 = slice_scatter_608 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_21: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_26, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_192: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_21, as_strided_scatter_26);  new_empty_strided_21 = as_strided_scatter_26 = None
    as_strided_189: "f32[12, 256, 257]" = torch.ops.aten.as_strided.default(copy_192, [12, 256, 257], [525312, 513, 1], 394240)
    clone_105: "f32[12, 256, 257]" = torch.ops.aten.clone.default(as_strided_189, memory_format = torch.contiguous_format)
    copy_193: "f32[12, 256, 257]" = torch.ops.aten.copy.default(as_strided_189, full_default_152);  as_strided_189 = None
    as_strided_scatter_27: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_192, copy_193, [12, 256, 257], [525312, 513, 1], 394240);  copy_192 = copy_193 = None
    slice_scatter_609: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_153, clone_105, 2, 0, 257);  clone_105 = None
    slice_scatter_610: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_609, 1, 256, 9223372036854775807);  slice_scatter_609 = None
    select_scatter_53: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_610, 1, -1);  slice_scatter_610 = None
    slice_scatter_611: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_53, 0, 0, 9223372036854775807);  select_scatter_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_160: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_159, slice_scatter_611);  add_159 = slice_scatter_611 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_22: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_27, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_194: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_22, as_strided_scatter_27);  new_empty_strided_22 = as_strided_scatter_27 = None
    as_strided_192: "f32[12, 3, 256, 257]" = torch.ops.aten.as_strided.default(copy_194, [12, 3, 256, 257], [525312, 131328, 513, 1], 256);  copy_194 = None
    clone_106: "f32[12, 3, 256, 257]" = torch.ops.aten.clone.default(as_strided_192, memory_format = torch.contiguous_format);  as_strided_192 = None
    slice_scatter_612: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_106, 3, 0, 257);  clone_106 = None
    slice_scatter_613: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_612, 2, 0, 256);  slice_scatter_612 = None
    slice_scatter_614: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_613, 1, 0, 9223372036854775807);  slice_scatter_613 = None
    slice_scatter_615: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_614, 0, 0, 9223372036854775807);  slice_scatter_614 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_161: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_160, slice_scatter_615);  add_160 = slice_scatter_615 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_1037: "f32[12, 3, 513, 512]" = torch.ops.aten.view.default(add_161, [12, 3, 513, 512]);  add_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_56: "f32[12, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1037, [0, 0, 0, -1]);  view_1037 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    view_1038: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_56, [12, 3, 512, 512, 1]);  constant_pad_nd_56 = None
    permute_898: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1038, [0, 1, 2, 4, 3]);  view_1038 = None
    view_1039: "f32[36, 512, 512]" = torch.ops.aten.view.default(permute_898, [36, 512, 512]);  permute_898 = None
    bmm_34: "f32[36, 64, 512]" = torch.ops.aten.bmm.default(permute_899, view_1039);  permute_899 = None
    bmm_35: "f32[36, 512, 64]" = torch.ops.aten.bmm.default(view_1039, permute_900);  view_1039 = permute_900 = None
    view_1040: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_34, [12, 3, 64, 512, 1]);  bmm_34 = None
    permute_901: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1040, [0, 1, 4, 3, 2]);  view_1040 = None
    view_1041: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_35, [12, 3, 512, 64, 1]);  bmm_35 = None
    permute_902: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(view_1041, [0, 1, 2, 4, 3]);  view_1041 = None
    permute_903: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_901, [0, 1, 3, 4, 2]);  permute_901 = None
    squeeze_20: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_903, 4);  permute_903 = None
    permute_904: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_902, [0, 1, 2, 4, 3]);  permute_902 = None
    squeeze_21: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_904, 4);  permute_904 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    clone_107: "f32[12, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_20, memory_format = torch.contiguous_format);  squeeze_20 = None
    view_1042: "f32[1179648]" = torch.ops.aten.view.default(clone_107, [1179648]);  clone_107 = None
    index_put_7: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1042, True);  view_1042 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    view_1045: "f32[1179648]" = torch.ops.aten.view.default(squeeze_21, [-1]);  squeeze_21 = None
    index_put_8: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1045, True);  view_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    as_strided_197: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_8, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_8 = None
    view_1052: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_197, [12, 1024, 64]);  as_strided_197 = None
    view_1053: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1052, [1, 12, 1024, 64]);  view_1052 = None
    permute_909: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1053, [0, 2, 1, 3]);  view_1053 = None
    permute_910: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_909, [1, 0, 2, 3]);  permute_909 = None
    view_1054: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_910, [1024, 1, 768]);  permute_910 = None
    squeeze_23: "f32[1024, 768]" = torch.ops.aten.squeeze.dim(view_1054, 1);  view_1054 = None
    copy_195: "f32[1024, 768]" = torch.ops.aten.copy.default(as_strided_108, squeeze_23);  squeeze_23 = None
    as_strided_scatter_28: "f32[786432]" = torch.ops.aten.as_strided_scatter.default(full_default_161, copy_195, [1024, 768], [768, 1], 0);  copy_195 = None
    as_strided_201: "f32[1024, 768]" = torch.ops.aten.as_strided.default(as_strided_scatter_28, [1024, 768], [768, 1], 0);  as_strided_scatter_28 = None
    new_empty_strided_23: "f32[1024, 768]" = torch.ops.aten.new_empty_strided.default(as_strided_201, [1024, 768], [768, 1])
    copy_196: "f32[1024, 768]" = torch.ops.aten.copy.default(new_empty_strided_23, as_strided_201);  new_empty_strided_23 = as_strided_201 = None
    as_strided_203: "f32[1024, 1, 768]" = torch.ops.aten.as_strided.default(copy_196, [1024, 1, 768], [768, 768, 1], 0)
    clone_110: "f32[1024, 1, 768]" = torch.ops.aten.clone.default(as_strided_203, memory_format = torch.contiguous_format)
    div_128: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(clone_110, 8.0);  clone_110 = None
    copy_197: "f32[1024, 1, 768]" = torch.ops.aten.copy.default(as_strided_203, div_128);  as_strided_203 = div_128 = None
    as_strided_scatter_29: "f32[1024, 768]" = torch.ops.aten.as_strided_scatter.default(copy_196, copy_197, [1024, 1, 768], [768, 768, 1], 0);  copy_196 = copy_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_1055: "f32[1024, 768]" = torch.ops.aten.view.default(view_1035, [1024, 768]);  view_1035 = None
    mm_30: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1055, permute_911);  permute_911 = None
    permute_912: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1055, [1, 0])
    mm_31: "f32[768, 768]" = torch.ops.aten.mm.default(permute_912, view_675);  permute_912 = None
    permute_913: "f32[768, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_55: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1055, [0], True);  view_1055 = None
    view_1056: "f32[768]" = torch.ops.aten.view.default(sum_55, [768]);  sum_55 = None
    permute_914: "f32[768, 768]" = torch.ops.aten.permute.default(permute_913, [1, 0]);  permute_913 = None
    view_1057: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_30, [1024, 1, 768]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    as_strided_205: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_7, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_7 = None
    view_1059: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_205, [12, 1024, 64]);  as_strided_205 = None
    view_1060: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1059, [1, 12, 1024, 64]);  view_1059 = None
    permute_916: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1060, [0, 2, 1, 3]);  view_1060 = None
    permute_917: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_916, [1, 0, 2, 3]);  permute_916 = None
    view_1061: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_917, [1024, 1, 768]);  permute_917 = None
    view_1062: "f32[1024, 768]" = torch.ops.aten.view.default(view_1061, [1024, 768]);  view_1061 = None
    mm_32: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1062, permute_915);  permute_915 = None
    permute_921: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1062, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_921, view_675);  permute_921 = None
    permute_922: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_56: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1062, [0], True);  view_1062 = None
    view_1067: "f32[768]" = torch.ops.aten.view.default(sum_56, [768]);  sum_56 = None
    permute_923: "f32[768, 768]" = torch.ops.aten.permute.default(permute_922, [1, 0]);  permute_922 = None
    view_1068: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_32, [1024, 1, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    add_162: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(view_1057, view_1068);  view_1057 = view_1068 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    mm_34: "f32[1024, 768]" = torch.ops.aten.mm.default(as_strided_scatter_29, permute_924);  permute_924 = None
    permute_926: "f32[768, 1024]" = torch.ops.aten.permute.default(as_strided_scatter_29, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_926, view_675);  permute_926 = view_675 = None
    permute_927: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_57: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(as_strided_scatter_29, [0], True);  as_strided_scatter_29 = None
    view_1069: "f32[768]" = torch.ops.aten.view.default(sum_57, [768]);  sum_57 = None
    permute_928: "f32[768, 768]" = torch.ops.aten.permute.default(permute_927, [1, 0]);  permute_927 = None
    view_1070: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_34, [1024, 1, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    add_163: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(add_162, view_1070);  add_162 = view_1070 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_929: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(add_163, [1, 0, 2]);  add_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    add_164: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_175, permute_929);  mul_175 = permute_929 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_184: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_164, primals_143);  primals_143 = None
    mul_185: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_184, 768)
    sum_58: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_184, [2], True)
    mul_186: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_184, mul_70);  mul_184 = None
    sum_59: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True);  mul_186 = None
    mul_187: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_70, sum_59);  sum_59 = None
    sub_118: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_185, sum_58);  mul_185 = sum_58 = None
    sub_119: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_118, mul_187);  sub_118 = mul_187 = None
    mul_188: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_129, sub_119);  div_129 = sub_119 = None
    mul_189: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_164, mul_70);  mul_70 = None
    sum_60: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_189, [0, 1]);  mul_189 = None
    sum_61: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_164, [0, 1]);  add_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_69: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_87, torch.float32);  getitem_87 = None
    mul_190: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
    mul_191: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_188, mul_190);  mul_190 = None
    clone_111: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_191, memory_format = torch.contiguous_format);  mul_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_1071: "f32[1024, 768]" = torch.ops.aten.view.default(clone_111, [1024, 768]);  clone_111 = None
    mm_36: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_1071, permute_930);  permute_930 = None
    permute_931: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1071, [1, 0])
    mm_37: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_931, view_673);  permute_931 = view_673 = None
    permute_932: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_62: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1071, [0], True);  view_1071 = None
    view_1072: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    permute_933: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_932, [1, 0]);  permute_932 = None
    view_1073: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_36, [1, 1024, 3072]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_193: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_95, 0.5);  add_95 = None
    mul_194: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_672, view_672)
    mul_195: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_194, -0.5);  mul_194 = None
    exp_15: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_195);  mul_195 = None
    mul_196: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_197: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_672, mul_196);  view_672 = mul_196 = None
    add_166: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_193, mul_197);  mul_193 = mul_197 = None
    mul_198: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1073, add_166);  view_1073 = add_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_1074: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_198, [1024, 3072]);  mul_198 = None
    mm_38: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1074, permute_934);  permute_934 = None
    permute_935: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_1074, [1, 0])
    mm_39: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_935, view_671);  permute_935 = view_671 = None
    permute_936: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_63: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1074, [0], True);  view_1074 = None
    view_1075: "f32[3072]" = torch.ops.aten.view.default(sum_63, [3072]);  sum_63 = None
    permute_937: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_936, [1, 0]);  permute_936 = None
    view_1076: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_38, [1, 1024, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    add_167: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_188, view_1076);  mul_188 = view_1076 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_200: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_167, primals_137);  primals_137 = None
    mul_201: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_200, 768)
    sum_64: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_200, [2], True)
    mul_202: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_200, mul_65);  mul_200 = None
    sum_65: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_202, [2], True);  mul_202 = None
    mul_203: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_65, sum_65);  sum_65 = None
    sub_121: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_201, sum_64);  mul_201 = sum_64 = None
    sub_122: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_121, mul_203);  sub_121 = mul_203 = None
    mul_204: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_130, sub_122);  div_130 = sub_122 = None
    mul_205: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_167, mul_65);  mul_65 = None
    sum_66: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_205, [0, 1]);  mul_205 = None
    sum_67: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_167, [0, 1]);  add_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_70: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_83, torch.float32);  getitem_83 = None
    mul_206: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
    mul_207: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_204, mul_206);  mul_206 = None
    clone_112: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_207, memory_format = torch.contiguous_format);  mul_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_1077: "f32[1024, 768]" = torch.ops.aten.view.default(clone_112, [1024, 768]);  clone_112 = None
    mm_40: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1077, permute_938);  permute_938 = None
    permute_939: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1077, [1, 0])
    mm_41: "f32[768, 768]" = torch.ops.aten.mm.default(permute_939, view_669);  permute_939 = view_669 = None
    permute_940: "f32[768, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1077, [0], True);  view_1077 = None
    view_1078: "f32[768]" = torch.ops.aten.view.default(sum_68, [768]);  sum_68 = None
    permute_941: "f32[768, 768]" = torch.ops.aten.permute.default(permute_940, [1, 0]);  permute_940 = None
    view_1079: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_40, [1, 1024, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_942: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(view_1079, [1, 0, 2]);  view_1079 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    view_1080: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(permute_942, [1024, 1, 12, 64]);  permute_942 = None
    permute_943: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1080, [1, 0, 2, 3]);  view_1080 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    permute_944: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_943, [0, 2, 1, 3]);  permute_943 = None
    view_1081: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_944, [12, 4, 256, 64]);  permute_944 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    view_1082: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_1081, [12, 4, 256, 64, 1]);  view_1081 = None
    permute_945: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1082, [0, 1, 2, 4, 3]);  view_1082 = None
    clone_113: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.clone.default(permute_945, memory_format = torch.contiguous_format);  permute_945 = None
    view_1083: "f32[48, 256, 64]" = torch.ops.aten.view.default(clone_113, [48, 256, 64]);  clone_113 = None
    bmm_36: "f32[48, 768, 64]" = torch.ops.aten.bmm.default(permute_946, view_1083);  permute_946 = None
    bmm_37: "f32[48, 256, 768]" = torch.ops.aten.bmm.default(view_1083, permute_947);  view_1083 = permute_947 = None
    view_1084: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_36, [12, 4, 768, 64, 1]);  bmm_36 = None
    permute_948: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(view_1084, [0, 1, 4, 3, 2]);  view_1084 = None
    view_1085: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_37, [12, 4, 256, 768, 1]);  bmm_37 = None
    permute_949: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(view_1085, [0, 1, 2, 4, 3]);  view_1085 = None
    permute_950: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_948, [0, 1, 4, 3, 2]);  permute_948 = None
    squeeze_24: "f32[12, 4, 768, 64]" = torch.ops.aten.squeeze.dim(permute_950, 4);  permute_950 = None
    permute_951: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_949, [0, 1, 2, 4, 3]);  permute_949 = None
    squeeze_25: "f32[12, 4, 256, 768]" = torch.ops.aten.squeeze.dim(permute_951, 4);  permute_951 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_scatter_616: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, squeeze_25, 3, 0, -1);  squeeze_25 = None
    slice_scatter_617: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_616, 2, 0, 9223372036854775807);  slice_scatter_616 = None
    slice_scatter_618: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_617, 1, 0, 9223372036854775807);  slice_scatter_617 = None
    slice_scatter_619: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_618, 0, 0, 9223372036854775807);  slice_scatter_618 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1086: "f32[12, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_619, [12, 4, 196864]);  slice_scatter_619 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_scatter_620: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, view_1086, 2, 0, -256);  view_1086 = None
    slice_scatter_621: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_620, 1, 0, 9223372036854775807);  slice_scatter_620 = None
    slice_scatter_622: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_621, 0, 0, 9223372036854775807);  slice_scatter_621 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1087: "f32[12, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_622, [12, 4, 256, 770]);  slice_scatter_622 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_57: "f32[12, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1087, [0, -257]);  view_1087 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    view_1088: "f32[2359296]" = torch.ops.aten.view.default(squeeze_24, [-1]);  squeeze_24 = None
    index_put_9: "f32[1179648]" = torch.ops.aten.index_put.default(full_default_128, [view_918], view_1088, True);  view_1088 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    as_strided_209: "f32[12, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_9, [12, 1536, 64], [98304, 64, 1], 0);  index_put_9 = None
    constant_pad_nd_58: "f32[12, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_209, [0, 0, -256, -256]);  as_strided_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    view_1090: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_58, [1, 12, 1024, 64]);  constant_pad_nd_58 = None
    permute_952: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1090, [0, 2, 1, 3]);  view_1090 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    view_1091: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_57, [1, 12, 1024, 513]);  constant_pad_nd_57 = None
    permute_953: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1091, [0, 2, 1, 3]);  view_1091 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    permute_954: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_952, [1, 0, 2, 3]);  permute_952 = None
    clone_115: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_954, memory_format = torch.contiguous_format);  permute_954 = None
    view_1092: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_115, [1024, 1, 768]);  clone_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    convert_element_type_71: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(getitem_81, torch.float32);  getitem_81 = None
    mul_208: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
    mul_209: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_953, mul_208);  permute_953 = mul_208 = None
    clone_116: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_209, memory_format = torch.contiguous_format);  mul_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_105: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, clone_116);  clone_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    mul_210: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_105, alias_15);  where_105 = None
    sum_69: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_210, [-1], True)
    mul_211: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(alias_15, sum_69);  alias_15 = sum_69 = None
    sub_123: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(mul_210, mul_211);  mul_210 = mul_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    squeeze_26: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(sub_123, 0);  sub_123 = None
    copy_198: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_26);  squeeze_26 = None
    as_strided_scatter_30: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_198, [1024, 12, 513], [513, 525312, 1], 0);  copy_198 = None
    as_strided_213: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_30, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_30 = None
    new_empty_strided_24: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_213, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_199: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_24, as_strided_213);  new_empty_strided_24 = as_strided_213 = None
    as_strided_215: "f32[1, 1024, 12, 513]" = torch.ops.aten.as_strided.default(copy_199, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0)
    clone_117: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(as_strided_215, memory_format = torch.contiguous_format)
    copy_200: "f32[1, 1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_215, clone_117);  as_strided_215 = clone_117 = None
    as_strided_scatter_31: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_199, copy_200, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0);  copy_199 = copy_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    new_empty_strided_25: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_31, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_201: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_25, as_strided_scatter_31);  new_empty_strided_25 = as_strided_scatter_31 = None
    as_strided_218: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_201, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240)
    clone_118: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_218, memory_format = torch.contiguous_format)
    copy_202: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_218, full_default_130);  as_strided_218 = None
    as_strided_scatter_32: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_201, copy_202, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240);  copy_201 = copy_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_106: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_118);  clone_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    slice_scatter_623: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_106, 3, -257, 9223372036854775807);  where_106 = None
    slice_scatter_624: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_623, 2, 0, 9223372036854775807);  slice_scatter_623 = None
    slice_scatter_625: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_624, 1, -256, 9223372036854775807);  slice_scatter_624 = None
    slice_scatter_626: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_625, 0, 0, 9223372036854775807);  slice_scatter_625 = None
    squeeze_27: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(slice_scatter_626, 0);  slice_scatter_626 = None
    copy_203: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_27);  squeeze_27 = None
    as_strided_scatter_33: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_203, [1024, 12, 513], [513, 525312, 1], 0);  copy_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    as_strided_223: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_33, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_33 = None
    add_168: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_32, as_strided_223);  as_strided_scatter_32 = as_strided_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    new_empty_strided_26: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_168, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_204: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_26, add_168);  new_empty_strided_26 = add_168 = None
    as_strided_225: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_204, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0)
    clone_119: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_225, memory_format = torch.contiguous_format)
    copy_205: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_225, full_default_130);  as_strided_225 = None
    as_strided_scatter_34: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_204, copy_205, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0);  copy_204 = copy_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_107: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_119);  clone_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    slice_scatter_627: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_107, 3, 0, 257);  where_107 = None
    slice_scatter_628: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_627, 2, 0, 9223372036854775807);  slice_scatter_627 = None
    slice_scatter_629: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_628, 1, 0, 256);  slice_scatter_628 = None
    slice_scatter_630: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_629, 0, 0, 9223372036854775807);  slice_scatter_629 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    permute_955: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_630, [0, 2, 1, 3]);  slice_scatter_630 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_1093: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_955, [12, 4, 256, 513]);  permute_955 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    add_169: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_34, view_1093);  as_strided_scatter_34 = view_1093 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_27: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_169, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_206: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_27, add_169);  new_empty_strided_27 = add_169 = None
    as_strided_228: "f32[12, 255, 255]" = torch.ops.aten.as_strided.default(copy_206, [12, 255, 255], [525312, 513, 1], 514)
    clone_120: "f32[12, 255, 255]" = torch.ops.aten.clone.default(as_strided_228, memory_format = torch.contiguous_format)
    copy_207: "f32[12, 255, 255]" = torch.ops.aten.copy.default(as_strided_228, full_default_142);  as_strided_228 = None
    as_strided_scatter_35: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_206, copy_207, [12, 255, 255], [525312, 513, 1], 514);  copy_206 = copy_207 = None
    slice_scatter_631: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_143, clone_120, 2, -255, 9223372036854775807);  clone_120 = None
    slice_scatter_632: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_631, 1, 0, 255);  slice_scatter_631 = None
    select_scatter_54: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_632, 1, 0);  slice_scatter_632 = None
    slice_scatter_633: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_54, 0, 0, 9223372036854775807);  select_scatter_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_28: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_35, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_208: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_28, as_strided_scatter_35);  new_empty_strided_28 = as_strided_scatter_35 = None
    as_strided_231: "f32[12, 3, 256, 256]" = torch.ops.aten.as_strided.default(copy_208, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328)
    clone_121: "f32[12, 3, 256, 256]" = torch.ops.aten.clone.default(as_strided_231, memory_format = torch.contiguous_format)
    copy_209: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(as_strided_231, full_default_147);  as_strided_231 = None
    as_strided_scatter_36: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_208, copy_209, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328);  copy_208 = copy_209 = None
    slice_scatter_634: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_121, 3, 257, 9223372036854775807);  clone_121 = None
    slice_scatter_635: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_634, 2, -257, -1);  slice_scatter_634 = None
    slice_scatter_636: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_635, 1, 0, 9223372036854775807);  slice_scatter_635 = None
    slice_scatter_637: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_636, 0, 0, 9223372036854775807);  slice_scatter_636 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    add_170: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(slice_scatter_633, slice_scatter_637);  slice_scatter_633 = slice_scatter_637 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_29: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_36, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_210: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_29, as_strided_scatter_36);  new_empty_strided_29 = as_strided_scatter_36 = None
    as_strided_234: "f32[12, 256, 257]" = torch.ops.aten.as_strided.default(copy_210, [12, 256, 257], [525312, 513, 1], 394240)
    clone_122: "f32[12, 256, 257]" = torch.ops.aten.clone.default(as_strided_234, memory_format = torch.contiguous_format)
    copy_211: "f32[12, 256, 257]" = torch.ops.aten.copy.default(as_strided_234, full_default_152);  as_strided_234 = None
    as_strided_scatter_37: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_210, copy_211, [12, 256, 257], [525312, 513, 1], 394240);  copy_210 = copy_211 = None
    slice_scatter_638: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_153, clone_122, 2, 0, 257);  clone_122 = None
    slice_scatter_639: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_638, 1, 256, 9223372036854775807);  slice_scatter_638 = None
    select_scatter_55: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_639, 1, -1);  slice_scatter_639 = None
    slice_scatter_640: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_55, 0, 0, 9223372036854775807);  select_scatter_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_171: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_170, slice_scatter_640);  add_170 = slice_scatter_640 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_30: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_37, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_212: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_30, as_strided_scatter_37);  new_empty_strided_30 = as_strided_scatter_37 = None
    as_strided_237: "f32[12, 3, 256, 257]" = torch.ops.aten.as_strided.default(copy_212, [12, 3, 256, 257], [525312, 131328, 513, 1], 256);  copy_212 = None
    clone_123: "f32[12, 3, 256, 257]" = torch.ops.aten.clone.default(as_strided_237, memory_format = torch.contiguous_format);  as_strided_237 = None
    slice_scatter_641: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_123, 3, 0, 257);  clone_123 = None
    slice_scatter_642: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_641, 2, 0, 256);  slice_scatter_641 = None
    slice_scatter_643: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_642, 1, 0, 9223372036854775807);  slice_scatter_642 = None
    slice_scatter_644: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_643, 0, 0, 9223372036854775807);  slice_scatter_643 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_172: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_171, slice_scatter_644);  add_171 = slice_scatter_644 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_1094: "f32[12, 3, 513, 512]" = torch.ops.aten.view.default(add_172, [12, 3, 513, 512]);  add_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_59: "f32[12, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1094, [0, 0, 0, -1]);  view_1094 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    view_1095: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_59, [12, 3, 512, 512, 1]);  constant_pad_nd_59 = None
    permute_956: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1095, [0, 1, 2, 4, 3]);  view_1095 = None
    view_1096: "f32[36, 512, 512]" = torch.ops.aten.view.default(permute_956, [36, 512, 512]);  permute_956 = None
    bmm_38: "f32[36, 64, 512]" = torch.ops.aten.bmm.default(permute_957, view_1096);  permute_957 = None
    bmm_39: "f32[36, 512, 64]" = torch.ops.aten.bmm.default(view_1096, permute_958);  view_1096 = permute_958 = None
    view_1097: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_38, [12, 3, 64, 512, 1]);  bmm_38 = None
    permute_959: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1097, [0, 1, 4, 3, 2]);  view_1097 = None
    view_1098: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_39, [12, 3, 512, 64, 1]);  bmm_39 = None
    permute_960: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(view_1098, [0, 1, 2, 4, 3]);  view_1098 = None
    permute_961: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_959, [0, 1, 3, 4, 2]);  permute_959 = None
    squeeze_28: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_961, 4);  permute_961 = None
    permute_962: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_960, [0, 1, 2, 4, 3]);  permute_960 = None
    squeeze_29: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_962, 4);  permute_962 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    clone_124: "f32[12, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_28, memory_format = torch.contiguous_format);  squeeze_28 = None
    view_1099: "f32[1179648]" = torch.ops.aten.view.default(clone_124, [1179648]);  clone_124 = None
    index_put_10: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1099, True);  view_1099 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    view_1102: "f32[1179648]" = torch.ops.aten.view.default(squeeze_29, [-1]);  squeeze_29 = None
    index_put_11: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1102, True);  view_1102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    as_strided_242: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_11, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_11 = None
    view_1109: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_242, [12, 1024, 64]);  as_strided_242 = None
    view_1110: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1109, [1, 12, 1024, 64]);  view_1109 = None
    permute_967: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1110, [0, 2, 1, 3]);  view_1110 = None
    permute_968: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_967, [1, 0, 2, 3]);  permute_967 = None
    view_1111: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_968, [1024, 1, 768]);  permute_968 = None
    squeeze_31: "f32[1024, 768]" = torch.ops.aten.squeeze.dim(view_1111, 1);  view_1111 = None
    copy_213: "f32[1024, 768]" = torch.ops.aten.copy.default(as_strided_108, squeeze_31);  squeeze_31 = None
    as_strided_scatter_38: "f32[786432]" = torch.ops.aten.as_strided_scatter.default(full_default_161, copy_213, [1024, 768], [768, 1], 0);  copy_213 = None
    as_strided_246: "f32[1024, 768]" = torch.ops.aten.as_strided.default(as_strided_scatter_38, [1024, 768], [768, 1], 0);  as_strided_scatter_38 = None
    new_empty_strided_31: "f32[1024, 768]" = torch.ops.aten.new_empty_strided.default(as_strided_246, [1024, 768], [768, 1])
    copy_214: "f32[1024, 768]" = torch.ops.aten.copy.default(new_empty_strided_31, as_strided_246);  new_empty_strided_31 = as_strided_246 = None
    as_strided_248: "f32[1024, 1, 768]" = torch.ops.aten.as_strided.default(copy_214, [1024, 1, 768], [768, 768, 1], 0)
    clone_127: "f32[1024, 1, 768]" = torch.ops.aten.clone.default(as_strided_248, memory_format = torch.contiguous_format)
    div_131: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(clone_127, 8.0);  clone_127 = None
    copy_215: "f32[1024, 1, 768]" = torch.ops.aten.copy.default(as_strided_248, div_131);  as_strided_248 = div_131 = None
    as_strided_scatter_39: "f32[1024, 768]" = torch.ops.aten.as_strided_scatter.default(copy_214, copy_215, [1024, 1, 768], [768, 768, 1], 0);  copy_214 = copy_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_1112: "f32[1024, 768]" = torch.ops.aten.view.default(view_1092, [1024, 768]);  view_1092 = None
    mm_42: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1112, permute_969);  permute_969 = None
    permute_970: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1112, [1, 0])
    mm_43: "f32[768, 768]" = torch.ops.aten.mm.default(permute_970, view_600);  permute_970 = None
    permute_971: "f32[768, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_70: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1112, [0], True);  view_1112 = None
    view_1113: "f32[768]" = torch.ops.aten.view.default(sum_70, [768]);  sum_70 = None
    permute_972: "f32[768, 768]" = torch.ops.aten.permute.default(permute_971, [1, 0]);  permute_971 = None
    view_1114: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_42, [1024, 1, 768]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    as_strided_250: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_10, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_10 = None
    view_1116: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_250, [12, 1024, 64]);  as_strided_250 = None
    view_1117: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1116, [1, 12, 1024, 64]);  view_1116 = None
    permute_974: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1117, [0, 2, 1, 3]);  view_1117 = None
    permute_975: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_974, [1, 0, 2, 3]);  permute_974 = None
    view_1118: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_975, [1024, 1, 768]);  permute_975 = None
    view_1119: "f32[1024, 768]" = torch.ops.aten.view.default(view_1118, [1024, 768]);  view_1118 = None
    mm_44: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1119, permute_973);  permute_973 = None
    permute_979: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1119, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_979, view_600);  permute_979 = None
    permute_980: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_71: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1119, [0], True);  view_1119 = None
    view_1124: "f32[768]" = torch.ops.aten.view.default(sum_71, [768]);  sum_71 = None
    permute_981: "f32[768, 768]" = torch.ops.aten.permute.default(permute_980, [1, 0]);  permute_980 = None
    view_1125: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_44, [1024, 1, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    add_173: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(view_1114, view_1125);  view_1114 = view_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    mm_46: "f32[1024, 768]" = torch.ops.aten.mm.default(as_strided_scatter_39, permute_982);  permute_982 = None
    permute_984: "f32[768, 1024]" = torch.ops.aten.permute.default(as_strided_scatter_39, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_984, view_600);  permute_984 = view_600 = None
    permute_985: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_72: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(as_strided_scatter_39, [0], True);  as_strided_scatter_39 = None
    view_1126: "f32[768]" = torch.ops.aten.view.default(sum_72, [768]);  sum_72 = None
    permute_986: "f32[768, 768]" = torch.ops.aten.permute.default(permute_985, [1, 0]);  permute_985 = None
    view_1127: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_46, [1024, 1, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    add_174: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(add_173, view_1127);  add_173 = view_1127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_987: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(add_174, [1, 0, 2]);  add_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    add_175: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_204, permute_987);  mul_204 = permute_987 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_213: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_175, primals_127);  primals_127 = None
    mul_214: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_213, 768)
    sum_73: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_213, [2], True)
    mul_215: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_213, mul_62);  mul_213 = None
    sum_74: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_215, [2], True);  mul_215 = None
    mul_216: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_62, sum_74);  sum_74 = None
    sub_125: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_214, sum_73);  mul_214 = sum_73 = None
    sub_126: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_125, mul_216);  sub_125 = mul_216 = None
    mul_217: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_132, sub_126);  div_132 = sub_126 = None
    mul_218: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_175, mul_62);  mul_62 = None
    sum_75: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_218, [0, 1]);  mul_218 = None
    sum_76: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_175, [0, 1]);  add_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_72: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_77, torch.float32);  getitem_77 = None
    mul_219: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 1.1111111111111112);  convert_element_type_72 = None
    mul_220: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_217, mul_219);  mul_219 = None
    clone_128: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_220, memory_format = torch.contiguous_format);  mul_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_1128: "f32[1024, 768]" = torch.ops.aten.view.default(clone_128, [1024, 768]);  clone_128 = None
    mm_48: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_1128, permute_988);  permute_988 = None
    permute_989: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1128, [1, 0])
    mm_49: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_989, view_598);  permute_989 = view_598 = None
    permute_990: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_77: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1128, [0], True);  view_1128 = None
    view_1129: "f32[768]" = torch.ops.aten.view.default(sum_77, [768]);  sum_77 = None
    permute_991: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_990, [1, 0]);  permute_990 = None
    view_1130: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_48, [1, 1024, 3072]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_222: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_84, 0.5);  add_84 = None
    mul_223: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_597, view_597)
    mul_224: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_223, -0.5);  mul_223 = None
    exp_16: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_224);  mul_224 = None
    mul_225: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_226: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_597, mul_225);  view_597 = mul_225 = None
    add_177: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_222, mul_226);  mul_222 = mul_226 = None
    mul_227: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1130, add_177);  view_1130 = add_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_1131: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_227, [1024, 3072]);  mul_227 = None
    mm_50: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1131, permute_992);  permute_992 = None
    permute_993: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_1131, [1, 0])
    mm_51: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_993, view_596);  permute_993 = view_596 = None
    permute_994: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_78: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1131, [0], True);  view_1131 = None
    view_1132: "f32[3072]" = torch.ops.aten.view.default(sum_78, [3072]);  sum_78 = None
    permute_995: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_994, [1, 0]);  permute_994 = None
    view_1133: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_50, [1, 1024, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    add_178: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_217, view_1133);  mul_217 = view_1133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_229: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_178, primals_121);  primals_121 = None
    mul_230: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_229, 768)
    sum_79: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_229, [2], True)
    mul_231: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_229, mul_57);  mul_229 = None
    sum_80: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True);  mul_231 = None
    mul_232: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_57, sum_80);  sum_80 = None
    sub_128: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_230, sum_79);  mul_230 = sum_79 = None
    sub_129: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_128, mul_232);  sub_128 = mul_232 = None
    mul_233: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_133, sub_129);  div_133 = sub_129 = None
    mul_234: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_178, mul_57);  mul_57 = None
    sum_81: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_234, [0, 1]);  mul_234 = None
    sum_82: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_178, [0, 1]);  add_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_73: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_73, torch.float32);  getitem_73 = None
    mul_235: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.1111111111111112);  convert_element_type_73 = None
    mul_236: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_233, mul_235);  mul_235 = None
    clone_129: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_236, memory_format = torch.contiguous_format);  mul_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_1134: "f32[1024, 768]" = torch.ops.aten.view.default(clone_129, [1024, 768]);  clone_129 = None
    mm_52: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1134, permute_996);  permute_996 = None
    permute_997: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1134, [1, 0])
    mm_53: "f32[768, 768]" = torch.ops.aten.mm.default(permute_997, view_594);  permute_997 = view_594 = None
    permute_998: "f32[768, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1134, [0], True);  view_1134 = None
    view_1135: "f32[768]" = torch.ops.aten.view.default(sum_83, [768]);  sum_83 = None
    permute_999: "f32[768, 768]" = torch.ops.aten.permute.default(permute_998, [1, 0]);  permute_998 = None
    view_1136: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_52, [1, 1024, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_1000: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(view_1136, [1, 0, 2]);  view_1136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    view_1137: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(permute_1000, [1024, 1, 12, 64]);  permute_1000 = None
    permute_1001: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1137, [1, 0, 2, 3]);  view_1137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    permute_1002: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1001, [0, 2, 1, 3]);  permute_1001 = None
    view_1138: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_1002, [12, 4, 256, 64]);  permute_1002 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    view_1139: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_1138, [12, 4, 256, 64, 1]);  view_1138 = None
    permute_1003: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1139, [0, 1, 2, 4, 3]);  view_1139 = None
    clone_130: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.clone.default(permute_1003, memory_format = torch.contiguous_format);  permute_1003 = None
    view_1140: "f32[48, 256, 64]" = torch.ops.aten.view.default(clone_130, [48, 256, 64]);  clone_130 = None
    bmm_40: "f32[48, 768, 64]" = torch.ops.aten.bmm.default(permute_1004, view_1140);  permute_1004 = None
    bmm_41: "f32[48, 256, 768]" = torch.ops.aten.bmm.default(view_1140, permute_1005);  view_1140 = permute_1005 = None
    view_1141: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_40, [12, 4, 768, 64, 1]);  bmm_40 = None
    permute_1006: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(view_1141, [0, 1, 4, 3, 2]);  view_1141 = None
    view_1142: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_41, [12, 4, 256, 768, 1]);  bmm_41 = None
    permute_1007: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(view_1142, [0, 1, 2, 4, 3]);  view_1142 = None
    permute_1008: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_1006, [0, 1, 4, 3, 2]);  permute_1006 = None
    squeeze_32: "f32[12, 4, 768, 64]" = torch.ops.aten.squeeze.dim(permute_1008, 4);  permute_1008 = None
    permute_1009: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_1007, [0, 1, 2, 4, 3]);  permute_1007 = None
    squeeze_33: "f32[12, 4, 256, 768]" = torch.ops.aten.squeeze.dim(permute_1009, 4);  permute_1009 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_scatter_645: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, squeeze_33, 3, 0, -1);  squeeze_33 = None
    slice_scatter_646: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_645, 2, 0, 9223372036854775807);  slice_scatter_645 = None
    slice_scatter_647: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_646, 1, 0, 9223372036854775807);  slice_scatter_646 = None
    slice_scatter_648: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_647, 0, 0, 9223372036854775807);  slice_scatter_647 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1143: "f32[12, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_648, [12, 4, 196864]);  slice_scatter_648 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_scatter_649: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, view_1143, 2, 0, -256);  view_1143 = None
    slice_scatter_650: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_649, 1, 0, 9223372036854775807);  slice_scatter_649 = None
    slice_scatter_651: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_650, 0, 0, 9223372036854775807);  slice_scatter_650 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1144: "f32[12, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_651, [12, 4, 256, 770]);  slice_scatter_651 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_60: "f32[12, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1144, [0, -257]);  view_1144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    view_1145: "f32[2359296]" = torch.ops.aten.view.default(squeeze_32, [-1]);  squeeze_32 = None
    index_put_12: "f32[1179648]" = torch.ops.aten.index_put.default(full_default_128, [view_918], view_1145, True);  view_1145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    as_strided_254: "f32[12, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_12, [12, 1536, 64], [98304, 64, 1], 0);  index_put_12 = None
    constant_pad_nd_61: "f32[12, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_254, [0, 0, -256, -256]);  as_strided_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    view_1147: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_61, [1, 12, 1024, 64]);  constant_pad_nd_61 = None
    permute_1010: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1147, [0, 2, 1, 3]);  view_1147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    view_1148: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_60, [1, 12, 1024, 513]);  constant_pad_nd_60 = None
    permute_1011: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1148, [0, 2, 1, 3]);  view_1148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    permute_1012: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1010, [1, 0, 2, 3]);  permute_1010 = None
    clone_132: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_1012, memory_format = torch.contiguous_format);  permute_1012 = None
    view_1149: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_132, [1024, 1, 768]);  clone_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    convert_element_type_74: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(getitem_71, torch.float32);  getitem_71 = None
    mul_237: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 1.1111111111111112);  convert_element_type_74 = None
    mul_238: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1011, mul_237);  permute_1011 = mul_237 = None
    clone_133: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_238, memory_format = torch.contiguous_format);  mul_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_108: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, clone_133);  clone_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    mul_239: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_108, alias_16);  where_108 = None
    sum_84: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_239, [-1], True)
    mul_240: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(alias_16, sum_84);  alias_16 = sum_84 = None
    sub_130: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(mul_239, mul_240);  mul_239 = mul_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    squeeze_34: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(sub_130, 0);  sub_130 = None
    copy_216: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_34);  squeeze_34 = None
    as_strided_scatter_40: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_216, [1024, 12, 513], [513, 525312, 1], 0);  copy_216 = None
    as_strided_258: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_40, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_40 = None
    new_empty_strided_32: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_258, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_217: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_32, as_strided_258);  new_empty_strided_32 = as_strided_258 = None
    as_strided_260: "f32[1, 1024, 12, 513]" = torch.ops.aten.as_strided.default(copy_217, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0)
    clone_134: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(as_strided_260, memory_format = torch.contiguous_format)
    copy_218: "f32[1, 1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_260, clone_134);  as_strided_260 = clone_134 = None
    as_strided_scatter_41: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_217, copy_218, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0);  copy_217 = copy_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    new_empty_strided_33: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_41, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_219: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_33, as_strided_scatter_41);  new_empty_strided_33 = as_strided_scatter_41 = None
    as_strided_263: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_219, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240)
    clone_135: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_263, memory_format = torch.contiguous_format)
    copy_220: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_263, full_default_130);  as_strided_263 = None
    as_strided_scatter_42: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_219, copy_220, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240);  copy_219 = copy_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_109: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_135);  clone_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    slice_scatter_652: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_109, 3, -257, 9223372036854775807);  where_109 = None
    slice_scatter_653: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_652, 2, 0, 9223372036854775807);  slice_scatter_652 = None
    slice_scatter_654: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_653, 1, -256, 9223372036854775807);  slice_scatter_653 = None
    slice_scatter_655: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_654, 0, 0, 9223372036854775807);  slice_scatter_654 = None
    squeeze_35: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(slice_scatter_655, 0);  slice_scatter_655 = None
    copy_221: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_35);  squeeze_35 = None
    as_strided_scatter_43: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_221, [1024, 12, 513], [513, 525312, 1], 0);  copy_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    as_strided_268: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_43, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_43 = None
    add_179: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_42, as_strided_268);  as_strided_scatter_42 = as_strided_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    new_empty_strided_34: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_179, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_222: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_34, add_179);  new_empty_strided_34 = add_179 = None
    as_strided_270: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_222, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0)
    clone_136: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_270, memory_format = torch.contiguous_format)
    copy_223: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_270, full_default_130);  as_strided_270 = None
    as_strided_scatter_44: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_222, copy_223, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0);  copy_222 = copy_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_110: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_136);  clone_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    slice_scatter_656: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_110, 3, 0, 257);  where_110 = None
    slice_scatter_657: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_656, 2, 0, 9223372036854775807);  slice_scatter_656 = None
    slice_scatter_658: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_657, 1, 0, 256);  slice_scatter_657 = None
    slice_scatter_659: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_658, 0, 0, 9223372036854775807);  slice_scatter_658 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    permute_1013: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_659, [0, 2, 1, 3]);  slice_scatter_659 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_1150: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_1013, [12, 4, 256, 513]);  permute_1013 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    add_180: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_44, view_1150);  as_strided_scatter_44 = view_1150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_35: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_180, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_224: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_35, add_180);  new_empty_strided_35 = add_180 = None
    as_strided_273: "f32[12, 255, 255]" = torch.ops.aten.as_strided.default(copy_224, [12, 255, 255], [525312, 513, 1], 514)
    clone_137: "f32[12, 255, 255]" = torch.ops.aten.clone.default(as_strided_273, memory_format = torch.contiguous_format)
    copy_225: "f32[12, 255, 255]" = torch.ops.aten.copy.default(as_strided_273, full_default_142);  as_strided_273 = None
    as_strided_scatter_45: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_224, copy_225, [12, 255, 255], [525312, 513, 1], 514);  copy_224 = copy_225 = None
    slice_scatter_660: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_143, clone_137, 2, -255, 9223372036854775807);  clone_137 = None
    slice_scatter_661: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_660, 1, 0, 255);  slice_scatter_660 = None
    select_scatter_56: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_661, 1, 0);  slice_scatter_661 = None
    slice_scatter_662: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_56, 0, 0, 9223372036854775807);  select_scatter_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_36: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_45, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_226: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_36, as_strided_scatter_45);  new_empty_strided_36 = as_strided_scatter_45 = None
    as_strided_276: "f32[12, 3, 256, 256]" = torch.ops.aten.as_strided.default(copy_226, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328)
    clone_138: "f32[12, 3, 256, 256]" = torch.ops.aten.clone.default(as_strided_276, memory_format = torch.contiguous_format)
    copy_227: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(as_strided_276, full_default_147);  as_strided_276 = None
    as_strided_scatter_46: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_226, copy_227, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328);  copy_226 = copy_227 = None
    slice_scatter_663: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_138, 3, 257, 9223372036854775807);  clone_138 = None
    slice_scatter_664: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_663, 2, -257, -1);  slice_scatter_663 = None
    slice_scatter_665: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_664, 1, 0, 9223372036854775807);  slice_scatter_664 = None
    slice_scatter_666: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_665, 0, 0, 9223372036854775807);  slice_scatter_665 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    add_181: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(slice_scatter_662, slice_scatter_666);  slice_scatter_662 = slice_scatter_666 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_37: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_46, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_228: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_37, as_strided_scatter_46);  new_empty_strided_37 = as_strided_scatter_46 = None
    as_strided_279: "f32[12, 256, 257]" = torch.ops.aten.as_strided.default(copy_228, [12, 256, 257], [525312, 513, 1], 394240)
    clone_139: "f32[12, 256, 257]" = torch.ops.aten.clone.default(as_strided_279, memory_format = torch.contiguous_format)
    copy_229: "f32[12, 256, 257]" = torch.ops.aten.copy.default(as_strided_279, full_default_152);  as_strided_279 = None
    as_strided_scatter_47: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_228, copy_229, [12, 256, 257], [525312, 513, 1], 394240);  copy_228 = copy_229 = None
    slice_scatter_667: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_153, clone_139, 2, 0, 257);  clone_139 = None
    slice_scatter_668: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_667, 1, 256, 9223372036854775807);  slice_scatter_667 = None
    select_scatter_57: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_668, 1, -1);  slice_scatter_668 = None
    slice_scatter_669: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_57, 0, 0, 9223372036854775807);  select_scatter_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_182: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_181, slice_scatter_669);  add_181 = slice_scatter_669 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_38: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_47, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_230: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_38, as_strided_scatter_47);  new_empty_strided_38 = as_strided_scatter_47 = None
    as_strided_282: "f32[12, 3, 256, 257]" = torch.ops.aten.as_strided.default(copy_230, [12, 3, 256, 257], [525312, 131328, 513, 1], 256);  copy_230 = None
    clone_140: "f32[12, 3, 256, 257]" = torch.ops.aten.clone.default(as_strided_282, memory_format = torch.contiguous_format);  as_strided_282 = None
    slice_scatter_670: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_140, 3, 0, 257);  clone_140 = None
    slice_scatter_671: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_670, 2, 0, 256);  slice_scatter_670 = None
    slice_scatter_672: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_671, 1, 0, 9223372036854775807);  slice_scatter_671 = None
    slice_scatter_673: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_672, 0, 0, 9223372036854775807);  slice_scatter_672 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_183: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_182, slice_scatter_673);  add_182 = slice_scatter_673 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_1151: "f32[12, 3, 513, 512]" = torch.ops.aten.view.default(add_183, [12, 3, 513, 512]);  add_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_62: "f32[12, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1151, [0, 0, 0, -1]);  view_1151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    view_1152: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_62, [12, 3, 512, 512, 1]);  constant_pad_nd_62 = None
    permute_1014: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1152, [0, 1, 2, 4, 3]);  view_1152 = None
    view_1153: "f32[36, 512, 512]" = torch.ops.aten.view.default(permute_1014, [36, 512, 512]);  permute_1014 = None
    bmm_42: "f32[36, 64, 512]" = torch.ops.aten.bmm.default(permute_1015, view_1153);  permute_1015 = None
    bmm_43: "f32[36, 512, 64]" = torch.ops.aten.bmm.default(view_1153, permute_1016);  view_1153 = permute_1016 = None
    view_1154: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_42, [12, 3, 64, 512, 1]);  bmm_42 = None
    permute_1017: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1154, [0, 1, 4, 3, 2]);  view_1154 = None
    view_1155: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_43, [12, 3, 512, 64, 1]);  bmm_43 = None
    permute_1018: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(view_1155, [0, 1, 2, 4, 3]);  view_1155 = None
    permute_1019: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1017, [0, 1, 3, 4, 2]);  permute_1017 = None
    squeeze_36: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1019, 4);  permute_1019 = None
    permute_1020: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1018, [0, 1, 2, 4, 3]);  permute_1018 = None
    squeeze_37: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1020, 4);  permute_1020 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    clone_141: "f32[12, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_36, memory_format = torch.contiguous_format);  squeeze_36 = None
    view_1156: "f32[1179648]" = torch.ops.aten.view.default(clone_141, [1179648]);  clone_141 = None
    index_put_13: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1156, True);  view_1156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    view_1159: "f32[1179648]" = torch.ops.aten.view.default(squeeze_37, [-1]);  squeeze_37 = None
    index_put_14: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1159, True);  view_1159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    as_strided_287: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_14, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_14 = None
    view_1166: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_287, [12, 1024, 64]);  as_strided_287 = None
    view_1167: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1166, [1, 12, 1024, 64]);  view_1166 = None
    permute_1025: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1167, [0, 2, 1, 3]);  view_1167 = None
    permute_1026: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1025, [1, 0, 2, 3]);  permute_1025 = None
    view_1168: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1026, [1024, 1, 768]);  permute_1026 = None
    squeeze_39: "f32[1024, 768]" = torch.ops.aten.squeeze.dim(view_1168, 1);  view_1168 = None
    copy_231: "f32[1024, 768]" = torch.ops.aten.copy.default(as_strided_108, squeeze_39);  squeeze_39 = None
    as_strided_scatter_48: "f32[786432]" = torch.ops.aten.as_strided_scatter.default(full_default_161, copy_231, [1024, 768], [768, 1], 0);  copy_231 = None
    as_strided_291: "f32[1024, 768]" = torch.ops.aten.as_strided.default(as_strided_scatter_48, [1024, 768], [768, 1], 0);  as_strided_scatter_48 = None
    new_empty_strided_39: "f32[1024, 768]" = torch.ops.aten.new_empty_strided.default(as_strided_291, [1024, 768], [768, 1])
    copy_232: "f32[1024, 768]" = torch.ops.aten.copy.default(new_empty_strided_39, as_strided_291);  new_empty_strided_39 = as_strided_291 = None
    as_strided_293: "f32[1024, 1, 768]" = torch.ops.aten.as_strided.default(copy_232, [1024, 1, 768], [768, 768, 1], 0)
    clone_144: "f32[1024, 1, 768]" = torch.ops.aten.clone.default(as_strided_293, memory_format = torch.contiguous_format)
    div_134: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(clone_144, 8.0);  clone_144 = None
    copy_233: "f32[1024, 1, 768]" = torch.ops.aten.copy.default(as_strided_293, div_134);  as_strided_293 = div_134 = None
    as_strided_scatter_49: "f32[1024, 768]" = torch.ops.aten.as_strided_scatter.default(copy_232, copy_233, [1024, 1, 768], [768, 768, 1], 0);  copy_232 = copy_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_1169: "f32[1024, 768]" = torch.ops.aten.view.default(view_1149, [1024, 768]);  view_1149 = None
    mm_54: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1169, permute_1027);  permute_1027 = None
    permute_1028: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1169, [1, 0])
    mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1028, view_525);  permute_1028 = None
    permute_1029: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_85: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1169, [0], True);  view_1169 = None
    view_1170: "f32[768]" = torch.ops.aten.view.default(sum_85, [768]);  sum_85 = None
    permute_1030: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1029, [1, 0]);  permute_1029 = None
    view_1171: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_54, [1024, 1, 768]);  mm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    as_strided_295: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_13, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_13 = None
    view_1173: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_295, [12, 1024, 64]);  as_strided_295 = None
    view_1174: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1173, [1, 12, 1024, 64]);  view_1173 = None
    permute_1032: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1174, [0, 2, 1, 3]);  view_1174 = None
    permute_1033: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1032, [1, 0, 2, 3]);  permute_1032 = None
    view_1175: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1033, [1024, 1, 768]);  permute_1033 = None
    view_1176: "f32[1024, 768]" = torch.ops.aten.view.default(view_1175, [1024, 768]);  view_1175 = None
    mm_56: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1176, permute_1031);  permute_1031 = None
    permute_1037: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1176, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1037, view_525);  permute_1037 = None
    permute_1038: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_86: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1176, [0], True);  view_1176 = None
    view_1181: "f32[768]" = torch.ops.aten.view.default(sum_86, [768]);  sum_86 = None
    permute_1039: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1038, [1, 0]);  permute_1038 = None
    view_1182: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_56, [1024, 1, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    add_184: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(view_1171, view_1182);  view_1171 = view_1182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    mm_58: "f32[1024, 768]" = torch.ops.aten.mm.default(as_strided_scatter_49, permute_1040);  permute_1040 = None
    permute_1042: "f32[768, 1024]" = torch.ops.aten.permute.default(as_strided_scatter_49, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1042, view_525);  permute_1042 = view_525 = None
    permute_1043: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_87: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(as_strided_scatter_49, [0], True);  as_strided_scatter_49 = None
    view_1183: "f32[768]" = torch.ops.aten.view.default(sum_87, [768]);  sum_87 = None
    permute_1044: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1043, [1, 0]);  permute_1043 = None
    view_1184: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_58, [1024, 1, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    add_185: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(add_184, view_1184);  add_184 = view_1184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_1045: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(add_185, [1, 0, 2]);  add_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    add_186: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_233, permute_1045);  mul_233 = permute_1045 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_242: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_186, primals_111);  primals_111 = None
    mul_243: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_242, 768)
    sum_88: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_242, [2], True)
    mul_244: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_242, mul_54);  mul_242 = None
    sum_89: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_244, [2], True);  mul_244 = None
    mul_245: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_54, sum_89);  sum_89 = None
    sub_132: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_243, sum_88);  mul_243 = sum_88 = None
    sub_133: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_132, mul_245);  sub_132 = mul_245 = None
    mul_246: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_135, sub_133);  div_135 = sub_133 = None
    mul_247: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_186, mul_54);  mul_54 = None
    sum_90: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_247, [0, 1]);  mul_247 = None
    sum_91: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_186, [0, 1]);  add_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_75: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_67, torch.float32);  getitem_67 = None
    mul_248: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 1.1111111111111112);  convert_element_type_75 = None
    mul_249: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_246, mul_248);  mul_248 = None
    clone_145: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_249, memory_format = torch.contiguous_format);  mul_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_1185: "f32[1024, 768]" = torch.ops.aten.view.default(clone_145, [1024, 768]);  clone_145 = None
    mm_60: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_1185, permute_1046);  permute_1046 = None
    permute_1047: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1185, [1, 0])
    mm_61: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1047, view_523);  permute_1047 = view_523 = None
    permute_1048: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_92: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1185, [0], True);  view_1185 = None
    view_1186: "f32[768]" = torch.ops.aten.view.default(sum_92, [768]);  sum_92 = None
    permute_1049: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1048, [1, 0]);  permute_1048 = None
    view_1187: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_60, [1, 1024, 3072]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_251: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_73, 0.5);  add_73 = None
    mul_252: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_522, view_522)
    mul_253: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_252, -0.5);  mul_252 = None
    exp_17: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_253);  mul_253 = None
    mul_254: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_255: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_522, mul_254);  view_522 = mul_254 = None
    add_188: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_251, mul_255);  mul_251 = mul_255 = None
    mul_256: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1187, add_188);  view_1187 = add_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_1188: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_256, [1024, 3072]);  mul_256 = None
    mm_62: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1188, permute_1050);  permute_1050 = None
    permute_1051: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_1188, [1, 0])
    mm_63: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1051, view_521);  permute_1051 = view_521 = None
    permute_1052: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_93: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1188, [0], True);  view_1188 = None
    view_1189: "f32[3072]" = torch.ops.aten.view.default(sum_93, [3072]);  sum_93 = None
    permute_1053: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1052, [1, 0]);  permute_1052 = None
    view_1190: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_62, [1, 1024, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    add_189: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_246, view_1190);  mul_246 = view_1190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_258: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_189, primals_105);  primals_105 = None
    mul_259: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_258, 768)
    sum_94: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_258, [2], True)
    mul_260: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_258, mul_49);  mul_258 = None
    sum_95: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_260, [2], True);  mul_260 = None
    mul_261: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_49, sum_95);  sum_95 = None
    sub_135: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_259, sum_94);  mul_259 = sum_94 = None
    sub_136: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_135, mul_261);  sub_135 = mul_261 = None
    mul_262: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_136, sub_136);  div_136 = sub_136 = None
    mul_263: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_189, mul_49);  mul_49 = None
    sum_96: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_263, [0, 1]);  mul_263 = None
    sum_97: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_189, [0, 1]);  add_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_76: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_63, torch.float32);  getitem_63 = None
    mul_264: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_76, 1.1111111111111112);  convert_element_type_76 = None
    mul_265: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_262, mul_264);  mul_264 = None
    clone_146: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_265, memory_format = torch.contiguous_format);  mul_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_1191: "f32[1024, 768]" = torch.ops.aten.view.default(clone_146, [1024, 768]);  clone_146 = None
    mm_64: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1191, permute_1054);  permute_1054 = None
    permute_1055: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1191, [1, 0])
    mm_65: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1055, view_519);  permute_1055 = view_519 = None
    permute_1056: "f32[768, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_98: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1191, [0], True);  view_1191 = None
    view_1192: "f32[768]" = torch.ops.aten.view.default(sum_98, [768]);  sum_98 = None
    permute_1057: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1056, [1, 0]);  permute_1056 = None
    view_1193: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_64, [1, 1024, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_1058: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(view_1193, [1, 0, 2]);  view_1193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    view_1194: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(permute_1058, [1024, 1, 12, 64]);  permute_1058 = None
    permute_1059: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1194, [1, 0, 2, 3]);  view_1194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    permute_1060: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1059, [0, 2, 1, 3]);  permute_1059 = None
    view_1195: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_1060, [12, 4, 256, 64]);  permute_1060 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    view_1196: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_1195, [12, 4, 256, 64, 1]);  view_1195 = None
    permute_1061: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1196, [0, 1, 2, 4, 3]);  view_1196 = None
    clone_147: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.clone.default(permute_1061, memory_format = torch.contiguous_format);  permute_1061 = None
    view_1197: "f32[48, 256, 64]" = torch.ops.aten.view.default(clone_147, [48, 256, 64]);  clone_147 = None
    bmm_44: "f32[48, 768, 64]" = torch.ops.aten.bmm.default(permute_1062, view_1197);  permute_1062 = None
    bmm_45: "f32[48, 256, 768]" = torch.ops.aten.bmm.default(view_1197, permute_1063);  view_1197 = permute_1063 = None
    view_1198: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_44, [12, 4, 768, 64, 1]);  bmm_44 = None
    permute_1064: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(view_1198, [0, 1, 4, 3, 2]);  view_1198 = None
    view_1199: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_45, [12, 4, 256, 768, 1]);  bmm_45 = None
    permute_1065: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(view_1199, [0, 1, 2, 4, 3]);  view_1199 = None
    permute_1066: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_1064, [0, 1, 4, 3, 2]);  permute_1064 = None
    squeeze_40: "f32[12, 4, 768, 64]" = torch.ops.aten.squeeze.dim(permute_1066, 4);  permute_1066 = None
    permute_1067: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_1065, [0, 1, 2, 4, 3]);  permute_1065 = None
    squeeze_41: "f32[12, 4, 256, 768]" = torch.ops.aten.squeeze.dim(permute_1067, 4);  permute_1067 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_scatter_674: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, squeeze_41, 3, 0, -1);  squeeze_41 = None
    slice_scatter_675: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_674, 2, 0, 9223372036854775807);  slice_scatter_674 = None
    slice_scatter_676: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_675, 1, 0, 9223372036854775807);  slice_scatter_675 = None
    slice_scatter_677: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_676, 0, 0, 9223372036854775807);  slice_scatter_676 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1200: "f32[12, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_677, [12, 4, 196864]);  slice_scatter_677 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_scatter_678: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, view_1200, 2, 0, -256);  view_1200 = None
    slice_scatter_679: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_678, 1, 0, 9223372036854775807);  slice_scatter_678 = None
    slice_scatter_680: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_679, 0, 0, 9223372036854775807);  slice_scatter_679 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1201: "f32[12, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_680, [12, 4, 256, 770]);  slice_scatter_680 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_63: "f32[12, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1201, [0, -257]);  view_1201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    view_1202: "f32[2359296]" = torch.ops.aten.view.default(squeeze_40, [-1]);  squeeze_40 = None
    index_put_15: "f32[1179648]" = torch.ops.aten.index_put.default(full_default_128, [view_918], view_1202, True);  view_1202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    as_strided_299: "f32[12, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_15, [12, 1536, 64], [98304, 64, 1], 0);  index_put_15 = None
    constant_pad_nd_64: "f32[12, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_299, [0, 0, -256, -256]);  as_strided_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    view_1204: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_64, [1, 12, 1024, 64]);  constant_pad_nd_64 = None
    permute_1068: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1204, [0, 2, 1, 3]);  view_1204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    view_1205: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_63, [1, 12, 1024, 513]);  constant_pad_nd_63 = None
    permute_1069: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1205, [0, 2, 1, 3]);  view_1205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    permute_1070: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1068, [1, 0, 2, 3]);  permute_1068 = None
    clone_149: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_1070, memory_format = torch.contiguous_format);  permute_1070 = None
    view_1206: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_149, [1024, 1, 768]);  clone_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    convert_element_type_77: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(getitem_61, torch.float32);  getitem_61 = None
    mul_266: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_77, 1.1111111111111112);  convert_element_type_77 = None
    mul_267: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1069, mul_266);  permute_1069 = mul_266 = None
    clone_150: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_267, memory_format = torch.contiguous_format);  mul_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_111: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, clone_150);  clone_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    mul_268: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_111, alias_17);  where_111 = None
    sum_99: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_268, [-1], True)
    mul_269: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(alias_17, sum_99);  alias_17 = sum_99 = None
    sub_137: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(mul_268, mul_269);  mul_268 = mul_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    squeeze_42: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(sub_137, 0);  sub_137 = None
    copy_234: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_42);  squeeze_42 = None
    as_strided_scatter_50: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_234, [1024, 12, 513], [513, 525312, 1], 0);  copy_234 = None
    as_strided_303: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_50, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_50 = None
    new_empty_strided_40: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_303, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_235: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_40, as_strided_303);  new_empty_strided_40 = as_strided_303 = None
    as_strided_305: "f32[1, 1024, 12, 513]" = torch.ops.aten.as_strided.default(copy_235, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0)
    clone_151: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(as_strided_305, memory_format = torch.contiguous_format)
    copy_236: "f32[1, 1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_305, clone_151);  as_strided_305 = clone_151 = None
    as_strided_scatter_51: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_235, copy_236, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0);  copy_235 = copy_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    new_empty_strided_41: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_51, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_237: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_41, as_strided_scatter_51);  new_empty_strided_41 = as_strided_scatter_51 = None
    as_strided_308: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_237, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240)
    clone_152: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_308, memory_format = torch.contiguous_format)
    copy_238: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_308, full_default_130);  as_strided_308 = None
    as_strided_scatter_52: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_237, copy_238, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240);  copy_237 = copy_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_112: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_152);  clone_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    slice_scatter_681: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_112, 3, -257, 9223372036854775807);  where_112 = None
    slice_scatter_682: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_681, 2, 0, 9223372036854775807);  slice_scatter_681 = None
    slice_scatter_683: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_682, 1, -256, 9223372036854775807);  slice_scatter_682 = None
    slice_scatter_684: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_683, 0, 0, 9223372036854775807);  slice_scatter_683 = None
    squeeze_43: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(slice_scatter_684, 0);  slice_scatter_684 = None
    copy_239: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_43);  squeeze_43 = None
    as_strided_scatter_53: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_239, [1024, 12, 513], [513, 525312, 1], 0);  copy_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    as_strided_313: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_53, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_53 = None
    add_190: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_52, as_strided_313);  as_strided_scatter_52 = as_strided_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    new_empty_strided_42: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_190, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_240: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_42, add_190);  new_empty_strided_42 = add_190 = None
    as_strided_315: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_240, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0)
    clone_153: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_315, memory_format = torch.contiguous_format)
    copy_241: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_315, full_default_130);  as_strided_315 = None
    as_strided_scatter_54: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_240, copy_241, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0);  copy_240 = copy_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_113: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_153);  clone_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    slice_scatter_685: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_113, 3, 0, 257);  where_113 = None
    slice_scatter_686: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_685, 2, 0, 9223372036854775807);  slice_scatter_685 = None
    slice_scatter_687: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_686, 1, 0, 256);  slice_scatter_686 = None
    slice_scatter_688: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_687, 0, 0, 9223372036854775807);  slice_scatter_687 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    permute_1071: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_688, [0, 2, 1, 3]);  slice_scatter_688 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_1207: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_1071, [12, 4, 256, 513]);  permute_1071 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    add_191: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_54, view_1207);  as_strided_scatter_54 = view_1207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_43: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_191, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_242: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_43, add_191);  new_empty_strided_43 = add_191 = None
    as_strided_318: "f32[12, 255, 255]" = torch.ops.aten.as_strided.default(copy_242, [12, 255, 255], [525312, 513, 1], 514)
    clone_154: "f32[12, 255, 255]" = torch.ops.aten.clone.default(as_strided_318, memory_format = torch.contiguous_format)
    copy_243: "f32[12, 255, 255]" = torch.ops.aten.copy.default(as_strided_318, full_default_142);  as_strided_318 = None
    as_strided_scatter_55: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_242, copy_243, [12, 255, 255], [525312, 513, 1], 514);  copy_242 = copy_243 = None
    slice_scatter_689: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_143, clone_154, 2, -255, 9223372036854775807);  clone_154 = None
    slice_scatter_690: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_689, 1, 0, 255);  slice_scatter_689 = None
    select_scatter_58: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_690, 1, 0);  slice_scatter_690 = None
    slice_scatter_691: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_58, 0, 0, 9223372036854775807);  select_scatter_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_44: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_55, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_244: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_44, as_strided_scatter_55);  new_empty_strided_44 = as_strided_scatter_55 = None
    as_strided_321: "f32[12, 3, 256, 256]" = torch.ops.aten.as_strided.default(copy_244, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328)
    clone_155: "f32[12, 3, 256, 256]" = torch.ops.aten.clone.default(as_strided_321, memory_format = torch.contiguous_format)
    copy_245: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(as_strided_321, full_default_147);  as_strided_321 = None
    as_strided_scatter_56: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_244, copy_245, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328);  copy_244 = copy_245 = None
    slice_scatter_692: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_155, 3, 257, 9223372036854775807);  clone_155 = None
    slice_scatter_693: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_692, 2, -257, -1);  slice_scatter_692 = None
    slice_scatter_694: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_693, 1, 0, 9223372036854775807);  slice_scatter_693 = None
    slice_scatter_695: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_694, 0, 0, 9223372036854775807);  slice_scatter_694 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    add_192: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(slice_scatter_691, slice_scatter_695);  slice_scatter_691 = slice_scatter_695 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_45: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_56, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_246: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_45, as_strided_scatter_56);  new_empty_strided_45 = as_strided_scatter_56 = None
    as_strided_324: "f32[12, 256, 257]" = torch.ops.aten.as_strided.default(copy_246, [12, 256, 257], [525312, 513, 1], 394240)
    clone_156: "f32[12, 256, 257]" = torch.ops.aten.clone.default(as_strided_324, memory_format = torch.contiguous_format)
    copy_247: "f32[12, 256, 257]" = torch.ops.aten.copy.default(as_strided_324, full_default_152);  as_strided_324 = None
    as_strided_scatter_57: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_246, copy_247, [12, 256, 257], [525312, 513, 1], 394240);  copy_246 = copy_247 = None
    slice_scatter_696: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_153, clone_156, 2, 0, 257);  clone_156 = None
    slice_scatter_697: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_696, 1, 256, 9223372036854775807);  slice_scatter_696 = None
    select_scatter_59: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_697, 1, -1);  slice_scatter_697 = None
    slice_scatter_698: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_59, 0, 0, 9223372036854775807);  select_scatter_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_193: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_192, slice_scatter_698);  add_192 = slice_scatter_698 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_46: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_57, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_248: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_46, as_strided_scatter_57);  new_empty_strided_46 = as_strided_scatter_57 = None
    as_strided_327: "f32[12, 3, 256, 257]" = torch.ops.aten.as_strided.default(copy_248, [12, 3, 256, 257], [525312, 131328, 513, 1], 256);  copy_248 = None
    clone_157: "f32[12, 3, 256, 257]" = torch.ops.aten.clone.default(as_strided_327, memory_format = torch.contiguous_format);  as_strided_327 = None
    slice_scatter_699: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_157, 3, 0, 257);  clone_157 = None
    slice_scatter_700: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_699, 2, 0, 256);  slice_scatter_699 = None
    slice_scatter_701: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_700, 1, 0, 9223372036854775807);  slice_scatter_700 = None
    slice_scatter_702: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_701, 0, 0, 9223372036854775807);  slice_scatter_701 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_194: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_193, slice_scatter_702);  add_193 = slice_scatter_702 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_1208: "f32[12, 3, 513, 512]" = torch.ops.aten.view.default(add_194, [12, 3, 513, 512]);  add_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_65: "f32[12, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1208, [0, 0, 0, -1]);  view_1208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    view_1209: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_65, [12, 3, 512, 512, 1]);  constant_pad_nd_65 = None
    permute_1072: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1209, [0, 1, 2, 4, 3]);  view_1209 = None
    view_1210: "f32[36, 512, 512]" = torch.ops.aten.view.default(permute_1072, [36, 512, 512]);  permute_1072 = None
    bmm_46: "f32[36, 64, 512]" = torch.ops.aten.bmm.default(permute_1073, view_1210);  permute_1073 = None
    bmm_47: "f32[36, 512, 64]" = torch.ops.aten.bmm.default(view_1210, permute_1074);  view_1210 = permute_1074 = None
    view_1211: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_46, [12, 3, 64, 512, 1]);  bmm_46 = None
    permute_1075: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1211, [0, 1, 4, 3, 2]);  view_1211 = None
    view_1212: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_47, [12, 3, 512, 64, 1]);  bmm_47 = None
    permute_1076: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(view_1212, [0, 1, 2, 4, 3]);  view_1212 = None
    permute_1077: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1075, [0, 1, 3, 4, 2]);  permute_1075 = None
    squeeze_44: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1077, 4);  permute_1077 = None
    permute_1078: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1076, [0, 1, 2, 4, 3]);  permute_1076 = None
    squeeze_45: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1078, 4);  permute_1078 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    clone_158: "f32[12, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_44, memory_format = torch.contiguous_format);  squeeze_44 = None
    view_1213: "f32[1179648]" = torch.ops.aten.view.default(clone_158, [1179648]);  clone_158 = None
    index_put_16: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1213, True);  view_1213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    view_1216: "f32[1179648]" = torch.ops.aten.view.default(squeeze_45, [-1]);  squeeze_45 = None
    index_put_17: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1216, True);  view_1216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    as_strided_332: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_17, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_17 = None
    view_1223: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_332, [12, 1024, 64]);  as_strided_332 = None
    view_1224: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1223, [1, 12, 1024, 64]);  view_1223 = None
    permute_1083: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1224, [0, 2, 1, 3]);  view_1224 = None
    permute_1084: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1083, [1, 0, 2, 3]);  permute_1083 = None
    view_1225: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1084, [1024, 1, 768]);  permute_1084 = None
    squeeze_47: "f32[1024, 768]" = torch.ops.aten.squeeze.dim(view_1225, 1);  view_1225 = None
    copy_249: "f32[1024, 768]" = torch.ops.aten.copy.default(as_strided_108, squeeze_47);  squeeze_47 = None
    as_strided_scatter_58: "f32[786432]" = torch.ops.aten.as_strided_scatter.default(full_default_161, copy_249, [1024, 768], [768, 1], 0);  copy_249 = None
    as_strided_336: "f32[1024, 768]" = torch.ops.aten.as_strided.default(as_strided_scatter_58, [1024, 768], [768, 1], 0);  as_strided_scatter_58 = None
    new_empty_strided_47: "f32[1024, 768]" = torch.ops.aten.new_empty_strided.default(as_strided_336, [1024, 768], [768, 1])
    copy_250: "f32[1024, 768]" = torch.ops.aten.copy.default(new_empty_strided_47, as_strided_336);  new_empty_strided_47 = as_strided_336 = None
    as_strided_338: "f32[1024, 1, 768]" = torch.ops.aten.as_strided.default(copy_250, [1024, 1, 768], [768, 768, 1], 0)
    clone_161: "f32[1024, 1, 768]" = torch.ops.aten.clone.default(as_strided_338, memory_format = torch.contiguous_format)
    div_137: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(clone_161, 8.0);  clone_161 = None
    copy_251: "f32[1024, 1, 768]" = torch.ops.aten.copy.default(as_strided_338, div_137);  as_strided_338 = div_137 = None
    as_strided_scatter_59: "f32[1024, 768]" = torch.ops.aten.as_strided_scatter.default(copy_250, copy_251, [1024, 1, 768], [768, 768, 1], 0);  copy_250 = copy_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_1226: "f32[1024, 768]" = torch.ops.aten.view.default(view_1206, [1024, 768]);  view_1206 = None
    mm_66: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1226, permute_1085);  permute_1085 = None
    permute_1086: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1226, [1, 0])
    mm_67: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1086, view_450);  permute_1086 = None
    permute_1087: "f32[768, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_100: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1226, [0], True);  view_1226 = None
    view_1227: "f32[768]" = torch.ops.aten.view.default(sum_100, [768]);  sum_100 = None
    permute_1088: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1087, [1, 0]);  permute_1087 = None
    view_1228: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_66, [1024, 1, 768]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    as_strided_340: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_16, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_16 = None
    view_1230: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_340, [12, 1024, 64]);  as_strided_340 = None
    view_1231: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1230, [1, 12, 1024, 64]);  view_1230 = None
    permute_1090: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1231, [0, 2, 1, 3]);  view_1231 = None
    permute_1091: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1090, [1, 0, 2, 3]);  permute_1090 = None
    view_1232: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1091, [1024, 1, 768]);  permute_1091 = None
    view_1233: "f32[1024, 768]" = torch.ops.aten.view.default(view_1232, [1024, 768]);  view_1232 = None
    mm_68: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1233, permute_1089);  permute_1089 = None
    permute_1095: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1233, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1095, view_450);  permute_1095 = None
    permute_1096: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_101: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1233, [0], True);  view_1233 = None
    view_1238: "f32[768]" = torch.ops.aten.view.default(sum_101, [768]);  sum_101 = None
    permute_1097: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1096, [1, 0]);  permute_1096 = None
    view_1239: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_68, [1024, 1, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    add_195: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(view_1228, view_1239);  view_1228 = view_1239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    mm_70: "f32[1024, 768]" = torch.ops.aten.mm.default(as_strided_scatter_59, permute_1098);  permute_1098 = None
    permute_1100: "f32[768, 1024]" = torch.ops.aten.permute.default(as_strided_scatter_59, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1100, view_450);  permute_1100 = view_450 = None
    permute_1101: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_102: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(as_strided_scatter_59, [0], True);  as_strided_scatter_59 = None
    view_1240: "f32[768]" = torch.ops.aten.view.default(sum_102, [768]);  sum_102 = None
    permute_1102: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1101, [1, 0]);  permute_1101 = None
    view_1241: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_70, [1024, 1, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    add_196: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(add_195, view_1241);  add_195 = view_1241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_1103: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(add_196, [1, 0, 2]);  add_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    add_197: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_262, permute_1103);  mul_262 = permute_1103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_271: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_197, primals_95);  primals_95 = None
    mul_272: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_271, 768)
    sum_103: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True)
    mul_273: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_271, mul_46);  mul_271 = None
    sum_104: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_273, [2], True);  mul_273 = None
    mul_274: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_46, sum_104);  sum_104 = None
    sub_139: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_272, sum_103);  mul_272 = sum_103 = None
    sub_140: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_139, mul_274);  sub_139 = mul_274 = None
    mul_275: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_138, sub_140);  div_138 = sub_140 = None
    mul_276: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_197, mul_46);  mul_46 = None
    sum_105: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_276, [0, 1]);  mul_276 = None
    sum_106: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_197, [0, 1]);  add_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_78: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_57, torch.float32);  getitem_57 = None
    mul_277: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 1.1111111111111112);  convert_element_type_78 = None
    mul_278: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_275, mul_277);  mul_277 = None
    clone_162: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_278, memory_format = torch.contiguous_format);  mul_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_1242: "f32[1024, 768]" = torch.ops.aten.view.default(clone_162, [1024, 768]);  clone_162 = None
    mm_72: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_1242, permute_1104);  permute_1104 = None
    permute_1105: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1242, [1, 0])
    mm_73: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1105, view_448);  permute_1105 = view_448 = None
    permute_1106: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_107: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1242, [0], True);  view_1242 = None
    view_1243: "f32[768]" = torch.ops.aten.view.default(sum_107, [768]);  sum_107 = None
    permute_1107: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1106, [1, 0]);  permute_1106 = None
    view_1244: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_72, [1, 1024, 3072]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_280: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_62, 0.5);  add_62 = None
    mul_281: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_447, view_447)
    mul_282: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_281, -0.5);  mul_281 = None
    exp_18: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_282);  mul_282 = None
    mul_283: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_284: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_447, mul_283);  view_447 = mul_283 = None
    add_199: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_280, mul_284);  mul_280 = mul_284 = None
    mul_285: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1244, add_199);  view_1244 = add_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_1245: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_285, [1024, 3072]);  mul_285 = None
    mm_74: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1245, permute_1108);  permute_1108 = None
    permute_1109: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_1245, [1, 0])
    mm_75: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1109, view_446);  permute_1109 = view_446 = None
    permute_1110: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_108: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1245, [0], True);  view_1245 = None
    view_1246: "f32[3072]" = torch.ops.aten.view.default(sum_108, [3072]);  sum_108 = None
    permute_1111: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1110, [1, 0]);  permute_1110 = None
    view_1247: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_74, [1, 1024, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    add_200: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_275, view_1247);  mul_275 = view_1247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_287: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_200, primals_89);  primals_89 = None
    mul_288: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_287, 768)
    sum_109: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_287, [2], True)
    mul_289: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_287, mul_41);  mul_287 = None
    sum_110: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_289, [2], True);  mul_289 = None
    mul_290: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_41, sum_110);  sum_110 = None
    sub_142: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_288, sum_109);  mul_288 = sum_109 = None
    sub_143: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_142, mul_290);  sub_142 = mul_290 = None
    mul_291: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_139, sub_143);  div_139 = sub_143 = None
    mul_292: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_200, mul_41);  mul_41 = None
    sum_111: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_292, [0, 1]);  mul_292 = None
    sum_112: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_200, [0, 1]);  add_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_79: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_53, torch.float32);  getitem_53 = None
    mul_293: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_79, 1.1111111111111112);  convert_element_type_79 = None
    mul_294: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_291, mul_293);  mul_293 = None
    clone_163: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_294, memory_format = torch.contiguous_format);  mul_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_1248: "f32[1024, 768]" = torch.ops.aten.view.default(clone_163, [1024, 768]);  clone_163 = None
    mm_76: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1248, permute_1112);  permute_1112 = None
    permute_1113: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1248, [1, 0])
    mm_77: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1113, view_444);  permute_1113 = view_444 = None
    permute_1114: "f32[768, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1248, [0], True);  view_1248 = None
    view_1249: "f32[768]" = torch.ops.aten.view.default(sum_113, [768]);  sum_113 = None
    permute_1115: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1114, [1, 0]);  permute_1114 = None
    view_1250: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_76, [1, 1024, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_1116: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(view_1250, [1, 0, 2]);  view_1250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    view_1251: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(permute_1116, [1024, 1, 12, 64]);  permute_1116 = None
    permute_1117: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1251, [1, 0, 2, 3]);  view_1251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    permute_1118: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1117, [0, 2, 1, 3]);  permute_1117 = None
    view_1252: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_1118, [12, 4, 256, 64]);  permute_1118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    view_1253: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_1252, [12, 4, 256, 64, 1]);  view_1252 = None
    permute_1119: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1253, [0, 1, 2, 4, 3]);  view_1253 = None
    clone_164: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.clone.default(permute_1119, memory_format = torch.contiguous_format);  permute_1119 = None
    view_1254: "f32[48, 256, 64]" = torch.ops.aten.view.default(clone_164, [48, 256, 64]);  clone_164 = None
    bmm_48: "f32[48, 768, 64]" = torch.ops.aten.bmm.default(permute_1120, view_1254);  permute_1120 = None
    bmm_49: "f32[48, 256, 768]" = torch.ops.aten.bmm.default(view_1254, permute_1121);  view_1254 = permute_1121 = None
    view_1255: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_48, [12, 4, 768, 64, 1]);  bmm_48 = None
    permute_1122: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(view_1255, [0, 1, 4, 3, 2]);  view_1255 = None
    view_1256: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_49, [12, 4, 256, 768, 1]);  bmm_49 = None
    permute_1123: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(view_1256, [0, 1, 2, 4, 3]);  view_1256 = None
    permute_1124: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_1122, [0, 1, 4, 3, 2]);  permute_1122 = None
    squeeze_48: "f32[12, 4, 768, 64]" = torch.ops.aten.squeeze.dim(permute_1124, 4);  permute_1124 = None
    permute_1125: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_1123, [0, 1, 2, 4, 3]);  permute_1123 = None
    squeeze_49: "f32[12, 4, 256, 768]" = torch.ops.aten.squeeze.dim(permute_1125, 4);  permute_1125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_scatter_703: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, squeeze_49, 3, 0, -1);  squeeze_49 = None
    slice_scatter_704: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_703, 2, 0, 9223372036854775807);  slice_scatter_703 = None
    slice_scatter_705: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_704, 1, 0, 9223372036854775807);  slice_scatter_704 = None
    slice_scatter_706: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_705, 0, 0, 9223372036854775807);  slice_scatter_705 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1257: "f32[12, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_706, [12, 4, 196864]);  slice_scatter_706 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_scatter_707: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, view_1257, 2, 0, -256);  view_1257 = None
    slice_scatter_708: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_707, 1, 0, 9223372036854775807);  slice_scatter_707 = None
    slice_scatter_709: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_708, 0, 0, 9223372036854775807);  slice_scatter_708 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1258: "f32[12, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_709, [12, 4, 256, 770]);  slice_scatter_709 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_66: "f32[12, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1258, [0, -257]);  view_1258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    view_1259: "f32[2359296]" = torch.ops.aten.view.default(squeeze_48, [-1]);  squeeze_48 = None
    index_put_18: "f32[1179648]" = torch.ops.aten.index_put.default(full_default_128, [view_918], view_1259, True);  view_1259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    as_strided_344: "f32[12, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_18, [12, 1536, 64], [98304, 64, 1], 0);  index_put_18 = None
    constant_pad_nd_67: "f32[12, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_344, [0, 0, -256, -256]);  as_strided_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    view_1261: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_67, [1, 12, 1024, 64]);  constant_pad_nd_67 = None
    permute_1126: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1261, [0, 2, 1, 3]);  view_1261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    view_1262: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_66, [1, 12, 1024, 513]);  constant_pad_nd_66 = None
    permute_1127: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1262, [0, 2, 1, 3]);  view_1262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    permute_1128: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1126, [1, 0, 2, 3]);  permute_1126 = None
    clone_166: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_1128, memory_format = torch.contiguous_format);  permute_1128 = None
    view_1263: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_166, [1024, 1, 768]);  clone_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    convert_element_type_80: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(getitem_51, torch.float32);  getitem_51 = None
    mul_295: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 1.1111111111111112);  convert_element_type_80 = None
    mul_296: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1127, mul_295);  permute_1127 = mul_295 = None
    clone_167: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_296, memory_format = torch.contiguous_format);  mul_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_114: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, clone_167);  clone_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    mul_297: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_114, alias_18);  where_114 = None
    sum_114: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_297, [-1], True)
    mul_298: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(alias_18, sum_114);  alias_18 = sum_114 = None
    sub_144: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(mul_297, mul_298);  mul_297 = mul_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    squeeze_50: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(sub_144, 0);  sub_144 = None
    copy_252: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_50);  squeeze_50 = None
    as_strided_scatter_60: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_252, [1024, 12, 513], [513, 525312, 1], 0);  copy_252 = None
    as_strided_348: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_60, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_60 = None
    new_empty_strided_48: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_348, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_253: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_48, as_strided_348);  new_empty_strided_48 = as_strided_348 = None
    as_strided_350: "f32[1, 1024, 12, 513]" = torch.ops.aten.as_strided.default(copy_253, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0)
    clone_168: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(as_strided_350, memory_format = torch.contiguous_format)
    copy_254: "f32[1, 1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_350, clone_168);  as_strided_350 = clone_168 = None
    as_strided_scatter_61: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_253, copy_254, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0);  copy_253 = copy_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    new_empty_strided_49: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_61, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_255: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_49, as_strided_scatter_61);  new_empty_strided_49 = as_strided_scatter_61 = None
    as_strided_353: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_255, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240)
    clone_169: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_353, memory_format = torch.contiguous_format)
    copy_256: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_353, full_default_130);  as_strided_353 = None
    as_strided_scatter_62: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_255, copy_256, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240);  copy_255 = copy_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_115: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_169);  clone_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    slice_scatter_710: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_115, 3, -257, 9223372036854775807);  where_115 = None
    slice_scatter_711: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_710, 2, 0, 9223372036854775807);  slice_scatter_710 = None
    slice_scatter_712: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_711, 1, -256, 9223372036854775807);  slice_scatter_711 = None
    slice_scatter_713: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_712, 0, 0, 9223372036854775807);  slice_scatter_712 = None
    squeeze_51: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(slice_scatter_713, 0);  slice_scatter_713 = None
    copy_257: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_51);  squeeze_51 = None
    as_strided_scatter_63: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_257, [1024, 12, 513], [513, 525312, 1], 0);  copy_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    as_strided_358: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_63, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_63 = None
    add_201: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_62, as_strided_358);  as_strided_scatter_62 = as_strided_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    new_empty_strided_50: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_201, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_258: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_50, add_201);  new_empty_strided_50 = add_201 = None
    as_strided_360: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_258, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0)
    clone_170: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_360, memory_format = torch.contiguous_format)
    copy_259: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_360, full_default_130);  as_strided_360 = None
    as_strided_scatter_64: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_258, copy_259, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0);  copy_258 = copy_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_116: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_170);  clone_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    slice_scatter_714: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_116, 3, 0, 257);  where_116 = None
    slice_scatter_715: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_714, 2, 0, 9223372036854775807);  slice_scatter_714 = None
    slice_scatter_716: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_715, 1, 0, 256);  slice_scatter_715 = None
    slice_scatter_717: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_716, 0, 0, 9223372036854775807);  slice_scatter_716 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    permute_1129: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_717, [0, 2, 1, 3]);  slice_scatter_717 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_1264: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_1129, [12, 4, 256, 513]);  permute_1129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    add_202: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_64, view_1264);  as_strided_scatter_64 = view_1264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_51: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_202, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_260: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_51, add_202);  new_empty_strided_51 = add_202 = None
    as_strided_363: "f32[12, 255, 255]" = torch.ops.aten.as_strided.default(copy_260, [12, 255, 255], [525312, 513, 1], 514)
    clone_171: "f32[12, 255, 255]" = torch.ops.aten.clone.default(as_strided_363, memory_format = torch.contiguous_format)
    copy_261: "f32[12, 255, 255]" = torch.ops.aten.copy.default(as_strided_363, full_default_142);  as_strided_363 = None
    as_strided_scatter_65: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_260, copy_261, [12, 255, 255], [525312, 513, 1], 514);  copy_260 = copy_261 = None
    slice_scatter_718: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_143, clone_171, 2, -255, 9223372036854775807);  clone_171 = None
    slice_scatter_719: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_718, 1, 0, 255);  slice_scatter_718 = None
    select_scatter_60: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_719, 1, 0);  slice_scatter_719 = None
    slice_scatter_720: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_60, 0, 0, 9223372036854775807);  select_scatter_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_52: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_65, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_262: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_52, as_strided_scatter_65);  new_empty_strided_52 = as_strided_scatter_65 = None
    as_strided_366: "f32[12, 3, 256, 256]" = torch.ops.aten.as_strided.default(copy_262, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328)
    clone_172: "f32[12, 3, 256, 256]" = torch.ops.aten.clone.default(as_strided_366, memory_format = torch.contiguous_format)
    copy_263: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(as_strided_366, full_default_147);  as_strided_366 = None
    as_strided_scatter_66: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_262, copy_263, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328);  copy_262 = copy_263 = None
    slice_scatter_721: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_172, 3, 257, 9223372036854775807);  clone_172 = None
    slice_scatter_722: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_721, 2, -257, -1);  slice_scatter_721 = None
    slice_scatter_723: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_722, 1, 0, 9223372036854775807);  slice_scatter_722 = None
    slice_scatter_724: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_723, 0, 0, 9223372036854775807);  slice_scatter_723 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    add_203: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(slice_scatter_720, slice_scatter_724);  slice_scatter_720 = slice_scatter_724 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_53: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_66, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_264: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_53, as_strided_scatter_66);  new_empty_strided_53 = as_strided_scatter_66 = None
    as_strided_369: "f32[12, 256, 257]" = torch.ops.aten.as_strided.default(copy_264, [12, 256, 257], [525312, 513, 1], 394240)
    clone_173: "f32[12, 256, 257]" = torch.ops.aten.clone.default(as_strided_369, memory_format = torch.contiguous_format)
    copy_265: "f32[12, 256, 257]" = torch.ops.aten.copy.default(as_strided_369, full_default_152);  as_strided_369 = None
    as_strided_scatter_67: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_264, copy_265, [12, 256, 257], [525312, 513, 1], 394240);  copy_264 = copy_265 = None
    slice_scatter_725: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_153, clone_173, 2, 0, 257);  clone_173 = None
    slice_scatter_726: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_725, 1, 256, 9223372036854775807);  slice_scatter_725 = None
    select_scatter_61: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_726, 1, -1);  slice_scatter_726 = None
    slice_scatter_727: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_61, 0, 0, 9223372036854775807);  select_scatter_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_204: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_203, slice_scatter_727);  add_203 = slice_scatter_727 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_54: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_67, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_266: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_54, as_strided_scatter_67);  new_empty_strided_54 = as_strided_scatter_67 = None
    as_strided_372: "f32[12, 3, 256, 257]" = torch.ops.aten.as_strided.default(copy_266, [12, 3, 256, 257], [525312, 131328, 513, 1], 256);  copy_266 = None
    clone_174: "f32[12, 3, 256, 257]" = torch.ops.aten.clone.default(as_strided_372, memory_format = torch.contiguous_format);  as_strided_372 = None
    slice_scatter_728: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_174, 3, 0, 257);  clone_174 = None
    slice_scatter_729: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_728, 2, 0, 256);  slice_scatter_728 = None
    slice_scatter_730: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_729, 1, 0, 9223372036854775807);  slice_scatter_729 = None
    slice_scatter_731: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_730, 0, 0, 9223372036854775807);  slice_scatter_730 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_205: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_204, slice_scatter_731);  add_204 = slice_scatter_731 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_1265: "f32[12, 3, 513, 512]" = torch.ops.aten.view.default(add_205, [12, 3, 513, 512]);  add_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_68: "f32[12, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1265, [0, 0, 0, -1]);  view_1265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    view_1266: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_68, [12, 3, 512, 512, 1]);  constant_pad_nd_68 = None
    permute_1130: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1266, [0, 1, 2, 4, 3]);  view_1266 = None
    view_1267: "f32[36, 512, 512]" = torch.ops.aten.view.default(permute_1130, [36, 512, 512]);  permute_1130 = None
    bmm_50: "f32[36, 64, 512]" = torch.ops.aten.bmm.default(permute_1131, view_1267);  permute_1131 = None
    bmm_51: "f32[36, 512, 64]" = torch.ops.aten.bmm.default(view_1267, permute_1132);  view_1267 = permute_1132 = None
    view_1268: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_50, [12, 3, 64, 512, 1]);  bmm_50 = None
    permute_1133: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1268, [0, 1, 4, 3, 2]);  view_1268 = None
    view_1269: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_51, [12, 3, 512, 64, 1]);  bmm_51 = None
    permute_1134: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(view_1269, [0, 1, 2, 4, 3]);  view_1269 = None
    permute_1135: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1133, [0, 1, 3, 4, 2]);  permute_1133 = None
    squeeze_52: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1135, 4);  permute_1135 = None
    permute_1136: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1134, [0, 1, 2, 4, 3]);  permute_1134 = None
    squeeze_53: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1136, 4);  permute_1136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    clone_175: "f32[12, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_52, memory_format = torch.contiguous_format);  squeeze_52 = None
    view_1270: "f32[1179648]" = torch.ops.aten.view.default(clone_175, [1179648]);  clone_175 = None
    index_put_19: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1270, True);  view_1270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    view_1273: "f32[1179648]" = torch.ops.aten.view.default(squeeze_53, [-1]);  squeeze_53 = None
    index_put_20: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1273, True);  view_1273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    as_strided_377: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_20, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_20 = None
    view_1280: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_377, [12, 1024, 64]);  as_strided_377 = None
    view_1281: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1280, [1, 12, 1024, 64]);  view_1280 = None
    permute_1141: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1281, [0, 2, 1, 3]);  view_1281 = None
    permute_1142: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1141, [1, 0, 2, 3]);  permute_1141 = None
    view_1282: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1142, [1024, 1, 768]);  permute_1142 = None
    squeeze_55: "f32[1024, 768]" = torch.ops.aten.squeeze.dim(view_1282, 1);  view_1282 = None
    copy_267: "f32[1024, 768]" = torch.ops.aten.copy.default(as_strided_108, squeeze_55);  squeeze_55 = None
    as_strided_scatter_68: "f32[786432]" = torch.ops.aten.as_strided_scatter.default(full_default_161, copy_267, [1024, 768], [768, 1], 0);  copy_267 = None
    as_strided_381: "f32[1024, 768]" = torch.ops.aten.as_strided.default(as_strided_scatter_68, [1024, 768], [768, 1], 0);  as_strided_scatter_68 = None
    new_empty_strided_55: "f32[1024, 768]" = torch.ops.aten.new_empty_strided.default(as_strided_381, [1024, 768], [768, 1])
    copy_268: "f32[1024, 768]" = torch.ops.aten.copy.default(new_empty_strided_55, as_strided_381);  new_empty_strided_55 = as_strided_381 = None
    as_strided_383: "f32[1024, 1, 768]" = torch.ops.aten.as_strided.default(copy_268, [1024, 1, 768], [768, 768, 1], 0)
    clone_178: "f32[1024, 1, 768]" = torch.ops.aten.clone.default(as_strided_383, memory_format = torch.contiguous_format)
    div_140: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(clone_178, 8.0);  clone_178 = None
    copy_269: "f32[1024, 1, 768]" = torch.ops.aten.copy.default(as_strided_383, div_140);  as_strided_383 = div_140 = None
    as_strided_scatter_69: "f32[1024, 768]" = torch.ops.aten.as_strided_scatter.default(copy_268, copy_269, [1024, 1, 768], [768, 768, 1], 0);  copy_268 = copy_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_1283: "f32[1024, 768]" = torch.ops.aten.view.default(view_1263, [1024, 768]);  view_1263 = None
    mm_78: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1283, permute_1143);  permute_1143 = None
    permute_1144: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1283, [1, 0])
    mm_79: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1144, view_375);  permute_1144 = None
    permute_1145: "f32[768, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_115: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1283, [0], True);  view_1283 = None
    view_1284: "f32[768]" = torch.ops.aten.view.default(sum_115, [768]);  sum_115 = None
    permute_1146: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1145, [1, 0]);  permute_1145 = None
    view_1285: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_78, [1024, 1, 768]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    as_strided_385: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_19, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_19 = None
    view_1287: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_385, [12, 1024, 64]);  as_strided_385 = None
    view_1288: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1287, [1, 12, 1024, 64]);  view_1287 = None
    permute_1148: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1288, [0, 2, 1, 3]);  view_1288 = None
    permute_1149: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1148, [1, 0, 2, 3]);  permute_1148 = None
    view_1289: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1149, [1024, 1, 768]);  permute_1149 = None
    view_1290: "f32[1024, 768]" = torch.ops.aten.view.default(view_1289, [1024, 768]);  view_1289 = None
    mm_80: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1290, permute_1147);  permute_1147 = None
    permute_1153: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1290, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1153, view_375);  permute_1153 = None
    permute_1154: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_116: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1290, [0], True);  view_1290 = None
    view_1295: "f32[768]" = torch.ops.aten.view.default(sum_116, [768]);  sum_116 = None
    permute_1155: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1154, [1, 0]);  permute_1154 = None
    view_1296: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_80, [1024, 1, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    add_206: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(view_1285, view_1296);  view_1285 = view_1296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    mm_82: "f32[1024, 768]" = torch.ops.aten.mm.default(as_strided_scatter_69, permute_1156);  permute_1156 = None
    permute_1158: "f32[768, 1024]" = torch.ops.aten.permute.default(as_strided_scatter_69, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1158, view_375);  permute_1158 = view_375 = None
    permute_1159: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_117: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(as_strided_scatter_69, [0], True);  as_strided_scatter_69 = None
    view_1297: "f32[768]" = torch.ops.aten.view.default(sum_117, [768]);  sum_117 = None
    permute_1160: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1159, [1, 0]);  permute_1159 = None
    view_1298: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_82, [1024, 1, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    add_207: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(add_206, view_1298);  add_206 = view_1298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_1161: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(add_207, [1, 0, 2]);  add_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    add_208: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_291, permute_1161);  mul_291 = permute_1161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_300: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_208, primals_79);  primals_79 = None
    mul_301: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_300, 768)
    sum_118: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True)
    mul_302: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_300, mul_38);  mul_300 = None
    sum_119: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True);  mul_302 = None
    mul_303: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_38, sum_119);  sum_119 = None
    sub_146: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_301, sum_118);  mul_301 = sum_118 = None
    sub_147: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_146, mul_303);  sub_146 = mul_303 = None
    mul_304: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_141, sub_147);  div_141 = sub_147 = None
    mul_305: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_208, mul_38);  mul_38 = None
    sum_120: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 1]);  mul_305 = None
    sum_121: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_208, [0, 1]);  add_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_81: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_47, torch.float32);  getitem_47 = None
    mul_306: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 1.1111111111111112);  convert_element_type_81 = None
    mul_307: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_304, mul_306);  mul_306 = None
    clone_179: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_307, memory_format = torch.contiguous_format);  mul_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_1299: "f32[1024, 768]" = torch.ops.aten.view.default(clone_179, [1024, 768]);  clone_179 = None
    mm_84: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_1299, permute_1162);  permute_1162 = None
    permute_1163: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1299, [1, 0])
    mm_85: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1163, view_373);  permute_1163 = view_373 = None
    permute_1164: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_122: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1299, [0], True);  view_1299 = None
    view_1300: "f32[768]" = torch.ops.aten.view.default(sum_122, [768]);  sum_122 = None
    permute_1165: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1164, [1, 0]);  permute_1164 = None
    view_1301: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_84, [1, 1024, 3072]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_309: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_51, 0.5);  add_51 = None
    mul_310: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_372, view_372)
    mul_311: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_310, -0.5);  mul_310 = None
    exp_19: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_311);  mul_311 = None
    mul_312: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_313: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_372, mul_312);  view_372 = mul_312 = None
    add_210: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_309, mul_313);  mul_309 = mul_313 = None
    mul_314: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1301, add_210);  view_1301 = add_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_1302: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_314, [1024, 3072]);  mul_314 = None
    mm_86: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1302, permute_1166);  permute_1166 = None
    permute_1167: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_1302, [1, 0])
    mm_87: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1167, view_371);  permute_1167 = view_371 = None
    permute_1168: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_123: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1302, [0], True);  view_1302 = None
    view_1303: "f32[3072]" = torch.ops.aten.view.default(sum_123, [3072]);  sum_123 = None
    permute_1169: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1168, [1, 0]);  permute_1168 = None
    view_1304: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_86, [1, 1024, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    add_211: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_304, view_1304);  mul_304 = view_1304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_316: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_211, primals_73);  primals_73 = None
    mul_317: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_316, 768)
    sum_124: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_316, [2], True)
    mul_318: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_316, mul_33);  mul_316 = None
    sum_125: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_318, [2], True);  mul_318 = None
    mul_319: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_33, sum_125);  sum_125 = None
    sub_149: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_317, sum_124);  mul_317 = sum_124 = None
    sub_150: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_149, mul_319);  sub_149 = mul_319 = None
    mul_320: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_142, sub_150);  div_142 = sub_150 = None
    mul_321: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_211, mul_33);  mul_33 = None
    sum_126: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_321, [0, 1]);  mul_321 = None
    sum_127: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_211, [0, 1]);  add_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_82: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_43, torch.float32);  getitem_43 = None
    mul_322: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_82, 1.1111111111111112);  convert_element_type_82 = None
    mul_323: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_320, mul_322);  mul_322 = None
    clone_180: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_323, memory_format = torch.contiguous_format);  mul_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_1305: "f32[1024, 768]" = torch.ops.aten.view.default(clone_180, [1024, 768]);  clone_180 = None
    mm_88: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1305, permute_1170);  permute_1170 = None
    permute_1171: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1305, [1, 0])
    mm_89: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1171, view_369);  permute_1171 = view_369 = None
    permute_1172: "f32[768, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_128: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1305, [0], True);  view_1305 = None
    view_1306: "f32[768]" = torch.ops.aten.view.default(sum_128, [768]);  sum_128 = None
    permute_1173: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1172, [1, 0]);  permute_1172 = None
    view_1307: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_88, [1, 1024, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_1174: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(view_1307, [1, 0, 2]);  view_1307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    view_1308: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(permute_1174, [1024, 1, 12, 64]);  permute_1174 = None
    permute_1175: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1308, [1, 0, 2, 3]);  view_1308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    permute_1176: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1175, [0, 2, 1, 3]);  permute_1175 = None
    view_1309: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_1176, [12, 4, 256, 64]);  permute_1176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    view_1310: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_1309, [12, 4, 256, 64, 1]);  view_1309 = None
    permute_1177: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1310, [0, 1, 2, 4, 3]);  view_1310 = None
    clone_181: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.clone.default(permute_1177, memory_format = torch.contiguous_format);  permute_1177 = None
    view_1311: "f32[48, 256, 64]" = torch.ops.aten.view.default(clone_181, [48, 256, 64]);  clone_181 = None
    bmm_52: "f32[48, 768, 64]" = torch.ops.aten.bmm.default(permute_1178, view_1311);  permute_1178 = None
    bmm_53: "f32[48, 256, 768]" = torch.ops.aten.bmm.default(view_1311, permute_1179);  view_1311 = permute_1179 = None
    view_1312: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_52, [12, 4, 768, 64, 1]);  bmm_52 = None
    permute_1180: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(view_1312, [0, 1, 4, 3, 2]);  view_1312 = None
    view_1313: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_53, [12, 4, 256, 768, 1]);  bmm_53 = None
    permute_1181: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(view_1313, [0, 1, 2, 4, 3]);  view_1313 = None
    permute_1182: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_1180, [0, 1, 4, 3, 2]);  permute_1180 = None
    squeeze_56: "f32[12, 4, 768, 64]" = torch.ops.aten.squeeze.dim(permute_1182, 4);  permute_1182 = None
    permute_1183: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_1181, [0, 1, 2, 4, 3]);  permute_1181 = None
    squeeze_57: "f32[12, 4, 256, 768]" = torch.ops.aten.squeeze.dim(permute_1183, 4);  permute_1183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_scatter_732: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, squeeze_57, 3, 0, -1);  squeeze_57 = None
    slice_scatter_733: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_732, 2, 0, 9223372036854775807);  slice_scatter_732 = None
    slice_scatter_734: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_733, 1, 0, 9223372036854775807);  slice_scatter_733 = None
    slice_scatter_735: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_734, 0, 0, 9223372036854775807);  slice_scatter_734 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1314: "f32[12, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_735, [12, 4, 196864]);  slice_scatter_735 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_scatter_736: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, view_1314, 2, 0, -256);  view_1314 = None
    slice_scatter_737: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_736, 1, 0, 9223372036854775807);  slice_scatter_736 = None
    slice_scatter_738: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_737, 0, 0, 9223372036854775807);  slice_scatter_737 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1315: "f32[12, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_738, [12, 4, 256, 770]);  slice_scatter_738 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_69: "f32[12, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1315, [0, -257]);  view_1315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    view_1316: "f32[2359296]" = torch.ops.aten.view.default(squeeze_56, [-1]);  squeeze_56 = None
    index_put_21: "f32[1179648]" = torch.ops.aten.index_put.default(full_default_128, [view_918], view_1316, True);  view_1316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    as_strided_389: "f32[12, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_21, [12, 1536, 64], [98304, 64, 1], 0);  index_put_21 = None
    constant_pad_nd_70: "f32[12, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_389, [0, 0, -256, -256]);  as_strided_389 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    view_1318: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_70, [1, 12, 1024, 64]);  constant_pad_nd_70 = None
    permute_1184: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1318, [0, 2, 1, 3]);  view_1318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    view_1319: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_69, [1, 12, 1024, 513]);  constant_pad_nd_69 = None
    permute_1185: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1319, [0, 2, 1, 3]);  view_1319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    permute_1186: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1184, [1, 0, 2, 3]);  permute_1184 = None
    clone_183: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_1186, memory_format = torch.contiguous_format);  permute_1186 = None
    view_1320: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_183, [1024, 1, 768]);  clone_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    convert_element_type_83: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(getitem_41, torch.float32);  getitem_41 = None
    mul_324: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_83, 1.1111111111111112);  convert_element_type_83 = None
    mul_325: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1185, mul_324);  permute_1185 = mul_324 = None
    clone_184: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_325, memory_format = torch.contiguous_format);  mul_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_117: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, clone_184);  clone_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    mul_326: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_117, alias_19);  where_117 = None
    sum_129: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_326, [-1], True)
    mul_327: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(alias_19, sum_129);  alias_19 = sum_129 = None
    sub_151: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(mul_326, mul_327);  mul_326 = mul_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    squeeze_58: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(sub_151, 0);  sub_151 = None
    copy_270: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_58);  squeeze_58 = None
    as_strided_scatter_70: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_270, [1024, 12, 513], [513, 525312, 1], 0);  copy_270 = None
    as_strided_393: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_70, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_70 = None
    new_empty_strided_56: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_393, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_271: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_56, as_strided_393);  new_empty_strided_56 = as_strided_393 = None
    as_strided_395: "f32[1, 1024, 12, 513]" = torch.ops.aten.as_strided.default(copy_271, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0)
    clone_185: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(as_strided_395, memory_format = torch.contiguous_format)
    copy_272: "f32[1, 1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_395, clone_185);  as_strided_395 = clone_185 = None
    as_strided_scatter_71: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_271, copy_272, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0);  copy_271 = copy_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    new_empty_strided_57: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_71, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_273: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_57, as_strided_scatter_71);  new_empty_strided_57 = as_strided_scatter_71 = None
    as_strided_398: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_273, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240)
    clone_186: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_398, memory_format = torch.contiguous_format)
    copy_274: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_398, full_default_130);  as_strided_398 = None
    as_strided_scatter_72: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_273, copy_274, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240);  copy_273 = copy_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_118: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_186);  clone_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    slice_scatter_739: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_118, 3, -257, 9223372036854775807);  where_118 = None
    slice_scatter_740: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_739, 2, 0, 9223372036854775807);  slice_scatter_739 = None
    slice_scatter_741: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_740, 1, -256, 9223372036854775807);  slice_scatter_740 = None
    slice_scatter_742: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_741, 0, 0, 9223372036854775807);  slice_scatter_741 = None
    squeeze_59: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(slice_scatter_742, 0);  slice_scatter_742 = None
    copy_275: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_59);  squeeze_59 = None
    as_strided_scatter_73: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_275, [1024, 12, 513], [513, 525312, 1], 0);  copy_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    as_strided_403: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_73, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_73 = None
    add_212: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_72, as_strided_403);  as_strided_scatter_72 = as_strided_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    new_empty_strided_58: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_212, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_276: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_58, add_212);  new_empty_strided_58 = add_212 = None
    as_strided_405: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_276, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0)
    clone_187: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_405, memory_format = torch.contiguous_format)
    copy_277: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_405, full_default_130);  as_strided_405 = None
    as_strided_scatter_74: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_276, copy_277, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0);  copy_276 = copy_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_119: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_187);  clone_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    slice_scatter_743: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_119, 3, 0, 257);  where_119 = None
    slice_scatter_744: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_743, 2, 0, 9223372036854775807);  slice_scatter_743 = None
    slice_scatter_745: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_744, 1, 0, 256);  slice_scatter_744 = None
    slice_scatter_746: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_745, 0, 0, 9223372036854775807);  slice_scatter_745 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    permute_1187: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_746, [0, 2, 1, 3]);  slice_scatter_746 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_1321: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_1187, [12, 4, 256, 513]);  permute_1187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    add_213: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_74, view_1321);  as_strided_scatter_74 = view_1321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_59: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_213, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_278: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_59, add_213);  new_empty_strided_59 = add_213 = None
    as_strided_408: "f32[12, 255, 255]" = torch.ops.aten.as_strided.default(copy_278, [12, 255, 255], [525312, 513, 1], 514)
    clone_188: "f32[12, 255, 255]" = torch.ops.aten.clone.default(as_strided_408, memory_format = torch.contiguous_format)
    copy_279: "f32[12, 255, 255]" = torch.ops.aten.copy.default(as_strided_408, full_default_142);  as_strided_408 = None
    as_strided_scatter_75: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_278, copy_279, [12, 255, 255], [525312, 513, 1], 514);  copy_278 = copy_279 = None
    slice_scatter_747: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_143, clone_188, 2, -255, 9223372036854775807);  clone_188 = None
    slice_scatter_748: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_747, 1, 0, 255);  slice_scatter_747 = None
    select_scatter_62: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_748, 1, 0);  slice_scatter_748 = None
    slice_scatter_749: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_62, 0, 0, 9223372036854775807);  select_scatter_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_60: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_75, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_280: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_60, as_strided_scatter_75);  new_empty_strided_60 = as_strided_scatter_75 = None
    as_strided_411: "f32[12, 3, 256, 256]" = torch.ops.aten.as_strided.default(copy_280, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328)
    clone_189: "f32[12, 3, 256, 256]" = torch.ops.aten.clone.default(as_strided_411, memory_format = torch.contiguous_format)
    copy_281: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(as_strided_411, full_default_147);  as_strided_411 = None
    as_strided_scatter_76: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_280, copy_281, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328);  copy_280 = copy_281 = None
    slice_scatter_750: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_189, 3, 257, 9223372036854775807);  clone_189 = None
    slice_scatter_751: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_750, 2, -257, -1);  slice_scatter_750 = None
    slice_scatter_752: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_751, 1, 0, 9223372036854775807);  slice_scatter_751 = None
    slice_scatter_753: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_752, 0, 0, 9223372036854775807);  slice_scatter_752 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    add_214: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(slice_scatter_749, slice_scatter_753);  slice_scatter_749 = slice_scatter_753 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_61: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_76, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_282: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_61, as_strided_scatter_76);  new_empty_strided_61 = as_strided_scatter_76 = None
    as_strided_414: "f32[12, 256, 257]" = torch.ops.aten.as_strided.default(copy_282, [12, 256, 257], [525312, 513, 1], 394240)
    clone_190: "f32[12, 256, 257]" = torch.ops.aten.clone.default(as_strided_414, memory_format = torch.contiguous_format)
    copy_283: "f32[12, 256, 257]" = torch.ops.aten.copy.default(as_strided_414, full_default_152);  as_strided_414 = None
    as_strided_scatter_77: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_282, copy_283, [12, 256, 257], [525312, 513, 1], 394240);  copy_282 = copy_283 = None
    slice_scatter_754: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_153, clone_190, 2, 0, 257);  clone_190 = None
    slice_scatter_755: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_754, 1, 256, 9223372036854775807);  slice_scatter_754 = None
    select_scatter_63: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_755, 1, -1);  slice_scatter_755 = None
    slice_scatter_756: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_63, 0, 0, 9223372036854775807);  select_scatter_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_215: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_214, slice_scatter_756);  add_214 = slice_scatter_756 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_62: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_77, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_284: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_62, as_strided_scatter_77);  new_empty_strided_62 = as_strided_scatter_77 = None
    as_strided_417: "f32[12, 3, 256, 257]" = torch.ops.aten.as_strided.default(copy_284, [12, 3, 256, 257], [525312, 131328, 513, 1], 256);  copy_284 = None
    clone_191: "f32[12, 3, 256, 257]" = torch.ops.aten.clone.default(as_strided_417, memory_format = torch.contiguous_format);  as_strided_417 = None
    slice_scatter_757: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_191, 3, 0, 257);  clone_191 = None
    slice_scatter_758: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_757, 2, 0, 256);  slice_scatter_757 = None
    slice_scatter_759: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_758, 1, 0, 9223372036854775807);  slice_scatter_758 = None
    slice_scatter_760: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_759, 0, 0, 9223372036854775807);  slice_scatter_759 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_216: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_215, slice_scatter_760);  add_215 = slice_scatter_760 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_1322: "f32[12, 3, 513, 512]" = torch.ops.aten.view.default(add_216, [12, 3, 513, 512]);  add_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_71: "f32[12, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1322, [0, 0, 0, -1]);  view_1322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    view_1323: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_71, [12, 3, 512, 512, 1]);  constant_pad_nd_71 = None
    permute_1188: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1323, [0, 1, 2, 4, 3]);  view_1323 = None
    view_1324: "f32[36, 512, 512]" = torch.ops.aten.view.default(permute_1188, [36, 512, 512]);  permute_1188 = None
    bmm_54: "f32[36, 64, 512]" = torch.ops.aten.bmm.default(permute_1189, view_1324);  permute_1189 = None
    bmm_55: "f32[36, 512, 64]" = torch.ops.aten.bmm.default(view_1324, permute_1190);  view_1324 = permute_1190 = None
    view_1325: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_54, [12, 3, 64, 512, 1]);  bmm_54 = None
    permute_1191: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1325, [0, 1, 4, 3, 2]);  view_1325 = None
    view_1326: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_55, [12, 3, 512, 64, 1]);  bmm_55 = None
    permute_1192: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(view_1326, [0, 1, 2, 4, 3]);  view_1326 = None
    permute_1193: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1191, [0, 1, 3, 4, 2]);  permute_1191 = None
    squeeze_60: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1193, 4);  permute_1193 = None
    permute_1194: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1192, [0, 1, 2, 4, 3]);  permute_1192 = None
    squeeze_61: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1194, 4);  permute_1194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    clone_192: "f32[12, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_60, memory_format = torch.contiguous_format);  squeeze_60 = None
    view_1327: "f32[1179648]" = torch.ops.aten.view.default(clone_192, [1179648]);  clone_192 = None
    index_put_22: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1327, True);  view_1327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    view_1330: "f32[1179648]" = torch.ops.aten.view.default(squeeze_61, [-1]);  squeeze_61 = None
    index_put_23: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1330, True);  view_1330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    as_strided_422: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_23, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_23 = None
    view_1337: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_422, [12, 1024, 64]);  as_strided_422 = None
    view_1338: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1337, [1, 12, 1024, 64]);  view_1337 = None
    permute_1199: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1338, [0, 2, 1, 3]);  view_1338 = None
    permute_1200: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1199, [1, 0, 2, 3]);  permute_1199 = None
    view_1339: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1200, [1024, 1, 768]);  permute_1200 = None
    squeeze_63: "f32[1024, 768]" = torch.ops.aten.squeeze.dim(view_1339, 1);  view_1339 = None
    copy_285: "f32[1024, 768]" = torch.ops.aten.copy.default(as_strided_108, squeeze_63);  squeeze_63 = None
    as_strided_scatter_78: "f32[786432]" = torch.ops.aten.as_strided_scatter.default(full_default_161, copy_285, [1024, 768], [768, 1], 0);  copy_285 = None
    as_strided_426: "f32[1024, 768]" = torch.ops.aten.as_strided.default(as_strided_scatter_78, [1024, 768], [768, 1], 0);  as_strided_scatter_78 = None
    new_empty_strided_63: "f32[1024, 768]" = torch.ops.aten.new_empty_strided.default(as_strided_426, [1024, 768], [768, 1])
    copy_286: "f32[1024, 768]" = torch.ops.aten.copy.default(new_empty_strided_63, as_strided_426);  new_empty_strided_63 = as_strided_426 = None
    as_strided_428: "f32[1024, 1, 768]" = torch.ops.aten.as_strided.default(copy_286, [1024, 1, 768], [768, 768, 1], 0)
    clone_195: "f32[1024, 1, 768]" = torch.ops.aten.clone.default(as_strided_428, memory_format = torch.contiguous_format)
    div_143: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(clone_195, 8.0);  clone_195 = None
    copy_287: "f32[1024, 1, 768]" = torch.ops.aten.copy.default(as_strided_428, div_143);  as_strided_428 = div_143 = None
    as_strided_scatter_79: "f32[1024, 768]" = torch.ops.aten.as_strided_scatter.default(copy_286, copy_287, [1024, 1, 768], [768, 768, 1], 0);  copy_286 = copy_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_1340: "f32[1024, 768]" = torch.ops.aten.view.default(view_1320, [1024, 768]);  view_1320 = None
    mm_90: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1340, permute_1201);  permute_1201 = None
    permute_1202: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1340, [1, 0])
    mm_91: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1202, view_300);  permute_1202 = None
    permute_1203: "f32[768, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_130: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1340, [0], True);  view_1340 = None
    view_1341: "f32[768]" = torch.ops.aten.view.default(sum_130, [768]);  sum_130 = None
    permute_1204: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1203, [1, 0]);  permute_1203 = None
    view_1342: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_90, [1024, 1, 768]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    as_strided_430: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_22, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_22 = None
    view_1344: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_430, [12, 1024, 64]);  as_strided_430 = None
    view_1345: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1344, [1, 12, 1024, 64]);  view_1344 = None
    permute_1206: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1345, [0, 2, 1, 3]);  view_1345 = None
    permute_1207: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1206, [1, 0, 2, 3]);  permute_1206 = None
    view_1346: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1207, [1024, 1, 768]);  permute_1207 = None
    view_1347: "f32[1024, 768]" = torch.ops.aten.view.default(view_1346, [1024, 768]);  view_1346 = None
    mm_92: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1347, permute_1205);  permute_1205 = None
    permute_1211: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1347, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1211, view_300);  permute_1211 = None
    permute_1212: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_131: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1347, [0], True);  view_1347 = None
    view_1352: "f32[768]" = torch.ops.aten.view.default(sum_131, [768]);  sum_131 = None
    permute_1213: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1212, [1, 0]);  permute_1212 = None
    view_1353: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_92, [1024, 1, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    add_217: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(view_1342, view_1353);  view_1342 = view_1353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    mm_94: "f32[1024, 768]" = torch.ops.aten.mm.default(as_strided_scatter_79, permute_1214);  permute_1214 = None
    permute_1216: "f32[768, 1024]" = torch.ops.aten.permute.default(as_strided_scatter_79, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1216, view_300);  permute_1216 = view_300 = None
    permute_1217: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_132: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(as_strided_scatter_79, [0], True);  as_strided_scatter_79 = None
    view_1354: "f32[768]" = torch.ops.aten.view.default(sum_132, [768]);  sum_132 = None
    permute_1218: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1217, [1, 0]);  permute_1217 = None
    view_1355: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_94, [1024, 1, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    add_218: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(add_217, view_1355);  add_217 = view_1355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_1219: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(add_218, [1, 0, 2]);  add_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    add_219: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_320, permute_1219);  mul_320 = permute_1219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_329: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_219, primals_63);  primals_63 = None
    mul_330: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_329, 768)
    sum_133: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_329, [2], True)
    mul_331: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_329, mul_30);  mul_329 = None
    sum_134: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_331, [2], True);  mul_331 = None
    mul_332: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_30, sum_134);  sum_134 = None
    sub_153: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_330, sum_133);  mul_330 = sum_133 = None
    sub_154: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_153, mul_332);  sub_153 = mul_332 = None
    mul_333: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_144, sub_154);  div_144 = sub_154 = None
    mul_334: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_219, mul_30);  mul_30 = None
    sum_135: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_334, [0, 1]);  mul_334 = None
    sum_136: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_219, [0, 1]);  add_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_84: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_37, torch.float32);  getitem_37 = None
    mul_335: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 1.1111111111111112);  convert_element_type_84 = None
    mul_336: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_333, mul_335);  mul_335 = None
    clone_196: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_336, memory_format = torch.contiguous_format);  mul_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_1356: "f32[1024, 768]" = torch.ops.aten.view.default(clone_196, [1024, 768]);  clone_196 = None
    mm_96: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_1356, permute_1220);  permute_1220 = None
    permute_1221: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1356, [1, 0])
    mm_97: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1221, view_298);  permute_1221 = view_298 = None
    permute_1222: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_137: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1356, [0], True);  view_1356 = None
    view_1357: "f32[768]" = torch.ops.aten.view.default(sum_137, [768]);  sum_137 = None
    permute_1223: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1222, [1, 0]);  permute_1222 = None
    view_1358: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_96, [1, 1024, 3072]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_338: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
    mul_339: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_297, view_297)
    mul_340: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_339, -0.5);  mul_339 = None
    exp_20: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_340);  mul_340 = None
    mul_341: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_342: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_297, mul_341);  view_297 = mul_341 = None
    add_221: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_338, mul_342);  mul_338 = mul_342 = None
    mul_343: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1358, add_221);  view_1358 = add_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_1359: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_343, [1024, 3072]);  mul_343 = None
    mm_98: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1359, permute_1224);  permute_1224 = None
    permute_1225: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_1359, [1, 0])
    mm_99: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1225, view_296);  permute_1225 = view_296 = None
    permute_1226: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_138: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1359, [0], True);  view_1359 = None
    view_1360: "f32[3072]" = torch.ops.aten.view.default(sum_138, [3072]);  sum_138 = None
    permute_1227: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1226, [1, 0]);  permute_1226 = None
    view_1361: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_98, [1, 1024, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    add_222: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_333, view_1361);  mul_333 = view_1361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_345: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_222, primals_57);  primals_57 = None
    mul_346: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_345, 768)
    sum_139: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_345, [2], True)
    mul_347: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_345, mul_25);  mul_345 = None
    sum_140: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True);  mul_347 = None
    mul_348: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_25, sum_140);  sum_140 = None
    sub_156: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_346, sum_139);  mul_346 = sum_139 = None
    sub_157: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_156, mul_348);  sub_156 = mul_348 = None
    mul_349: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_145, sub_157);  div_145 = sub_157 = None
    mul_350: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_222, mul_25);  mul_25 = None
    sum_141: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 1]);  mul_350 = None
    sum_142: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_222, [0, 1]);  add_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_85: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_33, torch.float32);  getitem_33 = None
    mul_351: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 1.1111111111111112);  convert_element_type_85 = None
    mul_352: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_349, mul_351);  mul_351 = None
    clone_197: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_352, memory_format = torch.contiguous_format);  mul_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_1362: "f32[1024, 768]" = torch.ops.aten.view.default(clone_197, [1024, 768]);  clone_197 = None
    mm_100: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1362, permute_1228);  permute_1228 = None
    permute_1229: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1362, [1, 0])
    mm_101: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1229, view_294);  permute_1229 = view_294 = None
    permute_1230: "f32[768, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_143: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1362, [0], True);  view_1362 = None
    view_1363: "f32[768]" = torch.ops.aten.view.default(sum_143, [768]);  sum_143 = None
    permute_1231: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1230, [1, 0]);  permute_1230 = None
    view_1364: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_100, [1, 1024, 768]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_1232: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(view_1364, [1, 0, 2]);  view_1364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    view_1365: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(permute_1232, [1024, 1, 12, 64]);  permute_1232 = None
    permute_1233: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1365, [1, 0, 2, 3]);  view_1365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    permute_1234: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1233, [0, 2, 1, 3]);  permute_1233 = None
    view_1366: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_1234, [12, 4, 256, 64]);  permute_1234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    view_1367: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_1366, [12, 4, 256, 64, 1]);  view_1366 = None
    permute_1235: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1367, [0, 1, 2, 4, 3]);  view_1367 = None
    clone_198: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.clone.default(permute_1235, memory_format = torch.contiguous_format);  permute_1235 = None
    view_1368: "f32[48, 256, 64]" = torch.ops.aten.view.default(clone_198, [48, 256, 64]);  clone_198 = None
    bmm_56: "f32[48, 768, 64]" = torch.ops.aten.bmm.default(permute_1236, view_1368);  permute_1236 = None
    bmm_57: "f32[48, 256, 768]" = torch.ops.aten.bmm.default(view_1368, permute_1237);  view_1368 = permute_1237 = None
    view_1369: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_56, [12, 4, 768, 64, 1]);  bmm_56 = None
    permute_1238: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(view_1369, [0, 1, 4, 3, 2]);  view_1369 = None
    view_1370: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_57, [12, 4, 256, 768, 1]);  bmm_57 = None
    permute_1239: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(view_1370, [0, 1, 2, 4, 3]);  view_1370 = None
    permute_1240: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_1238, [0, 1, 4, 3, 2]);  permute_1238 = None
    squeeze_64: "f32[12, 4, 768, 64]" = torch.ops.aten.squeeze.dim(permute_1240, 4);  permute_1240 = None
    permute_1241: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_1239, [0, 1, 2, 4, 3]);  permute_1239 = None
    squeeze_65: "f32[12, 4, 256, 768]" = torch.ops.aten.squeeze.dim(permute_1241, 4);  permute_1241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_scatter_761: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, squeeze_65, 3, 0, -1);  squeeze_65 = None
    slice_scatter_762: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_761, 2, 0, 9223372036854775807);  slice_scatter_761 = None
    slice_scatter_763: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_762, 1, 0, 9223372036854775807);  slice_scatter_762 = None
    slice_scatter_764: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_763, 0, 0, 9223372036854775807);  slice_scatter_763 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1371: "f32[12, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_764, [12, 4, 196864]);  slice_scatter_764 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_scatter_765: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, view_1371, 2, 0, -256);  view_1371 = None
    slice_scatter_766: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_765, 1, 0, 9223372036854775807);  slice_scatter_765 = None
    slice_scatter_767: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_766, 0, 0, 9223372036854775807);  slice_scatter_766 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1372: "f32[12, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_767, [12, 4, 256, 770]);  slice_scatter_767 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_72: "f32[12, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1372, [0, -257]);  view_1372 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    view_1373: "f32[2359296]" = torch.ops.aten.view.default(squeeze_64, [-1]);  squeeze_64 = None
    index_put_24: "f32[1179648]" = torch.ops.aten.index_put.default(full_default_128, [view_918], view_1373, True);  view_1373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    as_strided_434: "f32[12, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_24, [12, 1536, 64], [98304, 64, 1], 0);  index_put_24 = None
    constant_pad_nd_73: "f32[12, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_434, [0, 0, -256, -256]);  as_strided_434 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    view_1375: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_73, [1, 12, 1024, 64]);  constant_pad_nd_73 = None
    permute_1242: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1375, [0, 2, 1, 3]);  view_1375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    view_1376: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_72, [1, 12, 1024, 513]);  constant_pad_nd_72 = None
    permute_1243: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1376, [0, 2, 1, 3]);  view_1376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    permute_1244: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1242, [1, 0, 2, 3]);  permute_1242 = None
    clone_200: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_1244, memory_format = torch.contiguous_format);  permute_1244 = None
    view_1377: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_200, [1024, 1, 768]);  clone_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    convert_element_type_86: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(getitem_31, torch.float32);  getitem_31 = None
    mul_353: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 1.1111111111111112);  convert_element_type_86 = None
    mul_354: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1243, mul_353);  permute_1243 = mul_353 = None
    clone_201: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_354, memory_format = torch.contiguous_format);  mul_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_120: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, clone_201);  clone_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    mul_355: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_120, alias_20);  where_120 = None
    sum_144: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [-1], True)
    mul_356: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(alias_20, sum_144);  alias_20 = sum_144 = None
    sub_158: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(mul_355, mul_356);  mul_355 = mul_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    squeeze_66: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(sub_158, 0);  sub_158 = None
    copy_288: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_66);  squeeze_66 = None
    as_strided_scatter_80: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_288, [1024, 12, 513], [513, 525312, 1], 0);  copy_288 = None
    as_strided_438: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_80, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_80 = None
    new_empty_strided_64: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_438, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_289: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_64, as_strided_438);  new_empty_strided_64 = as_strided_438 = None
    as_strided_440: "f32[1, 1024, 12, 513]" = torch.ops.aten.as_strided.default(copy_289, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0)
    clone_202: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(as_strided_440, memory_format = torch.contiguous_format)
    copy_290: "f32[1, 1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_440, clone_202);  as_strided_440 = clone_202 = None
    as_strided_scatter_81: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_289, copy_290, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0);  copy_289 = copy_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    new_empty_strided_65: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_81, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_291: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_65, as_strided_scatter_81);  new_empty_strided_65 = as_strided_scatter_81 = None
    as_strided_443: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_291, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240)
    clone_203: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_443, memory_format = torch.contiguous_format)
    copy_292: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_443, full_default_130);  as_strided_443 = None
    as_strided_scatter_82: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_291, copy_292, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240);  copy_291 = copy_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_121: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_203);  clone_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    slice_scatter_768: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_121, 3, -257, 9223372036854775807);  where_121 = None
    slice_scatter_769: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_768, 2, 0, 9223372036854775807);  slice_scatter_768 = None
    slice_scatter_770: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_769, 1, -256, 9223372036854775807);  slice_scatter_769 = None
    slice_scatter_771: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_770, 0, 0, 9223372036854775807);  slice_scatter_770 = None
    squeeze_67: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(slice_scatter_771, 0);  slice_scatter_771 = None
    copy_293: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_67);  squeeze_67 = None
    as_strided_scatter_83: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_293, [1024, 12, 513], [513, 525312, 1], 0);  copy_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    as_strided_448: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_83, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_83 = None
    add_223: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_82, as_strided_448);  as_strided_scatter_82 = as_strided_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    new_empty_strided_66: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_223, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_294: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_66, add_223);  new_empty_strided_66 = add_223 = None
    as_strided_450: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_294, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0)
    clone_204: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_450, memory_format = torch.contiguous_format)
    copy_295: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_450, full_default_130);  as_strided_450 = None
    as_strided_scatter_84: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_294, copy_295, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0);  copy_294 = copy_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_122: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_204);  clone_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    slice_scatter_772: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_122, 3, 0, 257);  where_122 = None
    slice_scatter_773: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_772, 2, 0, 9223372036854775807);  slice_scatter_772 = None
    slice_scatter_774: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_773, 1, 0, 256);  slice_scatter_773 = None
    slice_scatter_775: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_774, 0, 0, 9223372036854775807);  slice_scatter_774 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    permute_1245: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_775, [0, 2, 1, 3]);  slice_scatter_775 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_1378: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_1245, [12, 4, 256, 513]);  permute_1245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    add_224: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_84, view_1378);  as_strided_scatter_84 = view_1378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_67: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_224, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_296: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_67, add_224);  new_empty_strided_67 = add_224 = None
    as_strided_453: "f32[12, 255, 255]" = torch.ops.aten.as_strided.default(copy_296, [12, 255, 255], [525312, 513, 1], 514)
    clone_205: "f32[12, 255, 255]" = torch.ops.aten.clone.default(as_strided_453, memory_format = torch.contiguous_format)
    copy_297: "f32[12, 255, 255]" = torch.ops.aten.copy.default(as_strided_453, full_default_142);  as_strided_453 = None
    as_strided_scatter_85: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_296, copy_297, [12, 255, 255], [525312, 513, 1], 514);  copy_296 = copy_297 = None
    slice_scatter_776: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_143, clone_205, 2, -255, 9223372036854775807);  clone_205 = None
    slice_scatter_777: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_776, 1, 0, 255);  slice_scatter_776 = None
    select_scatter_64: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_777, 1, 0);  slice_scatter_777 = None
    slice_scatter_778: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_64, 0, 0, 9223372036854775807);  select_scatter_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_68: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_85, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_298: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_68, as_strided_scatter_85);  new_empty_strided_68 = as_strided_scatter_85 = None
    as_strided_456: "f32[12, 3, 256, 256]" = torch.ops.aten.as_strided.default(copy_298, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328)
    clone_206: "f32[12, 3, 256, 256]" = torch.ops.aten.clone.default(as_strided_456, memory_format = torch.contiguous_format)
    copy_299: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(as_strided_456, full_default_147);  as_strided_456 = None
    as_strided_scatter_86: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_298, copy_299, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328);  copy_298 = copy_299 = None
    slice_scatter_779: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_206, 3, 257, 9223372036854775807);  clone_206 = None
    slice_scatter_780: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_779, 2, -257, -1);  slice_scatter_779 = None
    slice_scatter_781: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_780, 1, 0, 9223372036854775807);  slice_scatter_780 = None
    slice_scatter_782: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_781, 0, 0, 9223372036854775807);  slice_scatter_781 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    add_225: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(slice_scatter_778, slice_scatter_782);  slice_scatter_778 = slice_scatter_782 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_69: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_86, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_300: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_69, as_strided_scatter_86);  new_empty_strided_69 = as_strided_scatter_86 = None
    as_strided_459: "f32[12, 256, 257]" = torch.ops.aten.as_strided.default(copy_300, [12, 256, 257], [525312, 513, 1], 394240)
    clone_207: "f32[12, 256, 257]" = torch.ops.aten.clone.default(as_strided_459, memory_format = torch.contiguous_format)
    copy_301: "f32[12, 256, 257]" = torch.ops.aten.copy.default(as_strided_459, full_default_152);  as_strided_459 = None
    as_strided_scatter_87: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_300, copy_301, [12, 256, 257], [525312, 513, 1], 394240);  copy_300 = copy_301 = None
    slice_scatter_783: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_153, clone_207, 2, 0, 257);  clone_207 = None
    slice_scatter_784: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_783, 1, 256, 9223372036854775807);  slice_scatter_783 = None
    select_scatter_65: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_784, 1, -1);  slice_scatter_784 = None
    slice_scatter_785: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_65, 0, 0, 9223372036854775807);  select_scatter_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_226: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_225, slice_scatter_785);  add_225 = slice_scatter_785 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_70: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_87, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_302: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_70, as_strided_scatter_87);  new_empty_strided_70 = as_strided_scatter_87 = None
    as_strided_462: "f32[12, 3, 256, 257]" = torch.ops.aten.as_strided.default(copy_302, [12, 3, 256, 257], [525312, 131328, 513, 1], 256);  copy_302 = None
    clone_208: "f32[12, 3, 256, 257]" = torch.ops.aten.clone.default(as_strided_462, memory_format = torch.contiguous_format);  as_strided_462 = None
    slice_scatter_786: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_208, 3, 0, 257);  clone_208 = None
    slice_scatter_787: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_786, 2, 0, 256);  slice_scatter_786 = None
    slice_scatter_788: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_787, 1, 0, 9223372036854775807);  slice_scatter_787 = None
    slice_scatter_789: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_788, 0, 0, 9223372036854775807);  slice_scatter_788 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_227: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_226, slice_scatter_789);  add_226 = slice_scatter_789 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_1379: "f32[12, 3, 513, 512]" = torch.ops.aten.view.default(add_227, [12, 3, 513, 512]);  add_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_74: "f32[12, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1379, [0, 0, 0, -1]);  view_1379 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    view_1380: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_74, [12, 3, 512, 512, 1]);  constant_pad_nd_74 = None
    permute_1246: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1380, [0, 1, 2, 4, 3]);  view_1380 = None
    view_1381: "f32[36, 512, 512]" = torch.ops.aten.view.default(permute_1246, [36, 512, 512]);  permute_1246 = None
    bmm_58: "f32[36, 64, 512]" = torch.ops.aten.bmm.default(permute_1247, view_1381);  permute_1247 = None
    bmm_59: "f32[36, 512, 64]" = torch.ops.aten.bmm.default(view_1381, permute_1248);  view_1381 = permute_1248 = None
    view_1382: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_58, [12, 3, 64, 512, 1]);  bmm_58 = None
    permute_1249: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1382, [0, 1, 4, 3, 2]);  view_1382 = None
    view_1383: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_59, [12, 3, 512, 64, 1]);  bmm_59 = None
    permute_1250: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(view_1383, [0, 1, 2, 4, 3]);  view_1383 = None
    permute_1251: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1249, [0, 1, 3, 4, 2]);  permute_1249 = None
    squeeze_68: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1251, 4);  permute_1251 = None
    permute_1252: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1250, [0, 1, 2, 4, 3]);  permute_1250 = None
    squeeze_69: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1252, 4);  permute_1252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    clone_209: "f32[12, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_68, memory_format = torch.contiguous_format);  squeeze_68 = None
    view_1384: "f32[1179648]" = torch.ops.aten.view.default(clone_209, [1179648]);  clone_209 = None
    index_put_25: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1384, True);  view_1384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    view_1387: "f32[1179648]" = torch.ops.aten.view.default(squeeze_69, [-1]);  squeeze_69 = None
    index_put_26: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1387, True);  view_1387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    as_strided_467: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_26, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_26 = None
    view_1394: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_467, [12, 1024, 64]);  as_strided_467 = None
    view_1395: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1394, [1, 12, 1024, 64]);  view_1394 = None
    permute_1257: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1395, [0, 2, 1, 3]);  view_1395 = None
    permute_1258: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1257, [1, 0, 2, 3]);  permute_1257 = None
    view_1396: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1258, [1024, 1, 768]);  permute_1258 = None
    squeeze_71: "f32[1024, 768]" = torch.ops.aten.squeeze.dim(view_1396, 1);  view_1396 = None
    copy_303: "f32[1024, 768]" = torch.ops.aten.copy.default(as_strided_108, squeeze_71);  squeeze_71 = None
    as_strided_scatter_88: "f32[786432]" = torch.ops.aten.as_strided_scatter.default(full_default_161, copy_303, [1024, 768], [768, 1], 0);  copy_303 = None
    as_strided_471: "f32[1024, 768]" = torch.ops.aten.as_strided.default(as_strided_scatter_88, [1024, 768], [768, 1], 0);  as_strided_scatter_88 = None
    new_empty_strided_71: "f32[1024, 768]" = torch.ops.aten.new_empty_strided.default(as_strided_471, [1024, 768], [768, 1])
    copy_304: "f32[1024, 768]" = torch.ops.aten.copy.default(new_empty_strided_71, as_strided_471);  new_empty_strided_71 = as_strided_471 = None
    as_strided_473: "f32[1024, 1, 768]" = torch.ops.aten.as_strided.default(copy_304, [1024, 1, 768], [768, 768, 1], 0)
    clone_212: "f32[1024, 1, 768]" = torch.ops.aten.clone.default(as_strided_473, memory_format = torch.contiguous_format)
    div_146: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(clone_212, 8.0);  clone_212 = None
    copy_305: "f32[1024, 1, 768]" = torch.ops.aten.copy.default(as_strided_473, div_146);  as_strided_473 = div_146 = None
    as_strided_scatter_89: "f32[1024, 768]" = torch.ops.aten.as_strided_scatter.default(copy_304, copy_305, [1024, 1, 768], [768, 768, 1], 0);  copy_304 = copy_305 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_1397: "f32[1024, 768]" = torch.ops.aten.view.default(view_1377, [1024, 768]);  view_1377 = None
    mm_102: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1397, permute_1259);  permute_1259 = None
    permute_1260: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1397, [1, 0])
    mm_103: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1260, view_225);  permute_1260 = None
    permute_1261: "f32[768, 768]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_145: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1397, [0], True);  view_1397 = None
    view_1398: "f32[768]" = torch.ops.aten.view.default(sum_145, [768]);  sum_145 = None
    permute_1262: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1261, [1, 0]);  permute_1261 = None
    view_1399: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_102, [1024, 1, 768]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    as_strided_475: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_25, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_25 = None
    view_1401: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_475, [12, 1024, 64]);  as_strided_475 = None
    view_1402: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1401, [1, 12, 1024, 64]);  view_1401 = None
    permute_1264: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1402, [0, 2, 1, 3]);  view_1402 = None
    permute_1265: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1264, [1, 0, 2, 3]);  permute_1264 = None
    view_1403: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1265, [1024, 1, 768]);  permute_1265 = None
    view_1404: "f32[1024, 768]" = torch.ops.aten.view.default(view_1403, [1024, 768]);  view_1403 = None
    mm_104: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1404, permute_1263);  permute_1263 = None
    permute_1269: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1404, [1, 0])
    mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1269, view_225);  permute_1269 = None
    permute_1270: "f32[768, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_146: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1404, [0], True);  view_1404 = None
    view_1409: "f32[768]" = torch.ops.aten.view.default(sum_146, [768]);  sum_146 = None
    permute_1271: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1270, [1, 0]);  permute_1270 = None
    view_1410: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_104, [1024, 1, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    add_228: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(view_1399, view_1410);  view_1399 = view_1410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    mm_106: "f32[1024, 768]" = torch.ops.aten.mm.default(as_strided_scatter_89, permute_1272);  permute_1272 = None
    permute_1274: "f32[768, 1024]" = torch.ops.aten.permute.default(as_strided_scatter_89, [1, 0])
    mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1274, view_225);  permute_1274 = view_225 = None
    permute_1275: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_147: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(as_strided_scatter_89, [0], True);  as_strided_scatter_89 = None
    view_1411: "f32[768]" = torch.ops.aten.view.default(sum_147, [768]);  sum_147 = None
    permute_1276: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1275, [1, 0]);  permute_1275 = None
    view_1412: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_106, [1024, 1, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    add_229: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(add_228, view_1412);  add_228 = view_1412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_1277: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(add_229, [1, 0, 2]);  add_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    add_230: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_349, permute_1277);  mul_349 = permute_1277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_358: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_230, primals_47);  primals_47 = None
    mul_359: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_358, 768)
    sum_148: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_358, [2], True)
    mul_360: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_358, mul_22);  mul_358 = None
    sum_149: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_360, [2], True);  mul_360 = None
    mul_361: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_22, sum_149);  sum_149 = None
    sub_160: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_359, sum_148);  mul_359 = sum_148 = None
    sub_161: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_160, mul_361);  sub_160 = mul_361 = None
    mul_362: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_147, sub_161);  div_147 = sub_161 = None
    mul_363: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_230, mul_22);  mul_22 = None
    sum_150: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 1]);  mul_363 = None
    sum_151: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_87: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_27, torch.float32);  getitem_27 = None
    mul_364: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 1.1111111111111112);  convert_element_type_87 = None
    mul_365: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_362, mul_364);  mul_364 = None
    clone_213: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_365, memory_format = torch.contiguous_format);  mul_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_1413: "f32[1024, 768]" = torch.ops.aten.view.default(clone_213, [1024, 768]);  clone_213 = None
    mm_108: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_1413, permute_1278);  permute_1278 = None
    permute_1279: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1413, [1, 0])
    mm_109: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1279, view_223);  permute_1279 = view_223 = None
    permute_1280: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_152: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1413, [0], True);  view_1413 = None
    view_1414: "f32[768]" = torch.ops.aten.view.default(sum_152, [768]);  sum_152 = None
    permute_1281: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1280, [1, 0]);  permute_1280 = None
    view_1415: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_108, [1, 1024, 3072]);  mm_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_367: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_29, 0.5);  add_29 = None
    mul_368: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_222, view_222)
    mul_369: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_368, -0.5);  mul_368 = None
    exp_21: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_369);  mul_369 = None
    mul_370: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_371: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_222, mul_370);  view_222 = mul_370 = None
    add_232: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_367, mul_371);  mul_367 = mul_371 = None
    mul_372: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1415, add_232);  view_1415 = add_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_1416: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_372, [1024, 3072]);  mul_372 = None
    mm_110: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1416, permute_1282);  permute_1282 = None
    permute_1283: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_1416, [1, 0])
    mm_111: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1283, view_221);  permute_1283 = view_221 = None
    permute_1284: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_153: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1416, [0], True);  view_1416 = None
    view_1417: "f32[3072]" = torch.ops.aten.view.default(sum_153, [3072]);  sum_153 = None
    permute_1285: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1284, [1, 0]);  permute_1284 = None
    view_1418: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_110, [1, 1024, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    add_233: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_362, view_1418);  mul_362 = view_1418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_374: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_233, primals_41);  primals_41 = None
    mul_375: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_374, 768)
    sum_154: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [2], True)
    mul_376: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_374, mul_17);  mul_374 = None
    sum_155: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_376, [2], True);  mul_376 = None
    mul_377: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_17, sum_155);  sum_155 = None
    sub_163: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_375, sum_154);  mul_375 = sum_154 = None
    sub_164: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_163, mul_377);  sub_163 = mul_377 = None
    mul_378: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_148, sub_164);  div_148 = sub_164 = None
    mul_379: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_233, mul_17);  mul_17 = None
    sum_156: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_379, [0, 1]);  mul_379 = None
    sum_157: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_88: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_23, torch.float32);  getitem_23 = None
    mul_380: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 1.1111111111111112);  convert_element_type_88 = None
    mul_381: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_378, mul_380);  mul_380 = None
    clone_214: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_381, memory_format = torch.contiguous_format);  mul_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_1419: "f32[1024, 768]" = torch.ops.aten.view.default(clone_214, [1024, 768]);  clone_214 = None
    mm_112: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1419, permute_1286);  permute_1286 = None
    permute_1287: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1419, [1, 0])
    mm_113: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1287, view_219);  permute_1287 = view_219 = None
    permute_1288: "f32[768, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_158: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1419, [0], True);  view_1419 = None
    view_1420: "f32[768]" = torch.ops.aten.view.default(sum_158, [768]);  sum_158 = None
    permute_1289: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1288, [1, 0]);  permute_1288 = None
    view_1421: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_112, [1, 1024, 768]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_1290: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(view_1421, [1, 0, 2]);  view_1421 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    view_1422: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(permute_1290, [1024, 1, 12, 64]);  permute_1290 = None
    permute_1291: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1422, [1, 0, 2, 3]);  view_1422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    permute_1292: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1291, [0, 2, 1, 3]);  permute_1291 = None
    view_1423: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_1292, [12, 4, 256, 64]);  permute_1292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    view_1424: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_1423, [12, 4, 256, 64, 1]);  view_1423 = None
    permute_1293: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1424, [0, 1, 2, 4, 3]);  view_1424 = None
    clone_215: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.clone.default(permute_1293, memory_format = torch.contiguous_format);  permute_1293 = None
    view_1425: "f32[48, 256, 64]" = torch.ops.aten.view.default(clone_215, [48, 256, 64]);  clone_215 = None
    bmm_60: "f32[48, 768, 64]" = torch.ops.aten.bmm.default(permute_1294, view_1425);  permute_1294 = None
    bmm_61: "f32[48, 256, 768]" = torch.ops.aten.bmm.default(view_1425, permute_1295);  view_1425 = permute_1295 = None
    view_1426: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_60, [12, 4, 768, 64, 1]);  bmm_60 = None
    permute_1296: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(view_1426, [0, 1, 4, 3, 2]);  view_1426 = None
    view_1427: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_61, [12, 4, 256, 768, 1]);  bmm_61 = None
    permute_1297: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(view_1427, [0, 1, 2, 4, 3]);  view_1427 = None
    permute_1298: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_1296, [0, 1, 4, 3, 2]);  permute_1296 = None
    squeeze_72: "f32[12, 4, 768, 64]" = torch.ops.aten.squeeze.dim(permute_1298, 4);  permute_1298 = None
    permute_1299: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_1297, [0, 1, 2, 4, 3]);  permute_1297 = None
    squeeze_73: "f32[12, 4, 256, 768]" = torch.ops.aten.squeeze.dim(permute_1299, 4);  permute_1299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_scatter_790: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, squeeze_73, 3, 0, -1);  squeeze_73 = None
    slice_scatter_791: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_790, 2, 0, 9223372036854775807);  slice_scatter_790 = None
    slice_scatter_792: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_791, 1, 0, 9223372036854775807);  slice_scatter_791 = None
    slice_scatter_793: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_792, 0, 0, 9223372036854775807);  slice_scatter_792 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1428: "f32[12, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_793, [12, 4, 196864]);  slice_scatter_793 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_scatter_794: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, view_1428, 2, 0, -256);  view_1428 = None
    slice_scatter_795: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_794, 1, 0, 9223372036854775807);  slice_scatter_794 = None
    slice_scatter_796: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_795, 0, 0, 9223372036854775807);  slice_scatter_795 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1429: "f32[12, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_796, [12, 4, 256, 770]);  slice_scatter_796 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_75: "f32[12, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1429, [0, -257]);  view_1429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    view_1430: "f32[2359296]" = torch.ops.aten.view.default(squeeze_72, [-1]);  squeeze_72 = None
    index_put_27: "f32[1179648]" = torch.ops.aten.index_put.default(full_default_128, [view_918], view_1430, True);  view_1430 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    as_strided_479: "f32[12, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_27, [12, 1536, 64], [98304, 64, 1], 0);  index_put_27 = None
    constant_pad_nd_76: "f32[12, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_479, [0, 0, -256, -256]);  as_strided_479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    view_1432: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_76, [1, 12, 1024, 64]);  constant_pad_nd_76 = None
    permute_1300: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1432, [0, 2, 1, 3]);  view_1432 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    view_1433: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_75, [1, 12, 1024, 513]);  constant_pad_nd_75 = None
    permute_1301: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1433, [0, 2, 1, 3]);  view_1433 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    permute_1302: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1300, [1, 0, 2, 3]);  permute_1300 = None
    clone_217: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_1302, memory_format = torch.contiguous_format);  permute_1302 = None
    view_1434: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_217, [1024, 1, 768]);  clone_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    convert_element_type_89: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(getitem_21, torch.float32);  getitem_21 = None
    mul_382: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_89, 1.1111111111111112);  convert_element_type_89 = None
    mul_383: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1301, mul_382);  permute_1301 = mul_382 = None
    clone_218: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_383, memory_format = torch.contiguous_format);  mul_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_123: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, clone_218);  clone_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    mul_384: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_123, alias_21);  where_123 = None
    sum_159: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_384, [-1], True)
    mul_385: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(alias_21, sum_159);  alias_21 = sum_159 = None
    sub_165: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(mul_384, mul_385);  mul_384 = mul_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    squeeze_74: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(sub_165, 0);  sub_165 = None
    copy_306: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_74);  squeeze_74 = None
    as_strided_scatter_90: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_306, [1024, 12, 513], [513, 525312, 1], 0);  copy_306 = None
    as_strided_483: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_90, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_90 = None
    new_empty_strided_72: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_483, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_307: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_72, as_strided_483);  new_empty_strided_72 = as_strided_483 = None
    as_strided_485: "f32[1, 1024, 12, 513]" = torch.ops.aten.as_strided.default(copy_307, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0)
    clone_219: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(as_strided_485, memory_format = torch.contiguous_format)
    copy_308: "f32[1, 1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_485, clone_219);  as_strided_485 = clone_219 = None
    as_strided_scatter_91: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_307, copy_308, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0);  copy_307 = copy_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    new_empty_strided_73: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_91, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_309: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_73, as_strided_scatter_91);  new_empty_strided_73 = as_strided_scatter_91 = None
    as_strided_488: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_309, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240)
    clone_220: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_488, memory_format = torch.contiguous_format)
    copy_310: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_488, full_default_130);  as_strided_488 = None
    as_strided_scatter_92: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_309, copy_310, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240);  copy_309 = copy_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_124: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_220);  clone_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    slice_scatter_797: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_124, 3, -257, 9223372036854775807);  where_124 = None
    slice_scatter_798: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_797, 2, 0, 9223372036854775807);  slice_scatter_797 = None
    slice_scatter_799: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_798, 1, -256, 9223372036854775807);  slice_scatter_798 = None
    slice_scatter_800: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_799, 0, 0, 9223372036854775807);  slice_scatter_799 = None
    squeeze_75: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(slice_scatter_800, 0);  slice_scatter_800 = None
    copy_311: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_75);  squeeze_75 = None
    as_strided_scatter_93: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_311, [1024, 12, 513], [513, 525312, 1], 0);  copy_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    as_strided_493: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_93, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_93 = None
    add_234: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_92, as_strided_493);  as_strided_scatter_92 = as_strided_493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    new_empty_strided_74: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_234, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_312: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_74, add_234);  new_empty_strided_74 = add_234 = None
    as_strided_495: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_312, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0)
    clone_221: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_495, memory_format = torch.contiguous_format)
    copy_313: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_495, full_default_130);  as_strided_495 = None
    as_strided_scatter_94: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_312, copy_313, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0);  copy_312 = copy_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_125: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_221);  clone_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    slice_scatter_801: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_125, 3, 0, 257);  where_125 = None
    slice_scatter_802: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_801, 2, 0, 9223372036854775807);  slice_scatter_801 = None
    slice_scatter_803: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_802, 1, 0, 256);  slice_scatter_802 = None
    slice_scatter_804: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_803, 0, 0, 9223372036854775807);  slice_scatter_803 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    permute_1303: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_804, [0, 2, 1, 3]);  slice_scatter_804 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_1435: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_1303, [12, 4, 256, 513]);  permute_1303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    add_235: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_94, view_1435);  as_strided_scatter_94 = view_1435 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_75: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_235, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_314: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_75, add_235);  new_empty_strided_75 = add_235 = None
    as_strided_498: "f32[12, 255, 255]" = torch.ops.aten.as_strided.default(copy_314, [12, 255, 255], [525312, 513, 1], 514)
    clone_222: "f32[12, 255, 255]" = torch.ops.aten.clone.default(as_strided_498, memory_format = torch.contiguous_format)
    copy_315: "f32[12, 255, 255]" = torch.ops.aten.copy.default(as_strided_498, full_default_142);  as_strided_498 = None
    as_strided_scatter_95: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_314, copy_315, [12, 255, 255], [525312, 513, 1], 514);  copy_314 = copy_315 = None
    slice_scatter_805: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_143, clone_222, 2, -255, 9223372036854775807);  clone_222 = None
    slice_scatter_806: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_805, 1, 0, 255);  slice_scatter_805 = None
    select_scatter_66: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_806, 1, 0);  slice_scatter_806 = None
    slice_scatter_807: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_66, 0, 0, 9223372036854775807);  select_scatter_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_76: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_95, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_316: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_76, as_strided_scatter_95);  new_empty_strided_76 = as_strided_scatter_95 = None
    as_strided_501: "f32[12, 3, 256, 256]" = torch.ops.aten.as_strided.default(copy_316, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328)
    clone_223: "f32[12, 3, 256, 256]" = torch.ops.aten.clone.default(as_strided_501, memory_format = torch.contiguous_format)
    copy_317: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(as_strided_501, full_default_147);  as_strided_501 = None
    as_strided_scatter_96: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_316, copy_317, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328);  copy_316 = copy_317 = None
    slice_scatter_808: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_223, 3, 257, 9223372036854775807);  clone_223 = None
    slice_scatter_809: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_808, 2, -257, -1);  slice_scatter_808 = None
    slice_scatter_810: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_809, 1, 0, 9223372036854775807);  slice_scatter_809 = None
    slice_scatter_811: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_810, 0, 0, 9223372036854775807);  slice_scatter_810 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    add_236: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(slice_scatter_807, slice_scatter_811);  slice_scatter_807 = slice_scatter_811 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_77: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_96, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_318: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_77, as_strided_scatter_96);  new_empty_strided_77 = as_strided_scatter_96 = None
    as_strided_504: "f32[12, 256, 257]" = torch.ops.aten.as_strided.default(copy_318, [12, 256, 257], [525312, 513, 1], 394240)
    clone_224: "f32[12, 256, 257]" = torch.ops.aten.clone.default(as_strided_504, memory_format = torch.contiguous_format)
    copy_319: "f32[12, 256, 257]" = torch.ops.aten.copy.default(as_strided_504, full_default_152);  as_strided_504 = None
    as_strided_scatter_97: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_318, copy_319, [12, 256, 257], [525312, 513, 1], 394240);  copy_318 = copy_319 = None
    slice_scatter_812: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_153, clone_224, 2, 0, 257);  clone_224 = None
    slice_scatter_813: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_812, 1, 256, 9223372036854775807);  slice_scatter_812 = None
    select_scatter_67: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_813, 1, -1);  slice_scatter_813 = None
    slice_scatter_814: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_67, 0, 0, 9223372036854775807);  select_scatter_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_237: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_236, slice_scatter_814);  add_236 = slice_scatter_814 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_78: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_97, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_320: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_78, as_strided_scatter_97);  new_empty_strided_78 = as_strided_scatter_97 = None
    as_strided_507: "f32[12, 3, 256, 257]" = torch.ops.aten.as_strided.default(copy_320, [12, 3, 256, 257], [525312, 131328, 513, 1], 256);  copy_320 = None
    clone_225: "f32[12, 3, 256, 257]" = torch.ops.aten.clone.default(as_strided_507, memory_format = torch.contiguous_format);  as_strided_507 = None
    slice_scatter_815: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_225, 3, 0, 257);  clone_225 = None
    slice_scatter_816: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_815, 2, 0, 256);  slice_scatter_815 = None
    slice_scatter_817: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_816, 1, 0, 9223372036854775807);  slice_scatter_816 = None
    slice_scatter_818: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_817, 0, 0, 9223372036854775807);  slice_scatter_817 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_238: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_237, slice_scatter_818);  add_237 = slice_scatter_818 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_1436: "f32[12, 3, 513, 512]" = torch.ops.aten.view.default(add_238, [12, 3, 513, 512]);  add_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_77: "f32[12, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1436, [0, 0, 0, -1]);  view_1436 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    view_1437: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_77, [12, 3, 512, 512, 1]);  constant_pad_nd_77 = None
    permute_1304: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1437, [0, 1, 2, 4, 3]);  view_1437 = None
    view_1438: "f32[36, 512, 512]" = torch.ops.aten.view.default(permute_1304, [36, 512, 512]);  permute_1304 = None
    bmm_62: "f32[36, 64, 512]" = torch.ops.aten.bmm.default(permute_1305, view_1438);  permute_1305 = None
    bmm_63: "f32[36, 512, 64]" = torch.ops.aten.bmm.default(view_1438, permute_1306);  view_1438 = permute_1306 = None
    view_1439: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_62, [12, 3, 64, 512, 1]);  bmm_62 = None
    permute_1307: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1439, [0, 1, 4, 3, 2]);  view_1439 = None
    view_1440: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_63, [12, 3, 512, 64, 1]);  bmm_63 = None
    permute_1308: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(view_1440, [0, 1, 2, 4, 3]);  view_1440 = None
    permute_1309: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1307, [0, 1, 3, 4, 2]);  permute_1307 = None
    squeeze_76: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1309, 4);  permute_1309 = None
    permute_1310: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1308, [0, 1, 2, 4, 3]);  permute_1308 = None
    squeeze_77: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1310, 4);  permute_1310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    clone_226: "f32[12, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_76, memory_format = torch.contiguous_format);  squeeze_76 = None
    view_1441: "f32[1179648]" = torch.ops.aten.view.default(clone_226, [1179648]);  clone_226 = None
    index_put_28: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1441, True);  view_1441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    view_1444: "f32[1179648]" = torch.ops.aten.view.default(squeeze_77, [-1]);  squeeze_77 = None
    index_put_29: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1444, True);  view_1444 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    as_strided_512: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_29, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_29 = None
    view_1451: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_512, [12, 1024, 64]);  as_strided_512 = None
    view_1452: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1451, [1, 12, 1024, 64]);  view_1451 = None
    permute_1315: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1452, [0, 2, 1, 3]);  view_1452 = None
    permute_1316: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1315, [1, 0, 2, 3]);  permute_1315 = None
    view_1453: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1316, [1024, 1, 768]);  permute_1316 = None
    squeeze_79: "f32[1024, 768]" = torch.ops.aten.squeeze.dim(view_1453, 1);  view_1453 = None
    copy_321: "f32[1024, 768]" = torch.ops.aten.copy.default(as_strided_108, squeeze_79);  squeeze_79 = None
    as_strided_scatter_98: "f32[786432]" = torch.ops.aten.as_strided_scatter.default(full_default_161, copy_321, [1024, 768], [768, 1], 0);  copy_321 = None
    as_strided_516: "f32[1024, 768]" = torch.ops.aten.as_strided.default(as_strided_scatter_98, [1024, 768], [768, 1], 0);  as_strided_scatter_98 = None
    new_empty_strided_79: "f32[1024, 768]" = torch.ops.aten.new_empty_strided.default(as_strided_516, [1024, 768], [768, 1])
    copy_322: "f32[1024, 768]" = torch.ops.aten.copy.default(new_empty_strided_79, as_strided_516);  new_empty_strided_79 = as_strided_516 = None
    as_strided_518: "f32[1024, 1, 768]" = torch.ops.aten.as_strided.default(copy_322, [1024, 1, 768], [768, 768, 1], 0)
    clone_229: "f32[1024, 1, 768]" = torch.ops.aten.clone.default(as_strided_518, memory_format = torch.contiguous_format)
    div_149: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(clone_229, 8.0);  clone_229 = None
    copy_323: "f32[1024, 1, 768]" = torch.ops.aten.copy.default(as_strided_518, div_149);  as_strided_518 = div_149 = None
    as_strided_scatter_99: "f32[1024, 768]" = torch.ops.aten.as_strided_scatter.default(copy_322, copy_323, [1024, 1, 768], [768, 768, 1], 0);  copy_322 = copy_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_1454: "f32[1024, 768]" = torch.ops.aten.view.default(view_1434, [1024, 768]);  view_1434 = None
    mm_114: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1454, permute_1317);  permute_1317 = None
    permute_1318: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1454, [1, 0])
    mm_115: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1318, view_150);  permute_1318 = None
    permute_1319: "f32[768, 768]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_160: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1454, [0], True);  view_1454 = None
    view_1455: "f32[768]" = torch.ops.aten.view.default(sum_160, [768]);  sum_160 = None
    permute_1320: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1319, [1, 0]);  permute_1319 = None
    view_1456: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_114, [1024, 1, 768]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    as_strided_520: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_28, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_28 = None
    view_1458: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_520, [12, 1024, 64]);  as_strided_520 = None
    view_1459: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1458, [1, 12, 1024, 64]);  view_1458 = None
    permute_1322: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1459, [0, 2, 1, 3]);  view_1459 = None
    permute_1323: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1322, [1, 0, 2, 3]);  permute_1322 = None
    view_1460: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1323, [1024, 1, 768]);  permute_1323 = None
    view_1461: "f32[1024, 768]" = torch.ops.aten.view.default(view_1460, [1024, 768]);  view_1460 = None
    mm_116: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1461, permute_1321);  permute_1321 = None
    permute_1327: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1461, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1327, view_150);  permute_1327 = None
    permute_1328: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_161: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1461, [0], True);  view_1461 = None
    view_1466: "f32[768]" = torch.ops.aten.view.default(sum_161, [768]);  sum_161 = None
    permute_1329: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1328, [1, 0]);  permute_1328 = None
    view_1467: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_116, [1024, 1, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    add_239: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(view_1456, view_1467);  view_1456 = view_1467 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    mm_118: "f32[1024, 768]" = torch.ops.aten.mm.default(as_strided_scatter_99, permute_1330);  permute_1330 = None
    permute_1332: "f32[768, 1024]" = torch.ops.aten.permute.default(as_strided_scatter_99, [1, 0])
    mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1332, view_150);  permute_1332 = view_150 = None
    permute_1333: "f32[768, 768]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_162: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(as_strided_scatter_99, [0], True);  as_strided_scatter_99 = None
    view_1468: "f32[768]" = torch.ops.aten.view.default(sum_162, [768]);  sum_162 = None
    permute_1334: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1333, [1, 0]);  permute_1333 = None
    view_1469: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_118, [1024, 1, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    add_240: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(add_239, view_1469);  add_239 = view_1469 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_1335: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(add_240, [1, 0, 2]);  add_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    add_241: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_378, permute_1335);  mul_378 = permute_1335 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_387: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_241, primals_31);  primals_31 = None
    mul_388: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_387, 768)
    sum_163: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_387, [2], True)
    mul_389: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_387, mul_14);  mul_387 = None
    sum_164: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_389, [2], True);  mul_389 = None
    mul_390: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_14, sum_164);  sum_164 = None
    sub_167: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_388, sum_163);  mul_388 = sum_163 = None
    sub_168: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_167, mul_390);  sub_167 = mul_390 = None
    mul_391: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_150, sub_168);  div_150 = sub_168 = None
    mul_392: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_241, mul_14);  mul_14 = None
    sum_165: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_392, [0, 1]);  mul_392 = None
    sum_166: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_241, [0, 1]);  add_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_90: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_17, torch.float32);  getitem_17 = None
    mul_393: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_90, 1.1111111111111112);  convert_element_type_90 = None
    mul_394: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_391, mul_393);  mul_393 = None
    clone_230: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_394, memory_format = torch.contiguous_format);  mul_394 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_1470: "f32[1024, 768]" = torch.ops.aten.view.default(clone_230, [1024, 768]);  clone_230 = None
    mm_120: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_1470, permute_1336);  permute_1336 = None
    permute_1337: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1470, [1, 0])
    mm_121: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1337, view_148);  permute_1337 = view_148 = None
    permute_1338: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_167: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1470, [0], True);  view_1470 = None
    view_1471: "f32[768]" = torch.ops.aten.view.default(sum_167, [768]);  sum_167 = None
    permute_1339: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1338, [1, 0]);  permute_1338 = None
    view_1472: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_120, [1, 1024, 3072]);  mm_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_396: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_18, 0.5);  add_18 = None
    mul_397: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_147, view_147)
    mul_398: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_397, -0.5);  mul_397 = None
    exp_22: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_398);  mul_398 = None
    mul_399: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_400: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_147, mul_399);  view_147 = mul_399 = None
    add_243: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_396, mul_400);  mul_396 = mul_400 = None
    mul_401: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1472, add_243);  view_1472 = add_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_1473: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_401, [1024, 3072]);  mul_401 = None
    mm_122: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1473, permute_1340);  permute_1340 = None
    permute_1341: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_1473, [1, 0])
    mm_123: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1341, view_146);  permute_1341 = view_146 = None
    permute_1342: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_168: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1473, [0], True);  view_1473 = None
    view_1474: "f32[3072]" = torch.ops.aten.view.default(sum_168, [3072]);  sum_168 = None
    permute_1343: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1342, [1, 0]);  permute_1342 = None
    view_1475: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_122, [1, 1024, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    add_244: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_391, view_1475);  mul_391 = view_1475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_403: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_244, primals_25);  primals_25 = None
    mul_404: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_403, 768)
    sum_169: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_403, [2], True)
    mul_405: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_403, mul_9);  mul_403 = None
    sum_170: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_405, [2], True);  mul_405 = None
    mul_406: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_9, sum_170);  sum_170 = None
    sub_170: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_404, sum_169);  mul_404 = sum_169 = None
    sub_171: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_170, mul_406);  sub_170 = mul_406 = None
    mul_407: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_151, sub_171);  div_151 = sub_171 = None
    mul_408: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_244, mul_9);  mul_9 = None
    sum_171: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_408, [0, 1]);  mul_408 = None
    sum_172: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_244, [0, 1]);  add_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_91: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_13, torch.float32);  getitem_13 = None
    mul_409: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_91, 1.1111111111111112);  convert_element_type_91 = None
    mul_410: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_407, mul_409);  mul_409 = None
    clone_231: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_410, memory_format = torch.contiguous_format);  mul_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_1476: "f32[1024, 768]" = torch.ops.aten.view.default(clone_231, [1024, 768]);  clone_231 = None
    mm_124: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1476, permute_1344);  permute_1344 = None
    permute_1345: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1476, [1, 0])
    mm_125: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1345, view_144);  permute_1345 = view_144 = None
    permute_1346: "f32[768, 768]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_173: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1476, [0], True);  view_1476 = None
    view_1477: "f32[768]" = torch.ops.aten.view.default(sum_173, [768]);  sum_173 = None
    permute_1347: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1346, [1, 0]);  permute_1346 = None
    view_1478: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_124, [1, 1024, 768]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_1348: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(view_1478, [1, 0, 2]);  view_1478 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    view_1479: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(permute_1348, [1024, 1, 12, 64]);  permute_1348 = None
    permute_1349: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1479, [1, 0, 2, 3]);  view_1479 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    permute_1350: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1349, [0, 2, 1, 3]);  permute_1349 = None
    view_1480: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_1350, [12, 4, 256, 64]);  permute_1350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    view_1481: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_1480, [12, 4, 256, 64, 1]);  view_1480 = None
    permute_1351: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1481, [0, 1, 2, 4, 3]);  view_1481 = None
    clone_232: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.clone.default(permute_1351, memory_format = torch.contiguous_format);  permute_1351 = None
    view_1482: "f32[48, 256, 64]" = torch.ops.aten.view.default(clone_232, [48, 256, 64]);  clone_232 = None
    bmm_64: "f32[48, 768, 64]" = torch.ops.aten.bmm.default(permute_1352, view_1482);  permute_1352 = None
    bmm_65: "f32[48, 256, 768]" = torch.ops.aten.bmm.default(view_1482, permute_1353);  view_1482 = permute_1353 = None
    view_1483: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_64, [12, 4, 768, 64, 1]);  bmm_64 = None
    permute_1354: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(view_1483, [0, 1, 4, 3, 2]);  view_1483 = None
    view_1484: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_65, [12, 4, 256, 768, 1]);  bmm_65 = None
    permute_1355: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(view_1484, [0, 1, 2, 4, 3]);  view_1484 = None
    permute_1356: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_1354, [0, 1, 4, 3, 2]);  permute_1354 = None
    squeeze_80: "f32[12, 4, 768, 64]" = torch.ops.aten.squeeze.dim(permute_1356, 4);  permute_1356 = None
    permute_1357: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_1355, [0, 1, 2, 4, 3]);  permute_1355 = None
    squeeze_81: "f32[12, 4, 256, 768]" = torch.ops.aten.squeeze.dim(permute_1357, 4);  permute_1357 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_scatter_819: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, squeeze_81, 3, 0, -1);  squeeze_81 = None
    slice_scatter_820: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_819, 2, 0, 9223372036854775807);  slice_scatter_819 = None
    slice_scatter_821: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_820, 1, 0, 9223372036854775807);  slice_scatter_820 = None
    slice_scatter_822: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_821, 0, 0, 9223372036854775807);  slice_scatter_821 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1485: "f32[12, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_822, [12, 4, 196864]);  slice_scatter_822 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_scatter_823: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, view_1485, 2, 0, -256);  view_1485 = None
    slice_scatter_824: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_823, 1, 0, 9223372036854775807);  slice_scatter_823 = None
    slice_scatter_825: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_824, 0, 0, 9223372036854775807);  slice_scatter_824 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1486: "f32[12, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_825, [12, 4, 256, 770]);  slice_scatter_825 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_78: "f32[12, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1486, [0, -257]);  view_1486 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    view_1487: "f32[2359296]" = torch.ops.aten.view.default(squeeze_80, [-1]);  squeeze_80 = None
    index_put_30: "f32[1179648]" = torch.ops.aten.index_put.default(full_default_128, [view_918], view_1487, True);  view_1487 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    as_strided_524: "f32[12, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_30, [12, 1536, 64], [98304, 64, 1], 0);  index_put_30 = None
    constant_pad_nd_79: "f32[12, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_524, [0, 0, -256, -256]);  as_strided_524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    view_1489: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_79, [1, 12, 1024, 64]);  constant_pad_nd_79 = None
    permute_1358: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1489, [0, 2, 1, 3]);  view_1489 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    view_1490: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_78, [1, 12, 1024, 513]);  constant_pad_nd_78 = None
    permute_1359: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1490, [0, 2, 1, 3]);  view_1490 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    permute_1360: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1358, [1, 0, 2, 3]);  permute_1358 = None
    clone_234: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_1360, memory_format = torch.contiguous_format);  permute_1360 = None
    view_1491: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_234, [1024, 1, 768]);  clone_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    convert_element_type_92: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(getitem_11, torch.float32);  getitem_11 = None
    mul_411: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_92, 1.1111111111111112);  convert_element_type_92 = None
    mul_412: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1359, mul_411);  permute_1359 = mul_411 = None
    clone_235: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_412, memory_format = torch.contiguous_format);  mul_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_126: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, clone_235);  clone_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    mul_413: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_126, alias_22);  where_126 = None
    sum_174: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [-1], True)
    mul_414: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(alias_22, sum_174);  alias_22 = sum_174 = None
    sub_172: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(mul_413, mul_414);  mul_413 = mul_414 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    squeeze_82: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(sub_172, 0);  sub_172 = None
    copy_324: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_82);  squeeze_82 = None
    as_strided_scatter_100: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_324, [1024, 12, 513], [513, 525312, 1], 0);  copy_324 = None
    as_strided_528: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_100, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_100 = None
    new_empty_strided_80: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_528, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_325: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_80, as_strided_528);  new_empty_strided_80 = as_strided_528 = None
    as_strided_530: "f32[1, 1024, 12, 513]" = torch.ops.aten.as_strided.default(copy_325, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0)
    clone_236: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(as_strided_530, memory_format = torch.contiguous_format)
    copy_326: "f32[1, 1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_530, clone_236);  as_strided_530 = clone_236 = None
    as_strided_scatter_101: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_325, copy_326, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0);  copy_325 = copy_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    new_empty_strided_81: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_101, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_327: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_81, as_strided_scatter_101);  new_empty_strided_81 = as_strided_scatter_101 = None
    as_strided_533: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_327, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240)
    clone_237: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_533, memory_format = torch.contiguous_format)
    copy_328: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_533, full_default_130);  as_strided_533 = None
    as_strided_scatter_102: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_327, copy_328, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240);  copy_327 = copy_328 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_127: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_237);  clone_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    slice_scatter_826: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_127, 3, -257, 9223372036854775807);  where_127 = None
    slice_scatter_827: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_826, 2, 0, 9223372036854775807);  slice_scatter_826 = None
    slice_scatter_828: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_827, 1, -256, 9223372036854775807);  slice_scatter_827 = None
    slice_scatter_829: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_828, 0, 0, 9223372036854775807);  slice_scatter_828 = None
    squeeze_83: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(slice_scatter_829, 0);  slice_scatter_829 = None
    copy_329: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_83);  squeeze_83 = None
    as_strided_scatter_103: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_329, [1024, 12, 513], [513, 525312, 1], 0);  copy_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    as_strided_538: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_103, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_103 = None
    add_245: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_102, as_strided_538);  as_strided_scatter_102 = as_strided_538 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    new_empty_strided_82: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_245, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_330: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_82, add_245);  new_empty_strided_82 = add_245 = None
    as_strided_540: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_330, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0)
    clone_238: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_540, memory_format = torch.contiguous_format)
    copy_331: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_540, full_default_130);  as_strided_540 = None
    as_strided_scatter_104: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_330, copy_331, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0);  copy_330 = copy_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_128: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_238);  clone_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    slice_scatter_830: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_128, 3, 0, 257);  where_128 = None
    slice_scatter_831: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_830, 2, 0, 9223372036854775807);  slice_scatter_830 = None
    slice_scatter_832: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_831, 1, 0, 256);  slice_scatter_831 = None
    slice_scatter_833: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_832, 0, 0, 9223372036854775807);  slice_scatter_832 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    permute_1361: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_833, [0, 2, 1, 3]);  slice_scatter_833 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_1492: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_1361, [12, 4, 256, 513]);  permute_1361 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    add_246: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_104, view_1492);  as_strided_scatter_104 = view_1492 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_83: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_246, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_332: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_83, add_246);  new_empty_strided_83 = add_246 = None
    as_strided_543: "f32[12, 255, 255]" = torch.ops.aten.as_strided.default(copy_332, [12, 255, 255], [525312, 513, 1], 514)
    clone_239: "f32[12, 255, 255]" = torch.ops.aten.clone.default(as_strided_543, memory_format = torch.contiguous_format)
    copy_333: "f32[12, 255, 255]" = torch.ops.aten.copy.default(as_strided_543, full_default_142);  as_strided_543 = None
    as_strided_scatter_105: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_332, copy_333, [12, 255, 255], [525312, 513, 1], 514);  copy_332 = copy_333 = None
    slice_scatter_834: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_143, clone_239, 2, -255, 9223372036854775807);  clone_239 = None
    slice_scatter_835: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_834, 1, 0, 255);  slice_scatter_834 = None
    select_scatter_68: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_835, 1, 0);  slice_scatter_835 = None
    slice_scatter_836: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_68, 0, 0, 9223372036854775807);  select_scatter_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_84: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_105, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_334: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_84, as_strided_scatter_105);  new_empty_strided_84 = as_strided_scatter_105 = None
    as_strided_546: "f32[12, 3, 256, 256]" = torch.ops.aten.as_strided.default(copy_334, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328)
    clone_240: "f32[12, 3, 256, 256]" = torch.ops.aten.clone.default(as_strided_546, memory_format = torch.contiguous_format)
    copy_335: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(as_strided_546, full_default_147);  as_strided_546 = None
    as_strided_scatter_106: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_334, copy_335, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328);  copy_334 = copy_335 = None
    slice_scatter_837: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_240, 3, 257, 9223372036854775807);  clone_240 = None
    slice_scatter_838: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_837, 2, -257, -1);  slice_scatter_837 = None
    slice_scatter_839: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_838, 1, 0, 9223372036854775807);  slice_scatter_838 = None
    slice_scatter_840: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_839, 0, 0, 9223372036854775807);  slice_scatter_839 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    add_247: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(slice_scatter_836, slice_scatter_840);  slice_scatter_836 = slice_scatter_840 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_85: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_106, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_336: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_85, as_strided_scatter_106);  new_empty_strided_85 = as_strided_scatter_106 = None
    as_strided_549: "f32[12, 256, 257]" = torch.ops.aten.as_strided.default(copy_336, [12, 256, 257], [525312, 513, 1], 394240)
    clone_241: "f32[12, 256, 257]" = torch.ops.aten.clone.default(as_strided_549, memory_format = torch.contiguous_format)
    copy_337: "f32[12, 256, 257]" = torch.ops.aten.copy.default(as_strided_549, full_default_152);  as_strided_549 = None
    as_strided_scatter_107: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_336, copy_337, [12, 256, 257], [525312, 513, 1], 394240);  copy_336 = copy_337 = None
    slice_scatter_841: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_153, clone_241, 2, 0, 257);  clone_241 = None
    slice_scatter_842: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_841, 1, 256, 9223372036854775807);  slice_scatter_841 = None
    select_scatter_69: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_842, 1, -1);  slice_scatter_842 = None
    slice_scatter_843: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_69, 0, 0, 9223372036854775807);  select_scatter_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_248: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_247, slice_scatter_843);  add_247 = slice_scatter_843 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_86: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_107, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_338: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_86, as_strided_scatter_107);  new_empty_strided_86 = as_strided_scatter_107 = None
    as_strided_552: "f32[12, 3, 256, 257]" = torch.ops.aten.as_strided.default(copy_338, [12, 3, 256, 257], [525312, 131328, 513, 1], 256);  copy_338 = None
    clone_242: "f32[12, 3, 256, 257]" = torch.ops.aten.clone.default(as_strided_552, memory_format = torch.contiguous_format);  as_strided_552 = None
    slice_scatter_844: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_242, 3, 0, 257);  clone_242 = None
    slice_scatter_845: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_844, 2, 0, 256);  slice_scatter_844 = None
    slice_scatter_846: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_845, 1, 0, 9223372036854775807);  slice_scatter_845 = None
    slice_scatter_847: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_846, 0, 0, 9223372036854775807);  slice_scatter_846 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_249: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_248, slice_scatter_847);  add_248 = slice_scatter_847 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_1493: "f32[12, 3, 513, 512]" = torch.ops.aten.view.default(add_249, [12, 3, 513, 512]);  add_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_80: "f32[12, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1493, [0, 0, 0, -1]);  view_1493 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    view_1494: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_80, [12, 3, 512, 512, 1]);  constant_pad_nd_80 = None
    permute_1362: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1494, [0, 1, 2, 4, 3]);  view_1494 = None
    view_1495: "f32[36, 512, 512]" = torch.ops.aten.view.default(permute_1362, [36, 512, 512]);  permute_1362 = None
    bmm_66: "f32[36, 64, 512]" = torch.ops.aten.bmm.default(permute_1363, view_1495);  permute_1363 = None
    bmm_67: "f32[36, 512, 64]" = torch.ops.aten.bmm.default(view_1495, permute_1364);  view_1495 = permute_1364 = None
    view_1496: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_66, [12, 3, 64, 512, 1]);  bmm_66 = None
    permute_1365: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1496, [0, 1, 4, 3, 2]);  view_1496 = None
    view_1497: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_67, [12, 3, 512, 64, 1]);  bmm_67 = None
    permute_1366: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(view_1497, [0, 1, 2, 4, 3]);  view_1497 = None
    permute_1367: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1365, [0, 1, 3, 4, 2]);  permute_1365 = None
    squeeze_84: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1367, 4);  permute_1367 = None
    permute_1368: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1366, [0, 1, 2, 4, 3]);  permute_1366 = None
    squeeze_85: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1368, 4);  permute_1368 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    clone_243: "f32[12, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_84, memory_format = torch.contiguous_format);  squeeze_84 = None
    view_1498: "f32[1179648]" = torch.ops.aten.view.default(clone_243, [1179648]);  clone_243 = None
    index_put_31: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1498, True);  view_1498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    view_1501: "f32[1179648]" = torch.ops.aten.view.default(squeeze_85, [-1]);  squeeze_85 = None
    index_put_32: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1501, True);  view_1501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    as_strided_557: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_32, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_32 = None
    view_1508: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_557, [12, 1024, 64]);  as_strided_557 = None
    view_1509: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1508, [1, 12, 1024, 64]);  view_1508 = None
    permute_1373: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1509, [0, 2, 1, 3]);  view_1509 = None
    permute_1374: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1373, [1, 0, 2, 3]);  permute_1373 = None
    view_1510: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1374, [1024, 1, 768]);  permute_1374 = None
    squeeze_87: "f32[1024, 768]" = torch.ops.aten.squeeze.dim(view_1510, 1);  view_1510 = None
    copy_339: "f32[1024, 768]" = torch.ops.aten.copy.default(as_strided_108, squeeze_87);  squeeze_87 = None
    as_strided_scatter_108: "f32[786432]" = torch.ops.aten.as_strided_scatter.default(full_default_161, copy_339, [1024, 768], [768, 1], 0);  copy_339 = None
    as_strided_561: "f32[1024, 768]" = torch.ops.aten.as_strided.default(as_strided_scatter_108, [1024, 768], [768, 1], 0);  as_strided_scatter_108 = None
    new_empty_strided_87: "f32[1024, 768]" = torch.ops.aten.new_empty_strided.default(as_strided_561, [1024, 768], [768, 1])
    copy_340: "f32[1024, 768]" = torch.ops.aten.copy.default(new_empty_strided_87, as_strided_561);  new_empty_strided_87 = as_strided_561 = None
    as_strided_563: "f32[1024, 1, 768]" = torch.ops.aten.as_strided.default(copy_340, [1024, 1, 768], [768, 768, 1], 0)
    clone_246: "f32[1024, 1, 768]" = torch.ops.aten.clone.default(as_strided_563, memory_format = torch.contiguous_format)
    div_152: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(clone_246, 8.0);  clone_246 = None
    copy_341: "f32[1024, 1, 768]" = torch.ops.aten.copy.default(as_strided_563, div_152);  as_strided_563 = div_152 = None
    as_strided_scatter_109: "f32[1024, 768]" = torch.ops.aten.as_strided_scatter.default(copy_340, copy_341, [1024, 1, 768], [768, 768, 1], 0);  copy_340 = copy_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_1511: "f32[1024, 768]" = torch.ops.aten.view.default(view_1491, [1024, 768]);  view_1491 = None
    mm_126: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1511, permute_1375);  permute_1375 = None
    permute_1376: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1511, [1, 0])
    mm_127: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1376, view_75);  permute_1376 = None
    permute_1377: "f32[768, 768]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_175: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1511, [0], True);  view_1511 = None
    view_1512: "f32[768]" = torch.ops.aten.view.default(sum_175, [768]);  sum_175 = None
    permute_1378: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1377, [1, 0]);  permute_1377 = None
    view_1513: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_126, [1024, 1, 768]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    as_strided_565: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_31, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_31 = None
    view_1515: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_565, [12, 1024, 64]);  as_strided_565 = None
    view_1516: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1515, [1, 12, 1024, 64]);  view_1515 = None
    permute_1380: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1516, [0, 2, 1, 3]);  view_1516 = None
    permute_1381: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1380, [1, 0, 2, 3]);  permute_1380 = None
    view_1517: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1381, [1024, 1, 768]);  permute_1381 = None
    view_1518: "f32[1024, 768]" = torch.ops.aten.view.default(view_1517, [1024, 768]);  view_1517 = None
    mm_128: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1518, permute_1379);  permute_1379 = None
    permute_1385: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1518, [1, 0])
    mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1385, view_75);  permute_1385 = None
    permute_1386: "f32[768, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_176: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1518, [0], True);  view_1518 = None
    view_1523: "f32[768]" = torch.ops.aten.view.default(sum_176, [768]);  sum_176 = None
    permute_1387: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1386, [1, 0]);  permute_1386 = None
    view_1524: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_128, [1024, 1, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    add_250: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(view_1513, view_1524);  view_1513 = view_1524 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    mm_130: "f32[1024, 768]" = torch.ops.aten.mm.default(as_strided_scatter_109, permute_1388);  permute_1388 = None
    permute_1390: "f32[768, 1024]" = torch.ops.aten.permute.default(as_strided_scatter_109, [1, 0])
    mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1390, view_75);  permute_1390 = view_75 = None
    permute_1391: "f32[768, 768]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_177: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(as_strided_scatter_109, [0], True);  as_strided_scatter_109 = None
    view_1525: "f32[768]" = torch.ops.aten.view.default(sum_177, [768]);  sum_177 = None
    permute_1392: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1391, [1, 0]);  permute_1391 = None
    view_1526: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_130, [1024, 1, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    add_251: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(add_250, view_1526);  add_250 = view_1526 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_1393: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(add_251, [1, 0, 2]);  add_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    add_252: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_407, permute_1393);  mul_407 = permute_1393 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1223, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_416: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_252, primals_15);  primals_15 = None
    mul_417: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_416, 768)
    sum_178: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_416, [2], True)
    mul_418: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_416, mul_6);  mul_416 = None
    sum_179: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_418, [2], True);  mul_418 = None
    mul_419: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_6, sum_179);  sum_179 = None
    sub_174: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_417, sum_178);  mul_417 = sum_178 = None
    sub_175: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_174, mul_419);  sub_174 = mul_419 = None
    mul_420: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_153, sub_175);  div_153 = sub_175 = None
    mul_421: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_252, mul_6);  mul_6 = None
    sum_180: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_421, [0, 1]);  mul_421 = None
    sum_181: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_252, [0, 1]);  add_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1222, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_93: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_7, torch.float32);  getitem_7 = None
    mul_422: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_93, 1.1111111111111112);  convert_element_type_93 = None
    mul_423: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_420, mul_422);  mul_422 = None
    clone_247: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_423, memory_format = torch.contiguous_format);  mul_423 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1221, code: hidden_states = self.dense(hidden_states)
    view_1527: "f32[1024, 768]" = torch.ops.aten.view.default(clone_247, [1024, 768]);  clone_247 = None
    mm_132: "f32[1024, 3072]" = torch.ops.aten.mm.default(view_1527, permute_1394);  permute_1394 = None
    permute_1395: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1527, [1, 0])
    mm_133: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_1395, view_73);  permute_1395 = view_73 = None
    permute_1396: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_182: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1527, [0], True);  view_1527 = None
    view_1528: "f32[768]" = torch.ops.aten.view.default(sum_182, [768]);  sum_182 = None
    permute_1397: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_1396, [1, 0]);  permute_1396 = None
    view_1529: "f32[1, 1024, 3072]" = torch.ops.aten.view.default(mm_132, [1, 1024, 3072]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    mul_425: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(add_7, 0.5);  add_7 = None
    mul_426: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_72, view_72)
    mul_427: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(mul_426, -0.5);  mul_426 = None
    exp_23: "f32[1, 1024, 3072]" = torch.ops.aten.exp.default(mul_427);  mul_427 = None
    mul_428: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_429: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_72, mul_428);  view_72 = mul_428 = None
    add_254: "f32[1, 1024, 3072]" = torch.ops.aten.add.Tensor(mul_425, mul_429);  mul_425 = mul_429 = None
    mul_430: "f32[1, 1024, 3072]" = torch.ops.aten.mul.Tensor(view_1529, add_254);  view_1529 = add_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    view_1530: "f32[1024, 3072]" = torch.ops.aten.view.default(mul_430, [1024, 3072]);  mul_430 = None
    mm_134: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1530, permute_1398);  permute_1398 = None
    permute_1399: "f32[3072, 1024]" = torch.ops.aten.permute.default(view_1530, [1, 0])
    mm_135: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_1399, view_71);  permute_1399 = view_71 = None
    permute_1400: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_183: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_1530, [0], True);  view_1530 = None
    view_1531: "f32[3072]" = torch.ops.aten.view.default(sum_183, [3072]);  sum_183 = None
    permute_1401: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_1400, [1, 0]);  permute_1400 = None
    view_1532: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_134, [1, 1024, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1207, code: hidden_states = self.dense(hidden_states)
    add_255: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_420, view_1532);  mul_420 = view_1532 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1143, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
    mul_432: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_255, primals_9);  primals_9 = None
    mul_433: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_432, 768)
    sum_184: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_432, [2], True)
    mul_434: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_432, mul_1);  mul_432 = None
    sum_185: "f32[1, 1024, 1]" = torch.ops.aten.sum.dim_IntList(mul_434, [2], True);  mul_434 = None
    mul_435: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_1, sum_185);  sum_185 = None
    sub_177: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(mul_433, sum_184);  mul_433 = sum_184 = None
    sub_178: "f32[1, 1024, 768]" = torch.ops.aten.sub.Tensor(sub_177, mul_435);  sub_177 = mul_435 = None
    mul_436: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(div_154, sub_178);  div_154 = sub_178 = None
    mul_437: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(add_255, mul_1);  mul_1 = None
    sum_186: "f32[768]" = torch.ops.aten.sum.dim_IntList(mul_437, [0, 1]);  mul_437 = None
    sum_187: "f32[768]" = torch.ops.aten.sum.dim_IntList(add_255, [0, 1]);  add_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1142, code: hidden_states = self.dropout(hidden_states)
    convert_element_type_94: "f32[1, 1024, 768]" = torch.ops.prims.convert_element_type.default(getitem_3, torch.float32);  getitem_3 = None
    mul_438: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(convert_element_type_94, 1.1111111111111112);  convert_element_type_94 = None
    mul_439: "f32[1, 1024, 768]" = torch.ops.aten.mul.Tensor(mul_436, mul_438);  mul_438 = None
    clone_248: "f32[1, 1024, 768]" = torch.ops.aten.clone.default(mul_439, memory_format = torch.contiguous_format);  mul_439 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:1141, code: hidden_states = self.dense(hidden_states)
    view_1533: "f32[1024, 768]" = torch.ops.aten.view.default(clone_248, [1024, 768]);  clone_248 = None
    mm_136: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1533, permute_1402);  permute_1402 = None
    permute_1403: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1533, [1, 0])
    mm_137: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1403, view_69);  permute_1403 = view_69 = None
    permute_1404: "f32[768, 768]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_188: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1533, [0], True);  view_1533 = None
    view_1534: "f32[768]" = torch.ops.aten.view.default(sum_188, [768]);  sum_188 = None
    permute_1405: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1404, [1, 0]);  permute_1404 = None
    view_1535: "f32[1, 1024, 768]" = torch.ops.aten.view.default(mm_136, [1, 1024, 768]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:694, code: outputs = (attn_output.transpose(0, 1),)
    permute_1406: "f32[1024, 1, 768]" = torch.ops.aten.permute.default(view_1535, [1, 0, 2]);  view_1535 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:665, code: attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
    view_1536: "f32[1024, 1, 12, 64]" = torch.ops.aten.view.default(permute_1406, [1024, 1, 12, 64]);  permute_1406 = None
    permute_1407: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1536, [1, 0, 2, 3]);  view_1536 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:926, code: return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    permute_1408: "f32[1, 12, 1024, 64]" = torch.ops.aten.permute.default(permute_1407, [0, 2, 1, 3]);  permute_1407 = None
    view_1537: "f32[12, 4, 256, 64]" = torch.ops.aten.view.default(permute_1408, [12, 4, 256, 64]);  permute_1408 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:925, code: context = torch.einsum("bcwd,bcdh->bcwh", (chunked_attn_probs, chunked_value))
    view_1538: "f32[12, 4, 256, 64, 1]" = torch.ops.aten.view.default(view_1537, [12, 4, 256, 64, 1]);  view_1537 = None
    permute_1409: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.permute.default(view_1538, [0, 1, 2, 4, 3]);  view_1538 = None
    clone_249: "f32[12, 4, 256, 1, 64]" = torch.ops.aten.clone.default(permute_1409, memory_format = torch.contiguous_format);  permute_1409 = None
    view_1539: "f32[48, 256, 64]" = torch.ops.aten.view.default(clone_249, [48, 256, 64]);  clone_249 = None
    bmm_68: "f32[48, 768, 64]" = torch.ops.aten.bmm.default(permute_1410, view_1539);  permute_1410 = None
    bmm_69: "f32[48, 256, 768]" = torch.ops.aten.bmm.default(view_1539, permute_1411);  view_1539 = permute_1411 = None
    view_1540: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.view.default(bmm_68, [12, 4, 768, 64, 1]);  bmm_68 = None
    permute_1412: "f32[12, 4, 1, 64, 768]" = torch.ops.aten.permute.default(view_1540, [0, 1, 4, 3, 2]);  view_1540 = None
    view_1541: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.view.default(bmm_69, [12, 4, 256, 768, 1]);  bmm_69 = None
    permute_1413: "f32[12, 4, 256, 1, 768]" = torch.ops.aten.permute.default(view_1541, [0, 1, 2, 4, 3]);  view_1541 = None
    permute_1414: "f32[12, 4, 768, 64, 1]" = torch.ops.aten.permute.default(permute_1412, [0, 1, 4, 3, 2]);  permute_1412 = None
    squeeze_88: "f32[12, 4, 768, 64]" = torch.ops.aten.squeeze.dim(permute_1414, 4);  permute_1414 = None
    permute_1415: "f32[12, 4, 256, 768, 1]" = torch.ops.aten.permute.default(permute_1413, [0, 1, 2, 4, 3]);  permute_1413 = None
    squeeze_89: "f32[12, 4, 256, 768]" = torch.ops.aten.squeeze.dim(permute_1415, 4);  permute_1415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:758, code: chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
    slice_scatter_848: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, squeeze_89, 3, 0, -1);  squeeze_89 = None
    slice_scatter_849: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_848, 2, 0, 9223372036854775807);  slice_scatter_848 = None
    slice_scatter_850: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_849, 1, 0, 9223372036854775807);  slice_scatter_849 = None
    slice_scatter_851: "f32[12, 4, 256, 769]" = torch.ops.aten.slice_scatter.default(full_default_121, slice_scatter_850, 0, 0, 9223372036854775807);  full_default_121 = slice_scatter_850 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:755, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1542: "f32[12, 4, 196864]" = torch.ops.aten.view.default(slice_scatter_851, [12, 4, 196864]);  slice_scatter_851 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:752, code: chunked_hidden_states = chunked_hidden_states[
    slice_scatter_852: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, view_1542, 2, 0, -256);  view_1542 = None
    slice_scatter_853: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_852, 1, 0, 9223372036854775807);  slice_scatter_852 = None
    slice_scatter_854: "f32[12, 4, 197120]" = torch.ops.aten.slice_scatter.default(full_default_125, slice_scatter_853, 0, 0, 9223372036854775807);  full_default_125 = slice_scatter_853 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:749, code: chunked_hidden_states = chunked_hidden_states.view(
    view_1543: "f32[12, 4, 256, 770]" = torch.ops.aten.view.default(slice_scatter_854, [12, 4, 256, 770]);  slice_scatter_854 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:746, code: chunked_hidden_states = nn.functional.pad(
    constant_pad_nd_81: "f32[12, 4, 256, 513]" = torch.ops.aten.constant_pad_nd.default(view_1543, [0, -257]);  view_1543 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:921, code: chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
    view_1544: "f32[2359296]" = torch.ops.aten.view.default(squeeze_88, [-1]);  squeeze_88 = None
    index_put_33: "f32[1179648]" = torch.ops.aten.index_put.default(full_default_128, [view_918], view_1544, True);  full_default_128 = view_918 = view_1544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:910, code: padded_value = nn.functional.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
    as_strided_569: "f32[12, 1536, 64]" = torch.ops.aten.as_strided.default(index_put_33, [12, 1536, 64], [98304, 64, 1], 0);  index_put_33 = None
    constant_pad_nd_82: "f32[12, 1024, 64]" = torch.ops.aten.constant_pad_nd.default(as_strided_569, [0, 0, -256, -256]);  as_strided_569 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:907, code: value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
    view_1546: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(constant_pad_nd_82, [1, 12, 1024, 64]);  constant_pad_nd_82 = None
    permute_1416: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1546, [0, 2, 1, 3]);  view_1546 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:899, code: chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
    view_1547: "f32[1, 12, 1024, 513]" = torch.ops.aten.view.default(constant_pad_nd_81, [1, 12, 1024, 513]);  constant_pad_nd_81 = None
    permute_1417: "f32[1, 1024, 12, 513]" = torch.ops.aten.permute.default(view_1547, [0, 2, 1, 3]);  view_1547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:646, code: value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
    permute_1418: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1416, [1, 0, 2, 3]);  permute_1416 = None
    clone_251: "f32[1024, 1, 12, 64]" = torch.ops.aten.clone.default(permute_1418, memory_format = torch.contiguous_format);  permute_1418 = None
    view_1548: "f32[1024, 1, 768]" = torch.ops.aten.view.default(clone_251, [1024, 1, 768]);  clone_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:644, code: attn_probs = nn.functional.dropout(attn_probs, p=self.dropout, training=self.training)
    convert_element_type_95: "f32[1, 1024, 12, 513]" = torch.ops.prims.convert_element_type.default(getitem_1, torch.float32);  getitem_1 = None
    mul_440: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(convert_element_type_95, 1.1111111111111112);  convert_element_type_95 = None
    mul_441: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(permute_1417, mul_440);  permute_1417 = mul_440 = None
    clone_252: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(mul_441, memory_format = torch.contiguous_format);  mul_441 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:637, code: attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
    where_129: "f32[1, 1024, 12, 513]" = torch.ops.aten.where.self(unsqueeze_16, full_default_1, clone_252);  unsqueeze_16 = clone_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:626, code: attn_probs = nn.functional.softmax(
    mul_442: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(where_129, alias_23);  where_129 = None
    sum_189: "f32[1, 1024, 12, 1]" = torch.ops.aten.sum.dim_IntList(mul_442, [-1], True)
    mul_443: "f32[1, 1024, 12, 513]" = torch.ops.aten.mul.Tensor(alias_23, sum_189);  alias_23 = sum_189 = None
    sub_179: "f32[1, 1024, 12, 513]" = torch.ops.aten.sub.Tensor(mul_442, mul_443);  mul_442 = mul_443 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:588, code: attn_scores += diagonal_mask
    squeeze_90: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(sub_179, 0);  sub_179 = None
    copy_342: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_90);  squeeze_90 = None
    as_strided_scatter_110: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_342, [1024, 12, 513], [513, 525312, 1], 0);  copy_342 = None
    as_strided_573: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_110, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_110 = None
    new_empty_strided_88: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_573, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_343: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_88, as_strided_573);  new_empty_strided_88 = as_strided_573 = None
    as_strided_575: "f32[1, 1024, 12, 513]" = torch.ops.aten.as_strided.default(copy_343, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0)
    clone_253: "f32[1, 1024, 12, 513]" = torch.ops.aten.clone.default(as_strided_575, memory_format = torch.contiguous_format)
    copy_344: "f32[1, 1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_575, clone_253);  as_strided_575 = clone_253 = None
    as_strided_scatter_111: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_343, copy_344, [1, 1024, 12, 513], [6303744, 513, 525312, 1], 0);  copy_343 = copy_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:814, code: input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :] = torch.full_like(
    new_empty_strided_89: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_111, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_345: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_89, as_strided_scatter_111);  new_empty_strided_89 = as_strided_scatter_111 = None
    as_strided_578: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_345, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240)
    clone_254: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_578, memory_format = torch.contiguous_format)
    copy_346: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_578, full_default_130);  as_strided_578 = None
    as_strided_scatter_112: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_345, copy_346, [1, 256, 12, 257], [6303744, 513, 525312, 1], 394240);  copy_345 = copy_346 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:816, code: ).where(ending_mask.bool(), ending_input)
    where_130: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type_1, full_default_1, clone_254);  convert_element_type_1 = clone_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    slice_scatter_855: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_130, 3, -257, 9223372036854775807);  where_130 = None
    slice_scatter_856: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_855, 2, 0, 9223372036854775807);  slice_scatter_855 = None
    slice_scatter_857: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_856, 1, -256, 9223372036854775807);  slice_scatter_856 = None
    slice_scatter_858: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_857, 0, 0, 9223372036854775807);  slice_scatter_857 = None
    squeeze_91: "f32[1024, 12, 513]" = torch.ops.aten.squeeze.dim(slice_scatter_858, 0);  slice_scatter_858 = None
    copy_347: "f32[1024, 12, 513]" = torch.ops.aten.copy.default(as_strided_75, squeeze_91);  as_strided_75 = squeeze_91 = None
    as_strided_scatter_113: "f32[6303744]" = torch.ops.aten.as_strided_scatter.default(full_117, copy_347, [1024, 12, 513], [513, 525312, 1], 0);  full_117 = copy_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:812, code: ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1) :]
    as_strided_583: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided.default(as_strided_scatter_113, [12, 4, 256, 513], [525312, 131328, 513, 1], 0);  as_strided_scatter_113 = None
    add_256: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_112, as_strided_583);  as_strided_scatter_112 = as_strided_583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:809, code: input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1] = torch.full_like(
    new_empty_strided_90: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_256, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_348: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_90, add_256);  new_empty_strided_90 = add_256 = None
    as_strided_585: "f32[1, 256, 12, 257]" = torch.ops.aten.as_strided.default(copy_348, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0)
    clone_255: "f32[1, 256, 12, 257]" = torch.ops.aten.clone.default(as_strided_585, memory_format = torch.contiguous_format)
    copy_349: "f32[1, 256, 12, 257]" = torch.ops.aten.copy.default(as_strided_585, full_default_130);  as_strided_585 = full_default_130 = None
    as_strided_scatter_114: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_348, copy_349, [1, 256, 12, 257], [6303744, 513, 525312, 1], 0);  copy_348 = copy_349 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:811, code: ).where(beginning_mask.bool(), beginning_input)
    where_131: "f32[1, 256, 12, 257]" = torch.ops.aten.where.self(convert_element_type, full_default_1, clone_255);  convert_element_type = full_default_1 = clone_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:807, code: beginning_input = input_tensor[:, :affected_seq_len, :, : affected_seq_len + 1]
    slice_scatter_859: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, where_131, 3, 0, 257);  where_131 = None
    slice_scatter_860: "f32[1, 256, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_132, slice_scatter_859, 2, 0, 9223372036854775807);  full_default_132 = slice_scatter_859 = None
    slice_scatter_861: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_860, 1, 0, 256);  slice_scatter_860 = None
    slice_scatter_862: "f32[1, 1024, 12, 513]" = torch.ops.aten.slice_scatter.default(full_default_134, slice_scatter_861, 0, 0, 9223372036854775807);  full_default_134 = slice_scatter_861 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:879, code: ).transpose(2, 1)
    permute_1419: "f32[1, 12, 1024, 513]" = torch.ops.aten.permute.default(slice_scatter_862, [0, 2, 1, 3]);  slice_scatter_862 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    view_1549: "f32[12, 4, 256, 513]" = torch.ops.aten.view.default(permute_1419, [12, 4, 256, 513]);  permute_1419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:877, code: diagonal_attention_scores = diagonal_attention_scores.view(
    add_257: "f32[12, 4, 256, 513]" = torch.ops.aten.add.Tensor(as_strided_scatter_114, view_1549);  as_strided_scatter_114 = view_1549 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:872, code: diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_91: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(add_257, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_350: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_91, add_257);  new_empty_strided_91 = add_257 = None
    as_strided_588: "f32[12, 255, 255]" = torch.ops.aten.as_strided.default(copy_350, [12, 255, 255], [525312, 513, 1], 514)
    clone_256: "f32[12, 255, 255]" = torch.ops.aten.clone.default(as_strided_588, memory_format = torch.contiguous_format)
    copy_351: "f32[12, 255, 255]" = torch.ops.aten.copy.default(as_strided_588, full_default_142);  as_strided_588 = full_default_142 = None
    as_strided_scatter_115: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_350, copy_351, [12, 255, 255], [525312, 513, 1], 514);  copy_350 = copy_351 = None
    slice_scatter_863: "f32[12, 255, 513]" = torch.ops.aten.slice_scatter.default(full_default_143, clone_256, 2, -255, 9223372036854775807);  full_default_143 = clone_256 = None
    slice_scatter_864: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_863, 1, 0, 255);  slice_scatter_863 = None
    select_scatter_70: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_864, 1, 0);  slice_scatter_864 = None
    slice_scatter_865: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_70, 0, 0, 9223372036854775807);  select_scatter_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    new_empty_strided_92: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_115, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_352: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_92, as_strided_scatter_115);  new_empty_strided_92 = as_strided_scatter_115 = None
    as_strided_591: "f32[12, 3, 256, 256]" = torch.ops.aten.as_strided.default(copy_352, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328)
    clone_257: "f32[12, 3, 256, 256]" = torch.ops.aten.clone.default(as_strided_591, memory_format = torch.contiguous_format)
    copy_353: "f32[12, 3, 256, 256]" = torch.ops.aten.copy.default(as_strided_591, full_default_147);  as_strided_591 = full_default_147 = None
    as_strided_scatter_116: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_352, copy_353, [12, 3, 256, 256], [525312, 131328, 513, 1], 131328);  copy_352 = copy_353 = None
    slice_scatter_866: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_257, 3, 257, 9223372036854775807);  clone_257 = None
    slice_scatter_867: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_866, 2, -257, -1);  slice_scatter_866 = None
    slice_scatter_868: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_867, 1, 0, 9223372036854775807);  slice_scatter_867 = None
    slice_scatter_869: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_868, 0, 0, 9223372036854775807);  slice_scatter_868 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:868, code: diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[
    add_258: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(slice_scatter_865, slice_scatter_869);  slice_scatter_865 = slice_scatter_869 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_93: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_116, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_354: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_93, as_strided_scatter_116);  new_empty_strided_93 = as_strided_scatter_116 = None
    as_strided_594: "f32[12, 256, 257]" = torch.ops.aten.as_strided.default(copy_354, [12, 256, 257], [525312, 513, 1], 394240)
    clone_258: "f32[12, 256, 257]" = torch.ops.aten.clone.default(as_strided_594, memory_format = torch.contiguous_format)
    copy_355: "f32[12, 256, 257]" = torch.ops.aten.copy.default(as_strided_594, full_default_152);  as_strided_594 = full_default_152 = None
    as_strided_scatter_117: "f32[12, 4, 256, 513]" = torch.ops.aten.as_strided_scatter.default(copy_354, copy_355, [12, 256, 257], [525312, 513, 1], 394240);  copy_354 = copy_355 = None
    slice_scatter_870: "f32[12, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_153, clone_258, 2, 0, 257);  full_default_153 = clone_258 = None
    slice_scatter_871: "f32[12, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_144, slice_scatter_870, 1, 256, 9223372036854775807);  full_default_144 = slice_scatter_870 = None
    select_scatter_71: "f32[12, 3, 512, 513]" = torch.ops.aten.select_scatter.default(full_default_145, slice_scatter_871, 1, -1);  slice_scatter_871 = None
    slice_scatter_872: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, select_scatter_71, 0, 0, 9223372036854775807);  select_scatter_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:864, code: diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_259: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_258, slice_scatter_872);  add_258 = slice_scatter_872 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    new_empty_strided_94: "f32[12, 4, 256, 513]" = torch.ops.aten.new_empty_strided.default(as_strided_scatter_117, [12, 4, 256, 513], [525312, 131328, 513, 1])
    copy_356: "f32[12, 4, 256, 513]" = torch.ops.aten.copy.default(new_empty_strided_94, as_strided_scatter_117);  new_empty_strided_94 = as_strided_scatter_117 = None
    as_strided_597: "f32[12, 3, 256, 257]" = torch.ops.aten.as_strided.default(copy_356, [12, 3, 256, 257], [525312, 131328, 513, 1], 256);  copy_356 = None
    clone_259: "f32[12, 3, 256, 257]" = torch.ops.aten.clone.default(as_strided_597, memory_format = torch.contiguous_format);  as_strided_597 = None
    slice_scatter_873: "f32[12, 3, 256, 513]" = torch.ops.aten.slice_scatter.default(full_default_148, clone_259, 3, 0, 257);  full_default_148 = clone_259 = None
    slice_scatter_874: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_873, 2, 0, 256);  slice_scatter_873 = None
    slice_scatter_875: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_874, 1, 0, 9223372036854775807);  slice_scatter_874 = None
    slice_scatter_876: "f32[12, 3, 512, 513]" = torch.ops.aten.slice_scatter.default(full_default_145, slice_scatter_875, 0, 0, 9223372036854775807);  full_default_145 = slice_scatter_875 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:861, code: diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[
    add_260: "f32[12, 3, 512, 513]" = torch.ops.aten.add.Tensor(add_259, slice_scatter_876);  add_259 = slice_scatter_876 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:707, code: hidden_states_padded = hidden_states_padded.view(
    view_1550: "f32[12, 3, 513, 512]" = torch.ops.aten.view.default(add_260, [12, 3, 513, 512]);  add_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:704, code: hidden_states_padded = nn.functional.pad(
    constant_pad_nd_83: "f32[12, 3, 512, 512]" = torch.ops.aten.constant_pad_nd.default(view_1550, [0, 0, 0, -1]);  view_1550 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:843, code: diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (query, key))  # multiply
    view_1551: "f32[12, 3, 512, 512, 1]" = torch.ops.aten.view.default(constant_pad_nd_83, [12, 3, 512, 512, 1]);  constant_pad_nd_83 = None
    permute_1420: "f32[12, 3, 512, 1, 512]" = torch.ops.aten.permute.default(view_1551, [0, 1, 2, 4, 3]);  view_1551 = None
    view_1552: "f32[36, 512, 512]" = torch.ops.aten.view.default(permute_1420, [36, 512, 512]);  permute_1420 = None
    bmm_70: "f32[36, 64, 512]" = torch.ops.aten.bmm.default(permute_1421, view_1552);  permute_1421 = None
    bmm_71: "f32[36, 512, 64]" = torch.ops.aten.bmm.default(view_1552, permute_1422);  view_1552 = permute_1422 = None
    view_1553: "f32[12, 3, 64, 512, 1]" = torch.ops.aten.view.default(bmm_70, [12, 3, 64, 512, 1]);  bmm_70 = None
    permute_1423: "f32[12, 3, 1, 512, 64]" = torch.ops.aten.permute.default(view_1553, [0, 1, 4, 3, 2]);  view_1553 = None
    view_1554: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.view.default(bmm_71, [12, 3, 512, 64, 1]);  bmm_71 = None
    permute_1424: "f32[12, 3, 512, 1, 64]" = torch.ops.aten.permute.default(view_1554, [0, 1, 2, 4, 3]);  view_1554 = None
    permute_1425: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1423, [0, 1, 3, 4, 2]);  permute_1423 = None
    squeeze_92: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1425, 4);  permute_1425 = None
    permute_1426: "f32[12, 3, 512, 64, 1]" = torch.ops.aten.permute.default(permute_1424, [0, 1, 2, 4, 3]);  permute_1424 = None
    squeeze_93: "f32[12, 3, 512, 64]" = torch.ops.aten.squeeze.dim(permute_1426, 4);  permute_1426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    clone_260: "f32[12, 3, 512, 64]" = torch.ops.aten.clone.default(squeeze_92, memory_format = torch.contiguous_format);  squeeze_92 = None
    view_1555: "f32[1179648]" = torch.ops.aten.view.default(clone_260, [1179648]);  clone_260 = None
    index_put_34: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1555, True);  view_1555 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:778, code: return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    view_1558: "f32[1179648]" = torch.ops.aten.view.default(squeeze_93, [-1]);  squeeze_93 = None
    index_put_35: "f32[786432]" = torch.ops.aten.index_put.default(full_default_161, [view_929], view_1558, True);  view_929 = view_1558 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:566, code: query_vectors /= math.sqrt(self.head_dim)
    as_strided_602: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_35, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_35 = None
    view_1565: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_602, [12, 1024, 64]);  as_strided_602 = None
    view_1566: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1565, [1, 12, 1024, 64]);  view_1565 = None
    permute_1431: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1566, [0, 2, 1, 3]);  view_1566 = None
    permute_1432: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1431, [1, 0, 2, 3]);  permute_1431 = None
    view_1567: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1432, [1024, 1, 768]);  permute_1432 = None
    squeeze_95: "f32[1024, 768]" = torch.ops.aten.squeeze.dim(view_1567, 1);  view_1567 = None
    copy_357: "f32[1024, 768]" = torch.ops.aten.copy.default(as_strided_108, squeeze_95);  as_strided_108 = squeeze_95 = None
    as_strided_scatter_118: "f32[786432]" = torch.ops.aten.as_strided_scatter.default(full_default_161, copy_357, [1024, 768], [768, 1], 0);  full_default_161 = copy_357 = None
    as_strided_606: "f32[1024, 768]" = torch.ops.aten.as_strided.default(as_strided_scatter_118, [1024, 768], [768, 1], 0);  as_strided_scatter_118 = None
    new_empty_strided_95: "f32[1024, 768]" = torch.ops.aten.new_empty_strided.default(as_strided_606, [1024, 768], [768, 1])
    copy_358: "f32[1024, 768]" = torch.ops.aten.copy.default(new_empty_strided_95, as_strided_606);  new_empty_strided_95 = as_strided_606 = None
    as_strided_608: "f32[1024, 1, 768]" = torch.ops.aten.as_strided.default(copy_358, [1024, 1, 768], [768, 768, 1], 0)
    clone_263: "f32[1024, 1, 768]" = torch.ops.aten.clone.default(as_strided_608, memory_format = torch.contiguous_format)
    div_155: "f32[1024, 1, 768]" = torch.ops.aten.div.Tensor(clone_263, 8.0);  clone_263 = None
    copy_359: "f32[1024, 1, 768]" = torch.ops.aten.copy.default(as_strided_608, div_155);  as_strided_608 = div_155 = None
    as_strided_scatter_119: "f32[1024, 768]" = torch.ops.aten.as_strided_scatter.default(copy_358, copy_359, [1024, 1, 768], [768, 768, 1], 0);  copy_358 = copy_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:558, code: value_vectors = self.value(hidden_states)
    view_1568: "f32[1024, 768]" = torch.ops.aten.view.default(view_1548, [1024, 768]);  view_1548 = None
    mm_138: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1568, permute_1433);  permute_1433 = None
    permute_1434: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1568, [1, 0])
    mm_139: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1434, view);  permute_1434 = None
    permute_1435: "f32[768, 768]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_190: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1568, [0], True);  view_1568 = None
    view_1569: "f32[768]" = torch.ops.aten.view.default(sum_190, [768]);  sum_190 = None
    permute_1436: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1435, [1, 0]);  permute_1435 = None
    view_1570: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_138, [1024, 1, 768]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    as_strided_610: "f32[12, 2, 512, 64]" = torch.ops.aten.as_strided.default(index_put_34, [12, 2, 512, 64], [64, 393216, 768, 1], 0);  index_put_34 = None
    view_1572: "f32[12, 1024, 64]" = torch.ops.aten.view.default(as_strided_610, [12, 1024, 64]);  as_strided_610 = None
    view_1573: "f32[1, 12, 1024, 64]" = torch.ops.aten.view.default(view_1572, [1, 12, 1024, 64]);  view_1572 = None
    permute_1438: "f32[1, 1024, 12, 64]" = torch.ops.aten.permute.default(view_1573, [0, 2, 1, 3]);  view_1573 = None
    permute_1439: "f32[1024, 1, 12, 64]" = torch.ops.aten.permute.default(permute_1438, [1, 0, 2, 3]);  permute_1438 = None
    view_1574: "f32[1024, 1, 768]" = torch.ops.aten.view.default(permute_1439, [1024, 1, 768]);  permute_1439 = None
    view_1575: "f32[1024, 768]" = torch.ops.aten.view.default(view_1574, [1024, 768]);  view_1574 = None
    mm_140: "f32[1024, 768]" = torch.ops.aten.mm.default(view_1575, permute_1437);  permute_1437 = None
    permute_1443: "f32[768, 1024]" = torch.ops.aten.permute.default(view_1575, [1, 0])
    mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1443, view);  permute_1443 = None
    permute_1444: "f32[768, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_191: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1575, [0], True);  view_1575 = None
    view_1580: "f32[768]" = torch.ops.aten.view.default(sum_191, [768]);  sum_191 = None
    permute_1445: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1444, [1, 0]);  permute_1444 = None
    view_1581: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_140, [1024, 1, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:557, code: key_vectors = self.key(hidden_states)
    add_261: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(view_1570, view_1581);  view_1570 = view_1581 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    mm_142: "f32[1024, 768]" = torch.ops.aten.mm.default(as_strided_scatter_119, permute_1446);  permute_1446 = None
    permute_1448: "f32[768, 1024]" = torch.ops.aten.permute.default(as_strided_scatter_119, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_1448, view);  permute_1448 = view = None
    permute_1449: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_192: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(as_strided_scatter_119, [0], True);  as_strided_scatter_119 = None
    view_1582: "f32[768]" = torch.ops.aten.view.default(sum_192, [768]);  sum_192 = None
    permute_1450: "f32[768, 768]" = torch.ops.aten.permute.default(permute_1449, [1, 0]);  permute_1449 = None
    view_1583: "f32[1024, 1, 768]" = torch.ops.aten.view.default(mm_142, [1024, 1, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:556, code: query_vectors = self.query(hidden_states)
    add_262: "f32[1024, 1, 768]" = torch.ops.aten.add.Tensor(add_261, view_1583);  add_261 = view_1583 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    permute_1451: "f32[1, 1024, 768]" = torch.ops.aten.permute.default(add_262, [1, 0, 2]);  add_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/longformer/modeling_longformer.py:553, code: hidden_states = hidden_states.transpose(0, 1)
    add_263: "f32[1, 1024, 768]" = torch.ops.aten.add.Tensor(mul_436, permute_1451);  mul_436 = permute_1451 = None
    return [permute_1450, view_1582, permute_1445, view_1580, permute_1436, view_1569, permute_1405, view_1534, sum_186, sum_187, permute_1401, view_1531, permute_1397, view_1528, sum_180, sum_181, permute_1392, view_1525, permute_1387, view_1523, permute_1378, view_1512, permute_1347, view_1477, sum_171, sum_172, permute_1343, view_1474, permute_1339, view_1471, sum_165, sum_166, permute_1334, view_1468, permute_1329, view_1466, permute_1320, view_1455, permute_1289, view_1420, sum_156, sum_157, permute_1285, view_1417, permute_1281, view_1414, sum_150, sum_151, permute_1276, view_1411, permute_1271, view_1409, permute_1262, view_1398, permute_1231, view_1363, sum_141, sum_142, permute_1227, view_1360, permute_1223, view_1357, sum_135, sum_136, permute_1218, view_1354, permute_1213, view_1352, permute_1204, view_1341, permute_1173, view_1306, sum_126, sum_127, permute_1169, view_1303, permute_1165, view_1300, sum_120, sum_121, permute_1160, view_1297, permute_1155, view_1295, permute_1146, view_1284, permute_1115, view_1249, sum_111, sum_112, permute_1111, view_1246, permute_1107, view_1243, sum_105, sum_106, permute_1102, view_1240, permute_1097, view_1238, permute_1088, view_1227, permute_1057, view_1192, sum_96, sum_97, permute_1053, view_1189, permute_1049, view_1186, sum_90, sum_91, permute_1044, view_1183, permute_1039, view_1181, permute_1030, view_1170, permute_999, view_1135, sum_81, sum_82, permute_995, view_1132, permute_991, view_1129, sum_75, sum_76, permute_986, view_1126, permute_981, view_1124, permute_972, view_1113, permute_941, view_1078, sum_66, sum_67, permute_937, view_1075, permute_933, view_1072, sum_60, sum_61, permute_928, view_1069, permute_923, view_1067, permute_914, view_1056, permute_883, view_1021, sum_51, sum_52, permute_879, view_1018, permute_875, view_1015, sum_45, sum_46, permute_870, view_1012, permute_865, view_1010, permute_856, view_999, permute_825, view_964, sum_36, sum_37, permute_821, view_961, permute_817, view_958, sum_30, sum_31, permute_812, view_955, permute_807, view_953, permute_798, view_942, permute_767, view_907, sum_21, sum_22, permute_763, view_904, permute_759, view_901, sum_15, sum_16, add_263, None, None]
    