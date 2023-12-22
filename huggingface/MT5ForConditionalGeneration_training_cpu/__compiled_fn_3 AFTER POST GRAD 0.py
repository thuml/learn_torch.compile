from __future__ import annotations



def forward(self, primals_1: "f32[512]", primals_2: "f32[512]", primals_3: "f32[512]", primals_4: "f32[512]", primals_5: "f32[512]", primals_6: "f32[512]", primals_7: "f32[512]", primals_8: "f32[512]", primals_9: "f32[512]", primals_10: "f32[512]", primals_11: "f32[512]", primals_12: "f32[512]", primals_13: "f32[512]", primals_14: "f32[512]", primals_15: "f32[512]", primals_16: "f32[512]", primals_17: "f32[512]", primals_18: "f32[512]", primals_19: "f32[512]", primals_20: "f32[512]", primals_21: "f32[512]", primals_22: "f32[512]", primals_23: "f32[512]", primals_24: "f32[512]", primals_25: "f32[512]", primals_26: "f32[512]", primals_27: "f32[512]", primals_28: "f32[512]", primals_29: "f32[512]", primals_30: "f32[512]", primals_31: "f32[512]", primals_32: "f32[512]", primals_33: "f32[512]", primals_34: "f32[512]", primals_35: "f32[512]", primals_36: "f32[512]", primals_37: "f32[512]", primals_38: "f32[512]", primals_39: "f32[512]", primals_40: "f32[512]", primals_41: "f32[512]", primals_42: "f32[512]", primals_43: "f32[250112, 512]", primals_44: "f32[384, 512]", primals_45: "f32[384, 512]", primals_46: "f32[384, 512]", primals_47: "f32[32, 6]", primals_48: "f32[512, 384]", primals_49: "f32[1024, 512]", primals_50: "f32[1024, 512]", primals_51: "f32[512, 1024]", primals_52: "f32[384, 512]", primals_53: "f32[384, 512]", primals_54: "f32[384, 512]", primals_55: "f32[512, 384]", primals_56: "f32[1024, 512]", primals_57: "f32[1024, 512]", primals_58: "f32[512, 1024]", primals_59: "f32[384, 512]", primals_60: "f32[384, 512]", primals_61: "f32[384, 512]", primals_62: "f32[512, 384]", primals_63: "f32[1024, 512]", primals_64: "f32[1024, 512]", primals_65: "f32[512, 1024]", primals_66: "f32[384, 512]", primals_67: "f32[384, 512]", primals_68: "f32[384, 512]", primals_69: "f32[512, 384]", primals_70: "f32[1024, 512]", primals_71: "f32[1024, 512]", primals_72: "f32[512, 1024]", primals_73: "f32[384, 512]", primals_74: "f32[384, 512]", primals_75: "f32[384, 512]", primals_76: "f32[512, 384]", primals_77: "f32[1024, 512]", primals_78: "f32[1024, 512]", primals_79: "f32[512, 1024]", primals_80: "f32[384, 512]", primals_81: "f32[384, 512]", primals_82: "f32[384, 512]", primals_83: "f32[512, 384]", primals_84: "f32[1024, 512]", primals_85: "f32[1024, 512]", primals_86: "f32[512, 1024]", primals_87: "f32[384, 512]", primals_88: "f32[384, 512]", primals_89: "f32[384, 512]", primals_90: "f32[512, 384]", primals_91: "f32[1024, 512]", primals_92: "f32[1024, 512]", primals_93: "f32[512, 1024]", primals_94: "f32[384, 512]", primals_95: "f32[384, 512]", primals_96: "f32[384, 512]", primals_97: "f32[512, 384]", primals_98: "f32[1024, 512]", primals_99: "f32[1024, 512]", primals_100: "f32[512, 1024]", primals_101: "f32[384, 512]", primals_102: "f32[384, 512]", primals_103: "f32[384, 512]", primals_104: "f32[32, 6]", primals_105: "f32[512, 384]", primals_106: "f32[384, 512]", primals_107: "f32[384, 512]", primals_108: "f32[384, 512]", primals_109: "f32[512, 384]", primals_110: "f32[1024, 512]", primals_111: "f32[1024, 512]", primals_112: "f32[512, 1024]", primals_113: "f32[384, 512]", primals_114: "f32[384, 512]", primals_115: "f32[384, 512]", primals_116: "f32[512, 384]", primals_117: "f32[384, 512]", primals_118: "f32[384, 512]", primals_119: "f32[384, 512]", primals_120: "f32[512, 384]", primals_121: "f32[1024, 512]", primals_122: "f32[1024, 512]", primals_123: "f32[512, 1024]", primals_124: "f32[384, 512]", primals_125: "f32[384, 512]", primals_126: "f32[384, 512]", primals_127: "f32[512, 384]", primals_128: "f32[384, 512]", primals_129: "f32[384, 512]", primals_130: "f32[384, 512]", primals_131: "f32[512, 384]", primals_132: "f32[1024, 512]", primals_133: "f32[1024, 512]", primals_134: "f32[512, 1024]", primals_135: "f32[384, 512]", primals_136: "f32[384, 512]", primals_137: "f32[384, 512]", primals_138: "f32[512, 384]", primals_139: "f32[384, 512]", primals_140: "f32[384, 512]", primals_141: "f32[384, 512]", primals_142: "f32[512, 384]", primals_143: "f32[1024, 512]", primals_144: "f32[1024, 512]", primals_145: "f32[512, 1024]", primals_146: "f32[384, 512]", primals_147: "f32[384, 512]", primals_148: "f32[384, 512]", primals_149: "f32[512, 384]", primals_150: "f32[384, 512]", primals_151: "f32[384, 512]", primals_152: "f32[384, 512]", primals_153: "f32[512, 384]", primals_154: "f32[1024, 512]", primals_155: "f32[1024, 512]", primals_156: "f32[512, 1024]", primals_157: "f32[384, 512]", primals_158: "f32[384, 512]", primals_159: "f32[384, 512]", primals_160: "f32[512, 384]", primals_161: "f32[384, 512]", primals_162: "f32[384, 512]", primals_163: "f32[384, 512]", primals_164: "f32[512, 384]", primals_165: "f32[1024, 512]", primals_166: "f32[1024, 512]", primals_167: "f32[512, 1024]", primals_168: "f32[384, 512]", primals_169: "f32[384, 512]", primals_170: "f32[384, 512]", primals_171: "f32[512, 384]", primals_172: "f32[384, 512]", primals_173: "f32[384, 512]", primals_174: "f32[384, 512]", primals_175: "f32[512, 384]", primals_176: "f32[1024, 512]", primals_177: "f32[1024, 512]", primals_178: "f32[512, 1024]", primals_179: "f32[384, 512]", primals_180: "f32[384, 512]", primals_181: "f32[384, 512]", primals_182: "f32[512, 384]", primals_183: "f32[384, 512]", primals_184: "f32[384, 512]", primals_185: "f32[384, 512]", primals_186: "f32[512, 384]", primals_187: "f32[1024, 512]", primals_188: "f32[1024, 512]", primals_189: "f32[512, 1024]", primals_190: "f32[250112, 512]", primals_191: "i64[1, 128]", primals_192: "i64[1, 128]", primals_193: "i64[1, 128]"):
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:984, code: input_ids = input_ids.view(-1, input_shape[-1])
    view: "i64[1, 128]" = torch.ops.aten.reshape.default(primals_191, [-1, 128]);  primals_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:994, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(primals_43, view)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1049, code: hidden_states = self.dropout(inputs_embeds)
    native_dropout = torch.ops.aten.native_dropout.default(embedding, 0.1, True);  embedding = None
    getitem: "f32[1, 128, 512]" = native_dropout[0]
    getitem_1: "b8[1, 128, 512]" = native_dropout[1];  native_dropout = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_1: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(getitem, 2)
    mean: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
    rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
    mul_1: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(getitem, rsqrt)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_2: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_1, mul_1);  mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute: "f32[512, 384]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
    view_1: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_2, [128, 512]);  mul_2 = None
    mm: "f32[128, 384]" = torch.ops.aten.mm.default(view_1, permute)
    view_2: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm, [1, 128, 384]);  mm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_3: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_2, [1, -1, 6, 64]);  view_2 = None
    permute_1: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_2: "f32[512, 384]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
    mm_1: "f32[128, 384]" = torch.ops.aten.mm.default(view_1, permute_2)
    view_5: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_1, [1, 128, 384]);  mm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_6: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_5, [1, -1, 6, 64]);  view_5 = None
    permute_3: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_4: "f32[512, 384]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
    mm_2: "f32[128, 384]" = torch.ops.aten.mm.default(view_1, permute_4)
    view_8: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_2, [1, 128, 384]);  mm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_9: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_8, [1, -1, 6, 64]);  view_8 = None
    permute_5: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_6: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_1, [1, 6, 128, 64]);  permute_1 = None
    view_10: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand, [6, 128, 64]);  expand = None
    expand_1: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_6, [1, 6, 64, 128]);  permute_6 = None
    view_11: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_1, [6, 64, 128]);  expand_1 = None
    bmm: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_10, view_11)
    view_12: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm, [1, 6, 128, 128]);  bmm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:302, code: context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
    unsqueeze_2: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(iota, 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:303, code: memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    unsqueeze_3: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(iota, 0);  iota = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:304, code: relative_position = memory_position - context_position  # shape (query_length, key_length)
    sub_1: "i64[128, 128]" = torch.ops.aten.sub.Tensor(unsqueeze_3, unsqueeze_2);  unsqueeze_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:275, code: relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
    gt: "b8[128, 128]" = torch.ops.aten.gt.Scalar(sub_1, 0)
    convert_element_type: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(gt, torch.int64);  gt = None
    mul_3: "i64[128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type, 16);  convert_element_type = None
    add_1: "i64[128, 128]" = torch.ops.aten.add.Tensor(mul_3, 0);  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:276, code: relative_position = torch.abs(relative_position)
    abs_1: "i64[128, 128]" = torch.ops.aten.abs.default(sub_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:283, code: is_small = relative_position < max_exact
    lt: "b8[128, 128]" = torch.ops.aten.lt.Scalar(abs_1, 8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:287, code: torch.log(relative_position.float() / max_exact)
    convert_element_type_1: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(abs_1, torch.float32)
    div: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_1, 8);  convert_element_type_1 = None
    log: "f32[128, 128]" = torch.ops.aten.log.default(div);  div = None
    div_1: "f32[128, 128]" = torch.ops.aten.div.Tensor(log, 2.772588722239781);  log = None
    mul_4: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_1, 8);  div_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:290, code: ).to(torch.long)
    convert_element_type_2: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_4, torch.int64);  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:286, code: relative_position_if_large = max_exact + (
    add_2: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_2, 8);  convert_element_type_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:292, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    full_default_1: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 15, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:291, code: relative_position_if_large = torch.min(
    minimum: "i64[128, 128]" = torch.ops.aten.minimum.default(add_2, full_default_1);  add_2 = full_default_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:295, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    where: "i64[128, 128]" = torch.ops.aten.where.self(lt, abs_1, minimum);  lt = abs_1 = minimum = None
    add_3: "i64[128, 128]" = torch.ops.aten.add.Tensor(add_1, where);  add_1 = where = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:311, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    embedding_1: "f32[128, 128, 6]" = torch.ops.aten.embedding.default(primals_47, add_3);  primals_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:312, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    permute_7: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:413, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    unsqueeze_4: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_5: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_12, unsqueeze_4);  view_12 = None
    view_13: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_5, [6, 128, 128]);  add_5 = None
    view_14: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_13, [1, 6, 128, 128]);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_14, [-1], True)
    sub_2: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_14, amax);  view_14 = amax = None
    exp: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
    sum_1: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_2: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias_1: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_1 = torch.ops.aten.native_dropout.default(div_2, 0.1, True);  div_2 = None
    getitem_2: "f32[1, 6, 128, 128]" = native_dropout_1[0]
    getitem_3: "b8[1, 6, 128, 128]" = native_dropout_1[1];  native_dropout_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_2: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_2, [1, 6, 128, 128]);  getitem_2 = None
    view_15: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_2, [6, 128, 128]);  expand_2 = None
    expand_3: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_5, [1, 6, 128, 64]);  permute_5 = None
    view_16: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_3, [6, 128, 64]);  expand_3 = None
    bmm_1: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_15, view_16)
    view_17: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_1, [1, 6, 128, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_8: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
    clone: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_8, memory_format = torch.contiguous_format);  permute_8 = None
    view_18: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone, [1, -1, 384]);  clone = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_9: "f32[384, 512]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
    view_19: "f32[128, 384]" = torch.ops.aten.reshape.default(view_18, [128, 384]);  view_18 = None
    mm_3: "f32[128, 512]" = torch.ops.aten.mm.default(view_19, permute_9)
    view_20: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_3, [1, 128, 512]);  mm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_2 = torch.ops.aten.native_dropout.default(view_20, 0.1, True);  view_20 = None
    getitem_4: "f32[1, 128, 512]" = native_dropout_2[0]
    getitem_5: "b8[1, 128, 512]" = native_dropout_2[1];  native_dropout_2 = None
    add_6: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(getitem, getitem_4);  getitem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_2: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_6, 2)
    mean_1: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_7: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
    rsqrt_1: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
    mul_5: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_6, rsqrt_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_6: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_2, mul_5);  mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_10: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
    view_21: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_6, [128, 512]);  mul_6 = None
    mm_4: "f32[128, 1024]" = torch.ops.aten.mm.default(view_21, permute_10)
    view_22: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_4, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_7: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
    pow_3: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
    mul_8: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
    add_8: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_22, mul_8);  view_22 = mul_8 = None
    mul_9: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_8, 0.7978845608028654);  add_8 = None
    tanh: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_9);  mul_9 = None
    add_9: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh, 1.0)
    mul_10: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_11: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
    mm_5: "f32[128, 1024]" = torch.ops.aten.mm.default(view_21, permute_11)
    view_24: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_5, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_11: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_10, view_24);  mul_10 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_3 = torch.ops.aten.native_dropout.default(mul_11, 0.1, True);  mul_11 = None
    getitem_6: "f32[1, 128, 1024]" = native_dropout_3[0]
    getitem_7: "b8[1, 128, 1024]" = native_dropout_3[1];  native_dropout_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_12: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    view_25: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_6, [128, 1024]);  getitem_6 = None
    mm_6: "f32[128, 512]" = torch.ops.aten.mm.default(view_25, permute_12)
    view_26: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_6, [1, 128, 512]);  mm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_4 = torch.ops.aten.native_dropout.default(view_26, 0.1, True);  view_26 = None
    getitem_8: "f32[1, 128, 512]" = native_dropout_4[0]
    getitem_9: "b8[1, 128, 512]" = native_dropout_4[1];  native_dropout_4 = None
    add_10: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_6, getitem_8);  getitem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_4: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_10, 2)
    mean_2: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_11: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
    rsqrt_2: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
    mul_12: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_10, rsqrt_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_13: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_3, mul_12);  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_13: "f32[512, 384]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
    view_27: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_13, [128, 512]);  mul_13 = None
    mm_7: "f32[128, 384]" = torch.ops.aten.mm.default(view_27, permute_13)
    view_28: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_7, [1, 128, 384]);  mm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_29: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_28, [1, -1, 6, 64]);  view_28 = None
    permute_14: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_15: "f32[512, 384]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    mm_8: "f32[128, 384]" = torch.ops.aten.mm.default(view_27, permute_15)
    view_31: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_8, [1, 128, 384]);  mm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_32: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_31, [1, -1, 6, 64]);  view_31 = None
    permute_16: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1, 3]);  view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_17: "f32[512, 384]" = torch.ops.aten.permute.default(primals_54, [1, 0]);  primals_54 = None
    mm_9: "f32[128, 384]" = torch.ops.aten.mm.default(view_27, permute_17)
    view_34: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_9, [1, 128, 384]);  mm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_35: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_34, [1, -1, 6, 64]);  view_34 = None
    permute_18: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_35, [0, 2, 1, 3]);  view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_19: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_16, [0, 1, 3, 2]);  permute_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_4: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_14, [1, 6, 128, 64]);  permute_14 = None
    view_36: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_4, [6, 128, 64]);  expand_4 = None
    expand_5: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_19, [1, 6, 64, 128]);  permute_19 = None
    view_37: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_5, [6, 64, 128]);  expand_5 = None
    bmm_2: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_36, view_37)
    view_38: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [1, 6, 128, 128]);  bmm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_12: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_38, unsqueeze_4);  view_38 = None
    view_39: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_12, [6, 128, 128]);  add_12 = None
    view_40: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_39, [1, 6, 128, 128]);  view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_1: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_40, [-1], True)
    sub_3: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_40, amax_1);  view_40 = amax_1 = None
    exp_1: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
    sum_2: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_3: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_5: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_5 = torch.ops.aten.native_dropout.default(div_3, 0.1, True);  div_3 = None
    getitem_10: "f32[1, 6, 128, 128]" = native_dropout_5[0]
    getitem_11: "b8[1, 6, 128, 128]" = native_dropout_5[1];  native_dropout_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_6: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_10, [1, 6, 128, 128]);  getitem_10 = None
    view_41: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_6, [6, 128, 128]);  expand_6 = None
    expand_7: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_18, [1, 6, 128, 64]);  permute_18 = None
    view_42: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_7, [6, 128, 64]);  expand_7 = None
    bmm_3: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_41, view_42)
    view_43: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_3, [1, 6, 128, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_20: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_43, [0, 2, 1, 3]);  view_43 = None
    clone_1: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_20, memory_format = torch.contiguous_format);  permute_20 = None
    view_44: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_1, [1, -1, 384]);  clone_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_21: "f32[384, 512]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    view_45: "f32[128, 384]" = torch.ops.aten.reshape.default(view_44, [128, 384]);  view_44 = None
    mm_10: "f32[128, 512]" = torch.ops.aten.mm.default(view_45, permute_21)
    view_46: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_10, [1, 128, 512]);  mm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_6 = torch.ops.aten.native_dropout.default(view_46, 0.1, True);  view_46 = None
    getitem_12: "f32[1, 128, 512]" = native_dropout_6[0]
    getitem_13: "b8[1, 128, 512]" = native_dropout_6[1];  native_dropout_6 = None
    add_13: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_10, getitem_12);  getitem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_5: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_13, 2)
    mean_3: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_5, [-1], True);  pow_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_14: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
    rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_14);  add_14 = None
    mul_14: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_13, rsqrt_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_15: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_4, mul_14);  mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_22: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
    view_47: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_15, [128, 512]);  mul_15 = None
    mm_11: "f32[128, 1024]" = torch.ops.aten.mm.default(view_47, permute_22)
    view_48: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_11, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_16: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_48, 0.5)
    pow_6: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_48, 3.0)
    mul_17: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
    add_15: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_48, mul_17);  view_48 = mul_17 = None
    mul_18: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_15, 0.7978845608028654);  add_15 = None
    tanh_1: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_18);  mul_18 = None
    add_16: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
    mul_19: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_16, add_16);  mul_16 = add_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_23: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    mm_12: "f32[128, 1024]" = torch.ops.aten.mm.default(view_47, permute_23)
    view_50: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_12, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_20: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_19, view_50);  mul_19 = view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_7 = torch.ops.aten.native_dropout.default(mul_20, 0.1, True);  mul_20 = None
    getitem_14: "f32[1, 128, 1024]" = native_dropout_7[0]
    getitem_15: "b8[1, 128, 1024]" = native_dropout_7[1];  native_dropout_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_24: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_58, [1, 0]);  primals_58 = None
    view_51: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_14, [128, 1024]);  getitem_14 = None
    mm_13: "f32[128, 512]" = torch.ops.aten.mm.default(view_51, permute_24)
    view_52: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_13, [1, 128, 512]);  mm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_8 = torch.ops.aten.native_dropout.default(view_52, 0.1, True);  view_52 = None
    getitem_16: "f32[1, 128, 512]" = native_dropout_8[0]
    getitem_17: "b8[1, 128, 512]" = native_dropout_8[1];  native_dropout_8 = None
    add_17: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_13, getitem_16);  getitem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_7: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_17, 2)
    mean_4: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_7, [-1], True);  pow_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_18: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_4, 1e-06);  mean_4 = None
    rsqrt_4: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_18);  add_18 = None
    mul_21: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_17, rsqrt_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_22: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_5, mul_21);  mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_25: "f32[512, 384]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    view_53: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_22, [128, 512]);  mul_22 = None
    mm_14: "f32[128, 384]" = torch.ops.aten.mm.default(view_53, permute_25)
    view_54: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_14, [1, 128, 384]);  mm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_55: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_54, [1, -1, 6, 64]);  view_54 = None
    permute_26: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_55, [0, 2, 1, 3]);  view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_27: "f32[512, 384]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
    mm_15: "f32[128, 384]" = torch.ops.aten.mm.default(view_53, permute_27)
    view_57: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_15, [1, 128, 384]);  mm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_58: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_57, [1, -1, 6, 64]);  view_57 = None
    permute_28: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_29: "f32[512, 384]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    mm_16: "f32[128, 384]" = torch.ops.aten.mm.default(view_53, permute_29)
    view_60: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_16, [1, 128, 384]);  mm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_61: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_60, [1, -1, 6, 64]);  view_60 = None
    permute_30: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_61, [0, 2, 1, 3]);  view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_31: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_28, [0, 1, 3, 2]);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_8: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_26, [1, 6, 128, 64]);  permute_26 = None
    view_62: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_8, [6, 128, 64]);  expand_8 = None
    expand_9: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_31, [1, 6, 64, 128]);  permute_31 = None
    view_63: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_9, [6, 64, 128]);  expand_9 = None
    bmm_4: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_62, view_63)
    view_64: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [1, 6, 128, 128]);  bmm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_19: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_64, unsqueeze_4);  view_64 = None
    view_65: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_19, [6, 128, 128]);  add_19 = None
    view_66: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_65, [1, 6, 128, 128]);  view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_2: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_66, [-1], True)
    sub_4: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_66, amax_2);  view_66 = amax_2 = None
    exp_2: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_3: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_4: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_9: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_9 = torch.ops.aten.native_dropout.default(div_4, 0.1, True);  div_4 = None
    getitem_18: "f32[1, 6, 128, 128]" = native_dropout_9[0]
    getitem_19: "b8[1, 6, 128, 128]" = native_dropout_9[1];  native_dropout_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_10: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_18, [1, 6, 128, 128]);  getitem_18 = None
    view_67: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_10, [6, 128, 128]);  expand_10 = None
    expand_11: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_30, [1, 6, 128, 64]);  permute_30 = None
    view_68: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_11, [6, 128, 64]);  expand_11 = None
    bmm_5: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_67, view_68)
    view_69: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_5, [1, 6, 128, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_32: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_69, [0, 2, 1, 3]);  view_69 = None
    clone_2: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_32, memory_format = torch.contiguous_format);  permute_32 = None
    view_70: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_2, [1, -1, 384]);  clone_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_33: "f32[384, 512]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
    view_71: "f32[128, 384]" = torch.ops.aten.reshape.default(view_70, [128, 384]);  view_70 = None
    mm_17: "f32[128, 512]" = torch.ops.aten.mm.default(view_71, permute_33)
    view_72: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_17, [1, 128, 512]);  mm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_10 = torch.ops.aten.native_dropout.default(view_72, 0.1, True);  view_72 = None
    getitem_20: "f32[1, 128, 512]" = native_dropout_10[0]
    getitem_21: "b8[1, 128, 512]" = native_dropout_10[1];  native_dropout_10 = None
    add_20: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_17, getitem_20);  getitem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_8: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_20, 2)
    mean_5: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_8, [-1], True);  pow_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_21: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_5, 1e-06);  mean_5 = None
    rsqrt_5: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_21);  add_21 = None
    mul_23: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_20, rsqrt_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_24: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_6, mul_23);  mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_34: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    view_73: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_24, [128, 512]);  mul_24 = None
    mm_18: "f32[128, 1024]" = torch.ops.aten.mm.default(view_73, permute_34)
    view_74: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_18, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_25: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_74, 0.5)
    pow_9: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_74, 3.0)
    mul_26: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
    add_22: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_74, mul_26);  view_74 = mul_26 = None
    mul_27: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_22, 0.7978845608028654);  add_22 = None
    tanh_2: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_27);  mul_27 = None
    add_23: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
    mul_28: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_25, add_23);  mul_25 = add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_35: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
    mm_19: "f32[128, 1024]" = torch.ops.aten.mm.default(view_73, permute_35)
    view_76: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_19, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_29: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_28, view_76);  mul_28 = view_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_11 = torch.ops.aten.native_dropout.default(mul_29, 0.1, True);  mul_29 = None
    getitem_22: "f32[1, 128, 1024]" = native_dropout_11[0]
    getitem_23: "b8[1, 128, 1024]" = native_dropout_11[1];  native_dropout_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_36: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    view_77: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_22, [128, 1024]);  getitem_22 = None
    mm_20: "f32[128, 512]" = torch.ops.aten.mm.default(view_77, permute_36)
    view_78: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_20, [1, 128, 512]);  mm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_12 = torch.ops.aten.native_dropout.default(view_78, 0.1, True);  view_78 = None
    getitem_24: "f32[1, 128, 512]" = native_dropout_12[0]
    getitem_25: "b8[1, 128, 512]" = native_dropout_12[1];  native_dropout_12 = None
    add_24: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_20, getitem_24);  getitem_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_10: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_24, 2)
    mean_6: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_10, [-1], True);  pow_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_25: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_6, 1e-06);  mean_6 = None
    rsqrt_6: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_25);  add_25 = None
    mul_30: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_24, rsqrt_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_31: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_7, mul_30);  mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_37: "f32[512, 384]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
    view_79: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_31, [128, 512]);  mul_31 = None
    mm_21: "f32[128, 384]" = torch.ops.aten.mm.default(view_79, permute_37)
    view_80: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_21, [1, 128, 384]);  mm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_81: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_80, [1, -1, 6, 64]);  view_80 = None
    permute_38: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_39: "f32[512, 384]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    mm_22: "f32[128, 384]" = torch.ops.aten.mm.default(view_79, permute_39)
    view_83: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_22, [1, 128, 384]);  mm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_84: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_83, [1, -1, 6, 64]);  view_83 = None
    permute_40: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_84, [0, 2, 1, 3]);  view_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_41: "f32[512, 384]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
    mm_23: "f32[128, 384]" = torch.ops.aten.mm.default(view_79, permute_41)
    view_86: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_23, [1, 128, 384]);  mm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_87: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_86, [1, -1, 6, 64]);  view_86 = None
    permute_42: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_87, [0, 2, 1, 3]);  view_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_43: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_40, [0, 1, 3, 2]);  permute_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_12: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_38, [1, 6, 128, 64]);  permute_38 = None
    view_88: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_12, [6, 128, 64]);  expand_12 = None
    expand_13: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_43, [1, 6, 64, 128]);  permute_43 = None
    view_89: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_13, [6, 64, 128]);  expand_13 = None
    bmm_6: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_88, view_89)
    view_90: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [1, 6, 128, 128]);  bmm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_26: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_90, unsqueeze_4);  view_90 = None
    view_91: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_26, [6, 128, 128]);  add_26 = None
    view_92: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_91, [1, 6, 128, 128]);  view_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_3: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_92, [-1], True)
    sub_5: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_92, amax_3);  view_92 = amax_3 = None
    exp_3: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
    sum_4: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_5: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_13: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_13 = torch.ops.aten.native_dropout.default(div_5, 0.1, True);  div_5 = None
    getitem_26: "f32[1, 6, 128, 128]" = native_dropout_13[0]
    getitem_27: "b8[1, 6, 128, 128]" = native_dropout_13[1];  native_dropout_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_14: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_26, [1, 6, 128, 128]);  getitem_26 = None
    view_93: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_14, [6, 128, 128]);  expand_14 = None
    expand_15: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_42, [1, 6, 128, 64]);  permute_42 = None
    view_94: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_15, [6, 128, 64]);  expand_15 = None
    bmm_7: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_93, view_94)
    view_95: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_7, [1, 6, 128, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_44: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_95, [0, 2, 1, 3]);  view_95 = None
    clone_3: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_44, memory_format = torch.contiguous_format);  permute_44 = None
    view_96: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_3, [1, -1, 384]);  clone_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_45: "f32[384, 512]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    view_97: "f32[128, 384]" = torch.ops.aten.reshape.default(view_96, [128, 384]);  view_96 = None
    mm_24: "f32[128, 512]" = torch.ops.aten.mm.default(view_97, permute_45)
    view_98: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_24, [1, 128, 512]);  mm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_14 = torch.ops.aten.native_dropout.default(view_98, 0.1, True);  view_98 = None
    getitem_28: "f32[1, 128, 512]" = native_dropout_14[0]
    getitem_29: "b8[1, 128, 512]" = native_dropout_14[1];  native_dropout_14 = None
    add_27: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_24, getitem_28);  getitem_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_11: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_27, 2)
    mean_7: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_11, [-1], True);  pow_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_28: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_7, 1e-06);  mean_7 = None
    rsqrt_7: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_28);  add_28 = None
    mul_32: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_27, rsqrt_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_33: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_8, mul_32);  mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_46: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
    view_99: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_33, [128, 512]);  mul_33 = None
    mm_25: "f32[128, 1024]" = torch.ops.aten.mm.default(view_99, permute_46)
    view_100: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_25, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_34: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_100, 0.5)
    pow_12: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_100, 3.0)
    mul_35: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
    add_29: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_100, mul_35);  view_100 = mul_35 = None
    mul_36: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_29, 0.7978845608028654);  add_29 = None
    tanh_3: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_36);  mul_36 = None
    add_30: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
    mul_37: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_34, add_30);  mul_34 = add_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_47: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    mm_26: "f32[128, 1024]" = torch.ops.aten.mm.default(view_99, permute_47)
    view_102: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_26, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_38: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_37, view_102);  mul_37 = view_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_15 = torch.ops.aten.native_dropout.default(mul_38, 0.1, True);  mul_38 = None
    getitem_30: "f32[1, 128, 1024]" = native_dropout_15[0]
    getitem_31: "b8[1, 128, 1024]" = native_dropout_15[1];  native_dropout_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_48: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_72, [1, 0]);  primals_72 = None
    view_103: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_30, [128, 1024]);  getitem_30 = None
    mm_27: "f32[128, 512]" = torch.ops.aten.mm.default(view_103, permute_48)
    view_104: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_27, [1, 128, 512]);  mm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_16 = torch.ops.aten.native_dropout.default(view_104, 0.1, True);  view_104 = None
    getitem_32: "f32[1, 128, 512]" = native_dropout_16[0]
    getitem_33: "b8[1, 128, 512]" = native_dropout_16[1];  native_dropout_16 = None
    add_31: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_27, getitem_32);  getitem_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_13: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_31, 2)
    mean_8: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_13, [-1], True);  pow_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_32: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_8, 1e-06);  mean_8 = None
    rsqrt_8: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_32);  add_32 = None
    mul_39: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_31, rsqrt_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_40: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_9, mul_39);  mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_49: "f32[512, 384]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    view_105: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_40, [128, 512]);  mul_40 = None
    mm_28: "f32[128, 384]" = torch.ops.aten.mm.default(view_105, permute_49)
    view_106: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_28, [1, 128, 384]);  mm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_107: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_106, [1, -1, 6, 64]);  view_106 = None
    permute_50: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_107, [0, 2, 1, 3]);  view_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_51: "f32[512, 384]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
    mm_29: "f32[128, 384]" = torch.ops.aten.mm.default(view_105, permute_51)
    view_109: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_29, [1, 128, 384]);  mm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_110: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_109, [1, -1, 6, 64]);  view_109 = None
    permute_52: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_110, [0, 2, 1, 3]);  view_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_53: "f32[512, 384]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    mm_30: "f32[128, 384]" = torch.ops.aten.mm.default(view_105, permute_53)
    view_112: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_30, [1, 128, 384]);  mm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_113: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_112, [1, -1, 6, 64]);  view_112 = None
    permute_54: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_113, [0, 2, 1, 3]);  view_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_55: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_52, [0, 1, 3, 2]);  permute_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_16: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_50, [1, 6, 128, 64]);  permute_50 = None
    view_114: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_16, [6, 128, 64]);  expand_16 = None
    expand_17: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_55, [1, 6, 64, 128]);  permute_55 = None
    view_115: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_17, [6, 64, 128]);  expand_17 = None
    bmm_8: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_114, view_115)
    view_116: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [1, 6, 128, 128]);  bmm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_33: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_116, unsqueeze_4);  view_116 = None
    view_117: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_33, [6, 128, 128]);  add_33 = None
    view_118: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_117, [1, 6, 128, 128]);  view_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_4: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_118, [-1], True)
    sub_6: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_118, amax_4);  view_118 = amax_4 = None
    exp_4: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
    sum_5: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_6: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_17: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_17 = torch.ops.aten.native_dropout.default(div_6, 0.1, True);  div_6 = None
    getitem_34: "f32[1, 6, 128, 128]" = native_dropout_17[0]
    getitem_35: "b8[1, 6, 128, 128]" = native_dropout_17[1];  native_dropout_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_18: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_34, [1, 6, 128, 128]);  getitem_34 = None
    view_119: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_18, [6, 128, 128]);  expand_18 = None
    expand_19: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_54, [1, 6, 128, 64]);  permute_54 = None
    view_120: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_19, [6, 128, 64]);  expand_19 = None
    bmm_9: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_119, view_120)
    view_121: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_9, [1, 6, 128, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_56: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_121, [0, 2, 1, 3]);  view_121 = None
    clone_4: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None
    view_122: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_4, [1, -1, 384]);  clone_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_57: "f32[384, 512]" = torch.ops.aten.permute.default(primals_76, [1, 0]);  primals_76 = None
    view_123: "f32[128, 384]" = torch.ops.aten.reshape.default(view_122, [128, 384]);  view_122 = None
    mm_31: "f32[128, 512]" = torch.ops.aten.mm.default(view_123, permute_57)
    view_124: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_31, [1, 128, 512]);  mm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_18 = torch.ops.aten.native_dropout.default(view_124, 0.1, True);  view_124 = None
    getitem_36: "f32[1, 128, 512]" = native_dropout_18[0]
    getitem_37: "b8[1, 128, 512]" = native_dropout_18[1];  native_dropout_18 = None
    add_34: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_31, getitem_36);  getitem_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_14: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 2)
    mean_9: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_14, [-1], True);  pow_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_35: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_9, 1e-06);  mean_9 = None
    rsqrt_9: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_35);  add_35 = None
    mul_41: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_42: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_10, mul_41);  mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_58: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    view_125: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_42, [128, 512]);  mul_42 = None
    mm_32: "f32[128, 1024]" = torch.ops.aten.mm.default(view_125, permute_58)
    view_126: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_32, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_43: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_126, 0.5)
    pow_15: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_126, 3.0)
    mul_44: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
    add_36: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_126, mul_44);  view_126 = mul_44 = None
    mul_45: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_36, 0.7978845608028654);  add_36 = None
    tanh_4: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_45);  mul_45 = None
    add_37: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
    mul_46: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_43, add_37);  mul_43 = add_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_59: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_78, [1, 0]);  primals_78 = None
    mm_33: "f32[128, 1024]" = torch.ops.aten.mm.default(view_125, permute_59)
    view_128: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_33, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_47: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_46, view_128);  mul_46 = view_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_19 = torch.ops.aten.native_dropout.default(mul_47, 0.1, True);  mul_47 = None
    getitem_38: "f32[1, 128, 1024]" = native_dropout_19[0]
    getitem_39: "b8[1, 128, 1024]" = native_dropout_19[1];  native_dropout_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_60: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    view_129: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_38, [128, 1024]);  getitem_38 = None
    mm_34: "f32[128, 512]" = torch.ops.aten.mm.default(view_129, permute_60)
    view_130: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_34, [1, 128, 512]);  mm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_20 = torch.ops.aten.native_dropout.default(view_130, 0.1, True);  view_130 = None
    getitem_40: "f32[1, 128, 512]" = native_dropout_20[0]
    getitem_41: "b8[1, 128, 512]" = native_dropout_20[1];  native_dropout_20 = None
    add_38: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_34, getitem_40);  getitem_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_16: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_38, 2)
    mean_10: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_16, [-1], True);  pow_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_39: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_10, 1e-06);  mean_10 = None
    rsqrt_10: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_39);  add_39 = None
    mul_48: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_38, rsqrt_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_49: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_11, mul_48);  mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_61: "f32[512, 384]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
    view_131: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_49, [128, 512]);  mul_49 = None
    mm_35: "f32[128, 384]" = torch.ops.aten.mm.default(view_131, permute_61)
    view_132: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_35, [1, 128, 384]);  mm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_133: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_132, [1, -1, 6, 64]);  view_132 = None
    permute_62: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_133, [0, 2, 1, 3]);  view_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_63: "f32[512, 384]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    mm_36: "f32[128, 384]" = torch.ops.aten.mm.default(view_131, permute_63)
    view_135: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_36, [1, 128, 384]);  mm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_136: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_135, [1, -1, 6, 64]);  view_135 = None
    permute_64: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_65: "f32[512, 384]" = torch.ops.aten.permute.default(primals_82, [1, 0]);  primals_82 = None
    mm_37: "f32[128, 384]" = torch.ops.aten.mm.default(view_131, permute_65)
    view_138: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_37, [1, 128, 384]);  mm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_139: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_138, [1, -1, 6, 64]);  view_138 = None
    permute_66: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_139, [0, 2, 1, 3]);  view_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_67: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_64, [0, 1, 3, 2]);  permute_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_20: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_62, [1, 6, 128, 64]);  permute_62 = None
    view_140: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_20, [6, 128, 64]);  expand_20 = None
    expand_21: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_67, [1, 6, 64, 128]);  permute_67 = None
    view_141: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_21, [6, 64, 128]);  expand_21 = None
    bmm_10: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_140, view_141)
    view_142: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [1, 6, 128, 128]);  bmm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_40: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_142, unsqueeze_4);  view_142 = None
    view_143: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_40, [6, 128, 128]);  add_40 = None
    view_144: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_143, [1, 6, 128, 128]);  view_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_5: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_144, [-1], True)
    sub_7: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_144, amax_5);  view_144 = amax_5 = None
    exp_5: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_6: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_7: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_21: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_21 = torch.ops.aten.native_dropout.default(div_7, 0.1, True);  div_7 = None
    getitem_42: "f32[1, 6, 128, 128]" = native_dropout_21[0]
    getitem_43: "b8[1, 6, 128, 128]" = native_dropout_21[1];  native_dropout_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_22: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_42, [1, 6, 128, 128]);  getitem_42 = None
    view_145: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_22, [6, 128, 128]);  expand_22 = None
    expand_23: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_66, [1, 6, 128, 64]);  permute_66 = None
    view_146: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_23, [6, 128, 64]);  expand_23 = None
    bmm_11: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_145, view_146)
    view_147: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_11, [1, 6, 128, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_68: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_147, [0, 2, 1, 3]);  view_147 = None
    clone_5: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_68, memory_format = torch.contiguous_format);  permute_68 = None
    view_148: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_5, [1, -1, 384]);  clone_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_69: "f32[384, 512]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    view_149: "f32[128, 384]" = torch.ops.aten.reshape.default(view_148, [128, 384]);  view_148 = None
    mm_38: "f32[128, 512]" = torch.ops.aten.mm.default(view_149, permute_69)
    view_150: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_38, [1, 128, 512]);  mm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_22 = torch.ops.aten.native_dropout.default(view_150, 0.1, True);  view_150 = None
    getitem_44: "f32[1, 128, 512]" = native_dropout_22[0]
    getitem_45: "b8[1, 128, 512]" = native_dropout_22[1];  native_dropout_22 = None
    add_41: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_38, getitem_44);  getitem_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_17: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_41, 2)
    mean_11: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_17, [-1], True);  pow_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_42: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_11, 1e-06);  mean_11 = None
    rsqrt_11: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
    mul_50: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_41, rsqrt_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_51: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_12, mul_50);  mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_70: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_84, [1, 0]);  primals_84 = None
    view_151: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_51, [128, 512]);  mul_51 = None
    mm_39: "f32[128, 1024]" = torch.ops.aten.mm.default(view_151, permute_70)
    view_152: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_39, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_52: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_152, 0.5)
    pow_18: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_152, 3.0)
    mul_53: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
    add_43: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_152, mul_53);  view_152 = mul_53 = None
    mul_54: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_43, 0.7978845608028654);  add_43 = None
    tanh_5: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_54);  mul_54 = None
    add_44: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
    mul_55: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_52, add_44);  mul_52 = add_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_71: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    mm_40: "f32[128, 1024]" = torch.ops.aten.mm.default(view_151, permute_71)
    view_154: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_40, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_56: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_55, view_154);  mul_55 = view_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_23 = torch.ops.aten.native_dropout.default(mul_56, 0.1, True);  mul_56 = None
    getitem_46: "f32[1, 128, 1024]" = native_dropout_23[0]
    getitem_47: "b8[1, 128, 1024]" = native_dropout_23[1];  native_dropout_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_72: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
    view_155: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_46, [128, 1024]);  getitem_46 = None
    mm_41: "f32[128, 512]" = torch.ops.aten.mm.default(view_155, permute_72)
    view_156: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_41, [1, 128, 512]);  mm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_24 = torch.ops.aten.native_dropout.default(view_156, 0.1, True);  view_156 = None
    getitem_48: "f32[1, 128, 512]" = native_dropout_24[0]
    getitem_49: "b8[1, 128, 512]" = native_dropout_24[1];  native_dropout_24 = None
    add_45: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_41, getitem_48);  getitem_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_19: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_45, 2)
    mean_12: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_19, [-1], True);  pow_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_46: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_12, 1e-06);  mean_12 = None
    rsqrt_12: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_46);  add_46 = None
    mul_57: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_45, rsqrt_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_58: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_13, mul_57);  mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_73: "f32[512, 384]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    view_157: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_58, [128, 512]);  mul_58 = None
    mm_42: "f32[128, 384]" = torch.ops.aten.mm.default(view_157, permute_73)
    view_158: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_42, [1, 128, 384]);  mm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_159: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_158, [1, -1, 6, 64]);  view_158 = None
    permute_74: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_75: "f32[512, 384]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
    mm_43: "f32[128, 384]" = torch.ops.aten.mm.default(view_157, permute_75)
    view_161: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_43, [1, 128, 384]);  mm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_162: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_161, [1, -1, 6, 64]);  view_161 = None
    permute_76: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_77: "f32[512, 384]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    mm_44: "f32[128, 384]" = torch.ops.aten.mm.default(view_157, permute_77)
    view_164: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_44, [1, 128, 384]);  mm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_165: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_164, [1, -1, 6, 64]);  view_164 = None
    permute_78: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_79: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_76, [0, 1, 3, 2]);  permute_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_24: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_74, [1, 6, 128, 64]);  permute_74 = None
    view_166: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_24, [6, 128, 64]);  expand_24 = None
    expand_25: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_79, [1, 6, 64, 128]);  permute_79 = None
    view_167: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_25, [6, 64, 128]);  expand_25 = None
    bmm_12: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_166, view_167)
    view_168: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [1, 6, 128, 128]);  bmm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_47: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_168, unsqueeze_4);  view_168 = None
    view_169: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_47, [6, 128, 128]);  add_47 = None
    view_170: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_169, [1, 6, 128, 128]);  view_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_6: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_170, [-1], True)
    sub_8: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_170, amax_6);  view_170 = amax_6 = None
    exp_6: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
    sum_7: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_8: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_25: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_25 = torch.ops.aten.native_dropout.default(div_8, 0.1, True);  div_8 = None
    getitem_50: "f32[1, 6, 128, 128]" = native_dropout_25[0]
    getitem_51: "b8[1, 6, 128, 128]" = native_dropout_25[1];  native_dropout_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_26: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_50, [1, 6, 128, 128]);  getitem_50 = None
    view_171: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_26, [6, 128, 128]);  expand_26 = None
    expand_27: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_78, [1, 6, 128, 64]);  permute_78 = None
    view_172: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_27, [6, 128, 64]);  expand_27 = None
    bmm_13: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_171, view_172)
    view_173: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_13, [1, 6, 128, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_80: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_173, [0, 2, 1, 3]);  view_173 = None
    clone_6: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_80, memory_format = torch.contiguous_format);  permute_80 = None
    view_174: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_6, [1, -1, 384]);  clone_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_81: "f32[384, 512]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
    view_175: "f32[128, 384]" = torch.ops.aten.reshape.default(view_174, [128, 384]);  view_174 = None
    mm_45: "f32[128, 512]" = torch.ops.aten.mm.default(view_175, permute_81)
    view_176: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_45, [1, 128, 512]);  mm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_26 = torch.ops.aten.native_dropout.default(view_176, 0.1, True);  view_176 = None
    getitem_52: "f32[1, 128, 512]" = native_dropout_26[0]
    getitem_53: "b8[1, 128, 512]" = native_dropout_26[1];  native_dropout_26 = None
    add_48: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_45, getitem_52);  getitem_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_20: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_48, 2)
    mean_13: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_20, [-1], True);  pow_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_49: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_13, 1e-06);  mean_13 = None
    rsqrt_13: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
    mul_59: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_48, rsqrt_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_60: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_14, mul_59);  mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_82: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    view_177: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_60, [128, 512]);  mul_60 = None
    mm_46: "f32[128, 1024]" = torch.ops.aten.mm.default(view_177, permute_82)
    view_178: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_46, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_61: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
    pow_21: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_178, 3.0)
    mul_62: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
    add_50: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_178, mul_62);  view_178 = mul_62 = None
    mul_63: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_50, 0.7978845608028654);  add_50 = None
    tanh_6: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_63);  mul_63 = None
    add_51: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
    mul_64: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_61, add_51);  mul_61 = add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_83: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
    mm_47: "f32[128, 1024]" = torch.ops.aten.mm.default(view_177, permute_83)
    view_180: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_47, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_65: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_64, view_180);  mul_64 = view_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_27 = torch.ops.aten.native_dropout.default(mul_65, 0.1, True);  mul_65 = None
    getitem_54: "f32[1, 128, 1024]" = native_dropout_27[0]
    getitem_55: "b8[1, 128, 1024]" = native_dropout_27[1];  native_dropout_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_84: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    view_181: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_54, [128, 1024]);  getitem_54 = None
    mm_48: "f32[128, 512]" = torch.ops.aten.mm.default(view_181, permute_84)
    view_182: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_48, [1, 128, 512]);  mm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_28 = torch.ops.aten.native_dropout.default(view_182, 0.1, True);  view_182 = None
    getitem_56: "f32[1, 128, 512]" = native_dropout_28[0]
    getitem_57: "b8[1, 128, 512]" = native_dropout_28[1];  native_dropout_28 = None
    add_52: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_48, getitem_56);  getitem_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_22: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_52, 2)
    mean_14: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_22, [-1], True);  pow_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_53: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_14, 1e-06);  mean_14 = None
    rsqrt_14: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
    mul_66: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_52, rsqrt_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_67: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_15, mul_66);  mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_85: "f32[512, 384]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
    view_183: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_67, [128, 512]);  mul_67 = None
    mm_49: "f32[128, 384]" = torch.ops.aten.mm.default(view_183, permute_85)
    view_184: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_49, [1, 128, 384]);  mm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_185: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_184, [1, -1, 6, 64]);  view_184 = None
    permute_86: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_185, [0, 2, 1, 3]);  view_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_87: "f32[512, 384]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    mm_50: "f32[128, 384]" = torch.ops.aten.mm.default(view_183, permute_87)
    view_187: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_50, [1, 128, 384]);  mm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_188: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_187, [1, -1, 6, 64]);  view_187 = None
    permute_88: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_89: "f32[512, 384]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
    mm_51: "f32[128, 384]" = torch.ops.aten.mm.default(view_183, permute_89)
    view_190: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_51, [1, 128, 384]);  mm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_191: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_190, [1, -1, 6, 64]);  view_190 = None
    permute_90: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_191, [0, 2, 1, 3]);  view_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_91: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_88, [0, 1, 3, 2]);  permute_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_28: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_86, [1, 6, 128, 64]);  permute_86 = None
    view_192: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_28, [6, 128, 64]);  expand_28 = None
    expand_29: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_91, [1, 6, 64, 128]);  permute_91 = None
    view_193: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_29, [6, 64, 128]);  expand_29 = None
    bmm_14: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_192, view_193)
    view_194: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, [1, 6, 128, 128]);  bmm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_54: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_194, unsqueeze_4);  view_194 = unsqueeze_4 = None
    view_195: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_54, [6, 128, 128]);  add_54 = None
    view_196: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_195, [1, 6, 128, 128]);  view_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_7: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_196, [-1], True)
    sub_9: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_196, amax_7);  view_196 = amax_7 = None
    exp_7: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
    sum_8: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_9: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_29: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_29 = torch.ops.aten.native_dropout.default(div_9, 0.1, True);  div_9 = None
    getitem_58: "f32[1, 6, 128, 128]" = native_dropout_29[0]
    getitem_59: "b8[1, 6, 128, 128]" = native_dropout_29[1];  native_dropout_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_30: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_58, [1, 6, 128, 128]);  getitem_58 = None
    view_197: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_30, [6, 128, 128]);  expand_30 = None
    expand_31: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_90, [1, 6, 128, 64]);  permute_90 = None
    view_198: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_31, [6, 128, 64]);  expand_31 = None
    bmm_15: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_197, view_198)
    view_199: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_15, [1, 6, 128, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_92: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None
    clone_7: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_92, memory_format = torch.contiguous_format);  permute_92 = None
    view_200: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_7, [1, -1, 384]);  clone_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_93: "f32[384, 512]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    view_201: "f32[128, 384]" = torch.ops.aten.reshape.default(view_200, [128, 384]);  view_200 = None
    mm_52: "f32[128, 512]" = torch.ops.aten.mm.default(view_201, permute_93)
    view_202: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_52, [1, 128, 512]);  mm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_30 = torch.ops.aten.native_dropout.default(view_202, 0.1, True);  view_202 = None
    getitem_60: "f32[1, 128, 512]" = native_dropout_30[0]
    getitem_61: "b8[1, 128, 512]" = native_dropout_30[1];  native_dropout_30 = None
    add_55: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_52, getitem_60);  getitem_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_23: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_55, 2)
    mean_15: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_23, [-1], True);  pow_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_56: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_15, 1e-06);  mean_15 = None
    rsqrt_15: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
    mul_68: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_55, rsqrt_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_69: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_16, mul_68);  mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_94: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_98, [1, 0]);  primals_98 = None
    view_203: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_69, [128, 512]);  mul_69 = None
    mm_53: "f32[128, 1024]" = torch.ops.aten.mm.default(view_203, permute_94)
    view_204: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_53, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_70: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_204, 0.5)
    pow_24: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_204, 3.0)
    mul_71: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
    add_57: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_204, mul_71);  view_204 = mul_71 = None
    mul_72: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
    tanh_7: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_72);  mul_72 = None
    add_58: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
    mul_73: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_70, add_58);  mul_70 = add_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_95: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    mm_54: "f32[128, 1024]" = torch.ops.aten.mm.default(view_203, permute_95)
    view_206: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_54, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_74: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_73, view_206);  mul_73 = view_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_31 = torch.ops.aten.native_dropout.default(mul_74, 0.1, True);  mul_74 = None
    getitem_62: "f32[1, 128, 1024]" = native_dropout_31[0]
    getitem_63: "b8[1, 128, 1024]" = native_dropout_31[1];  native_dropout_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_96: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
    view_207: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_62, [128, 1024]);  getitem_62 = None
    mm_55: "f32[128, 512]" = torch.ops.aten.mm.default(view_207, permute_96)
    view_208: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_55, [1, 128, 512]);  mm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_32 = torch.ops.aten.native_dropout.default(view_208, 0.1, True);  view_208 = None
    getitem_64: "f32[1, 128, 512]" = native_dropout_32[0]
    getitem_65: "b8[1, 128, 512]" = native_dropout_32[1];  native_dropout_32 = None
    add_59: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_55, getitem_64);  getitem_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_25: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_59, 2)
    mean_16: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_25, [-1], True);  pow_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_60: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_16, 1e-06);  mean_16 = None
    rsqrt_16: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
    mul_75: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_59, rsqrt_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_76: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_17, mul_75);  mul_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1139, code: hidden_states = self.dropout(hidden_states)
    native_dropout_33 = torch.ops.aten.native_dropout.default(mul_76, 0.1, True);  mul_76 = None
    getitem_66: "f32[1, 128, 512]" = native_dropout_33[0]
    getitem_67: "b8[1, 128, 512]" = native_dropout_33[1];  native_dropout_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:984, code: input_ids = input_ids.view(-1, input_shape[-1])
    view_209: "i64[1, 128]" = torch.ops.aten.reshape.default(primals_193, [-1, 128]);  primals_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:994, code: inputs_embeds = self.embed_tokens(input_ids)
    embedding_2: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(primals_43, view_209);  primals_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:861, code: causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    unsqueeze_6: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1)
    repeat: "i64[1, 128, 128]" = torch.ops.aten.repeat.default(unsqueeze_6, [1, 128, 1]);  unsqueeze_6 = None
    unsqueeze_8: "i64[1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 2);  unsqueeze_3 = None
    le: "b8[1, 128, 128]" = torch.ops.aten.le.Tensor(repeat, unsqueeze_8);  repeat = unsqueeze_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:864, code: causal_mask = causal_mask.to(attention_mask.dtype)
    convert_element_type_3: "f32[1, 128, 128]" = torch.ops.prims.convert_element_type.default(le, torch.float32);  le = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:876, code: extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    unsqueeze_9: "f32[1, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/modeling_utils.py:928, code: extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    sub_10: "f32[1, 1, 128, 128]" = torch.ops.aten.sub.Tensor(1.0, unsqueeze_9);  unsqueeze_9 = None
    mul_78: "f32[1, 1, 128, 128]" = torch.ops.aten.mul.Tensor(sub_10, -3.4028234663852886e+38);  sub_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1049, code: hidden_states = self.dropout(inputs_embeds)
    native_dropout_34 = torch.ops.aten.native_dropout.default(embedding_2, 0.1, True);  embedding_2 = None
    getitem_68: "f32[1, 128, 512]" = native_dropout_34[0]
    getitem_69: "b8[1, 128, 512]" = native_dropout_34[1];  native_dropout_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_26: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(getitem_68, 2)
    mean_17: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_26, [-1], True);  pow_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_61: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_17, 1e-06);  mean_17 = None
    rsqrt_17: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_61);  add_61 = None
    mul_80: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(getitem_68, rsqrt_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_81: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_18, mul_80);  mul_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_97: "f32[512, 384]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    view_210: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_81, [128, 512]);  mul_81 = None
    mm_56: "f32[128, 384]" = torch.ops.aten.mm.default(view_210, permute_97)
    view_211: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_56, [1, 128, 384]);  mm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_212: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_211, [1, -1, 6, 64]);  view_211 = None
    permute_98: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_99: "f32[512, 384]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
    mm_57: "f32[128, 384]" = torch.ops.aten.mm.default(view_210, permute_99)
    view_214: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_57, [1, 128, 384]);  mm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_215: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_214, [1, -1, 6, 64]);  view_214 = None
    permute_100: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_215, [0, 2, 1, 3]);  view_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_101: "f32[512, 384]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    mm_58: "f32[128, 384]" = torch.ops.aten.mm.default(view_210, permute_101)
    view_217: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_58, [1, 128, 384]);  mm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_218: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_217, [1, -1, 6, 64]);  view_217 = None
    permute_102: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_103: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_100, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_32: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_98, [1, 6, 128, 64]);  permute_98 = None
    view_219: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_32, [6, 128, 64]);  expand_32 = None
    expand_33: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_103, [1, 6, 64, 128]);  permute_103 = None
    view_220: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_33, [6, 64, 128]);  expand_33 = None
    bmm_16: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_219, view_220)
    view_221: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [1, 6, 128, 128]);  bmm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:278, code: relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
    full_default_3: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    minimum_1: "i64[128, 128]" = torch.ops.aten.minimum.default(sub_1, full_default_3);  sub_1 = full_default_3 = None
    neg: "i64[128, 128]" = torch.ops.aten.neg.default(minimum_1);  minimum_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:283, code: is_small = relative_position < max_exact
    lt_1: "b8[128, 128]" = torch.ops.aten.lt.Scalar(neg, 16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:287, code: torch.log(relative_position.float() / max_exact)
    convert_element_type_5: "f32[128, 128]" = torch.ops.prims.convert_element_type.default(neg, torch.float32)
    div_10: "f32[128, 128]" = torch.ops.aten.div.Tensor(convert_element_type_5, 16);  convert_element_type_5 = None
    log_1: "f32[128, 128]" = torch.ops.aten.log.default(div_10);  div_10 = None
    div_11: "f32[128, 128]" = torch.ops.aten.div.Tensor(log_1, 2.0794415416798357);  log_1 = None
    mul_82: "f32[128, 128]" = torch.ops.aten.mul.Tensor(div_11, 16);  div_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:290, code: ).to(torch.long)
    convert_element_type_6: "i64[128, 128]" = torch.ops.prims.convert_element_type.default(mul_82, torch.int64);  mul_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:286, code: relative_position_if_large = max_exact + (
    add_62: "i64[128, 128]" = torch.ops.aten.add.Tensor(convert_element_type_6, 16);  convert_element_type_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:292, code: relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
    full_default_4: "i64[128, 128]" = torch.ops.aten.full.default([128, 128], 31, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:291, code: relative_position_if_large = torch.min(
    minimum_2: "i64[128, 128]" = torch.ops.aten.minimum.default(add_62, full_default_4);  add_62 = full_default_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:295, code: relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
    where_1: "i64[128, 128]" = torch.ops.aten.where.self(lt_1, neg, minimum_2);  lt_1 = neg = minimum_2 = None
    add_63: "i64[128, 128]" = torch.ops.aten.add.Tensor(where_1, 0);  where_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:311, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
    embedding_3: "f32[128, 128, 6]" = torch.ops.aten.embedding.default(primals_104, add_63);  primals_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:312, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    permute_104: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_3, [2, 0, 1]);  embedding_3 = None
    unsqueeze_16: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_104, 0);  permute_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:413, code: position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)
    add_64: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_16, mul_78);  unsqueeze_16 = mul_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_65: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_221, add_64);  view_221 = None
    view_222: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_65, [6, 128, 128]);  add_65 = None
    view_223: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_222, [1, 6, 128, 128]);  view_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_8: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_223, [-1], True)
    sub_13: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_223, amax_8);  view_223 = amax_8 = None
    exp_8: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_9: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_12: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_34: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_35 = torch.ops.aten.native_dropout.default(div_12, 0.1, True);  div_12 = None
    getitem_70: "f32[1, 6, 128, 128]" = native_dropout_35[0]
    getitem_71: "b8[1, 6, 128, 128]" = native_dropout_35[1];  native_dropout_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_34: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_70, [1, 6, 128, 128]);  getitem_70 = None
    view_224: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_34, [6, 128, 128]);  expand_34 = None
    expand_35: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_102, [1, 6, 128, 64])
    view_225: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_35, [6, 128, 64]);  expand_35 = None
    bmm_17: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_224, view_225)
    view_226: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_17, [1, 6, 128, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_105: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_226, [0, 2, 1, 3]);  view_226 = None
    clone_8: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_105, memory_format = torch.contiguous_format);  permute_105 = None
    view_227: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_8, [1, -1, 384]);  clone_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_106: "f32[384, 512]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    view_228: "f32[128, 384]" = torch.ops.aten.reshape.default(view_227, [128, 384]);  view_227 = None
    mm_59: "f32[128, 512]" = torch.ops.aten.mm.default(view_228, permute_106)
    view_229: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_59, [1, 128, 512]);  mm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_36 = torch.ops.aten.native_dropout.default(view_229, 0.1, True);  view_229 = None
    getitem_72: "f32[1, 128, 512]" = native_dropout_36[0]
    getitem_73: "b8[1, 128, 512]" = native_dropout_36[1];  native_dropout_36 = None
    add_66: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(getitem_68, getitem_72);  getitem_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_27: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_66, 2)
    mean_18: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_27, [-1], True);  pow_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_67: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_18, 1e-06);  mean_18 = None
    rsqrt_18: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
    mul_83: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_66, rsqrt_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_84: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_19, mul_83);  mul_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_107: "f32[512, 384]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
    view_230: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_84, [128, 512]);  mul_84 = None
    mm_60: "f32[128, 384]" = torch.ops.aten.mm.default(view_230, permute_107)
    view_231: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_60, [1, 128, 384]);  mm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_232: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_231, [1, -1, 6, 64]);  view_231 = None
    permute_108: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_232, [0, 2, 1, 3]);  view_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_109: "f32[512, 384]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    view_233: "f32[128, 512]" = torch.ops.aten.reshape.default(getitem_66, [128, 512])
    mm_61: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_109)
    view_234: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_61, [1, 128, 384]);  mm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_235: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_234, [1, -1, 6, 64]);  view_234 = None
    permute_110: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_235, [0, 2, 1, 3]);  view_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_111: "f32[512, 384]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
    mm_62: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_111)
    view_237: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_62, [1, 128, 384]);  mm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_238: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_237, [1, -1, 6, 64]);  view_237 = None
    permute_112: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_238, [0, 2, 1, 3]);  view_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_113: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_110, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_36: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_108, [1, 6, 128, 64]);  permute_108 = None
    view_239: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_36, [6, 128, 64]);  expand_36 = None
    expand_37: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_113, [1, 6, 64, 128]);  permute_113 = None
    view_240: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_37, [6, 64, 128]);  expand_37 = None
    bmm_18: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_239, view_240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_241: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_18, [1, 6, 128, 128]);  bmm_18 = None
    view_242: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_241, [6, 128, 128]);  view_241 = None
    view_243: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_242, [1, 6, 128, 128]);  view_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_9: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_243, [-1], True)
    sub_14: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_243, amax_9);  view_243 = amax_9 = None
    exp_9: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
    sum_10: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_13: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_36: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_37 = torch.ops.aten.native_dropout.default(div_13, 0.1, True);  div_13 = None
    getitem_74: "f32[1, 6, 128, 128]" = native_dropout_37[0]
    getitem_75: "b8[1, 6, 128, 128]" = native_dropout_37[1];  native_dropout_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_38: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_74, [1, 6, 128, 128]);  getitem_74 = None
    view_244: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_38, [6, 128, 128]);  expand_38 = None
    expand_39: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_112, [1, 6, 128, 64])
    view_245: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_39, [6, 128, 64]);  expand_39 = None
    bmm_19: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_244, view_245)
    view_246: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_19, [1, 6, 128, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_114: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
    clone_9: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_114, memory_format = torch.contiguous_format);  permute_114 = None
    view_247: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_9, [1, -1, 384]);  clone_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_115: "f32[384, 512]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    view_248: "f32[128, 384]" = torch.ops.aten.reshape.default(view_247, [128, 384]);  view_247 = None
    mm_63: "f32[128, 512]" = torch.ops.aten.mm.default(view_248, permute_115)
    view_249: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_63, [1, 128, 512]);  mm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_38 = torch.ops.aten.native_dropout.default(view_249, 0.1, True);  view_249 = None
    getitem_76: "f32[1, 128, 512]" = native_dropout_38[0]
    getitem_77: "b8[1, 128, 512]" = native_dropout_38[1];  native_dropout_38 = None
    add_70: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_66, getitem_76);  getitem_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_28: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_70, 2)
    mean_19: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_28, [-1], True);  pow_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_71: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_19, 1e-06);  mean_19 = None
    rsqrt_19: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
    mul_85: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_70, rsqrt_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_86: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_20, mul_85);  mul_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_116: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
    view_250: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_86, [128, 512]);  mul_86 = None
    mm_64: "f32[128, 1024]" = torch.ops.aten.mm.default(view_250, permute_116)
    view_251: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_64, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_87: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_251, 0.5)
    pow_29: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_251, 3.0)
    mul_88: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
    add_72: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_251, mul_88);  view_251 = mul_88 = None
    mul_89: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_72, 0.7978845608028654);  add_72 = None
    tanh_8: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_89);  mul_89 = None
    add_73: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
    mul_90: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_87, add_73);  mul_87 = add_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_117: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    mm_65: "f32[128, 1024]" = torch.ops.aten.mm.default(view_250, permute_117)
    view_253: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_65, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_91: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_90, view_253);  mul_90 = view_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_39 = torch.ops.aten.native_dropout.default(mul_91, 0.1, True);  mul_91 = None
    getitem_78: "f32[1, 128, 1024]" = native_dropout_39[0]
    getitem_79: "b8[1, 128, 1024]" = native_dropout_39[1];  native_dropout_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_118: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_112, [1, 0]);  primals_112 = None
    view_254: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_78, [128, 1024]);  getitem_78 = None
    mm_66: "f32[128, 512]" = torch.ops.aten.mm.default(view_254, permute_118)
    view_255: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_66, [1, 128, 512]);  mm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_40 = torch.ops.aten.native_dropout.default(view_255, 0.1, True);  view_255 = None
    getitem_80: "f32[1, 128, 512]" = native_dropout_40[0]
    getitem_81: "b8[1, 128, 512]" = native_dropout_40[1];  native_dropout_40 = None
    add_74: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_70, getitem_80);  getitem_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_30: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_74, 2)
    mean_20: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_30, [-1], True);  pow_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_75: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_20, 1e-06);  mean_20 = None
    rsqrt_20: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
    mul_92: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_74, rsqrt_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_93: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_21, mul_92);  mul_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_119: "f32[512, 384]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    view_256: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_93, [128, 512]);  mul_93 = None
    mm_67: "f32[128, 384]" = torch.ops.aten.mm.default(view_256, permute_119)
    view_257: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_67, [1, 128, 384]);  mm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_258: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_257, [1, -1, 6, 64]);  view_257 = None
    permute_120: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_121: "f32[512, 384]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
    mm_68: "f32[128, 384]" = torch.ops.aten.mm.default(view_256, permute_121)
    view_260: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_68, [1, 128, 384]);  mm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_261: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_260, [1, -1, 6, 64]);  view_260 = None
    permute_122: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_261, [0, 2, 1, 3]);  view_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_123: "f32[512, 384]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    mm_69: "f32[128, 384]" = torch.ops.aten.mm.default(view_256, permute_123)
    view_263: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_69, [1, 128, 384]);  mm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_264: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_263, [1, -1, 6, 64]);  view_263 = None
    permute_124: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_264, [0, 2, 1, 3]);  view_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_125: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_122, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_40: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_120, [1, 6, 128, 64]);  permute_120 = None
    view_265: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_40, [6, 128, 64]);  expand_40 = None
    expand_41: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_125, [1, 6, 64, 128]);  permute_125 = None
    view_266: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_41, [6, 64, 128]);  expand_41 = None
    bmm_20: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_265, view_266)
    view_267: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [1, 6, 128, 128]);  bmm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_76: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_267, add_64);  view_267 = None
    view_268: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_76, [6, 128, 128]);  add_76 = None
    view_269: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_268, [1, 6, 128, 128]);  view_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_10: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_269, [-1], True)
    sub_15: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_269, amax_10);  view_269 = amax_10 = None
    exp_10: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
    sum_11: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_14: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_40: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_41 = torch.ops.aten.native_dropout.default(div_14, 0.1, True);  div_14 = None
    getitem_82: "f32[1, 6, 128, 128]" = native_dropout_41[0]
    getitem_83: "b8[1, 6, 128, 128]" = native_dropout_41[1];  native_dropout_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_42: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_82, [1, 6, 128, 128]);  getitem_82 = None
    view_270: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_42, [6, 128, 128]);  expand_42 = None
    expand_43: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_124, [1, 6, 128, 64])
    view_271: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_43, [6, 128, 64]);  expand_43 = None
    bmm_21: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_270, view_271)
    view_272: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_21, [1, 6, 128, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_126: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_272, [0, 2, 1, 3]);  view_272 = None
    clone_10: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_126, memory_format = torch.contiguous_format);  permute_126 = None
    view_273: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_10, [1, -1, 384]);  clone_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_127: "f32[384, 512]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
    view_274: "f32[128, 384]" = torch.ops.aten.reshape.default(view_273, [128, 384]);  view_273 = None
    mm_70: "f32[128, 512]" = torch.ops.aten.mm.default(view_274, permute_127)
    view_275: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_70, [1, 128, 512]);  mm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_42 = torch.ops.aten.native_dropout.default(view_275, 0.1, True);  view_275 = None
    getitem_84: "f32[1, 128, 512]" = native_dropout_42[0]
    getitem_85: "b8[1, 128, 512]" = native_dropout_42[1];  native_dropout_42 = None
    add_77: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_74, getitem_84);  getitem_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_31: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_77, 2)
    mean_21: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_31, [-1], True);  pow_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_78: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_21, 1e-06);  mean_21 = None
    rsqrt_21: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
    mul_94: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_77, rsqrt_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_95: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_22, mul_94);  mul_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_128: "f32[512, 384]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    view_276: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_95, [128, 512]);  mul_95 = None
    mm_71: "f32[128, 384]" = torch.ops.aten.mm.default(view_276, permute_128)
    view_277: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_71, [1, 128, 384]);  mm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_278: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_277, [1, -1, 6, 64]);  view_277 = None
    permute_129: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_278, [0, 2, 1, 3]);  view_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_130: "f32[512, 384]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
    mm_72: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_130)
    view_280: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_72, [1, 128, 384]);  mm_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_281: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_280, [1, -1, 6, 64]);  view_280 = None
    permute_131: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_132: "f32[512, 384]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    mm_73: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_132)
    view_283: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_73, [1, 128, 384]);  mm_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_284: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_283, [1, -1, 6, 64]);  view_283 = None
    permute_133: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_134: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_131, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_44: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_129, [1, 6, 128, 64]);  permute_129 = None
    view_285: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_44, [6, 128, 64]);  expand_44 = None
    expand_45: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_134, [1, 6, 64, 128]);  permute_134 = None
    view_286: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_45, [6, 64, 128]);  expand_45 = None
    bmm_22: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_285, view_286)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_287: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, [1, 6, 128, 128]);  bmm_22 = None
    view_288: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_287, [6, 128, 128]);  view_287 = None
    view_289: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_288, [1, 6, 128, 128]);  view_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_11: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_289, [-1], True)
    sub_16: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_289, amax_11);  view_289 = amax_11 = None
    exp_11: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_12: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_15: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_42: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_43 = torch.ops.aten.native_dropout.default(div_15, 0.1, True);  div_15 = None
    getitem_86: "f32[1, 6, 128, 128]" = native_dropout_43[0]
    getitem_87: "b8[1, 6, 128, 128]" = native_dropout_43[1];  native_dropout_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_46: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_86, [1, 6, 128, 128]);  getitem_86 = None
    view_290: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_46, [6, 128, 128]);  expand_46 = None
    expand_47: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_133, [1, 6, 128, 64])
    view_291: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_47, [6, 128, 64]);  expand_47 = None
    bmm_23: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_290, view_291)
    view_292: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_23, [1, 6, 128, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_135: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_292, [0, 2, 1, 3]);  view_292 = None
    clone_11: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_135, memory_format = torch.contiguous_format);  permute_135 = None
    view_293: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_11, [1, -1, 384]);  clone_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_136: "f32[384, 512]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
    view_294: "f32[128, 384]" = torch.ops.aten.reshape.default(view_293, [128, 384]);  view_293 = None
    mm_74: "f32[128, 512]" = torch.ops.aten.mm.default(view_294, permute_136)
    view_295: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_74, [1, 128, 512]);  mm_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_44 = torch.ops.aten.native_dropout.default(view_295, 0.1, True);  view_295 = None
    getitem_88: "f32[1, 128, 512]" = native_dropout_44[0]
    getitem_89: "b8[1, 128, 512]" = native_dropout_44[1];  native_dropout_44 = None
    add_80: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_77, getitem_88);  getitem_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_32: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_80, 2)
    mean_22: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_32, [-1], True);  pow_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_81: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_22, 1e-06);  mean_22 = None
    rsqrt_22: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_81);  add_81 = None
    mul_96: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_80, rsqrt_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_97: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_23, mul_96);  mul_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_137: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    view_296: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_97, [128, 512]);  mul_97 = None
    mm_75: "f32[128, 1024]" = torch.ops.aten.mm.default(view_296, permute_137)
    view_297: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_75, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_98: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_297, 0.5)
    pow_33: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_297, 3.0)
    mul_99: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
    add_82: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_297, mul_99);  view_297 = mul_99 = None
    mul_100: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_82, 0.7978845608028654);  add_82 = None
    tanh_9: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_100);  mul_100 = None
    add_83: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
    mul_101: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_98, add_83);  mul_98 = add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_138: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
    mm_76: "f32[128, 1024]" = torch.ops.aten.mm.default(view_296, permute_138)
    view_299: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_76, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_102: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_101, view_299);  mul_101 = view_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_45 = torch.ops.aten.native_dropout.default(mul_102, 0.1, True);  mul_102 = None
    getitem_90: "f32[1, 128, 1024]" = native_dropout_45[0]
    getitem_91: "b8[1, 128, 1024]" = native_dropout_45[1];  native_dropout_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_139: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    view_300: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_90, [128, 1024]);  getitem_90 = None
    mm_77: "f32[128, 512]" = torch.ops.aten.mm.default(view_300, permute_139)
    view_301: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_77, [1, 128, 512]);  mm_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_46 = torch.ops.aten.native_dropout.default(view_301, 0.1, True);  view_301 = None
    getitem_92: "f32[1, 128, 512]" = native_dropout_46[0]
    getitem_93: "b8[1, 128, 512]" = native_dropout_46[1];  native_dropout_46 = None
    add_84: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_80, getitem_92);  getitem_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_34: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_84, 2)
    mean_23: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_34, [-1], True);  pow_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_85: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_23, 1e-06);  mean_23 = None
    rsqrt_23: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
    mul_103: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_84, rsqrt_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_104: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_24, mul_103);  mul_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_140: "f32[512, 384]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
    view_302: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_104, [128, 512]);  mul_104 = None
    mm_78: "f32[128, 384]" = torch.ops.aten.mm.default(view_302, permute_140)
    view_303: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_78, [1, 128, 384]);  mm_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_304: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_303, [1, -1, 6, 64]);  view_303 = None
    permute_141: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_304, [0, 2, 1, 3]);  view_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_142: "f32[512, 384]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    mm_79: "f32[128, 384]" = torch.ops.aten.mm.default(view_302, permute_142)
    view_306: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_79, [1, 128, 384]);  mm_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_307: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_306, [1, -1, 6, 64]);  view_306 = None
    permute_143: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_144: "f32[512, 384]" = torch.ops.aten.permute.default(primals_126, [1, 0]);  primals_126 = None
    mm_80: "f32[128, 384]" = torch.ops.aten.mm.default(view_302, permute_144)
    view_309: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_80, [1, 128, 384]);  mm_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_310: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_309, [1, -1, 6, 64]);  view_309 = None
    permute_145: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_310, [0, 2, 1, 3]);  view_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_146: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_143, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_48: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_141, [1, 6, 128, 64]);  permute_141 = None
    view_311: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_48, [6, 128, 64]);  expand_48 = None
    expand_49: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_146, [1, 6, 64, 128]);  permute_146 = None
    view_312: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_49, [6, 64, 128]);  expand_49 = None
    bmm_24: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_311, view_312)
    view_313: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [1, 6, 128, 128]);  bmm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_86: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_313, add_64);  view_313 = None
    view_314: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_86, [6, 128, 128]);  add_86 = None
    view_315: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_314, [1, 6, 128, 128]);  view_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_12: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_315, [-1], True)
    sub_17: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_315, amax_12);  view_315 = amax_12 = None
    exp_12: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
    sum_13: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
    div_16: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
    alias_46: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_47 = torch.ops.aten.native_dropout.default(div_16, 0.1, True);  div_16 = None
    getitem_94: "f32[1, 6, 128, 128]" = native_dropout_47[0]
    getitem_95: "b8[1, 6, 128, 128]" = native_dropout_47[1];  native_dropout_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_50: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_94, [1, 6, 128, 128]);  getitem_94 = None
    view_316: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_50, [6, 128, 128]);  expand_50 = None
    expand_51: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_145, [1, 6, 128, 64])
    view_317: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_51, [6, 128, 64]);  expand_51 = None
    bmm_25: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_316, view_317)
    view_318: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_25, [1, 6, 128, 64]);  bmm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_147: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None
    clone_12: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_147, memory_format = torch.contiguous_format);  permute_147 = None
    view_319: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_12, [1, -1, 384]);  clone_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_148: "f32[384, 512]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    view_320: "f32[128, 384]" = torch.ops.aten.reshape.default(view_319, [128, 384]);  view_319 = None
    mm_81: "f32[128, 512]" = torch.ops.aten.mm.default(view_320, permute_148)
    view_321: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_81, [1, 128, 512]);  mm_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_48 = torch.ops.aten.native_dropout.default(view_321, 0.1, True);  view_321 = None
    getitem_96: "f32[1, 128, 512]" = native_dropout_48[0]
    getitem_97: "b8[1, 128, 512]" = native_dropout_48[1];  native_dropout_48 = None
    add_87: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_84, getitem_96);  getitem_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_35: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_87, 2)
    mean_24: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_35, [-1], True);  pow_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_88: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_24, 1e-06);  mean_24 = None
    rsqrt_24: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_88);  add_88 = None
    mul_105: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_87, rsqrt_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_106: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_25, mul_105);  mul_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_149: "f32[512, 384]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
    view_322: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_106, [128, 512]);  mul_106 = None
    mm_82: "f32[128, 384]" = torch.ops.aten.mm.default(view_322, permute_149)
    view_323: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_82, [1, 128, 384]);  mm_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_324: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_323, [1, -1, 6, 64]);  view_323 = None
    permute_150: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_151: "f32[512, 384]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    mm_83: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_151)
    view_326: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_83, [1, 128, 384]);  mm_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_327: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_326, [1, -1, 6, 64]);  view_326 = None
    permute_152: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_327, [0, 2, 1, 3]);  view_327 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_153: "f32[512, 384]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
    mm_84: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_153)
    view_329: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_84, [1, 128, 384]);  mm_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_330: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_329, [1, -1, 6, 64]);  view_329 = None
    permute_154: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_155: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_152, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_52: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_150, [1, 6, 128, 64]);  permute_150 = None
    view_331: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_52, [6, 128, 64]);  expand_52 = None
    expand_53: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_155, [1, 6, 64, 128]);  permute_155 = None
    view_332: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_53, [6, 64, 128]);  expand_53 = None
    bmm_26: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_331, view_332)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_333: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_26, [1, 6, 128, 128]);  bmm_26 = None
    view_334: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_333, [6, 128, 128]);  view_333 = None
    view_335: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_334, [1, 6, 128, 128]);  view_334 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_13: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_335, [-1], True)
    sub_18: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_335, amax_13);  view_335 = amax_13 = None
    exp_13: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
    sum_14: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
    div_17: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
    alias_48: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_49 = torch.ops.aten.native_dropout.default(div_17, 0.1, True);  div_17 = None
    getitem_98: "f32[1, 6, 128, 128]" = native_dropout_49[0]
    getitem_99: "b8[1, 6, 128, 128]" = native_dropout_49[1];  native_dropout_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_54: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_98, [1, 6, 128, 128]);  getitem_98 = None
    view_336: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_54, [6, 128, 128]);  expand_54 = None
    expand_55: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_154, [1, 6, 128, 64])
    view_337: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_55, [6, 128, 64]);  expand_55 = None
    bmm_27: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_336, view_337)
    view_338: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_27, [1, 6, 128, 64]);  bmm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_156: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
    clone_13: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_156, memory_format = torch.contiguous_format);  permute_156 = None
    view_339: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_13, [1, -1, 384]);  clone_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_157: "f32[384, 512]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    view_340: "f32[128, 384]" = torch.ops.aten.reshape.default(view_339, [128, 384]);  view_339 = None
    mm_85: "f32[128, 512]" = torch.ops.aten.mm.default(view_340, permute_157)
    view_341: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_85, [1, 128, 512]);  mm_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_50 = torch.ops.aten.native_dropout.default(view_341, 0.1, True);  view_341 = None
    getitem_100: "f32[1, 128, 512]" = native_dropout_50[0]
    getitem_101: "b8[1, 128, 512]" = native_dropout_50[1];  native_dropout_50 = None
    add_90: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_87, getitem_100);  getitem_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_36: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_90, 2)
    mean_25: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_36, [-1], True);  pow_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_91: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_25, 1e-06);  mean_25 = None
    rsqrt_25: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_91);  add_91 = None
    mul_107: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_90, rsqrt_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_108: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_26, mul_107);  mul_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_158: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
    view_342: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_108, [128, 512]);  mul_108 = None
    mm_86: "f32[128, 1024]" = torch.ops.aten.mm.default(view_342, permute_158)
    view_343: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_86, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_109: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_343, 0.5)
    pow_37: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_343, 3.0)
    mul_110: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
    add_92: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_343, mul_110);  view_343 = mul_110 = None
    mul_111: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_92, 0.7978845608028654);  add_92 = None
    tanh_10: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_111);  mul_111 = None
    add_93: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
    mul_112: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_109, add_93);  mul_109 = add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_159: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    mm_87: "f32[128, 1024]" = torch.ops.aten.mm.default(view_342, permute_159)
    view_345: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_87, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_113: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_112, view_345);  mul_112 = view_345 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_51 = torch.ops.aten.native_dropout.default(mul_113, 0.1, True);  mul_113 = None
    getitem_102: "f32[1, 128, 1024]" = native_dropout_51[0]
    getitem_103: "b8[1, 128, 1024]" = native_dropout_51[1];  native_dropout_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_160: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
    view_346: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_102, [128, 1024]);  getitem_102 = None
    mm_88: "f32[128, 512]" = torch.ops.aten.mm.default(view_346, permute_160)
    view_347: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_88, [1, 128, 512]);  mm_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_52 = torch.ops.aten.native_dropout.default(view_347, 0.1, True);  view_347 = None
    getitem_104: "f32[1, 128, 512]" = native_dropout_52[0]
    getitem_105: "b8[1, 128, 512]" = native_dropout_52[1];  native_dropout_52 = None
    add_94: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_90, getitem_104);  getitem_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_38: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_94, 2)
    mean_26: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_38, [-1], True);  pow_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_95: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_26, 1e-06);  mean_26 = None
    rsqrt_26: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_95);  add_95 = None
    mul_114: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_94, rsqrt_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_115: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_27, mul_114);  mul_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_161: "f32[512, 384]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    view_348: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_115, [128, 512]);  mul_115 = None
    mm_89: "f32[128, 384]" = torch.ops.aten.mm.default(view_348, permute_161)
    view_349: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_89, [1, 128, 384]);  mm_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_350: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_349, [1, -1, 6, 64]);  view_349 = None
    permute_162: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_163: "f32[512, 384]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
    mm_90: "f32[128, 384]" = torch.ops.aten.mm.default(view_348, permute_163)
    view_352: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_90, [1, 128, 384]);  mm_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_353: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_352, [1, -1, 6, 64]);  view_352 = None
    permute_164: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_353, [0, 2, 1, 3]);  view_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_165: "f32[512, 384]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    mm_91: "f32[128, 384]" = torch.ops.aten.mm.default(view_348, permute_165)
    view_355: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_91, [1, 128, 384]);  mm_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_356: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_355, [1, -1, 6, 64]);  view_355 = None
    permute_166: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_356, [0, 2, 1, 3]);  view_356 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_167: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_56: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_162, [1, 6, 128, 64]);  permute_162 = None
    view_357: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_56, [6, 128, 64]);  expand_56 = None
    expand_57: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_167, [1, 6, 64, 128]);  permute_167 = None
    view_358: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_57, [6, 64, 128]);  expand_57 = None
    bmm_28: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_357, view_358)
    view_359: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [1, 6, 128, 128]);  bmm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_96: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_359, add_64);  view_359 = None
    view_360: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_96, [6, 128, 128]);  add_96 = None
    view_361: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_360, [1, 6, 128, 128]);  view_360 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_14: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_361, [-1], True)
    sub_19: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_361, amax_14);  view_361 = amax_14 = None
    exp_14: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_15: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
    div_18: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
    alias_52: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_53 = torch.ops.aten.native_dropout.default(div_18, 0.1, True);  div_18 = None
    getitem_106: "f32[1, 6, 128, 128]" = native_dropout_53[0]
    getitem_107: "b8[1, 6, 128, 128]" = native_dropout_53[1];  native_dropout_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_58: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_106, [1, 6, 128, 128]);  getitem_106 = None
    view_362: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_58, [6, 128, 128]);  expand_58 = None
    expand_59: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_166, [1, 6, 128, 64])
    view_363: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_59, [6, 128, 64]);  expand_59 = None
    bmm_29: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_362, view_363)
    view_364: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_29, [1, 6, 128, 64]);  bmm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_168: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None
    clone_14: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_168, memory_format = torch.contiguous_format);  permute_168 = None
    view_365: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_14, [1, -1, 384]);  clone_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_169: "f32[384, 512]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
    view_366: "f32[128, 384]" = torch.ops.aten.reshape.default(view_365, [128, 384]);  view_365 = None
    mm_92: "f32[128, 512]" = torch.ops.aten.mm.default(view_366, permute_169)
    view_367: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_92, [1, 128, 512]);  mm_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_54 = torch.ops.aten.native_dropout.default(view_367, 0.1, True);  view_367 = None
    getitem_108: "f32[1, 128, 512]" = native_dropout_54[0]
    getitem_109: "b8[1, 128, 512]" = native_dropout_54[1];  native_dropout_54 = None
    add_97: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_94, getitem_108);  getitem_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_39: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_97, 2)
    mean_27: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_39, [-1], True);  pow_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_98: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_27, 1e-06);  mean_27 = None
    rsqrt_27: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_98);  add_98 = None
    mul_116: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_97, rsqrt_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_117: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_28, mul_116);  mul_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_170: "f32[512, 384]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    view_368: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_117, [128, 512]);  mul_117 = None
    mm_93: "f32[128, 384]" = torch.ops.aten.mm.default(view_368, permute_170)
    view_369: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_93, [1, 128, 384]);  mm_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_370: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_369, [1, -1, 6, 64]);  view_369 = None
    permute_171: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_172: "f32[512, 384]" = torch.ops.aten.permute.default(primals_140, [1, 0]);  primals_140 = None
    mm_94: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_172)
    view_372: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_94, [1, 128, 384]);  mm_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_373: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_372, [1, -1, 6, 64]);  view_372 = None
    permute_173: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_174: "f32[512, 384]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    mm_95: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_174)
    view_375: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_95, [1, 128, 384]);  mm_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_376: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_375, [1, -1, 6, 64]);  view_375 = None
    permute_175: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_176: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_173, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_60: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_171, [1, 6, 128, 64]);  permute_171 = None
    view_377: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_60, [6, 128, 64]);  expand_60 = None
    expand_61: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_176, [1, 6, 64, 128]);  permute_176 = None
    view_378: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_61, [6, 64, 128]);  expand_61 = None
    bmm_30: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_377, view_378)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_379: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_30, [1, 6, 128, 128]);  bmm_30 = None
    view_380: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_379, [6, 128, 128]);  view_379 = None
    view_381: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_380, [1, 6, 128, 128]);  view_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_15: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_381, [-1], True)
    sub_20: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_381, amax_15);  view_381 = amax_15 = None
    exp_15: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
    sum_16: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
    div_19: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
    alias_54: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_55 = torch.ops.aten.native_dropout.default(div_19, 0.1, True);  div_19 = None
    getitem_110: "f32[1, 6, 128, 128]" = native_dropout_55[0]
    getitem_111: "b8[1, 6, 128, 128]" = native_dropout_55[1];  native_dropout_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_62: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_110, [1, 6, 128, 128]);  getitem_110 = None
    view_382: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_62, [6, 128, 128]);  expand_62 = None
    expand_63: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_175, [1, 6, 128, 64])
    view_383: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_63, [6, 128, 64]);  expand_63 = None
    bmm_31: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_382, view_383)
    view_384: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_31, [1, 6, 128, 64]);  bmm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_177: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_384, [0, 2, 1, 3]);  view_384 = None
    clone_15: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    view_385: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_15, [1, -1, 384]);  clone_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_178: "f32[384, 512]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
    view_386: "f32[128, 384]" = torch.ops.aten.reshape.default(view_385, [128, 384]);  view_385 = None
    mm_96: "f32[128, 512]" = torch.ops.aten.mm.default(view_386, permute_178)
    view_387: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_96, [1, 128, 512]);  mm_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_56 = torch.ops.aten.native_dropout.default(view_387, 0.1, True);  view_387 = None
    getitem_112: "f32[1, 128, 512]" = native_dropout_56[0]
    getitem_113: "b8[1, 128, 512]" = native_dropout_56[1];  native_dropout_56 = None
    add_100: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_97, getitem_112);  getitem_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_40: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_100, 2)
    mean_28: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_40, [-1], True);  pow_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_101: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_28, 1e-06);  mean_28 = None
    rsqrt_28: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_101);  add_101 = None
    mul_118: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_100, rsqrt_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_119: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_29, mul_118);  mul_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_179: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    view_388: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_119, [128, 512]);  mul_119 = None
    mm_97: "f32[128, 1024]" = torch.ops.aten.mm.default(view_388, permute_179)
    view_389: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_97, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_120: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_389, 0.5)
    pow_41: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_389, 3.0)
    mul_121: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
    add_102: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_389, mul_121);  view_389 = mul_121 = None
    mul_122: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_102, 0.7978845608028654);  add_102 = None
    tanh_11: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_122);  mul_122 = None
    add_103: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
    mul_123: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_120, add_103);  mul_120 = add_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_180: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
    mm_98: "f32[128, 1024]" = torch.ops.aten.mm.default(view_388, permute_180)
    view_391: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_98, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_124: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_123, view_391);  mul_123 = view_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_57 = torch.ops.aten.native_dropout.default(mul_124, 0.1, True);  mul_124 = None
    getitem_114: "f32[1, 128, 1024]" = native_dropout_57[0]
    getitem_115: "b8[1, 128, 1024]" = native_dropout_57[1];  native_dropout_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_181: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    view_392: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_114, [128, 1024]);  getitem_114 = None
    mm_99: "f32[128, 512]" = torch.ops.aten.mm.default(view_392, permute_181)
    view_393: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_99, [1, 128, 512]);  mm_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_58 = torch.ops.aten.native_dropout.default(view_393, 0.1, True);  view_393 = None
    getitem_116: "f32[1, 128, 512]" = native_dropout_58[0]
    getitem_117: "b8[1, 128, 512]" = native_dropout_58[1];  native_dropout_58 = None
    add_104: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_100, getitem_116);  getitem_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_42: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_104, 2)
    mean_29: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_42, [-1], True);  pow_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_105: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_29, 1e-06);  mean_29 = None
    rsqrt_29: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_105);  add_105 = None
    mul_125: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_104, rsqrt_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_126: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_30, mul_125);  mul_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_182: "f32[512, 384]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
    view_394: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_126, [128, 512]);  mul_126 = None
    mm_100: "f32[128, 384]" = torch.ops.aten.mm.default(view_394, permute_182)
    view_395: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_100, [1, 128, 384]);  mm_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_396: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_395, [1, -1, 6, 64]);  view_395 = None
    permute_183: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_396, [0, 2, 1, 3]);  view_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_184: "f32[512, 384]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    mm_101: "f32[128, 384]" = torch.ops.aten.mm.default(view_394, permute_184)
    view_398: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_101, [1, 128, 384]);  mm_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_399: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_398, [1, -1, 6, 64]);  view_398 = None
    permute_185: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_399, [0, 2, 1, 3]);  view_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_186: "f32[512, 384]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
    mm_102: "f32[128, 384]" = torch.ops.aten.mm.default(view_394, permute_186)
    view_401: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_102, [1, 128, 384]);  mm_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_402: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_401, [1, -1, 6, 64]);  view_401 = None
    permute_187: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_402, [0, 2, 1, 3]);  view_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_188: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_185, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_64: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_183, [1, 6, 128, 64]);  permute_183 = None
    view_403: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_64, [6, 128, 64]);  expand_64 = None
    expand_65: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_188, [1, 6, 64, 128]);  permute_188 = None
    view_404: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_65, [6, 64, 128]);  expand_65 = None
    bmm_32: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_403, view_404)
    view_405: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [1, 6, 128, 128]);  bmm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_106: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_405, add_64);  view_405 = None
    view_406: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_106, [6, 128, 128]);  add_106 = None
    view_407: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_406, [1, 6, 128, 128]);  view_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_16: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_407, [-1], True)
    sub_21: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_407, amax_16);  view_407 = amax_16 = None
    exp_16: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
    sum_17: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
    div_20: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
    alias_58: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_59 = torch.ops.aten.native_dropout.default(div_20, 0.1, True);  div_20 = None
    getitem_118: "f32[1, 6, 128, 128]" = native_dropout_59[0]
    getitem_119: "b8[1, 6, 128, 128]" = native_dropout_59[1];  native_dropout_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_66: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_118, [1, 6, 128, 128]);  getitem_118 = None
    view_408: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_66, [6, 128, 128]);  expand_66 = None
    expand_67: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_187, [1, 6, 128, 64])
    view_409: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_67, [6, 128, 64]);  expand_67 = None
    bmm_33: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_408, view_409)
    view_410: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_33, [1, 6, 128, 64]);  bmm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_189: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
    clone_16: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
    view_411: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_16, [1, -1, 384]);  clone_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_190: "f32[384, 512]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    view_412: "f32[128, 384]" = torch.ops.aten.reshape.default(view_411, [128, 384]);  view_411 = None
    mm_103: "f32[128, 512]" = torch.ops.aten.mm.default(view_412, permute_190)
    view_413: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_103, [1, 128, 512]);  mm_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_60 = torch.ops.aten.native_dropout.default(view_413, 0.1, True);  view_413 = None
    getitem_120: "f32[1, 128, 512]" = native_dropout_60[0]
    getitem_121: "b8[1, 128, 512]" = native_dropout_60[1];  native_dropout_60 = None
    add_107: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_104, getitem_120);  getitem_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_43: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_107, 2)
    mean_30: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_43, [-1], True);  pow_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_108: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_30, 1e-06);  mean_30 = None
    rsqrt_30: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
    mul_127: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_107, rsqrt_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_128: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_31, mul_127);  mul_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_191: "f32[512, 384]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
    view_414: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_128, [128, 512]);  mul_128 = None
    mm_104: "f32[128, 384]" = torch.ops.aten.mm.default(view_414, permute_191)
    view_415: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_104, [1, 128, 384]);  mm_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_416: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_415, [1, -1, 6, 64]);  view_415 = None
    permute_192: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_193: "f32[512, 384]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    mm_105: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_193)
    view_418: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_105, [1, 128, 384]);  mm_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_419: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_418, [1, -1, 6, 64]);  view_418 = None
    permute_194: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_419, [0, 2, 1, 3]);  view_419 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_195: "f32[512, 384]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
    mm_106: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_195)
    view_421: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_106, [1, 128, 384]);  mm_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_422: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_421, [1, -1, 6, 64]);  view_421 = None
    permute_196: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_422, [0, 2, 1, 3]);  view_422 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_197: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_194, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_68: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_192, [1, 6, 128, 64]);  permute_192 = None
    view_423: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_68, [6, 128, 64]);  expand_68 = None
    expand_69: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_197, [1, 6, 64, 128]);  permute_197 = None
    view_424: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_69, [6, 64, 128]);  expand_69 = None
    bmm_34: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_423, view_424)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_425: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_34, [1, 6, 128, 128]);  bmm_34 = None
    view_426: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_425, [6, 128, 128]);  view_425 = None
    view_427: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_426, [1, 6, 128, 128]);  view_426 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_17: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_427, [-1], True)
    sub_22: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_427, amax_17);  view_427 = amax_17 = None
    exp_17: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_18: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
    div_21: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
    alias_60: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_21)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_61 = torch.ops.aten.native_dropout.default(div_21, 0.1, True);  div_21 = None
    getitem_122: "f32[1, 6, 128, 128]" = native_dropout_61[0]
    getitem_123: "b8[1, 6, 128, 128]" = native_dropout_61[1];  native_dropout_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_70: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_122, [1, 6, 128, 128]);  getitem_122 = None
    view_428: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_70, [6, 128, 128]);  expand_70 = None
    expand_71: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_196, [1, 6, 128, 64])
    view_429: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_71, [6, 128, 64]);  expand_71 = None
    bmm_35: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_428, view_429)
    view_430: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_35, [1, 6, 128, 64]);  bmm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_198: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_430, [0, 2, 1, 3]);  view_430 = None
    clone_17: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_198, memory_format = torch.contiguous_format);  permute_198 = None
    view_431: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_17, [1, -1, 384]);  clone_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_199: "f32[384, 512]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    view_432: "f32[128, 384]" = torch.ops.aten.reshape.default(view_431, [128, 384]);  view_431 = None
    mm_107: "f32[128, 512]" = torch.ops.aten.mm.default(view_432, permute_199)
    view_433: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_107, [1, 128, 512]);  mm_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_62 = torch.ops.aten.native_dropout.default(view_433, 0.1, True);  view_433 = None
    getitem_124: "f32[1, 128, 512]" = native_dropout_62[0]
    getitem_125: "b8[1, 128, 512]" = native_dropout_62[1];  native_dropout_62 = None
    add_110: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_107, getitem_124);  getitem_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_44: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_110, 2)
    mean_31: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_44, [-1], True);  pow_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_111: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_31, 1e-06);  mean_31 = None
    rsqrt_31: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
    mul_129: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_110, rsqrt_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_130: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_32, mul_129);  mul_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_200: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_154, [1, 0]);  primals_154 = None
    view_434: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_130, [128, 512]);  mul_130 = None
    mm_108: "f32[128, 1024]" = torch.ops.aten.mm.default(view_434, permute_200)
    view_435: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_108, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_131: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_435, 0.5)
    pow_45: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_435, 3.0)
    mul_132: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
    add_112: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_435, mul_132);  view_435 = mul_132 = None
    mul_133: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_112, 0.7978845608028654);  add_112 = None
    tanh_12: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_133);  mul_133 = None
    add_113: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
    mul_134: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_131, add_113);  mul_131 = add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_201: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    mm_109: "f32[128, 1024]" = torch.ops.aten.mm.default(view_434, permute_201)
    view_437: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_109, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_135: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_134, view_437);  mul_134 = view_437 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_63 = torch.ops.aten.native_dropout.default(mul_135, 0.1, True);  mul_135 = None
    getitem_126: "f32[1, 128, 1024]" = native_dropout_63[0]
    getitem_127: "b8[1, 128, 1024]" = native_dropout_63[1];  native_dropout_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_202: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
    view_438: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_126, [128, 1024]);  getitem_126 = None
    mm_110: "f32[128, 512]" = torch.ops.aten.mm.default(view_438, permute_202)
    view_439: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_110, [1, 128, 512]);  mm_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_64 = torch.ops.aten.native_dropout.default(view_439, 0.1, True);  view_439 = None
    getitem_128: "f32[1, 128, 512]" = native_dropout_64[0]
    getitem_129: "b8[1, 128, 512]" = native_dropout_64[1];  native_dropout_64 = None
    add_114: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_110, getitem_128);  getitem_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_46: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_114, 2)
    mean_32: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_46, [-1], True);  pow_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_115: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_32, 1e-06);  mean_32 = None
    rsqrt_32: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
    mul_136: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_114, rsqrt_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_137: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_33, mul_136);  mul_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_203: "f32[512, 384]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    view_440: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_137, [128, 512]);  mul_137 = None
    mm_111: "f32[128, 384]" = torch.ops.aten.mm.default(view_440, permute_203)
    view_441: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_111, [1, 128, 384]);  mm_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_442: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_441, [1, -1, 6, 64]);  view_441 = None
    permute_204: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_442, [0, 2, 1, 3]);  view_442 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_205: "f32[512, 384]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
    mm_112: "f32[128, 384]" = torch.ops.aten.mm.default(view_440, permute_205)
    view_444: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_112, [1, 128, 384]);  mm_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_445: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_444, [1, -1, 6, 64]);  view_444 = None
    permute_206: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_445, [0, 2, 1, 3]);  view_445 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_207: "f32[512, 384]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    mm_113: "f32[128, 384]" = torch.ops.aten.mm.default(view_440, permute_207)
    view_447: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_113, [1, 128, 384]);  mm_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_448: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_447, [1, -1, 6, 64]);  view_447 = None
    permute_208: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_209: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_72: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_204, [1, 6, 128, 64]);  permute_204 = None
    view_449: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_72, [6, 128, 64]);  expand_72 = None
    expand_73: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_209, [1, 6, 64, 128]);  permute_209 = None
    view_450: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_73, [6, 64, 128]);  expand_73 = None
    bmm_36: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_449, view_450)
    view_451: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [1, 6, 128, 128]);  bmm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_116: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_451, add_64);  view_451 = None
    view_452: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_116, [6, 128, 128]);  add_116 = None
    view_453: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_452, [1, 6, 128, 128]);  view_452 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_18: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_453, [-1], True)
    sub_23: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_453, amax_18);  view_453 = amax_18 = None
    exp_18: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
    sum_19: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
    div_22: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
    alias_64: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_65 = torch.ops.aten.native_dropout.default(div_22, 0.1, True);  div_22 = None
    getitem_130: "f32[1, 6, 128, 128]" = native_dropout_65[0]
    getitem_131: "b8[1, 6, 128, 128]" = native_dropout_65[1];  native_dropout_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_74: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_130, [1, 6, 128, 128]);  getitem_130 = None
    view_454: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_74, [6, 128, 128]);  expand_74 = None
    expand_75: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_208, [1, 6, 128, 64])
    view_455: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_75, [6, 128, 64]);  expand_75 = None
    bmm_37: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_454, view_455)
    view_456: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_37, [1, 6, 128, 64]);  bmm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_210: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
    clone_18: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
    view_457: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_18, [1, -1, 384]);  clone_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_211: "f32[384, 512]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
    view_458: "f32[128, 384]" = torch.ops.aten.reshape.default(view_457, [128, 384]);  view_457 = None
    mm_114: "f32[128, 512]" = torch.ops.aten.mm.default(view_458, permute_211)
    view_459: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_114, [1, 128, 512]);  mm_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_66 = torch.ops.aten.native_dropout.default(view_459, 0.1, True);  view_459 = None
    getitem_132: "f32[1, 128, 512]" = native_dropout_66[0]
    getitem_133: "b8[1, 128, 512]" = native_dropout_66[1];  native_dropout_66 = None
    add_117: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_114, getitem_132);  getitem_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_47: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_117, 2)
    mean_33: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_47, [-1], True);  pow_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_118: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_33, 1e-06);  mean_33 = None
    rsqrt_33: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
    mul_138: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_117, rsqrt_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_139: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_34, mul_138);  mul_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_212: "f32[512, 384]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    view_460: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_139, [128, 512]);  mul_139 = None
    mm_115: "f32[128, 384]" = torch.ops.aten.mm.default(view_460, permute_212)
    view_461: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_115, [1, 128, 384]);  mm_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_462: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_461, [1, -1, 6, 64]);  view_461 = None
    permute_213: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_462, [0, 2, 1, 3]);  view_462 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_214: "f32[512, 384]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
    mm_116: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_214)
    view_464: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_116, [1, 128, 384]);  mm_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_465: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_464, [1, -1, 6, 64]);  view_464 = None
    permute_215: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_465, [0, 2, 1, 3]);  view_465 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_216: "f32[512, 384]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    mm_117: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_216)
    view_467: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_117, [1, 128, 384]);  mm_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_468: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_467, [1, -1, 6, 64]);  view_467 = None
    permute_217: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_468, [0, 2, 1, 3]);  view_468 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_218: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_215, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_76: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_213, [1, 6, 128, 64]);  permute_213 = None
    view_469: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_76, [6, 128, 64]);  expand_76 = None
    expand_77: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_218, [1, 6, 64, 128]);  permute_218 = None
    view_470: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_77, [6, 64, 128]);  expand_77 = None
    bmm_38: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_469, view_470)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_471: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_38, [1, 6, 128, 128]);  bmm_38 = None
    view_472: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_471, [6, 128, 128]);  view_471 = None
    view_473: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_472, [1, 6, 128, 128]);  view_472 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_19: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_473, [-1], True)
    sub_24: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_473, amax_19);  view_473 = amax_19 = None
    exp_19: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_24);  sub_24 = None
    sum_20: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
    div_23: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
    alias_66: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_23)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_67 = torch.ops.aten.native_dropout.default(div_23, 0.1, True);  div_23 = None
    getitem_134: "f32[1, 6, 128, 128]" = native_dropout_67[0]
    getitem_135: "b8[1, 6, 128, 128]" = native_dropout_67[1];  native_dropout_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_78: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_134, [1, 6, 128, 128]);  getitem_134 = None
    view_474: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_78, [6, 128, 128]);  expand_78 = None
    expand_79: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_217, [1, 6, 128, 64])
    view_475: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_79, [6, 128, 64]);  expand_79 = None
    bmm_39: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_474, view_475)
    view_476: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_39, [1, 6, 128, 64]);  bmm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_219: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_476, [0, 2, 1, 3]);  view_476 = None
    clone_19: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_219, memory_format = torch.contiguous_format);  permute_219 = None
    view_477: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_19, [1, -1, 384]);  clone_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_220: "f32[384, 512]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
    view_478: "f32[128, 384]" = torch.ops.aten.reshape.default(view_477, [128, 384]);  view_477 = None
    mm_118: "f32[128, 512]" = torch.ops.aten.mm.default(view_478, permute_220)
    view_479: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_118, [1, 128, 512]);  mm_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_68 = torch.ops.aten.native_dropout.default(view_479, 0.1, True);  view_479 = None
    getitem_136: "f32[1, 128, 512]" = native_dropout_68[0]
    getitem_137: "b8[1, 128, 512]" = native_dropout_68[1];  native_dropout_68 = None
    add_120: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_117, getitem_136);  getitem_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_48: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_120, 2)
    mean_34: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_48, [-1], True);  pow_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_121: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_34, 1e-06);  mean_34 = None
    rsqrt_34: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_121);  add_121 = None
    mul_140: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_120, rsqrt_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_141: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_35, mul_140);  mul_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_221: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    view_480: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_141, [128, 512]);  mul_141 = None
    mm_119: "f32[128, 1024]" = torch.ops.aten.mm.default(view_480, permute_221)
    view_481: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_119, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_142: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_481, 0.5)
    pow_49: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_481, 3.0)
    mul_143: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
    add_122: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_481, mul_143);  view_481 = mul_143 = None
    mul_144: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_122, 0.7978845608028654);  add_122 = None
    tanh_13: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_144);  mul_144 = None
    add_123: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
    mul_145: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_142, add_123);  mul_142 = add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_222: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
    mm_120: "f32[128, 1024]" = torch.ops.aten.mm.default(view_480, permute_222)
    view_483: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_120, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_146: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_145, view_483);  mul_145 = view_483 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_69 = torch.ops.aten.native_dropout.default(mul_146, 0.1, True);  mul_146 = None
    getitem_138: "f32[1, 128, 1024]" = native_dropout_69[0]
    getitem_139: "b8[1, 128, 1024]" = native_dropout_69[1];  native_dropout_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_223: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
    view_484: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_138, [128, 1024]);  getitem_138 = None
    mm_121: "f32[128, 512]" = torch.ops.aten.mm.default(view_484, permute_223)
    view_485: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_121, [1, 128, 512]);  mm_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_70 = torch.ops.aten.native_dropout.default(view_485, 0.1, True);  view_485 = None
    getitem_140: "f32[1, 128, 512]" = native_dropout_70[0]
    getitem_141: "b8[1, 128, 512]" = native_dropout_70[1];  native_dropout_70 = None
    add_124: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_120, getitem_140);  getitem_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_50: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_124, 2)
    mean_35: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_50, [-1], True);  pow_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_125: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_35, 1e-06);  mean_35 = None
    rsqrt_35: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_125);  add_125 = None
    mul_147: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_124, rsqrt_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_148: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_36, mul_147);  mul_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_224: "f32[512, 384]" = torch.ops.aten.permute.default(primals_168, [1, 0]);  primals_168 = None
    view_486: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_148, [128, 512]);  mul_148 = None
    mm_122: "f32[128, 384]" = torch.ops.aten.mm.default(view_486, permute_224)
    view_487: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_122, [1, 128, 384]);  mm_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_488: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_487, [1, -1, 6, 64]);  view_487 = None
    permute_225: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_226: "f32[512, 384]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    mm_123: "f32[128, 384]" = torch.ops.aten.mm.default(view_486, permute_226)
    view_490: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_123, [1, 128, 384]);  mm_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_491: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_490, [1, -1, 6, 64]);  view_490 = None
    permute_227: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_491, [0, 2, 1, 3]);  view_491 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_228: "f32[512, 384]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
    mm_124: "f32[128, 384]" = torch.ops.aten.mm.default(view_486, permute_228)
    view_493: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_124, [1, 128, 384]);  mm_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_494: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_493, [1, -1, 6, 64]);  view_493 = None
    permute_229: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_230: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_227, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_80: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_225, [1, 6, 128, 64]);  permute_225 = None
    view_495: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_80, [6, 128, 64]);  expand_80 = None
    expand_81: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_230, [1, 6, 64, 128]);  permute_230 = None
    view_496: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_81, [6, 64, 128]);  expand_81 = None
    bmm_40: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_495, view_496)
    view_497: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [1, 6, 128, 128]);  bmm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_126: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_497, add_64);  view_497 = None
    view_498: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_126, [6, 128, 128]);  add_126 = None
    view_499: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_498, [1, 6, 128, 128]);  view_498 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_20: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_499, [-1], True)
    sub_25: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_499, amax_20);  view_499 = amax_20 = None
    exp_20: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_21: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
    div_24: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
    alias_70: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_71 = torch.ops.aten.native_dropout.default(div_24, 0.1, True);  div_24 = None
    getitem_142: "f32[1, 6, 128, 128]" = native_dropout_71[0]
    getitem_143: "b8[1, 6, 128, 128]" = native_dropout_71[1];  native_dropout_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_82: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_142, [1, 6, 128, 128]);  getitem_142 = None
    view_500: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_82, [6, 128, 128]);  expand_82 = None
    expand_83: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_229, [1, 6, 128, 64])
    view_501: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_83, [6, 128, 64]);  expand_83 = None
    bmm_41: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_500, view_501)
    view_502: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_41, [1, 6, 128, 64]);  bmm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_231: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_502, [0, 2, 1, 3]);  view_502 = None
    clone_20: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_231, memory_format = torch.contiguous_format);  permute_231 = None
    view_503: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_20, [1, -1, 384]);  clone_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_232: "f32[384, 512]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    view_504: "f32[128, 384]" = torch.ops.aten.reshape.default(view_503, [128, 384]);  view_503 = None
    mm_125: "f32[128, 512]" = torch.ops.aten.mm.default(view_504, permute_232)
    view_505: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_125, [1, 128, 512]);  mm_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_72 = torch.ops.aten.native_dropout.default(view_505, 0.1, True);  view_505 = None
    getitem_144: "f32[1, 128, 512]" = native_dropout_72[0]
    getitem_145: "b8[1, 128, 512]" = native_dropout_72[1];  native_dropout_72 = None
    add_127: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_124, getitem_144);  getitem_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_51: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_127, 2)
    mean_36: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_51, [-1], True);  pow_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_128: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_36, 1e-06);  mean_36 = None
    rsqrt_36: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_128);  add_128 = None
    mul_149: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_127, rsqrt_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_150: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_37, mul_149);  mul_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_233: "f32[512, 384]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
    view_506: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_150, [128, 512]);  mul_150 = None
    mm_126: "f32[128, 384]" = torch.ops.aten.mm.default(view_506, permute_233)
    view_507: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_126, [1, 128, 384]);  mm_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_508: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_507, [1, -1, 6, 64]);  view_507 = None
    permute_234: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_508, [0, 2, 1, 3]);  view_508 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_235: "f32[512, 384]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
    mm_127: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_235)
    view_510: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_127, [1, 128, 384]);  mm_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_511: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_510, [1, -1, 6, 64]);  view_510 = None
    permute_236: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_511, [0, 2, 1, 3]);  view_511 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_237: "f32[512, 384]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
    mm_128: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_237)
    view_513: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_128, [1, 128, 384]);  mm_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_514: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_513, [1, -1, 6, 64]);  view_513 = None
    permute_238: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_514, [0, 2, 1, 3]);  view_514 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_239: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_236, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_84: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_234, [1, 6, 128, 64]);  permute_234 = None
    view_515: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_84, [6, 128, 64]);  expand_84 = None
    expand_85: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_239, [1, 6, 64, 128]);  permute_239 = None
    view_516: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_85, [6, 64, 128]);  expand_85 = None
    bmm_42: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_515, view_516)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_517: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_42, [1, 6, 128, 128]);  bmm_42 = None
    view_518: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_517, [6, 128, 128]);  view_517 = None
    view_519: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_518, [1, 6, 128, 128]);  view_518 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_21: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_519, [-1], True)
    sub_26: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_519, amax_21);  view_519 = amax_21 = None
    exp_21: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_26);  sub_26 = None
    sum_22: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
    div_25: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
    alias_72: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_73 = torch.ops.aten.native_dropout.default(div_25, 0.1, True);  div_25 = None
    getitem_146: "f32[1, 6, 128, 128]" = native_dropout_73[0]
    getitem_147: "b8[1, 6, 128, 128]" = native_dropout_73[1];  native_dropout_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_86: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_146, [1, 6, 128, 128]);  getitem_146 = None
    view_520: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_86, [6, 128, 128]);  expand_86 = None
    expand_87: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_238, [1, 6, 128, 64])
    view_521: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_87, [6, 128, 64]);  expand_87 = None
    bmm_43: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_520, view_521)
    view_522: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_43, [1, 6, 128, 64]);  bmm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_240: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None
    clone_21: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_240, memory_format = torch.contiguous_format);  permute_240 = None
    view_523: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_21, [1, -1, 384]);  clone_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_241: "f32[384, 512]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    view_524: "f32[128, 384]" = torch.ops.aten.reshape.default(view_523, [128, 384]);  view_523 = None
    mm_129: "f32[128, 512]" = torch.ops.aten.mm.default(view_524, permute_241)
    view_525: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_129, [1, 128, 512]);  mm_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_74 = torch.ops.aten.native_dropout.default(view_525, 0.1, True);  view_525 = None
    getitem_148: "f32[1, 128, 512]" = native_dropout_74[0]
    getitem_149: "b8[1, 128, 512]" = native_dropout_74[1];  native_dropout_74 = None
    add_130: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_127, getitem_148);  getitem_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_52: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_130, 2)
    mean_37: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_52, [-1], True);  pow_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_131: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_37, 1e-06);  mean_37 = None
    rsqrt_37: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_131);  add_131 = None
    mul_151: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_130, rsqrt_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_152: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_38, mul_151);  mul_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_242: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
    view_526: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_152, [128, 512]);  mul_152 = None
    mm_130: "f32[128, 1024]" = torch.ops.aten.mm.default(view_526, permute_242)
    view_527: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_130, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_153: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_527, 0.5)
    pow_53: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_527, 3.0)
    mul_154: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
    add_132: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_527, mul_154);  view_527 = mul_154 = None
    mul_155: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_132, 0.7978845608028654);  add_132 = None
    tanh_14: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_155);  mul_155 = None
    add_133: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
    mul_156: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_153, add_133);  mul_153 = add_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_243: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    mm_131: "f32[128, 1024]" = torch.ops.aten.mm.default(view_526, permute_243)
    view_529: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_131, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_157: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_156, view_529);  mul_156 = view_529 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_75 = torch.ops.aten.native_dropout.default(mul_157, 0.1, True);  mul_157 = None
    getitem_150: "f32[1, 128, 1024]" = native_dropout_75[0]
    getitem_151: "b8[1, 128, 1024]" = native_dropout_75[1];  native_dropout_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_244: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
    view_530: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_150, [128, 1024]);  getitem_150 = None
    mm_132: "f32[128, 512]" = torch.ops.aten.mm.default(view_530, permute_244)
    view_531: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_132, [1, 128, 512]);  mm_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_76 = torch.ops.aten.native_dropout.default(view_531, 0.1, True);  view_531 = None
    getitem_152: "f32[1, 128, 512]" = native_dropout_76[0]
    getitem_153: "b8[1, 128, 512]" = native_dropout_76[1];  native_dropout_76 = None
    add_134: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_130, getitem_152);  getitem_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_54: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_134, 2)
    mean_38: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_54, [-1], True);  pow_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_135: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_38, 1e-06);  mean_38 = None
    rsqrt_38: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_135);  add_135 = None
    mul_158: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_134, rsqrt_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_159: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_39, mul_158);  mul_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_245: "f32[512, 384]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    view_532: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_159, [128, 512]);  mul_159 = None
    mm_133: "f32[128, 384]" = torch.ops.aten.mm.default(view_532, permute_245)
    view_533: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_133, [1, 128, 384]);  mm_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_534: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_533, [1, -1, 6, 64]);  view_533 = None
    permute_246: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_247: "f32[512, 384]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
    mm_134: "f32[128, 384]" = torch.ops.aten.mm.default(view_532, permute_247)
    view_536: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_134, [1, 128, 384]);  mm_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_537: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_536, [1, -1, 6, 64]);  view_536 = None
    permute_248: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_249: "f32[512, 384]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    mm_135: "f32[128, 384]" = torch.ops.aten.mm.default(view_532, permute_249)
    view_539: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_135, [1, 128, 384]);  mm_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_540: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_539, [1, -1, 6, 64]);  view_539 = None
    permute_250: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_251: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_248, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_88: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_246, [1, 6, 128, 64]);  permute_246 = None
    view_541: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_88, [6, 128, 64]);  expand_88 = None
    expand_89: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_251, [1, 6, 64, 128]);  permute_251 = None
    view_542: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_89, [6, 64, 128]);  expand_89 = None
    bmm_44: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_541, view_542)
    view_543: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [1, 6, 128, 128]);  bmm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    add_136: "f32[1, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_543, add_64);  view_543 = add_64 = None
    view_544: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(add_136, [6, 128, 128]);  add_136 = None
    view_545: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_544, [1, 6, 128, 128]);  view_544 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_22: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_545, [-1], True)
    sub_27: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_545, amax_22);  view_545 = amax_22 = None
    exp_22: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
    sum_23: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
    div_26: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
    alias_76: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_77 = torch.ops.aten.native_dropout.default(div_26, 0.1, True);  div_26 = None
    getitem_154: "f32[1, 6, 128, 128]" = native_dropout_77[0]
    getitem_155: "b8[1, 6, 128, 128]" = native_dropout_77[1];  native_dropout_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_90: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_154, [1, 6, 128, 128]);  getitem_154 = None
    view_546: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_90, [6, 128, 128]);  expand_90 = None
    expand_91: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_250, [1, 6, 128, 64])
    view_547: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_91, [6, 128, 64]);  expand_91 = None
    bmm_45: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_546, view_547)
    view_548: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_45, [1, 6, 128, 64]);  bmm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_252: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_548, [0, 2, 1, 3]);  view_548 = None
    clone_22: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_252, memory_format = torch.contiguous_format);  permute_252 = None
    view_549: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_22, [1, -1, 384]);  clone_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_253: "f32[384, 512]" = torch.ops.aten.permute.default(primals_182, [1, 0]);  primals_182 = None
    view_550: "f32[128, 384]" = torch.ops.aten.reshape.default(view_549, [128, 384]);  view_549 = None
    mm_136: "f32[128, 512]" = torch.ops.aten.mm.default(view_550, permute_253)
    view_551: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_136, [1, 128, 512]);  mm_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:473, code: hidden_states = hidden_states + self.dropout(attention_output[0])
    native_dropout_78 = torch.ops.aten.native_dropout.default(view_551, 0.1, True);  view_551 = None
    getitem_156: "f32[1, 128, 512]" = native_dropout_78[0]
    getitem_157: "b8[1, 128, 512]" = native_dropout_78[1];  native_dropout_78 = None
    add_137: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_134, getitem_156);  getitem_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_55: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_137, 2)
    mean_39: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_55, [-1], True);  pow_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_138: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_39, 1e-06);  mean_39 = None
    rsqrt_39: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_138);  add_138 = None
    mul_160: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_137, rsqrt_39)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_161: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_40, mul_160);  mul_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_254: "f32[512, 384]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    view_552: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_161, [128, 512]);  mul_161 = None
    mm_137: "f32[128, 384]" = torch.ops.aten.mm.default(view_552, permute_254)
    view_553: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_137, [1, 128, 384]);  mm_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_554: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_553, [1, -1, 6, 64]);  view_553 = None
    permute_255: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_554, [0, 2, 1, 3]);  view_554 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_256: "f32[512, 384]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
    mm_138: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_256)
    view_556: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_138, [1, 128, 384]);  mm_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_557: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_556, [1, -1, 6, 64]);  view_556 = None
    permute_257: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_557, [0, 2, 1, 3]);  view_557 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_258: "f32[512, 384]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    mm_139: "f32[128, 384]" = torch.ops.aten.mm.default(view_233, permute_258)
    view_559: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(mm_139, [1, 128, 384]);  mm_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:348, code: return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    view_560: "f32[1, 128, 6, 64]" = torch.ops.aten.reshape.default(view_559, [1, -1, 6, 64]);  view_559 = None
    permute_259: "f32[1, 6, 128, 64]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:394, code: query_states, key_states.transpose(3, 2)
    permute_260: "f32[1, 6, 64, 128]" = torch.ops.aten.permute.default(permute_257, [0, 1, 3, 2])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    expand_92: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_255, [1, 6, 128, 64]);  permute_255 = None
    view_561: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_92, [6, 128, 64]);  expand_92 = None
    expand_93: "f32[1, 6, 64, 128]" = torch.ops.aten.expand.default(permute_260, [1, 6, 64, 128]);  permute_260 = None
    view_562: "f32[6, 64, 128]" = torch.ops.aten.reshape.default(expand_93, [6, 64, 128]);  expand_93 = None
    bmm_46: "f32[6, 128, 128]" = torch.ops.aten.bmm.default(view_561, view_562)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:422, code: scores += position_bias_masked
    view_563: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, [1, 6, 128, 128]);  bmm_46 = None
    view_564: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(view_563, [6, 128, 128]);  view_563 = None
    view_565: "f32[1, 6, 128, 128]" = torch.ops.aten.reshape.default(view_564, [1, 6, 128, 128]);  view_564 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    amax_23: "f32[1, 6, 128, 1]" = torch.ops.aten.amax.default(view_565, [-1], True)
    sub_28: "f32[1, 6, 128, 128]" = torch.ops.aten.sub.Tensor(view_565, amax_23);  view_565 = amax_23 = None
    exp_23: "f32[1, 6, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_24: "f32[1, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
    div_27: "f32[1, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
    alias_78: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(div_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:426, code: attn_weights = nn.functional.dropout(
    native_dropout_79 = torch.ops.aten.native_dropout.default(div_27, 0.1, True);  div_27 = None
    getitem_158: "f32[1, 6, 128, 128]" = native_dropout_79[0]
    getitem_159: "b8[1, 6, 128, 128]" = native_dropout_79[1];  native_dropout_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    expand_94: "f32[1, 6, 128, 128]" = torch.ops.aten.expand.default(getitem_158, [1, 6, 128, 128]);  getitem_158 = None
    view_566: "f32[6, 128, 128]" = torch.ops.aten.reshape.default(expand_94, [6, 128, 128]);  expand_94 = None
    expand_95: "f32[1, 6, 128, 64]" = torch.ops.aten.expand.default(permute_259, [1, 6, 128, 64])
    view_567: "f32[6, 128, 64]" = torch.ops.aten.reshape.default(expand_95, [6, 128, 64]);  expand_95 = None
    bmm_47: "f32[6, 128, 64]" = torch.ops.aten.bmm.default(view_566, view_567)
    view_568: "f32[1, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_47, [1, 6, 128, 64]);  bmm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:352, code: return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    permute_261: "f32[1, 128, 6, 64]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None
    clone_23: "f32[1, 128, 6, 64]" = torch.ops.aten.clone.default(permute_261, memory_format = torch.contiguous_format);  permute_261 = None
    view_569: "f32[1, 128, 384]" = torch.ops.aten.reshape.default(clone_23, [1, -1, 384]);  clone_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_262: "f32[384, 512]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
    view_570: "f32[128, 384]" = torch.ops.aten.reshape.default(view_569, [128, 384]);  view_569 = None
    mm_140: "f32[128, 512]" = torch.ops.aten.mm.default(view_570, permute_262)
    view_571: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_140, [1, 128, 512]);  mm_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:510, code: layer_output = hidden_states + self.dropout(attention_output[0])
    native_dropout_80 = torch.ops.aten.native_dropout.default(view_571, 0.1, True);  view_571 = None
    getitem_160: "f32[1, 128, 512]" = native_dropout_80[0]
    getitem_161: "b8[1, 128, 512]" = native_dropout_80[1];  native_dropout_80 = None
    add_140: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_137, getitem_160);  getitem_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_56: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_140, 2)
    mean_40: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_56, [-1], True);  pow_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_141: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_40, 1e-06);  mean_40 = None
    rsqrt_40: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
    mul_162: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_140, rsqrt_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_163: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_41, mul_162);  mul_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_263: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    view_572: "f32[128, 512]" = torch.ops.aten.reshape.default(mul_163, [128, 512]);  mul_163 = None
    mm_141: "f32[128, 1024]" = torch.ops.aten.mm.default(view_572, permute_263)
    view_573: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_141, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:56, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
    mul_164: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(view_573, 0.5)
    pow_57: "f32[1, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_573, 3.0)
    mul_165: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_57, 0.044715);  pow_57 = None
    add_142: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(view_573, mul_165);  view_573 = mul_165 = None
    mul_166: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(add_142, 0.7978845608028654);  add_142 = None
    tanh_15: "f32[1, 128, 1024]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
    add_143: "f32[1, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
    mul_167: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_164, add_143);  mul_164 = add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_264: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
    mm_142: "f32[128, 1024]" = torch.ops.aten.mm.default(view_572, permute_264)
    view_575: "f32[1, 128, 1024]" = torch.ops.aten.reshape.default(mm_142, [1, 128, 1024])
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:173, code: hidden_states = hidden_gelu * hidden_linear
    mul_168: "f32[1, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_167, view_575);  mul_167 = view_575 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:174, code: hidden_states = self.dropout(hidden_states)
    native_dropout_81 = torch.ops.aten.native_dropout.default(mul_168, 0.1, True);  mul_168 = None
    getitem_162: "f32[1, 128, 1024]" = native_dropout_81[0]
    getitem_163: "b8[1, 128, 1024]" = native_dropout_81[1];  native_dropout_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_265: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    view_576: "f32[128, 1024]" = torch.ops.aten.reshape.default(getitem_162, [128, 1024]);  getitem_162 = None
    mm_143: "f32[128, 512]" = torch.ops.aten.mm.default(view_576, permute_265)
    view_577: "f32[1, 128, 512]" = torch.ops.aten.reshape.default(mm_143, [1, 128, 512]);  mm_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:205, code: hidden_states = hidden_states + self.dropout(forwarded_states)
    native_dropout_82 = torch.ops.aten.native_dropout.default(view_577, 0.1, True);  view_577 = None
    getitem_164: "f32[1, 128, 512]" = native_dropout_82[0]
    getitem_165: "b8[1, 128, 512]" = native_dropout_82[1];  native_dropout_82 = None
    add_144: "f32[1, 128, 512]" = torch.ops.aten.add.Tensor(add_140, getitem_164);  getitem_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:127, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    pow_58: "f32[1, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_144, 2)
    mean_41: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_58, [-1], True);  pow_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:128, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    add_145: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_41, 1e-06);  mean_41 = None
    rsqrt_41: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_145);  add_145 = None
    mul_169: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(add_144, rsqrt_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:134, code: return self.weight * hidden_states
    mul_170: "f32[1, 128, 512]" = torch.ops.aten.mul.Tensor(primals_42, mul_169);  mul_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1139, code: hidden_states = self.dropout(hidden_states)
    native_dropout_83 = torch.ops.aten.native_dropout.default(mul_170, 0.1, True);  mul_170 = None
    getitem_166: "f32[1, 128, 512]" = native_dropout_83[0]
    getitem_167: "b8[1, 128, 512]" = native_dropout_83[1];  native_dropout_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1799, code: lm_logits = self.lm_head(sequence_output)
    permute_266: "f32[512, 250112]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
    view_578: "f32[128, 512]" = torch.ops.aten.reshape.default(getitem_166, [128, 512]);  getitem_166 = None
    mm_144: "f32[128, 250112]" = torch.ops.aten.mm.default(view_578, permute_266)
    view_579: "f32[1, 128, 250112]" = torch.ops.aten.reshape.default(mm_144, [1, 128, 250112]);  mm_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1806, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
    view_580: "f32[128, 250112]" = torch.ops.aten.reshape.default(view_579, [-1, 250112])
    view_581: "i64[128]" = torch.ops.aten.reshape.default(primals_192, [-1])
    amax_24: "f32[128, 1]" = torch.ops.aten.amax.default(view_580, [1], True)
    sub_29: "f32[128, 250112]" = torch.ops.aten.sub.Tensor(view_580, amax_24);  view_580 = amax_24 = None
    exp_24: "f32[128, 250112]" = torch.ops.aten.exp.default(sub_29)
    sum_25: "f32[128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
    log_2: "f32[128, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
    sub_30: "f32[128, 250112]" = torch.ops.aten.sub.Tensor(sub_29, log_2);  sub_29 = log_2 = None
    ne: "b8[128]" = torch.ops.aten.ne.Scalar(view_581, -100)
    full_default_6: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_2: "i64[128]" = torch.ops.aten.where.self(ne, view_581, full_default_6);  view_581 = full_default_6 = None
    unsqueeze_17: "i64[128, 1]" = torch.ops.aten.unsqueeze.default(where_2, 1);  where_2 = None
    gather: "f32[128, 1]" = torch.ops.aten.gather.default(sub_30, 1, unsqueeze_17);  unsqueeze_17 = None
    squeeze: "f32[128]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
    neg_1: "f32[128]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
    full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    where_3: "f32[128]" = torch.ops.aten.where.self(ne, neg_1, full_default_7);  neg_1 = full_default_7 = None
    sum_26: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
    convert_element_type_7: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
    sum_27: "f32[]" = torch.ops.aten.sum.default(where_3);  where_3 = None
    div_28: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_7);  sum_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:1799, code: lm_logits = self.lm_head(sequence_output)
    permute_269: "f32[250112, 512]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_273: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_277: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_281: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_285: "f32[512, 384]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_288: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_566, [0, 2, 1]);  view_566 = None
    permute_289: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_567, [0, 2, 1]);  view_567 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_87: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_78);  alias_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_290: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_561, [0, 2, 1]);  view_561 = None
    permute_291: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_562, [0, 2, 1]);  view_562 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_296: "f32[384, 512]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_301: "f32[384, 512]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_306: "f32[384, 512]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_310: "f32[512, 384]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_313: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_546, [0, 2, 1]);  view_546 = None
    permute_314: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_547, [0, 2, 1]);  view_547 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_89: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_76);  alias_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_315: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_541, [0, 2, 1]);  view_541 = None
    permute_316: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_542, [0, 2, 1]);  view_542 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_321: "f32[384, 512]" = torch.ops.aten.permute.default(permute_249, [1, 0]);  permute_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_326: "f32[384, 512]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_331: "f32[384, 512]" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_335: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_339: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_343: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_347: "f32[512, 384]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_350: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_520, [0, 2, 1]);  view_520 = None
    permute_351: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_93: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_72);  alias_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_352: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None
    permute_353: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_516, [0, 2, 1]);  view_516 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_358: "f32[384, 512]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_363: "f32[384, 512]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_368: "f32[384, 512]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_372: "f32[512, 384]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_375: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_500, [0, 2, 1]);  view_500 = None
    permute_376: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_501, [0, 2, 1]);  view_501 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_95: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_70);  alias_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_377: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None
    permute_378: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_383: "f32[384, 512]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_388: "f32[384, 512]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_393: "f32[384, 512]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_397: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_401: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_405: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_409: "f32[512, 384]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_412: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_474, [0, 2, 1]);  view_474 = None
    permute_413: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_475, [0, 2, 1]);  view_475 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_99: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_66);  alias_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_414: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_469, [0, 2, 1]);  view_469 = None
    permute_415: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_470, [0, 2, 1]);  view_470 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_420: "f32[384, 512]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_425: "f32[384, 512]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_430: "f32[384, 512]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_434: "f32[512, 384]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_437: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_454, [0, 2, 1]);  view_454 = None
    permute_438: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_455, [0, 2, 1]);  view_455 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_101: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_64);  alias_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_439: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None
    permute_440: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_450, [0, 2, 1]);  view_450 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_445: "f32[384, 512]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_450: "f32[384, 512]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_455: "f32[384, 512]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_459: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_463: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_467: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_471: "f32[512, 384]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_474: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_428, [0, 2, 1]);  view_428 = None
    permute_475: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_429, [0, 2, 1]);  view_429 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_105: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_60);  alias_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_476: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_423, [0, 2, 1]);  view_423 = None
    permute_477: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_424, [0, 2, 1]);  view_424 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_482: "f32[384, 512]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_487: "f32[384, 512]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_492: "f32[384, 512]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_496: "f32[512, 384]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_499: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_408, [0, 2, 1]);  view_408 = None
    permute_500: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_409, [0, 2, 1]);  view_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_107: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_58);  alias_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_501: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_403, [0, 2, 1]);  view_403 = None
    permute_502: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_404, [0, 2, 1]);  view_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_507: "f32[384, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_512: "f32[384, 512]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_517: "f32[384, 512]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_521: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_525: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_529: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_533: "f32[512, 384]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_536: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_382, [0, 2, 1]);  view_382 = None
    permute_537: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_111: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_54);  alias_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_538: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_377, [0, 2, 1]);  view_377 = None
    permute_539: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_378, [0, 2, 1]);  view_378 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_544: "f32[384, 512]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_549: "f32[384, 512]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_554: "f32[384, 512]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_558: "f32[512, 384]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_561: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_362, [0, 2, 1]);  view_362 = None
    permute_562: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_113: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_52);  alias_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_563: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_357, [0, 2, 1]);  view_357 = None
    permute_564: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_358, [0, 2, 1]);  view_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_569: "f32[384, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_574: "f32[384, 512]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_579: "f32[384, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_583: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_587: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_591: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_595: "f32[512, 384]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_598: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_336, [0, 2, 1]);  view_336 = None
    permute_599: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_337, [0, 2, 1]);  view_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_117: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_48);  alias_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_600: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None
    permute_601: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_332, [0, 2, 1]);  view_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_606: "f32[384, 512]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_611: "f32[384, 512]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_616: "f32[384, 512]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_620: "f32[512, 384]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_623: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_316, [0, 2, 1]);  view_316 = None
    permute_624: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_317, [0, 2, 1]);  view_317 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_119: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_46);  alias_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_625: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
    permute_626: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_631: "f32[384, 512]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_636: "f32[384, 512]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_641: "f32[384, 512]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_645: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_649: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_653: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_657: "f32[512, 384]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_660: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_290, [0, 2, 1]);  view_290 = None
    permute_661: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_291, [0, 2, 1]);  view_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_123: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_42);  alias_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_662: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_285, [0, 2, 1]);  view_285 = None
    permute_663: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_668: "f32[384, 512]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_673: "f32[384, 512]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_678: "f32[384, 512]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_682: "f32[512, 384]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_685: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_270, [0, 2, 1]);  view_270 = None
    permute_686: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_125: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_40);  alias_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_687: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_265, [0, 2, 1]);  view_265 = None
    permute_688: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_693: "f32[384, 512]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_698: "f32[384, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_703: "f32[384, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_707: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_711: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_715: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_719: "f32[512, 384]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_722: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
    permute_723: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_129: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_36);  alias_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_724: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_239, [0, 2, 1]);  view_239 = None
    permute_725: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_240, [0, 2, 1]);  view_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_730: "f32[384, 512]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:363, code: hidden_states = shape(proj_layer(key_value_states))
    permute_735: "f32[384, 512]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_740: "f32[384, 512]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_744: "f32[512, 384]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_747: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_224, [0, 2, 1]);  view_224 = None
    permute_748: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_225, [0, 2, 1]);  view_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_131: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_750: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_219, [0, 2, 1]);  view_219 = None
    permute_751: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_756: "f32[384, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_761: "f32[384, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_766: "f32[384, 512]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_770: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_774: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_778: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_782: "f32[512, 384]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_785: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_197, [0, 2, 1]);  view_197 = None
    permute_786: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_198, [0, 2, 1]);  view_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_136: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_787: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_192, [0, 2, 1]);  view_192 = None
    permute_788: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_193, [0, 2, 1]);  view_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_793: "f32[384, 512]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_798: "f32[384, 512]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_803: "f32[384, 512]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_807: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_811: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_815: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_819: "f32[512, 384]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_822: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_171, [0, 2, 1]);  view_171 = None
    permute_823: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_140: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_824: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    permute_825: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_830: "f32[384, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_835: "f32[384, 512]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_840: "f32[384, 512]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_844: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_848: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_852: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_856: "f32[512, 384]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_859: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    permute_860: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_146, [0, 2, 1]);  view_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_144: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_861: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_140, [0, 2, 1]);  view_140 = None
    permute_862: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_867: "f32[384, 512]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_872: "f32[384, 512]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_877: "f32[384, 512]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_881: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_885: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_889: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_893: "f32[512, 384]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_896: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
    permute_897: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_148: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_898: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
    permute_899: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_904: "f32[384, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_909: "f32[384, 512]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_914: "f32[384, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_918: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_922: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_926: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_930: "f32[512, 384]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_933: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None
    permute_934: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_94, [0, 2, 1]);  view_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_152: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_935: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_88, [0, 2, 1]);  view_88 = None
    permute_936: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_89, [0, 2, 1]);  view_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_941: "f32[384, 512]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_946: "f32[384, 512]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_951: "f32[384, 512]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_955: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_959: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_963: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_967: "f32[512, 384]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_970: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_67, [0, 2, 1]);  view_67 = None
    permute_971: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_156: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_972: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_62, [0, 2, 1]);  view_62 = None
    permute_973: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_63, [0, 2, 1]);  view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_978: "f32[384, 512]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_983: "f32[384, 512]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_988: "f32[384, 512]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_992: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_996: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_1000: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_1004: "f32[512, 384]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_1007: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_41, [0, 2, 1]);  view_41 = None
    permute_1008: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_42, [0, 2, 1]);  view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_160: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_1009: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_36, [0, 2, 1]);  view_36 = None
    permute_1010: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_37, [0, 2, 1]);  view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_1015: "f32[384, 512]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_1020: "f32[384, 512]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_1025: "f32[384, 512]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:186, code: hidden_states = self.wo(hidden_states)
    permute_1029: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:172, code: hidden_linear = self.wi_1(hidden_states)
    permute_1033: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:171, code: hidden_gelu = self.act(self.wi_0(hidden_states))
    permute_1037: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:435, code: attn_output = self.o(attn_output)
    permute_1041: "f32[512, 384]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:434, code: attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    permute_1044: "f32[6, 128, 128]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
    permute_1045: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:423, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
    alias_164: "f32[1, 6, 128, 128]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:393, code: scores = torch.matmul(
    permute_1047: "f32[6, 64, 128]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    permute_1048: "f32[6, 128, 64]" = torch.ops.aten.permute.default(view_11, [0, 2, 1]);  view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_1053: "f32[384, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:359, code: hidden_states = shape(proj_layer(hidden_states))
    permute_1058: "f32[384, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/mt5/modeling_mt5.py:382, code: query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)
    permute_1063: "f32[384, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    return [div_28, view_579, permute_100, permute_102, permute_110, permute_112, permute_122, permute_124, permute_131, permute_133, permute_143, permute_145, permute_152, permute_154, permute_164, permute_166, permute_173, permute_175, permute_185, permute_187, permute_194, permute_196, permute_206, permute_208, permute_215, permute_217, permute_227, permute_229, permute_236, permute_238, permute_248, permute_250, permute_257, permute_259, getitem_66, primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_192, view, getitem, getitem_1, rsqrt, view_1, add_3, getitem_3, view_19, getitem_5, add_6, rsqrt_1, view_21, mm_4, tanh, mm_5, getitem_7, view_25, getitem_9, add_10, rsqrt_2, view_27, getitem_11, view_45, getitem_13, add_13, rsqrt_3, view_47, mm_11, tanh_1, mm_12, getitem_15, view_51, getitem_17, add_17, rsqrt_4, view_53, getitem_19, view_71, getitem_21, add_20, rsqrt_5, view_73, mm_18, tanh_2, mm_19, getitem_23, view_77, getitem_25, add_24, rsqrt_6, view_79, getitem_27, view_97, getitem_29, add_27, rsqrt_7, view_99, mm_25, tanh_3, mm_26, getitem_31, view_103, getitem_33, add_31, rsqrt_8, view_105, getitem_35, view_123, getitem_37, add_34, rsqrt_9, view_125, mm_32, tanh_4, mm_33, getitem_39, view_129, getitem_41, add_38, rsqrt_10, view_131, getitem_43, view_149, getitem_45, add_41, rsqrt_11, view_151, mm_39, tanh_5, mm_40, getitem_47, view_155, getitem_49, add_45, rsqrt_12, view_157, getitem_51, view_175, getitem_53, add_48, rsqrt_13, view_177, mm_46, tanh_6, mm_47, getitem_55, view_181, getitem_57, add_52, rsqrt_14, view_183, getitem_59, view_201, getitem_61, add_55, rsqrt_15, view_203, mm_53, tanh_7, mm_54, getitem_63, view_207, getitem_65, add_59, rsqrt_16, getitem_67, view_209, getitem_68, getitem_69, rsqrt_17, view_210, add_63, getitem_71, view_228, getitem_73, add_66, rsqrt_18, view_230, view_233, getitem_75, view_248, getitem_77, add_70, rsqrt_19, view_250, mm_64, tanh_8, mm_65, getitem_79, view_254, getitem_81, add_74, rsqrt_20, view_256, getitem_83, view_274, getitem_85, add_77, rsqrt_21, view_276, getitem_87, view_294, getitem_89, add_80, rsqrt_22, view_296, mm_75, tanh_9, mm_76, getitem_91, view_300, getitem_93, add_84, rsqrt_23, view_302, getitem_95, view_320, getitem_97, add_87, rsqrt_24, view_322, getitem_99, view_340, getitem_101, add_90, rsqrt_25, view_342, mm_86, tanh_10, mm_87, getitem_103, view_346, getitem_105, add_94, rsqrt_26, view_348, getitem_107, view_366, getitem_109, add_97, rsqrt_27, view_368, getitem_111, view_386, getitem_113, add_100, rsqrt_28, view_388, mm_97, tanh_11, mm_98, getitem_115, view_392, getitem_117, add_104, rsqrt_29, view_394, getitem_119, view_412, getitem_121, add_107, rsqrt_30, view_414, getitem_123, view_432, getitem_125, add_110, rsqrt_31, view_434, mm_108, tanh_12, mm_109, getitem_127, view_438, getitem_129, add_114, rsqrt_32, view_440, getitem_131, view_458, getitem_133, add_117, rsqrt_33, view_460, getitem_135, view_478, getitem_137, add_120, rsqrt_34, view_480, mm_119, tanh_13, mm_120, getitem_139, view_484, getitem_141, add_124, rsqrt_35, view_486, getitem_143, view_504, getitem_145, add_127, rsqrt_36, view_506, getitem_147, view_524, getitem_149, add_130, rsqrt_37, view_526, mm_130, tanh_14, mm_131, getitem_151, view_530, getitem_153, add_134, rsqrt_38, view_532, getitem_155, view_550, getitem_157, add_137, rsqrt_39, view_552, getitem_159, view_570, getitem_161, add_140, rsqrt_40, view_572, mm_141, tanh_15, mm_142, getitem_163, view_576, getitem_165, add_144, rsqrt_41, getitem_167, view_578, sub_30, convert_element_type_7, permute_269, permute_273, permute_277, permute_281, permute_285, permute_288, permute_289, alias_87, permute_290, permute_291, permute_296, permute_301, permute_306, permute_310, permute_313, permute_314, alias_89, permute_315, permute_316, permute_321, permute_326, permute_331, permute_335, permute_339, permute_343, permute_347, permute_350, permute_351, alias_93, permute_352, permute_353, permute_358, permute_363, permute_368, permute_372, permute_375, permute_376, alias_95, permute_377, permute_378, permute_383, permute_388, permute_393, permute_397, permute_401, permute_405, permute_409, permute_412, permute_413, alias_99, permute_414, permute_415, permute_420, permute_425, permute_430, permute_434, permute_437, permute_438, alias_101, permute_439, permute_440, permute_445, permute_450, permute_455, permute_459, permute_463, permute_467, permute_471, permute_474, permute_475, alias_105, permute_476, permute_477, permute_482, permute_487, permute_492, permute_496, permute_499, permute_500, alias_107, permute_501, permute_502, permute_507, permute_512, permute_517, permute_521, permute_525, permute_529, permute_533, permute_536, permute_537, alias_111, permute_538, permute_539, permute_544, permute_549, permute_554, permute_558, permute_561, permute_562, alias_113, permute_563, permute_564, permute_569, permute_574, permute_579, permute_583, permute_587, permute_591, permute_595, permute_598, permute_599, alias_117, permute_600, permute_601, permute_606, permute_611, permute_616, permute_620, permute_623, permute_624, alias_119, permute_625, permute_626, permute_631, permute_636, permute_641, permute_645, permute_649, permute_653, permute_657, permute_660, permute_661, alias_123, permute_662, permute_663, permute_668, permute_673, permute_678, permute_682, permute_685, permute_686, alias_125, permute_687, permute_688, permute_693, permute_698, permute_703, permute_707, permute_711, permute_715, permute_719, permute_722, permute_723, alias_129, permute_724, permute_725, permute_730, permute_735, permute_740, permute_744, permute_747, permute_748, alias_131, permute_750, permute_751, permute_756, permute_761, permute_766, permute_770, permute_774, permute_778, permute_782, permute_785, permute_786, alias_136, permute_787, permute_788, permute_793, permute_798, permute_803, permute_807, permute_811, permute_815, permute_819, permute_822, permute_823, alias_140, permute_824, permute_825, permute_830, permute_835, permute_840, permute_844, permute_848, permute_852, permute_856, permute_859, permute_860, alias_144, permute_861, permute_862, permute_867, permute_872, permute_877, permute_881, permute_885, permute_889, permute_893, permute_896, permute_897, alias_148, permute_898, permute_899, permute_904, permute_909, permute_914, permute_918, permute_922, permute_926, permute_930, permute_933, permute_934, alias_152, permute_935, permute_936, permute_941, permute_946, permute_951, permute_955, permute_959, permute_963, permute_967, permute_970, permute_971, alias_156, permute_972, permute_973, permute_978, permute_983, permute_988, permute_992, permute_996, permute_1000, permute_1004, permute_1007, permute_1008, alias_160, permute_1009, permute_1010, permute_1015, permute_1020, permute_1025, permute_1029, permute_1033, permute_1037, permute_1041, permute_1044, permute_1045, alias_164, permute_1047, permute_1048, permute_1053, permute_1058, permute_1063]
    