from __future__ import annotations



def forward(self, primals, tangents):
    primals_1: "f32[768]"; primals_2: "f32[768]"; primals_3: "f32[768]"; primals_4: "f32[768]"; primals_5: "f32[768]"; primals_6: "f32[768]"; primals_7: "f32[768]"; primals_8: "f32[768]"; primals_9: "f32[768]"; primals_10: "f32[768]"; primals_11: "f32[768]"; primals_12: "f32[768]"; primals_13: "f32[768]"; primals_14: "f32[768]"; primals_15: "f32[768]"; primals_16: "f32[768]"; primals_17: "f32[768]"; primals_18: "f32[768]"; primals_19: "f32[768]"; primals_20: "f32[768]"; primals_21: "f32[768]"; primals_22: "f32[768]"; primals_23: "f32[768]"; primals_24: "f32[768]"; primals_25: "f32[768]"; primals_26: "f32[768]"; primals_27: "f32[768]"; primals_28: "f32[768]"; primals_29: "f32[768]"; primals_30: "f32[768]"; primals_31: "f32[768]"; primals_32: "f32[768]"; primals_33: "f32[768]"; primals_34: "f32[768]"; primals_35: "f32[768]"; primals_36: "f32[768]"; primals_37: "f32[768]"; primals_38: "f32[768]"; primals_39: "f32[768]"; primals_40: "f32[768]"; primals_41: "f32[768]"; primals_42: "f32[768]"; primals_43: "f32[768]"; primals_44: "f32[768]"; primals_45: "f32[768]"; primals_46: "f32[768]"; primals_47: "f32[768]"; primals_48: "f32[768]"; primals_49: "f32[20005, 768]"; primals_50: "f32[3, 768]"; primals_51: "f32[768, 768]"; primals_52: "f32[768]"; primals_53: "f32[768, 768]"; primals_54: "f32[768]"; primals_55: "f32[768, 768]"; primals_56: "f32[768]"; primals_57: "f32[768, 768]"; primals_58: "f32[768]"; primals_59: "f32[3072, 768]"; primals_60: "f32[3072]"; primals_61: "f32[768, 3072]"; primals_62: "f32[768]"; primals_63: "f32[768, 768]"; primals_64: "f32[768]"; primals_65: "f32[768, 768]"; primals_66: "f32[768]"; primals_67: "f32[768, 768]"; primals_68: "f32[768]"; primals_69: "f32[768, 768]"; primals_70: "f32[768]"; primals_71: "f32[3072, 768]"; primals_72: "f32[3072]"; primals_73: "f32[768, 3072]"; primals_74: "f32[768]"; primals_75: "f32[768, 768]"; primals_76: "f32[768]"; primals_77: "f32[768, 768]"; primals_78: "f32[768]"; primals_79: "f32[768, 768]"; primals_80: "f32[768]"; primals_81: "f32[768, 768]"; primals_82: "f32[768]"; primals_83: "f32[3072, 768]"; primals_84: "f32[3072]"; primals_85: "f32[768, 3072]"; primals_86: "f32[768]"; primals_87: "f32[768, 768]"; primals_88: "f32[768]"; primals_89: "f32[768, 768]"; primals_90: "f32[768]"; primals_91: "f32[768, 768]"; primals_92: "f32[768]"; primals_93: "f32[768, 768]"; primals_94: "f32[768]"; primals_95: "f32[3072, 768]"; primals_96: "f32[3072]"; primals_97: "f32[768, 3072]"; primals_98: "f32[768]"; primals_99: "f32[768, 768]"; primals_100: "f32[768]"; primals_101: "f32[768, 768]"; primals_102: "f32[768]"; primals_103: "f32[768, 768]"; primals_104: "f32[768]"; primals_105: "f32[768, 768]"; primals_106: "f32[768]"; primals_107: "f32[3072, 768]"; primals_108: "f32[3072]"; primals_109: "f32[768, 3072]"; primals_110: "f32[768]"; primals_111: "f32[768, 768]"; primals_112: "f32[768]"; primals_113: "f32[768, 768]"; primals_114: "f32[768]"; primals_115: "f32[768, 768]"; primals_116: "f32[768]"; primals_117: "f32[768, 768]"; primals_118: "f32[768]"; primals_119: "f32[3072, 768]"; primals_120: "f32[3072]"; primals_121: "f32[768, 3072]"; primals_122: "f32[768]"; primals_123: "f32[768, 768]"; primals_124: "f32[768]"; primals_125: "f32[768, 768]"; primals_126: "f32[768]"; primals_127: "f32[768, 768]"; primals_128: "f32[768]"; primals_129: "f32[768, 768]"; primals_130: "f32[768]"; primals_131: "f32[3072, 768]"; primals_132: "f32[3072]"; primals_133: "f32[768, 3072]"; primals_134: "f32[768]"; primals_135: "f32[768, 768]"; primals_136: "f32[768]"; primals_137: "f32[768, 768]"; primals_138: "f32[768]"; primals_139: "f32[768, 768]"; primals_140: "f32[768]"; primals_141: "f32[768, 768]"; primals_142: "f32[768]"; primals_143: "f32[3072, 768]"; primals_144: "f32[3072]"; primals_145: "f32[768, 3072]"; primals_146: "f32[768]"; primals_147: "f32[768, 768]"; primals_148: "f32[768]"; primals_149: "f32[768, 768]"; primals_150: "f32[768]"; primals_151: "f32[768, 768]"; primals_152: "f32[768]"; primals_153: "f32[768, 768]"; primals_154: "f32[768]"; primals_155: "f32[3072, 768]"; primals_156: "f32[3072]"; primals_157: "f32[768, 3072]"; primals_158: "f32[768]"; primals_159: "f32[768, 768]"; primals_160: "f32[768]"; primals_161: "f32[768, 768]"; primals_162: "f32[768]"; primals_163: "f32[768, 768]"; primals_164: "f32[768]"; primals_165: "f32[768, 768]"; primals_166: "f32[768]"; primals_167: "f32[3072, 768]"; primals_168: "f32[3072]"; primals_169: "f32[768, 3072]"; primals_170: "f32[768]"; primals_171: "f32[768, 768]"; primals_172: "f32[768]"; primals_173: "f32[768, 768]"; primals_174: "f32[768]"; primals_175: "f32[768, 768]"; primals_176: "f32[768]"; primals_177: "f32[768, 768]"; primals_178: "f32[768]"; primals_179: "f32[3072, 768]"; primals_180: "f32[3072]"; primals_181: "f32[768, 3072]"; primals_182: "f32[768]"; primals_183: "f32[768, 768]"; primals_184: "f32[768]"; primals_185: "f32[768, 768]"; primals_186: "f32[768]"; primals_187: "f32[768, 768]"; primals_188: "f32[768]"; primals_189: "f32[768, 768]"; primals_190: "f32[768]"; primals_191: "f32[3072, 768]"; primals_192: "f32[3072]"; primals_193: "f32[768, 3072]"; primals_194: "f32[768]"; primals_195: "f32[1, 512, 768]"; primals_196: "i64[4, 128]"; primals_197: "i64[4, 128]"; tangents_1: "f32[4, 128, 768]"; 

    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, tangents_1, = fx_pytree.tree_flatten_spec([primals, tangents], self._in_spec)
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/bert.py:40, code: mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    gt: "b8[4, 128]" = torch.ops.aten.gt.Scalar(primals_196, 0)
    unsqueeze: "b8[4, 1, 128]" = torch.ops.aten.unsqueeze.default(gt, 1);  gt = None
    repeat: "b8[4, 128, 128]" = torch.ops.aten.repeat.default(unsqueeze, [1, 128, 1]);  unsqueeze = None
    unsqueeze_1: "b8[4, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(repeat, 1);  repeat = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/module.py:1520, code: return forward_call(*args, **kwargs)
    embedding: "f32[4, 128, 768]" = torch.ops.aten.embedding.default(primals_49, primals_196, 0);  primals_49 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/position.py:26, code: return self.pe[:, :x.size(1)]
    slice_1: "f32[1, 512, 768]" = torch.ops.aten.slice.Tensor(primals_195, 0, 0, 9223372036854775807);  primals_195 = None
    slice_2: "f32[1, 128, 768]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 128);  slice_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
    add: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(embedding, slice_2);  embedding = slice_2 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/module.py:1520, code: return forward_call(*args, **kwargs)
    embedding_1: "f32[4, 128, 768]" = torch.ops.aten.embedding.default(primals_50, primals_197, 0);  primals_50 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
    add_1: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add, embedding_1);  add = embedding_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:33, code: return self.dropout(x)
    clone: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone, [-1], correction = 1.0, keepdim = True)
    sqrt: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var);  var = None
    alias: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone, mean);  mean = None
    mul: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_1, sub)
    add_2: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt, 1e-06);  sqrt = None
    div: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul, add_2)
    add_3: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div, primals_2);  div = primals_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view: "f32[512, 768]" = torch.ops.aten.view.default(add_3, [512, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
    addmm: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_52, view, permute);  primals_52 = None
    view_1: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm, [4, 128, 768]);  addmm = None
    view_2: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_1, [4, -1, 12, 64]);  view_1 = None
    permute_1: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    view_3: "f32[512, 768]" = torch.ops.aten.view.default(add_3, [512, 768])
    permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
    addmm_1: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_54, view_3, permute_2);  primals_54 = None
    view_4: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_1, [4, 128, 768]);  addmm_1 = None
    view_5: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_4, [4, -1, 12, 64]);  view_4 = None
    permute_3: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    view_6: "f32[512, 768]" = torch.ops.aten.view.default(add_3, [512, 768]);  add_3 = None
    permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
    addmm_2: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_56, view_6, permute_4);  primals_56 = None
    view_7: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_2, [4, 128, 768]);  addmm_2 = None
    view_8: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_7, [4, -1, 12, 64]);  view_7 = None
    permute_5: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    permute_6: "f32[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_3, [0, 1, 3, 2]);  permute_3 = None
    expand: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_1, [4, 12, 128, 64]);  permute_1 = None
    clone_1: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_9: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_1, [48, 128, 64]);  clone_1 = None
    expand_1: "f32[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_6, [4, 12, 64, 128]);  permute_6 = None
    clone_2: "f32[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_10: "f32[48, 64, 128]" = torch.ops.aten.view.default(clone_2, [48, 64, 128]);  clone_2 = None
    bmm: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_9, view_10)
    view_11: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm, [4, 12, 128, 128]);  bmm = None
    div_1: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq, scalar_tensor, div_1);  scalar_tensor = div_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_2: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    alias_1: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(div_2)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_3: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_2: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_3, [4, 12, 128, 128]);  clone_3 = None
    view_12: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_2, [48, 128, 128]);  expand_2 = None
    expand_3: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_5, [4, 12, 128, 64]);  permute_5 = None
    clone_4: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_13: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_4, [48, 128, 64]);  clone_4 = None
    bmm_1: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_12, view_13)
    view_14: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_1, [4, 12, 128, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_7: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    clone_5: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_15: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_5, [4, -1, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_16: "f32[512, 768]" = torch.ops.aten.view.default(view_15, [512, 768]);  view_15 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
    addmm_3: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_58, view_16, permute_8);  primals_58 = None
    view_17: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_3, [4, 128, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_6: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_17);  view_17 = None
    add_4: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone, clone_6);  clone_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_1: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_4, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_1: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_4, [-1], correction = 1.0, keepdim = True)
    sqrt_1: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_1);  var_1 = None
    alias_2: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_1)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_2: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_4, mean_1);  mean_1 = None
    mul_1: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_3, sub_2)
    add_5: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_1, 1e-06);  sqrt_1 = None
    div_3: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_1, add_5)
    add_6: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_3, primals_4);  div_3 = primals_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_18: "f32[512, 768]" = torch.ops.aten.view.default(add_6, [512, 768]);  add_6 = None
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
    addmm_4: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_60, view_18, permute_9);  primals_60 = None
    view_19: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_4, [4, 128, 3072]);  addmm_4 = None
    mul_2: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_3: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
    add_7: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_4: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_2, add_7);  mul_2 = add_7 = None
    clone_7: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_4);  mul_4 = None
    view_20: "f32[512, 3072]" = torch.ops.aten.view.default(clone_7, [512, 3072]);  clone_7 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
    addmm_5: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_62, view_20, permute_10);  primals_62 = None
    view_21: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_5, [4, 128, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_8: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_21);  view_21 = None
    add_8: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_4, clone_8);  clone_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_9: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_8);  add_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_2: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_9, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_2: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_9, [-1], correction = 1.0, keepdim = True)
    sqrt_2: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_2);  var_2 = None
    alias_3: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_2)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_3: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_9, mean_2);  mean_2 = None
    mul_5: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_5, sub_3)
    add_9: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_2, 1e-06);  sqrt_2 = None
    div_4: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_5, add_9)
    add_10: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_4, primals_6);  div_4 = primals_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_22: "f32[512, 768]" = torch.ops.aten.view.default(add_10, [512, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_64, view_22, permute_11);  primals_64 = None
    view_23: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_6, [4, 128, 768]);  addmm_6 = None
    view_24: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_23, [4, -1, 12, 64]);  view_23 = None
    permute_12: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    view_25: "f32[512, 768]" = torch.ops.aten.view.default(add_10, [512, 768])
    permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
    addmm_7: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_66, view_25, permute_13);  primals_66 = None
    view_26: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_7, [4, 128, 768]);  addmm_7 = None
    view_27: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_26, [4, -1, 12, 64]);  view_26 = None
    permute_14: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    view_28: "f32[512, 768]" = torch.ops.aten.view.default(add_10, [512, 768]);  add_10 = None
    permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
    addmm_8: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_68, view_28, permute_15);  primals_68 = None
    view_29: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_8, [4, 128, 768]);  addmm_8 = None
    view_30: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_29, [4, -1, 12, 64]);  view_29 = None
    permute_16: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    permute_17: "f32[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_14, [0, 1, 3, 2]);  permute_14 = None
    expand_4: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_12, [4, 12, 128, 64]);  permute_12 = None
    clone_10: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
    view_31: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_10, [48, 128, 64]);  clone_10 = None
    expand_5: "f32[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_17, [4, 12, 64, 128]);  permute_17 = None
    clone_11: "f32[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
    view_32: "f32[48, 64, 128]" = torch.ops.aten.view.default(clone_11, [48, 64, 128]);  clone_11 = None
    bmm_2: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_31, view_32)
    view_33: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_2, [4, 12, 128, 128]);  bmm_2 = None
    div_5: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_33, 8.0);  view_33 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_1: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_1: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, scalar_tensor_1, div_5);  scalar_tensor_1 = div_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_1: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_6: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    alias_4: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(div_6)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_12: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_6: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_12, [4, 12, 128, 128]);  clone_12 = None
    view_34: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_6, [48, 128, 128]);  expand_6 = None
    expand_7: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_16, [4, 12, 128, 64]);  permute_16 = None
    clone_13: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_35: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_13, [48, 128, 64]);  clone_13 = None
    bmm_3: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_34, view_35)
    view_36: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_3, [4, 12, 128, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_18: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
    clone_14: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_37: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_14, [4, -1, 768]);  clone_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_38: "f32[512, 768]" = torch.ops.aten.view.default(view_37, [512, 768]);  view_37 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
    addmm_9: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_70, view_38, permute_19);  primals_70 = None
    view_39: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_9, [4, 128, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_15: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_39);  view_39 = None
    add_11: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_9, clone_15);  clone_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_3: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_11, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_3: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_11, [-1], correction = 1.0, keepdim = True)
    sqrt_3: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_3);  var_3 = None
    alias_5: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_3)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_5: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_11, mean_3);  mean_3 = None
    mul_6: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_7, sub_5)
    add_12: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_3, 1e-06);  sqrt_3 = None
    div_7: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_6, add_12)
    add_13: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_7, primals_8);  div_7 = primals_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_40: "f32[512, 768]" = torch.ops.aten.view.default(add_13, [512, 768]);  add_13 = None
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
    addmm_10: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_72, view_40, permute_20);  primals_72 = None
    view_41: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_10, [4, 128, 3072]);  addmm_10 = None
    mul_7: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_8: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_1: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_14: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_9: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_14);  mul_7 = add_14 = None
    clone_16: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_9);  mul_9 = None
    view_42: "f32[512, 3072]" = torch.ops.aten.view.default(clone_16, [512, 3072]);  clone_16 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_74, view_42, permute_21);  primals_74 = None
    view_43: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_11, [4, 128, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_17: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_43);  view_43 = None
    add_15: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_11, clone_17);  clone_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_18: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_15);  add_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_4: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_18, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_4: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_18, [-1], correction = 1.0, keepdim = True)
    sqrt_4: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_4);  var_4 = None
    alias_6: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_4)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_6: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_18, mean_4);  mean_4 = None
    mul_10: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_9, sub_6)
    add_16: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_4, 1e-06);  sqrt_4 = None
    div_8: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_10, add_16)
    add_17: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_8, primals_10);  div_8 = primals_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_44: "f32[512, 768]" = torch.ops.aten.view.default(add_17, [512, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
    addmm_12: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_76, view_44, permute_22);  primals_76 = None
    view_45: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_12, [4, 128, 768]);  addmm_12 = None
    view_46: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_45, [4, -1, 12, 64]);  view_45 = None
    permute_23: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
    view_47: "f32[512, 768]" = torch.ops.aten.view.default(add_17, [512, 768])
    permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
    addmm_13: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_78, view_47, permute_24);  primals_78 = None
    view_48: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_13, [4, 128, 768]);  addmm_13 = None
    view_49: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_48, [4, -1, 12, 64]);  view_48 = None
    permute_25: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    view_50: "f32[512, 768]" = torch.ops.aten.view.default(add_17, [512, 768]);  add_17 = None
    permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
    addmm_14: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_80, view_50, permute_26);  primals_80 = None
    view_51: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_14, [4, 128, 768]);  addmm_14 = None
    view_52: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_51, [4, -1, 12, 64]);  view_51 = None
    permute_27: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    permute_28: "f32[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
    expand_8: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_23, [4, 12, 128, 64]);  permute_23 = None
    clone_19: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
    view_53: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_19, [48, 128, 64]);  clone_19 = None
    expand_9: "f32[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_28, [4, 12, 64, 128]);  permute_28 = None
    clone_20: "f32[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
    view_54: "f32[48, 64, 128]" = torch.ops.aten.view.default(clone_20, [48, 64, 128]);  clone_20 = None
    bmm_4: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_53, view_54)
    view_55: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_4, [4, 12, 128, 128]);  bmm_4 = None
    div_9: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_55, 8.0);  view_55 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_2: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_2: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_2, scalar_tensor_2, div_9);  scalar_tensor_2 = div_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_2: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_10: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    alias_7: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(div_10)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_21: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_10: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_21, [4, 12, 128, 128]);  clone_21 = None
    view_56: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_10, [48, 128, 128]);  expand_10 = None
    expand_11: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_27, [4, 12, 128, 64]);  permute_27 = None
    clone_22: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_57: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_22, [48, 128, 64]);  clone_22 = None
    bmm_5: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_56, view_57)
    view_58: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_5, [4, 12, 128, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_29: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    clone_23: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_59: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_23, [4, -1, 768]);  clone_23 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_60: "f32[512, 768]" = torch.ops.aten.view.default(view_59, [512, 768]);  view_59 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
    addmm_15: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_82, view_60, permute_30);  primals_82 = None
    view_61: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_15, [4, 128, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_24: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_61);  view_61 = None
    add_18: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_18, clone_24);  clone_24 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_5: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_18, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_5: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_18, [-1], correction = 1.0, keepdim = True)
    sqrt_5: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_5);  var_5 = None
    alias_8: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_5)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_8: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_18, mean_5);  mean_5 = None
    mul_11: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_11, sub_8)
    add_19: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_5, 1e-06);  sqrt_5 = None
    div_11: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_11, add_19)
    add_20: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_11, primals_12);  div_11 = primals_12 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_62: "f32[512, 768]" = torch.ops.aten.view.default(add_20, [512, 768]);  add_20 = None
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
    addmm_16: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_84, view_62, permute_31);  primals_84 = None
    view_63: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_16, [4, 128, 3072]);  addmm_16 = None
    mul_12: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_13: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_2: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_21: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_14: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_21);  mul_12 = add_21 = None
    clone_25: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_14);  mul_14 = None
    view_64: "f32[512, 3072]" = torch.ops.aten.view.default(clone_25, [512, 3072]);  clone_25 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
    addmm_17: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_86, view_64, permute_32);  primals_86 = None
    view_65: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_17, [4, 128, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_26: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_65);  view_65 = None
    add_22: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_18, clone_26);  clone_26 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_27: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_22);  add_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_6: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_27, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_6: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_27, [-1], correction = 1.0, keepdim = True)
    sqrt_6: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_6);  var_6 = None
    alias_9: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_6)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_9: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_27, mean_6);  mean_6 = None
    mul_15: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_13, sub_9)
    add_23: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_6, 1e-06);  sqrt_6 = None
    div_12: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_15, add_23)
    add_24: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_12, primals_14);  div_12 = primals_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_66: "f32[512, 768]" = torch.ops.aten.view.default(add_24, [512, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_88, view_66, permute_33);  primals_88 = None
    view_67: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_18, [4, 128, 768]);  addmm_18 = None
    view_68: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_67, [4, -1, 12, 64]);  view_67 = None
    permute_34: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    view_69: "f32[512, 768]" = torch.ops.aten.view.default(add_24, [512, 768])
    permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
    addmm_19: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_90, view_69, permute_35);  primals_90 = None
    view_70: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_19, [4, 128, 768]);  addmm_19 = None
    view_71: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_70, [4, -1, 12, 64]);  view_70 = None
    permute_36: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    view_72: "f32[512, 768]" = torch.ops.aten.view.default(add_24, [512, 768]);  add_24 = None
    permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_92, view_72, permute_37);  primals_92 = None
    view_73: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_20, [4, 128, 768]);  addmm_20 = None
    view_74: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_73, [4, -1, 12, 64]);  view_73 = None
    permute_38: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_74, [0, 2, 1, 3]);  view_74 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    permute_39: "f32[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_36, [0, 1, 3, 2]);  permute_36 = None
    expand_12: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_34, [4, 12, 128, 64]);  permute_34 = None
    clone_28: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
    view_75: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_28, [48, 128, 64]);  clone_28 = None
    expand_13: "f32[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_39, [4, 12, 64, 128]);  permute_39 = None
    clone_29: "f32[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
    view_76: "f32[48, 64, 128]" = torch.ops.aten.view.default(clone_29, [48, 64, 128]);  clone_29 = None
    bmm_6: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_75, view_76)
    view_77: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_6, [4, 12, 128, 128]);  bmm_6 = None
    div_13: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_77, 8.0);  view_77 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_3: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_3: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_3, scalar_tensor_3, div_13);  scalar_tensor_3 = div_13 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_3: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_14: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    alias_10: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(div_14)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_30: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_14: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_30, [4, 12, 128, 128]);  clone_30 = None
    view_78: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_14, [48, 128, 128]);  expand_14 = None
    expand_15: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_38, [4, 12, 128, 64]);  permute_38 = None
    clone_31: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_79: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_31, [48, 128, 64]);  clone_31 = None
    bmm_7: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_78, view_79)
    view_80: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_7, [4, 12, 128, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_40: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    clone_32: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_81: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_32, [4, -1, 768]);  clone_32 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_82: "f32[512, 768]" = torch.ops.aten.view.default(view_81, [512, 768]);  view_81 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
    addmm_21: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_94, view_82, permute_41);  primals_94 = None
    view_83: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_21, [4, 128, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_33: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_83);  view_83 = None
    add_25: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_27, clone_33);  clone_33 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_7: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_25, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_7: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_25, [-1], correction = 1.0, keepdim = True)
    sqrt_7: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_7);  var_7 = None
    alias_11: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_7)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_11: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_25, mean_7);  mean_7 = None
    mul_16: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_15, sub_11)
    add_26: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_7, 1e-06);  sqrt_7 = None
    div_15: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_16, add_26)
    add_27: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_15, primals_16);  div_15 = primals_16 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_84: "f32[512, 768]" = torch.ops.aten.view.default(add_27, [512, 768]);  add_27 = None
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
    addmm_22: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_96, view_84, permute_42);  primals_96 = None
    view_85: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_22, [4, 128, 3072]);  addmm_22 = None
    mul_17: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_18: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_3: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_28: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_19: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_17, add_28);  mul_17 = add_28 = None
    clone_34: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_19);  mul_19 = None
    view_86: "f32[512, 3072]" = torch.ops.aten.view.default(clone_34, [512, 3072]);  clone_34 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
    addmm_23: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_98, view_86, permute_43);  primals_98 = None
    view_87: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_23, [4, 128, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_35: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_87);  view_87 = None
    add_29: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_25, clone_35);  clone_35 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_36: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_8: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_36, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_8: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_36, [-1], correction = 1.0, keepdim = True)
    sqrt_8: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_8);  var_8 = None
    alias_12: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_8)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_12: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_36, mean_8);  mean_8 = None
    mul_20: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_17, sub_12)
    add_30: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_8, 1e-06);  sqrt_8 = None
    div_16: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_20, add_30)
    add_31: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_16, primals_18);  div_16 = primals_18 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_88: "f32[512, 768]" = torch.ops.aten.view.default(add_31, [512, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
    addmm_24: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_100, view_88, permute_44);  primals_100 = None
    view_89: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_24, [4, 128, 768]);  addmm_24 = None
    view_90: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_89, [4, -1, 12, 64]);  view_89 = None
    permute_45: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
    view_91: "f32[512, 768]" = torch.ops.aten.view.default(add_31, [512, 768])
    permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
    addmm_25: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_102, view_91, permute_46);  primals_102 = None
    view_92: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_25, [4, 128, 768]);  addmm_25 = None
    view_93: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_92, [4, -1, 12, 64]);  view_92 = None
    permute_47: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    view_94: "f32[512, 768]" = torch.ops.aten.view.default(add_31, [512, 768]);  add_31 = None
    permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
    addmm_26: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_104, view_94, permute_48);  primals_104 = None
    view_95: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_26, [4, 128, 768]);  addmm_26 = None
    view_96: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_95, [4, -1, 12, 64]);  view_95 = None
    permute_49: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    permute_50: "f32[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_47, [0, 1, 3, 2]);  permute_47 = None
    expand_16: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_45, [4, 12, 128, 64]);  permute_45 = None
    clone_37: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
    view_97: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_37, [48, 128, 64]);  clone_37 = None
    expand_17: "f32[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_50, [4, 12, 64, 128]);  permute_50 = None
    clone_38: "f32[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
    view_98: "f32[48, 64, 128]" = torch.ops.aten.view.default(clone_38, [48, 64, 128]);  clone_38 = None
    bmm_8: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_97, view_98)
    view_99: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_8, [4, 12, 128, 128]);  bmm_8 = None
    div_17: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_99, 8.0);  view_99 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_4: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_4: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_4, scalar_tensor_4, div_17);  scalar_tensor_4 = div_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_4: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_18: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    alias_13: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(div_18)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_39: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_18);  div_18 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_18: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_39, [4, 12, 128, 128]);  clone_39 = None
    view_100: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_18, [48, 128, 128]);  expand_18 = None
    expand_19: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_49, [4, 12, 128, 64]);  permute_49 = None
    clone_40: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_101: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_40, [48, 128, 64]);  clone_40 = None
    bmm_9: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_100, view_101)
    view_102: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_9, [4, 12, 128, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_51: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_41: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_103: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_41, [4, -1, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_104: "f32[512, 768]" = torch.ops.aten.view.default(view_103, [512, 768]);  view_103 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
    addmm_27: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_106, view_104, permute_52);  primals_106 = None
    view_105: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_27, [4, 128, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_42: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_105);  view_105 = None
    add_32: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_36, clone_42);  clone_42 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_9: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_32, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_9: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_32, [-1], correction = 1.0, keepdim = True)
    sqrt_9: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_9);  var_9 = None
    alias_14: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_9)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_14: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_32, mean_9);  mean_9 = None
    mul_21: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_19, sub_14)
    add_33: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_9, 1e-06);  sqrt_9 = None
    div_19: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_21, add_33)
    add_34: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_19, primals_20);  div_19 = primals_20 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_106: "f32[512, 768]" = torch.ops.aten.view.default(add_34, [512, 768]);  add_34 = None
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
    addmm_28: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_108, view_106, permute_53);  primals_108 = None
    view_107: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_28, [4, 128, 3072]);  addmm_28 = None
    mul_22: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_23: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_4: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_35: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_24: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_22, add_35);  mul_22 = add_35 = None
    clone_43: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_24);  mul_24 = None
    view_108: "f32[512, 3072]" = torch.ops.aten.view.default(clone_43, [512, 3072]);  clone_43 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
    addmm_29: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_110, view_108, permute_54);  primals_110 = None
    view_109: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_29, [4, 128, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_44: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_109);  view_109 = None
    add_36: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_32, clone_44);  clone_44 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_45: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_36);  add_36 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_10: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_45, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_10: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_45, [-1], correction = 1.0, keepdim = True)
    sqrt_10: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_10);  var_10 = None
    alias_15: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_10)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_15: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_45, mean_10);  mean_10 = None
    mul_25: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_21, sub_15)
    add_37: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_10, 1e-06);  sqrt_10 = None
    div_20: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_25, add_37)
    add_38: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_20, primals_22);  div_20 = primals_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_110: "f32[512, 768]" = torch.ops.aten.view.default(add_38, [512, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
    addmm_30: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_112, view_110, permute_55);  primals_112 = None
    view_111: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_30, [4, 128, 768]);  addmm_30 = None
    view_112: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_111, [4, -1, 12, 64]);  view_111 = None
    permute_56: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
    view_113: "f32[512, 768]" = torch.ops.aten.view.default(add_38, [512, 768])
    permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
    addmm_31: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_114, view_113, permute_57);  primals_114 = None
    view_114: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_31, [4, 128, 768]);  addmm_31 = None
    view_115: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_114, [4, -1, 12, 64]);  view_114 = None
    permute_58: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
    view_116: "f32[512, 768]" = torch.ops.aten.view.default(add_38, [512, 768]);  add_38 = None
    permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
    addmm_32: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_116, view_116, permute_59);  primals_116 = None
    view_117: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_32, [4, 128, 768]);  addmm_32 = None
    view_118: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_117, [4, -1, 12, 64]);  view_117 = None
    permute_60: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    permute_61: "f32[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_58, [0, 1, 3, 2]);  permute_58 = None
    expand_20: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_56, [4, 12, 128, 64]);  permute_56 = None
    clone_46: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
    view_119: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_46, [48, 128, 64]);  clone_46 = None
    expand_21: "f32[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_61, [4, 12, 64, 128]);  permute_61 = None
    clone_47: "f32[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
    view_120: "f32[48, 64, 128]" = torch.ops.aten.view.default(clone_47, [48, 64, 128]);  clone_47 = None
    bmm_10: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_119, view_120)
    view_121: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_10, [4, 12, 128, 128]);  bmm_10 = None
    div_21: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_121, 8.0);  view_121 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_5: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_5: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_5, scalar_tensor_5, div_21);  scalar_tensor_5 = div_21 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_5: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_22: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    alias_16: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(div_22)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_48: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_22);  div_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_22: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_48, [4, 12, 128, 128]);  clone_48 = None
    view_122: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_22, [48, 128, 128]);  expand_22 = None
    expand_23: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_60, [4, 12, 128, 64]);  permute_60 = None
    clone_49: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_123: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_49, [48, 128, 64]);  clone_49 = None
    bmm_11: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_122, view_123)
    view_124: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_11, [4, 12, 128, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_62: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
    clone_50: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_125: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_50, [4, -1, 768]);  clone_50 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_126: "f32[512, 768]" = torch.ops.aten.view.default(view_125, [512, 768]);  view_125 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
    addmm_33: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_118, view_126, permute_63);  primals_118 = None
    view_127: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_33, [4, 128, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_51: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_127);  view_127 = None
    add_39: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_45, clone_51);  clone_51 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_11: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_39, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_11: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_39, [-1], correction = 1.0, keepdim = True)
    sqrt_11: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_11);  var_11 = None
    alias_17: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_11)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_17: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_39, mean_11);  mean_11 = None
    mul_26: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_23, sub_17)
    add_40: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_11, 1e-06);  sqrt_11 = None
    div_23: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_26, add_40)
    add_41: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_23, primals_24);  div_23 = primals_24 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_128: "f32[512, 768]" = torch.ops.aten.view.default(add_41, [512, 768]);  add_41 = None
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
    addmm_34: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_120, view_128, permute_64);  primals_120 = None
    view_129: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_34, [4, 128, 3072]);  addmm_34 = None
    mul_27: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
    mul_28: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_5: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_42: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_29: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_42);  mul_27 = add_42 = None
    clone_52: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_29);  mul_29 = None
    view_130: "f32[512, 3072]" = torch.ops.aten.view.default(clone_52, [512, 3072]);  clone_52 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
    addmm_35: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_122, view_130, permute_65);  primals_122 = None
    view_131: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_35, [4, 128, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_53: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_131);  view_131 = None
    add_43: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_39, clone_53);  clone_53 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_54: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_43);  add_43 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_12: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_54, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_12: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_54, [-1], correction = 1.0, keepdim = True)
    sqrt_12: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_12);  var_12 = None
    alias_18: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_12)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_18: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_54, mean_12);  mean_12 = None
    mul_30: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_25, sub_18)
    add_44: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_12, 1e-06);  sqrt_12 = None
    div_24: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_30, add_44)
    add_45: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_24, primals_26);  div_24 = primals_26 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_132: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
    addmm_36: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_124, view_132, permute_66);  primals_124 = None
    view_133: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_36, [4, 128, 768]);  addmm_36 = None
    view_134: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_133, [4, -1, 12, 64]);  view_133 = None
    permute_67: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
    view_135: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768])
    permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
    addmm_37: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_126, view_135, permute_68);  primals_126 = None
    view_136: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_37, [4, 128, 768]);  addmm_37 = None
    view_137: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_136, [4, -1, 12, 64]);  view_136 = None
    permute_69: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
    view_138: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768]);  add_45 = None
    permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
    addmm_38: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_128, view_138, permute_70);  primals_128 = None
    view_139: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_38, [4, 128, 768]);  addmm_38 = None
    view_140: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_139, [4, -1, 12, 64]);  view_139 = None
    permute_71: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_140, [0, 2, 1, 3]);  view_140 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    permute_72: "f32[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_69, [0, 1, 3, 2]);  permute_69 = None
    expand_24: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_67, [4, 12, 128, 64]);  permute_67 = None
    clone_55: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
    view_141: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_55, [48, 128, 64]);  clone_55 = None
    expand_25: "f32[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_72, [4, 12, 64, 128]);  permute_72 = None
    clone_56: "f32[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
    view_142: "f32[48, 64, 128]" = torch.ops.aten.view.default(clone_56, [48, 64, 128]);  clone_56 = None
    bmm_12: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_141, view_142)
    view_143: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_12, [4, 12, 128, 128]);  bmm_12 = None
    div_25: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_143, 8.0);  view_143 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_6: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_6: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_6, scalar_tensor_6, div_25);  scalar_tensor_6 = div_25 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_6: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_19: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_6, amax_6);  where_6 = amax_6 = None
    exp_6: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_26: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    alias_19: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(div_26)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_57: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_26);  div_26 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_26: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_57, [4, 12, 128, 128]);  clone_57 = None
    view_144: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_26, [48, 128, 128]);  expand_26 = None
    expand_27: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_71, [4, 12, 128, 64]);  permute_71 = None
    clone_58: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_145: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_58, [48, 128, 64]);  clone_58 = None
    bmm_13: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_144, view_145)
    view_146: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_13, [4, 12, 128, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_73: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    clone_59: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_147: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_59, [4, -1, 768]);  clone_59 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_148: "f32[512, 768]" = torch.ops.aten.view.default(view_147, [512, 768]);  view_147 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
    addmm_39: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_130, view_148, permute_74);  primals_130 = None
    view_149: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_39, [4, 128, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_60: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_149);  view_149 = None
    add_46: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_54, clone_60);  clone_60 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_13: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_46, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_13: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_46, [-1], correction = 1.0, keepdim = True)
    sqrt_13: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_13);  var_13 = None
    alias_20: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_13)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_20: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_46, mean_13);  mean_13 = None
    mul_31: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_27, sub_20)
    add_47: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_13, 1e-06);  sqrt_13 = None
    div_27: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_31, add_47)
    add_48: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_27, primals_28);  div_27 = primals_28 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_150: "f32[512, 768]" = torch.ops.aten.view.default(add_48, [512, 768]);  add_48 = None
    permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
    addmm_40: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_132, view_150, permute_75);  primals_132 = None
    view_151: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_40, [4, 128, 3072]);  addmm_40 = None
    mul_32: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_33: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_6: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_49: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_34: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_49);  mul_32 = add_49 = None
    clone_61: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_34);  mul_34 = None
    view_152: "f32[512, 3072]" = torch.ops.aten.view.default(clone_61, [512, 3072]);  clone_61 = None
    permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
    addmm_41: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_134, view_152, permute_76);  primals_134 = None
    view_153: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_41, [4, 128, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_62: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_153);  view_153 = None
    add_50: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_46, clone_62);  clone_62 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_63: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_50);  add_50 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_14: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_63, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_14: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_63, [-1], correction = 1.0, keepdim = True)
    sqrt_14: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_14);  var_14 = None
    alias_21: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_14)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_21: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_63, mean_14);  mean_14 = None
    mul_35: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_29, sub_21)
    add_51: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_14, 1e-06);  sqrt_14 = None
    div_28: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_35, add_51)
    add_52: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_28, primals_30);  div_28 = primals_30 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_154: "f32[512, 768]" = torch.ops.aten.view.default(add_52, [512, 768])
    permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
    addmm_42: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_136, view_154, permute_77);  primals_136 = None
    view_155: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_42, [4, 128, 768]);  addmm_42 = None
    view_156: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_155, [4, -1, 12, 64]);  view_155 = None
    permute_78: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    view_157: "f32[512, 768]" = torch.ops.aten.view.default(add_52, [512, 768])
    permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
    addmm_43: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_138, view_157, permute_79);  primals_138 = None
    view_158: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_43, [4, 128, 768]);  addmm_43 = None
    view_159: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_158, [4, -1, 12, 64]);  view_158 = None
    permute_80: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
    view_160: "f32[512, 768]" = torch.ops.aten.view.default(add_52, [512, 768]);  add_52 = None
    permute_81: "f32[768, 768]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
    addmm_44: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_140, view_160, permute_81);  primals_140 = None
    view_161: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_44, [4, 128, 768]);  addmm_44 = None
    view_162: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_161, [4, -1, 12, 64]);  view_161 = None
    permute_82: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_162, [0, 2, 1, 3]);  view_162 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    permute_83: "f32[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 2]);  permute_80 = None
    expand_28: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_78, [4, 12, 128, 64]);  permute_78 = None
    clone_64: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
    view_163: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_64, [48, 128, 64]);  clone_64 = None
    expand_29: "f32[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_83, [4, 12, 64, 128]);  permute_83 = None
    clone_65: "f32[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
    view_164: "f32[48, 64, 128]" = torch.ops.aten.view.default(clone_65, [48, 64, 128]);  clone_65 = None
    bmm_14: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_163, view_164)
    view_165: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_14, [4, 12, 128, 128]);  bmm_14 = None
    div_29: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_165, 8.0);  view_165 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_7: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_7: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_7, scalar_tensor_7, div_29);  scalar_tensor_7 = div_29 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_7: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_7, [-1], True)
    sub_22: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_7, amax_7);  where_7 = amax_7 = None
    exp_7: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_30: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    alias_22: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(div_30)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_66: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_30);  div_30 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_30: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_66, [4, 12, 128, 128]);  clone_66 = None
    view_166: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_30, [48, 128, 128]);  expand_30 = None
    expand_31: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_82, [4, 12, 128, 64]);  permute_82 = None
    clone_67: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_167: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_67, [48, 128, 64]);  clone_67 = None
    bmm_15: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_166, view_167)
    view_168: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_15, [4, 12, 128, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_84: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    clone_68: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_169: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_68, [4, -1, 768]);  clone_68 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_170: "f32[512, 768]" = torch.ops.aten.view.default(view_169, [512, 768]);  view_169 = None
    permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
    addmm_45: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_142, view_170, permute_85);  primals_142 = None
    view_171: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_45, [4, 128, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_69: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_171);  view_171 = None
    add_53: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_63, clone_69);  clone_69 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_15: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_53, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_15: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_53, [-1], correction = 1.0, keepdim = True)
    sqrt_15: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_15);  var_15 = None
    alias_23: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_15)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_23: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_53, mean_15);  mean_15 = None
    mul_36: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_31, sub_23)
    add_54: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_15, 1e-06);  sqrt_15 = None
    div_31: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_36, add_54)
    add_55: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_31, primals_32);  div_31 = primals_32 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_172: "f32[512, 768]" = torch.ops.aten.view.default(add_55, [512, 768]);  add_55 = None
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
    addmm_46: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_144, view_172, permute_86);  primals_144 = None
    view_173: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_46, [4, 128, 3072]);  addmm_46 = None
    mul_37: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
    mul_38: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_7: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_56: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_39: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_37, add_56);  mul_37 = add_56 = None
    clone_70: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_39);  mul_39 = None
    view_174: "f32[512, 3072]" = torch.ops.aten.view.default(clone_70, [512, 3072]);  clone_70 = None
    permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
    addmm_47: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_146, view_174, permute_87);  primals_146 = None
    view_175: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_47, [4, 128, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_71: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_175);  view_175 = None
    add_57: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_53, clone_71);  clone_71 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_72: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_16: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_72, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_16: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_72, [-1], correction = 1.0, keepdim = True)
    sqrt_16: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_16);  var_16 = None
    alias_24: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_16)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_24: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_72, mean_16);  mean_16 = None
    mul_40: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_33, sub_24)
    add_58: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_16, 1e-06);  sqrt_16 = None
    div_32: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_40, add_58)
    add_59: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_32, primals_34);  div_32 = primals_34 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_176: "f32[512, 768]" = torch.ops.aten.view.default(add_59, [512, 768])
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
    addmm_48: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_148, view_176, permute_88);  primals_148 = None
    view_177: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_48, [4, 128, 768]);  addmm_48 = None
    view_178: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_177, [4, -1, 12, 64]);  view_177 = None
    permute_89: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
    view_179: "f32[512, 768]" = torch.ops.aten.view.default(add_59, [512, 768])
    permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
    addmm_49: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_150, view_179, permute_90);  primals_150 = None
    view_180: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_49, [4, 128, 768]);  addmm_49 = None
    view_181: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_180, [4, -1, 12, 64]);  view_180 = None
    permute_91: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
    view_182: "f32[512, 768]" = torch.ops.aten.view.default(add_59, [512, 768]);  add_59 = None
    permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
    addmm_50: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_152, view_182, permute_92);  primals_152 = None
    view_183: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_50, [4, 128, 768]);  addmm_50 = None
    view_184: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_183, [4, -1, 12, 64]);  view_183 = None
    permute_93: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_184, [0, 2, 1, 3]);  view_184 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    permute_94: "f32[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
    expand_32: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_89, [4, 12, 128, 64]);  permute_89 = None
    clone_73: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
    view_185: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_73, [48, 128, 64]);  clone_73 = None
    expand_33: "f32[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_94, [4, 12, 64, 128]);  permute_94 = None
    clone_74: "f32[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
    view_186: "f32[48, 64, 128]" = torch.ops.aten.view.default(clone_74, [48, 64, 128]);  clone_74 = None
    bmm_16: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_185, view_186)
    view_187: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_16, [4, 12, 128, 128]);  bmm_16 = None
    div_33: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_187, 8.0);  view_187 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_8: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_8: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_8, scalar_tensor_8, div_33);  scalar_tensor_8 = div_33 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_8: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
    sub_25: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_8, amax_8);  where_8 = amax_8 = None
    exp_8: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_34: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    alias_25: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(div_34)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_75: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_34);  div_34 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_34: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_75, [4, 12, 128, 128]);  clone_75 = None
    view_188: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_34, [48, 128, 128]);  expand_34 = None
    expand_35: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_93, [4, 12, 128, 64]);  permute_93 = None
    clone_76: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_189: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_76, [48, 128, 64]);  clone_76 = None
    bmm_17: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_188, view_189)
    view_190: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_17, [4, 12, 128, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_95: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_77: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_191: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_77, [4, -1, 768]);  clone_77 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_192: "f32[512, 768]" = torch.ops.aten.view.default(view_191, [512, 768]);  view_191 = None
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
    addmm_51: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_154, view_192, permute_96);  primals_154 = None
    view_193: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_51, [4, 128, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_78: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_193);  view_193 = None
    add_60: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_72, clone_78);  clone_78 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_17: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_60, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_17: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_60, [-1], correction = 1.0, keepdim = True)
    sqrt_17: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_17);  var_17 = None
    alias_26: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_17)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_26: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_60, mean_17);  mean_17 = None
    mul_41: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_35, sub_26)
    add_61: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_17, 1e-06);  sqrt_17 = None
    div_35: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_41, add_61)
    add_62: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_35, primals_36);  div_35 = primals_36 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_194: "f32[512, 768]" = torch.ops.aten.view.default(add_62, [512, 768]);  add_62 = None
    permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
    addmm_52: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_156, view_194, permute_97);  primals_156 = None
    view_195: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_52, [4, 128, 3072]);  addmm_52 = None
    mul_42: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_43: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_8: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_63: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_44: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_42, add_63);  mul_42 = add_63 = None
    clone_79: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_44);  mul_44 = None
    view_196: "f32[512, 3072]" = torch.ops.aten.view.default(clone_79, [512, 3072]);  clone_79 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
    addmm_53: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_158, view_196, permute_98);  primals_158 = None
    view_197: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_53, [4, 128, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_80: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_197);  view_197 = None
    add_64: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_60, clone_80);  clone_80 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_81: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_64);  add_64 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_18: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_81, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_18: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_81, [-1], correction = 1.0, keepdim = True)
    sqrt_18: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_18);  var_18 = None
    alias_27: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_18)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_27: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_81, mean_18);  mean_18 = None
    mul_45: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_37, sub_27)
    add_65: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_18, 1e-06);  sqrt_18 = None
    div_36: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_45, add_65)
    add_66: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_36, primals_38);  div_36 = primals_38 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_198: "f32[512, 768]" = torch.ops.aten.view.default(add_66, [512, 768])
    permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
    addmm_54: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_160, view_198, permute_99);  primals_160 = None
    view_199: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_54, [4, 128, 768]);  addmm_54 = None
    view_200: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_199, [4, -1, 12, 64]);  view_199 = None
    permute_100: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
    view_201: "f32[512, 768]" = torch.ops.aten.view.default(add_66, [512, 768])
    permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
    addmm_55: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_162, view_201, permute_101);  primals_162 = None
    view_202: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_55, [4, 128, 768]);  addmm_55 = None
    view_203: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_202, [4, -1, 12, 64]);  view_202 = None
    permute_102: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    view_204: "f32[512, 768]" = torch.ops.aten.view.default(add_66, [512, 768]);  add_66 = None
    permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
    addmm_56: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_164, view_204, permute_103);  primals_164 = None
    view_205: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_56, [4, 128, 768]);  addmm_56 = None
    view_206: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_205, [4, -1, 12, 64]);  view_205 = None
    permute_104: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    permute_105: "f32[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_102, [0, 1, 3, 2]);  permute_102 = None
    expand_36: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_100, [4, 12, 128, 64]);  permute_100 = None
    clone_82: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
    view_207: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_82, [48, 128, 64]);  clone_82 = None
    expand_37: "f32[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_105, [4, 12, 64, 128]);  permute_105 = None
    clone_83: "f32[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
    view_208: "f32[48, 64, 128]" = torch.ops.aten.view.default(clone_83, [48, 64, 128]);  clone_83 = None
    bmm_18: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_207, view_208)
    view_209: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_18, [4, 12, 128, 128]);  bmm_18 = None
    div_37: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_209, 8.0);  view_209 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_9: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_9: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_9, scalar_tensor_9, div_37);  scalar_tensor_9 = div_37 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_9: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_9, [-1], True)
    sub_28: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_9, amax_9);  where_9 = amax_9 = None
    exp_9: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_38: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    alias_28: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(div_38)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_84: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_38);  div_38 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_38: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_84, [4, 12, 128, 128]);  clone_84 = None
    view_210: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_38, [48, 128, 128]);  expand_38 = None
    expand_39: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_104, [4, 12, 128, 64]);  permute_104 = None
    clone_85: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_211: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_85, [48, 128, 64]);  clone_85 = None
    bmm_19: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_210, view_211)
    view_212: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_19, [4, 12, 128, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_106: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    clone_86: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    view_213: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_86, [4, -1, 768]);  clone_86 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_214: "f32[512, 768]" = torch.ops.aten.view.default(view_213, [512, 768]);  view_213 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
    addmm_57: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_166, view_214, permute_107);  primals_166 = None
    view_215: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_57, [4, 128, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_87: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_215);  view_215 = None
    add_67: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_81, clone_87);  clone_87 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_19: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_67, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_19: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_67, [-1], correction = 1.0, keepdim = True)
    sqrt_19: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_19);  var_19 = None
    alias_29: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_19)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_29: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_67, mean_19);  mean_19 = None
    mul_46: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_39, sub_29)
    add_68: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_19, 1e-06);  sqrt_19 = None
    div_39: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_46, add_68)
    add_69: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_39, primals_40);  div_39 = primals_40 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_216: "f32[512, 768]" = torch.ops.aten.view.default(add_69, [512, 768]);  add_69 = None
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
    addmm_58: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_168, view_216, permute_108);  primals_168 = None
    view_217: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_58, [4, 128, 3072]);  addmm_58 = None
    mul_47: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_48: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_9: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_70: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_49: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_70);  mul_47 = add_70 = None
    clone_88: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_49);  mul_49 = None
    view_218: "f32[512, 3072]" = torch.ops.aten.view.default(clone_88, [512, 3072]);  clone_88 = None
    permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
    addmm_59: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_170, view_218, permute_109);  primals_170 = None
    view_219: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_59, [4, 128, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_89: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_219);  view_219 = None
    add_71: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_67, clone_89);  clone_89 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_90: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_71);  add_71 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_20: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_90, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_20: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_90, [-1], correction = 1.0, keepdim = True)
    sqrt_20: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_20);  var_20 = None
    alias_30: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_20)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_30: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_90, mean_20);  mean_20 = None
    mul_50: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_41, sub_30)
    add_72: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_20, 1e-06);  sqrt_20 = None
    div_40: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_50, add_72)
    add_73: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_40, primals_42);  div_40 = primals_42 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_220: "f32[512, 768]" = torch.ops.aten.view.default(add_73, [512, 768])
    permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
    addmm_60: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_172, view_220, permute_110);  primals_172 = None
    view_221: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_60, [4, 128, 768]);  addmm_60 = None
    view_222: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_221, [4, -1, 12, 64]);  view_221 = None
    permute_111: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    view_223: "f32[512, 768]" = torch.ops.aten.view.default(add_73, [512, 768])
    permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
    addmm_61: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_174, view_223, permute_112);  primals_174 = None
    view_224: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_61, [4, 128, 768]);  addmm_61 = None
    view_225: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_224, [4, -1, 12, 64]);  view_224 = None
    permute_113: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
    view_226: "f32[512, 768]" = torch.ops.aten.view.default(add_73, [512, 768]);  add_73 = None
    permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
    addmm_62: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_176, view_226, permute_114);  primals_176 = None
    view_227: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_62, [4, 128, 768]);  addmm_62 = None
    view_228: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_227, [4, -1, 12, 64]);  view_227 = None
    permute_115: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    permute_116: "f32[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_113, [0, 1, 3, 2]);  permute_113 = None
    expand_40: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_111, [4, 12, 128, 64]);  permute_111 = None
    clone_91: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
    view_229: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_91, [48, 128, 64]);  clone_91 = None
    expand_41: "f32[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_116, [4, 12, 64, 128]);  permute_116 = None
    clone_92: "f32[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
    view_230: "f32[48, 64, 128]" = torch.ops.aten.view.default(clone_92, [48, 64, 128]);  clone_92 = None
    bmm_20: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_229, view_230)
    view_231: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_20, [4, 12, 128, 128]);  bmm_20 = None
    div_41: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_231, 8.0);  view_231 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_10: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_10: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_10, scalar_tensor_10, div_41);  scalar_tensor_10 = div_41 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_10: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
    sub_31: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_10, amax_10);  where_10 = amax_10 = None
    exp_10: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_42: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    alias_31: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(div_42)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_93: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_42);  div_42 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_42: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_93, [4, 12, 128, 128]);  clone_93 = None
    view_232: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_42, [48, 128, 128]);  expand_42 = None
    expand_43: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_115, [4, 12, 128, 64]);  permute_115 = None
    clone_94: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_233: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_94, [48, 128, 64]);  clone_94 = None
    bmm_21: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_232, view_233)
    view_234: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_21, [4, 12, 128, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_117: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
    clone_95: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_235: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_95, [4, -1, 768]);  clone_95 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_236: "f32[512, 768]" = torch.ops.aten.view.default(view_235, [512, 768]);  view_235 = None
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
    addmm_63: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_178, view_236, permute_118);  primals_178 = None
    view_237: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_63, [4, 128, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_96: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_237);  view_237 = None
    add_74: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_90, clone_96);  clone_96 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_21: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_74, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_21: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_74, [-1], correction = 1.0, keepdim = True)
    sqrt_21: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_21);  var_21 = None
    alias_32: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_21)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_32: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_74, mean_21);  mean_21 = None
    mul_51: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_43, sub_32)
    add_75: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_21, 1e-06);  sqrt_21 = None
    div_43: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_51, add_75)
    add_76: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_43, primals_44);  div_43 = primals_44 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_238: "f32[512, 768]" = torch.ops.aten.view.default(add_76, [512, 768]);  add_76 = None
    permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
    addmm_64: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_180, view_238, permute_119);  primals_180 = None
    view_239: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_64, [4, 128, 3072]);  addmm_64 = None
    mul_52: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
    mul_53: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_10: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_53);  mul_53 = None
    add_77: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_54: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_77);  mul_52 = add_77 = None
    clone_97: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_54);  mul_54 = None
    view_240: "f32[512, 3072]" = torch.ops.aten.view.default(clone_97, [512, 3072]);  clone_97 = None
    permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
    addmm_65: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_182, view_240, permute_120);  primals_182 = None
    view_241: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_65, [4, 128, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_98: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_241);  view_241 = None
    add_78: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_74, clone_98);  clone_98 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_99: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_78);  add_78 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_22: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_99, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_22: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_99, [-1], correction = 1.0, keepdim = True)
    sqrt_22: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_22);  var_22 = None
    alias_33: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_22)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_33: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_99, mean_22);  mean_22 = None
    mul_55: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_45, sub_33)
    add_79: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_22, 1e-06);  sqrt_22 = None
    div_44: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_55, add_79)
    add_80: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_44, primals_46);  div_44 = primals_46 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_242: "f32[512, 768]" = torch.ops.aten.view.default(add_80, [512, 768])
    permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
    addmm_66: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_184, view_242, permute_121);  primals_184 = None
    view_243: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_66, [4, 128, 768]);  addmm_66 = None
    view_244: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_243, [4, -1, 12, 64]);  view_243 = None
    permute_122: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
    view_245: "f32[512, 768]" = torch.ops.aten.view.default(add_80, [512, 768])
    permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
    addmm_67: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_186, view_245, permute_123);  primals_186 = None
    view_246: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_67, [4, 128, 768]);  addmm_67 = None
    view_247: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_246, [4, -1, 12, 64]);  view_246 = None
    permute_124: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
    view_248: "f32[512, 768]" = torch.ops.aten.view.default(add_80, [512, 768]);  add_80 = None
    permute_125: "f32[768, 768]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
    addmm_68: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_188, view_248, permute_125);  primals_188 = None
    view_249: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_68, [4, 128, 768]);  addmm_68 = None
    view_250: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_249, [4, -1, 12, 64]);  view_249 = None
    permute_126: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    permute_127: "f32[4, 12, 64, 128]" = torch.ops.aten.permute.default(permute_124, [0, 1, 3, 2]);  permute_124 = None
    expand_44: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_122, [4, 12, 128, 64]);  permute_122 = None
    clone_100: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
    view_251: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_100, [48, 128, 64]);  clone_100 = None
    expand_45: "f32[4, 12, 64, 128]" = torch.ops.aten.expand.default(permute_127, [4, 12, 64, 128]);  permute_127 = None
    clone_101: "f32[4, 12, 64, 128]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
    view_252: "f32[48, 64, 128]" = torch.ops.aten.view.default(clone_101, [48, 64, 128]);  clone_101 = None
    bmm_22: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_251, view_252)
    view_253: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_22, [4, 12, 128, 128]);  bmm_22 = None
    div_45: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_253, 8.0);  view_253 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_11: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_11: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_11, scalar_tensor_11, div_45);  scalar_tensor_11 = div_45 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_11: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
    sub_34: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_11, amax_11);  where_11 = amax_11 = None
    exp_11: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_46: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    alias_34: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(div_46)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_102: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_46);  div_46 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_46: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_102, [4, 12, 128, 128]);  clone_102 = None
    view_254: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_46, [48, 128, 128]);  expand_46 = None
    expand_47: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_126, [4, 12, 128, 64]);  permute_126 = None
    clone_103: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_255: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_103, [48, 128, 64]);  clone_103 = None
    bmm_23: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_254, view_255)
    view_256: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_23, [4, 12, 128, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_128: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    clone_104: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_257: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_104, [4, -1, 768]);  clone_104 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_258: "f32[512, 768]" = torch.ops.aten.view.default(view_257, [512, 768]);  view_257 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
    addmm_69: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_190, view_258, permute_129);  primals_190 = None
    view_259: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_69, [4, 128, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_105: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_259);  view_259 = None
    add_81: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_99, clone_105);  clone_105 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_23: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_81, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_23: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_81, [-1], correction = 1.0, keepdim = True)
    sqrt_23: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_23);  var_23 = None
    alias_35: "f32[4, 128, 1]" = torch.ops.aten.alias.default(sqrt_23)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_35: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_81, mean_23);  mean_23 = None
    mul_56: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(primals_47, sub_35)
    add_82: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_23, 1e-06);  sqrt_23 = None
    div_47: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_56, add_82)
    add_83: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_47, primals_48);  div_47 = primals_48 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_260: "f32[512, 768]" = torch.ops.aten.view.default(add_83, [512, 768]);  add_83 = None
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
    addmm_70: "f32[512, 3072]" = torch.ops.aten.addmm.default(primals_192, view_260, permute_130);  primals_192 = None
    view_261: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_70, [4, 128, 3072]);  addmm_70 = None
    mul_57: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_58: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_11: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_84: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_59: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_57, add_84);  mul_57 = add_84 = None
    clone_106: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_59);  mul_59 = None
    view_262: "f32[512, 3072]" = torch.ops.aten.view.default(clone_106, [512, 3072]);  clone_106 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
    addmm_71: "f32[512, 768]" = torch.ops.aten.addmm.default(primals_194, view_262, permute_131);  primals_194 = None
    view_263: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_71, [4, 128, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_107: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_263);  view_263 = None
    add_85: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_81, clone_107);  clone_107 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_108: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_85);  add_85 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_264: "f32[512, 768]" = torch.ops.aten.view.default(tangents_1, [512, 768])
    permute_132: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
    mm: "f32[512, 3072]" = torch.ops.aten.mm.default(view_264, permute_132);  permute_132 = None
    permute_133: "f32[768, 512]" = torch.ops.aten.permute.default(view_264, [1, 0])
    mm_1: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_133, view_262);  permute_133 = view_262 = None
    permute_134: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_1, [1, 0]);  mm_1 = None
    sum_13: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_264, [0], True);  view_264 = None
    view_265: "f32[768]" = torch.ops.aten.view.default(sum_13, [768]);  sum_13 = None
    permute_135: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
    view_266: "f32[4, 128, 3072]" = torch.ops.aten.view.default(mm, [4, 128, 3072]);  mm = None
    mul_60: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
    erf_12: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_60);  mul_60 = None
    add_86: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
    mul_61: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(add_86, 0.5);  add_86 = None
    mul_62: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_261, view_261)
    mul_63: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_62, -0.5);  mul_62 = None
    exp_12: "f32[4, 128, 3072]" = torch.ops.aten.exp.default(mul_63);  mul_63 = None
    mul_64: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_12, 0.3989422804014327);  exp_12 = None
    mul_65: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_261, mul_64);  view_261 = mul_64 = None
    add_87: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(mul_61, mul_65);  mul_61 = mul_65 = None
    mul_66: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_266, add_87);  view_266 = add_87 = None
    view_267: "f32[512, 3072]" = torch.ops.aten.view.default(mul_66, [512, 3072]);  mul_66 = None
    permute_136: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
    mm_2: "f32[512, 768]" = torch.ops.aten.mm.default(view_267, permute_136);  permute_136 = None
    permute_137: "f32[3072, 512]" = torch.ops.aten.permute.default(view_267, [1, 0])
    mm_3: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_137, view_260);  permute_137 = view_260 = None
    permute_138: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_3, [1, 0]);  mm_3 = None
    sum_14: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_267, [0], True);  view_267 = None
    view_268: "f32[3072]" = torch.ops.aten.view.default(sum_14, [3072]);  sum_14 = None
    permute_139: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
    view_269: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_2, [4, 128, 768]);  mm_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_15: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_269, [0, 1], True)
    view_270: "f32[768]" = torch.ops.aten.view.default(sum_15, [768]);  sum_15 = None
    div_48: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_56, add_82);  mul_56 = None
    div_49: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_48, add_82);  div_48 = None
    neg: "f32[4, 128, 768]" = torch.ops.aten.neg.default(view_269)
    mul_67: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg, div_49);  neg = div_49 = None
    div_50: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(view_269, add_82);  view_269 = add_82 = None
    sum_16: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_67, [2], True);  mul_67 = None
    mul_68: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_50, primals_47);  primals_47 = None
    mul_69: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_50, sub_35);  div_50 = sub_35 = None
    sum_17: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_69, [0, 1], True);  mul_69 = None
    view_271: "f32[768]" = torch.ops.aten.view.default(sum_17, [768]);  sum_17 = None
    neg_1: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_68)
    sum_18: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_1, [2], True);  neg_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_88: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(tangents_1, mul_68);  tangents_1 = mul_68 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_36: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_35);  alias_35 = None
    mul_70: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_36, 2)
    div_51: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_16, mul_70);  sum_16 = mul_70 = None
    eq_12: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_36, 0);  alias_36 = None
    scalar_tensor_12: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_12: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_12, scalar_tensor_12, div_51);  eq_12 = scalar_tensor_12 = div_51 = None
    mean_24: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_81, [-1], True)
    sub_36: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_81, mean_24);  add_81 = mean_24 = None
    mul_71: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_12, 0.002607561929595828);  where_12 = None
    mul_72: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_71, sub_36);  mul_71 = sub_36 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_89: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_88, mul_72);  add_88 = mul_72 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_48: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_18, [4, 128, 768]);  sum_18 = None
    div_52: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_48, 768);  expand_48 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_90: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_89, div_52);  add_89 = div_52 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_272: "f32[512, 768]" = torch.ops.aten.view.default(add_90, [512, 768])
    permute_140: "f32[768, 768]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
    mm_4: "f32[512, 768]" = torch.ops.aten.mm.default(view_272, permute_140);  permute_140 = None
    permute_141: "f32[768, 512]" = torch.ops.aten.permute.default(view_272, [1, 0])
    mm_5: "f32[768, 768]" = torch.ops.aten.mm.default(permute_141, view_258);  permute_141 = view_258 = None
    permute_142: "f32[768, 768]" = torch.ops.aten.permute.default(mm_5, [1, 0]);  mm_5 = None
    sum_19: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_272, [0], True);  view_272 = None
    view_273: "f32[768]" = torch.ops.aten.view.default(sum_19, [768]);  sum_19 = None
    permute_143: "f32[768, 768]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
    view_274: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_4, [4, 128, 768]);  mm_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    view_275: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_274, [4, 128, 12, 64]);  view_274 = None
    permute_144: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_275, [0, 2, 1, 3]);  view_275 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    clone_109: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(permute_144, memory_format = torch.contiguous_format);  permute_144 = None
    view_276: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_109, [48, 128, 64]);  clone_109 = None
    permute_145: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
    bmm_24: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(permute_145, view_276);  permute_145 = None
    permute_146: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None
    bmm_25: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_276, permute_146);  view_276 = permute_146 = None
    view_277: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_24, [4, 12, 128, 64]);  bmm_24 = None
    view_278: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_25, [4, 12, 128, 128]);  bmm_25 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    alias_37: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(alias_34);  alias_34 = None
    mul_73: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_278, alias_37);  view_278 = None
    sum_20: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_73, [-1], True)
    mul_74: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_37, sum_20);  alias_37 = sum_20 = None
    sub_37: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_73, mul_74);  mul_73 = mul_74 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    scalar_tensor_13: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_13: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_11, scalar_tensor_13, sub_37);  eq_11 = scalar_tensor_13 = sub_37 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    div_53: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_13, 8.0);  where_13 = None
    view_279: "f32[48, 128, 128]" = torch.ops.aten.view.default(div_53, [48, 128, 128]);  div_53 = None
    permute_147: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_251, [0, 2, 1]);  view_251 = None
    bmm_26: "f32[48, 64, 128]" = torch.ops.aten.bmm.default(permute_147, view_279);  permute_147 = None
    permute_148: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_252, [0, 2, 1]);  view_252 = None
    bmm_27: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_279, permute_148);  view_279 = permute_148 = None
    view_280: "f32[4, 12, 64, 128]" = torch.ops.aten.view.default(bmm_26, [4, 12, 64, 128]);  bmm_26 = None
    view_281: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_27, [4, 12, 128, 64]);  bmm_27 = None
    permute_149: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_280, [0, 1, 3, 2]);  view_280 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_150: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_277, [0, 2, 1, 3]);  view_277 = None
    clone_110: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
    view_282: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_110, [4, 128, 768]);  clone_110 = None
    view_283: "f32[512, 768]" = torch.ops.aten.view.default(view_282, [512, 768]);  view_282 = None
    permute_151: "f32[768, 768]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
    mm_6: "f32[512, 768]" = torch.ops.aten.mm.default(view_283, permute_151);  permute_151 = None
    permute_152: "f32[768, 512]" = torch.ops.aten.permute.default(view_283, [1, 0])
    mm_7: "f32[768, 768]" = torch.ops.aten.mm.default(permute_152, view_248);  permute_152 = view_248 = None
    permute_153: "f32[768, 768]" = torch.ops.aten.permute.default(mm_7, [1, 0]);  mm_7 = None
    sum_21: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_283, [0], True);  view_283 = None
    view_284: "f32[768]" = torch.ops.aten.view.default(sum_21, [768]);  sum_21 = None
    permute_154: "f32[768, 768]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
    view_285: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_6, [4, 128, 768]);  mm_6 = None
    permute_155: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(permute_149, [0, 2, 1, 3]);  permute_149 = None
    view_286: "f32[4, 128, 768]" = torch.ops.aten.view.default(permute_155, [4, 128, 768]);  permute_155 = None
    clone_111: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_286, memory_format = torch.contiguous_format);  view_286 = None
    view_287: "f32[512, 768]" = torch.ops.aten.view.default(clone_111, [512, 768]);  clone_111 = None
    permute_156: "f32[768, 768]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
    mm_8: "f32[512, 768]" = torch.ops.aten.mm.default(view_287, permute_156);  permute_156 = None
    permute_157: "f32[768, 512]" = torch.ops.aten.permute.default(view_287, [1, 0])
    mm_9: "f32[768, 768]" = torch.ops.aten.mm.default(permute_157, view_245);  permute_157 = view_245 = None
    permute_158: "f32[768, 768]" = torch.ops.aten.permute.default(mm_9, [1, 0]);  mm_9 = None
    sum_22: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_287, [0], True);  view_287 = None
    view_288: "f32[768]" = torch.ops.aten.view.default(sum_22, [768]);  sum_22 = None
    permute_159: "f32[768, 768]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
    view_289: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_8, [4, 128, 768]);  mm_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_91: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(view_285, view_289);  view_285 = view_289 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_160: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_281, [0, 2, 1, 3]);  view_281 = None
    clone_112: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_160, memory_format = torch.contiguous_format);  permute_160 = None
    view_290: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_112, [4, 128, 768]);  clone_112 = None
    view_291: "f32[512, 768]" = torch.ops.aten.view.default(view_290, [512, 768]);  view_290 = None
    permute_161: "f32[768, 768]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
    mm_10: "f32[512, 768]" = torch.ops.aten.mm.default(view_291, permute_161);  permute_161 = None
    permute_162: "f32[768, 512]" = torch.ops.aten.permute.default(view_291, [1, 0])
    mm_11: "f32[768, 768]" = torch.ops.aten.mm.default(permute_162, view_242);  permute_162 = view_242 = None
    permute_163: "f32[768, 768]" = torch.ops.aten.permute.default(mm_11, [1, 0]);  mm_11 = None
    sum_23: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_291, [0], True);  view_291 = None
    view_292: "f32[768]" = torch.ops.aten.view.default(sum_23, [768]);  sum_23 = None
    permute_164: "f32[768, 768]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
    view_293: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_10, [4, 128, 768]);  mm_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_92: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_91, view_293);  add_91 = view_293 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_24: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_92, [0, 1], True)
    view_294: "f32[768]" = torch.ops.aten.view.default(sum_24, [768]);  sum_24 = None
    div_54: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_55, add_79);  mul_55 = None
    div_55: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_54, add_79);  div_54 = None
    neg_2: "f32[4, 128, 768]" = torch.ops.aten.neg.default(add_92)
    mul_75: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_2, div_55);  neg_2 = div_55 = None
    div_56: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(add_92, add_79);  add_92 = add_79 = None
    sum_25: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_75, [2], True);  mul_75 = None
    mul_76: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_56, primals_45);  primals_45 = None
    mul_77: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_56, sub_33);  div_56 = sub_33 = None
    sum_26: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_77, [0, 1], True);  mul_77 = None
    view_295: "f32[768]" = torch.ops.aten.view.default(sum_26, [768]);  sum_26 = None
    neg_3: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_76)
    sum_27: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_3, [2], True);  neg_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_93: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_90, mul_76);  add_90 = mul_76 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_38: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_33);  alias_33 = None
    mul_78: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_38, 2)
    div_57: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_25, mul_78);  sum_25 = mul_78 = None
    eq_13: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_38, 0);  alias_38 = None
    scalar_tensor_14: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_14: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_13, scalar_tensor_14, div_57);  eq_13 = scalar_tensor_14 = div_57 = None
    mean_25: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_99, [-1], True)
    sub_38: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_99, mean_25);  clone_99 = mean_25 = None
    mul_79: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_14, 0.002607561929595828);  where_14 = None
    mul_80: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_79, sub_38);  mul_79 = sub_38 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_94: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_93, mul_80);  add_93 = mul_80 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_49: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_27, [4, 128, 768]);  sum_27 = None
    div_58: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_49, 768);  expand_49 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_95: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_94, div_58);  add_94 = div_58 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_296: "f32[512, 768]" = torch.ops.aten.view.default(add_95, [512, 768])
    permute_165: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
    mm_12: "f32[512, 3072]" = torch.ops.aten.mm.default(view_296, permute_165);  permute_165 = None
    permute_166: "f32[768, 512]" = torch.ops.aten.permute.default(view_296, [1, 0])
    mm_13: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_166, view_240);  permute_166 = view_240 = None
    permute_167: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_13, [1, 0]);  mm_13 = None
    sum_28: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_296, [0], True);  view_296 = None
    view_297: "f32[768]" = torch.ops.aten.view.default(sum_28, [768]);  sum_28 = None
    permute_168: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
    view_298: "f32[4, 128, 3072]" = torch.ops.aten.view.default(mm_12, [4, 128, 3072]);  mm_12 = None
    mul_81: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
    erf_13: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_81);  mul_81 = None
    add_96: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
    mul_82: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
    mul_83: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_239, view_239)
    mul_84: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_83, -0.5);  mul_83 = None
    exp_13: "f32[4, 128, 3072]" = torch.ops.aten.exp.default(mul_84);  mul_84 = None
    mul_85: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_13, 0.3989422804014327);  exp_13 = None
    mul_86: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_239, mul_85);  view_239 = mul_85 = None
    add_97: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(mul_82, mul_86);  mul_82 = mul_86 = None
    mul_87: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_298, add_97);  view_298 = add_97 = None
    view_299: "f32[512, 3072]" = torch.ops.aten.view.default(mul_87, [512, 3072]);  mul_87 = None
    permute_169: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
    mm_14: "f32[512, 768]" = torch.ops.aten.mm.default(view_299, permute_169);  permute_169 = None
    permute_170: "f32[3072, 512]" = torch.ops.aten.permute.default(view_299, [1, 0])
    mm_15: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_170, view_238);  permute_170 = view_238 = None
    permute_171: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_15, [1, 0]);  mm_15 = None
    sum_29: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_299, [0], True);  view_299 = None
    view_300: "f32[3072]" = torch.ops.aten.view.default(sum_29, [3072]);  sum_29 = None
    permute_172: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
    view_301: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_14, [4, 128, 768]);  mm_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_30: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_301, [0, 1], True)
    view_302: "f32[768]" = torch.ops.aten.view.default(sum_30, [768]);  sum_30 = None
    div_59: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_51, add_75);  mul_51 = None
    div_60: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_59, add_75);  div_59 = None
    neg_4: "f32[4, 128, 768]" = torch.ops.aten.neg.default(view_301)
    mul_88: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_4, div_60);  neg_4 = div_60 = None
    div_61: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(view_301, add_75);  view_301 = add_75 = None
    sum_31: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_88, [2], True);  mul_88 = None
    mul_89: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_61, primals_43);  primals_43 = None
    mul_90: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_61, sub_32);  div_61 = sub_32 = None
    sum_32: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_90, [0, 1], True);  mul_90 = None
    view_303: "f32[768]" = torch.ops.aten.view.default(sum_32, [768]);  sum_32 = None
    neg_5: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_89)
    sum_33: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_5, [2], True);  neg_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_98: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_95, mul_89);  add_95 = mul_89 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_39: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_32);  alias_32 = None
    mul_91: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_39, 2)
    div_62: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_31, mul_91);  sum_31 = mul_91 = None
    eq_14: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_39, 0);  alias_39 = None
    scalar_tensor_15: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_15: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_14, scalar_tensor_15, div_62);  eq_14 = scalar_tensor_15 = div_62 = None
    mean_26: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_74, [-1], True)
    sub_39: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_74, mean_26);  add_74 = mean_26 = None
    mul_92: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_15, 0.002607561929595828);  where_15 = None
    mul_93: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_92, sub_39);  mul_92 = sub_39 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_99: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_98, mul_93);  add_98 = mul_93 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_50: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_33, [4, 128, 768]);  sum_33 = None
    div_63: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_50, 768);  expand_50 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_100: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_99, div_63);  add_99 = div_63 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_304: "f32[512, 768]" = torch.ops.aten.view.default(add_100, [512, 768])
    permute_173: "f32[768, 768]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
    mm_16: "f32[512, 768]" = torch.ops.aten.mm.default(view_304, permute_173);  permute_173 = None
    permute_174: "f32[768, 512]" = torch.ops.aten.permute.default(view_304, [1, 0])
    mm_17: "f32[768, 768]" = torch.ops.aten.mm.default(permute_174, view_236);  permute_174 = view_236 = None
    permute_175: "f32[768, 768]" = torch.ops.aten.permute.default(mm_17, [1, 0]);  mm_17 = None
    sum_34: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_304, [0], True);  view_304 = None
    view_305: "f32[768]" = torch.ops.aten.view.default(sum_34, [768]);  sum_34 = None
    permute_176: "f32[768, 768]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
    view_306: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_16, [4, 128, 768]);  mm_16 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    view_307: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_306, [4, 128, 12, 64]);  view_306 = None
    permute_177: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    clone_113: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(permute_177, memory_format = torch.contiguous_format);  permute_177 = None
    view_308: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_113, [48, 128, 64]);  clone_113 = None
    permute_178: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_232, [0, 2, 1]);  view_232 = None
    bmm_28: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(permute_178, view_308);  permute_178 = None
    permute_179: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
    bmm_29: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_308, permute_179);  view_308 = permute_179 = None
    view_309: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_28, [4, 12, 128, 64]);  bmm_28 = None
    view_310: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_29, [4, 12, 128, 128]);  bmm_29 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    alias_40: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(alias_31);  alias_31 = None
    mul_94: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_310, alias_40);  view_310 = None
    sum_35: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_94, [-1], True)
    mul_95: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_40, sum_35);  alias_40 = sum_35 = None
    sub_40: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_94, mul_95);  mul_94 = mul_95 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    scalar_tensor_16: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_16: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_10, scalar_tensor_16, sub_40);  eq_10 = scalar_tensor_16 = sub_40 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    div_64: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_16, 8.0);  where_16 = None
    view_311: "f32[48, 128, 128]" = torch.ops.aten.view.default(div_64, [48, 128, 128]);  div_64 = None
    permute_180: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
    bmm_30: "f32[48, 64, 128]" = torch.ops.aten.bmm.default(permute_180, view_311);  permute_180 = None
    permute_181: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_230, [0, 2, 1]);  view_230 = None
    bmm_31: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_311, permute_181);  view_311 = permute_181 = None
    view_312: "f32[4, 12, 64, 128]" = torch.ops.aten.view.default(bmm_30, [4, 12, 64, 128]);  bmm_30 = None
    view_313: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_31, [4, 12, 128, 64]);  bmm_31 = None
    permute_182: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_312, [0, 1, 3, 2]);  view_312 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_183: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_309, [0, 2, 1, 3]);  view_309 = None
    clone_114: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_183, memory_format = torch.contiguous_format);  permute_183 = None
    view_314: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_114, [4, 128, 768]);  clone_114 = None
    view_315: "f32[512, 768]" = torch.ops.aten.view.default(view_314, [512, 768]);  view_314 = None
    permute_184: "f32[768, 768]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
    mm_18: "f32[512, 768]" = torch.ops.aten.mm.default(view_315, permute_184);  permute_184 = None
    permute_185: "f32[768, 512]" = torch.ops.aten.permute.default(view_315, [1, 0])
    mm_19: "f32[768, 768]" = torch.ops.aten.mm.default(permute_185, view_226);  permute_185 = view_226 = None
    permute_186: "f32[768, 768]" = torch.ops.aten.permute.default(mm_19, [1, 0]);  mm_19 = None
    sum_36: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_315, [0], True);  view_315 = None
    view_316: "f32[768]" = torch.ops.aten.view.default(sum_36, [768]);  sum_36 = None
    permute_187: "f32[768, 768]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
    view_317: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_18, [4, 128, 768]);  mm_18 = None
    permute_188: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(permute_182, [0, 2, 1, 3]);  permute_182 = None
    view_318: "f32[4, 128, 768]" = torch.ops.aten.view.default(permute_188, [4, 128, 768]);  permute_188 = None
    clone_115: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_318, memory_format = torch.contiguous_format);  view_318 = None
    view_319: "f32[512, 768]" = torch.ops.aten.view.default(clone_115, [512, 768]);  clone_115 = None
    permute_189: "f32[768, 768]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
    mm_20: "f32[512, 768]" = torch.ops.aten.mm.default(view_319, permute_189);  permute_189 = None
    permute_190: "f32[768, 512]" = torch.ops.aten.permute.default(view_319, [1, 0])
    mm_21: "f32[768, 768]" = torch.ops.aten.mm.default(permute_190, view_223);  permute_190 = view_223 = None
    permute_191: "f32[768, 768]" = torch.ops.aten.permute.default(mm_21, [1, 0]);  mm_21 = None
    sum_37: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_319, [0], True);  view_319 = None
    view_320: "f32[768]" = torch.ops.aten.view.default(sum_37, [768]);  sum_37 = None
    permute_192: "f32[768, 768]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
    view_321: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_20, [4, 128, 768]);  mm_20 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_101: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(view_317, view_321);  view_317 = view_321 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_193: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
    clone_116: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_193, memory_format = torch.contiguous_format);  permute_193 = None
    view_322: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_116, [4, 128, 768]);  clone_116 = None
    view_323: "f32[512, 768]" = torch.ops.aten.view.default(view_322, [512, 768]);  view_322 = None
    permute_194: "f32[768, 768]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
    mm_22: "f32[512, 768]" = torch.ops.aten.mm.default(view_323, permute_194);  permute_194 = None
    permute_195: "f32[768, 512]" = torch.ops.aten.permute.default(view_323, [1, 0])
    mm_23: "f32[768, 768]" = torch.ops.aten.mm.default(permute_195, view_220);  permute_195 = view_220 = None
    permute_196: "f32[768, 768]" = torch.ops.aten.permute.default(mm_23, [1, 0]);  mm_23 = None
    sum_38: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_323, [0], True);  view_323 = None
    view_324: "f32[768]" = torch.ops.aten.view.default(sum_38, [768]);  sum_38 = None
    permute_197: "f32[768, 768]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
    view_325: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_22, [4, 128, 768]);  mm_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_102: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_101, view_325);  add_101 = view_325 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_39: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_102, [0, 1], True)
    view_326: "f32[768]" = torch.ops.aten.view.default(sum_39, [768]);  sum_39 = None
    div_65: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_50, add_72);  mul_50 = None
    div_66: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_65, add_72);  div_65 = None
    neg_6: "f32[4, 128, 768]" = torch.ops.aten.neg.default(add_102)
    mul_96: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_6, div_66);  neg_6 = div_66 = None
    div_67: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(add_102, add_72);  add_102 = add_72 = None
    sum_40: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_96, [2], True);  mul_96 = None
    mul_97: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_67, primals_41);  primals_41 = None
    mul_98: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_67, sub_30);  div_67 = sub_30 = None
    sum_41: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_98, [0, 1], True);  mul_98 = None
    view_327: "f32[768]" = torch.ops.aten.view.default(sum_41, [768]);  sum_41 = None
    neg_7: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_97)
    sum_42: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_7, [2], True);  neg_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_103: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_100, mul_97);  add_100 = mul_97 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_41: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_30);  alias_30 = None
    mul_99: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_41, 2)
    div_68: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_40, mul_99);  sum_40 = mul_99 = None
    eq_15: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_41, 0);  alias_41 = None
    scalar_tensor_17: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_17: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_15, scalar_tensor_17, div_68);  eq_15 = scalar_tensor_17 = div_68 = None
    mean_27: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_90, [-1], True)
    sub_41: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_90, mean_27);  clone_90 = mean_27 = None
    mul_100: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_17, 0.002607561929595828);  where_17 = None
    mul_101: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_100, sub_41);  mul_100 = sub_41 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_104: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_103, mul_101);  add_103 = mul_101 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_51: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_42, [4, 128, 768]);  sum_42 = None
    div_69: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_51, 768);  expand_51 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_105: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_104, div_69);  add_104 = div_69 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_328: "f32[512, 768]" = torch.ops.aten.view.default(add_105, [512, 768])
    permute_198: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
    mm_24: "f32[512, 3072]" = torch.ops.aten.mm.default(view_328, permute_198);  permute_198 = None
    permute_199: "f32[768, 512]" = torch.ops.aten.permute.default(view_328, [1, 0])
    mm_25: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_199, view_218);  permute_199 = view_218 = None
    permute_200: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_25, [1, 0]);  mm_25 = None
    sum_43: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_328, [0], True);  view_328 = None
    view_329: "f32[768]" = torch.ops.aten.view.default(sum_43, [768]);  sum_43 = None
    permute_201: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
    view_330: "f32[4, 128, 3072]" = torch.ops.aten.view.default(mm_24, [4, 128, 3072]);  mm_24 = None
    mul_102: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
    erf_14: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_102);  mul_102 = None
    add_106: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
    mul_103: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(add_106, 0.5);  add_106 = None
    mul_104: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_217, view_217)
    mul_105: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_104, -0.5);  mul_104 = None
    exp_14: "f32[4, 128, 3072]" = torch.ops.aten.exp.default(mul_105);  mul_105 = None
    mul_106: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_14, 0.3989422804014327);  exp_14 = None
    mul_107: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_217, mul_106);  view_217 = mul_106 = None
    add_107: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(mul_103, mul_107);  mul_103 = mul_107 = None
    mul_108: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_330, add_107);  view_330 = add_107 = None
    view_331: "f32[512, 3072]" = torch.ops.aten.view.default(mul_108, [512, 3072]);  mul_108 = None
    permute_202: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
    mm_26: "f32[512, 768]" = torch.ops.aten.mm.default(view_331, permute_202);  permute_202 = None
    permute_203: "f32[3072, 512]" = torch.ops.aten.permute.default(view_331, [1, 0])
    mm_27: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_203, view_216);  permute_203 = view_216 = None
    permute_204: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_27, [1, 0]);  mm_27 = None
    sum_44: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_331, [0], True);  view_331 = None
    view_332: "f32[3072]" = torch.ops.aten.view.default(sum_44, [3072]);  sum_44 = None
    permute_205: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_204, [1, 0]);  permute_204 = None
    view_333: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_26, [4, 128, 768]);  mm_26 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_45: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_333, [0, 1], True)
    view_334: "f32[768]" = torch.ops.aten.view.default(sum_45, [768]);  sum_45 = None
    div_70: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_46, add_68);  mul_46 = None
    div_71: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_70, add_68);  div_70 = None
    neg_8: "f32[4, 128, 768]" = torch.ops.aten.neg.default(view_333)
    mul_109: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_8, div_71);  neg_8 = div_71 = None
    div_72: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(view_333, add_68);  view_333 = add_68 = None
    sum_46: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_109, [2], True);  mul_109 = None
    mul_110: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_72, primals_39);  primals_39 = None
    mul_111: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_72, sub_29);  div_72 = sub_29 = None
    sum_47: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_111, [0, 1], True);  mul_111 = None
    view_335: "f32[768]" = torch.ops.aten.view.default(sum_47, [768]);  sum_47 = None
    neg_9: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_110)
    sum_48: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_9, [2], True);  neg_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_108: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_105, mul_110);  add_105 = mul_110 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_42: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_29);  alias_29 = None
    mul_112: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_42, 2)
    div_73: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_46, mul_112);  sum_46 = mul_112 = None
    eq_16: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_42, 0);  alias_42 = None
    scalar_tensor_18: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_18: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_16, scalar_tensor_18, div_73);  eq_16 = scalar_tensor_18 = div_73 = None
    mean_28: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_67, [-1], True)
    sub_42: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_67, mean_28);  add_67 = mean_28 = None
    mul_113: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_18, 0.002607561929595828);  where_18 = None
    mul_114: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_113, sub_42);  mul_113 = sub_42 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_109: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_108, mul_114);  add_108 = mul_114 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_52: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_48, [4, 128, 768]);  sum_48 = None
    div_74: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_52, 768);  expand_52 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_110: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_109, div_74);  add_109 = div_74 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_336: "f32[512, 768]" = torch.ops.aten.view.default(add_110, [512, 768])
    permute_206: "f32[768, 768]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
    mm_28: "f32[512, 768]" = torch.ops.aten.mm.default(view_336, permute_206);  permute_206 = None
    permute_207: "f32[768, 512]" = torch.ops.aten.permute.default(view_336, [1, 0])
    mm_29: "f32[768, 768]" = torch.ops.aten.mm.default(permute_207, view_214);  permute_207 = view_214 = None
    permute_208: "f32[768, 768]" = torch.ops.aten.permute.default(mm_29, [1, 0]);  mm_29 = None
    sum_49: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_336, [0], True);  view_336 = None
    view_337: "f32[768]" = torch.ops.aten.view.default(sum_49, [768]);  sum_49 = None
    permute_209: "f32[768, 768]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
    view_338: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_28, [4, 128, 768]);  mm_28 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    view_339: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_338, [4, 128, 12, 64]);  view_338 = None
    permute_210: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_339, [0, 2, 1, 3]);  view_339 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    clone_117: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
    view_340: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_117, [48, 128, 64]);  clone_117 = None
    permute_211: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
    bmm_32: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(permute_211, view_340);  permute_211 = None
    permute_212: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None
    bmm_33: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_340, permute_212);  view_340 = permute_212 = None
    view_341: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_32, [4, 12, 128, 64]);  bmm_32 = None
    view_342: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_33, [4, 12, 128, 128]);  bmm_33 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    alias_43: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(alias_28);  alias_28 = None
    mul_115: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_342, alias_43);  view_342 = None
    sum_50: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_115, [-1], True)
    mul_116: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_43, sum_50);  alias_43 = sum_50 = None
    sub_43: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_115, mul_116);  mul_115 = mul_116 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    scalar_tensor_19: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_19: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_9, scalar_tensor_19, sub_43);  eq_9 = scalar_tensor_19 = sub_43 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    div_75: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_19, 8.0);  where_19 = None
    view_343: "f32[48, 128, 128]" = torch.ops.aten.view.default(div_75, [48, 128, 128]);  div_75 = None
    permute_213: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None
    bmm_34: "f32[48, 64, 128]" = torch.ops.aten.bmm.default(permute_213, view_343);  permute_213 = None
    permute_214: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_208, [0, 2, 1]);  view_208 = None
    bmm_35: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_343, permute_214);  view_343 = permute_214 = None
    view_344: "f32[4, 12, 64, 128]" = torch.ops.aten.view.default(bmm_34, [4, 12, 64, 128]);  bmm_34 = None
    view_345: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_35, [4, 12, 128, 64]);  bmm_35 = None
    permute_215: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_344, [0, 1, 3, 2]);  view_344 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_216: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_341, [0, 2, 1, 3]);  view_341 = None
    clone_118: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_216, memory_format = torch.contiguous_format);  permute_216 = None
    view_346: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_118, [4, 128, 768]);  clone_118 = None
    view_347: "f32[512, 768]" = torch.ops.aten.view.default(view_346, [512, 768]);  view_346 = None
    permute_217: "f32[768, 768]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
    mm_30: "f32[512, 768]" = torch.ops.aten.mm.default(view_347, permute_217);  permute_217 = None
    permute_218: "f32[768, 512]" = torch.ops.aten.permute.default(view_347, [1, 0])
    mm_31: "f32[768, 768]" = torch.ops.aten.mm.default(permute_218, view_204);  permute_218 = view_204 = None
    permute_219: "f32[768, 768]" = torch.ops.aten.permute.default(mm_31, [1, 0]);  mm_31 = None
    sum_51: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_347, [0], True);  view_347 = None
    view_348: "f32[768]" = torch.ops.aten.view.default(sum_51, [768]);  sum_51 = None
    permute_220: "f32[768, 768]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
    view_349: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_30, [4, 128, 768]);  mm_30 = None
    permute_221: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(permute_215, [0, 2, 1, 3]);  permute_215 = None
    view_350: "f32[4, 128, 768]" = torch.ops.aten.view.default(permute_221, [4, 128, 768]);  permute_221 = None
    clone_119: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_350, memory_format = torch.contiguous_format);  view_350 = None
    view_351: "f32[512, 768]" = torch.ops.aten.view.default(clone_119, [512, 768]);  clone_119 = None
    permute_222: "f32[768, 768]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
    mm_32: "f32[512, 768]" = torch.ops.aten.mm.default(view_351, permute_222);  permute_222 = None
    permute_223: "f32[768, 512]" = torch.ops.aten.permute.default(view_351, [1, 0])
    mm_33: "f32[768, 768]" = torch.ops.aten.mm.default(permute_223, view_201);  permute_223 = view_201 = None
    permute_224: "f32[768, 768]" = torch.ops.aten.permute.default(mm_33, [1, 0]);  mm_33 = None
    sum_52: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_351, [0], True);  view_351 = None
    view_352: "f32[768]" = torch.ops.aten.view.default(sum_52, [768]);  sum_52 = None
    permute_225: "f32[768, 768]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
    view_353: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_32, [4, 128, 768]);  mm_32 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_111: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(view_349, view_353);  view_349 = view_353 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_226: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_345, [0, 2, 1, 3]);  view_345 = None
    clone_120: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_226, memory_format = torch.contiguous_format);  permute_226 = None
    view_354: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_120, [4, 128, 768]);  clone_120 = None
    view_355: "f32[512, 768]" = torch.ops.aten.view.default(view_354, [512, 768]);  view_354 = None
    permute_227: "f32[768, 768]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
    mm_34: "f32[512, 768]" = torch.ops.aten.mm.default(view_355, permute_227);  permute_227 = None
    permute_228: "f32[768, 512]" = torch.ops.aten.permute.default(view_355, [1, 0])
    mm_35: "f32[768, 768]" = torch.ops.aten.mm.default(permute_228, view_198);  permute_228 = view_198 = None
    permute_229: "f32[768, 768]" = torch.ops.aten.permute.default(mm_35, [1, 0]);  mm_35 = None
    sum_53: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_355, [0], True);  view_355 = None
    view_356: "f32[768]" = torch.ops.aten.view.default(sum_53, [768]);  sum_53 = None
    permute_230: "f32[768, 768]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
    view_357: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_34, [4, 128, 768]);  mm_34 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_112: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_111, view_357);  add_111 = view_357 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_54: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_112, [0, 1], True)
    view_358: "f32[768]" = torch.ops.aten.view.default(sum_54, [768]);  sum_54 = None
    div_76: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_45, add_65);  mul_45 = None
    div_77: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_76, add_65);  div_76 = None
    neg_10: "f32[4, 128, 768]" = torch.ops.aten.neg.default(add_112)
    mul_117: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_10, div_77);  neg_10 = div_77 = None
    div_78: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(add_112, add_65);  add_112 = add_65 = None
    sum_55: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_117, [2], True);  mul_117 = None
    mul_118: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_78, primals_37);  primals_37 = None
    mul_119: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_78, sub_27);  div_78 = sub_27 = None
    sum_56: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_119, [0, 1], True);  mul_119 = None
    view_359: "f32[768]" = torch.ops.aten.view.default(sum_56, [768]);  sum_56 = None
    neg_11: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_118)
    sum_57: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_11, [2], True);  neg_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_113: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_110, mul_118);  add_110 = mul_118 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_44: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_27);  alias_27 = None
    mul_120: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_44, 2)
    div_79: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_55, mul_120);  sum_55 = mul_120 = None
    eq_17: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_44, 0);  alias_44 = None
    scalar_tensor_20: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_20: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_17, scalar_tensor_20, div_79);  eq_17 = scalar_tensor_20 = div_79 = None
    mean_29: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_81, [-1], True)
    sub_44: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_81, mean_29);  clone_81 = mean_29 = None
    mul_121: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_20, 0.002607561929595828);  where_20 = None
    mul_122: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_121, sub_44);  mul_121 = sub_44 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_114: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_113, mul_122);  add_113 = mul_122 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_53: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_57, [4, 128, 768]);  sum_57 = None
    div_80: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_53, 768);  expand_53 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_115: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_114, div_80);  add_114 = div_80 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_360: "f32[512, 768]" = torch.ops.aten.view.default(add_115, [512, 768])
    permute_231: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
    mm_36: "f32[512, 3072]" = torch.ops.aten.mm.default(view_360, permute_231);  permute_231 = None
    permute_232: "f32[768, 512]" = torch.ops.aten.permute.default(view_360, [1, 0])
    mm_37: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_232, view_196);  permute_232 = view_196 = None
    permute_233: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_37, [1, 0]);  mm_37 = None
    sum_58: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_360, [0], True);  view_360 = None
    view_361: "f32[768]" = torch.ops.aten.view.default(sum_58, [768]);  sum_58 = None
    permute_234: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
    view_362: "f32[4, 128, 3072]" = torch.ops.aten.view.default(mm_36, [4, 128, 3072]);  mm_36 = None
    mul_123: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
    erf_15: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_123);  mul_123 = None
    add_116: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
    mul_124: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(add_116, 0.5);  add_116 = None
    mul_125: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_195, view_195)
    mul_126: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_125, -0.5);  mul_125 = None
    exp_15: "f32[4, 128, 3072]" = torch.ops.aten.exp.default(mul_126);  mul_126 = None
    mul_127: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_15, 0.3989422804014327);  exp_15 = None
    mul_128: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_195, mul_127);  view_195 = mul_127 = None
    add_117: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(mul_124, mul_128);  mul_124 = mul_128 = None
    mul_129: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_362, add_117);  view_362 = add_117 = None
    view_363: "f32[512, 3072]" = torch.ops.aten.view.default(mul_129, [512, 3072]);  mul_129 = None
    permute_235: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
    mm_38: "f32[512, 768]" = torch.ops.aten.mm.default(view_363, permute_235);  permute_235 = None
    permute_236: "f32[3072, 512]" = torch.ops.aten.permute.default(view_363, [1, 0])
    mm_39: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_236, view_194);  permute_236 = view_194 = None
    permute_237: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_39, [1, 0]);  mm_39 = None
    sum_59: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_363, [0], True);  view_363 = None
    view_364: "f32[3072]" = torch.ops.aten.view.default(sum_59, [3072]);  sum_59 = None
    permute_238: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
    view_365: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_38, [4, 128, 768]);  mm_38 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_60: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_365, [0, 1], True)
    view_366: "f32[768]" = torch.ops.aten.view.default(sum_60, [768]);  sum_60 = None
    div_81: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_41, add_61);  mul_41 = None
    div_82: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_81, add_61);  div_81 = None
    neg_12: "f32[4, 128, 768]" = torch.ops.aten.neg.default(view_365)
    mul_130: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_12, div_82);  neg_12 = div_82 = None
    div_83: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(view_365, add_61);  view_365 = add_61 = None
    sum_61: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_130, [2], True);  mul_130 = None
    mul_131: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_83, primals_35);  primals_35 = None
    mul_132: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_83, sub_26);  div_83 = sub_26 = None
    sum_62: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_132, [0, 1], True);  mul_132 = None
    view_367: "f32[768]" = torch.ops.aten.view.default(sum_62, [768]);  sum_62 = None
    neg_13: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_131)
    sum_63: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_13, [2], True);  neg_13 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_118: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_115, mul_131);  add_115 = mul_131 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_45: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_26);  alias_26 = None
    mul_133: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_45, 2)
    div_84: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_61, mul_133);  sum_61 = mul_133 = None
    eq_18: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_45, 0);  alias_45 = None
    scalar_tensor_21: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_21: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_18, scalar_tensor_21, div_84);  eq_18 = scalar_tensor_21 = div_84 = None
    mean_30: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_60, [-1], True)
    sub_45: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_60, mean_30);  add_60 = mean_30 = None
    mul_134: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_21, 0.002607561929595828);  where_21 = None
    mul_135: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_134, sub_45);  mul_134 = sub_45 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_119: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_118, mul_135);  add_118 = mul_135 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_54: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_63, [4, 128, 768]);  sum_63 = None
    div_85: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_54, 768);  expand_54 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_120: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_119, div_85);  add_119 = div_85 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_368: "f32[512, 768]" = torch.ops.aten.view.default(add_120, [512, 768])
    permute_239: "f32[768, 768]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
    mm_40: "f32[512, 768]" = torch.ops.aten.mm.default(view_368, permute_239);  permute_239 = None
    permute_240: "f32[768, 512]" = torch.ops.aten.permute.default(view_368, [1, 0])
    mm_41: "f32[768, 768]" = torch.ops.aten.mm.default(permute_240, view_192);  permute_240 = view_192 = None
    permute_241: "f32[768, 768]" = torch.ops.aten.permute.default(mm_41, [1, 0]);  mm_41 = None
    sum_64: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_368, [0], True);  view_368 = None
    view_369: "f32[768]" = torch.ops.aten.view.default(sum_64, [768]);  sum_64 = None
    permute_242: "f32[768, 768]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
    view_370: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_40, [4, 128, 768]);  mm_40 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    view_371: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_370, [4, 128, 12, 64]);  view_370 = None
    permute_243: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_371, [0, 2, 1, 3]);  view_371 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    clone_121: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(permute_243, memory_format = torch.contiguous_format);  permute_243 = None
    view_372: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_121, [48, 128, 64]);  clone_121 = None
    permute_244: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_188, [0, 2, 1]);  view_188 = None
    bmm_36: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(permute_244, view_372);  permute_244 = None
    permute_245: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_189, [0, 2, 1]);  view_189 = None
    bmm_37: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_372, permute_245);  view_372 = permute_245 = None
    view_373: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_36, [4, 12, 128, 64]);  bmm_36 = None
    view_374: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_37, [4, 12, 128, 128]);  bmm_37 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    alias_46: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(alias_25);  alias_25 = None
    mul_136: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_374, alias_46);  view_374 = None
    sum_65: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_136, [-1], True)
    mul_137: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_46, sum_65);  alias_46 = sum_65 = None
    sub_46: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_136, mul_137);  mul_136 = mul_137 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    scalar_tensor_22: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_22: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_8, scalar_tensor_22, sub_46);  eq_8 = scalar_tensor_22 = sub_46 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    div_86: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_22, 8.0);  where_22 = None
    view_375: "f32[48, 128, 128]" = torch.ops.aten.view.default(div_86, [48, 128, 128]);  div_86 = None
    permute_246: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_185, [0, 2, 1]);  view_185 = None
    bmm_38: "f32[48, 64, 128]" = torch.ops.aten.bmm.default(permute_246, view_375);  permute_246 = None
    permute_247: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_186, [0, 2, 1]);  view_186 = None
    bmm_39: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_375, permute_247);  view_375 = permute_247 = None
    view_376: "f32[4, 12, 64, 128]" = torch.ops.aten.view.default(bmm_38, [4, 12, 64, 128]);  bmm_38 = None
    view_377: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_39, [4, 12, 128, 64]);  bmm_39 = None
    permute_248: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_376, [0, 1, 3, 2]);  view_376 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_249: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_373, [0, 2, 1, 3]);  view_373 = None
    clone_122: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
    view_378: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_122, [4, 128, 768]);  clone_122 = None
    view_379: "f32[512, 768]" = torch.ops.aten.view.default(view_378, [512, 768]);  view_378 = None
    permute_250: "f32[768, 768]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
    mm_42: "f32[512, 768]" = torch.ops.aten.mm.default(view_379, permute_250);  permute_250 = None
    permute_251: "f32[768, 512]" = torch.ops.aten.permute.default(view_379, [1, 0])
    mm_43: "f32[768, 768]" = torch.ops.aten.mm.default(permute_251, view_182);  permute_251 = view_182 = None
    permute_252: "f32[768, 768]" = torch.ops.aten.permute.default(mm_43, [1, 0]);  mm_43 = None
    sum_66: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_379, [0], True);  view_379 = None
    view_380: "f32[768]" = torch.ops.aten.view.default(sum_66, [768]);  sum_66 = None
    permute_253: "f32[768, 768]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
    view_381: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_42, [4, 128, 768]);  mm_42 = None
    permute_254: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(permute_248, [0, 2, 1, 3]);  permute_248 = None
    view_382: "f32[4, 128, 768]" = torch.ops.aten.view.default(permute_254, [4, 128, 768]);  permute_254 = None
    clone_123: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_382, memory_format = torch.contiguous_format);  view_382 = None
    view_383: "f32[512, 768]" = torch.ops.aten.view.default(clone_123, [512, 768]);  clone_123 = None
    permute_255: "f32[768, 768]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
    mm_44: "f32[512, 768]" = torch.ops.aten.mm.default(view_383, permute_255);  permute_255 = None
    permute_256: "f32[768, 512]" = torch.ops.aten.permute.default(view_383, [1, 0])
    mm_45: "f32[768, 768]" = torch.ops.aten.mm.default(permute_256, view_179);  permute_256 = view_179 = None
    permute_257: "f32[768, 768]" = torch.ops.aten.permute.default(mm_45, [1, 0]);  mm_45 = None
    sum_67: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
    view_384: "f32[768]" = torch.ops.aten.view.default(sum_67, [768]);  sum_67 = None
    permute_258: "f32[768, 768]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
    view_385: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_44, [4, 128, 768]);  mm_44 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_121: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(view_381, view_385);  view_381 = view_385 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_259: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_377, [0, 2, 1, 3]);  view_377 = None
    clone_124: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_259, memory_format = torch.contiguous_format);  permute_259 = None
    view_386: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_124, [4, 128, 768]);  clone_124 = None
    view_387: "f32[512, 768]" = torch.ops.aten.view.default(view_386, [512, 768]);  view_386 = None
    permute_260: "f32[768, 768]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
    mm_46: "f32[512, 768]" = torch.ops.aten.mm.default(view_387, permute_260);  permute_260 = None
    permute_261: "f32[768, 512]" = torch.ops.aten.permute.default(view_387, [1, 0])
    mm_47: "f32[768, 768]" = torch.ops.aten.mm.default(permute_261, view_176);  permute_261 = view_176 = None
    permute_262: "f32[768, 768]" = torch.ops.aten.permute.default(mm_47, [1, 0]);  mm_47 = None
    sum_68: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_387, [0], True);  view_387 = None
    view_388: "f32[768]" = torch.ops.aten.view.default(sum_68, [768]);  sum_68 = None
    permute_263: "f32[768, 768]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
    view_389: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_46, [4, 128, 768]);  mm_46 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_122: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_121, view_389);  add_121 = view_389 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_69: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_122, [0, 1], True)
    view_390: "f32[768]" = torch.ops.aten.view.default(sum_69, [768]);  sum_69 = None
    div_87: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_40, add_58);  mul_40 = None
    div_88: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_87, add_58);  div_87 = None
    neg_14: "f32[4, 128, 768]" = torch.ops.aten.neg.default(add_122)
    mul_138: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_14, div_88);  neg_14 = div_88 = None
    div_89: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(add_122, add_58);  add_122 = add_58 = None
    sum_70: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_138, [2], True);  mul_138 = None
    mul_139: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_89, primals_33);  primals_33 = None
    mul_140: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_89, sub_24);  div_89 = sub_24 = None
    sum_71: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_140, [0, 1], True);  mul_140 = None
    view_391: "f32[768]" = torch.ops.aten.view.default(sum_71, [768]);  sum_71 = None
    neg_15: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_139)
    sum_72: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_15, [2], True);  neg_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_123: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_120, mul_139);  add_120 = mul_139 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_47: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_24);  alias_24 = None
    mul_141: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_47, 2)
    div_90: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_70, mul_141);  sum_70 = mul_141 = None
    eq_19: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_47, 0);  alias_47 = None
    scalar_tensor_23: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_23: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_19, scalar_tensor_23, div_90);  eq_19 = scalar_tensor_23 = div_90 = None
    mean_31: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_72, [-1], True)
    sub_47: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_72, mean_31);  clone_72 = mean_31 = None
    mul_142: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_23, 0.002607561929595828);  where_23 = None
    mul_143: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_142, sub_47);  mul_142 = sub_47 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_124: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_123, mul_143);  add_123 = mul_143 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_55: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_72, [4, 128, 768]);  sum_72 = None
    div_91: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_55, 768);  expand_55 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_125: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_124, div_91);  add_124 = div_91 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_392: "f32[512, 768]" = torch.ops.aten.view.default(add_125, [512, 768])
    permute_264: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
    mm_48: "f32[512, 3072]" = torch.ops.aten.mm.default(view_392, permute_264);  permute_264 = None
    permute_265: "f32[768, 512]" = torch.ops.aten.permute.default(view_392, [1, 0])
    mm_49: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_265, view_174);  permute_265 = view_174 = None
    permute_266: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_49, [1, 0]);  mm_49 = None
    sum_73: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_392, [0], True);  view_392 = None
    view_393: "f32[768]" = torch.ops.aten.view.default(sum_73, [768]);  sum_73 = None
    permute_267: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
    view_394: "f32[4, 128, 3072]" = torch.ops.aten.view.default(mm_48, [4, 128, 3072]);  mm_48 = None
    mul_144: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
    erf_16: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_144);  mul_144 = None
    add_126: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
    mul_145: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(add_126, 0.5);  add_126 = None
    mul_146: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, view_173)
    mul_147: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_146, -0.5);  mul_146 = None
    exp_16: "f32[4, 128, 3072]" = torch.ops.aten.exp.default(mul_147);  mul_147 = None
    mul_148: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_16, 0.3989422804014327);  exp_16 = None
    mul_149: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, mul_148);  view_173 = mul_148 = None
    add_127: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(mul_145, mul_149);  mul_145 = mul_149 = None
    mul_150: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_394, add_127);  view_394 = add_127 = None
    view_395: "f32[512, 3072]" = torch.ops.aten.view.default(mul_150, [512, 3072]);  mul_150 = None
    permute_268: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
    mm_50: "f32[512, 768]" = torch.ops.aten.mm.default(view_395, permute_268);  permute_268 = None
    permute_269: "f32[3072, 512]" = torch.ops.aten.permute.default(view_395, [1, 0])
    mm_51: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_269, view_172);  permute_269 = view_172 = None
    permute_270: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_51, [1, 0]);  mm_51 = None
    sum_74: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_395, [0], True);  view_395 = None
    view_396: "f32[3072]" = torch.ops.aten.view.default(sum_74, [3072]);  sum_74 = None
    permute_271: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_270, [1, 0]);  permute_270 = None
    view_397: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_50, [4, 128, 768]);  mm_50 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_75: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_397, [0, 1], True)
    view_398: "f32[768]" = torch.ops.aten.view.default(sum_75, [768]);  sum_75 = None
    div_92: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_36, add_54);  mul_36 = None
    div_93: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_92, add_54);  div_92 = None
    neg_16: "f32[4, 128, 768]" = torch.ops.aten.neg.default(view_397)
    mul_151: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_16, div_93);  neg_16 = div_93 = None
    div_94: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(view_397, add_54);  view_397 = add_54 = None
    sum_76: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_151, [2], True);  mul_151 = None
    mul_152: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_94, primals_31);  primals_31 = None
    mul_153: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_94, sub_23);  div_94 = sub_23 = None
    sum_77: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_153, [0, 1], True);  mul_153 = None
    view_399: "f32[768]" = torch.ops.aten.view.default(sum_77, [768]);  sum_77 = None
    neg_17: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_152)
    sum_78: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_17, [2], True);  neg_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_128: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_125, mul_152);  add_125 = mul_152 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_48: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_23);  alias_23 = None
    mul_154: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_48, 2)
    div_95: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_76, mul_154);  sum_76 = mul_154 = None
    eq_20: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_48, 0);  alias_48 = None
    scalar_tensor_24: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_24: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_20, scalar_tensor_24, div_95);  eq_20 = scalar_tensor_24 = div_95 = None
    mean_32: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_53, [-1], True)
    sub_48: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_53, mean_32);  add_53 = mean_32 = None
    mul_155: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_24, 0.002607561929595828);  where_24 = None
    mul_156: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_155, sub_48);  mul_155 = sub_48 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_129: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_128, mul_156);  add_128 = mul_156 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_56: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_78, [4, 128, 768]);  sum_78 = None
    div_96: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_56, 768);  expand_56 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_130: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_129, div_96);  add_129 = div_96 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_400: "f32[512, 768]" = torch.ops.aten.view.default(add_130, [512, 768])
    permute_272: "f32[768, 768]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
    mm_52: "f32[512, 768]" = torch.ops.aten.mm.default(view_400, permute_272);  permute_272 = None
    permute_273: "f32[768, 512]" = torch.ops.aten.permute.default(view_400, [1, 0])
    mm_53: "f32[768, 768]" = torch.ops.aten.mm.default(permute_273, view_170);  permute_273 = view_170 = None
    permute_274: "f32[768, 768]" = torch.ops.aten.permute.default(mm_53, [1, 0]);  mm_53 = None
    sum_79: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_400, [0], True);  view_400 = None
    view_401: "f32[768]" = torch.ops.aten.view.default(sum_79, [768]);  sum_79 = None
    permute_275: "f32[768, 768]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
    view_402: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_52, [4, 128, 768]);  mm_52 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    view_403: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_402, [4, 128, 12, 64]);  view_402 = None
    permute_276: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    clone_125: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(permute_276, memory_format = torch.contiguous_format);  permute_276 = None
    view_404: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_125, [48, 128, 64]);  clone_125 = None
    permute_277: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_166, [0, 2, 1]);  view_166 = None
    bmm_40: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(permute_277, view_404);  permute_277 = None
    permute_278: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_167, [0, 2, 1]);  view_167 = None
    bmm_41: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_404, permute_278);  view_404 = permute_278 = None
    view_405: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_40, [4, 12, 128, 64]);  bmm_40 = None
    view_406: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_41, [4, 12, 128, 128]);  bmm_41 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    alias_49: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(alias_22);  alias_22 = None
    mul_157: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_406, alias_49);  view_406 = None
    sum_80: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_157, [-1], True)
    mul_158: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_49, sum_80);  alias_49 = sum_80 = None
    sub_49: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_157, mul_158);  mul_157 = mul_158 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    scalar_tensor_25: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_25: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_7, scalar_tensor_25, sub_49);  eq_7 = scalar_tensor_25 = sub_49 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    div_97: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_25, 8.0);  where_25 = None
    view_407: "f32[48, 128, 128]" = torch.ops.aten.view.default(div_97, [48, 128, 128]);  div_97 = None
    permute_279: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
    bmm_42: "f32[48, 64, 128]" = torch.ops.aten.bmm.default(permute_279, view_407);  permute_279 = None
    permute_280: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
    bmm_43: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_407, permute_280);  view_407 = permute_280 = None
    view_408: "f32[4, 12, 64, 128]" = torch.ops.aten.view.default(bmm_42, [4, 12, 64, 128]);  bmm_42 = None
    view_409: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_43, [4, 12, 128, 64]);  bmm_43 = None
    permute_281: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_408, [0, 1, 3, 2]);  view_408 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_282: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_405, [0, 2, 1, 3]);  view_405 = None
    clone_126: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_282, memory_format = torch.contiguous_format);  permute_282 = None
    view_410: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_126, [4, 128, 768]);  clone_126 = None
    view_411: "f32[512, 768]" = torch.ops.aten.view.default(view_410, [512, 768]);  view_410 = None
    permute_283: "f32[768, 768]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
    mm_54: "f32[512, 768]" = torch.ops.aten.mm.default(view_411, permute_283);  permute_283 = None
    permute_284: "f32[768, 512]" = torch.ops.aten.permute.default(view_411, [1, 0])
    mm_55: "f32[768, 768]" = torch.ops.aten.mm.default(permute_284, view_160);  permute_284 = view_160 = None
    permute_285: "f32[768, 768]" = torch.ops.aten.permute.default(mm_55, [1, 0]);  mm_55 = None
    sum_81: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_411, [0], True);  view_411 = None
    view_412: "f32[768]" = torch.ops.aten.view.default(sum_81, [768]);  sum_81 = None
    permute_286: "f32[768, 768]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
    view_413: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_54, [4, 128, 768]);  mm_54 = None
    permute_287: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(permute_281, [0, 2, 1, 3]);  permute_281 = None
    view_414: "f32[4, 128, 768]" = torch.ops.aten.view.default(permute_287, [4, 128, 768]);  permute_287 = None
    clone_127: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_414, memory_format = torch.contiguous_format);  view_414 = None
    view_415: "f32[512, 768]" = torch.ops.aten.view.default(clone_127, [512, 768]);  clone_127 = None
    permute_288: "f32[768, 768]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
    mm_56: "f32[512, 768]" = torch.ops.aten.mm.default(view_415, permute_288);  permute_288 = None
    permute_289: "f32[768, 512]" = torch.ops.aten.permute.default(view_415, [1, 0])
    mm_57: "f32[768, 768]" = torch.ops.aten.mm.default(permute_289, view_157);  permute_289 = view_157 = None
    permute_290: "f32[768, 768]" = torch.ops.aten.permute.default(mm_57, [1, 0]);  mm_57 = None
    sum_82: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_415, [0], True);  view_415 = None
    view_416: "f32[768]" = torch.ops.aten.view.default(sum_82, [768]);  sum_82 = None
    permute_291: "f32[768, 768]" = torch.ops.aten.permute.default(permute_290, [1, 0]);  permute_290 = None
    view_417: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_56, [4, 128, 768]);  mm_56 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_131: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(view_413, view_417);  view_413 = view_417 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_292: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_409, [0, 2, 1, 3]);  view_409 = None
    clone_128: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
    view_418: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_128, [4, 128, 768]);  clone_128 = None
    view_419: "f32[512, 768]" = torch.ops.aten.view.default(view_418, [512, 768]);  view_418 = None
    permute_293: "f32[768, 768]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
    mm_58: "f32[512, 768]" = torch.ops.aten.mm.default(view_419, permute_293);  permute_293 = None
    permute_294: "f32[768, 512]" = torch.ops.aten.permute.default(view_419, [1, 0])
    mm_59: "f32[768, 768]" = torch.ops.aten.mm.default(permute_294, view_154);  permute_294 = view_154 = None
    permute_295: "f32[768, 768]" = torch.ops.aten.permute.default(mm_59, [1, 0]);  mm_59 = None
    sum_83: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_419, [0], True);  view_419 = None
    view_420: "f32[768]" = torch.ops.aten.view.default(sum_83, [768]);  sum_83 = None
    permute_296: "f32[768, 768]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
    view_421: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_58, [4, 128, 768]);  mm_58 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_132: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_131, view_421);  add_131 = view_421 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_84: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_132, [0, 1], True)
    view_422: "f32[768]" = torch.ops.aten.view.default(sum_84, [768]);  sum_84 = None
    div_98: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_35, add_51);  mul_35 = None
    div_99: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_98, add_51);  div_98 = None
    neg_18: "f32[4, 128, 768]" = torch.ops.aten.neg.default(add_132)
    mul_159: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_18, div_99);  neg_18 = div_99 = None
    div_100: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(add_132, add_51);  add_132 = add_51 = None
    sum_85: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_159, [2], True);  mul_159 = None
    mul_160: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_100, primals_29);  primals_29 = None
    mul_161: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_100, sub_21);  div_100 = sub_21 = None
    sum_86: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_161, [0, 1], True);  mul_161 = None
    view_423: "f32[768]" = torch.ops.aten.view.default(sum_86, [768]);  sum_86 = None
    neg_19: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_160)
    sum_87: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_19, [2], True);  neg_19 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_133: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_130, mul_160);  add_130 = mul_160 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_50: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_21);  alias_21 = None
    mul_162: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_50, 2)
    div_101: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_85, mul_162);  sum_85 = mul_162 = None
    eq_21: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_50, 0);  alias_50 = None
    scalar_tensor_26: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_26: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_21, scalar_tensor_26, div_101);  eq_21 = scalar_tensor_26 = div_101 = None
    mean_33: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_63, [-1], True)
    sub_50: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_63, mean_33);  clone_63 = mean_33 = None
    mul_163: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_26, 0.002607561929595828);  where_26 = None
    mul_164: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_163, sub_50);  mul_163 = sub_50 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_134: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_133, mul_164);  add_133 = mul_164 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_57: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_87, [4, 128, 768]);  sum_87 = None
    div_102: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_57, 768);  expand_57 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_135: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_134, div_102);  add_134 = div_102 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_424: "f32[512, 768]" = torch.ops.aten.view.default(add_135, [512, 768])
    permute_297: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
    mm_60: "f32[512, 3072]" = torch.ops.aten.mm.default(view_424, permute_297);  permute_297 = None
    permute_298: "f32[768, 512]" = torch.ops.aten.permute.default(view_424, [1, 0])
    mm_61: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_298, view_152);  permute_298 = view_152 = None
    permute_299: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_61, [1, 0]);  mm_61 = None
    sum_88: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_424, [0], True);  view_424 = None
    view_425: "f32[768]" = torch.ops.aten.view.default(sum_88, [768]);  sum_88 = None
    permute_300: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
    view_426: "f32[4, 128, 3072]" = torch.ops.aten.view.default(mm_60, [4, 128, 3072]);  mm_60 = None
    mul_165: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
    erf_17: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_165);  mul_165 = None
    add_136: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
    mul_166: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(add_136, 0.5);  add_136 = None
    mul_167: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_151, view_151)
    mul_168: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_167, -0.5);  mul_167 = None
    exp_17: "f32[4, 128, 3072]" = torch.ops.aten.exp.default(mul_168);  mul_168 = None
    mul_169: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_17, 0.3989422804014327);  exp_17 = None
    mul_170: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_151, mul_169);  view_151 = mul_169 = None
    add_137: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(mul_166, mul_170);  mul_166 = mul_170 = None
    mul_171: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_426, add_137);  view_426 = add_137 = None
    view_427: "f32[512, 3072]" = torch.ops.aten.view.default(mul_171, [512, 3072]);  mul_171 = None
    permute_301: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
    mm_62: "f32[512, 768]" = torch.ops.aten.mm.default(view_427, permute_301);  permute_301 = None
    permute_302: "f32[3072, 512]" = torch.ops.aten.permute.default(view_427, [1, 0])
    mm_63: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_302, view_150);  permute_302 = view_150 = None
    permute_303: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_63, [1, 0]);  mm_63 = None
    sum_89: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_427, [0], True);  view_427 = None
    view_428: "f32[3072]" = torch.ops.aten.view.default(sum_89, [3072]);  sum_89 = None
    permute_304: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_303, [1, 0]);  permute_303 = None
    view_429: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_62, [4, 128, 768]);  mm_62 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_90: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_429, [0, 1], True)
    view_430: "f32[768]" = torch.ops.aten.view.default(sum_90, [768]);  sum_90 = None
    div_103: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_31, add_47);  mul_31 = None
    div_104: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_103, add_47);  div_103 = None
    neg_20: "f32[4, 128, 768]" = torch.ops.aten.neg.default(view_429)
    mul_172: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_20, div_104);  neg_20 = div_104 = None
    div_105: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(view_429, add_47);  view_429 = add_47 = None
    sum_91: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_172, [2], True);  mul_172 = None
    mul_173: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_105, primals_27);  primals_27 = None
    mul_174: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_105, sub_20);  div_105 = sub_20 = None
    sum_92: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_174, [0, 1], True);  mul_174 = None
    view_431: "f32[768]" = torch.ops.aten.view.default(sum_92, [768]);  sum_92 = None
    neg_21: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_173)
    sum_93: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_21, [2], True);  neg_21 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_138: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_135, mul_173);  add_135 = mul_173 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_51: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_20);  alias_20 = None
    mul_175: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_51, 2)
    div_106: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_91, mul_175);  sum_91 = mul_175 = None
    eq_22: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_51, 0);  alias_51 = None
    scalar_tensor_27: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_27: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_22, scalar_tensor_27, div_106);  eq_22 = scalar_tensor_27 = div_106 = None
    mean_34: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_46, [-1], True)
    sub_51: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_46, mean_34);  add_46 = mean_34 = None
    mul_176: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_27, 0.002607561929595828);  where_27 = None
    mul_177: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_176, sub_51);  mul_176 = sub_51 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_139: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_138, mul_177);  add_138 = mul_177 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_58: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_93, [4, 128, 768]);  sum_93 = None
    div_107: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_58, 768);  expand_58 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_140: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_139, div_107);  add_139 = div_107 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_432: "f32[512, 768]" = torch.ops.aten.view.default(add_140, [512, 768])
    permute_305: "f32[768, 768]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
    mm_64: "f32[512, 768]" = torch.ops.aten.mm.default(view_432, permute_305);  permute_305 = None
    permute_306: "f32[768, 512]" = torch.ops.aten.permute.default(view_432, [1, 0])
    mm_65: "f32[768, 768]" = torch.ops.aten.mm.default(permute_306, view_148);  permute_306 = view_148 = None
    permute_307: "f32[768, 768]" = torch.ops.aten.permute.default(mm_65, [1, 0]);  mm_65 = None
    sum_94: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_432, [0], True);  view_432 = None
    view_433: "f32[768]" = torch.ops.aten.view.default(sum_94, [768]);  sum_94 = None
    permute_308: "f32[768, 768]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
    view_434: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_64, [4, 128, 768]);  mm_64 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    view_435: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_434, [4, 128, 12, 64]);  view_434 = None
    permute_309: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_435, [0, 2, 1, 3]);  view_435 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    clone_129: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
    view_436: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_129, [48, 128, 64]);  clone_129 = None
    permute_310: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
    bmm_44: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(permute_310, view_436);  permute_310 = None
    permute_311: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
    bmm_45: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_436, permute_311);  view_436 = permute_311 = None
    view_437: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_44, [4, 12, 128, 64]);  bmm_44 = None
    view_438: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_45, [4, 12, 128, 128]);  bmm_45 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    alias_52: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(alias_19);  alias_19 = None
    mul_178: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_438, alias_52);  view_438 = None
    sum_95: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_178, [-1], True)
    mul_179: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_52, sum_95);  alias_52 = sum_95 = None
    sub_52: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_178, mul_179);  mul_178 = mul_179 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    scalar_tensor_28: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_28: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_6, scalar_tensor_28, sub_52);  eq_6 = scalar_tensor_28 = sub_52 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    div_108: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_28, 8.0);  where_28 = None
    view_439: "f32[48, 128, 128]" = torch.ops.aten.view.default(div_108, [48, 128, 128]);  div_108 = None
    permute_312: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None
    bmm_46: "f32[48, 64, 128]" = torch.ops.aten.bmm.default(permute_312, view_439);  permute_312 = None
    permute_313: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_142, [0, 2, 1]);  view_142 = None
    bmm_47: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_439, permute_313);  view_439 = permute_313 = None
    view_440: "f32[4, 12, 64, 128]" = torch.ops.aten.view.default(bmm_46, [4, 12, 64, 128]);  bmm_46 = None
    view_441: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_47, [4, 12, 128, 64]);  bmm_47 = None
    permute_314: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_440, [0, 1, 3, 2]);  view_440 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_315: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None
    clone_130: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_315, memory_format = torch.contiguous_format);  permute_315 = None
    view_442: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_130, [4, 128, 768]);  clone_130 = None
    view_443: "f32[512, 768]" = torch.ops.aten.view.default(view_442, [512, 768]);  view_442 = None
    permute_316: "f32[768, 768]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
    mm_66: "f32[512, 768]" = torch.ops.aten.mm.default(view_443, permute_316);  permute_316 = None
    permute_317: "f32[768, 512]" = torch.ops.aten.permute.default(view_443, [1, 0])
    mm_67: "f32[768, 768]" = torch.ops.aten.mm.default(permute_317, view_138);  permute_317 = view_138 = None
    permute_318: "f32[768, 768]" = torch.ops.aten.permute.default(mm_67, [1, 0]);  mm_67 = None
    sum_96: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_443, [0], True);  view_443 = None
    view_444: "f32[768]" = torch.ops.aten.view.default(sum_96, [768]);  sum_96 = None
    permute_319: "f32[768, 768]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
    view_445: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_66, [4, 128, 768]);  mm_66 = None
    permute_320: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(permute_314, [0, 2, 1, 3]);  permute_314 = None
    view_446: "f32[4, 128, 768]" = torch.ops.aten.view.default(permute_320, [4, 128, 768]);  permute_320 = None
    clone_131: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_446, memory_format = torch.contiguous_format);  view_446 = None
    view_447: "f32[512, 768]" = torch.ops.aten.view.default(clone_131, [512, 768]);  clone_131 = None
    permute_321: "f32[768, 768]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
    mm_68: "f32[512, 768]" = torch.ops.aten.mm.default(view_447, permute_321);  permute_321 = None
    permute_322: "f32[768, 512]" = torch.ops.aten.permute.default(view_447, [1, 0])
    mm_69: "f32[768, 768]" = torch.ops.aten.mm.default(permute_322, view_135);  permute_322 = view_135 = None
    permute_323: "f32[768, 768]" = torch.ops.aten.permute.default(mm_69, [1, 0]);  mm_69 = None
    sum_97: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_447, [0], True);  view_447 = None
    view_448: "f32[768]" = torch.ops.aten.view.default(sum_97, [768]);  sum_97 = None
    permute_324: "f32[768, 768]" = torch.ops.aten.permute.default(permute_323, [1, 0]);  permute_323 = None
    view_449: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_68, [4, 128, 768]);  mm_68 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_141: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(view_445, view_449);  view_445 = view_449 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_325: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_441, [0, 2, 1, 3]);  view_441 = None
    clone_132: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_325, memory_format = torch.contiguous_format);  permute_325 = None
    view_450: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_132, [4, 128, 768]);  clone_132 = None
    view_451: "f32[512, 768]" = torch.ops.aten.view.default(view_450, [512, 768]);  view_450 = None
    permute_326: "f32[768, 768]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
    mm_70: "f32[512, 768]" = torch.ops.aten.mm.default(view_451, permute_326);  permute_326 = None
    permute_327: "f32[768, 512]" = torch.ops.aten.permute.default(view_451, [1, 0])
    mm_71: "f32[768, 768]" = torch.ops.aten.mm.default(permute_327, view_132);  permute_327 = view_132 = None
    permute_328: "f32[768, 768]" = torch.ops.aten.permute.default(mm_71, [1, 0]);  mm_71 = None
    sum_98: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_451, [0], True);  view_451 = None
    view_452: "f32[768]" = torch.ops.aten.view.default(sum_98, [768]);  sum_98 = None
    permute_329: "f32[768, 768]" = torch.ops.aten.permute.default(permute_328, [1, 0]);  permute_328 = None
    view_453: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_70, [4, 128, 768]);  mm_70 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_142: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_141, view_453);  add_141 = view_453 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_99: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_142, [0, 1], True)
    view_454: "f32[768]" = torch.ops.aten.view.default(sum_99, [768]);  sum_99 = None
    div_109: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_30, add_44);  mul_30 = None
    div_110: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_109, add_44);  div_109 = None
    neg_22: "f32[4, 128, 768]" = torch.ops.aten.neg.default(add_142)
    mul_180: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_22, div_110);  neg_22 = div_110 = None
    div_111: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(add_142, add_44);  add_142 = add_44 = None
    sum_100: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_180, [2], True);  mul_180 = None
    mul_181: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_111, primals_25);  primals_25 = None
    mul_182: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_111, sub_18);  div_111 = sub_18 = None
    sum_101: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_182, [0, 1], True);  mul_182 = None
    view_455: "f32[768]" = torch.ops.aten.view.default(sum_101, [768]);  sum_101 = None
    neg_23: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_181)
    sum_102: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_23, [2], True);  neg_23 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_143: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_140, mul_181);  add_140 = mul_181 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_53: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_18);  alias_18 = None
    mul_183: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_53, 2)
    div_112: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_100, mul_183);  sum_100 = mul_183 = None
    eq_23: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_53, 0);  alias_53 = None
    scalar_tensor_29: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_29: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_23, scalar_tensor_29, div_112);  eq_23 = scalar_tensor_29 = div_112 = None
    mean_35: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_54, [-1], True)
    sub_53: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_54, mean_35);  clone_54 = mean_35 = None
    mul_184: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_29, 0.002607561929595828);  where_29 = None
    mul_185: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_184, sub_53);  mul_184 = sub_53 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_144: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_143, mul_185);  add_143 = mul_185 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_59: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_102, [4, 128, 768]);  sum_102 = None
    div_113: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_59, 768);  expand_59 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_145: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_144, div_113);  add_144 = div_113 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_456: "f32[512, 768]" = torch.ops.aten.view.default(add_145, [512, 768])
    permute_330: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
    mm_72: "f32[512, 3072]" = torch.ops.aten.mm.default(view_456, permute_330);  permute_330 = None
    permute_331: "f32[768, 512]" = torch.ops.aten.permute.default(view_456, [1, 0])
    mm_73: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_331, view_130);  permute_331 = view_130 = None
    permute_332: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_73, [1, 0]);  mm_73 = None
    sum_103: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_456, [0], True);  view_456 = None
    view_457: "f32[768]" = torch.ops.aten.view.default(sum_103, [768]);  sum_103 = None
    permute_333: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
    view_458: "f32[4, 128, 3072]" = torch.ops.aten.view.default(mm_72, [4, 128, 3072]);  mm_72 = None
    mul_186: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
    erf_18: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_186);  mul_186 = None
    add_146: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
    mul_187: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(add_146, 0.5);  add_146 = None
    mul_188: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, view_129)
    mul_189: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_188, -0.5);  mul_188 = None
    exp_18: "f32[4, 128, 3072]" = torch.ops.aten.exp.default(mul_189);  mul_189 = None
    mul_190: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_18, 0.3989422804014327);  exp_18 = None
    mul_191: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, mul_190);  view_129 = mul_190 = None
    add_147: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(mul_187, mul_191);  mul_187 = mul_191 = None
    mul_192: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_458, add_147);  view_458 = add_147 = None
    view_459: "f32[512, 3072]" = torch.ops.aten.view.default(mul_192, [512, 3072]);  mul_192 = None
    permute_334: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
    mm_74: "f32[512, 768]" = torch.ops.aten.mm.default(view_459, permute_334);  permute_334 = None
    permute_335: "f32[3072, 512]" = torch.ops.aten.permute.default(view_459, [1, 0])
    mm_75: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_335, view_128);  permute_335 = view_128 = None
    permute_336: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_75, [1, 0]);  mm_75 = None
    sum_104: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_459, [0], True);  view_459 = None
    view_460: "f32[3072]" = torch.ops.aten.view.default(sum_104, [3072]);  sum_104 = None
    permute_337: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
    view_461: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_74, [4, 128, 768]);  mm_74 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_105: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_461, [0, 1], True)
    view_462: "f32[768]" = torch.ops.aten.view.default(sum_105, [768]);  sum_105 = None
    div_114: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_26, add_40);  mul_26 = None
    div_115: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_114, add_40);  div_114 = None
    neg_24: "f32[4, 128, 768]" = torch.ops.aten.neg.default(view_461)
    mul_193: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_24, div_115);  neg_24 = div_115 = None
    div_116: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(view_461, add_40);  view_461 = add_40 = None
    sum_106: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_193, [2], True);  mul_193 = None
    mul_194: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_116, primals_23);  primals_23 = None
    mul_195: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_116, sub_17);  div_116 = sub_17 = None
    sum_107: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_195, [0, 1], True);  mul_195 = None
    view_463: "f32[768]" = torch.ops.aten.view.default(sum_107, [768]);  sum_107 = None
    neg_25: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_194)
    sum_108: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_25, [2], True);  neg_25 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_148: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_145, mul_194);  add_145 = mul_194 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_54: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_17);  alias_17 = None
    mul_196: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_54, 2)
    div_117: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_106, mul_196);  sum_106 = mul_196 = None
    eq_24: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_54, 0);  alias_54 = None
    scalar_tensor_30: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_30: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_24, scalar_tensor_30, div_117);  eq_24 = scalar_tensor_30 = div_117 = None
    mean_36: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_39, [-1], True)
    sub_54: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_39, mean_36);  add_39 = mean_36 = None
    mul_197: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_30, 0.002607561929595828);  where_30 = None
    mul_198: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_197, sub_54);  mul_197 = sub_54 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_149: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_148, mul_198);  add_148 = mul_198 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_60: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_108, [4, 128, 768]);  sum_108 = None
    div_118: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_60, 768);  expand_60 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_150: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_149, div_118);  add_149 = div_118 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_464: "f32[512, 768]" = torch.ops.aten.view.default(add_150, [512, 768])
    permute_338: "f32[768, 768]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
    mm_76: "f32[512, 768]" = torch.ops.aten.mm.default(view_464, permute_338);  permute_338 = None
    permute_339: "f32[768, 512]" = torch.ops.aten.permute.default(view_464, [1, 0])
    mm_77: "f32[768, 768]" = torch.ops.aten.mm.default(permute_339, view_126);  permute_339 = view_126 = None
    permute_340: "f32[768, 768]" = torch.ops.aten.permute.default(mm_77, [1, 0]);  mm_77 = None
    sum_109: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_464, [0], True);  view_464 = None
    view_465: "f32[768]" = torch.ops.aten.view.default(sum_109, [768]);  sum_109 = None
    permute_341: "f32[768, 768]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
    view_466: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_76, [4, 128, 768]);  mm_76 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    view_467: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_466, [4, 128, 12, 64]);  view_466 = None
    permute_342: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_467, [0, 2, 1, 3]);  view_467 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    clone_133: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(permute_342, memory_format = torch.contiguous_format);  permute_342 = None
    view_468: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_133, [48, 128, 64]);  clone_133 = None
    permute_343: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_122, [0, 2, 1]);  view_122 = None
    bmm_48: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(permute_343, view_468);  permute_343 = None
    permute_344: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
    bmm_49: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_468, permute_344);  view_468 = permute_344 = None
    view_469: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_48, [4, 12, 128, 64]);  bmm_48 = None
    view_470: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_49, [4, 12, 128, 128]);  bmm_49 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    alias_55: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(alias_16);  alias_16 = None
    mul_199: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_470, alias_55);  view_470 = None
    sum_110: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_199, [-1], True)
    mul_200: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_55, sum_110);  alias_55 = sum_110 = None
    sub_55: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_199, mul_200);  mul_199 = mul_200 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    scalar_tensor_31: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_31: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_5, scalar_tensor_31, sub_55);  eq_5 = scalar_tensor_31 = sub_55 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    div_119: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_31, 8.0);  where_31 = None
    view_471: "f32[48, 128, 128]" = torch.ops.aten.view.default(div_119, [48, 128, 128]);  div_119 = None
    permute_345: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
    bmm_50: "f32[48, 64, 128]" = torch.ops.aten.bmm.default(permute_345, view_471);  permute_345 = None
    permute_346: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_120, [0, 2, 1]);  view_120 = None
    bmm_51: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_471, permute_346);  view_471 = permute_346 = None
    view_472: "f32[4, 12, 64, 128]" = torch.ops.aten.view.default(bmm_50, [4, 12, 64, 128]);  bmm_50 = None
    view_473: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_51, [4, 12, 128, 64]);  bmm_51 = None
    permute_347: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_472, [0, 1, 3, 2]);  view_472 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_348: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_469, [0, 2, 1, 3]);  view_469 = None
    clone_134: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_348, memory_format = torch.contiguous_format);  permute_348 = None
    view_474: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_134, [4, 128, 768]);  clone_134 = None
    view_475: "f32[512, 768]" = torch.ops.aten.view.default(view_474, [512, 768]);  view_474 = None
    permute_349: "f32[768, 768]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
    mm_78: "f32[512, 768]" = torch.ops.aten.mm.default(view_475, permute_349);  permute_349 = None
    permute_350: "f32[768, 512]" = torch.ops.aten.permute.default(view_475, [1, 0])
    mm_79: "f32[768, 768]" = torch.ops.aten.mm.default(permute_350, view_116);  permute_350 = view_116 = None
    permute_351: "f32[768, 768]" = torch.ops.aten.permute.default(mm_79, [1, 0]);  mm_79 = None
    sum_111: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
    view_476: "f32[768]" = torch.ops.aten.view.default(sum_111, [768]);  sum_111 = None
    permute_352: "f32[768, 768]" = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
    view_477: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_78, [4, 128, 768]);  mm_78 = None
    permute_353: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(permute_347, [0, 2, 1, 3]);  permute_347 = None
    view_478: "f32[4, 128, 768]" = torch.ops.aten.view.default(permute_353, [4, 128, 768]);  permute_353 = None
    clone_135: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_478, memory_format = torch.contiguous_format);  view_478 = None
    view_479: "f32[512, 768]" = torch.ops.aten.view.default(clone_135, [512, 768]);  clone_135 = None
    permute_354: "f32[768, 768]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
    mm_80: "f32[512, 768]" = torch.ops.aten.mm.default(view_479, permute_354);  permute_354 = None
    permute_355: "f32[768, 512]" = torch.ops.aten.permute.default(view_479, [1, 0])
    mm_81: "f32[768, 768]" = torch.ops.aten.mm.default(permute_355, view_113);  permute_355 = view_113 = None
    permute_356: "f32[768, 768]" = torch.ops.aten.permute.default(mm_81, [1, 0]);  mm_81 = None
    sum_112: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_479, [0], True);  view_479 = None
    view_480: "f32[768]" = torch.ops.aten.view.default(sum_112, [768]);  sum_112 = None
    permute_357: "f32[768, 768]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
    view_481: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_80, [4, 128, 768]);  mm_80 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_151: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(view_477, view_481);  view_477 = view_481 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_358: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_473, [0, 2, 1, 3]);  view_473 = None
    clone_136: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_358, memory_format = torch.contiguous_format);  permute_358 = None
    view_482: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_136, [4, 128, 768]);  clone_136 = None
    view_483: "f32[512, 768]" = torch.ops.aten.view.default(view_482, [512, 768]);  view_482 = None
    permute_359: "f32[768, 768]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
    mm_82: "f32[512, 768]" = torch.ops.aten.mm.default(view_483, permute_359);  permute_359 = None
    permute_360: "f32[768, 512]" = torch.ops.aten.permute.default(view_483, [1, 0])
    mm_83: "f32[768, 768]" = torch.ops.aten.mm.default(permute_360, view_110);  permute_360 = view_110 = None
    permute_361: "f32[768, 768]" = torch.ops.aten.permute.default(mm_83, [1, 0]);  mm_83 = None
    sum_113: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_483, [0], True);  view_483 = None
    view_484: "f32[768]" = torch.ops.aten.view.default(sum_113, [768]);  sum_113 = None
    permute_362: "f32[768, 768]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
    view_485: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_82, [4, 128, 768]);  mm_82 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_152: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_151, view_485);  add_151 = view_485 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_114: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_152, [0, 1], True)
    view_486: "f32[768]" = torch.ops.aten.view.default(sum_114, [768]);  sum_114 = None
    div_120: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_25, add_37);  mul_25 = None
    div_121: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_120, add_37);  div_120 = None
    neg_26: "f32[4, 128, 768]" = torch.ops.aten.neg.default(add_152)
    mul_201: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_26, div_121);  neg_26 = div_121 = None
    div_122: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(add_152, add_37);  add_152 = add_37 = None
    sum_115: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_201, [2], True);  mul_201 = None
    mul_202: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_122, primals_21);  primals_21 = None
    mul_203: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_122, sub_15);  div_122 = sub_15 = None
    sum_116: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_203, [0, 1], True);  mul_203 = None
    view_487: "f32[768]" = torch.ops.aten.view.default(sum_116, [768]);  sum_116 = None
    neg_27: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_202)
    sum_117: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_27, [2], True);  neg_27 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_153: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_150, mul_202);  add_150 = mul_202 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_56: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_15);  alias_15 = None
    mul_204: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_56, 2)
    div_123: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_115, mul_204);  sum_115 = mul_204 = None
    eq_25: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_56, 0);  alias_56 = None
    scalar_tensor_32: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_32: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_25, scalar_tensor_32, div_123);  eq_25 = scalar_tensor_32 = div_123 = None
    mean_37: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_45, [-1], True)
    sub_56: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_45, mean_37);  clone_45 = mean_37 = None
    mul_205: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_32, 0.002607561929595828);  where_32 = None
    mul_206: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_205, sub_56);  mul_205 = sub_56 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_154: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_153, mul_206);  add_153 = mul_206 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_61: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_117, [4, 128, 768]);  sum_117 = None
    div_124: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_61, 768);  expand_61 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_155: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_154, div_124);  add_154 = div_124 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_488: "f32[512, 768]" = torch.ops.aten.view.default(add_155, [512, 768])
    permute_363: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
    mm_84: "f32[512, 3072]" = torch.ops.aten.mm.default(view_488, permute_363);  permute_363 = None
    permute_364: "f32[768, 512]" = torch.ops.aten.permute.default(view_488, [1, 0])
    mm_85: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_364, view_108);  permute_364 = view_108 = None
    permute_365: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_85, [1, 0]);  mm_85 = None
    sum_118: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_488, [0], True);  view_488 = None
    view_489: "f32[768]" = torch.ops.aten.view.default(sum_118, [768]);  sum_118 = None
    permute_366: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_365, [1, 0]);  permute_365 = None
    view_490: "f32[4, 128, 3072]" = torch.ops.aten.view.default(mm_84, [4, 128, 3072]);  mm_84 = None
    mul_207: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
    erf_19: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
    add_156: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
    mul_208: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(add_156, 0.5);  add_156 = None
    mul_209: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, view_107)
    mul_210: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_209, -0.5);  mul_209 = None
    exp_19: "f32[4, 128, 3072]" = torch.ops.aten.exp.default(mul_210);  mul_210 = None
    mul_211: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_19, 0.3989422804014327);  exp_19 = None
    mul_212: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, mul_211);  view_107 = mul_211 = None
    add_157: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(mul_208, mul_212);  mul_208 = mul_212 = None
    mul_213: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_490, add_157);  view_490 = add_157 = None
    view_491: "f32[512, 3072]" = torch.ops.aten.view.default(mul_213, [512, 3072]);  mul_213 = None
    permute_367: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
    mm_86: "f32[512, 768]" = torch.ops.aten.mm.default(view_491, permute_367);  permute_367 = None
    permute_368: "f32[3072, 512]" = torch.ops.aten.permute.default(view_491, [1, 0])
    mm_87: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_368, view_106);  permute_368 = view_106 = None
    permute_369: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_87, [1, 0]);  mm_87 = None
    sum_119: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_491, [0], True);  view_491 = None
    view_492: "f32[3072]" = torch.ops.aten.view.default(sum_119, [3072]);  sum_119 = None
    permute_370: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_369, [1, 0]);  permute_369 = None
    view_493: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_86, [4, 128, 768]);  mm_86 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_120: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_493, [0, 1], True)
    view_494: "f32[768]" = torch.ops.aten.view.default(sum_120, [768]);  sum_120 = None
    div_125: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_21, add_33);  mul_21 = None
    div_126: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_125, add_33);  div_125 = None
    neg_28: "f32[4, 128, 768]" = torch.ops.aten.neg.default(view_493)
    mul_214: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_28, div_126);  neg_28 = div_126 = None
    div_127: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(view_493, add_33);  view_493 = add_33 = None
    sum_121: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_214, [2], True);  mul_214 = None
    mul_215: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_127, primals_19);  primals_19 = None
    mul_216: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_127, sub_14);  div_127 = sub_14 = None
    sum_122: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_216, [0, 1], True);  mul_216 = None
    view_495: "f32[768]" = torch.ops.aten.view.default(sum_122, [768]);  sum_122 = None
    neg_29: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_215)
    sum_123: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_29, [2], True);  neg_29 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_158: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_155, mul_215);  add_155 = mul_215 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_57: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_14);  alias_14 = None
    mul_217: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_57, 2)
    div_128: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_121, mul_217);  sum_121 = mul_217 = None
    eq_26: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_57, 0);  alias_57 = None
    scalar_tensor_33: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_33: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_26, scalar_tensor_33, div_128);  eq_26 = scalar_tensor_33 = div_128 = None
    mean_38: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_32, [-1], True)
    sub_57: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_32, mean_38);  add_32 = mean_38 = None
    mul_218: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_33, 0.002607561929595828);  where_33 = None
    mul_219: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_218, sub_57);  mul_218 = sub_57 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_159: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_158, mul_219);  add_158 = mul_219 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_62: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_123, [4, 128, 768]);  sum_123 = None
    div_129: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_62, 768);  expand_62 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_160: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_159, div_129);  add_159 = div_129 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_496: "f32[512, 768]" = torch.ops.aten.view.default(add_160, [512, 768])
    permute_371: "f32[768, 768]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
    mm_88: "f32[512, 768]" = torch.ops.aten.mm.default(view_496, permute_371);  permute_371 = None
    permute_372: "f32[768, 512]" = torch.ops.aten.permute.default(view_496, [1, 0])
    mm_89: "f32[768, 768]" = torch.ops.aten.mm.default(permute_372, view_104);  permute_372 = view_104 = None
    permute_373: "f32[768, 768]" = torch.ops.aten.permute.default(mm_89, [1, 0]);  mm_89 = None
    sum_124: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_496, [0], True);  view_496 = None
    view_497: "f32[768]" = torch.ops.aten.view.default(sum_124, [768]);  sum_124 = None
    permute_374: "f32[768, 768]" = torch.ops.aten.permute.default(permute_373, [1, 0]);  permute_373 = None
    view_498: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_88, [4, 128, 768]);  mm_88 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    view_499: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_498, [4, 128, 12, 64]);  view_498 = None
    permute_375: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_499, [0, 2, 1, 3]);  view_499 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    clone_137: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(permute_375, memory_format = torch.contiguous_format);  permute_375 = None
    view_500: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_137, [48, 128, 64]);  clone_137 = None
    permute_376: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_100, [0, 2, 1]);  view_100 = None
    bmm_52: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(permute_376, view_500);  permute_376 = None
    permute_377: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_101, [0, 2, 1]);  view_101 = None
    bmm_53: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_500, permute_377);  view_500 = permute_377 = None
    view_501: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_52, [4, 12, 128, 64]);  bmm_52 = None
    view_502: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_53, [4, 12, 128, 128]);  bmm_53 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    alias_58: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(alias_13);  alias_13 = None
    mul_220: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_502, alias_58);  view_502 = None
    sum_125: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_220, [-1], True)
    mul_221: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_58, sum_125);  alias_58 = sum_125 = None
    sub_58: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_220, mul_221);  mul_220 = mul_221 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    scalar_tensor_34: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_34: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_4, scalar_tensor_34, sub_58);  eq_4 = scalar_tensor_34 = sub_58 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    div_130: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_34, 8.0);  where_34 = None
    view_503: "f32[48, 128, 128]" = torch.ops.aten.view.default(div_130, [48, 128, 128]);  div_130 = None
    permute_378: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None
    bmm_54: "f32[48, 64, 128]" = torch.ops.aten.bmm.default(permute_378, view_503);  permute_378 = None
    permute_379: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_98, [0, 2, 1]);  view_98 = None
    bmm_55: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_503, permute_379);  view_503 = permute_379 = None
    view_504: "f32[4, 12, 64, 128]" = torch.ops.aten.view.default(bmm_54, [4, 12, 64, 128]);  bmm_54 = None
    view_505: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_55, [4, 12, 128, 64]);  bmm_55 = None
    permute_380: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_504, [0, 1, 3, 2]);  view_504 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_381: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_501, [0, 2, 1, 3]);  view_501 = None
    clone_138: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_381, memory_format = torch.contiguous_format);  permute_381 = None
    view_506: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_138, [4, 128, 768]);  clone_138 = None
    view_507: "f32[512, 768]" = torch.ops.aten.view.default(view_506, [512, 768]);  view_506 = None
    permute_382: "f32[768, 768]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
    mm_90: "f32[512, 768]" = torch.ops.aten.mm.default(view_507, permute_382);  permute_382 = None
    permute_383: "f32[768, 512]" = torch.ops.aten.permute.default(view_507, [1, 0])
    mm_91: "f32[768, 768]" = torch.ops.aten.mm.default(permute_383, view_94);  permute_383 = view_94 = None
    permute_384: "f32[768, 768]" = torch.ops.aten.permute.default(mm_91, [1, 0]);  mm_91 = None
    sum_126: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_507, [0], True);  view_507 = None
    view_508: "f32[768]" = torch.ops.aten.view.default(sum_126, [768]);  sum_126 = None
    permute_385: "f32[768, 768]" = torch.ops.aten.permute.default(permute_384, [1, 0]);  permute_384 = None
    view_509: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_90, [4, 128, 768]);  mm_90 = None
    permute_386: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(permute_380, [0, 2, 1, 3]);  permute_380 = None
    view_510: "f32[4, 128, 768]" = torch.ops.aten.view.default(permute_386, [4, 128, 768]);  permute_386 = None
    clone_139: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_510, memory_format = torch.contiguous_format);  view_510 = None
    view_511: "f32[512, 768]" = torch.ops.aten.view.default(clone_139, [512, 768]);  clone_139 = None
    permute_387: "f32[768, 768]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
    mm_92: "f32[512, 768]" = torch.ops.aten.mm.default(view_511, permute_387);  permute_387 = None
    permute_388: "f32[768, 512]" = torch.ops.aten.permute.default(view_511, [1, 0])
    mm_93: "f32[768, 768]" = torch.ops.aten.mm.default(permute_388, view_91);  permute_388 = view_91 = None
    permute_389: "f32[768, 768]" = torch.ops.aten.permute.default(mm_93, [1, 0]);  mm_93 = None
    sum_127: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_511, [0], True);  view_511 = None
    view_512: "f32[768]" = torch.ops.aten.view.default(sum_127, [768]);  sum_127 = None
    permute_390: "f32[768, 768]" = torch.ops.aten.permute.default(permute_389, [1, 0]);  permute_389 = None
    view_513: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_92, [4, 128, 768]);  mm_92 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_161: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(view_509, view_513);  view_509 = view_513 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_391: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_505, [0, 2, 1, 3]);  view_505 = None
    clone_140: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_391, memory_format = torch.contiguous_format);  permute_391 = None
    view_514: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_140, [4, 128, 768]);  clone_140 = None
    view_515: "f32[512, 768]" = torch.ops.aten.view.default(view_514, [512, 768]);  view_514 = None
    permute_392: "f32[768, 768]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
    mm_94: "f32[512, 768]" = torch.ops.aten.mm.default(view_515, permute_392);  permute_392 = None
    permute_393: "f32[768, 512]" = torch.ops.aten.permute.default(view_515, [1, 0])
    mm_95: "f32[768, 768]" = torch.ops.aten.mm.default(permute_393, view_88);  permute_393 = view_88 = None
    permute_394: "f32[768, 768]" = torch.ops.aten.permute.default(mm_95, [1, 0]);  mm_95 = None
    sum_128: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_515, [0], True);  view_515 = None
    view_516: "f32[768]" = torch.ops.aten.view.default(sum_128, [768]);  sum_128 = None
    permute_395: "f32[768, 768]" = torch.ops.aten.permute.default(permute_394, [1, 0]);  permute_394 = None
    view_517: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_94, [4, 128, 768]);  mm_94 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_162: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_161, view_517);  add_161 = view_517 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_129: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_162, [0, 1], True)
    view_518: "f32[768]" = torch.ops.aten.view.default(sum_129, [768]);  sum_129 = None
    div_131: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_20, add_30);  mul_20 = None
    div_132: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_131, add_30);  div_131 = None
    neg_30: "f32[4, 128, 768]" = torch.ops.aten.neg.default(add_162)
    mul_222: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_30, div_132);  neg_30 = div_132 = None
    div_133: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(add_162, add_30);  add_162 = add_30 = None
    sum_130: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_222, [2], True);  mul_222 = None
    mul_223: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_133, primals_17);  primals_17 = None
    mul_224: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_133, sub_12);  div_133 = sub_12 = None
    sum_131: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_224, [0, 1], True);  mul_224 = None
    view_519: "f32[768]" = torch.ops.aten.view.default(sum_131, [768]);  sum_131 = None
    neg_31: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_223)
    sum_132: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_31, [2], True);  neg_31 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_163: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_160, mul_223);  add_160 = mul_223 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_59: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_12);  alias_12 = None
    mul_225: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_59, 2)
    div_134: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_130, mul_225);  sum_130 = mul_225 = None
    eq_27: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_59, 0);  alias_59 = None
    scalar_tensor_35: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_35: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_27, scalar_tensor_35, div_134);  eq_27 = scalar_tensor_35 = div_134 = None
    mean_39: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_36, [-1], True)
    sub_59: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_36, mean_39);  clone_36 = mean_39 = None
    mul_226: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_35, 0.002607561929595828);  where_35 = None
    mul_227: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_226, sub_59);  mul_226 = sub_59 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_164: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_163, mul_227);  add_163 = mul_227 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_63: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_132, [4, 128, 768]);  sum_132 = None
    div_135: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_63, 768);  expand_63 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_165: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_164, div_135);  add_164 = div_135 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_520: "f32[512, 768]" = torch.ops.aten.view.default(add_165, [512, 768])
    permute_396: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
    mm_96: "f32[512, 3072]" = torch.ops.aten.mm.default(view_520, permute_396);  permute_396 = None
    permute_397: "f32[768, 512]" = torch.ops.aten.permute.default(view_520, [1, 0])
    mm_97: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_397, view_86);  permute_397 = view_86 = None
    permute_398: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_97, [1, 0]);  mm_97 = None
    sum_133: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_520, [0], True);  view_520 = None
    view_521: "f32[768]" = torch.ops.aten.view.default(sum_133, [768]);  sum_133 = None
    permute_399: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
    view_522: "f32[4, 128, 3072]" = torch.ops.aten.view.default(mm_96, [4, 128, 3072]);  mm_96 = None
    mul_228: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
    erf_20: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_228);  mul_228 = None
    add_166: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
    mul_229: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(add_166, 0.5);  add_166 = None
    mul_230: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, view_85)
    mul_231: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_230, -0.5);  mul_230 = None
    exp_20: "f32[4, 128, 3072]" = torch.ops.aten.exp.default(mul_231);  mul_231 = None
    mul_232: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_20, 0.3989422804014327);  exp_20 = None
    mul_233: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, mul_232);  view_85 = mul_232 = None
    add_167: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(mul_229, mul_233);  mul_229 = mul_233 = None
    mul_234: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_522, add_167);  view_522 = add_167 = None
    view_523: "f32[512, 3072]" = torch.ops.aten.view.default(mul_234, [512, 3072]);  mul_234 = None
    permute_400: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
    mm_98: "f32[512, 768]" = torch.ops.aten.mm.default(view_523, permute_400);  permute_400 = None
    permute_401: "f32[3072, 512]" = torch.ops.aten.permute.default(view_523, [1, 0])
    mm_99: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_401, view_84);  permute_401 = view_84 = None
    permute_402: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_99, [1, 0]);  mm_99 = None
    sum_134: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_523, [0], True);  view_523 = None
    view_524: "f32[3072]" = torch.ops.aten.view.default(sum_134, [3072]);  sum_134 = None
    permute_403: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
    view_525: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_98, [4, 128, 768]);  mm_98 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_135: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_525, [0, 1], True)
    view_526: "f32[768]" = torch.ops.aten.view.default(sum_135, [768]);  sum_135 = None
    div_136: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_16, add_26);  mul_16 = None
    div_137: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_136, add_26);  div_136 = None
    neg_32: "f32[4, 128, 768]" = torch.ops.aten.neg.default(view_525)
    mul_235: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_32, div_137);  neg_32 = div_137 = None
    div_138: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(view_525, add_26);  view_525 = add_26 = None
    sum_136: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_235, [2], True);  mul_235 = None
    mul_236: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_138, primals_15);  primals_15 = None
    mul_237: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_138, sub_11);  div_138 = sub_11 = None
    sum_137: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_237, [0, 1], True);  mul_237 = None
    view_527: "f32[768]" = torch.ops.aten.view.default(sum_137, [768]);  sum_137 = None
    neg_33: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_236)
    sum_138: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_33, [2], True);  neg_33 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_168: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_165, mul_236);  add_165 = mul_236 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_60: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_11);  alias_11 = None
    mul_238: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_60, 2)
    div_139: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_136, mul_238);  sum_136 = mul_238 = None
    eq_28: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_60, 0);  alias_60 = None
    scalar_tensor_36: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_36: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_28, scalar_tensor_36, div_139);  eq_28 = scalar_tensor_36 = div_139 = None
    mean_40: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_25, [-1], True)
    sub_60: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_25, mean_40);  add_25 = mean_40 = None
    mul_239: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_36, 0.002607561929595828);  where_36 = None
    mul_240: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_239, sub_60);  mul_239 = sub_60 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_169: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_168, mul_240);  add_168 = mul_240 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_64: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_138, [4, 128, 768]);  sum_138 = None
    div_140: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_64, 768);  expand_64 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_170: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_169, div_140);  add_169 = div_140 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_528: "f32[512, 768]" = torch.ops.aten.view.default(add_170, [512, 768])
    permute_404: "f32[768, 768]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
    mm_100: "f32[512, 768]" = torch.ops.aten.mm.default(view_528, permute_404);  permute_404 = None
    permute_405: "f32[768, 512]" = torch.ops.aten.permute.default(view_528, [1, 0])
    mm_101: "f32[768, 768]" = torch.ops.aten.mm.default(permute_405, view_82);  permute_405 = view_82 = None
    permute_406: "f32[768, 768]" = torch.ops.aten.permute.default(mm_101, [1, 0]);  mm_101 = None
    sum_139: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_528, [0], True);  view_528 = None
    view_529: "f32[768]" = torch.ops.aten.view.default(sum_139, [768]);  sum_139 = None
    permute_407: "f32[768, 768]" = torch.ops.aten.permute.default(permute_406, [1, 0]);  permute_406 = None
    view_530: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_100, [4, 128, 768]);  mm_100 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    view_531: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_530, [4, 128, 12, 64]);  view_530 = None
    permute_408: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_531, [0, 2, 1, 3]);  view_531 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    clone_141: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(permute_408, memory_format = torch.contiguous_format);  permute_408 = None
    view_532: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_141, [48, 128, 64]);  clone_141 = None
    permute_409: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_78, [0, 2, 1]);  view_78 = None
    bmm_56: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(permute_409, view_532);  permute_409 = None
    permute_410: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_79, [0, 2, 1]);  view_79 = None
    bmm_57: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_532, permute_410);  view_532 = permute_410 = None
    view_533: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_56, [4, 12, 128, 64]);  bmm_56 = None
    view_534: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_57, [4, 12, 128, 128]);  bmm_57 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    alias_61: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(alias_10);  alias_10 = None
    mul_241: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_534, alias_61);  view_534 = None
    sum_140: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_241, [-1], True)
    mul_242: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_61, sum_140);  alias_61 = sum_140 = None
    sub_61: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_241, mul_242);  mul_241 = mul_242 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    scalar_tensor_37: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_37: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_3, scalar_tensor_37, sub_61);  eq_3 = scalar_tensor_37 = sub_61 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    div_141: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_37, 8.0);  where_37 = None
    view_535: "f32[48, 128, 128]" = torch.ops.aten.view.default(div_141, [48, 128, 128]);  div_141 = None
    permute_411: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_75, [0, 2, 1]);  view_75 = None
    bmm_58: "f32[48, 64, 128]" = torch.ops.aten.bmm.default(permute_411, view_535);  permute_411 = None
    permute_412: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
    bmm_59: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_535, permute_412);  view_535 = permute_412 = None
    view_536: "f32[4, 12, 64, 128]" = torch.ops.aten.view.default(bmm_58, [4, 12, 64, 128]);  bmm_58 = None
    view_537: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_59, [4, 12, 128, 64]);  bmm_59 = None
    permute_413: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_536, [0, 1, 3, 2]);  view_536 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_414: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_533, [0, 2, 1, 3]);  view_533 = None
    clone_142: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_414, memory_format = torch.contiguous_format);  permute_414 = None
    view_538: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_142, [4, 128, 768]);  clone_142 = None
    view_539: "f32[512, 768]" = torch.ops.aten.view.default(view_538, [512, 768]);  view_538 = None
    permute_415: "f32[768, 768]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
    mm_102: "f32[512, 768]" = torch.ops.aten.mm.default(view_539, permute_415);  permute_415 = None
    permute_416: "f32[768, 512]" = torch.ops.aten.permute.default(view_539, [1, 0])
    mm_103: "f32[768, 768]" = torch.ops.aten.mm.default(permute_416, view_72);  permute_416 = view_72 = None
    permute_417: "f32[768, 768]" = torch.ops.aten.permute.default(mm_103, [1, 0]);  mm_103 = None
    sum_141: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_539, [0], True);  view_539 = None
    view_540: "f32[768]" = torch.ops.aten.view.default(sum_141, [768]);  sum_141 = None
    permute_418: "f32[768, 768]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
    view_541: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_102, [4, 128, 768]);  mm_102 = None
    permute_419: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(permute_413, [0, 2, 1, 3]);  permute_413 = None
    view_542: "f32[4, 128, 768]" = torch.ops.aten.view.default(permute_419, [4, 128, 768]);  permute_419 = None
    clone_143: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_542, memory_format = torch.contiguous_format);  view_542 = None
    view_543: "f32[512, 768]" = torch.ops.aten.view.default(clone_143, [512, 768]);  clone_143 = None
    permute_420: "f32[768, 768]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
    mm_104: "f32[512, 768]" = torch.ops.aten.mm.default(view_543, permute_420);  permute_420 = None
    permute_421: "f32[768, 512]" = torch.ops.aten.permute.default(view_543, [1, 0])
    mm_105: "f32[768, 768]" = torch.ops.aten.mm.default(permute_421, view_69);  permute_421 = view_69 = None
    permute_422: "f32[768, 768]" = torch.ops.aten.permute.default(mm_105, [1, 0]);  mm_105 = None
    sum_142: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_543, [0], True);  view_543 = None
    view_544: "f32[768]" = torch.ops.aten.view.default(sum_142, [768]);  sum_142 = None
    permute_423: "f32[768, 768]" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
    view_545: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_104, [4, 128, 768]);  mm_104 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_171: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(view_541, view_545);  view_541 = view_545 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_424: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_537, [0, 2, 1, 3]);  view_537 = None
    clone_144: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_424, memory_format = torch.contiguous_format);  permute_424 = None
    view_546: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_144, [4, 128, 768]);  clone_144 = None
    view_547: "f32[512, 768]" = torch.ops.aten.view.default(view_546, [512, 768]);  view_546 = None
    permute_425: "f32[768, 768]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
    mm_106: "f32[512, 768]" = torch.ops.aten.mm.default(view_547, permute_425);  permute_425 = None
    permute_426: "f32[768, 512]" = torch.ops.aten.permute.default(view_547, [1, 0])
    mm_107: "f32[768, 768]" = torch.ops.aten.mm.default(permute_426, view_66);  permute_426 = view_66 = None
    permute_427: "f32[768, 768]" = torch.ops.aten.permute.default(mm_107, [1, 0]);  mm_107 = None
    sum_143: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
    view_548: "f32[768]" = torch.ops.aten.view.default(sum_143, [768]);  sum_143 = None
    permute_428: "f32[768, 768]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
    view_549: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_106, [4, 128, 768]);  mm_106 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_172: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_171, view_549);  add_171 = view_549 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_144: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_172, [0, 1], True)
    view_550: "f32[768]" = torch.ops.aten.view.default(sum_144, [768]);  sum_144 = None
    div_142: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_15, add_23);  mul_15 = None
    div_143: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_142, add_23);  div_142 = None
    neg_34: "f32[4, 128, 768]" = torch.ops.aten.neg.default(add_172)
    mul_243: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_34, div_143);  neg_34 = div_143 = None
    div_144: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(add_172, add_23);  add_172 = add_23 = None
    sum_145: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True);  mul_243 = None
    mul_244: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_144, primals_13);  primals_13 = None
    mul_245: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_144, sub_9);  div_144 = sub_9 = None
    sum_146: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_245, [0, 1], True);  mul_245 = None
    view_551: "f32[768]" = torch.ops.aten.view.default(sum_146, [768]);  sum_146 = None
    neg_35: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_244)
    sum_147: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_35, [2], True);  neg_35 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_173: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_170, mul_244);  add_170 = mul_244 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_62: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_9);  alias_9 = None
    mul_246: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_62, 2)
    div_145: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_145, mul_246);  sum_145 = mul_246 = None
    eq_29: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_62, 0);  alias_62 = None
    scalar_tensor_38: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_38: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_29, scalar_tensor_38, div_145);  eq_29 = scalar_tensor_38 = div_145 = None
    mean_41: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_27, [-1], True)
    sub_62: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_27, mean_41);  clone_27 = mean_41 = None
    mul_247: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_38, 0.002607561929595828);  where_38 = None
    mul_248: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_247, sub_62);  mul_247 = sub_62 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_174: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_173, mul_248);  add_173 = mul_248 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_65: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_147, [4, 128, 768]);  sum_147 = None
    div_146: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_65, 768);  expand_65 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_175: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_174, div_146);  add_174 = div_146 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_552: "f32[512, 768]" = torch.ops.aten.view.default(add_175, [512, 768])
    permute_429: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
    mm_108: "f32[512, 3072]" = torch.ops.aten.mm.default(view_552, permute_429);  permute_429 = None
    permute_430: "f32[768, 512]" = torch.ops.aten.permute.default(view_552, [1, 0])
    mm_109: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_430, view_64);  permute_430 = view_64 = None
    permute_431: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_109, [1, 0]);  mm_109 = None
    sum_148: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_552, [0], True);  view_552 = None
    view_553: "f32[768]" = torch.ops.aten.view.default(sum_148, [768]);  sum_148 = None
    permute_432: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
    view_554: "f32[4, 128, 3072]" = torch.ops.aten.view.default(mm_108, [4, 128, 3072]);  mm_108 = None
    mul_249: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
    erf_21: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_249);  mul_249 = None
    add_176: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
    mul_250: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(add_176, 0.5);  add_176 = None
    mul_251: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, view_63)
    mul_252: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_251, -0.5);  mul_251 = None
    exp_21: "f32[4, 128, 3072]" = torch.ops.aten.exp.default(mul_252);  mul_252 = None
    mul_253: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_21, 0.3989422804014327);  exp_21 = None
    mul_254: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, mul_253);  view_63 = mul_253 = None
    add_177: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(mul_250, mul_254);  mul_250 = mul_254 = None
    mul_255: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_554, add_177);  view_554 = add_177 = None
    view_555: "f32[512, 3072]" = torch.ops.aten.view.default(mul_255, [512, 3072]);  mul_255 = None
    permute_433: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
    mm_110: "f32[512, 768]" = torch.ops.aten.mm.default(view_555, permute_433);  permute_433 = None
    permute_434: "f32[3072, 512]" = torch.ops.aten.permute.default(view_555, [1, 0])
    mm_111: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_434, view_62);  permute_434 = view_62 = None
    permute_435: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_111, [1, 0]);  mm_111 = None
    sum_149: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_555, [0], True);  view_555 = None
    view_556: "f32[3072]" = torch.ops.aten.view.default(sum_149, [3072]);  sum_149 = None
    permute_436: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_435, [1, 0]);  permute_435 = None
    view_557: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_110, [4, 128, 768]);  mm_110 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_150: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_557, [0, 1], True)
    view_558: "f32[768]" = torch.ops.aten.view.default(sum_150, [768]);  sum_150 = None
    div_147: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_11, add_19);  mul_11 = None
    div_148: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_147, add_19);  div_147 = None
    neg_36: "f32[4, 128, 768]" = torch.ops.aten.neg.default(view_557)
    mul_256: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_36, div_148);  neg_36 = div_148 = None
    div_149: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(view_557, add_19);  view_557 = add_19 = None
    sum_151: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_256, [2], True);  mul_256 = None
    mul_257: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_149, primals_11);  primals_11 = None
    mul_258: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_149, sub_8);  div_149 = sub_8 = None
    sum_152: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_258, [0, 1], True);  mul_258 = None
    view_559: "f32[768]" = torch.ops.aten.view.default(sum_152, [768]);  sum_152 = None
    neg_37: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_257)
    sum_153: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_37, [2], True);  neg_37 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_178: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_175, mul_257);  add_175 = mul_257 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_63: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_8);  alias_8 = None
    mul_259: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_63, 2)
    div_150: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_151, mul_259);  sum_151 = mul_259 = None
    eq_30: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_63, 0);  alias_63 = None
    scalar_tensor_39: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_39: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_30, scalar_tensor_39, div_150);  eq_30 = scalar_tensor_39 = div_150 = None
    mean_42: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_18, [-1], True)
    sub_63: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_18, mean_42);  add_18 = mean_42 = None
    mul_260: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_39, 0.002607561929595828);  where_39 = None
    mul_261: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_260, sub_63);  mul_260 = sub_63 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_179: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_178, mul_261);  add_178 = mul_261 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_66: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_153, [4, 128, 768]);  sum_153 = None
    div_151: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_66, 768);  expand_66 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_180: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_179, div_151);  add_179 = div_151 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_560: "f32[512, 768]" = torch.ops.aten.view.default(add_180, [512, 768])
    permute_437: "f32[768, 768]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
    mm_112: "f32[512, 768]" = torch.ops.aten.mm.default(view_560, permute_437);  permute_437 = None
    permute_438: "f32[768, 512]" = torch.ops.aten.permute.default(view_560, [1, 0])
    mm_113: "f32[768, 768]" = torch.ops.aten.mm.default(permute_438, view_60);  permute_438 = view_60 = None
    permute_439: "f32[768, 768]" = torch.ops.aten.permute.default(mm_113, [1, 0]);  mm_113 = None
    sum_154: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_560, [0], True);  view_560 = None
    view_561: "f32[768]" = torch.ops.aten.view.default(sum_154, [768]);  sum_154 = None
    permute_440: "f32[768, 768]" = torch.ops.aten.permute.default(permute_439, [1, 0]);  permute_439 = None
    view_562: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_112, [4, 128, 768]);  mm_112 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    view_563: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_562, [4, 128, 12, 64]);  view_562 = None
    permute_441: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_563, [0, 2, 1, 3]);  view_563 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    clone_145: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(permute_441, memory_format = torch.contiguous_format);  permute_441 = None
    view_564: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_145, [48, 128, 64]);  clone_145 = None
    permute_442: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_56, [0, 2, 1]);  view_56 = None
    bmm_60: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(permute_442, view_564);  permute_442 = None
    permute_443: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_57, [0, 2, 1]);  view_57 = None
    bmm_61: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_564, permute_443);  view_564 = permute_443 = None
    view_565: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_60, [4, 12, 128, 64]);  bmm_60 = None
    view_566: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_61, [4, 12, 128, 128]);  bmm_61 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    alias_64: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(alias_7);  alias_7 = None
    mul_262: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_566, alias_64);  view_566 = None
    sum_155: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_262, [-1], True)
    mul_263: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_64, sum_155);  alias_64 = sum_155 = None
    sub_64: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_262, mul_263);  mul_262 = mul_263 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    scalar_tensor_40: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_40: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_2, scalar_tensor_40, sub_64);  eq_2 = scalar_tensor_40 = sub_64 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    div_152: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_40, 8.0);  where_40 = None
    view_567: "f32[48, 128, 128]" = torch.ops.aten.view.default(div_152, [48, 128, 128]);  div_152 = None
    permute_444: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
    bmm_62: "f32[48, 64, 128]" = torch.ops.aten.bmm.default(permute_444, view_567);  permute_444 = None
    permute_445: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
    bmm_63: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_567, permute_445);  view_567 = permute_445 = None
    view_568: "f32[4, 12, 64, 128]" = torch.ops.aten.view.default(bmm_62, [4, 12, 64, 128]);  bmm_62 = None
    view_569: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_63, [4, 12, 128, 64]);  bmm_63 = None
    permute_446: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_568, [0, 1, 3, 2]);  view_568 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_447: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_565, [0, 2, 1, 3]);  view_565 = None
    clone_146: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_447, memory_format = torch.contiguous_format);  permute_447 = None
    view_570: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_146, [4, 128, 768]);  clone_146 = None
    view_571: "f32[512, 768]" = torch.ops.aten.view.default(view_570, [512, 768]);  view_570 = None
    permute_448: "f32[768, 768]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
    mm_114: "f32[512, 768]" = torch.ops.aten.mm.default(view_571, permute_448);  permute_448 = None
    permute_449: "f32[768, 512]" = torch.ops.aten.permute.default(view_571, [1, 0])
    mm_115: "f32[768, 768]" = torch.ops.aten.mm.default(permute_449, view_50);  permute_449 = view_50 = None
    permute_450: "f32[768, 768]" = torch.ops.aten.permute.default(mm_115, [1, 0]);  mm_115 = None
    sum_156: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_571, [0], True);  view_571 = None
    view_572: "f32[768]" = torch.ops.aten.view.default(sum_156, [768]);  sum_156 = None
    permute_451: "f32[768, 768]" = torch.ops.aten.permute.default(permute_450, [1, 0]);  permute_450 = None
    view_573: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_114, [4, 128, 768]);  mm_114 = None
    permute_452: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(permute_446, [0, 2, 1, 3]);  permute_446 = None
    view_574: "f32[4, 128, 768]" = torch.ops.aten.view.default(permute_452, [4, 128, 768]);  permute_452 = None
    clone_147: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_574, memory_format = torch.contiguous_format);  view_574 = None
    view_575: "f32[512, 768]" = torch.ops.aten.view.default(clone_147, [512, 768]);  clone_147 = None
    permute_453: "f32[768, 768]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
    mm_116: "f32[512, 768]" = torch.ops.aten.mm.default(view_575, permute_453);  permute_453 = None
    permute_454: "f32[768, 512]" = torch.ops.aten.permute.default(view_575, [1, 0])
    mm_117: "f32[768, 768]" = torch.ops.aten.mm.default(permute_454, view_47);  permute_454 = view_47 = None
    permute_455: "f32[768, 768]" = torch.ops.aten.permute.default(mm_117, [1, 0]);  mm_117 = None
    sum_157: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
    view_576: "f32[768]" = torch.ops.aten.view.default(sum_157, [768]);  sum_157 = None
    permute_456: "f32[768, 768]" = torch.ops.aten.permute.default(permute_455, [1, 0]);  permute_455 = None
    view_577: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_116, [4, 128, 768]);  mm_116 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_181: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(view_573, view_577);  view_573 = view_577 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_457: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_569, [0, 2, 1, 3]);  view_569 = None
    clone_148: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_457, memory_format = torch.contiguous_format);  permute_457 = None
    view_578: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_148, [4, 128, 768]);  clone_148 = None
    view_579: "f32[512, 768]" = torch.ops.aten.view.default(view_578, [512, 768]);  view_578 = None
    permute_458: "f32[768, 768]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
    mm_118: "f32[512, 768]" = torch.ops.aten.mm.default(view_579, permute_458);  permute_458 = None
    permute_459: "f32[768, 512]" = torch.ops.aten.permute.default(view_579, [1, 0])
    mm_119: "f32[768, 768]" = torch.ops.aten.mm.default(permute_459, view_44);  permute_459 = view_44 = None
    permute_460: "f32[768, 768]" = torch.ops.aten.permute.default(mm_119, [1, 0]);  mm_119 = None
    sum_158: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_579, [0], True);  view_579 = None
    view_580: "f32[768]" = torch.ops.aten.view.default(sum_158, [768]);  sum_158 = None
    permute_461: "f32[768, 768]" = torch.ops.aten.permute.default(permute_460, [1, 0]);  permute_460 = None
    view_581: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_118, [4, 128, 768]);  mm_118 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_182: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_181, view_581);  add_181 = view_581 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_159: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1], True)
    view_582: "f32[768]" = torch.ops.aten.view.default(sum_159, [768]);  sum_159 = None
    div_153: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_10, add_16);  mul_10 = None
    div_154: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_153, add_16);  div_153 = None
    neg_38: "f32[4, 128, 768]" = torch.ops.aten.neg.default(add_182)
    mul_264: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_38, div_154);  neg_38 = div_154 = None
    div_155: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(add_182, add_16);  add_182 = add_16 = None
    sum_160: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_264, [2], True);  mul_264 = None
    mul_265: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_155, primals_9);  primals_9 = None
    mul_266: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_155, sub_6);  div_155 = sub_6 = None
    sum_161: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_266, [0, 1], True);  mul_266 = None
    view_583: "f32[768]" = torch.ops.aten.view.default(sum_161, [768]);  sum_161 = None
    neg_39: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_265)
    sum_162: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_39, [2], True);  neg_39 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_183: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_180, mul_265);  add_180 = mul_265 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_65: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_6);  alias_6 = None
    mul_267: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_65, 2)
    div_156: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_160, mul_267);  sum_160 = mul_267 = None
    eq_31: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_65, 0);  alias_65 = None
    scalar_tensor_41: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_41: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_31, scalar_tensor_41, div_156);  eq_31 = scalar_tensor_41 = div_156 = None
    mean_43: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_18, [-1], True)
    sub_65: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_18, mean_43);  clone_18 = mean_43 = None
    mul_268: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_41, 0.002607561929595828);  where_41 = None
    mul_269: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_268, sub_65);  mul_268 = sub_65 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_184: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_183, mul_269);  add_183 = mul_269 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_67: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_162, [4, 128, 768]);  sum_162 = None
    div_157: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_67, 768);  expand_67 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_185: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_184, div_157);  add_184 = div_157 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_584: "f32[512, 768]" = torch.ops.aten.view.default(add_185, [512, 768])
    permute_462: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
    mm_120: "f32[512, 3072]" = torch.ops.aten.mm.default(view_584, permute_462);  permute_462 = None
    permute_463: "f32[768, 512]" = torch.ops.aten.permute.default(view_584, [1, 0])
    mm_121: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_463, view_42);  permute_463 = view_42 = None
    permute_464: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_121, [1, 0]);  mm_121 = None
    sum_163: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
    view_585: "f32[768]" = torch.ops.aten.view.default(sum_163, [768]);  sum_163 = None
    permute_465: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_464, [1, 0]);  permute_464 = None
    view_586: "f32[4, 128, 3072]" = torch.ops.aten.view.default(mm_120, [4, 128, 3072]);  mm_120 = None
    mul_270: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
    erf_22: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_270);  mul_270 = None
    add_186: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
    mul_271: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(add_186, 0.5);  add_186 = None
    mul_272: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, view_41)
    mul_273: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_272, -0.5);  mul_272 = None
    exp_22: "f32[4, 128, 3072]" = torch.ops.aten.exp.default(mul_273);  mul_273 = None
    mul_274: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_22, 0.3989422804014327);  exp_22 = None
    mul_275: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, mul_274);  view_41 = mul_274 = None
    add_187: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(mul_271, mul_275);  mul_271 = mul_275 = None
    mul_276: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_586, add_187);  view_586 = add_187 = None
    view_587: "f32[512, 3072]" = torch.ops.aten.view.default(mul_276, [512, 3072]);  mul_276 = None
    permute_466: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
    mm_122: "f32[512, 768]" = torch.ops.aten.mm.default(view_587, permute_466);  permute_466 = None
    permute_467: "f32[3072, 512]" = torch.ops.aten.permute.default(view_587, [1, 0])
    mm_123: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_467, view_40);  permute_467 = view_40 = None
    permute_468: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_123, [1, 0]);  mm_123 = None
    sum_164: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_587, [0], True);  view_587 = None
    view_588: "f32[3072]" = torch.ops.aten.view.default(sum_164, [3072]);  sum_164 = None
    permute_469: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_468, [1, 0]);  permute_468 = None
    view_589: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_122, [4, 128, 768]);  mm_122 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_165: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_589, [0, 1], True)
    view_590: "f32[768]" = torch.ops.aten.view.default(sum_165, [768]);  sum_165 = None
    div_158: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_6, add_12);  mul_6 = None
    div_159: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_158, add_12);  div_158 = None
    neg_40: "f32[4, 128, 768]" = torch.ops.aten.neg.default(view_589)
    mul_277: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_40, div_159);  neg_40 = div_159 = None
    div_160: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(view_589, add_12);  view_589 = add_12 = None
    sum_166: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_277, [2], True);  mul_277 = None
    mul_278: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_160, primals_7);  primals_7 = None
    mul_279: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_160, sub_5);  div_160 = sub_5 = None
    sum_167: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_279, [0, 1], True);  mul_279 = None
    view_591: "f32[768]" = torch.ops.aten.view.default(sum_167, [768]);  sum_167 = None
    neg_41: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_278)
    sum_168: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_41, [2], True);  neg_41 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_188: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_185, mul_278);  add_185 = mul_278 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_66: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_5);  alias_5 = None
    mul_280: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_66, 2)
    div_161: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_166, mul_280);  sum_166 = mul_280 = None
    eq_32: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_66, 0);  alias_66 = None
    scalar_tensor_42: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_42: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_32, scalar_tensor_42, div_161);  eq_32 = scalar_tensor_42 = div_161 = None
    mean_44: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_11, [-1], True)
    sub_66: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_11, mean_44);  add_11 = mean_44 = None
    mul_281: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_42, 0.002607561929595828);  where_42 = None
    mul_282: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_281, sub_66);  mul_281 = sub_66 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_189: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_188, mul_282);  add_188 = mul_282 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_68: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_168, [4, 128, 768]);  sum_168 = None
    div_162: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_68, 768);  expand_68 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_190: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_189, div_162);  add_189 = div_162 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_592: "f32[512, 768]" = torch.ops.aten.view.default(add_190, [512, 768])
    permute_470: "f32[768, 768]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
    mm_124: "f32[512, 768]" = torch.ops.aten.mm.default(view_592, permute_470);  permute_470 = None
    permute_471: "f32[768, 512]" = torch.ops.aten.permute.default(view_592, [1, 0])
    mm_125: "f32[768, 768]" = torch.ops.aten.mm.default(permute_471, view_38);  permute_471 = view_38 = None
    permute_472: "f32[768, 768]" = torch.ops.aten.permute.default(mm_125, [1, 0]);  mm_125 = None
    sum_169: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_592, [0], True);  view_592 = None
    view_593: "f32[768]" = torch.ops.aten.view.default(sum_169, [768]);  sum_169 = None
    permute_473: "f32[768, 768]" = torch.ops.aten.permute.default(permute_472, [1, 0]);  permute_472 = None
    view_594: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_124, [4, 128, 768]);  mm_124 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    view_595: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_594, [4, 128, 12, 64]);  view_594 = None
    permute_474: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_595, [0, 2, 1, 3]);  view_595 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    clone_149: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(permute_474, memory_format = torch.contiguous_format);  permute_474 = None
    view_596: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_149, [48, 128, 64]);  clone_149 = None
    permute_475: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_34, [0, 2, 1]);  view_34 = None
    bmm_64: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(permute_475, view_596);  permute_475 = None
    permute_476: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_35, [0, 2, 1]);  view_35 = None
    bmm_65: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_596, permute_476);  view_596 = permute_476 = None
    view_597: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_64, [4, 12, 128, 64]);  bmm_64 = None
    view_598: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_65, [4, 12, 128, 128]);  bmm_65 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    alias_67: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(alias_4);  alias_4 = None
    mul_283: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_598, alias_67);  view_598 = None
    sum_170: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_283, [-1], True)
    mul_284: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_67, sum_170);  alias_67 = sum_170 = None
    sub_67: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_283, mul_284);  mul_283 = mul_284 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    scalar_tensor_43: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_43: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, scalar_tensor_43, sub_67);  eq_1 = scalar_tensor_43 = sub_67 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    div_163: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_43, 8.0);  where_43 = None
    view_599: "f32[48, 128, 128]" = torch.ops.aten.view.default(div_163, [48, 128, 128]);  div_163 = None
    permute_477: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
    bmm_66: "f32[48, 64, 128]" = torch.ops.aten.bmm.default(permute_477, view_599);  permute_477 = None
    permute_478: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_32, [0, 2, 1]);  view_32 = None
    bmm_67: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_599, permute_478);  view_599 = permute_478 = None
    view_600: "f32[4, 12, 64, 128]" = torch.ops.aten.view.default(bmm_66, [4, 12, 64, 128]);  bmm_66 = None
    view_601: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_67, [4, 12, 128, 64]);  bmm_67 = None
    permute_479: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_600, [0, 1, 3, 2]);  view_600 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_480: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_597, [0, 2, 1, 3]);  view_597 = None
    clone_150: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_480, memory_format = torch.contiguous_format);  permute_480 = None
    view_602: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_150, [4, 128, 768]);  clone_150 = None
    view_603: "f32[512, 768]" = torch.ops.aten.view.default(view_602, [512, 768]);  view_602 = None
    permute_481: "f32[768, 768]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
    mm_126: "f32[512, 768]" = torch.ops.aten.mm.default(view_603, permute_481);  permute_481 = None
    permute_482: "f32[768, 512]" = torch.ops.aten.permute.default(view_603, [1, 0])
    mm_127: "f32[768, 768]" = torch.ops.aten.mm.default(permute_482, view_28);  permute_482 = view_28 = None
    permute_483: "f32[768, 768]" = torch.ops.aten.permute.default(mm_127, [1, 0]);  mm_127 = None
    sum_171: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
    view_604: "f32[768]" = torch.ops.aten.view.default(sum_171, [768]);  sum_171 = None
    permute_484: "f32[768, 768]" = torch.ops.aten.permute.default(permute_483, [1, 0]);  permute_483 = None
    view_605: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_126, [4, 128, 768]);  mm_126 = None
    permute_485: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(permute_479, [0, 2, 1, 3]);  permute_479 = None
    view_606: "f32[4, 128, 768]" = torch.ops.aten.view.default(permute_485, [4, 128, 768]);  permute_485 = None
    clone_151: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_606, memory_format = torch.contiguous_format);  view_606 = None
    view_607: "f32[512, 768]" = torch.ops.aten.view.default(clone_151, [512, 768]);  clone_151 = None
    permute_486: "f32[768, 768]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
    mm_128: "f32[512, 768]" = torch.ops.aten.mm.default(view_607, permute_486);  permute_486 = None
    permute_487: "f32[768, 512]" = torch.ops.aten.permute.default(view_607, [1, 0])
    mm_129: "f32[768, 768]" = torch.ops.aten.mm.default(permute_487, view_25);  permute_487 = view_25 = None
    permute_488: "f32[768, 768]" = torch.ops.aten.permute.default(mm_129, [1, 0]);  mm_129 = None
    sum_172: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_607, [0], True);  view_607 = None
    view_608: "f32[768]" = torch.ops.aten.view.default(sum_172, [768]);  sum_172 = None
    permute_489: "f32[768, 768]" = torch.ops.aten.permute.default(permute_488, [1, 0]);  permute_488 = None
    view_609: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_128, [4, 128, 768]);  mm_128 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_191: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(view_605, view_609);  view_605 = view_609 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_490: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_601, [0, 2, 1, 3]);  view_601 = None
    clone_152: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_490, memory_format = torch.contiguous_format);  permute_490 = None
    view_610: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_152, [4, 128, 768]);  clone_152 = None
    view_611: "f32[512, 768]" = torch.ops.aten.view.default(view_610, [512, 768]);  view_610 = None
    permute_491: "f32[768, 768]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
    mm_130: "f32[512, 768]" = torch.ops.aten.mm.default(view_611, permute_491);  permute_491 = None
    permute_492: "f32[768, 512]" = torch.ops.aten.permute.default(view_611, [1, 0])
    mm_131: "f32[768, 768]" = torch.ops.aten.mm.default(permute_492, view_22);  permute_492 = view_22 = None
    permute_493: "f32[768, 768]" = torch.ops.aten.permute.default(mm_131, [1, 0]);  mm_131 = None
    sum_173: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_611, [0], True);  view_611 = None
    view_612: "f32[768]" = torch.ops.aten.view.default(sum_173, [768]);  sum_173 = None
    permute_494: "f32[768, 768]" = torch.ops.aten.permute.default(permute_493, [1, 0]);  permute_493 = None
    view_613: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_130, [4, 128, 768]);  mm_130 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_192: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_191, view_613);  add_191 = view_613 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_174: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_192, [0, 1], True)
    view_614: "f32[768]" = torch.ops.aten.view.default(sum_174, [768]);  sum_174 = None
    div_164: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_5, add_9);  mul_5 = None
    div_165: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_164, add_9);  div_164 = None
    neg_42: "f32[4, 128, 768]" = torch.ops.aten.neg.default(add_192)
    mul_285: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_42, div_165);  neg_42 = div_165 = None
    div_166: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(add_192, add_9);  add_192 = add_9 = None
    sum_175: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_285, [2], True);  mul_285 = None
    mul_286: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_166, primals_5);  primals_5 = None
    mul_287: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_166, sub_3);  div_166 = sub_3 = None
    sum_176: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_287, [0, 1], True);  mul_287 = None
    view_615: "f32[768]" = torch.ops.aten.view.default(sum_176, [768]);  sum_176 = None
    neg_43: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_286)
    sum_177: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_43, [2], True);  neg_43 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_193: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_190, mul_286);  add_190 = mul_286 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_68: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_3);  alias_3 = None
    mul_288: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_68, 2)
    div_167: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_175, mul_288);  sum_175 = mul_288 = None
    eq_33: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_68, 0);  alias_68 = None
    scalar_tensor_44: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_44: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_33, scalar_tensor_44, div_167);  eq_33 = scalar_tensor_44 = div_167 = None
    mean_45: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_9, [-1], True)
    sub_68: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_9, mean_45);  clone_9 = mean_45 = None
    mul_289: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_44, 0.002607561929595828);  where_44 = None
    mul_290: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_289, sub_68);  mul_289 = sub_68 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_194: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_193, mul_290);  add_193 = mul_290 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_69: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_177, [4, 128, 768]);  sum_177 = None
    div_168: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_69, 768);  expand_69 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_195: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_194, div_168);  add_194 = div_168 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_616: "f32[512, 768]" = torch.ops.aten.view.default(add_195, [512, 768])
    permute_495: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
    mm_132: "f32[512, 3072]" = torch.ops.aten.mm.default(view_616, permute_495);  permute_495 = None
    permute_496: "f32[768, 512]" = torch.ops.aten.permute.default(view_616, [1, 0])
    mm_133: "f32[768, 3072]" = torch.ops.aten.mm.default(permute_496, view_20);  permute_496 = view_20 = None
    permute_497: "f32[3072, 768]" = torch.ops.aten.permute.default(mm_133, [1, 0]);  mm_133 = None
    sum_178: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_616, [0], True);  view_616 = None
    view_617: "f32[768]" = torch.ops.aten.view.default(sum_178, [768]);  sum_178 = None
    permute_498: "f32[768, 3072]" = torch.ops.aten.permute.default(permute_497, [1, 0]);  permute_497 = None
    view_618: "f32[4, 128, 3072]" = torch.ops.aten.view.default(mm_132, [4, 128, 3072]);  mm_132 = None
    mul_291: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
    erf_23: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_291);  mul_291 = None
    add_196: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
    mul_292: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(add_196, 0.5);  add_196 = None
    mul_293: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, view_19)
    mul_294: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_293, -0.5);  mul_293 = None
    exp_23: "f32[4, 128, 3072]" = torch.ops.aten.exp.default(mul_294);  mul_294 = None
    mul_295: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(exp_23, 0.3989422804014327);  exp_23 = None
    mul_296: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, mul_295);  view_19 = mul_295 = None
    add_197: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(mul_292, mul_296);  mul_292 = mul_296 = None
    mul_297: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_618, add_197);  view_618 = add_197 = None
    view_619: "f32[512, 3072]" = torch.ops.aten.view.default(mul_297, [512, 3072]);  mul_297 = None
    permute_499: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
    mm_134: "f32[512, 768]" = torch.ops.aten.mm.default(view_619, permute_499);  permute_499 = None
    permute_500: "f32[3072, 512]" = torch.ops.aten.permute.default(view_619, [1, 0])
    mm_135: "f32[3072, 768]" = torch.ops.aten.mm.default(permute_500, view_18);  permute_500 = view_18 = None
    permute_501: "f32[768, 3072]" = torch.ops.aten.permute.default(mm_135, [1, 0]);  mm_135 = None
    sum_179: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_619, [0], True);  view_619 = None
    view_620: "f32[3072]" = torch.ops.aten.view.default(sum_179, [3072]);  sum_179 = None
    permute_502: "f32[3072, 768]" = torch.ops.aten.permute.default(permute_501, [1, 0]);  permute_501 = None
    view_621: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_134, [4, 128, 768]);  mm_134 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_180: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(view_621, [0, 1], True)
    view_622: "f32[768]" = torch.ops.aten.view.default(sum_180, [768]);  sum_180 = None
    div_169: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_1, add_5);  mul_1 = None
    div_170: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_169, add_5);  div_169 = None
    neg_44: "f32[4, 128, 768]" = torch.ops.aten.neg.default(view_621)
    mul_298: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_44, div_170);  neg_44 = div_170 = None
    div_171: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(view_621, add_5);  view_621 = add_5 = None
    sum_181: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_298, [2], True);  mul_298 = None
    mul_299: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_171, primals_3);  primals_3 = None
    mul_300: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_171, sub_2);  div_171 = sub_2 = None
    sum_182: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_300, [0, 1], True);  mul_300 = None
    view_623: "f32[768]" = torch.ops.aten.view.default(sum_182, [768]);  sum_182 = None
    neg_45: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_299)
    sum_183: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_45, [2], True);  neg_45 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_198: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_195, mul_299);  add_195 = mul_299 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_69: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias_2);  alias_2 = None
    mul_301: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_69, 2)
    div_172: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_181, mul_301);  sum_181 = mul_301 = None
    eq_34: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_69, 0);  alias_69 = None
    scalar_tensor_45: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_45: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_34, scalar_tensor_45, div_172);  eq_34 = scalar_tensor_45 = div_172 = None
    mean_46: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_4, [-1], True)
    sub_69: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_4, mean_46);  add_4 = mean_46 = None
    mul_302: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_45, 0.002607561929595828);  where_45 = None
    mul_303: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_302, sub_69);  mul_302 = sub_69 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_199: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_198, mul_303);  add_198 = mul_303 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_70: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_183, [4, 128, 768]);  sum_183 = None
    div_173: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_70, 768);  expand_70 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_200: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_199, div_173);  add_199 = div_173 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_624: "f32[512, 768]" = torch.ops.aten.view.default(add_200, [512, 768])
    permute_503: "f32[768, 768]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
    mm_136: "f32[512, 768]" = torch.ops.aten.mm.default(view_624, permute_503);  permute_503 = None
    permute_504: "f32[768, 512]" = torch.ops.aten.permute.default(view_624, [1, 0])
    mm_137: "f32[768, 768]" = torch.ops.aten.mm.default(permute_504, view_16);  permute_504 = view_16 = None
    permute_505: "f32[768, 768]" = torch.ops.aten.permute.default(mm_137, [1, 0]);  mm_137 = None
    sum_184: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_624, [0], True);  view_624 = None
    view_625: "f32[768]" = torch.ops.aten.view.default(sum_184, [768]);  sum_184 = None
    permute_506: "f32[768, 768]" = torch.ops.aten.permute.default(permute_505, [1, 0]);  permute_505 = None
    view_626: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_136, [4, 128, 768]);  mm_136 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    view_627: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_626, [4, 128, 12, 64]);  view_626 = None
    permute_507: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_627, [0, 2, 1, 3]);  view_627 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    clone_153: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(permute_507, memory_format = torch.contiguous_format);  permute_507 = None
    view_628: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_153, [48, 128, 64]);  clone_153 = None
    permute_508: "f32[48, 128, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
    bmm_68: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(permute_508, view_628);  permute_508 = None
    permute_509: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
    bmm_69: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_628, permute_509);  view_628 = permute_509 = None
    view_629: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_68, [4, 12, 128, 64]);  bmm_68 = None
    view_630: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_69, [4, 12, 128, 128]);  bmm_69 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    alias_70: "f32[4, 12, 128, 128]" = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    mul_304: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(view_630, alias_70);  view_630 = None
    sum_185: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_304, [-1], True)
    mul_305: "f32[4, 12, 128, 128]" = torch.ops.aten.mul.Tensor(alias_70, sum_185);  alias_70 = sum_185 = None
    sub_70: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(mul_304, mul_305);  mul_304 = mul_305 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    scalar_tensor_46: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_46: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq, scalar_tensor_46, sub_70);  eq = scalar_tensor_46 = sub_70 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:15, code: scores = torch.matmul(query, key.transpose(-2, -1)) \
    div_174: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(where_46, 8.0);  where_46 = None
    view_631: "f32[48, 128, 128]" = torch.ops.aten.view.default(div_174, [48, 128, 128]);  div_174 = None
    permute_510: "f32[48, 64, 128]" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
    bmm_70: "f32[48, 64, 128]" = torch.ops.aten.bmm.default(permute_510, view_631);  permute_510 = None
    permute_511: "f32[48, 128, 64]" = torch.ops.aten.permute.default(view_10, [0, 2, 1]);  view_10 = None
    bmm_71: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_631, permute_511);  view_631 = permute_511 = None
    view_632: "f32[4, 12, 64, 128]" = torch.ops.aten.view.default(bmm_70, [4, 12, 64, 128]);  bmm_70 = None
    view_633: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_71, [4, 12, 128, 64]);  bmm_71 = None
    permute_512: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_632, [0, 1, 3, 2]);  view_632 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_513: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_629, [0, 2, 1, 3]);  view_629 = None
    clone_154: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_513, memory_format = torch.contiguous_format);  permute_513 = None
    view_634: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_154, [4, 128, 768]);  clone_154 = None
    view_635: "f32[512, 768]" = torch.ops.aten.view.default(view_634, [512, 768]);  view_634 = None
    permute_514: "f32[768, 768]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
    mm_138: "f32[512, 768]" = torch.ops.aten.mm.default(view_635, permute_514);  permute_514 = None
    permute_515: "f32[768, 512]" = torch.ops.aten.permute.default(view_635, [1, 0])
    mm_139: "f32[768, 768]" = torch.ops.aten.mm.default(permute_515, view_6);  permute_515 = view_6 = None
    permute_516: "f32[768, 768]" = torch.ops.aten.permute.default(mm_139, [1, 0]);  mm_139 = None
    sum_186: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_635, [0], True);  view_635 = None
    view_636: "f32[768]" = torch.ops.aten.view.default(sum_186, [768]);  sum_186 = None
    permute_517: "f32[768, 768]" = torch.ops.aten.permute.default(permute_516, [1, 0]);  permute_516 = None
    view_637: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_138, [4, 128, 768]);  mm_138 = None
    permute_518: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(permute_512, [0, 2, 1, 3]);  permute_512 = None
    view_638: "f32[4, 128, 768]" = torch.ops.aten.view.default(permute_518, [4, 128, 768]);  permute_518 = None
    clone_155: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_638, memory_format = torch.contiguous_format);  view_638 = None
    view_639: "f32[512, 768]" = torch.ops.aten.view.default(clone_155, [512, 768]);  clone_155 = None
    permute_519: "f32[768, 768]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
    mm_140: "f32[512, 768]" = torch.ops.aten.mm.default(view_639, permute_519);  permute_519 = None
    permute_520: "f32[768, 512]" = torch.ops.aten.permute.default(view_639, [1, 0])
    mm_141: "f32[768, 768]" = torch.ops.aten.mm.default(permute_520, view_3);  permute_520 = view_3 = None
    permute_521: "f32[768, 768]" = torch.ops.aten.permute.default(mm_141, [1, 0]);  mm_141 = None
    sum_187: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_639, [0], True);  view_639 = None
    view_640: "f32[768]" = torch.ops.aten.view.default(sum_187, [768]);  sum_187 = None
    permute_522: "f32[768, 768]" = torch.ops.aten.permute.default(permute_521, [1, 0]);  permute_521 = None
    view_641: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_140, [4, 128, 768]);  mm_140 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_201: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(view_637, view_641);  view_637 = view_641 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    permute_523: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_633, [0, 2, 1, 3]);  view_633 = None
    clone_156: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_523, memory_format = torch.contiguous_format);  permute_523 = None
    view_642: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_156, [4, 128, 768]);  clone_156 = None
    view_643: "f32[512, 768]" = torch.ops.aten.view.default(view_642, [512, 768]);  view_642 = None
    permute_524: "f32[768, 768]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
    mm_142: "f32[512, 768]" = torch.ops.aten.mm.default(view_643, permute_524);  permute_524 = None
    permute_525: "f32[768, 512]" = torch.ops.aten.permute.default(view_643, [1, 0])
    mm_143: "f32[768, 768]" = torch.ops.aten.mm.default(permute_525, view);  permute_525 = view = None
    permute_526: "f32[768, 768]" = torch.ops.aten.permute.default(mm_143, [1, 0]);  mm_143 = None
    sum_188: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_643, [0], True);  view_643 = None
    view_644: "f32[768]" = torch.ops.aten.view.default(sum_188, [768]);  sum_188 = None
    permute_527: "f32[768, 768]" = torch.ops.aten.permute.default(permute_526, [1, 0]);  permute_526 = None
    view_645: "f32[4, 128, 768]" = torch.ops.aten.view.default(mm_142, [4, 128, 768]);  mm_142 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    add_202: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_201, view_645);  add_201 = view_645 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sum_189: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(add_202, [0, 1], True)
    view_646: "f32[768]" = torch.ops.aten.view.default(sum_189, [768]);  sum_189 = None
    div_175: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul, add_2);  mul = None
    div_176: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(div_175, add_2);  div_175 = None
    neg_46: "f32[4, 128, 768]" = torch.ops.aten.neg.default(add_202)
    mul_306: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(neg_46, div_176);  neg_46 = div_176 = None
    div_177: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(add_202, add_2);  add_202 = add_2 = None
    sum_190: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_306, [2], True);  mul_306 = None
    mul_307: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_177, primals_1);  primals_1 = None
    mul_308: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(div_177, sub);  div_177 = sub = None
    sum_191: "f32[1, 1, 768]" = torch.ops.aten.sum.dim_IntList(mul_308, [0, 1], True);  mul_308 = None
    view_647: "f32[768]" = torch.ops.aten.view.default(sum_191, [768]);  sum_191 = None
    neg_47: "f32[4, 128, 768]" = torch.ops.aten.neg.default(mul_307)
    sum_192: "f32[4, 128, 1]" = torch.ops.aten.sum.dim_IntList(neg_47, [2], True);  neg_47 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    add_203: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_200, mul_307);  add_200 = mul_307 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    alias_71: "f32[4, 128, 1]" = torch.ops.aten.alias.default(alias);  alias = None
    mul_309: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(alias_71, 2)
    div_178: "f32[4, 128, 1]" = torch.ops.aten.div.Tensor(sum_190, mul_309);  sum_190 = mul_309 = None
    eq_35: "b8[4, 128, 1]" = torch.ops.aten.eq.Scalar(alias_71, 0);  alias_71 = None
    scalar_tensor_47: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_47: "f32[4, 128, 1]" = torch.ops.aten.where.self(eq_35, scalar_tensor_47, div_178);  eq_35 = scalar_tensor_47 = div_178 = None
    mean_47: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone, [-1], True)
    sub_71: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone, mean_47);  clone = mean_47 = None
    mul_310: "f32[4, 128, 1]" = torch.ops.aten.mul.Scalar(where_47, 0.002607561929595828);  where_47 = None
    mul_311: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(mul_310, sub_71);  mul_310 = sub_71 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    add_204: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_203, mul_311);  add_203 = mul_311 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    expand_71: "f32[4, 128, 768]" = torch.ops.aten.expand.default(sum_192, [4, 128, 768]);  sum_192 = None
    div_179: "f32[4, 128, 768]" = torch.ops.aten.div.Scalar(expand_71, 768);  expand_71 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    add_205: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_204, div_179);  add_204 = div_179 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/module.py:1520, code: return forward_call(*args, **kwargs)
    eq_36: "b8[4, 128]" = torch.ops.aten.eq.Scalar(primals_197, 0)
    unsqueeze_2: "b8[4, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_36, -1);  eq_36 = None
    scalar_tensor_48: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_48: "f32[4, 128, 768]" = torch.ops.aten.where.self(unsqueeze_2, scalar_tensor_48, add_205);  unsqueeze_2 = scalar_tensor_48 = None
    full: "f32[3, 768]" = torch.ops.aten.full.default([3, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put: "f32[3, 768]" = torch.ops.aten._unsafe_index_put.default(full, [primals_197], where_48, True);  full = primals_197 = where_48 = None
    eq_37: "b8[4, 128]" = torch.ops.aten.eq.Scalar(primals_196, 0)
    unsqueeze_3: "b8[4, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_37, -1);  eq_37 = None
    scalar_tensor_49: "f32[]" = torch.ops.aten.scalar_tensor.default(0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'))
    where_49: "f32[4, 128, 768]" = torch.ops.aten.where.self(unsqueeze_3, scalar_tensor_49, add_205);  unsqueeze_3 = scalar_tensor_49 = add_205 = None
    full_1: "f32[20005, 768]" = torch.ops.aten.full.default([20005, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    _unsafe_index_put_1: "f32[20005, 768]" = torch.ops.aten._unsafe_index_put.default(full_1, [primals_196], where_49, True);  full_1 = primals_196 = where_49 = None
    return pytree.tree_unflatten([clone_108, unsqueeze_1, view_647, view_646, view_623, view_622, view_615, view_614, view_591, view_590, view_583, view_582, view_559, view_558, view_551, view_550, view_527, view_526, view_519, view_518, view_495, view_494, view_487, view_486, view_463, view_462, view_455, view_454, view_431, view_430, view_423, view_422, view_399, view_398, view_391, view_390, view_367, view_366, view_359, view_358, view_335, view_334, view_327, view_326, view_303, view_302, view_295, view_294, view_271, view_270, _unsafe_index_put_1, _unsafe_index_put, permute_527, view_644, permute_522, view_640, permute_517, view_636, permute_506, view_625, permute_502, view_620, permute_498, view_617, permute_494, view_612, permute_489, view_608, permute_484, view_604, permute_473, view_593, permute_469, view_588, permute_465, view_585, permute_461, view_580, permute_456, view_576, permute_451, view_572, permute_440, view_561, permute_436, view_556, permute_432, view_553, permute_428, view_548, permute_423, view_544, permute_418, view_540, permute_407, view_529, permute_403, view_524, permute_399, view_521, permute_395, view_516, permute_390, view_512, permute_385, view_508, permute_374, view_497, permute_370, view_492, permute_366, view_489, permute_362, view_484, permute_357, view_480, permute_352, view_476, permute_341, view_465, permute_337, view_460, permute_333, view_457, permute_329, view_452, permute_324, view_448, permute_319, view_444, permute_308, view_433, permute_304, view_428, permute_300, view_425, permute_296, view_420, permute_291, view_416, permute_286, view_412, permute_275, view_401, permute_271, view_396, permute_267, view_393, permute_263, view_388, permute_258, view_384, permute_253, view_380, permute_242, view_369, permute_238, view_364, permute_234, view_361, permute_230, view_356, permute_225, view_352, permute_220, view_348, permute_209, view_337, permute_205, view_332, permute_201, view_329, permute_197, view_324, permute_192, view_320, permute_187, view_316, permute_176, view_305, permute_172, view_300, permute_168, view_297, permute_164, view_292, permute_159, view_288, permute_154, view_284, permute_143, view_273, permute_139, view_268, permute_135, view_265, None, None, None], self._out_spec)
    