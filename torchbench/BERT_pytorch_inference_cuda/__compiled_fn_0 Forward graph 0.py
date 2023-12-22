from __future__ import annotations



def forward(self, arg0_1: "f32[768]", arg1_1: "f32[768]", arg2_1: "f32[768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[768]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[768]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[768]", arg26_1: "f32[768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[768]", arg43_1: "f32[768]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[20005, 768]", arg49_1: "f32[3, 768]", arg50_1: "f32[768, 768]", arg51_1: "f32[768]", arg52_1: "f32[768, 768]", arg53_1: "f32[768]", arg54_1: "f32[768, 768]", arg55_1: "f32[768]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[3072, 768]", arg59_1: "f32[3072]", arg60_1: "f32[768, 3072]", arg61_1: "f32[768]", arg62_1: "f32[768, 768]", arg63_1: "f32[768]", arg64_1: "f32[768, 768]", arg65_1: "f32[768]", arg66_1: "f32[768, 768]", arg67_1: "f32[768]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[3072, 768]", arg71_1: "f32[3072]", arg72_1: "f32[768, 3072]", arg73_1: "f32[768]", arg74_1: "f32[768, 768]", arg75_1: "f32[768]", arg76_1: "f32[768, 768]", arg77_1: "f32[768]", arg78_1: "f32[768, 768]", arg79_1: "f32[768]", arg80_1: "f32[768, 768]", arg81_1: "f32[768]", arg82_1: "f32[3072, 768]", arg83_1: "f32[3072]", arg84_1: "f32[768, 3072]", arg85_1: "f32[768]", arg86_1: "f32[768, 768]", arg87_1: "f32[768]", arg88_1: "f32[768, 768]", arg89_1: "f32[768]", arg90_1: "f32[768, 768]", arg91_1: "f32[768]", arg92_1: "f32[768, 768]", arg93_1: "f32[768]", arg94_1: "f32[3072, 768]", arg95_1: "f32[3072]", arg96_1: "f32[768, 3072]", arg97_1: "f32[768]", arg98_1: "f32[768, 768]", arg99_1: "f32[768]", arg100_1: "f32[768, 768]", arg101_1: "f32[768]", arg102_1: "f32[768, 768]", arg103_1: "f32[768]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[3072, 768]", arg107_1: "f32[3072]", arg108_1: "f32[768, 3072]", arg109_1: "f32[768]", arg110_1: "f32[768, 768]", arg111_1: "f32[768]", arg112_1: "f32[768, 768]", arg113_1: "f32[768]", arg114_1: "f32[768, 768]", arg115_1: "f32[768]", arg116_1: "f32[768, 768]", arg117_1: "f32[768]", arg118_1: "f32[3072, 768]", arg119_1: "f32[3072]", arg120_1: "f32[768, 3072]", arg121_1: "f32[768]", arg122_1: "f32[768, 768]", arg123_1: "f32[768]", arg124_1: "f32[768, 768]", arg125_1: "f32[768]", arg126_1: "f32[768, 768]", arg127_1: "f32[768]", arg128_1: "f32[768, 768]", arg129_1: "f32[768]", arg130_1: "f32[3072, 768]", arg131_1: "f32[3072]", arg132_1: "f32[768, 3072]", arg133_1: "f32[768]", arg134_1: "f32[768, 768]", arg135_1: "f32[768]", arg136_1: "f32[768, 768]", arg137_1: "f32[768]", arg138_1: "f32[768, 768]", arg139_1: "f32[768]", arg140_1: "f32[768, 768]", arg141_1: "f32[768]", arg142_1: "f32[3072, 768]", arg143_1: "f32[3072]", arg144_1: "f32[768, 3072]", arg145_1: "f32[768]", arg146_1: "f32[768, 768]", arg147_1: "f32[768]", arg148_1: "f32[768, 768]", arg149_1: "f32[768]", arg150_1: "f32[768, 768]", arg151_1: "f32[768]", arg152_1: "f32[768, 768]", arg153_1: "f32[768]", arg154_1: "f32[3072, 768]", arg155_1: "f32[3072]", arg156_1: "f32[768, 3072]", arg157_1: "f32[768]", arg158_1: "f32[768, 768]", arg159_1: "f32[768]", arg160_1: "f32[768, 768]", arg161_1: "f32[768]", arg162_1: "f32[768, 768]", arg163_1: "f32[768]", arg164_1: "f32[768, 768]", arg165_1: "f32[768]", arg166_1: "f32[3072, 768]", arg167_1: "f32[3072]", arg168_1: "f32[768, 3072]", arg169_1: "f32[768]", arg170_1: "f32[768, 768]", arg171_1: "f32[768]", arg172_1: "f32[768, 768]", arg173_1: "f32[768]", arg174_1: "f32[768, 768]", arg175_1: "f32[768]", arg176_1: "f32[768, 768]", arg177_1: "f32[768]", arg178_1: "f32[3072, 768]", arg179_1: "f32[3072]", arg180_1: "f32[768, 3072]", arg181_1: "f32[768]", arg182_1: "f32[768, 768]", arg183_1: "f32[768]", arg184_1: "f32[768, 768]", arg185_1: "f32[768]", arg186_1: "f32[768, 768]", arg187_1: "f32[768]", arg188_1: "f32[768, 768]", arg189_1: "f32[768]", arg190_1: "f32[3072, 768]", arg191_1: "f32[3072]", arg192_1: "f32[768, 3072]", arg193_1: "f32[768]", arg194_1: "f32[1, 512, 768]", arg195_1: "i64[4, 128]", arg196_1: "i64[4, 128]"):
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/bert.py:40, code: mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
    gt: "b8[4, 128]" = torch.ops.aten.gt.Scalar(arg195_1, 0)
    unsqueeze: "b8[4, 1, 128]" = torch.ops.aten.unsqueeze.default(gt, 1);  gt = None
    repeat: "b8[4, 128, 128]" = torch.ops.aten.repeat.default(unsqueeze, [1, 128, 1]);  unsqueeze = None
    unsqueeze_1: "b8[4, 1, 128, 128]" = torch.ops.aten.unsqueeze.default(repeat, 1);  repeat = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/module.py:1520, code: return forward_call(*args, **kwargs)
    embedding: "f32[4, 128, 768]" = torch.ops.aten.embedding.default(arg48_1, arg195_1, 0);  arg48_1 = arg195_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/position.py:26, code: return self.pe[:, :x.size(1)]
    slice_1: "f32[1, 512, 768]" = torch.ops.aten.slice.Tensor(arg194_1, 0, 0, 9223372036854775807);  arg194_1 = None
    slice_2: "f32[1, 128, 768]" = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 128);  slice_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
    add: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(embedding, slice_2);  embedding = slice_2 = None
    
    # File: /workspace/youkaichao/code/pytorch/torch/nn/modules/module.py:1520, code: return forward_call(*args, **kwargs)
    embedding_1: "f32[4, 128, 768]" = torch.ops.aten.embedding.default(arg49_1, arg196_1, 0);  arg49_1 = arg196_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:32, code: x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
    add_1: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add, embedding_1);  add = embedding_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/embedding/bert.py:33, code: return self.dropout(x)
    clone: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_1);  add_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone, [-1], correction = 1.0, keepdim = True)
    sqrt: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var);  var = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone, mean);  mean = None
    mul: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg0_1, sub);  arg0_1 = sub = None
    add_2: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt, 1e-06);  sqrt = None
    div: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul, add_2);  mul = add_2 = None
    add_3: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div, arg1_1);  div = arg1_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view: "f32[512, 768]" = torch.ops.aten.view.default(add_3, [512, 768])
    permute: "f32[768, 768]" = torch.ops.aten.permute.default(arg50_1, [1, 0]);  arg50_1 = None
    addmm: "f32[512, 768]" = torch.ops.aten.addmm.default(arg51_1, view, permute);  arg51_1 = view = permute = None
    view_1: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm, [4, 128, 768]);  addmm = None
    view_2: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_1, [4, -1, 12, 64]);  view_1 = None
    permute_1: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None
    view_3: "f32[512, 768]" = torch.ops.aten.view.default(add_3, [512, 768])
    permute_2: "f32[768, 768]" = torch.ops.aten.permute.default(arg52_1, [1, 0]);  arg52_1 = None
    addmm_1: "f32[512, 768]" = torch.ops.aten.addmm.default(arg53_1, view_3, permute_2);  arg53_1 = view_3 = permute_2 = None
    view_4: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_1, [4, 128, 768]);  addmm_1 = None
    view_5: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_4, [4, -1, 12, 64]);  view_4 = None
    permute_3: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_5, [0, 2, 1, 3]);  view_5 = None
    view_6: "f32[512, 768]" = torch.ops.aten.view.default(add_3, [512, 768]);  add_3 = None
    permute_4: "f32[768, 768]" = torch.ops.aten.permute.default(arg54_1, [1, 0]);  arg54_1 = None
    addmm_2: "f32[512, 768]" = torch.ops.aten.addmm.default(arg55_1, view_6, permute_4);  arg55_1 = view_6 = permute_4 = None
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
    bmm: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
    view_11: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm, [4, 12, 128, 128]);  bmm = None
    div_1: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_11, 8.0);  view_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq, scalar_tensor, div_1);  eq = scalar_tensor = div_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where, [-1], True)
    sub_1: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where, amax);  where = amax = None
    exp: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
    sum_1: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
    div_2: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_3: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_2);  div_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_2: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_3, [4, 12, 128, 128]);  clone_3 = None
    view_12: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_2, [48, 128, 128]);  expand_2 = None
    expand_3: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_5, [4, 12, 128, 64]);  permute_5 = None
    clone_4: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
    view_13: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_4, [48, 128, 64]);  clone_4 = None
    bmm_1: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
    view_14: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_1, [4, 12, 128, 64]);  bmm_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_7: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
    clone_5: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_15: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_5, [4, -1, 768]);  clone_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_16: "f32[512, 768]" = torch.ops.aten.view.default(view_15, [512, 768]);  view_15 = None
    permute_8: "f32[768, 768]" = torch.ops.aten.permute.default(arg56_1, [1, 0]);  arg56_1 = None
    addmm_3: "f32[512, 768]" = torch.ops.aten.addmm.default(arg57_1, view_16, permute_8);  arg57_1 = view_16 = permute_8 = None
    view_17: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_3, [4, 128, 768]);  addmm_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_6: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_17);  view_17 = None
    add_4: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone, clone_6);  clone = clone_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_1: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_4, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_1: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_4, [-1], correction = 1.0, keepdim = True)
    sqrt_1: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_1);  var_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_2: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_4, mean_1);  mean_1 = None
    mul_1: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg2_1, sub_2);  arg2_1 = sub_2 = None
    add_5: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_1, 1e-06);  sqrt_1 = None
    div_3: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_1, add_5);  mul_1 = add_5 = None
    add_6: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_3, arg3_1);  div_3 = arg3_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_18: "f32[512, 768]" = torch.ops.aten.view.default(add_6, [512, 768]);  add_6 = None
    permute_9: "f32[768, 3072]" = torch.ops.aten.permute.default(arg58_1, [1, 0]);  arg58_1 = None
    addmm_4: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg59_1, view_18, permute_9);  arg59_1 = view_18 = permute_9 = None
    view_19: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_4, [4, 128, 3072]);  addmm_4 = None
    mul_2: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
    mul_3: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
    erf: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_3);  mul_3 = None
    add_7: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
    mul_4: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_2, add_7);  mul_2 = add_7 = None
    clone_7: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_4);  mul_4 = None
    view_20: "f32[512, 3072]" = torch.ops.aten.view.default(clone_7, [512, 3072]);  clone_7 = None
    permute_10: "f32[3072, 768]" = torch.ops.aten.permute.default(arg60_1, [1, 0]);  arg60_1 = None
    addmm_5: "f32[512, 768]" = torch.ops.aten.addmm.default(arg61_1, view_20, permute_10);  arg61_1 = view_20 = permute_10 = None
    view_21: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_5, [4, 128, 768]);  addmm_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_8: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_21);  view_21 = None
    add_8: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_4, clone_8);  add_4 = clone_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_9: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_8);  add_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_2: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_9, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_2: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_9, [-1], correction = 1.0, keepdim = True)
    sqrt_2: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_2);  var_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_3: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_9, mean_2);  mean_2 = None
    mul_5: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg4_1, sub_3);  arg4_1 = sub_3 = None
    add_9: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_2, 1e-06);  sqrt_2 = None
    div_4: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_5, add_9);  mul_5 = add_9 = None
    add_10: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_4, arg5_1);  div_4 = arg5_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_22: "f32[512, 768]" = torch.ops.aten.view.default(add_10, [512, 768])
    permute_11: "f32[768, 768]" = torch.ops.aten.permute.default(arg62_1, [1, 0]);  arg62_1 = None
    addmm_6: "f32[512, 768]" = torch.ops.aten.addmm.default(arg63_1, view_22, permute_11);  arg63_1 = view_22 = permute_11 = None
    view_23: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_6, [4, 128, 768]);  addmm_6 = None
    view_24: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_23, [4, -1, 12, 64]);  view_23 = None
    permute_12: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None
    view_25: "f32[512, 768]" = torch.ops.aten.view.default(add_10, [512, 768])
    permute_13: "f32[768, 768]" = torch.ops.aten.permute.default(arg64_1, [1, 0]);  arg64_1 = None
    addmm_7: "f32[512, 768]" = torch.ops.aten.addmm.default(arg65_1, view_25, permute_13);  arg65_1 = view_25 = permute_13 = None
    view_26: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_7, [4, 128, 768]);  addmm_7 = None
    view_27: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_26, [4, -1, 12, 64]);  view_26 = None
    permute_14: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_27, [0, 2, 1, 3]);  view_27 = None
    view_28: "f32[512, 768]" = torch.ops.aten.view.default(add_10, [512, 768]);  add_10 = None
    permute_15: "f32[768, 768]" = torch.ops.aten.permute.default(arg66_1, [1, 0]);  arg66_1 = None
    addmm_8: "f32[512, 768]" = torch.ops.aten.addmm.default(arg67_1, view_28, permute_15);  arg67_1 = view_28 = permute_15 = None
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
    bmm_2: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_31, view_32);  view_31 = view_32 = None
    view_33: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_2, [4, 12, 128, 128]);  bmm_2 = None
    div_5: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_33, 8.0);  view_33 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_1: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_1: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_1, scalar_tensor_1, div_5);  eq_1 = scalar_tensor_1 = div_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_1: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_1, [-1], True)
    sub_4: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_1, amax_1);  where_1 = amax_1 = None
    exp_1: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
    sum_2: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
    div_6: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_12: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_6);  div_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_6: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_12, [4, 12, 128, 128]);  clone_12 = None
    view_34: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_6, [48, 128, 128]);  expand_6 = None
    expand_7: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_16, [4, 12, 128, 64]);  permute_16 = None
    clone_13: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
    view_35: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_13, [48, 128, 64]);  clone_13 = None
    bmm_3: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
    view_36: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_3, [4, 12, 128, 64]);  bmm_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_18: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
    clone_14: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None
    view_37: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_14, [4, -1, 768]);  clone_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_38: "f32[512, 768]" = torch.ops.aten.view.default(view_37, [512, 768]);  view_37 = None
    permute_19: "f32[768, 768]" = torch.ops.aten.permute.default(arg68_1, [1, 0]);  arg68_1 = None
    addmm_9: "f32[512, 768]" = torch.ops.aten.addmm.default(arg69_1, view_38, permute_19);  arg69_1 = view_38 = permute_19 = None
    view_39: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_9, [4, 128, 768]);  addmm_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_15: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_39);  view_39 = None
    add_11: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_9, clone_15);  clone_9 = clone_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_3: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_11, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_3: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_11, [-1], correction = 1.0, keepdim = True)
    sqrt_3: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_3);  var_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_5: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_11, mean_3);  mean_3 = None
    mul_6: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg6_1, sub_5);  arg6_1 = sub_5 = None
    add_12: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_3, 1e-06);  sqrt_3 = None
    div_7: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_6, add_12);  mul_6 = add_12 = None
    add_13: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_7, arg7_1);  div_7 = arg7_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_40: "f32[512, 768]" = torch.ops.aten.view.default(add_13, [512, 768]);  add_13 = None
    permute_20: "f32[768, 3072]" = torch.ops.aten.permute.default(arg70_1, [1, 0]);  arg70_1 = None
    addmm_10: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg71_1, view_40, permute_20);  arg71_1 = view_40 = permute_20 = None
    view_41: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_10, [4, 128, 3072]);  addmm_10 = None
    mul_7: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
    mul_8: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
    erf_1: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
    add_14: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
    mul_9: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_7, add_14);  mul_7 = add_14 = None
    clone_16: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_9);  mul_9 = None
    view_42: "f32[512, 3072]" = torch.ops.aten.view.default(clone_16, [512, 3072]);  clone_16 = None
    permute_21: "f32[3072, 768]" = torch.ops.aten.permute.default(arg72_1, [1, 0]);  arg72_1 = None
    addmm_11: "f32[512, 768]" = torch.ops.aten.addmm.default(arg73_1, view_42, permute_21);  arg73_1 = view_42 = permute_21 = None
    view_43: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_11, [4, 128, 768]);  addmm_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_17: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_43);  view_43 = None
    add_15: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_11, clone_17);  add_11 = clone_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_18: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_15);  add_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_4: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_18, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_4: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_18, [-1], correction = 1.0, keepdim = True)
    sqrt_4: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_4);  var_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_6: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_18, mean_4);  mean_4 = None
    mul_10: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg8_1, sub_6);  arg8_1 = sub_6 = None
    add_16: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_4, 1e-06);  sqrt_4 = None
    div_8: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_10, add_16);  mul_10 = add_16 = None
    add_17: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_8, arg9_1);  div_8 = arg9_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_44: "f32[512, 768]" = torch.ops.aten.view.default(add_17, [512, 768])
    permute_22: "f32[768, 768]" = torch.ops.aten.permute.default(arg74_1, [1, 0]);  arg74_1 = None
    addmm_12: "f32[512, 768]" = torch.ops.aten.addmm.default(arg75_1, view_44, permute_22);  arg75_1 = view_44 = permute_22 = None
    view_45: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_12, [4, 128, 768]);  addmm_12 = None
    view_46: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_45, [4, -1, 12, 64]);  view_45 = None
    permute_23: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
    view_47: "f32[512, 768]" = torch.ops.aten.view.default(add_17, [512, 768])
    permute_24: "f32[768, 768]" = torch.ops.aten.permute.default(arg76_1, [1, 0]);  arg76_1 = None
    addmm_13: "f32[512, 768]" = torch.ops.aten.addmm.default(arg77_1, view_47, permute_24);  arg77_1 = view_47 = permute_24 = None
    view_48: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_13, [4, 128, 768]);  addmm_13 = None
    view_49: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_48, [4, -1, 12, 64]);  view_48 = None
    permute_25: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_49, [0, 2, 1, 3]);  view_49 = None
    view_50: "f32[512, 768]" = torch.ops.aten.view.default(add_17, [512, 768]);  add_17 = None
    permute_26: "f32[768, 768]" = torch.ops.aten.permute.default(arg78_1, [1, 0]);  arg78_1 = None
    addmm_14: "f32[512, 768]" = torch.ops.aten.addmm.default(arg79_1, view_50, permute_26);  arg79_1 = view_50 = permute_26 = None
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
    bmm_4: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_53, view_54);  view_53 = view_54 = None
    view_55: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_4, [4, 12, 128, 128]);  bmm_4 = None
    div_9: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_55, 8.0);  view_55 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_2: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_2: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_2: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_2, scalar_tensor_2, div_9);  eq_2 = scalar_tensor_2 = div_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_2: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_2, [-1], True)
    sub_7: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_2, amax_2);  where_2 = amax_2 = None
    exp_2: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
    sum_3: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
    div_10: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_21: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_10);  div_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_10: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_21, [4, 12, 128, 128]);  clone_21 = None
    view_56: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_10, [48, 128, 128]);  expand_10 = None
    expand_11: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_27, [4, 12, 128, 64]);  permute_27 = None
    clone_22: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
    view_57: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_22, [48, 128, 64]);  clone_22 = None
    bmm_5: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_56, view_57);  view_56 = view_57 = None
    view_58: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_5, [4, 12, 128, 64]);  bmm_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_29: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
    clone_23: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
    view_59: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_23, [4, -1, 768]);  clone_23 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_60: "f32[512, 768]" = torch.ops.aten.view.default(view_59, [512, 768]);  view_59 = None
    permute_30: "f32[768, 768]" = torch.ops.aten.permute.default(arg80_1, [1, 0]);  arg80_1 = None
    addmm_15: "f32[512, 768]" = torch.ops.aten.addmm.default(arg81_1, view_60, permute_30);  arg81_1 = view_60 = permute_30 = None
    view_61: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_15, [4, 128, 768]);  addmm_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_24: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_61);  view_61 = None
    add_18: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_18, clone_24);  clone_18 = clone_24 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_5: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_18, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_5: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_18, [-1], correction = 1.0, keepdim = True)
    sqrt_5: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_5);  var_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_8: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_18, mean_5);  mean_5 = None
    mul_11: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg10_1, sub_8);  arg10_1 = sub_8 = None
    add_19: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_5, 1e-06);  sqrt_5 = None
    div_11: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_11, add_19);  mul_11 = add_19 = None
    add_20: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_11, arg11_1);  div_11 = arg11_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_62: "f32[512, 768]" = torch.ops.aten.view.default(add_20, [512, 768]);  add_20 = None
    permute_31: "f32[768, 3072]" = torch.ops.aten.permute.default(arg82_1, [1, 0]);  arg82_1 = None
    addmm_16: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg83_1, view_62, permute_31);  arg83_1 = view_62 = permute_31 = None
    view_63: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_16, [4, 128, 3072]);  addmm_16 = None
    mul_12: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.5)
    mul_13: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476);  view_63 = None
    erf_2: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_13);  mul_13 = None
    add_21: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
    mul_14: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_12, add_21);  mul_12 = add_21 = None
    clone_25: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_14);  mul_14 = None
    view_64: "f32[512, 3072]" = torch.ops.aten.view.default(clone_25, [512, 3072]);  clone_25 = None
    permute_32: "f32[3072, 768]" = torch.ops.aten.permute.default(arg84_1, [1, 0]);  arg84_1 = None
    addmm_17: "f32[512, 768]" = torch.ops.aten.addmm.default(arg85_1, view_64, permute_32);  arg85_1 = view_64 = permute_32 = None
    view_65: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_17, [4, 128, 768]);  addmm_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_26: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_65);  view_65 = None
    add_22: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_18, clone_26);  add_18 = clone_26 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_27: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_22);  add_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_6: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_27, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_6: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_27, [-1], correction = 1.0, keepdim = True)
    sqrt_6: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_6);  var_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_9: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_27, mean_6);  mean_6 = None
    mul_15: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg12_1, sub_9);  arg12_1 = sub_9 = None
    add_23: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_6, 1e-06);  sqrt_6 = None
    div_12: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_15, add_23);  mul_15 = add_23 = None
    add_24: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_12, arg13_1);  div_12 = arg13_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_66: "f32[512, 768]" = torch.ops.aten.view.default(add_24, [512, 768])
    permute_33: "f32[768, 768]" = torch.ops.aten.permute.default(arg86_1, [1, 0]);  arg86_1 = None
    addmm_18: "f32[512, 768]" = torch.ops.aten.addmm.default(arg87_1, view_66, permute_33);  arg87_1 = view_66 = permute_33 = None
    view_67: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_18, [4, 128, 768]);  addmm_18 = None
    view_68: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_67, [4, -1, 12, 64]);  view_67 = None
    permute_34: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_68, [0, 2, 1, 3]);  view_68 = None
    view_69: "f32[512, 768]" = torch.ops.aten.view.default(add_24, [512, 768])
    permute_35: "f32[768, 768]" = torch.ops.aten.permute.default(arg88_1, [1, 0]);  arg88_1 = None
    addmm_19: "f32[512, 768]" = torch.ops.aten.addmm.default(arg89_1, view_69, permute_35);  arg89_1 = view_69 = permute_35 = None
    view_70: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_19, [4, 128, 768]);  addmm_19 = None
    view_71: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_70, [4, -1, 12, 64]);  view_70 = None
    permute_36: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_71, [0, 2, 1, 3]);  view_71 = None
    view_72: "f32[512, 768]" = torch.ops.aten.view.default(add_24, [512, 768]);  add_24 = None
    permute_37: "f32[768, 768]" = torch.ops.aten.permute.default(arg90_1, [1, 0]);  arg90_1 = None
    addmm_20: "f32[512, 768]" = torch.ops.aten.addmm.default(arg91_1, view_72, permute_37);  arg91_1 = view_72 = permute_37 = None
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
    bmm_6: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_75, view_76);  view_75 = view_76 = None
    view_77: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_6, [4, 12, 128, 128]);  bmm_6 = None
    div_13: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_77, 8.0);  view_77 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_3: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_3: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_3: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_3, scalar_tensor_3, div_13);  eq_3 = scalar_tensor_3 = div_13 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_3: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_3, [-1], True)
    sub_10: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_3, amax_3);  where_3 = amax_3 = None
    exp_3: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
    sum_4: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
    div_14: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_30: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_14);  div_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_14: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_30, [4, 12, 128, 128]);  clone_30 = None
    view_78: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_14, [48, 128, 128]);  expand_14 = None
    expand_15: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_38, [4, 12, 128, 64]);  permute_38 = None
    clone_31: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
    view_79: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_31, [48, 128, 64]);  clone_31 = None
    bmm_7: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_78, view_79);  view_78 = view_79 = None
    view_80: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_7, [4, 12, 128, 64]);  bmm_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_40: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None
    clone_32: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
    view_81: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_32, [4, -1, 768]);  clone_32 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_82: "f32[512, 768]" = torch.ops.aten.view.default(view_81, [512, 768]);  view_81 = None
    permute_41: "f32[768, 768]" = torch.ops.aten.permute.default(arg92_1, [1, 0]);  arg92_1 = None
    addmm_21: "f32[512, 768]" = torch.ops.aten.addmm.default(arg93_1, view_82, permute_41);  arg93_1 = view_82 = permute_41 = None
    view_83: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_21, [4, 128, 768]);  addmm_21 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_33: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_83);  view_83 = None
    add_25: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_27, clone_33);  clone_27 = clone_33 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_7: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_25, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_7: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_25, [-1], correction = 1.0, keepdim = True)
    sqrt_7: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_7);  var_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_11: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_25, mean_7);  mean_7 = None
    mul_16: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg14_1, sub_11);  arg14_1 = sub_11 = None
    add_26: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_7, 1e-06);  sqrt_7 = None
    div_15: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_16, add_26);  mul_16 = add_26 = None
    add_27: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_15, arg15_1);  div_15 = arg15_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_84: "f32[512, 768]" = torch.ops.aten.view.default(add_27, [512, 768]);  add_27 = None
    permute_42: "f32[768, 3072]" = torch.ops.aten.permute.default(arg94_1, [1, 0]);  arg94_1 = None
    addmm_22: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg95_1, view_84, permute_42);  arg95_1 = view_84 = permute_42 = None
    view_85: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_22, [4, 128, 3072]);  addmm_22 = None
    mul_17: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.5)
    mul_18: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476);  view_85 = None
    erf_3: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_18);  mul_18 = None
    add_28: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
    mul_19: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_17, add_28);  mul_17 = add_28 = None
    clone_34: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_19);  mul_19 = None
    view_86: "f32[512, 3072]" = torch.ops.aten.view.default(clone_34, [512, 3072]);  clone_34 = None
    permute_43: "f32[3072, 768]" = torch.ops.aten.permute.default(arg96_1, [1, 0]);  arg96_1 = None
    addmm_23: "f32[512, 768]" = torch.ops.aten.addmm.default(arg97_1, view_86, permute_43);  arg97_1 = view_86 = permute_43 = None
    view_87: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_23, [4, 128, 768]);  addmm_23 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_35: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_87);  view_87 = None
    add_29: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_25, clone_35);  add_25 = clone_35 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_36: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_8: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_36, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_8: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_36, [-1], correction = 1.0, keepdim = True)
    sqrt_8: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_8);  var_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_12: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_36, mean_8);  mean_8 = None
    mul_20: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg16_1, sub_12);  arg16_1 = sub_12 = None
    add_30: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_8, 1e-06);  sqrt_8 = None
    div_16: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_20, add_30);  mul_20 = add_30 = None
    add_31: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_16, arg17_1);  div_16 = arg17_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_88: "f32[512, 768]" = torch.ops.aten.view.default(add_31, [512, 768])
    permute_44: "f32[768, 768]" = torch.ops.aten.permute.default(arg98_1, [1, 0]);  arg98_1 = None
    addmm_24: "f32[512, 768]" = torch.ops.aten.addmm.default(arg99_1, view_88, permute_44);  arg99_1 = view_88 = permute_44 = None
    view_89: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_24, [4, 128, 768]);  addmm_24 = None
    view_90: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_89, [4, -1, 12, 64]);  view_89 = None
    permute_45: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
    view_91: "f32[512, 768]" = torch.ops.aten.view.default(add_31, [512, 768])
    permute_46: "f32[768, 768]" = torch.ops.aten.permute.default(arg100_1, [1, 0]);  arg100_1 = None
    addmm_25: "f32[512, 768]" = torch.ops.aten.addmm.default(arg101_1, view_91, permute_46);  arg101_1 = view_91 = permute_46 = None
    view_92: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_25, [4, 128, 768]);  addmm_25 = None
    view_93: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_92, [4, -1, 12, 64]);  view_92 = None
    permute_47: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_93, [0, 2, 1, 3]);  view_93 = None
    view_94: "f32[512, 768]" = torch.ops.aten.view.default(add_31, [512, 768]);  add_31 = None
    permute_48: "f32[768, 768]" = torch.ops.aten.permute.default(arg102_1, [1, 0]);  arg102_1 = None
    addmm_26: "f32[512, 768]" = torch.ops.aten.addmm.default(arg103_1, view_94, permute_48);  arg103_1 = view_94 = permute_48 = None
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
    bmm_8: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_97, view_98);  view_97 = view_98 = None
    view_99: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_8, [4, 12, 128, 128]);  bmm_8 = None
    div_17: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_99, 8.0);  view_99 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_4: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_4: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_4: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_4, scalar_tensor_4, div_17);  eq_4 = scalar_tensor_4 = div_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_4: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_4, [-1], True)
    sub_13: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_4, amax_4);  where_4 = amax_4 = None
    exp_4: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
    sum_5: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
    div_18: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_39: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_18);  div_18 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_18: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_39, [4, 12, 128, 128]);  clone_39 = None
    view_100: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_18, [48, 128, 128]);  expand_18 = None
    expand_19: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_49, [4, 12, 128, 64]);  permute_49 = None
    clone_40: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
    view_101: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_40, [48, 128, 64]);  clone_40 = None
    bmm_9: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_100, view_101);  view_100 = view_101 = None
    view_102: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_9, [4, 12, 128, 64]);  bmm_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_51: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_102, [0, 2, 1, 3]);  view_102 = None
    clone_41: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_51, memory_format = torch.contiguous_format);  permute_51 = None
    view_103: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_41, [4, -1, 768]);  clone_41 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_104: "f32[512, 768]" = torch.ops.aten.view.default(view_103, [512, 768]);  view_103 = None
    permute_52: "f32[768, 768]" = torch.ops.aten.permute.default(arg104_1, [1, 0]);  arg104_1 = None
    addmm_27: "f32[512, 768]" = torch.ops.aten.addmm.default(arg105_1, view_104, permute_52);  arg105_1 = view_104 = permute_52 = None
    view_105: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_27, [4, 128, 768]);  addmm_27 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_42: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_105);  view_105 = None
    add_32: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_36, clone_42);  clone_36 = clone_42 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_9: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_32, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_9: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_32, [-1], correction = 1.0, keepdim = True)
    sqrt_9: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_9);  var_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_14: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_32, mean_9);  mean_9 = None
    mul_21: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg18_1, sub_14);  arg18_1 = sub_14 = None
    add_33: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_9, 1e-06);  sqrt_9 = None
    div_19: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_21, add_33);  mul_21 = add_33 = None
    add_34: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_19, arg19_1);  div_19 = arg19_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_106: "f32[512, 768]" = torch.ops.aten.view.default(add_34, [512, 768]);  add_34 = None
    permute_53: "f32[768, 3072]" = torch.ops.aten.permute.default(arg106_1, [1, 0]);  arg106_1 = None
    addmm_28: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg107_1, view_106, permute_53);  arg107_1 = view_106 = permute_53 = None
    view_107: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_28, [4, 128, 3072]);  addmm_28 = None
    mul_22: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.5)
    mul_23: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476);  view_107 = None
    erf_4: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_23);  mul_23 = None
    add_35: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
    mul_24: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_22, add_35);  mul_22 = add_35 = None
    clone_43: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_24);  mul_24 = None
    view_108: "f32[512, 3072]" = torch.ops.aten.view.default(clone_43, [512, 3072]);  clone_43 = None
    permute_54: "f32[3072, 768]" = torch.ops.aten.permute.default(arg108_1, [1, 0]);  arg108_1 = None
    addmm_29: "f32[512, 768]" = torch.ops.aten.addmm.default(arg109_1, view_108, permute_54);  arg109_1 = view_108 = permute_54 = None
    view_109: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_29, [4, 128, 768]);  addmm_29 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_44: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_109);  view_109 = None
    add_36: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_32, clone_44);  add_32 = clone_44 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_45: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_36);  add_36 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_10: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_45, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_10: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_45, [-1], correction = 1.0, keepdim = True)
    sqrt_10: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_10);  var_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_15: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_45, mean_10);  mean_10 = None
    mul_25: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg20_1, sub_15);  arg20_1 = sub_15 = None
    add_37: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_10, 1e-06);  sqrt_10 = None
    div_20: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_25, add_37);  mul_25 = add_37 = None
    add_38: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_20, arg21_1);  div_20 = arg21_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_110: "f32[512, 768]" = torch.ops.aten.view.default(add_38, [512, 768])
    permute_55: "f32[768, 768]" = torch.ops.aten.permute.default(arg110_1, [1, 0]);  arg110_1 = None
    addmm_30: "f32[512, 768]" = torch.ops.aten.addmm.default(arg111_1, view_110, permute_55);  arg111_1 = view_110 = permute_55 = None
    view_111: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_30, [4, 128, 768]);  addmm_30 = None
    view_112: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_111, [4, -1, 12, 64]);  view_111 = None
    permute_56: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_112, [0, 2, 1, 3]);  view_112 = None
    view_113: "f32[512, 768]" = torch.ops.aten.view.default(add_38, [512, 768])
    permute_57: "f32[768, 768]" = torch.ops.aten.permute.default(arg112_1, [1, 0]);  arg112_1 = None
    addmm_31: "f32[512, 768]" = torch.ops.aten.addmm.default(arg113_1, view_113, permute_57);  arg113_1 = view_113 = permute_57 = None
    view_114: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_31, [4, 128, 768]);  addmm_31 = None
    view_115: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_114, [4, -1, 12, 64]);  view_114 = None
    permute_58: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_115, [0, 2, 1, 3]);  view_115 = None
    view_116: "f32[512, 768]" = torch.ops.aten.view.default(add_38, [512, 768]);  add_38 = None
    permute_59: "f32[768, 768]" = torch.ops.aten.permute.default(arg114_1, [1, 0]);  arg114_1 = None
    addmm_32: "f32[512, 768]" = torch.ops.aten.addmm.default(arg115_1, view_116, permute_59);  arg115_1 = view_116 = permute_59 = None
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
    bmm_10: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_119, view_120);  view_119 = view_120 = None
    view_121: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_10, [4, 12, 128, 128]);  bmm_10 = None
    div_21: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_121, 8.0);  view_121 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_5: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_5: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_5: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_5, scalar_tensor_5, div_21);  eq_5 = scalar_tensor_5 = div_21 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_5: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_5, [-1], True)
    sub_16: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_5, amax_5);  where_5 = amax_5 = None
    exp_5: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
    sum_6: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
    div_22: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_48: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_22);  div_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_22: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_48, [4, 12, 128, 128]);  clone_48 = None
    view_122: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_22, [48, 128, 128]);  expand_22 = None
    expand_23: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_60, [4, 12, 128, 64]);  permute_60 = None
    clone_49: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
    view_123: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_49, [48, 128, 64]);  clone_49 = None
    bmm_11: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_122, view_123);  view_122 = view_123 = None
    view_124: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_11, [4, 12, 128, 64]);  bmm_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_62: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_124, [0, 2, 1, 3]);  view_124 = None
    clone_50: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_62, memory_format = torch.contiguous_format);  permute_62 = None
    view_125: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_50, [4, -1, 768]);  clone_50 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_126: "f32[512, 768]" = torch.ops.aten.view.default(view_125, [512, 768]);  view_125 = None
    permute_63: "f32[768, 768]" = torch.ops.aten.permute.default(arg116_1, [1, 0]);  arg116_1 = None
    addmm_33: "f32[512, 768]" = torch.ops.aten.addmm.default(arg117_1, view_126, permute_63);  arg117_1 = view_126 = permute_63 = None
    view_127: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_33, [4, 128, 768]);  addmm_33 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_51: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_127);  view_127 = None
    add_39: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_45, clone_51);  clone_45 = clone_51 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_11: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_39, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_11: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_39, [-1], correction = 1.0, keepdim = True)
    sqrt_11: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_11);  var_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_17: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_39, mean_11);  mean_11 = None
    mul_26: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg22_1, sub_17);  arg22_1 = sub_17 = None
    add_40: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_11, 1e-06);  sqrt_11 = None
    div_23: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_26, add_40);  mul_26 = add_40 = None
    add_41: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_23, arg23_1);  div_23 = arg23_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_128: "f32[512, 768]" = torch.ops.aten.view.default(add_41, [512, 768]);  add_41 = None
    permute_64: "f32[768, 3072]" = torch.ops.aten.permute.default(arg118_1, [1, 0]);  arg118_1 = None
    addmm_34: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg119_1, view_128, permute_64);  arg119_1 = view_128 = permute_64 = None
    view_129: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_34, [4, 128, 3072]);  addmm_34 = None
    mul_27: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.5)
    mul_28: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476);  view_129 = None
    erf_5: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_28);  mul_28 = None
    add_42: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
    mul_29: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_27, add_42);  mul_27 = add_42 = None
    clone_52: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_29);  mul_29 = None
    view_130: "f32[512, 3072]" = torch.ops.aten.view.default(clone_52, [512, 3072]);  clone_52 = None
    permute_65: "f32[3072, 768]" = torch.ops.aten.permute.default(arg120_1, [1, 0]);  arg120_1 = None
    addmm_35: "f32[512, 768]" = torch.ops.aten.addmm.default(arg121_1, view_130, permute_65);  arg121_1 = view_130 = permute_65 = None
    view_131: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_35, [4, 128, 768]);  addmm_35 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_53: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_131);  view_131 = None
    add_43: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_39, clone_53);  add_39 = clone_53 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_54: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_43);  add_43 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_12: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_54, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_12: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_54, [-1], correction = 1.0, keepdim = True)
    sqrt_12: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_12);  var_12 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_18: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_54, mean_12);  mean_12 = None
    mul_30: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg24_1, sub_18);  arg24_1 = sub_18 = None
    add_44: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_12, 1e-06);  sqrt_12 = None
    div_24: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_30, add_44);  mul_30 = add_44 = None
    add_45: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_24, arg25_1);  div_24 = arg25_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_132: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768])
    permute_66: "f32[768, 768]" = torch.ops.aten.permute.default(arg122_1, [1, 0]);  arg122_1 = None
    addmm_36: "f32[512, 768]" = torch.ops.aten.addmm.default(arg123_1, view_132, permute_66);  arg123_1 = view_132 = permute_66 = None
    view_133: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_36, [4, 128, 768]);  addmm_36 = None
    view_134: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_133, [4, -1, 12, 64]);  view_133 = None
    permute_67: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_134, [0, 2, 1, 3]);  view_134 = None
    view_135: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768])
    permute_68: "f32[768, 768]" = torch.ops.aten.permute.default(arg124_1, [1, 0]);  arg124_1 = None
    addmm_37: "f32[512, 768]" = torch.ops.aten.addmm.default(arg125_1, view_135, permute_68);  arg125_1 = view_135 = permute_68 = None
    view_136: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_37, [4, 128, 768]);  addmm_37 = None
    view_137: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_136, [4, -1, 12, 64]);  view_136 = None
    permute_69: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_137, [0, 2, 1, 3]);  view_137 = None
    view_138: "f32[512, 768]" = torch.ops.aten.view.default(add_45, [512, 768]);  add_45 = None
    permute_70: "f32[768, 768]" = torch.ops.aten.permute.default(arg126_1, [1, 0]);  arg126_1 = None
    addmm_38: "f32[512, 768]" = torch.ops.aten.addmm.default(arg127_1, view_138, permute_70);  arg127_1 = view_138 = permute_70 = None
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
    bmm_12: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_141, view_142);  view_141 = view_142 = None
    view_143: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_12, [4, 12, 128, 128]);  bmm_12 = None
    div_25: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_143, 8.0);  view_143 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_6: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_6: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_6: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_6, scalar_tensor_6, div_25);  eq_6 = scalar_tensor_6 = div_25 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_6: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_6, [-1], True)
    sub_19: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_6, amax_6);  where_6 = amax_6 = None
    exp_6: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
    sum_7: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
    div_26: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_57: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_26);  div_26 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_26: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_57, [4, 12, 128, 128]);  clone_57 = None
    view_144: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_26, [48, 128, 128]);  expand_26 = None
    expand_27: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_71, [4, 12, 128, 64]);  permute_71 = None
    clone_58: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
    view_145: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_58, [48, 128, 64]);  clone_58 = None
    bmm_13: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_144, view_145);  view_144 = view_145 = None
    view_146: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_13, [4, 12, 128, 64]);  bmm_13 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_73: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_146, [0, 2, 1, 3]);  view_146 = None
    clone_59: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_73, memory_format = torch.contiguous_format);  permute_73 = None
    view_147: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_59, [4, -1, 768]);  clone_59 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_148: "f32[512, 768]" = torch.ops.aten.view.default(view_147, [512, 768]);  view_147 = None
    permute_74: "f32[768, 768]" = torch.ops.aten.permute.default(arg128_1, [1, 0]);  arg128_1 = None
    addmm_39: "f32[512, 768]" = torch.ops.aten.addmm.default(arg129_1, view_148, permute_74);  arg129_1 = view_148 = permute_74 = None
    view_149: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_39, [4, 128, 768]);  addmm_39 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_60: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_149);  view_149 = None
    add_46: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_54, clone_60);  clone_54 = clone_60 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_13: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_46, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_13: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_46, [-1], correction = 1.0, keepdim = True)
    sqrt_13: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_13);  var_13 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_20: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_46, mean_13);  mean_13 = None
    mul_31: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg26_1, sub_20);  arg26_1 = sub_20 = None
    add_47: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_13, 1e-06);  sqrt_13 = None
    div_27: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_31, add_47);  mul_31 = add_47 = None
    add_48: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_27, arg27_1);  div_27 = arg27_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_150: "f32[512, 768]" = torch.ops.aten.view.default(add_48, [512, 768]);  add_48 = None
    permute_75: "f32[768, 3072]" = torch.ops.aten.permute.default(arg130_1, [1, 0]);  arg130_1 = None
    addmm_40: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg131_1, view_150, permute_75);  arg131_1 = view_150 = permute_75 = None
    view_151: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_40, [4, 128, 3072]);  addmm_40 = None
    mul_32: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.5)
    mul_33: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476);  view_151 = None
    erf_6: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_33);  mul_33 = None
    add_49: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
    mul_34: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_32, add_49);  mul_32 = add_49 = None
    clone_61: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_34);  mul_34 = None
    view_152: "f32[512, 3072]" = torch.ops.aten.view.default(clone_61, [512, 3072]);  clone_61 = None
    permute_76: "f32[3072, 768]" = torch.ops.aten.permute.default(arg132_1, [1, 0]);  arg132_1 = None
    addmm_41: "f32[512, 768]" = torch.ops.aten.addmm.default(arg133_1, view_152, permute_76);  arg133_1 = view_152 = permute_76 = None
    view_153: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_41, [4, 128, 768]);  addmm_41 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_62: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_153);  view_153 = None
    add_50: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_46, clone_62);  add_46 = clone_62 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_63: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_50);  add_50 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_14: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_63, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_14: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_63, [-1], correction = 1.0, keepdim = True)
    sqrt_14: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_14);  var_14 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_21: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_63, mean_14);  mean_14 = None
    mul_35: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg28_1, sub_21);  arg28_1 = sub_21 = None
    add_51: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_14, 1e-06);  sqrt_14 = None
    div_28: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_35, add_51);  mul_35 = add_51 = None
    add_52: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_28, arg29_1);  div_28 = arg29_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_154: "f32[512, 768]" = torch.ops.aten.view.default(add_52, [512, 768])
    permute_77: "f32[768, 768]" = torch.ops.aten.permute.default(arg134_1, [1, 0]);  arg134_1 = None
    addmm_42: "f32[512, 768]" = torch.ops.aten.addmm.default(arg135_1, view_154, permute_77);  arg135_1 = view_154 = permute_77 = None
    view_155: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_42, [4, 128, 768]);  addmm_42 = None
    view_156: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_155, [4, -1, 12, 64]);  view_155 = None
    permute_78: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_156, [0, 2, 1, 3]);  view_156 = None
    view_157: "f32[512, 768]" = torch.ops.aten.view.default(add_52, [512, 768])
    permute_79: "f32[768, 768]" = torch.ops.aten.permute.default(arg136_1, [1, 0]);  arg136_1 = None
    addmm_43: "f32[512, 768]" = torch.ops.aten.addmm.default(arg137_1, view_157, permute_79);  arg137_1 = view_157 = permute_79 = None
    view_158: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_43, [4, 128, 768]);  addmm_43 = None
    view_159: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_158, [4, -1, 12, 64]);  view_158 = None
    permute_80: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
    view_160: "f32[512, 768]" = torch.ops.aten.view.default(add_52, [512, 768]);  add_52 = None
    permute_81: "f32[768, 768]" = torch.ops.aten.permute.default(arg138_1, [1, 0]);  arg138_1 = None
    addmm_44: "f32[512, 768]" = torch.ops.aten.addmm.default(arg139_1, view_160, permute_81);  arg139_1 = view_160 = permute_81 = None
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
    bmm_14: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_163, view_164);  view_163 = view_164 = None
    view_165: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_14, [4, 12, 128, 128]);  bmm_14 = None
    div_29: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_165, 8.0);  view_165 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_7: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_7: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_7: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_7, scalar_tensor_7, div_29);  eq_7 = scalar_tensor_7 = div_29 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_7: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_7, [-1], True)
    sub_22: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_7, amax_7);  where_7 = amax_7 = None
    exp_7: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
    sum_8: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
    div_30: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_66: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_30);  div_30 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_30: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_66, [4, 12, 128, 128]);  clone_66 = None
    view_166: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_30, [48, 128, 128]);  expand_30 = None
    expand_31: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_82, [4, 12, 128, 64]);  permute_82 = None
    clone_67: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
    view_167: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_67, [48, 128, 64]);  clone_67 = None
    bmm_15: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_166, view_167);  view_166 = view_167 = None
    view_168: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_15, [4, 12, 128, 64]);  bmm_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_84: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_168, [0, 2, 1, 3]);  view_168 = None
    clone_68: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_84, memory_format = torch.contiguous_format);  permute_84 = None
    view_169: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_68, [4, -1, 768]);  clone_68 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_170: "f32[512, 768]" = torch.ops.aten.view.default(view_169, [512, 768]);  view_169 = None
    permute_85: "f32[768, 768]" = torch.ops.aten.permute.default(arg140_1, [1, 0]);  arg140_1 = None
    addmm_45: "f32[512, 768]" = torch.ops.aten.addmm.default(arg141_1, view_170, permute_85);  arg141_1 = view_170 = permute_85 = None
    view_171: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_45, [4, 128, 768]);  addmm_45 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_69: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_171);  view_171 = None
    add_53: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_63, clone_69);  clone_63 = clone_69 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_15: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_53, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_15: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_53, [-1], correction = 1.0, keepdim = True)
    sqrt_15: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_15);  var_15 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_23: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_53, mean_15);  mean_15 = None
    mul_36: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg30_1, sub_23);  arg30_1 = sub_23 = None
    add_54: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_15, 1e-06);  sqrt_15 = None
    div_31: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_36, add_54);  mul_36 = add_54 = None
    add_55: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_31, arg31_1);  div_31 = arg31_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_172: "f32[512, 768]" = torch.ops.aten.view.default(add_55, [512, 768]);  add_55 = None
    permute_86: "f32[768, 3072]" = torch.ops.aten.permute.default(arg142_1, [1, 0]);  arg142_1 = None
    addmm_46: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg143_1, view_172, permute_86);  arg143_1 = view_172 = permute_86 = None
    view_173: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_46, [4, 128, 3072]);  addmm_46 = None
    mul_37: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.5)
    mul_38: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476);  view_173 = None
    erf_7: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
    add_56: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
    mul_39: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_37, add_56);  mul_37 = add_56 = None
    clone_70: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_39);  mul_39 = None
    view_174: "f32[512, 3072]" = torch.ops.aten.view.default(clone_70, [512, 3072]);  clone_70 = None
    permute_87: "f32[3072, 768]" = torch.ops.aten.permute.default(arg144_1, [1, 0]);  arg144_1 = None
    addmm_47: "f32[512, 768]" = torch.ops.aten.addmm.default(arg145_1, view_174, permute_87);  arg145_1 = view_174 = permute_87 = None
    view_175: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_47, [4, 128, 768]);  addmm_47 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_71: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_175);  view_175 = None
    add_57: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_53, clone_71);  add_53 = clone_71 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_72: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_57);  add_57 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_16: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_72, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_16: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_72, [-1], correction = 1.0, keepdim = True)
    sqrt_16: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_16);  var_16 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_24: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_72, mean_16);  mean_16 = None
    mul_40: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg32_1, sub_24);  arg32_1 = sub_24 = None
    add_58: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_16, 1e-06);  sqrt_16 = None
    div_32: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_40, add_58);  mul_40 = add_58 = None
    add_59: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_32, arg33_1);  div_32 = arg33_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_176: "f32[512, 768]" = torch.ops.aten.view.default(add_59, [512, 768])
    permute_88: "f32[768, 768]" = torch.ops.aten.permute.default(arg146_1, [1, 0]);  arg146_1 = None
    addmm_48: "f32[512, 768]" = torch.ops.aten.addmm.default(arg147_1, view_176, permute_88);  arg147_1 = view_176 = permute_88 = None
    view_177: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_48, [4, 128, 768]);  addmm_48 = None
    view_178: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_177, [4, -1, 12, 64]);  view_177 = None
    permute_89: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
    view_179: "f32[512, 768]" = torch.ops.aten.view.default(add_59, [512, 768])
    permute_90: "f32[768, 768]" = torch.ops.aten.permute.default(arg148_1, [1, 0]);  arg148_1 = None
    addmm_49: "f32[512, 768]" = torch.ops.aten.addmm.default(arg149_1, view_179, permute_90);  arg149_1 = view_179 = permute_90 = None
    view_180: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_49, [4, 128, 768]);  addmm_49 = None
    view_181: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_180, [4, -1, 12, 64]);  view_180 = None
    permute_91: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_181, [0, 2, 1, 3]);  view_181 = None
    view_182: "f32[512, 768]" = torch.ops.aten.view.default(add_59, [512, 768]);  add_59 = None
    permute_92: "f32[768, 768]" = torch.ops.aten.permute.default(arg150_1, [1, 0]);  arg150_1 = None
    addmm_50: "f32[512, 768]" = torch.ops.aten.addmm.default(arg151_1, view_182, permute_92);  arg151_1 = view_182 = permute_92 = None
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
    bmm_16: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_185, view_186);  view_185 = view_186 = None
    view_187: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_16, [4, 12, 128, 128]);  bmm_16 = None
    div_33: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_187, 8.0);  view_187 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_8: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_8: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_8: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_8, scalar_tensor_8, div_33);  eq_8 = scalar_tensor_8 = div_33 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_8: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_8, [-1], True)
    sub_25: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_8, amax_8);  where_8 = amax_8 = None
    exp_8: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
    sum_9: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
    div_34: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_75: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_34);  div_34 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_34: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_75, [4, 12, 128, 128]);  clone_75 = None
    view_188: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_34, [48, 128, 128]);  expand_34 = None
    expand_35: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_93, [4, 12, 128, 64]);  permute_93 = None
    clone_76: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
    view_189: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_76, [48, 128, 64]);  clone_76 = None
    bmm_17: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_188, view_189);  view_188 = view_189 = None
    view_190: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_17, [4, 12, 128, 64]);  bmm_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_95: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_190, [0, 2, 1, 3]);  view_190 = None
    clone_77: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_95, memory_format = torch.contiguous_format);  permute_95 = None
    view_191: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_77, [4, -1, 768]);  clone_77 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_192: "f32[512, 768]" = torch.ops.aten.view.default(view_191, [512, 768]);  view_191 = None
    permute_96: "f32[768, 768]" = torch.ops.aten.permute.default(arg152_1, [1, 0]);  arg152_1 = None
    addmm_51: "f32[512, 768]" = torch.ops.aten.addmm.default(arg153_1, view_192, permute_96);  arg153_1 = view_192 = permute_96 = None
    view_193: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_51, [4, 128, 768]);  addmm_51 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_78: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_193);  view_193 = None
    add_60: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_72, clone_78);  clone_72 = clone_78 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_17: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_60, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_17: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_60, [-1], correction = 1.0, keepdim = True)
    sqrt_17: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_17);  var_17 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_26: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_60, mean_17);  mean_17 = None
    mul_41: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg34_1, sub_26);  arg34_1 = sub_26 = None
    add_61: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_17, 1e-06);  sqrt_17 = None
    div_35: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_41, add_61);  mul_41 = add_61 = None
    add_62: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_35, arg35_1);  div_35 = arg35_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_194: "f32[512, 768]" = torch.ops.aten.view.default(add_62, [512, 768]);  add_62 = None
    permute_97: "f32[768, 3072]" = torch.ops.aten.permute.default(arg154_1, [1, 0]);  arg154_1 = None
    addmm_52: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg155_1, view_194, permute_97);  arg155_1 = view_194 = permute_97 = None
    view_195: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_52, [4, 128, 3072]);  addmm_52 = None
    mul_42: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.5)
    mul_43: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476);  view_195 = None
    erf_8: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_43);  mul_43 = None
    add_63: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
    mul_44: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_42, add_63);  mul_42 = add_63 = None
    clone_79: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_44);  mul_44 = None
    view_196: "f32[512, 3072]" = torch.ops.aten.view.default(clone_79, [512, 3072]);  clone_79 = None
    permute_98: "f32[3072, 768]" = torch.ops.aten.permute.default(arg156_1, [1, 0]);  arg156_1 = None
    addmm_53: "f32[512, 768]" = torch.ops.aten.addmm.default(arg157_1, view_196, permute_98);  arg157_1 = view_196 = permute_98 = None
    view_197: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_53, [4, 128, 768]);  addmm_53 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_80: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_197);  view_197 = None
    add_64: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_60, clone_80);  add_60 = clone_80 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_81: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_64);  add_64 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_18: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_81, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_18: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_81, [-1], correction = 1.0, keepdim = True)
    sqrt_18: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_18);  var_18 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_27: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_81, mean_18);  mean_18 = None
    mul_45: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg36_1, sub_27);  arg36_1 = sub_27 = None
    add_65: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_18, 1e-06);  sqrt_18 = None
    div_36: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_45, add_65);  mul_45 = add_65 = None
    add_66: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_36, arg37_1);  div_36 = arg37_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_198: "f32[512, 768]" = torch.ops.aten.view.default(add_66, [512, 768])
    permute_99: "f32[768, 768]" = torch.ops.aten.permute.default(arg158_1, [1, 0]);  arg158_1 = None
    addmm_54: "f32[512, 768]" = torch.ops.aten.addmm.default(arg159_1, view_198, permute_99);  arg159_1 = view_198 = permute_99 = None
    view_199: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_54, [4, 128, 768]);  addmm_54 = None
    view_200: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_199, [4, -1, 12, 64]);  view_199 = None
    permute_100: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None
    view_201: "f32[512, 768]" = torch.ops.aten.view.default(add_66, [512, 768])
    permute_101: "f32[768, 768]" = torch.ops.aten.permute.default(arg160_1, [1, 0]);  arg160_1 = None
    addmm_55: "f32[512, 768]" = torch.ops.aten.addmm.default(arg161_1, view_201, permute_101);  arg161_1 = view_201 = permute_101 = None
    view_202: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_55, [4, 128, 768]);  addmm_55 = None
    view_203: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_202, [4, -1, 12, 64]);  view_202 = None
    permute_102: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_203, [0, 2, 1, 3]);  view_203 = None
    view_204: "f32[512, 768]" = torch.ops.aten.view.default(add_66, [512, 768]);  add_66 = None
    permute_103: "f32[768, 768]" = torch.ops.aten.permute.default(arg162_1, [1, 0]);  arg162_1 = None
    addmm_56: "f32[512, 768]" = torch.ops.aten.addmm.default(arg163_1, view_204, permute_103);  arg163_1 = view_204 = permute_103 = None
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
    bmm_18: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_207, view_208);  view_207 = view_208 = None
    view_209: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_18, [4, 12, 128, 128]);  bmm_18 = None
    div_37: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_209, 8.0);  view_209 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_9: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_9: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_9: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_9, scalar_tensor_9, div_37);  eq_9 = scalar_tensor_9 = div_37 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_9: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_9, [-1], True)
    sub_28: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_9, amax_9);  where_9 = amax_9 = None
    exp_9: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_28);  sub_28 = None
    sum_10: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
    div_38: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_84: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_38);  div_38 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_38: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_84, [4, 12, 128, 128]);  clone_84 = None
    view_210: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_38, [48, 128, 128]);  expand_38 = None
    expand_39: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_104, [4, 12, 128, 64]);  permute_104 = None
    clone_85: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
    view_211: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_85, [48, 128, 64]);  clone_85 = None
    bmm_19: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_210, view_211);  view_210 = view_211 = None
    view_212: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_19, [4, 12, 128, 64]);  bmm_19 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_106: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_212, [0, 2, 1, 3]);  view_212 = None
    clone_86: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_106, memory_format = torch.contiguous_format);  permute_106 = None
    view_213: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_86, [4, -1, 768]);  clone_86 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_214: "f32[512, 768]" = torch.ops.aten.view.default(view_213, [512, 768]);  view_213 = None
    permute_107: "f32[768, 768]" = torch.ops.aten.permute.default(arg164_1, [1, 0]);  arg164_1 = None
    addmm_57: "f32[512, 768]" = torch.ops.aten.addmm.default(arg165_1, view_214, permute_107);  arg165_1 = view_214 = permute_107 = None
    view_215: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_57, [4, 128, 768]);  addmm_57 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_87: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_215);  view_215 = None
    add_67: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_81, clone_87);  clone_81 = clone_87 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_19: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_67, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_19: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_67, [-1], correction = 1.0, keepdim = True)
    sqrt_19: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_19);  var_19 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_29: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_67, mean_19);  mean_19 = None
    mul_46: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg38_1, sub_29);  arg38_1 = sub_29 = None
    add_68: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_19, 1e-06);  sqrt_19 = None
    div_39: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_46, add_68);  mul_46 = add_68 = None
    add_69: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_39, arg39_1);  div_39 = arg39_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_216: "f32[512, 768]" = torch.ops.aten.view.default(add_69, [512, 768]);  add_69 = None
    permute_108: "f32[768, 3072]" = torch.ops.aten.permute.default(arg166_1, [1, 0]);  arg166_1 = None
    addmm_58: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg167_1, view_216, permute_108);  arg167_1 = view_216 = permute_108 = None
    view_217: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_58, [4, 128, 3072]);  addmm_58 = None
    mul_47: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.5)
    mul_48: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476);  view_217 = None
    erf_9: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_48);  mul_48 = None
    add_70: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
    mul_49: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_47, add_70);  mul_47 = add_70 = None
    clone_88: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_49);  mul_49 = None
    view_218: "f32[512, 3072]" = torch.ops.aten.view.default(clone_88, [512, 3072]);  clone_88 = None
    permute_109: "f32[3072, 768]" = torch.ops.aten.permute.default(arg168_1, [1, 0]);  arg168_1 = None
    addmm_59: "f32[512, 768]" = torch.ops.aten.addmm.default(arg169_1, view_218, permute_109);  arg169_1 = view_218 = permute_109 = None
    view_219: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_59, [4, 128, 768]);  addmm_59 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_89: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_219);  view_219 = None
    add_71: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_67, clone_89);  add_67 = clone_89 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_90: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_71);  add_71 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_20: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_90, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_20: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_90, [-1], correction = 1.0, keepdim = True)
    sqrt_20: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_20);  var_20 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_30: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_90, mean_20);  mean_20 = None
    mul_50: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg40_1, sub_30);  arg40_1 = sub_30 = None
    add_72: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_20, 1e-06);  sqrt_20 = None
    div_40: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_50, add_72);  mul_50 = add_72 = None
    add_73: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_40, arg41_1);  div_40 = arg41_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_220: "f32[512, 768]" = torch.ops.aten.view.default(add_73, [512, 768])
    permute_110: "f32[768, 768]" = torch.ops.aten.permute.default(arg170_1, [1, 0]);  arg170_1 = None
    addmm_60: "f32[512, 768]" = torch.ops.aten.addmm.default(arg171_1, view_220, permute_110);  arg171_1 = view_220 = permute_110 = None
    view_221: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_60, [4, 128, 768]);  addmm_60 = None
    view_222: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_221, [4, -1, 12, 64]);  view_221 = None
    permute_111: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None
    view_223: "f32[512, 768]" = torch.ops.aten.view.default(add_73, [512, 768])
    permute_112: "f32[768, 768]" = torch.ops.aten.permute.default(arg172_1, [1, 0]);  arg172_1 = None
    addmm_61: "f32[512, 768]" = torch.ops.aten.addmm.default(arg173_1, view_223, permute_112);  arg173_1 = view_223 = permute_112 = None
    view_224: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_61, [4, 128, 768]);  addmm_61 = None
    view_225: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_224, [4, -1, 12, 64]);  view_224 = None
    permute_113: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_225, [0, 2, 1, 3]);  view_225 = None
    view_226: "f32[512, 768]" = torch.ops.aten.view.default(add_73, [512, 768]);  add_73 = None
    permute_114: "f32[768, 768]" = torch.ops.aten.permute.default(arg174_1, [1, 0]);  arg174_1 = None
    addmm_62: "f32[512, 768]" = torch.ops.aten.addmm.default(arg175_1, view_226, permute_114);  arg175_1 = view_226 = permute_114 = None
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
    bmm_20: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_229, view_230);  view_229 = view_230 = None
    view_231: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_20, [4, 12, 128, 128]);  bmm_20 = None
    div_41: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_231, 8.0);  view_231 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_10: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_10: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_10: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_10, scalar_tensor_10, div_41);  eq_10 = scalar_tensor_10 = div_41 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_10: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_10, [-1], True)
    sub_31: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_10, amax_10);  where_10 = amax_10 = None
    exp_10: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_31);  sub_31 = None
    sum_11: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
    div_42: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_93: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_42);  div_42 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_42: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_93, [4, 12, 128, 128]);  clone_93 = None
    view_232: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_42, [48, 128, 128]);  expand_42 = None
    expand_43: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_115, [4, 12, 128, 64]);  permute_115 = None
    clone_94: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
    view_233: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_94, [48, 128, 64]);  clone_94 = None
    bmm_21: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_232, view_233);  view_232 = view_233 = None
    view_234: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_21, [4, 12, 128, 64]);  bmm_21 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_117: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None
    clone_95: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_117, memory_format = torch.contiguous_format);  permute_117 = None
    view_235: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_95, [4, -1, 768]);  clone_95 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_236: "f32[512, 768]" = torch.ops.aten.view.default(view_235, [512, 768]);  view_235 = None
    permute_118: "f32[768, 768]" = torch.ops.aten.permute.default(arg176_1, [1, 0]);  arg176_1 = None
    addmm_63: "f32[512, 768]" = torch.ops.aten.addmm.default(arg177_1, view_236, permute_118);  arg177_1 = view_236 = permute_118 = None
    view_237: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_63, [4, 128, 768]);  addmm_63 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_96: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_237);  view_237 = None
    add_74: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_90, clone_96);  clone_90 = clone_96 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_21: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_74, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_21: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_74, [-1], correction = 1.0, keepdim = True)
    sqrt_21: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_21);  var_21 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_32: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_74, mean_21);  mean_21 = None
    mul_51: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg42_1, sub_32);  arg42_1 = sub_32 = None
    add_75: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_21, 1e-06);  sqrt_21 = None
    div_43: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_51, add_75);  mul_51 = add_75 = None
    add_76: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_43, arg43_1);  div_43 = arg43_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_238: "f32[512, 768]" = torch.ops.aten.view.default(add_76, [512, 768]);  add_76 = None
    permute_119: "f32[768, 3072]" = torch.ops.aten.permute.default(arg178_1, [1, 0]);  arg178_1 = None
    addmm_64: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg179_1, view_238, permute_119);  arg179_1 = view_238 = permute_119 = None
    view_239: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_64, [4, 128, 3072]);  addmm_64 = None
    mul_52: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.5)
    mul_53: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476);  view_239 = None
    erf_10: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_53);  mul_53 = None
    add_77: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
    mul_54: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_52, add_77);  mul_52 = add_77 = None
    clone_97: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_54);  mul_54 = None
    view_240: "f32[512, 3072]" = torch.ops.aten.view.default(clone_97, [512, 3072]);  clone_97 = None
    permute_120: "f32[3072, 768]" = torch.ops.aten.permute.default(arg180_1, [1, 0]);  arg180_1 = None
    addmm_65: "f32[512, 768]" = torch.ops.aten.addmm.default(arg181_1, view_240, permute_120);  arg181_1 = view_240 = permute_120 = None
    view_241: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_65, [4, 128, 768]);  addmm_65 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_98: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_241);  view_241 = None
    add_78: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_74, clone_98);  add_74 = clone_98 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_99: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_78);  add_78 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_22: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(clone_99, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_22: "f32[4, 128, 1]" = torch.ops.aten.var.correction(clone_99, [-1], correction = 1.0, keepdim = True)
    sqrt_22: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_22);  var_22 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_33: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(clone_99, mean_22);  mean_22 = None
    mul_55: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg44_1, sub_33);  arg44_1 = sub_33 = None
    add_79: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_22, 1e-06);  sqrt_22 = None
    div_44: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_55, add_79);  mul_55 = add_79 = None
    add_80: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_44, arg45_1);  div_44 = arg45_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:37, code: query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    view_242: "f32[512, 768]" = torch.ops.aten.view.default(add_80, [512, 768])
    permute_121: "f32[768, 768]" = torch.ops.aten.permute.default(arg182_1, [1, 0]);  arg182_1 = None
    addmm_66: "f32[512, 768]" = torch.ops.aten.addmm.default(arg183_1, view_242, permute_121);  arg183_1 = view_242 = permute_121 = None
    view_243: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_66, [4, 128, 768]);  addmm_66 = None
    view_244: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_243, [4, -1, 12, 64]);  view_243 = None
    permute_122: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_244, [0, 2, 1, 3]);  view_244 = None
    view_245: "f32[512, 768]" = torch.ops.aten.view.default(add_80, [512, 768])
    permute_123: "f32[768, 768]" = torch.ops.aten.permute.default(arg184_1, [1, 0]);  arg184_1 = None
    addmm_67: "f32[512, 768]" = torch.ops.aten.addmm.default(arg185_1, view_245, permute_123);  arg185_1 = view_245 = permute_123 = None
    view_246: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_67, [4, 128, 768]);  addmm_67 = None
    view_247: "f32[4, 128, 12, 64]" = torch.ops.aten.view.default(view_246, [4, -1, 12, 64]);  view_246 = None
    permute_124: "f32[4, 12, 128, 64]" = torch.ops.aten.permute.default(view_247, [0, 2, 1, 3]);  view_247 = None
    view_248: "f32[512, 768]" = torch.ops.aten.view.default(add_80, [512, 768]);  add_80 = None
    permute_125: "f32[768, 768]" = torch.ops.aten.permute.default(arg186_1, [1, 0]);  arg186_1 = None
    addmm_68: "f32[512, 768]" = torch.ops.aten.addmm.default(arg187_1, view_248, permute_125);  arg187_1 = view_248 = permute_125 = None
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
    bmm_22: "f32[48, 128, 128]" = torch.ops.aten.bmm.default(view_251, view_252);  view_251 = view_252 = None
    view_253: "f32[4, 12, 128, 128]" = torch.ops.aten.view.default(bmm_22, [4, 12, 128, 128]);  bmm_22 = None
    div_45: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(view_253, 8.0);  view_253 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:27, code: scores = scores.masked_fill(mask == 0, min_mask)
    eq_11: "b8[4, 1, 128, 128]" = torch.ops.aten.eq.Scalar(unsqueeze_1, 0)
    scalar_tensor_11: "f32[]" = torch.ops.aten.scalar_tensor.default(-1000000000.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
    where_11: "f32[4, 12, 128, 128]" = torch.ops.aten.where.self(eq_11, scalar_tensor_11, div_45);  eq_11 = scalar_tensor_11 = div_45 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:29, code: p_attn = F.softmax(scores, dim=-1)
    amax_11: "f32[4, 12, 128, 1]" = torch.ops.aten.amax.default(where_11, [-1], True)
    sub_34: "f32[4, 12, 128, 128]" = torch.ops.aten.sub.Tensor(where_11, amax_11);  where_11 = amax_11 = None
    exp_11: "f32[4, 12, 128, 128]" = torch.ops.aten.exp.default(sub_34);  sub_34 = None
    sum_12: "f32[4, 12, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
    div_46: "f32[4, 12, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:12, code: return self.dropout(x)
    clone_102: "f32[4, 12, 128, 128]" = torch.ops.aten.clone.default(div_46);  div_46 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/single.py:33, code: return torch.matmul(p_attn, value), p_attn
    expand_46: "f32[4, 12, 128, 128]" = torch.ops.aten.expand.default(clone_102, [4, 12, 128, 128]);  clone_102 = None
    view_254: "f32[48, 128, 128]" = torch.ops.aten.view.default(expand_46, [48, 128, 128]);  expand_46 = None
    expand_47: "f32[4, 12, 128, 64]" = torch.ops.aten.expand.default(permute_126, [4, 12, 128, 64]);  permute_126 = None
    clone_103: "f32[4, 12, 128, 64]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
    view_255: "f32[48, 128, 64]" = torch.ops.aten.view.default(clone_103, [48, 128, 64]);  clone_103 = None
    bmm_23: "f32[48, 128, 64]" = torch.ops.aten.bmm.default(view_254, view_255);  view_254 = view_255 = None
    view_256: "f32[4, 12, 128, 64]" = torch.ops.aten.view.default(bmm_23, [4, 12, 128, 64]);  bmm_23 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:44, code: x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
    permute_128: "f32[4, 128, 12, 64]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
    clone_104: "f32[4, 128, 12, 64]" = torch.ops.aten.clone.default(permute_128, memory_format = torch.contiguous_format);  permute_128 = None
    view_257: "f32[4, 128, 768]" = torch.ops.aten.view.default(clone_104, [4, -1, 768]);  clone_104 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/attention/multi_head.py:46, code: return self.output_linear(x)
    view_258: "f32[512, 768]" = torch.ops.aten.view.default(view_257, [512, 768]);  view_257 = None
    permute_129: "f32[768, 768]" = torch.ops.aten.permute.default(arg188_1, [1, 0]);  arg188_1 = None
    addmm_69: "f32[512, 768]" = torch.ops.aten.addmm.default(arg189_1, view_258, permute_129);  arg189_1 = view_258 = permute_129 = None
    view_259: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_69, [4, 128, 768]);  addmm_69 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_105: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_259);  view_259 = None
    add_81: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(clone_99, clone_105);  clone_99 = clone_105 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:15, code: mean = x.mean(-1, keepdim=True)
    mean_23: "f32[4, 128, 1]" = torch.ops.aten.mean.dim(add_81, [-1], True)
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:16, code: std = x.std(-1, keepdim=True)
    var_23: "f32[4, 128, 1]" = torch.ops.aten.var.correction(add_81, [-1], correction = 1.0, keepdim = True)
    sqrt_23: "f32[4, 128, 1]" = torch.ops.aten.sqrt.default(var_23);  var_23 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/layer_norm.py:17, code: return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    sub_35: "f32[4, 128, 768]" = torch.ops.aten.sub.Tensor(add_81, mean_23);  mean_23 = None
    mul_56: "f32[4, 128, 768]" = torch.ops.aten.mul.Tensor(arg46_1, sub_35);  arg46_1 = sub_35 = None
    add_82: "f32[4, 128, 1]" = torch.ops.aten.add.Tensor(sqrt_23, 1e-06);  sqrt_23 = None
    div_47: "f32[4, 128, 768]" = torch.ops.aten.div.Tensor(mul_56, add_82);  mul_56 = add_82 = None
    add_83: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(div_47, arg47_1);  div_47 = arg47_1 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/feed_forward.py:16, code: return self.w_2(self.dropout(self.activation(self.w_1(x))))
    view_260: "f32[512, 768]" = torch.ops.aten.view.default(add_83, [512, 768]);  add_83 = None
    permute_130: "f32[768, 3072]" = torch.ops.aten.permute.default(arg190_1, [1, 0]);  arg190_1 = None
    addmm_70: "f32[512, 3072]" = torch.ops.aten.addmm.default(arg191_1, view_260, permute_130);  arg191_1 = view_260 = permute_130 = None
    view_261: "f32[4, 128, 3072]" = torch.ops.aten.view.default(addmm_70, [4, 128, 3072]);  addmm_70 = None
    mul_57: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
    mul_58: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476);  view_261 = None
    erf_11: "f32[4, 128, 3072]" = torch.ops.aten.erf.default(mul_58);  mul_58 = None
    add_84: "f32[4, 128, 3072]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
    mul_59: "f32[4, 128, 3072]" = torch.ops.aten.mul.Tensor(mul_57, add_84);  mul_57 = add_84 = None
    clone_106: "f32[4, 128, 3072]" = torch.ops.aten.clone.default(mul_59);  mul_59 = None
    view_262: "f32[512, 3072]" = torch.ops.aten.view.default(clone_106, [512, 3072]);  clone_106 = None
    permute_131: "f32[3072, 768]" = torch.ops.aten.permute.default(arg192_1, [1, 0]);  arg192_1 = None
    addmm_71: "f32[512, 768]" = torch.ops.aten.addmm.default(arg193_1, view_262, permute_131);  arg193_1 = view_262 = permute_131 = None
    view_263: "f32[4, 128, 768]" = torch.ops.aten.view.default(addmm_71, [4, 128, 768]);  addmm_71 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/utils/sublayer.py:19, code: return x + self.dropout(sublayer.forward(self.norm(x)))
    clone_107: "f32[4, 128, 768]" = torch.ops.aten.clone.default(view_263);  view_263 = None
    add_85: "f32[4, 128, 768]" = torch.ops.aten.add.Tensor(add_81, clone_107);  add_81 = clone_107 = None
    
    # File: /workspace/youkaichao/code/torchbenchmark/torchbenchmark/models/BERT_pytorch/bert_pytorch/model/transformer.py:49, code: return self.dropout(x)
    clone_108: "f32[4, 128, 768]" = torch.ops.aten.clone.default(add_85);  add_85 = None
    return (clone_108, unsqueeze_1)
    