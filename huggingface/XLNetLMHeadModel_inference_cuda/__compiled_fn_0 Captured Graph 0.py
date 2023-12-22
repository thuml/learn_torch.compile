from __future__ import annotations



def forward(self, L_inputs_input_ids_ : torch.Tensor, L_inputs_labels_ : torch.Tensor):
    l_inputs_input_ids_ = L_inputs_input_ids_
    l_inputs_labels_ = L_inputs_labels_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1107, code: input_ids = input_ids.transpose(0, 1).contiguous()
    transpose = l_inputs_input_ids_.transpose(0, 1);  l_inputs_input_ids_ = None
    input_ids = transpose.contiguous();  transpose = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1176, code: word_emb_k = self.word_embedding(input_ids)
    word_emb_k = self.L__mod___transformer_word_embedding(input_ids);  input_ids = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1177, code: output_h = self.dropout(word_emb_k)
    cat_1 = self.L__mod___transformer_dropout(word_emb_k);  word_emb_k = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1023, code: freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.float)
    freq_seq = torch.arange(0, 1024, 2.0, dtype = torch.float32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1024, code: inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))
    truediv = freq_seq / 1024;  freq_seq = None
    pow_1 = torch.pow(10000, truediv);  truediv = None
    inv_freq = 1 / pow_1;  pow_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1052, code: fwd_pos_seq = torch.arange(beg, end, -1.0)
    fwd_pos_seq = torch.arange(512, -512, -1.0)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1012, code: sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
    sinusoid_inp = torch.functional.einsum('i,d->id', fwd_pos_seq, inv_freq);  fwd_pos_seq = inv_freq = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1013, code: pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp);  sinusoid_inp = None
    pos_emb = torch.cat([sin, cos], dim = -1);  sin = cos = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1014, code: pos_emb = pos_emb[:, None, :]
    pos_emb_1 = pos_emb[(slice(None, None, None), None, slice(None, None, None))];  pos_emb = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1017, code: pos_emb = pos_emb.expand(-1, bsz, -1)
    pos_emb_4 = pos_emb_1.expand(-1, 1, -1);  pos_emb_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1204, code: pos_emb = pos_emb.to(output_h.device)
    pos_emb_5 = pos_emb_4.to(device(type='cuda', index=0));  pos_emb_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1205, code: pos_emb = self.dropout(pos_emb)
    pos_emb_6 = self.L__mod___transformer_dropout(pos_emb_5);  pos_emb_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem = cat_1[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach = new_mem.detach();  new_mem = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_0_rel_attn_q_2 = self.L__mod___transformer_layer_0_rel_attn_q
    q_head_h = torch.functional.einsum('ibh,hnd->ibnd', cat_1, l__mod___transformer_layer_0_rel_attn_q_2);  l__mod___transformer_layer_0_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_0_rel_attn_k_2 = self.L__mod___transformer_layer_0_rel_attn_k
    k_head_h = torch.functional.einsum('ibh,hnd->ibnd', cat_1, l__mod___transformer_layer_0_rel_attn_k_2);  l__mod___transformer_layer_0_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_0_rel_attn_v_2 = self.L__mod___transformer_layer_0_rel_attn_v
    v_head_h = torch.functional.einsum('ibh,hnd->ibnd', cat_1, l__mod___transformer_layer_0_rel_attn_v_2);  l__mod___transformer_layer_0_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_0_rel_attn_r_2 = self.L__mod___transformer_layer_0_rel_attn_r
    type_1 = pos_emb_6.type(torch.float32)
    k_head_r = torch.functional.einsum('ibh,hnd->ibnd', type_1, l__mod___transformer_layer_0_rel_attn_r_2);  type_1 = l__mod___transformer_layer_0_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_0_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_0_rel_attn_r_w_bias
    add = q_head_h + l__mod___transformer_layer_0_rel_attn_r_w_bias_2;  l__mod___transformer_layer_0_rel_attn_r_w_bias_2 = None
    ac = torch.functional.einsum('ibnd,jbnd->bnij', add, k_head_h);  add = k_head_h = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_0_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_0_rel_attn_r_r_bias
    add_1 = q_head_h + l__mod___transformer_layer_0_rel_attn_r_r_bias_2;  q_head_h = l__mod___transformer_layer_0_rel_attn_r_r_bias_2 = None
    bd = torch.functional.einsum('ibnd,jbnd->bnij', add_1, k_head_r);  add_1 = k_head_r = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x = bd.reshape(1, 16, 1024, 512);  bd = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_1 = x[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_2 = x_1.reshape(1, 16, 512, 1023);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_2 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_1 = torch.index_select(x_2, 3, arange_2);  x_2 = arange_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_2 = ac + bd_1;  ac = bd_1 = None
    add_3 = add_2 + 0;  add_2 = None
    attn_score = add_3 * 0.125;  add_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob = torch.nn.functional.softmax(attn_score, dim = 3);  attn_score = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_1 = self.L__mod___transformer_layer_0_rel_attn_dropout(attn_prob);  attn_prob = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_1, v_head_h);  attn_prob_1 = v_head_h = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_0_rel_attn_o_2 = self.L__mod___transformer_layer_0_rel_attn_o
    attn_out = torch.functional.einsum('ibnd,hnd->ibh', attn_vec, l__mod___transformer_layer_0_rel_attn_o_2);  attn_vec = l__mod___transformer_layer_0_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_1 = self.L__mod___transformer_layer_0_rel_attn_dropout(attn_out);  attn_out = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_2 = attn_out_1 + cat_1;  attn_out_1 = cat_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_1 = self.L__mod___transformer_layer_0_rel_attn_layer_norm(attn_out_2);  attn_out_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_2 = self.L__mod___transformer_layer_0_ff_layer_1(output_1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_3 = torch._C._nn.gelu(output_2);  output_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_4 = self.L__mod___transformer_layer_0_ff_dropout(output_3);  output_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_5 = self.L__mod___transformer_layer_0_ff_layer_2(output_4);  output_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_6 = self.L__mod___transformer_layer_0_ff_dropout(output_5);  output_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_5 = output_6 + output_1;  output_6 = output_1 = None
    cat_2 = self.L__mod___transformer_layer_0_ff_layer_norm(add_5);  add_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_1 = cat_2[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_1 = new_mem_1.detach();  new_mem_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_1_rel_attn_q_2 = self.L__mod___transformer_layer_1_rel_attn_q
    q_head_h_1 = torch.functional.einsum('ibh,hnd->ibnd', cat_2, l__mod___transformer_layer_1_rel_attn_q_2);  l__mod___transformer_layer_1_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_1_rel_attn_k_2 = self.L__mod___transformer_layer_1_rel_attn_k
    k_head_h_1 = torch.functional.einsum('ibh,hnd->ibnd', cat_2, l__mod___transformer_layer_1_rel_attn_k_2);  l__mod___transformer_layer_1_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_1_rel_attn_v_2 = self.L__mod___transformer_layer_1_rel_attn_v
    v_head_h_1 = torch.functional.einsum('ibh,hnd->ibnd', cat_2, l__mod___transformer_layer_1_rel_attn_v_2);  l__mod___transformer_layer_1_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_1_rel_attn_r_2 = self.L__mod___transformer_layer_1_rel_attn_r
    type_2 = pos_emb_6.type(torch.float32)
    k_head_r_1 = torch.functional.einsum('ibh,hnd->ibnd', type_2, l__mod___transformer_layer_1_rel_attn_r_2);  type_2 = l__mod___transformer_layer_1_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_1_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_1_rel_attn_r_w_bias
    add_6 = q_head_h_1 + l__mod___transformer_layer_1_rel_attn_r_w_bias_2;  l__mod___transformer_layer_1_rel_attn_r_w_bias_2 = None
    ac_1 = torch.functional.einsum('ibnd,jbnd->bnij', add_6, k_head_h_1);  add_6 = k_head_h_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_1_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_1_rel_attn_r_r_bias
    add_7 = q_head_h_1 + l__mod___transformer_layer_1_rel_attn_r_r_bias_2;  q_head_h_1 = l__mod___transformer_layer_1_rel_attn_r_r_bias_2 = None
    bd_2 = torch.functional.einsum('ibnd,jbnd->bnij', add_7, k_head_r_1);  add_7 = k_head_r_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_4 = bd_2.reshape(1, 16, 1024, 512);  bd_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_5 = x_4[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_6 = x_5.reshape(1, 16, 512, 1023);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_3 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_3 = torch.index_select(x_6, 3, arange_3);  x_6 = arange_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_8 = ac_1 + bd_3;  ac_1 = bd_3 = None
    add_9 = add_8 + 0;  add_8 = None
    attn_score_1 = add_9 * 0.125;  add_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_2 = torch.nn.functional.softmax(attn_score_1, dim = 3);  attn_score_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_3 = self.L__mod___transformer_layer_1_rel_attn_dropout(attn_prob_2);  attn_prob_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_2 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_3, v_head_h_1);  attn_prob_3 = v_head_h_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_1_rel_attn_o_2 = self.L__mod___transformer_layer_1_rel_attn_o
    attn_out_3 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_2, l__mod___transformer_layer_1_rel_attn_o_2);  attn_vec_2 = l__mod___transformer_layer_1_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_4 = self.L__mod___transformer_layer_1_rel_attn_dropout(attn_out_3);  attn_out_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_5 = attn_out_4 + cat_2;  attn_out_4 = cat_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_9 = self.L__mod___transformer_layer_1_rel_attn_layer_norm(attn_out_5);  attn_out_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_10 = self.L__mod___transformer_layer_1_ff_layer_1(output_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_11 = torch._C._nn.gelu(output_10);  output_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_12 = self.L__mod___transformer_layer_1_ff_dropout(output_11);  output_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_13 = self.L__mod___transformer_layer_1_ff_layer_2(output_12);  output_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_14 = self.L__mod___transformer_layer_1_ff_dropout(output_13);  output_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_11 = output_14 + output_9;  output_14 = output_9 = None
    cat_3 = self.L__mod___transformer_layer_1_ff_layer_norm(add_11);  add_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_2 = cat_3[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_2 = new_mem_2.detach();  new_mem_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_2_rel_attn_q_2 = self.L__mod___transformer_layer_2_rel_attn_q
    q_head_h_2 = torch.functional.einsum('ibh,hnd->ibnd', cat_3, l__mod___transformer_layer_2_rel_attn_q_2);  l__mod___transformer_layer_2_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_2_rel_attn_k_2 = self.L__mod___transformer_layer_2_rel_attn_k
    k_head_h_2 = torch.functional.einsum('ibh,hnd->ibnd', cat_3, l__mod___transformer_layer_2_rel_attn_k_2);  l__mod___transformer_layer_2_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_2_rel_attn_v_2 = self.L__mod___transformer_layer_2_rel_attn_v
    v_head_h_2 = torch.functional.einsum('ibh,hnd->ibnd', cat_3, l__mod___transformer_layer_2_rel_attn_v_2);  l__mod___transformer_layer_2_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_2_rel_attn_r_2 = self.L__mod___transformer_layer_2_rel_attn_r
    type_3 = pos_emb_6.type(torch.float32)
    k_head_r_2 = torch.functional.einsum('ibh,hnd->ibnd', type_3, l__mod___transformer_layer_2_rel_attn_r_2);  type_3 = l__mod___transformer_layer_2_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_2_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_2_rel_attn_r_w_bias
    add_12 = q_head_h_2 + l__mod___transformer_layer_2_rel_attn_r_w_bias_2;  l__mod___transformer_layer_2_rel_attn_r_w_bias_2 = None
    ac_2 = torch.functional.einsum('ibnd,jbnd->bnij', add_12, k_head_h_2);  add_12 = k_head_h_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_2_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_2_rel_attn_r_r_bias
    add_13 = q_head_h_2 + l__mod___transformer_layer_2_rel_attn_r_r_bias_2;  q_head_h_2 = l__mod___transformer_layer_2_rel_attn_r_r_bias_2 = None
    bd_4 = torch.functional.einsum('ibnd,jbnd->bnij', add_13, k_head_r_2);  add_13 = k_head_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_8 = bd_4.reshape(1, 16, 1024, 512);  bd_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_9 = x_8[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_10 = x_9.reshape(1, 16, 512, 1023);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_4 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_5 = torch.index_select(x_10, 3, arange_4);  x_10 = arange_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_14 = ac_2 + bd_5;  ac_2 = bd_5 = None
    add_15 = add_14 + 0;  add_14 = None
    attn_score_2 = add_15 * 0.125;  add_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_4 = torch.nn.functional.softmax(attn_score_2, dim = 3);  attn_score_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_5 = self.L__mod___transformer_layer_2_rel_attn_dropout(attn_prob_4);  attn_prob_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_4 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_5, v_head_h_2);  attn_prob_5 = v_head_h_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_2_rel_attn_o_2 = self.L__mod___transformer_layer_2_rel_attn_o
    attn_out_6 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_4, l__mod___transformer_layer_2_rel_attn_o_2);  attn_vec_4 = l__mod___transformer_layer_2_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_7 = self.L__mod___transformer_layer_2_rel_attn_dropout(attn_out_6);  attn_out_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_8 = attn_out_7 + cat_3;  attn_out_7 = cat_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_17 = self.L__mod___transformer_layer_2_rel_attn_layer_norm(attn_out_8);  attn_out_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_18 = self.L__mod___transformer_layer_2_ff_layer_1(output_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_19 = torch._C._nn.gelu(output_18);  output_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_20 = self.L__mod___transformer_layer_2_ff_dropout(output_19);  output_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_21 = self.L__mod___transformer_layer_2_ff_layer_2(output_20);  output_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_22 = self.L__mod___transformer_layer_2_ff_dropout(output_21);  output_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_17 = output_22 + output_17;  output_22 = output_17 = None
    cat_4 = self.L__mod___transformer_layer_2_ff_layer_norm(add_17);  add_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_3 = cat_4[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_3 = new_mem_3.detach();  new_mem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_3_rel_attn_q_2 = self.L__mod___transformer_layer_3_rel_attn_q
    q_head_h_3 = torch.functional.einsum('ibh,hnd->ibnd', cat_4, l__mod___transformer_layer_3_rel_attn_q_2);  l__mod___transformer_layer_3_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_3_rel_attn_k_2 = self.L__mod___transformer_layer_3_rel_attn_k
    k_head_h_3 = torch.functional.einsum('ibh,hnd->ibnd', cat_4, l__mod___transformer_layer_3_rel_attn_k_2);  l__mod___transformer_layer_3_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_3_rel_attn_v_2 = self.L__mod___transformer_layer_3_rel_attn_v
    v_head_h_3 = torch.functional.einsum('ibh,hnd->ibnd', cat_4, l__mod___transformer_layer_3_rel_attn_v_2);  l__mod___transformer_layer_3_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_3_rel_attn_r_2 = self.L__mod___transformer_layer_3_rel_attn_r
    type_4 = pos_emb_6.type(torch.float32)
    k_head_r_3 = torch.functional.einsum('ibh,hnd->ibnd', type_4, l__mod___transformer_layer_3_rel_attn_r_2);  type_4 = l__mod___transformer_layer_3_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_3_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_3_rel_attn_r_w_bias
    add_18 = q_head_h_3 + l__mod___transformer_layer_3_rel_attn_r_w_bias_2;  l__mod___transformer_layer_3_rel_attn_r_w_bias_2 = None
    ac_3 = torch.functional.einsum('ibnd,jbnd->bnij', add_18, k_head_h_3);  add_18 = k_head_h_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_3_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_3_rel_attn_r_r_bias
    add_19 = q_head_h_3 + l__mod___transformer_layer_3_rel_attn_r_r_bias_2;  q_head_h_3 = l__mod___transformer_layer_3_rel_attn_r_r_bias_2 = None
    bd_6 = torch.functional.einsum('ibnd,jbnd->bnij', add_19, k_head_r_3);  add_19 = k_head_r_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_12 = bd_6.reshape(1, 16, 1024, 512);  bd_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_13 = x_12[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_14 = x_13.reshape(1, 16, 512, 1023);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_5 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_7 = torch.index_select(x_14, 3, arange_5);  x_14 = arange_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_20 = ac_3 + bd_7;  ac_3 = bd_7 = None
    add_21 = add_20 + 0;  add_20 = None
    attn_score_3 = add_21 * 0.125;  add_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_6 = torch.nn.functional.softmax(attn_score_3, dim = 3);  attn_score_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_7 = self.L__mod___transformer_layer_3_rel_attn_dropout(attn_prob_6);  attn_prob_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_6 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_7, v_head_h_3);  attn_prob_7 = v_head_h_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_3_rel_attn_o_2 = self.L__mod___transformer_layer_3_rel_attn_o
    attn_out_9 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_6, l__mod___transformer_layer_3_rel_attn_o_2);  attn_vec_6 = l__mod___transformer_layer_3_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_10 = self.L__mod___transformer_layer_3_rel_attn_dropout(attn_out_9);  attn_out_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_11 = attn_out_10 + cat_4;  attn_out_10 = cat_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_25 = self.L__mod___transformer_layer_3_rel_attn_layer_norm(attn_out_11);  attn_out_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_26 = self.L__mod___transformer_layer_3_ff_layer_1(output_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_27 = torch._C._nn.gelu(output_26);  output_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_28 = self.L__mod___transformer_layer_3_ff_dropout(output_27);  output_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_29 = self.L__mod___transformer_layer_3_ff_layer_2(output_28);  output_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_30 = self.L__mod___transformer_layer_3_ff_dropout(output_29);  output_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_23 = output_30 + output_25;  output_30 = output_25 = None
    cat_5 = self.L__mod___transformer_layer_3_ff_layer_norm(add_23);  add_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_4 = cat_5[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_4 = new_mem_4.detach();  new_mem_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_4_rel_attn_q_2 = self.L__mod___transformer_layer_4_rel_attn_q
    q_head_h_4 = torch.functional.einsum('ibh,hnd->ibnd', cat_5, l__mod___transformer_layer_4_rel_attn_q_2);  l__mod___transformer_layer_4_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_4_rel_attn_k_2 = self.L__mod___transformer_layer_4_rel_attn_k
    k_head_h_4 = torch.functional.einsum('ibh,hnd->ibnd', cat_5, l__mod___transformer_layer_4_rel_attn_k_2);  l__mod___transformer_layer_4_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_4_rel_attn_v_2 = self.L__mod___transformer_layer_4_rel_attn_v
    v_head_h_4 = torch.functional.einsum('ibh,hnd->ibnd', cat_5, l__mod___transformer_layer_4_rel_attn_v_2);  l__mod___transformer_layer_4_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_4_rel_attn_r_2 = self.L__mod___transformer_layer_4_rel_attn_r
    type_5 = pos_emb_6.type(torch.float32)
    k_head_r_4 = torch.functional.einsum('ibh,hnd->ibnd', type_5, l__mod___transformer_layer_4_rel_attn_r_2);  type_5 = l__mod___transformer_layer_4_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_4_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_4_rel_attn_r_w_bias
    add_24 = q_head_h_4 + l__mod___transformer_layer_4_rel_attn_r_w_bias_2;  l__mod___transformer_layer_4_rel_attn_r_w_bias_2 = None
    ac_4 = torch.functional.einsum('ibnd,jbnd->bnij', add_24, k_head_h_4);  add_24 = k_head_h_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_4_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_4_rel_attn_r_r_bias
    add_25 = q_head_h_4 + l__mod___transformer_layer_4_rel_attn_r_r_bias_2;  q_head_h_4 = l__mod___transformer_layer_4_rel_attn_r_r_bias_2 = None
    bd_8 = torch.functional.einsum('ibnd,jbnd->bnij', add_25, k_head_r_4);  add_25 = k_head_r_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_16 = bd_8.reshape(1, 16, 1024, 512);  bd_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_17 = x_16[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_18 = x_17.reshape(1, 16, 512, 1023);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_6 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_9 = torch.index_select(x_18, 3, arange_6);  x_18 = arange_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_26 = ac_4 + bd_9;  ac_4 = bd_9 = None
    add_27 = add_26 + 0;  add_26 = None
    attn_score_4 = add_27 * 0.125;  add_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_8 = torch.nn.functional.softmax(attn_score_4, dim = 3);  attn_score_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_9 = self.L__mod___transformer_layer_4_rel_attn_dropout(attn_prob_8);  attn_prob_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_8 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_9, v_head_h_4);  attn_prob_9 = v_head_h_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_4_rel_attn_o_2 = self.L__mod___transformer_layer_4_rel_attn_o
    attn_out_12 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_8, l__mod___transformer_layer_4_rel_attn_o_2);  attn_vec_8 = l__mod___transformer_layer_4_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_13 = self.L__mod___transformer_layer_4_rel_attn_dropout(attn_out_12);  attn_out_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_14 = attn_out_13 + cat_5;  attn_out_13 = cat_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_33 = self.L__mod___transformer_layer_4_rel_attn_layer_norm(attn_out_14);  attn_out_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_34 = self.L__mod___transformer_layer_4_ff_layer_1(output_33)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_35 = torch._C._nn.gelu(output_34);  output_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_36 = self.L__mod___transformer_layer_4_ff_dropout(output_35);  output_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_37 = self.L__mod___transformer_layer_4_ff_layer_2(output_36);  output_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_38 = self.L__mod___transformer_layer_4_ff_dropout(output_37);  output_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_29 = output_38 + output_33;  output_38 = output_33 = None
    cat_6 = self.L__mod___transformer_layer_4_ff_layer_norm(add_29);  add_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_5 = cat_6[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_5 = new_mem_5.detach();  new_mem_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_5_rel_attn_q_2 = self.L__mod___transformer_layer_5_rel_attn_q
    q_head_h_5 = torch.functional.einsum('ibh,hnd->ibnd', cat_6, l__mod___transformer_layer_5_rel_attn_q_2);  l__mod___transformer_layer_5_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_5_rel_attn_k_2 = self.L__mod___transformer_layer_5_rel_attn_k
    k_head_h_5 = torch.functional.einsum('ibh,hnd->ibnd', cat_6, l__mod___transformer_layer_5_rel_attn_k_2);  l__mod___transformer_layer_5_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_5_rel_attn_v_2 = self.L__mod___transformer_layer_5_rel_attn_v
    v_head_h_5 = torch.functional.einsum('ibh,hnd->ibnd', cat_6, l__mod___transformer_layer_5_rel_attn_v_2);  l__mod___transformer_layer_5_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_5_rel_attn_r_2 = self.L__mod___transformer_layer_5_rel_attn_r
    type_6 = pos_emb_6.type(torch.float32)
    k_head_r_5 = torch.functional.einsum('ibh,hnd->ibnd', type_6, l__mod___transformer_layer_5_rel_attn_r_2);  type_6 = l__mod___transformer_layer_5_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_5_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_5_rel_attn_r_w_bias
    add_30 = q_head_h_5 + l__mod___transformer_layer_5_rel_attn_r_w_bias_2;  l__mod___transformer_layer_5_rel_attn_r_w_bias_2 = None
    ac_5 = torch.functional.einsum('ibnd,jbnd->bnij', add_30, k_head_h_5);  add_30 = k_head_h_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_5_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_5_rel_attn_r_r_bias
    add_31 = q_head_h_5 + l__mod___transformer_layer_5_rel_attn_r_r_bias_2;  q_head_h_5 = l__mod___transformer_layer_5_rel_attn_r_r_bias_2 = None
    bd_10 = torch.functional.einsum('ibnd,jbnd->bnij', add_31, k_head_r_5);  add_31 = k_head_r_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_20 = bd_10.reshape(1, 16, 1024, 512);  bd_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_21 = x_20[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_22 = x_21.reshape(1, 16, 512, 1023);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_7 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_11 = torch.index_select(x_22, 3, arange_7);  x_22 = arange_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_32 = ac_5 + bd_11;  ac_5 = bd_11 = None
    add_33 = add_32 + 0;  add_32 = None
    attn_score_5 = add_33 * 0.125;  add_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_10 = torch.nn.functional.softmax(attn_score_5, dim = 3);  attn_score_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_11 = self.L__mod___transformer_layer_5_rel_attn_dropout(attn_prob_10);  attn_prob_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_10 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_11, v_head_h_5);  attn_prob_11 = v_head_h_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_5_rel_attn_o_2 = self.L__mod___transformer_layer_5_rel_attn_o
    attn_out_15 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_10, l__mod___transformer_layer_5_rel_attn_o_2);  attn_vec_10 = l__mod___transformer_layer_5_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_16 = self.L__mod___transformer_layer_5_rel_attn_dropout(attn_out_15);  attn_out_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_17 = attn_out_16 + cat_6;  attn_out_16 = cat_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_41 = self.L__mod___transformer_layer_5_rel_attn_layer_norm(attn_out_17);  attn_out_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_42 = self.L__mod___transformer_layer_5_ff_layer_1(output_41)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_43 = torch._C._nn.gelu(output_42);  output_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_44 = self.L__mod___transformer_layer_5_ff_dropout(output_43);  output_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_45 = self.L__mod___transformer_layer_5_ff_layer_2(output_44);  output_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_46 = self.L__mod___transformer_layer_5_ff_dropout(output_45);  output_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_35 = output_46 + output_41;  output_46 = output_41 = None
    cat_7 = self.L__mod___transformer_layer_5_ff_layer_norm(add_35);  add_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_6 = cat_7[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_6 = new_mem_6.detach();  new_mem_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_6_rel_attn_q_2 = self.L__mod___transformer_layer_6_rel_attn_q
    q_head_h_6 = torch.functional.einsum('ibh,hnd->ibnd', cat_7, l__mod___transformer_layer_6_rel_attn_q_2);  l__mod___transformer_layer_6_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_6_rel_attn_k_2 = self.L__mod___transformer_layer_6_rel_attn_k
    k_head_h_6 = torch.functional.einsum('ibh,hnd->ibnd', cat_7, l__mod___transformer_layer_6_rel_attn_k_2);  l__mod___transformer_layer_6_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_6_rel_attn_v_2 = self.L__mod___transformer_layer_6_rel_attn_v
    v_head_h_6 = torch.functional.einsum('ibh,hnd->ibnd', cat_7, l__mod___transformer_layer_6_rel_attn_v_2);  l__mod___transformer_layer_6_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_6_rel_attn_r_2 = self.L__mod___transformer_layer_6_rel_attn_r
    type_7 = pos_emb_6.type(torch.float32)
    k_head_r_6 = torch.functional.einsum('ibh,hnd->ibnd', type_7, l__mod___transformer_layer_6_rel_attn_r_2);  type_7 = l__mod___transformer_layer_6_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_6_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_6_rel_attn_r_w_bias
    add_36 = q_head_h_6 + l__mod___transformer_layer_6_rel_attn_r_w_bias_2;  l__mod___transformer_layer_6_rel_attn_r_w_bias_2 = None
    ac_6 = torch.functional.einsum('ibnd,jbnd->bnij', add_36, k_head_h_6);  add_36 = k_head_h_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_6_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_6_rel_attn_r_r_bias
    add_37 = q_head_h_6 + l__mod___transformer_layer_6_rel_attn_r_r_bias_2;  q_head_h_6 = l__mod___transformer_layer_6_rel_attn_r_r_bias_2 = None
    bd_12 = torch.functional.einsum('ibnd,jbnd->bnij', add_37, k_head_r_6);  add_37 = k_head_r_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_24 = bd_12.reshape(1, 16, 1024, 512);  bd_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_25 = x_24[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_26 = x_25.reshape(1, 16, 512, 1023);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_8 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_13 = torch.index_select(x_26, 3, arange_8);  x_26 = arange_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_38 = ac_6 + bd_13;  ac_6 = bd_13 = None
    add_39 = add_38 + 0;  add_38 = None
    attn_score_6 = add_39 * 0.125;  add_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_12 = torch.nn.functional.softmax(attn_score_6, dim = 3);  attn_score_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_13 = self.L__mod___transformer_layer_6_rel_attn_dropout(attn_prob_12);  attn_prob_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_12 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_13, v_head_h_6);  attn_prob_13 = v_head_h_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_6_rel_attn_o_2 = self.L__mod___transformer_layer_6_rel_attn_o
    attn_out_18 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_12, l__mod___transformer_layer_6_rel_attn_o_2);  attn_vec_12 = l__mod___transformer_layer_6_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_19 = self.L__mod___transformer_layer_6_rel_attn_dropout(attn_out_18);  attn_out_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_20 = attn_out_19 + cat_7;  attn_out_19 = cat_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_49 = self.L__mod___transformer_layer_6_rel_attn_layer_norm(attn_out_20);  attn_out_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_50 = self.L__mod___transformer_layer_6_ff_layer_1(output_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_51 = torch._C._nn.gelu(output_50);  output_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_52 = self.L__mod___transformer_layer_6_ff_dropout(output_51);  output_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_53 = self.L__mod___transformer_layer_6_ff_layer_2(output_52);  output_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_54 = self.L__mod___transformer_layer_6_ff_dropout(output_53);  output_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_41 = output_54 + output_49;  output_54 = output_49 = None
    cat_8 = self.L__mod___transformer_layer_6_ff_layer_norm(add_41);  add_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_7 = cat_8[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_7 = new_mem_7.detach();  new_mem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_7_rel_attn_q_2 = self.L__mod___transformer_layer_7_rel_attn_q
    q_head_h_7 = torch.functional.einsum('ibh,hnd->ibnd', cat_8, l__mod___transformer_layer_7_rel_attn_q_2);  l__mod___transformer_layer_7_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_7_rel_attn_k_2 = self.L__mod___transformer_layer_7_rel_attn_k
    k_head_h_7 = torch.functional.einsum('ibh,hnd->ibnd', cat_8, l__mod___transformer_layer_7_rel_attn_k_2);  l__mod___transformer_layer_7_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_7_rel_attn_v_2 = self.L__mod___transformer_layer_7_rel_attn_v
    v_head_h_7 = torch.functional.einsum('ibh,hnd->ibnd', cat_8, l__mod___transformer_layer_7_rel_attn_v_2);  l__mod___transformer_layer_7_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_7_rel_attn_r_2 = self.L__mod___transformer_layer_7_rel_attn_r
    type_8 = pos_emb_6.type(torch.float32)
    k_head_r_7 = torch.functional.einsum('ibh,hnd->ibnd', type_8, l__mod___transformer_layer_7_rel_attn_r_2);  type_8 = l__mod___transformer_layer_7_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_7_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_7_rel_attn_r_w_bias
    add_42 = q_head_h_7 + l__mod___transformer_layer_7_rel_attn_r_w_bias_2;  l__mod___transformer_layer_7_rel_attn_r_w_bias_2 = None
    ac_7 = torch.functional.einsum('ibnd,jbnd->bnij', add_42, k_head_h_7);  add_42 = k_head_h_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_7_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_7_rel_attn_r_r_bias
    add_43 = q_head_h_7 + l__mod___transformer_layer_7_rel_attn_r_r_bias_2;  q_head_h_7 = l__mod___transformer_layer_7_rel_attn_r_r_bias_2 = None
    bd_14 = torch.functional.einsum('ibnd,jbnd->bnij', add_43, k_head_r_7);  add_43 = k_head_r_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_28 = bd_14.reshape(1, 16, 1024, 512);  bd_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_29 = x_28[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_30 = x_29.reshape(1, 16, 512, 1023);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_9 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_15 = torch.index_select(x_30, 3, arange_9);  x_30 = arange_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_44 = ac_7 + bd_15;  ac_7 = bd_15 = None
    add_45 = add_44 + 0;  add_44 = None
    attn_score_7 = add_45 * 0.125;  add_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_14 = torch.nn.functional.softmax(attn_score_7, dim = 3);  attn_score_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_15 = self.L__mod___transformer_layer_7_rel_attn_dropout(attn_prob_14);  attn_prob_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_14 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_15, v_head_h_7);  attn_prob_15 = v_head_h_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_7_rel_attn_o_2 = self.L__mod___transformer_layer_7_rel_attn_o
    attn_out_21 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_14, l__mod___transformer_layer_7_rel_attn_o_2);  attn_vec_14 = l__mod___transformer_layer_7_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_22 = self.L__mod___transformer_layer_7_rel_attn_dropout(attn_out_21);  attn_out_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_23 = attn_out_22 + cat_8;  attn_out_22 = cat_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_57 = self.L__mod___transformer_layer_7_rel_attn_layer_norm(attn_out_23);  attn_out_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_58 = self.L__mod___transformer_layer_7_ff_layer_1(output_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_59 = torch._C._nn.gelu(output_58);  output_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_60 = self.L__mod___transformer_layer_7_ff_dropout(output_59);  output_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_61 = self.L__mod___transformer_layer_7_ff_layer_2(output_60);  output_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_62 = self.L__mod___transformer_layer_7_ff_dropout(output_61);  output_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_47 = output_62 + output_57;  output_62 = output_57 = None
    cat_9 = self.L__mod___transformer_layer_7_ff_layer_norm(add_47);  add_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_8 = cat_9[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_8 = new_mem_8.detach();  new_mem_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_8_rel_attn_q_2 = self.L__mod___transformer_layer_8_rel_attn_q
    q_head_h_8 = torch.functional.einsum('ibh,hnd->ibnd', cat_9, l__mod___transformer_layer_8_rel_attn_q_2);  l__mod___transformer_layer_8_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_8_rel_attn_k_2 = self.L__mod___transformer_layer_8_rel_attn_k
    k_head_h_8 = torch.functional.einsum('ibh,hnd->ibnd', cat_9, l__mod___transformer_layer_8_rel_attn_k_2);  l__mod___transformer_layer_8_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_8_rel_attn_v_2 = self.L__mod___transformer_layer_8_rel_attn_v
    v_head_h_8 = torch.functional.einsum('ibh,hnd->ibnd', cat_9, l__mod___transformer_layer_8_rel_attn_v_2);  l__mod___transformer_layer_8_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_8_rel_attn_r_2 = self.L__mod___transformer_layer_8_rel_attn_r
    type_9 = pos_emb_6.type(torch.float32)
    k_head_r_8 = torch.functional.einsum('ibh,hnd->ibnd', type_9, l__mod___transformer_layer_8_rel_attn_r_2);  type_9 = l__mod___transformer_layer_8_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_8_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_8_rel_attn_r_w_bias
    add_48 = q_head_h_8 + l__mod___transformer_layer_8_rel_attn_r_w_bias_2;  l__mod___transformer_layer_8_rel_attn_r_w_bias_2 = None
    ac_8 = torch.functional.einsum('ibnd,jbnd->bnij', add_48, k_head_h_8);  add_48 = k_head_h_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_8_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_8_rel_attn_r_r_bias
    add_49 = q_head_h_8 + l__mod___transformer_layer_8_rel_attn_r_r_bias_2;  q_head_h_8 = l__mod___transformer_layer_8_rel_attn_r_r_bias_2 = None
    bd_16 = torch.functional.einsum('ibnd,jbnd->bnij', add_49, k_head_r_8);  add_49 = k_head_r_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_32 = bd_16.reshape(1, 16, 1024, 512);  bd_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_33 = x_32[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_34 = x_33.reshape(1, 16, 512, 1023);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_10 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_17 = torch.index_select(x_34, 3, arange_10);  x_34 = arange_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_50 = ac_8 + bd_17;  ac_8 = bd_17 = None
    add_51 = add_50 + 0;  add_50 = None
    attn_score_8 = add_51 * 0.125;  add_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_16 = torch.nn.functional.softmax(attn_score_8, dim = 3);  attn_score_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_17 = self.L__mod___transformer_layer_8_rel_attn_dropout(attn_prob_16);  attn_prob_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_16 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_17, v_head_h_8);  attn_prob_17 = v_head_h_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_8_rel_attn_o_2 = self.L__mod___transformer_layer_8_rel_attn_o
    attn_out_24 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_16, l__mod___transformer_layer_8_rel_attn_o_2);  attn_vec_16 = l__mod___transformer_layer_8_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_25 = self.L__mod___transformer_layer_8_rel_attn_dropout(attn_out_24);  attn_out_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_26 = attn_out_25 + cat_9;  attn_out_25 = cat_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_65 = self.L__mod___transformer_layer_8_rel_attn_layer_norm(attn_out_26);  attn_out_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_66 = self.L__mod___transformer_layer_8_ff_layer_1(output_65)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_67 = torch._C._nn.gelu(output_66);  output_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_68 = self.L__mod___transformer_layer_8_ff_dropout(output_67);  output_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_69 = self.L__mod___transformer_layer_8_ff_layer_2(output_68);  output_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_70 = self.L__mod___transformer_layer_8_ff_dropout(output_69);  output_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_53 = output_70 + output_65;  output_70 = output_65 = None
    cat_10 = self.L__mod___transformer_layer_8_ff_layer_norm(add_53);  add_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_9 = cat_10[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_9 = new_mem_9.detach();  new_mem_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_9_rel_attn_q_2 = self.L__mod___transformer_layer_9_rel_attn_q
    q_head_h_9 = torch.functional.einsum('ibh,hnd->ibnd', cat_10, l__mod___transformer_layer_9_rel_attn_q_2);  l__mod___transformer_layer_9_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_9_rel_attn_k_2 = self.L__mod___transformer_layer_9_rel_attn_k
    k_head_h_9 = torch.functional.einsum('ibh,hnd->ibnd', cat_10, l__mod___transformer_layer_9_rel_attn_k_2);  l__mod___transformer_layer_9_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_9_rel_attn_v_2 = self.L__mod___transformer_layer_9_rel_attn_v
    v_head_h_9 = torch.functional.einsum('ibh,hnd->ibnd', cat_10, l__mod___transformer_layer_9_rel_attn_v_2);  l__mod___transformer_layer_9_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_9_rel_attn_r_2 = self.L__mod___transformer_layer_9_rel_attn_r
    type_10 = pos_emb_6.type(torch.float32)
    k_head_r_9 = torch.functional.einsum('ibh,hnd->ibnd', type_10, l__mod___transformer_layer_9_rel_attn_r_2);  type_10 = l__mod___transformer_layer_9_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_9_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_9_rel_attn_r_w_bias
    add_54 = q_head_h_9 + l__mod___transformer_layer_9_rel_attn_r_w_bias_2;  l__mod___transformer_layer_9_rel_attn_r_w_bias_2 = None
    ac_9 = torch.functional.einsum('ibnd,jbnd->bnij', add_54, k_head_h_9);  add_54 = k_head_h_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_9_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_9_rel_attn_r_r_bias
    add_55 = q_head_h_9 + l__mod___transformer_layer_9_rel_attn_r_r_bias_2;  q_head_h_9 = l__mod___transformer_layer_9_rel_attn_r_r_bias_2 = None
    bd_18 = torch.functional.einsum('ibnd,jbnd->bnij', add_55, k_head_r_9);  add_55 = k_head_r_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_36 = bd_18.reshape(1, 16, 1024, 512);  bd_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_37 = x_36[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_38 = x_37.reshape(1, 16, 512, 1023);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_11 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_19 = torch.index_select(x_38, 3, arange_11);  x_38 = arange_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_56 = ac_9 + bd_19;  ac_9 = bd_19 = None
    add_57 = add_56 + 0;  add_56 = None
    attn_score_9 = add_57 * 0.125;  add_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_18 = torch.nn.functional.softmax(attn_score_9, dim = 3);  attn_score_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_19 = self.L__mod___transformer_layer_9_rel_attn_dropout(attn_prob_18);  attn_prob_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_18 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_19, v_head_h_9);  attn_prob_19 = v_head_h_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_9_rel_attn_o_2 = self.L__mod___transformer_layer_9_rel_attn_o
    attn_out_27 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_18, l__mod___transformer_layer_9_rel_attn_o_2);  attn_vec_18 = l__mod___transformer_layer_9_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_28 = self.L__mod___transformer_layer_9_rel_attn_dropout(attn_out_27);  attn_out_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_29 = attn_out_28 + cat_10;  attn_out_28 = cat_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_73 = self.L__mod___transformer_layer_9_rel_attn_layer_norm(attn_out_29);  attn_out_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_74 = self.L__mod___transformer_layer_9_ff_layer_1(output_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_75 = torch._C._nn.gelu(output_74);  output_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_76 = self.L__mod___transformer_layer_9_ff_dropout(output_75);  output_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_77 = self.L__mod___transformer_layer_9_ff_layer_2(output_76);  output_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_78 = self.L__mod___transformer_layer_9_ff_dropout(output_77);  output_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_59 = output_78 + output_73;  output_78 = output_73 = None
    cat_11 = self.L__mod___transformer_layer_9_ff_layer_norm(add_59);  add_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_10 = cat_11[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_10 = new_mem_10.detach();  new_mem_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_10_rel_attn_q_2 = self.L__mod___transformer_layer_10_rel_attn_q
    q_head_h_10 = torch.functional.einsum('ibh,hnd->ibnd', cat_11, l__mod___transformer_layer_10_rel_attn_q_2);  l__mod___transformer_layer_10_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_10_rel_attn_k_2 = self.L__mod___transformer_layer_10_rel_attn_k
    k_head_h_10 = torch.functional.einsum('ibh,hnd->ibnd', cat_11, l__mod___transformer_layer_10_rel_attn_k_2);  l__mod___transformer_layer_10_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_10_rel_attn_v_2 = self.L__mod___transformer_layer_10_rel_attn_v
    v_head_h_10 = torch.functional.einsum('ibh,hnd->ibnd', cat_11, l__mod___transformer_layer_10_rel_attn_v_2);  l__mod___transformer_layer_10_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_10_rel_attn_r_2 = self.L__mod___transformer_layer_10_rel_attn_r
    type_11 = pos_emb_6.type(torch.float32)
    k_head_r_10 = torch.functional.einsum('ibh,hnd->ibnd', type_11, l__mod___transformer_layer_10_rel_attn_r_2);  type_11 = l__mod___transformer_layer_10_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_10_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_10_rel_attn_r_w_bias
    add_60 = q_head_h_10 + l__mod___transformer_layer_10_rel_attn_r_w_bias_2;  l__mod___transformer_layer_10_rel_attn_r_w_bias_2 = None
    ac_10 = torch.functional.einsum('ibnd,jbnd->bnij', add_60, k_head_h_10);  add_60 = k_head_h_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_10_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_10_rel_attn_r_r_bias
    add_61 = q_head_h_10 + l__mod___transformer_layer_10_rel_attn_r_r_bias_2;  q_head_h_10 = l__mod___transformer_layer_10_rel_attn_r_r_bias_2 = None
    bd_20 = torch.functional.einsum('ibnd,jbnd->bnij', add_61, k_head_r_10);  add_61 = k_head_r_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_40 = bd_20.reshape(1, 16, 1024, 512);  bd_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_41 = x_40[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_42 = x_41.reshape(1, 16, 512, 1023);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_12 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_21 = torch.index_select(x_42, 3, arange_12);  x_42 = arange_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_62 = ac_10 + bd_21;  ac_10 = bd_21 = None
    add_63 = add_62 + 0;  add_62 = None
    attn_score_10 = add_63 * 0.125;  add_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_20 = torch.nn.functional.softmax(attn_score_10, dim = 3);  attn_score_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_21 = self.L__mod___transformer_layer_10_rel_attn_dropout(attn_prob_20);  attn_prob_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_20 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_21, v_head_h_10);  attn_prob_21 = v_head_h_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_10_rel_attn_o_2 = self.L__mod___transformer_layer_10_rel_attn_o
    attn_out_30 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_20, l__mod___transformer_layer_10_rel_attn_o_2);  attn_vec_20 = l__mod___transformer_layer_10_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_31 = self.L__mod___transformer_layer_10_rel_attn_dropout(attn_out_30);  attn_out_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_32 = attn_out_31 + cat_11;  attn_out_31 = cat_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_81 = self.L__mod___transformer_layer_10_rel_attn_layer_norm(attn_out_32);  attn_out_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_82 = self.L__mod___transformer_layer_10_ff_layer_1(output_81)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_83 = torch._C._nn.gelu(output_82);  output_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_84 = self.L__mod___transformer_layer_10_ff_dropout(output_83);  output_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_85 = self.L__mod___transformer_layer_10_ff_layer_2(output_84);  output_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_86 = self.L__mod___transformer_layer_10_ff_dropout(output_85);  output_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_65 = output_86 + output_81;  output_86 = output_81 = None
    cat_12 = self.L__mod___transformer_layer_10_ff_layer_norm(add_65);  add_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_11 = cat_12[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_11 = new_mem_11.detach();  new_mem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_11_rel_attn_q_2 = self.L__mod___transformer_layer_11_rel_attn_q
    q_head_h_11 = torch.functional.einsum('ibh,hnd->ibnd', cat_12, l__mod___transformer_layer_11_rel_attn_q_2);  l__mod___transformer_layer_11_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_11_rel_attn_k_2 = self.L__mod___transformer_layer_11_rel_attn_k
    k_head_h_11 = torch.functional.einsum('ibh,hnd->ibnd', cat_12, l__mod___transformer_layer_11_rel_attn_k_2);  l__mod___transformer_layer_11_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_11_rel_attn_v_2 = self.L__mod___transformer_layer_11_rel_attn_v
    v_head_h_11 = torch.functional.einsum('ibh,hnd->ibnd', cat_12, l__mod___transformer_layer_11_rel_attn_v_2);  l__mod___transformer_layer_11_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_11_rel_attn_r_2 = self.L__mod___transformer_layer_11_rel_attn_r
    type_12 = pos_emb_6.type(torch.float32)
    k_head_r_11 = torch.functional.einsum('ibh,hnd->ibnd', type_12, l__mod___transformer_layer_11_rel_attn_r_2);  type_12 = l__mod___transformer_layer_11_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_11_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_11_rel_attn_r_w_bias
    add_66 = q_head_h_11 + l__mod___transformer_layer_11_rel_attn_r_w_bias_2;  l__mod___transformer_layer_11_rel_attn_r_w_bias_2 = None
    ac_11 = torch.functional.einsum('ibnd,jbnd->bnij', add_66, k_head_h_11);  add_66 = k_head_h_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_11_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_11_rel_attn_r_r_bias
    add_67 = q_head_h_11 + l__mod___transformer_layer_11_rel_attn_r_r_bias_2;  q_head_h_11 = l__mod___transformer_layer_11_rel_attn_r_r_bias_2 = None
    bd_22 = torch.functional.einsum('ibnd,jbnd->bnij', add_67, k_head_r_11);  add_67 = k_head_r_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_44 = bd_22.reshape(1, 16, 1024, 512);  bd_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_45 = x_44[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_46 = x_45.reshape(1, 16, 512, 1023);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_13 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_23 = torch.index_select(x_46, 3, arange_13);  x_46 = arange_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_68 = ac_11 + bd_23;  ac_11 = bd_23 = None
    add_69 = add_68 + 0;  add_68 = None
    attn_score_11 = add_69 * 0.125;  add_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_22 = torch.nn.functional.softmax(attn_score_11, dim = 3);  attn_score_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_23 = self.L__mod___transformer_layer_11_rel_attn_dropout(attn_prob_22);  attn_prob_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_22 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_23, v_head_h_11);  attn_prob_23 = v_head_h_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_11_rel_attn_o_2 = self.L__mod___transformer_layer_11_rel_attn_o
    attn_out_33 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_22, l__mod___transformer_layer_11_rel_attn_o_2);  attn_vec_22 = l__mod___transformer_layer_11_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_34 = self.L__mod___transformer_layer_11_rel_attn_dropout(attn_out_33);  attn_out_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_35 = attn_out_34 + cat_12;  attn_out_34 = cat_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_89 = self.L__mod___transformer_layer_11_rel_attn_layer_norm(attn_out_35);  attn_out_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_90 = self.L__mod___transformer_layer_11_ff_layer_1(output_89)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_91 = torch._C._nn.gelu(output_90);  output_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_92 = self.L__mod___transformer_layer_11_ff_dropout(output_91);  output_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_93 = self.L__mod___transformer_layer_11_ff_layer_2(output_92);  output_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_94 = self.L__mod___transformer_layer_11_ff_dropout(output_93);  output_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_71 = output_94 + output_89;  output_94 = output_89 = None
    cat_13 = self.L__mod___transformer_layer_11_ff_layer_norm(add_71);  add_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_12 = cat_13[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_12 = new_mem_12.detach();  new_mem_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_12_rel_attn_q_2 = self.L__mod___transformer_layer_12_rel_attn_q
    q_head_h_12 = torch.functional.einsum('ibh,hnd->ibnd', cat_13, l__mod___transformer_layer_12_rel_attn_q_2);  l__mod___transformer_layer_12_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_12_rel_attn_k_2 = self.L__mod___transformer_layer_12_rel_attn_k
    k_head_h_12 = torch.functional.einsum('ibh,hnd->ibnd', cat_13, l__mod___transformer_layer_12_rel_attn_k_2);  l__mod___transformer_layer_12_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_12_rel_attn_v_2 = self.L__mod___transformer_layer_12_rel_attn_v
    v_head_h_12 = torch.functional.einsum('ibh,hnd->ibnd', cat_13, l__mod___transformer_layer_12_rel_attn_v_2);  l__mod___transformer_layer_12_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_12_rel_attn_r_2 = self.L__mod___transformer_layer_12_rel_attn_r
    type_13 = pos_emb_6.type(torch.float32)
    k_head_r_12 = torch.functional.einsum('ibh,hnd->ibnd', type_13, l__mod___transformer_layer_12_rel_attn_r_2);  type_13 = l__mod___transformer_layer_12_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_12_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_12_rel_attn_r_w_bias
    add_72 = q_head_h_12 + l__mod___transformer_layer_12_rel_attn_r_w_bias_2;  l__mod___transformer_layer_12_rel_attn_r_w_bias_2 = None
    ac_12 = torch.functional.einsum('ibnd,jbnd->bnij', add_72, k_head_h_12);  add_72 = k_head_h_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_12_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_12_rel_attn_r_r_bias
    add_73 = q_head_h_12 + l__mod___transformer_layer_12_rel_attn_r_r_bias_2;  q_head_h_12 = l__mod___transformer_layer_12_rel_attn_r_r_bias_2 = None
    bd_24 = torch.functional.einsum('ibnd,jbnd->bnij', add_73, k_head_r_12);  add_73 = k_head_r_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_48 = bd_24.reshape(1, 16, 1024, 512);  bd_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_49 = x_48[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_50 = x_49.reshape(1, 16, 512, 1023);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_14 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_25 = torch.index_select(x_50, 3, arange_14);  x_50 = arange_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_74 = ac_12 + bd_25;  ac_12 = bd_25 = None
    add_75 = add_74 + 0;  add_74 = None
    attn_score_12 = add_75 * 0.125;  add_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_24 = torch.nn.functional.softmax(attn_score_12, dim = 3);  attn_score_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_25 = self.L__mod___transformer_layer_12_rel_attn_dropout(attn_prob_24);  attn_prob_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_24 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_25, v_head_h_12);  attn_prob_25 = v_head_h_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_12_rel_attn_o_2 = self.L__mod___transformer_layer_12_rel_attn_o
    attn_out_36 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_24, l__mod___transformer_layer_12_rel_attn_o_2);  attn_vec_24 = l__mod___transformer_layer_12_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_37 = self.L__mod___transformer_layer_12_rel_attn_dropout(attn_out_36);  attn_out_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_38 = attn_out_37 + cat_13;  attn_out_37 = cat_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_97 = self.L__mod___transformer_layer_12_rel_attn_layer_norm(attn_out_38);  attn_out_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_98 = self.L__mod___transformer_layer_12_ff_layer_1(output_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_99 = torch._C._nn.gelu(output_98);  output_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_100 = self.L__mod___transformer_layer_12_ff_dropout(output_99);  output_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_101 = self.L__mod___transformer_layer_12_ff_layer_2(output_100);  output_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_102 = self.L__mod___transformer_layer_12_ff_dropout(output_101);  output_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_77 = output_102 + output_97;  output_102 = output_97 = None
    cat_14 = self.L__mod___transformer_layer_12_ff_layer_norm(add_77);  add_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_13 = cat_14[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_13 = new_mem_13.detach();  new_mem_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_13_rel_attn_q_2 = self.L__mod___transformer_layer_13_rel_attn_q
    q_head_h_13 = torch.functional.einsum('ibh,hnd->ibnd', cat_14, l__mod___transformer_layer_13_rel_attn_q_2);  l__mod___transformer_layer_13_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_13_rel_attn_k_2 = self.L__mod___transformer_layer_13_rel_attn_k
    k_head_h_13 = torch.functional.einsum('ibh,hnd->ibnd', cat_14, l__mod___transformer_layer_13_rel_attn_k_2);  l__mod___transformer_layer_13_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_13_rel_attn_v_2 = self.L__mod___transformer_layer_13_rel_attn_v
    v_head_h_13 = torch.functional.einsum('ibh,hnd->ibnd', cat_14, l__mod___transformer_layer_13_rel_attn_v_2);  l__mod___transformer_layer_13_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_13_rel_attn_r_2 = self.L__mod___transformer_layer_13_rel_attn_r
    type_14 = pos_emb_6.type(torch.float32)
    k_head_r_13 = torch.functional.einsum('ibh,hnd->ibnd', type_14, l__mod___transformer_layer_13_rel_attn_r_2);  type_14 = l__mod___transformer_layer_13_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_13_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_13_rel_attn_r_w_bias
    add_78 = q_head_h_13 + l__mod___transformer_layer_13_rel_attn_r_w_bias_2;  l__mod___transformer_layer_13_rel_attn_r_w_bias_2 = None
    ac_13 = torch.functional.einsum('ibnd,jbnd->bnij', add_78, k_head_h_13);  add_78 = k_head_h_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_13_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_13_rel_attn_r_r_bias
    add_79 = q_head_h_13 + l__mod___transformer_layer_13_rel_attn_r_r_bias_2;  q_head_h_13 = l__mod___transformer_layer_13_rel_attn_r_r_bias_2 = None
    bd_26 = torch.functional.einsum('ibnd,jbnd->bnij', add_79, k_head_r_13);  add_79 = k_head_r_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_52 = bd_26.reshape(1, 16, 1024, 512);  bd_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_53 = x_52[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_54 = x_53.reshape(1, 16, 512, 1023);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_15 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_27 = torch.index_select(x_54, 3, arange_15);  x_54 = arange_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_80 = ac_13 + bd_27;  ac_13 = bd_27 = None
    add_81 = add_80 + 0;  add_80 = None
    attn_score_13 = add_81 * 0.125;  add_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_26 = torch.nn.functional.softmax(attn_score_13, dim = 3);  attn_score_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_27 = self.L__mod___transformer_layer_13_rel_attn_dropout(attn_prob_26);  attn_prob_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_26 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_27, v_head_h_13);  attn_prob_27 = v_head_h_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_13_rel_attn_o_2 = self.L__mod___transformer_layer_13_rel_attn_o
    attn_out_39 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_26, l__mod___transformer_layer_13_rel_attn_o_2);  attn_vec_26 = l__mod___transformer_layer_13_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_40 = self.L__mod___transformer_layer_13_rel_attn_dropout(attn_out_39);  attn_out_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_41 = attn_out_40 + cat_14;  attn_out_40 = cat_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_105 = self.L__mod___transformer_layer_13_rel_attn_layer_norm(attn_out_41);  attn_out_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_106 = self.L__mod___transformer_layer_13_ff_layer_1(output_105)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_107 = torch._C._nn.gelu(output_106);  output_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_108 = self.L__mod___transformer_layer_13_ff_dropout(output_107);  output_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_109 = self.L__mod___transformer_layer_13_ff_layer_2(output_108);  output_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_110 = self.L__mod___transformer_layer_13_ff_dropout(output_109);  output_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_83 = output_110 + output_105;  output_110 = output_105 = None
    cat_15 = self.L__mod___transformer_layer_13_ff_layer_norm(add_83);  add_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_14 = cat_15[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_14 = new_mem_14.detach();  new_mem_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_14_rel_attn_q_2 = self.L__mod___transformer_layer_14_rel_attn_q
    q_head_h_14 = torch.functional.einsum('ibh,hnd->ibnd', cat_15, l__mod___transformer_layer_14_rel_attn_q_2);  l__mod___transformer_layer_14_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_14_rel_attn_k_2 = self.L__mod___transformer_layer_14_rel_attn_k
    k_head_h_14 = torch.functional.einsum('ibh,hnd->ibnd', cat_15, l__mod___transformer_layer_14_rel_attn_k_2);  l__mod___transformer_layer_14_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_14_rel_attn_v_2 = self.L__mod___transformer_layer_14_rel_attn_v
    v_head_h_14 = torch.functional.einsum('ibh,hnd->ibnd', cat_15, l__mod___transformer_layer_14_rel_attn_v_2);  l__mod___transformer_layer_14_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_14_rel_attn_r_2 = self.L__mod___transformer_layer_14_rel_attn_r
    type_15 = pos_emb_6.type(torch.float32)
    k_head_r_14 = torch.functional.einsum('ibh,hnd->ibnd', type_15, l__mod___transformer_layer_14_rel_attn_r_2);  type_15 = l__mod___transformer_layer_14_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_14_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_14_rel_attn_r_w_bias
    add_84 = q_head_h_14 + l__mod___transformer_layer_14_rel_attn_r_w_bias_2;  l__mod___transformer_layer_14_rel_attn_r_w_bias_2 = None
    ac_14 = torch.functional.einsum('ibnd,jbnd->bnij', add_84, k_head_h_14);  add_84 = k_head_h_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_14_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_14_rel_attn_r_r_bias
    add_85 = q_head_h_14 + l__mod___transformer_layer_14_rel_attn_r_r_bias_2;  q_head_h_14 = l__mod___transformer_layer_14_rel_attn_r_r_bias_2 = None
    bd_28 = torch.functional.einsum('ibnd,jbnd->bnij', add_85, k_head_r_14);  add_85 = k_head_r_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_56 = bd_28.reshape(1, 16, 1024, 512);  bd_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_57 = x_56[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_58 = x_57.reshape(1, 16, 512, 1023);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_16 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_29 = torch.index_select(x_58, 3, arange_16);  x_58 = arange_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_86 = ac_14 + bd_29;  ac_14 = bd_29 = None
    add_87 = add_86 + 0;  add_86 = None
    attn_score_14 = add_87 * 0.125;  add_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_28 = torch.nn.functional.softmax(attn_score_14, dim = 3);  attn_score_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_29 = self.L__mod___transformer_layer_14_rel_attn_dropout(attn_prob_28);  attn_prob_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_28 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_29, v_head_h_14);  attn_prob_29 = v_head_h_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_14_rel_attn_o_2 = self.L__mod___transformer_layer_14_rel_attn_o
    attn_out_42 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_28, l__mod___transformer_layer_14_rel_attn_o_2);  attn_vec_28 = l__mod___transformer_layer_14_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_43 = self.L__mod___transformer_layer_14_rel_attn_dropout(attn_out_42);  attn_out_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_44 = attn_out_43 + cat_15;  attn_out_43 = cat_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_113 = self.L__mod___transformer_layer_14_rel_attn_layer_norm(attn_out_44);  attn_out_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_114 = self.L__mod___transformer_layer_14_ff_layer_1(output_113)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_115 = torch._C._nn.gelu(output_114);  output_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_116 = self.L__mod___transformer_layer_14_ff_dropout(output_115);  output_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_117 = self.L__mod___transformer_layer_14_ff_layer_2(output_116);  output_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_118 = self.L__mod___transformer_layer_14_ff_dropout(output_117);  output_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_89 = output_118 + output_113;  output_118 = output_113 = None
    cat_16 = self.L__mod___transformer_layer_14_ff_layer_norm(add_89);  add_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_15 = cat_16[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_15 = new_mem_15.detach();  new_mem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_15_rel_attn_q_2 = self.L__mod___transformer_layer_15_rel_attn_q
    q_head_h_15 = torch.functional.einsum('ibh,hnd->ibnd', cat_16, l__mod___transformer_layer_15_rel_attn_q_2);  l__mod___transformer_layer_15_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_15_rel_attn_k_2 = self.L__mod___transformer_layer_15_rel_attn_k
    k_head_h_15 = torch.functional.einsum('ibh,hnd->ibnd', cat_16, l__mod___transformer_layer_15_rel_attn_k_2);  l__mod___transformer_layer_15_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_15_rel_attn_v_2 = self.L__mod___transformer_layer_15_rel_attn_v
    v_head_h_15 = torch.functional.einsum('ibh,hnd->ibnd', cat_16, l__mod___transformer_layer_15_rel_attn_v_2);  l__mod___transformer_layer_15_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_15_rel_attn_r_2 = self.L__mod___transformer_layer_15_rel_attn_r
    type_16 = pos_emb_6.type(torch.float32)
    k_head_r_15 = torch.functional.einsum('ibh,hnd->ibnd', type_16, l__mod___transformer_layer_15_rel_attn_r_2);  type_16 = l__mod___transformer_layer_15_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_15_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_15_rel_attn_r_w_bias
    add_90 = q_head_h_15 + l__mod___transformer_layer_15_rel_attn_r_w_bias_2;  l__mod___transformer_layer_15_rel_attn_r_w_bias_2 = None
    ac_15 = torch.functional.einsum('ibnd,jbnd->bnij', add_90, k_head_h_15);  add_90 = k_head_h_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_15_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_15_rel_attn_r_r_bias
    add_91 = q_head_h_15 + l__mod___transformer_layer_15_rel_attn_r_r_bias_2;  q_head_h_15 = l__mod___transformer_layer_15_rel_attn_r_r_bias_2 = None
    bd_30 = torch.functional.einsum('ibnd,jbnd->bnij', add_91, k_head_r_15);  add_91 = k_head_r_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_60 = bd_30.reshape(1, 16, 1024, 512);  bd_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_61 = x_60[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_62 = x_61.reshape(1, 16, 512, 1023);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_17 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_31 = torch.index_select(x_62, 3, arange_17);  x_62 = arange_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_92 = ac_15 + bd_31;  ac_15 = bd_31 = None
    add_93 = add_92 + 0;  add_92 = None
    attn_score_15 = add_93 * 0.125;  add_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_30 = torch.nn.functional.softmax(attn_score_15, dim = 3);  attn_score_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_31 = self.L__mod___transformer_layer_15_rel_attn_dropout(attn_prob_30);  attn_prob_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_30 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_31, v_head_h_15);  attn_prob_31 = v_head_h_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_15_rel_attn_o_2 = self.L__mod___transformer_layer_15_rel_attn_o
    attn_out_45 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_30, l__mod___transformer_layer_15_rel_attn_o_2);  attn_vec_30 = l__mod___transformer_layer_15_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_46 = self.L__mod___transformer_layer_15_rel_attn_dropout(attn_out_45);  attn_out_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_47 = attn_out_46 + cat_16;  attn_out_46 = cat_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_121 = self.L__mod___transformer_layer_15_rel_attn_layer_norm(attn_out_47);  attn_out_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_122 = self.L__mod___transformer_layer_15_ff_layer_1(output_121)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_123 = torch._C._nn.gelu(output_122);  output_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_124 = self.L__mod___transformer_layer_15_ff_dropout(output_123);  output_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_125 = self.L__mod___transformer_layer_15_ff_layer_2(output_124);  output_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_126 = self.L__mod___transformer_layer_15_ff_dropout(output_125);  output_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_95 = output_126 + output_121;  output_126 = output_121 = None
    cat_17 = self.L__mod___transformer_layer_15_ff_layer_norm(add_95);  add_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_16 = cat_17[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_16 = new_mem_16.detach();  new_mem_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_16_rel_attn_q_2 = self.L__mod___transformer_layer_16_rel_attn_q
    q_head_h_16 = torch.functional.einsum('ibh,hnd->ibnd', cat_17, l__mod___transformer_layer_16_rel_attn_q_2);  l__mod___transformer_layer_16_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_16_rel_attn_k_2 = self.L__mod___transformer_layer_16_rel_attn_k
    k_head_h_16 = torch.functional.einsum('ibh,hnd->ibnd', cat_17, l__mod___transformer_layer_16_rel_attn_k_2);  l__mod___transformer_layer_16_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_16_rel_attn_v_2 = self.L__mod___transformer_layer_16_rel_attn_v
    v_head_h_16 = torch.functional.einsum('ibh,hnd->ibnd', cat_17, l__mod___transformer_layer_16_rel_attn_v_2);  l__mod___transformer_layer_16_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_16_rel_attn_r_2 = self.L__mod___transformer_layer_16_rel_attn_r
    type_17 = pos_emb_6.type(torch.float32)
    k_head_r_16 = torch.functional.einsum('ibh,hnd->ibnd', type_17, l__mod___transformer_layer_16_rel_attn_r_2);  type_17 = l__mod___transformer_layer_16_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_16_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_16_rel_attn_r_w_bias
    add_96 = q_head_h_16 + l__mod___transformer_layer_16_rel_attn_r_w_bias_2;  l__mod___transformer_layer_16_rel_attn_r_w_bias_2 = None
    ac_16 = torch.functional.einsum('ibnd,jbnd->bnij', add_96, k_head_h_16);  add_96 = k_head_h_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_16_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_16_rel_attn_r_r_bias
    add_97 = q_head_h_16 + l__mod___transformer_layer_16_rel_attn_r_r_bias_2;  q_head_h_16 = l__mod___transformer_layer_16_rel_attn_r_r_bias_2 = None
    bd_32 = torch.functional.einsum('ibnd,jbnd->bnij', add_97, k_head_r_16);  add_97 = k_head_r_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_64 = bd_32.reshape(1, 16, 1024, 512);  bd_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_65 = x_64[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_66 = x_65.reshape(1, 16, 512, 1023);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_18 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_33 = torch.index_select(x_66, 3, arange_18);  x_66 = arange_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_98 = ac_16 + bd_33;  ac_16 = bd_33 = None
    add_99 = add_98 + 0;  add_98 = None
    attn_score_16 = add_99 * 0.125;  add_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_32 = torch.nn.functional.softmax(attn_score_16, dim = 3);  attn_score_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_33 = self.L__mod___transformer_layer_16_rel_attn_dropout(attn_prob_32);  attn_prob_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_32 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_33, v_head_h_16);  attn_prob_33 = v_head_h_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_16_rel_attn_o_2 = self.L__mod___transformer_layer_16_rel_attn_o
    attn_out_48 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_32, l__mod___transformer_layer_16_rel_attn_o_2);  attn_vec_32 = l__mod___transformer_layer_16_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_49 = self.L__mod___transformer_layer_16_rel_attn_dropout(attn_out_48);  attn_out_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_50 = attn_out_49 + cat_17;  attn_out_49 = cat_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_129 = self.L__mod___transformer_layer_16_rel_attn_layer_norm(attn_out_50);  attn_out_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_130 = self.L__mod___transformer_layer_16_ff_layer_1(output_129)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_131 = torch._C._nn.gelu(output_130);  output_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_132 = self.L__mod___transformer_layer_16_ff_dropout(output_131);  output_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_133 = self.L__mod___transformer_layer_16_ff_layer_2(output_132);  output_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_134 = self.L__mod___transformer_layer_16_ff_dropout(output_133);  output_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_101 = output_134 + output_129;  output_134 = output_129 = None
    cat_18 = self.L__mod___transformer_layer_16_ff_layer_norm(add_101);  add_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_17 = cat_18[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_17 = new_mem_17.detach();  new_mem_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_17_rel_attn_q_2 = self.L__mod___transformer_layer_17_rel_attn_q
    q_head_h_17 = torch.functional.einsum('ibh,hnd->ibnd', cat_18, l__mod___transformer_layer_17_rel_attn_q_2);  l__mod___transformer_layer_17_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_17_rel_attn_k_2 = self.L__mod___transformer_layer_17_rel_attn_k
    k_head_h_17 = torch.functional.einsum('ibh,hnd->ibnd', cat_18, l__mod___transformer_layer_17_rel_attn_k_2);  l__mod___transformer_layer_17_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_17_rel_attn_v_2 = self.L__mod___transformer_layer_17_rel_attn_v
    v_head_h_17 = torch.functional.einsum('ibh,hnd->ibnd', cat_18, l__mod___transformer_layer_17_rel_attn_v_2);  l__mod___transformer_layer_17_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_17_rel_attn_r_2 = self.L__mod___transformer_layer_17_rel_attn_r
    type_18 = pos_emb_6.type(torch.float32)
    k_head_r_17 = torch.functional.einsum('ibh,hnd->ibnd', type_18, l__mod___transformer_layer_17_rel_attn_r_2);  type_18 = l__mod___transformer_layer_17_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_17_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_17_rel_attn_r_w_bias
    add_102 = q_head_h_17 + l__mod___transformer_layer_17_rel_attn_r_w_bias_2;  l__mod___transformer_layer_17_rel_attn_r_w_bias_2 = None
    ac_17 = torch.functional.einsum('ibnd,jbnd->bnij', add_102, k_head_h_17);  add_102 = k_head_h_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_17_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_17_rel_attn_r_r_bias
    add_103 = q_head_h_17 + l__mod___transformer_layer_17_rel_attn_r_r_bias_2;  q_head_h_17 = l__mod___transformer_layer_17_rel_attn_r_r_bias_2 = None
    bd_34 = torch.functional.einsum('ibnd,jbnd->bnij', add_103, k_head_r_17);  add_103 = k_head_r_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_68 = bd_34.reshape(1, 16, 1024, 512);  bd_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_69 = x_68[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_70 = x_69.reshape(1, 16, 512, 1023);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_19 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_35 = torch.index_select(x_70, 3, arange_19);  x_70 = arange_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_104 = ac_17 + bd_35;  ac_17 = bd_35 = None
    add_105 = add_104 + 0;  add_104 = None
    attn_score_17 = add_105 * 0.125;  add_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_34 = torch.nn.functional.softmax(attn_score_17, dim = 3);  attn_score_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_35 = self.L__mod___transformer_layer_17_rel_attn_dropout(attn_prob_34);  attn_prob_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_34 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_35, v_head_h_17);  attn_prob_35 = v_head_h_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_17_rel_attn_o_2 = self.L__mod___transformer_layer_17_rel_attn_o
    attn_out_51 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_34, l__mod___transformer_layer_17_rel_attn_o_2);  attn_vec_34 = l__mod___transformer_layer_17_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_52 = self.L__mod___transformer_layer_17_rel_attn_dropout(attn_out_51);  attn_out_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_53 = attn_out_52 + cat_18;  attn_out_52 = cat_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_137 = self.L__mod___transformer_layer_17_rel_attn_layer_norm(attn_out_53);  attn_out_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_138 = self.L__mod___transformer_layer_17_ff_layer_1(output_137)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_139 = torch._C._nn.gelu(output_138);  output_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_140 = self.L__mod___transformer_layer_17_ff_dropout(output_139);  output_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_141 = self.L__mod___transformer_layer_17_ff_layer_2(output_140);  output_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_142 = self.L__mod___transformer_layer_17_ff_dropout(output_141);  output_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_107 = output_142 + output_137;  output_142 = output_137 = None
    cat_19 = self.L__mod___transformer_layer_17_ff_layer_norm(add_107);  add_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_18 = cat_19[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_18 = new_mem_18.detach();  new_mem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_18_rel_attn_q_2 = self.L__mod___transformer_layer_18_rel_attn_q
    q_head_h_18 = torch.functional.einsum('ibh,hnd->ibnd', cat_19, l__mod___transformer_layer_18_rel_attn_q_2);  l__mod___transformer_layer_18_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_18_rel_attn_k_2 = self.L__mod___transformer_layer_18_rel_attn_k
    k_head_h_18 = torch.functional.einsum('ibh,hnd->ibnd', cat_19, l__mod___transformer_layer_18_rel_attn_k_2);  l__mod___transformer_layer_18_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_18_rel_attn_v_2 = self.L__mod___transformer_layer_18_rel_attn_v
    v_head_h_18 = torch.functional.einsum('ibh,hnd->ibnd', cat_19, l__mod___transformer_layer_18_rel_attn_v_2);  l__mod___transformer_layer_18_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_18_rel_attn_r_2 = self.L__mod___transformer_layer_18_rel_attn_r
    type_19 = pos_emb_6.type(torch.float32)
    k_head_r_18 = torch.functional.einsum('ibh,hnd->ibnd', type_19, l__mod___transformer_layer_18_rel_attn_r_2);  type_19 = l__mod___transformer_layer_18_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_18_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_18_rel_attn_r_w_bias
    add_108 = q_head_h_18 + l__mod___transformer_layer_18_rel_attn_r_w_bias_2;  l__mod___transformer_layer_18_rel_attn_r_w_bias_2 = None
    ac_18 = torch.functional.einsum('ibnd,jbnd->bnij', add_108, k_head_h_18);  add_108 = k_head_h_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_18_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_18_rel_attn_r_r_bias
    add_109 = q_head_h_18 + l__mod___transformer_layer_18_rel_attn_r_r_bias_2;  q_head_h_18 = l__mod___transformer_layer_18_rel_attn_r_r_bias_2 = None
    bd_36 = torch.functional.einsum('ibnd,jbnd->bnij', add_109, k_head_r_18);  add_109 = k_head_r_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_72 = bd_36.reshape(1, 16, 1024, 512);  bd_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_73 = x_72[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_74 = x_73.reshape(1, 16, 512, 1023);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_20 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_37 = torch.index_select(x_74, 3, arange_20);  x_74 = arange_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_110 = ac_18 + bd_37;  ac_18 = bd_37 = None
    add_111 = add_110 + 0;  add_110 = None
    attn_score_18 = add_111 * 0.125;  add_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_36 = torch.nn.functional.softmax(attn_score_18, dim = 3);  attn_score_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_37 = self.L__mod___transformer_layer_18_rel_attn_dropout(attn_prob_36);  attn_prob_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_36 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_37, v_head_h_18);  attn_prob_37 = v_head_h_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_18_rel_attn_o_2 = self.L__mod___transformer_layer_18_rel_attn_o
    attn_out_54 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_36, l__mod___transformer_layer_18_rel_attn_o_2);  attn_vec_36 = l__mod___transformer_layer_18_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_55 = self.L__mod___transformer_layer_18_rel_attn_dropout(attn_out_54);  attn_out_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_56 = attn_out_55 + cat_19;  attn_out_55 = cat_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_145 = self.L__mod___transformer_layer_18_rel_attn_layer_norm(attn_out_56);  attn_out_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_146 = self.L__mod___transformer_layer_18_ff_layer_1(output_145)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_147 = torch._C._nn.gelu(output_146);  output_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_148 = self.L__mod___transformer_layer_18_ff_dropout(output_147);  output_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_149 = self.L__mod___transformer_layer_18_ff_layer_2(output_148);  output_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_150 = self.L__mod___transformer_layer_18_ff_dropout(output_149);  output_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_113 = output_150 + output_145;  output_150 = output_145 = None
    cat_20 = self.L__mod___transformer_layer_18_ff_layer_norm(add_113);  add_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_19 = cat_20[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_19 = new_mem_19.detach();  new_mem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_19_rel_attn_q_2 = self.L__mod___transformer_layer_19_rel_attn_q
    q_head_h_19 = torch.functional.einsum('ibh,hnd->ibnd', cat_20, l__mod___transformer_layer_19_rel_attn_q_2);  l__mod___transformer_layer_19_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_19_rel_attn_k_2 = self.L__mod___transformer_layer_19_rel_attn_k
    k_head_h_19 = torch.functional.einsum('ibh,hnd->ibnd', cat_20, l__mod___transformer_layer_19_rel_attn_k_2);  l__mod___transformer_layer_19_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_19_rel_attn_v_2 = self.L__mod___transformer_layer_19_rel_attn_v
    v_head_h_19 = torch.functional.einsum('ibh,hnd->ibnd', cat_20, l__mod___transformer_layer_19_rel_attn_v_2);  l__mod___transformer_layer_19_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_19_rel_attn_r_2 = self.L__mod___transformer_layer_19_rel_attn_r
    type_20 = pos_emb_6.type(torch.float32)
    k_head_r_19 = torch.functional.einsum('ibh,hnd->ibnd', type_20, l__mod___transformer_layer_19_rel_attn_r_2);  type_20 = l__mod___transformer_layer_19_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_19_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_19_rel_attn_r_w_bias
    add_114 = q_head_h_19 + l__mod___transformer_layer_19_rel_attn_r_w_bias_2;  l__mod___transformer_layer_19_rel_attn_r_w_bias_2 = None
    ac_19 = torch.functional.einsum('ibnd,jbnd->bnij', add_114, k_head_h_19);  add_114 = k_head_h_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_19_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_19_rel_attn_r_r_bias
    add_115 = q_head_h_19 + l__mod___transformer_layer_19_rel_attn_r_r_bias_2;  q_head_h_19 = l__mod___transformer_layer_19_rel_attn_r_r_bias_2 = None
    bd_38 = torch.functional.einsum('ibnd,jbnd->bnij', add_115, k_head_r_19);  add_115 = k_head_r_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_76 = bd_38.reshape(1, 16, 1024, 512);  bd_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_77 = x_76[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_78 = x_77.reshape(1, 16, 512, 1023);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_21 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_39 = torch.index_select(x_78, 3, arange_21);  x_78 = arange_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_116 = ac_19 + bd_39;  ac_19 = bd_39 = None
    add_117 = add_116 + 0;  add_116 = None
    attn_score_19 = add_117 * 0.125;  add_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_38 = torch.nn.functional.softmax(attn_score_19, dim = 3);  attn_score_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_39 = self.L__mod___transformer_layer_19_rel_attn_dropout(attn_prob_38);  attn_prob_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_38 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_39, v_head_h_19);  attn_prob_39 = v_head_h_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_19_rel_attn_o_2 = self.L__mod___transformer_layer_19_rel_attn_o
    attn_out_57 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_38, l__mod___transformer_layer_19_rel_attn_o_2);  attn_vec_38 = l__mod___transformer_layer_19_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_58 = self.L__mod___transformer_layer_19_rel_attn_dropout(attn_out_57);  attn_out_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_59 = attn_out_58 + cat_20;  attn_out_58 = cat_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_153 = self.L__mod___transformer_layer_19_rel_attn_layer_norm(attn_out_59);  attn_out_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_154 = self.L__mod___transformer_layer_19_ff_layer_1(output_153)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_155 = torch._C._nn.gelu(output_154);  output_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_156 = self.L__mod___transformer_layer_19_ff_dropout(output_155);  output_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_157 = self.L__mod___transformer_layer_19_ff_layer_2(output_156);  output_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_158 = self.L__mod___transformer_layer_19_ff_dropout(output_157);  output_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_119 = output_158 + output_153;  output_158 = output_153 = None
    cat_21 = self.L__mod___transformer_layer_19_ff_layer_norm(add_119);  add_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_20 = cat_21[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_20 = new_mem_20.detach();  new_mem_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_20_rel_attn_q_2 = self.L__mod___transformer_layer_20_rel_attn_q
    q_head_h_20 = torch.functional.einsum('ibh,hnd->ibnd', cat_21, l__mod___transformer_layer_20_rel_attn_q_2);  l__mod___transformer_layer_20_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_20_rel_attn_k_2 = self.L__mod___transformer_layer_20_rel_attn_k
    k_head_h_20 = torch.functional.einsum('ibh,hnd->ibnd', cat_21, l__mod___transformer_layer_20_rel_attn_k_2);  l__mod___transformer_layer_20_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_20_rel_attn_v_2 = self.L__mod___transformer_layer_20_rel_attn_v
    v_head_h_20 = torch.functional.einsum('ibh,hnd->ibnd', cat_21, l__mod___transformer_layer_20_rel_attn_v_2);  l__mod___transformer_layer_20_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_20_rel_attn_r_2 = self.L__mod___transformer_layer_20_rel_attn_r
    type_21 = pos_emb_6.type(torch.float32)
    k_head_r_20 = torch.functional.einsum('ibh,hnd->ibnd', type_21, l__mod___transformer_layer_20_rel_attn_r_2);  type_21 = l__mod___transformer_layer_20_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_20_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_20_rel_attn_r_w_bias
    add_120 = q_head_h_20 + l__mod___transformer_layer_20_rel_attn_r_w_bias_2;  l__mod___transformer_layer_20_rel_attn_r_w_bias_2 = None
    ac_20 = torch.functional.einsum('ibnd,jbnd->bnij', add_120, k_head_h_20);  add_120 = k_head_h_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_20_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_20_rel_attn_r_r_bias
    add_121 = q_head_h_20 + l__mod___transformer_layer_20_rel_attn_r_r_bias_2;  q_head_h_20 = l__mod___transformer_layer_20_rel_attn_r_r_bias_2 = None
    bd_40 = torch.functional.einsum('ibnd,jbnd->bnij', add_121, k_head_r_20);  add_121 = k_head_r_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_80 = bd_40.reshape(1, 16, 1024, 512);  bd_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_81 = x_80[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_82 = x_81.reshape(1, 16, 512, 1023);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_22 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_41 = torch.index_select(x_82, 3, arange_22);  x_82 = arange_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_122 = ac_20 + bd_41;  ac_20 = bd_41 = None
    add_123 = add_122 + 0;  add_122 = None
    attn_score_20 = add_123 * 0.125;  add_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_40 = torch.nn.functional.softmax(attn_score_20, dim = 3);  attn_score_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_41 = self.L__mod___transformer_layer_20_rel_attn_dropout(attn_prob_40);  attn_prob_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_40 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_41, v_head_h_20);  attn_prob_41 = v_head_h_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_20_rel_attn_o_2 = self.L__mod___transformer_layer_20_rel_attn_o
    attn_out_60 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_40, l__mod___transformer_layer_20_rel_attn_o_2);  attn_vec_40 = l__mod___transformer_layer_20_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_61 = self.L__mod___transformer_layer_20_rel_attn_dropout(attn_out_60);  attn_out_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_62 = attn_out_61 + cat_21;  attn_out_61 = cat_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_161 = self.L__mod___transformer_layer_20_rel_attn_layer_norm(attn_out_62);  attn_out_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_162 = self.L__mod___transformer_layer_20_ff_layer_1(output_161)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_163 = torch._C._nn.gelu(output_162);  output_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_164 = self.L__mod___transformer_layer_20_ff_dropout(output_163);  output_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_165 = self.L__mod___transformer_layer_20_ff_layer_2(output_164);  output_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_166 = self.L__mod___transformer_layer_20_ff_dropout(output_165);  output_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_125 = output_166 + output_161;  output_166 = output_161 = None
    cat_22 = self.L__mod___transformer_layer_20_ff_layer_norm(add_125);  add_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_21 = cat_22[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_21 = new_mem_21.detach();  new_mem_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_21_rel_attn_q_2 = self.L__mod___transformer_layer_21_rel_attn_q
    q_head_h_21 = torch.functional.einsum('ibh,hnd->ibnd', cat_22, l__mod___transformer_layer_21_rel_attn_q_2);  l__mod___transformer_layer_21_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_21_rel_attn_k_2 = self.L__mod___transformer_layer_21_rel_attn_k
    k_head_h_21 = torch.functional.einsum('ibh,hnd->ibnd', cat_22, l__mod___transformer_layer_21_rel_attn_k_2);  l__mod___transformer_layer_21_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_21_rel_attn_v_2 = self.L__mod___transformer_layer_21_rel_attn_v
    v_head_h_21 = torch.functional.einsum('ibh,hnd->ibnd', cat_22, l__mod___transformer_layer_21_rel_attn_v_2);  l__mod___transformer_layer_21_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_21_rel_attn_r_2 = self.L__mod___transformer_layer_21_rel_attn_r
    type_22 = pos_emb_6.type(torch.float32)
    k_head_r_21 = torch.functional.einsum('ibh,hnd->ibnd', type_22, l__mod___transformer_layer_21_rel_attn_r_2);  type_22 = l__mod___transformer_layer_21_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_21_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_21_rel_attn_r_w_bias
    add_126 = q_head_h_21 + l__mod___transformer_layer_21_rel_attn_r_w_bias_2;  l__mod___transformer_layer_21_rel_attn_r_w_bias_2 = None
    ac_21 = torch.functional.einsum('ibnd,jbnd->bnij', add_126, k_head_h_21);  add_126 = k_head_h_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_21_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_21_rel_attn_r_r_bias
    add_127 = q_head_h_21 + l__mod___transformer_layer_21_rel_attn_r_r_bias_2;  q_head_h_21 = l__mod___transformer_layer_21_rel_attn_r_r_bias_2 = None
    bd_42 = torch.functional.einsum('ibnd,jbnd->bnij', add_127, k_head_r_21);  add_127 = k_head_r_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_84 = bd_42.reshape(1, 16, 1024, 512);  bd_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_85 = x_84[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_86 = x_85.reshape(1, 16, 512, 1023);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_23 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_43 = torch.index_select(x_86, 3, arange_23);  x_86 = arange_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_128 = ac_21 + bd_43;  ac_21 = bd_43 = None
    add_129 = add_128 + 0;  add_128 = None
    attn_score_21 = add_129 * 0.125;  add_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_42 = torch.nn.functional.softmax(attn_score_21, dim = 3);  attn_score_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_43 = self.L__mod___transformer_layer_21_rel_attn_dropout(attn_prob_42);  attn_prob_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_42 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_43, v_head_h_21);  attn_prob_43 = v_head_h_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_21_rel_attn_o_2 = self.L__mod___transformer_layer_21_rel_attn_o
    attn_out_63 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_42, l__mod___transformer_layer_21_rel_attn_o_2);  attn_vec_42 = l__mod___transformer_layer_21_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_64 = self.L__mod___transformer_layer_21_rel_attn_dropout(attn_out_63);  attn_out_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_65 = attn_out_64 + cat_22;  attn_out_64 = cat_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_169 = self.L__mod___transformer_layer_21_rel_attn_layer_norm(attn_out_65);  attn_out_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_170 = self.L__mod___transformer_layer_21_ff_layer_1(output_169)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_171 = torch._C._nn.gelu(output_170);  output_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_172 = self.L__mod___transformer_layer_21_ff_dropout(output_171);  output_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_173 = self.L__mod___transformer_layer_21_ff_layer_2(output_172);  output_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_174 = self.L__mod___transformer_layer_21_ff_dropout(output_173);  output_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_131 = output_174 + output_169;  output_174 = output_169 = None
    cat_23 = self.L__mod___transformer_layer_21_ff_layer_norm(add_131);  add_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_22 = cat_23[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_22 = new_mem_22.detach();  new_mem_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_22_rel_attn_q_2 = self.L__mod___transformer_layer_22_rel_attn_q
    q_head_h_22 = torch.functional.einsum('ibh,hnd->ibnd', cat_23, l__mod___transformer_layer_22_rel_attn_q_2);  l__mod___transformer_layer_22_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_22_rel_attn_k_2 = self.L__mod___transformer_layer_22_rel_attn_k
    k_head_h_22 = torch.functional.einsum('ibh,hnd->ibnd', cat_23, l__mod___transformer_layer_22_rel_attn_k_2);  l__mod___transformer_layer_22_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_22_rel_attn_v_2 = self.L__mod___transformer_layer_22_rel_attn_v
    v_head_h_22 = torch.functional.einsum('ibh,hnd->ibnd', cat_23, l__mod___transformer_layer_22_rel_attn_v_2);  l__mod___transformer_layer_22_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_22_rel_attn_r_2 = self.L__mod___transformer_layer_22_rel_attn_r
    type_23 = pos_emb_6.type(torch.float32)
    k_head_r_22 = torch.functional.einsum('ibh,hnd->ibnd', type_23, l__mod___transformer_layer_22_rel_attn_r_2);  type_23 = l__mod___transformer_layer_22_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_22_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_22_rel_attn_r_w_bias
    add_132 = q_head_h_22 + l__mod___transformer_layer_22_rel_attn_r_w_bias_2;  l__mod___transformer_layer_22_rel_attn_r_w_bias_2 = None
    ac_22 = torch.functional.einsum('ibnd,jbnd->bnij', add_132, k_head_h_22);  add_132 = k_head_h_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_22_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_22_rel_attn_r_r_bias
    add_133 = q_head_h_22 + l__mod___transformer_layer_22_rel_attn_r_r_bias_2;  q_head_h_22 = l__mod___transformer_layer_22_rel_attn_r_r_bias_2 = None
    bd_44 = torch.functional.einsum('ibnd,jbnd->bnij', add_133, k_head_r_22);  add_133 = k_head_r_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_88 = bd_44.reshape(1, 16, 1024, 512);  bd_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_89 = x_88[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_90 = x_89.reshape(1, 16, 512, 1023);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_24 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_45 = torch.index_select(x_90, 3, arange_24);  x_90 = arange_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_134 = ac_22 + bd_45;  ac_22 = bd_45 = None
    add_135 = add_134 + 0;  add_134 = None
    attn_score_22 = add_135 * 0.125;  add_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_44 = torch.nn.functional.softmax(attn_score_22, dim = 3);  attn_score_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_45 = self.L__mod___transformer_layer_22_rel_attn_dropout(attn_prob_44);  attn_prob_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_44 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_45, v_head_h_22);  attn_prob_45 = v_head_h_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_22_rel_attn_o_2 = self.L__mod___transformer_layer_22_rel_attn_o
    attn_out_66 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_44, l__mod___transformer_layer_22_rel_attn_o_2);  attn_vec_44 = l__mod___transformer_layer_22_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_67 = self.L__mod___transformer_layer_22_rel_attn_dropout(attn_out_66);  attn_out_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_68 = attn_out_67 + cat_23;  attn_out_67 = cat_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_177 = self.L__mod___transformer_layer_22_rel_attn_layer_norm(attn_out_68);  attn_out_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_178 = self.L__mod___transformer_layer_22_ff_layer_1(output_177)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_179 = torch._C._nn.gelu(output_178);  output_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_180 = self.L__mod___transformer_layer_22_ff_dropout(output_179);  output_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_181 = self.L__mod___transformer_layer_22_ff_layer_2(output_180);  output_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_182 = self.L__mod___transformer_layer_22_ff_dropout(output_181);  output_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_137 = output_182 + output_177;  output_182 = output_177 = None
    cat_24 = self.L__mod___transformer_layer_22_ff_layer_norm(add_137);  add_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1004, code: new_mem = curr_out[cutoff:]
    new_mem_23 = cat_24[slice(-512, None, None)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1008, code: return new_mem.detach()
    detach_23 = new_mem_23.detach();  new_mem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:430, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
    l__mod___transformer_layer_23_rel_attn_q_2 = self.L__mod___transformer_layer_23_rel_attn_q
    q_head_h_23 = torch.functional.einsum('ibh,hnd->ibnd', cat_24, l__mod___transformer_layer_23_rel_attn_q_2);  l__mod___transformer_layer_23_rel_attn_q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:431, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
    l__mod___transformer_layer_23_rel_attn_k_2 = self.L__mod___transformer_layer_23_rel_attn_k
    k_head_h_23 = torch.functional.einsum('ibh,hnd->ibnd', cat_24, l__mod___transformer_layer_23_rel_attn_k_2);  l__mod___transformer_layer_23_rel_attn_k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:432, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
    l__mod___transformer_layer_23_rel_attn_v_2 = self.L__mod___transformer_layer_23_rel_attn_v
    v_head_h_23 = torch.functional.einsum('ibh,hnd->ibnd', cat_24, l__mod___transformer_layer_23_rel_attn_v_2);  l__mod___transformer_layer_23_rel_attn_v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:436, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
    l__mod___transformer_layer_23_rel_attn_r_2 = self.L__mod___transformer_layer_23_rel_attn_r
    type_24 = pos_emb_6.type(torch.float32);  pos_emb_6 = None
    k_head_r_23 = torch.functional.einsum('ibh,hnd->ibnd', type_24, l__mod___transformer_layer_23_rel_attn_r_2);  type_24 = l__mod___transformer_layer_23_rel_attn_r_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:277, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
    l__mod___transformer_layer_23_rel_attn_r_w_bias_2 = self.L__mod___transformer_layer_23_rel_attn_r_w_bias
    add_138 = q_head_h_23 + l__mod___transformer_layer_23_rel_attn_r_w_bias_2;  l__mod___transformer_layer_23_rel_attn_r_w_bias_2 = None
    ac_23 = torch.functional.einsum('ibnd,jbnd->bnij', add_138, k_head_h_23);  add_138 = k_head_h_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:280, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
    l__mod___transformer_layer_23_rel_attn_r_r_bias_2 = self.L__mod___transformer_layer_23_rel_attn_r_r_bias
    add_139 = q_head_h_23 + l__mod___transformer_layer_23_rel_attn_r_r_bias_2;  q_head_h_23 = l__mod___transformer_layer_23_rel_attn_r_r_bias_2 = None
    bd_46 = torch.functional.einsum('ibnd,jbnd->bnij', add_139, k_head_r_23);  add_139 = k_head_r_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:252, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
    x_92 = bd_46.reshape(1, 16, 1024, 512);  bd_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:253, code: x = x[:, :, 1:, :]
    x_93 = x_92[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))];  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:254, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
    x_94 = x_93.reshape(1, 16, 512, 1023);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:258, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
    arange_25 = torch.arange(512, device = device(type='cuda', index=0), dtype = torch.int64)
    bd_47 = torch.index_select(x_94, 3, arange_25);  x_94 = arange_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:291, code: attn_score = (ac + bd + ef) * self.scale
    add_140 = ac_23 + bd_47;  ac_23 = bd_47 = None
    add_141 = add_140 + 0;  add_140 = None
    attn_score_23 = add_141 * 0.125;  add_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:300, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
    attn_prob_46 = torch.nn.functional.softmax(attn_score_23, dim = 3);  attn_score_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:301, code: attn_prob = self.dropout(attn_prob)
    attn_prob_47 = self.L__mod___transformer_layer_23_rel_attn_dropout(attn_prob_46);  attn_prob_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:308, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
    attn_vec_46 = torch.functional.einsum('bnij,jbnd->ibnd', attn_prob_47, v_head_h_23);  attn_prob_47 = v_head_h_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:318, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
    l__mod___transformer_layer_23_rel_attn_o_2 = self.L__mod___transformer_layer_23_rel_attn_o
    attn_out_69 = torch.functional.einsum('ibnd,hnd->ibh', attn_vec_46, l__mod___transformer_layer_23_rel_attn_o_2);  attn_vec_46 = l__mod___transformer_layer_23_rel_attn_o_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:320, code: attn_out = self.dropout(attn_out)
    attn_out_70 = self.L__mod___transformer_layer_23_rel_attn_dropout(attn_out_69);  attn_out_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:322, code: attn_out = attn_out + h
    attn_out_71 = attn_out_70 + cat_24;  attn_out_70 = cat_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:323, code: output = self.layer_norm(attn_out)
    output_185 = self.L__mod___transformer_layer_23_rel_attn_layer_norm(attn_out_71);  attn_out_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:477, code: output = self.layer_1(output)
    output_186 = self.L__mod___transformer_layer_23_ff_layer_1(output_185)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/activations.py:78, code: return self.act(input)
    output_187 = torch._C._nn.gelu(output_186);  output_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:479, code: output = self.dropout(output)
    output_188 = self.L__mod___transformer_layer_23_ff_dropout(output_187);  output_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:480, code: output = self.layer_2(output)
    output_189 = self.L__mod___transformer_layer_23_ff_layer_2(output_188);  output_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:481, code: output = self.dropout(output)
    output_190 = self.L__mod___transformer_layer_23_ff_dropout(output_189);  output_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:482, code: output = self.layer_norm(output + inp)
    add_143 = output_190 + output_185;  output_190 = output_185 = None
    output_h_96 = self.L__mod___transformer_layer_23_ff_layer_norm(add_143);  add_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1257, code: output = self.dropout(output_g if output_g is not None else output_h)
    output_192 = self.L__mod___transformer_dropout(output_h_96);  output_h_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1260, code: output = output.permute(1, 0, 2).contiguous()
    permute = output_192.permute(1, 0, 2);  output_192 = None
    output_193 = permute.contiguous();  permute = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1463, code: logits = self.lm_loss(transformer_outputs[0])
    logits = self.L__mod___lm_loss(output_193);  output_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/transformers/models/xlnet/modeling_xlnet.py:1469, code: loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
    view = logits.view(-1, 32000)
    view_1 = l_inputs_labels_.view(-1);  l_inputs_labels_ = None
    loss = torch.nn.functional.cross_entropy(view, view_1, None, None, -100, None, 'mean', 0.0);  view = view_1 = None
    return (loss, logits, detach, detach_1, detach_2, detach_3, detach_4, detach_5, detach_6, detach_7, detach_8, detach_9, detach_10, detach_11, detach_12, detach_13, detach_14, detach_15, detach_16, detach_17, detach_18, detach_19, detach_20, detach_21, detach_22, detach_23)
    