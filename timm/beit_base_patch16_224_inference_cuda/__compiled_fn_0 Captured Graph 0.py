from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x = self.L__mod___patch_embed_proj(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    flatten = x.flatten(2);  x = None
    x_1 = flatten.transpose(1, 2);  flatten = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x_3 = self.L__mod___patch_embed_norm(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:405, code: x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    l__mod___cls_token = self.L__mod___cls_token
    expand = l__mod___cls_token.expand(8, -1, -1);  l__mod___cls_token = None
    x_4 = torch.cat((expand, x_3), dim = 1);  expand = x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:408, code: x = self.pos_drop(x)
    x_5 = self.L__mod___pos_drop(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:237, code: if self.gamma_1 is None:
    l__mod___blocks_0_gamma_1 = self.L__mod___blocks_0_gamma_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_0_norm1_weight = self.L__mod___blocks_0_norm1_weight
    l__mod___blocks_0_norm1_bias = self.L__mod___blocks_0_norm1_bias
    x_6 = torch.nn.functional.layer_norm(x_5, (768,), l__mod___blocks_0_norm1_weight, l__mod___blocks_0_norm1_bias, 1e-06);  l__mod___blocks_0_norm1_weight = l__mod___blocks_0_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    l__mod___blocks_0_attn_q_bias = self.L__mod___blocks_0_attn_q_bias
    l__mod___blocks_0_attn_k_bias = self.L__mod___blocks_0_attn_k_bias
    l__mod___blocks_0_attn_v_bias = self.L__mod___blocks_0_attn_v_bias
    qkv_bias = torch.cat((l__mod___blocks_0_attn_q_bias, l__mod___blocks_0_attn_k_bias, l__mod___blocks_0_attn_v_bias));  l__mod___blocks_0_attn_q_bias = l__mod___blocks_0_attn_k_bias = l__mod___blocks_0_attn_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    l__mod___blocks_0_attn_qkv_weight = self.L__mod___blocks_0_attn_qkv_weight
    qkv = torch._C._nn.linear(input = x_6, weight = l__mod___blocks_0_attn_qkv_weight, bias = qkv_bias);  x_6 = l__mod___blocks_0_attn_qkv_weight = qkv_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    reshape = qkv.reshape(8, 197, 3, 12, -1);  qkv = None
    qkv_1 = reshape.permute(2, 0, 3, 1, 4);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind = qkv_1.unbind(0);  qkv_1 = None
    q = unbind[0]
    k = unbind[1]
    v = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:148, code: if self.relative_position_bias_table is not None:
    l__mod___blocks_0_attn_relative_position_bias_table = self.L__mod___blocks_0_attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    l__mod___blocks_0_attn_relative_position_index = self.L__mod___blocks_0_attn_relative_position_index
    view = l__mod___blocks_0_attn_relative_position_index.view(-1);  l__mod___blocks_0_attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    getitem_3 = l__mod___blocks_0_attn_relative_position_bias_table[view];  l__mod___blocks_0_attn_relative_position_bias_table = view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    relative_position_bias = getitem_3.view(197, 197, -1);  getitem_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_1 = relative_position_bias.permute(2, 0, 1);  relative_position_bias = None
    relative_position_bias_1 = permute_1.contiguous();  permute_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    rel_pos_bias = relative_position_bias_1.unsqueeze(0);  relative_position_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    x_7 = torch._C._nn.scaled_dot_product_attention(q, k, v, attn_mask = rel_pos_bias, dropout_p = 0.0);  q = k = v = rel_pos_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_1 = x_7.transpose(1, 2);  x_7 = None
    x_8 = transpose_1.reshape(8, 197, 768);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    x_9 = self.L__mod___blocks_0_attn_proj(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    x_10 = self.L__mod___blocks_0_attn_proj_drop(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul = l__mod___blocks_0_gamma_1 * x_10;  l__mod___blocks_0_gamma_1 = x_10 = None
    l__mod___blocks_0_drop_path1 = self.L__mod___blocks_0_drop_path1(mul);  mul = None
    x_11 = x_5 + l__mod___blocks_0_drop_path1;  x_5 = l__mod___blocks_0_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    l__mod___blocks_0_gamma_2 = self.L__mod___blocks_0_gamma_2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_0_norm2_weight = self.L__mod___blocks_0_norm2_weight
    l__mod___blocks_0_norm2_bias = self.L__mod___blocks_0_norm2_bias
    x_12 = torch.nn.functional.layer_norm(x_11, (768,), l__mod___blocks_0_norm2_weight, l__mod___blocks_0_norm2_bias, 1e-06);  l__mod___blocks_0_norm2_weight = l__mod___blocks_0_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_13 = self.L__mod___blocks_0_mlp_fc1(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_14 = self.L__mod___blocks_0_mlp_act(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_15 = self.L__mod___blocks_0_mlp_drop1(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_16 = self.L__mod___blocks_0_mlp_norm(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_17 = self.L__mod___blocks_0_mlp_fc2(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_18 = self.L__mod___blocks_0_mlp_drop2(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_1 = l__mod___blocks_0_gamma_2 * x_18;  l__mod___blocks_0_gamma_2 = x_18 = None
    l__mod___blocks_0_drop_path2 = self.L__mod___blocks_0_drop_path2(mul_1);  mul_1 = None
    x_20 = x_11 + l__mod___blocks_0_drop_path2;  x_11 = l__mod___blocks_0_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:237, code: if self.gamma_1 is None:
    l__mod___blocks_1_gamma_1 = self.L__mod___blocks_1_gamma_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_1_norm1_weight = self.L__mod___blocks_1_norm1_weight
    l__mod___blocks_1_norm1_bias = self.L__mod___blocks_1_norm1_bias
    x_21 = torch.nn.functional.layer_norm(x_20, (768,), l__mod___blocks_1_norm1_weight, l__mod___blocks_1_norm1_bias, 1e-06);  l__mod___blocks_1_norm1_weight = l__mod___blocks_1_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    l__mod___blocks_1_attn_q_bias = self.L__mod___blocks_1_attn_q_bias
    l__mod___blocks_1_attn_k_bias = self.L__mod___blocks_1_attn_k_bias
    l__mod___blocks_1_attn_v_bias = self.L__mod___blocks_1_attn_v_bias
    qkv_bias_1 = torch.cat((l__mod___blocks_1_attn_q_bias, l__mod___blocks_1_attn_k_bias, l__mod___blocks_1_attn_v_bias));  l__mod___blocks_1_attn_q_bias = l__mod___blocks_1_attn_k_bias = l__mod___blocks_1_attn_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    l__mod___blocks_1_attn_qkv_weight = self.L__mod___blocks_1_attn_qkv_weight
    qkv_2 = torch._C._nn.linear(input = x_21, weight = l__mod___blocks_1_attn_qkv_weight, bias = qkv_bias_1);  x_21 = l__mod___blocks_1_attn_qkv_weight = qkv_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    reshape_2 = qkv_2.reshape(8, 197, 3, 12, -1);  qkv_2 = None
    qkv_3 = reshape_2.permute(2, 0, 3, 1, 4);  reshape_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_1 = qkv_3.unbind(0);  qkv_3 = None
    q_1 = unbind_1[0]
    k_1 = unbind_1[1]
    v_1 = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:148, code: if self.relative_position_bias_table is not None:
    l__mod___blocks_1_attn_relative_position_bias_table = self.L__mod___blocks_1_attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    l__mod___blocks_1_attn_relative_position_index = self.L__mod___blocks_1_attn_relative_position_index
    view_2 = l__mod___blocks_1_attn_relative_position_index.view(-1);  l__mod___blocks_1_attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    getitem_7 = l__mod___blocks_1_attn_relative_position_bias_table[view_2];  l__mod___blocks_1_attn_relative_position_bias_table = view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    relative_position_bias_2 = getitem_7.view(197, 197, -1);  getitem_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_3 = relative_position_bias_2.permute(2, 0, 1);  relative_position_bias_2 = None
    relative_position_bias_3 = permute_3.contiguous();  permute_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    rel_pos_bias_1 = relative_position_bias_3.unsqueeze(0);  relative_position_bias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    x_22 = torch._C._nn.scaled_dot_product_attention(q_1, k_1, v_1, attn_mask = rel_pos_bias_1, dropout_p = 0.0);  q_1 = k_1 = v_1 = rel_pos_bias_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_2 = x_22.transpose(1, 2);  x_22 = None
    x_23 = transpose_2.reshape(8, 197, 768);  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    x_24 = self.L__mod___blocks_1_attn_proj(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    x_25 = self.L__mod___blocks_1_attn_proj_drop(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_2 = l__mod___blocks_1_gamma_1 * x_25;  l__mod___blocks_1_gamma_1 = x_25 = None
    l__mod___blocks_1_drop_path1 = self.L__mod___blocks_1_drop_path1(mul_2);  mul_2 = None
    x_26 = x_20 + l__mod___blocks_1_drop_path1;  x_20 = l__mod___blocks_1_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    l__mod___blocks_1_gamma_2 = self.L__mod___blocks_1_gamma_2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_1_norm2_weight = self.L__mod___blocks_1_norm2_weight
    l__mod___blocks_1_norm2_bias = self.L__mod___blocks_1_norm2_bias
    x_27 = torch.nn.functional.layer_norm(x_26, (768,), l__mod___blocks_1_norm2_weight, l__mod___blocks_1_norm2_bias, 1e-06);  l__mod___blocks_1_norm2_weight = l__mod___blocks_1_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_28 = self.L__mod___blocks_1_mlp_fc1(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_29 = self.L__mod___blocks_1_mlp_act(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_30 = self.L__mod___blocks_1_mlp_drop1(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_31 = self.L__mod___blocks_1_mlp_norm(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_32 = self.L__mod___blocks_1_mlp_fc2(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_33 = self.L__mod___blocks_1_mlp_drop2(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_3 = l__mod___blocks_1_gamma_2 * x_33;  l__mod___blocks_1_gamma_2 = x_33 = None
    l__mod___blocks_1_drop_path2 = self.L__mod___blocks_1_drop_path2(mul_3);  mul_3 = None
    x_35 = x_26 + l__mod___blocks_1_drop_path2;  x_26 = l__mod___blocks_1_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:237, code: if self.gamma_1 is None:
    l__mod___blocks_2_gamma_1 = self.L__mod___blocks_2_gamma_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_2_norm1_weight = self.L__mod___blocks_2_norm1_weight
    l__mod___blocks_2_norm1_bias = self.L__mod___blocks_2_norm1_bias
    x_36 = torch.nn.functional.layer_norm(x_35, (768,), l__mod___blocks_2_norm1_weight, l__mod___blocks_2_norm1_bias, 1e-06);  l__mod___blocks_2_norm1_weight = l__mod___blocks_2_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    l__mod___blocks_2_attn_q_bias = self.L__mod___blocks_2_attn_q_bias
    l__mod___blocks_2_attn_k_bias = self.L__mod___blocks_2_attn_k_bias
    l__mod___blocks_2_attn_v_bias = self.L__mod___blocks_2_attn_v_bias
    qkv_bias_2 = torch.cat((l__mod___blocks_2_attn_q_bias, l__mod___blocks_2_attn_k_bias, l__mod___blocks_2_attn_v_bias));  l__mod___blocks_2_attn_q_bias = l__mod___blocks_2_attn_k_bias = l__mod___blocks_2_attn_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    l__mod___blocks_2_attn_qkv_weight = self.L__mod___blocks_2_attn_qkv_weight
    qkv_4 = torch._C._nn.linear(input = x_36, weight = l__mod___blocks_2_attn_qkv_weight, bias = qkv_bias_2);  x_36 = l__mod___blocks_2_attn_qkv_weight = qkv_bias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    reshape_4 = qkv_4.reshape(8, 197, 3, 12, -1);  qkv_4 = None
    qkv_5 = reshape_4.permute(2, 0, 3, 1, 4);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_2 = qkv_5.unbind(0);  qkv_5 = None
    q_2 = unbind_2[0]
    k_2 = unbind_2[1]
    v_2 = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:148, code: if self.relative_position_bias_table is not None:
    l__mod___blocks_2_attn_relative_position_bias_table = self.L__mod___blocks_2_attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    l__mod___blocks_2_attn_relative_position_index = self.L__mod___blocks_2_attn_relative_position_index
    view_4 = l__mod___blocks_2_attn_relative_position_index.view(-1);  l__mod___blocks_2_attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    getitem_11 = l__mod___blocks_2_attn_relative_position_bias_table[view_4];  l__mod___blocks_2_attn_relative_position_bias_table = view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    relative_position_bias_4 = getitem_11.view(197, 197, -1);  getitem_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_5 = relative_position_bias_4.permute(2, 0, 1);  relative_position_bias_4 = None
    relative_position_bias_5 = permute_5.contiguous();  permute_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    rel_pos_bias_2 = relative_position_bias_5.unsqueeze(0);  relative_position_bias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    x_37 = torch._C._nn.scaled_dot_product_attention(q_2, k_2, v_2, attn_mask = rel_pos_bias_2, dropout_p = 0.0);  q_2 = k_2 = v_2 = rel_pos_bias_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_3 = x_37.transpose(1, 2);  x_37 = None
    x_38 = transpose_3.reshape(8, 197, 768);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    x_39 = self.L__mod___blocks_2_attn_proj(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    x_40 = self.L__mod___blocks_2_attn_proj_drop(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_4 = l__mod___blocks_2_gamma_1 * x_40;  l__mod___blocks_2_gamma_1 = x_40 = None
    l__mod___blocks_2_drop_path1 = self.L__mod___blocks_2_drop_path1(mul_4);  mul_4 = None
    x_41 = x_35 + l__mod___blocks_2_drop_path1;  x_35 = l__mod___blocks_2_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    l__mod___blocks_2_gamma_2 = self.L__mod___blocks_2_gamma_2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_2_norm2_weight = self.L__mod___blocks_2_norm2_weight
    l__mod___blocks_2_norm2_bias = self.L__mod___blocks_2_norm2_bias
    x_42 = torch.nn.functional.layer_norm(x_41, (768,), l__mod___blocks_2_norm2_weight, l__mod___blocks_2_norm2_bias, 1e-06);  l__mod___blocks_2_norm2_weight = l__mod___blocks_2_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_43 = self.L__mod___blocks_2_mlp_fc1(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_44 = self.L__mod___blocks_2_mlp_act(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_45 = self.L__mod___blocks_2_mlp_drop1(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_46 = self.L__mod___blocks_2_mlp_norm(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_47 = self.L__mod___blocks_2_mlp_fc2(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_48 = self.L__mod___blocks_2_mlp_drop2(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_5 = l__mod___blocks_2_gamma_2 * x_48;  l__mod___blocks_2_gamma_2 = x_48 = None
    l__mod___blocks_2_drop_path2 = self.L__mod___blocks_2_drop_path2(mul_5);  mul_5 = None
    x_50 = x_41 + l__mod___blocks_2_drop_path2;  x_41 = l__mod___blocks_2_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:237, code: if self.gamma_1 is None:
    l__mod___blocks_3_gamma_1 = self.L__mod___blocks_3_gamma_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_3_norm1_weight = self.L__mod___blocks_3_norm1_weight
    l__mod___blocks_3_norm1_bias = self.L__mod___blocks_3_norm1_bias
    x_51 = torch.nn.functional.layer_norm(x_50, (768,), l__mod___blocks_3_norm1_weight, l__mod___blocks_3_norm1_bias, 1e-06);  l__mod___blocks_3_norm1_weight = l__mod___blocks_3_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    l__mod___blocks_3_attn_q_bias = self.L__mod___blocks_3_attn_q_bias
    l__mod___blocks_3_attn_k_bias = self.L__mod___blocks_3_attn_k_bias
    l__mod___blocks_3_attn_v_bias = self.L__mod___blocks_3_attn_v_bias
    qkv_bias_3 = torch.cat((l__mod___blocks_3_attn_q_bias, l__mod___blocks_3_attn_k_bias, l__mod___blocks_3_attn_v_bias));  l__mod___blocks_3_attn_q_bias = l__mod___blocks_3_attn_k_bias = l__mod___blocks_3_attn_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    l__mod___blocks_3_attn_qkv_weight = self.L__mod___blocks_3_attn_qkv_weight
    qkv_6 = torch._C._nn.linear(input = x_51, weight = l__mod___blocks_3_attn_qkv_weight, bias = qkv_bias_3);  x_51 = l__mod___blocks_3_attn_qkv_weight = qkv_bias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    reshape_6 = qkv_6.reshape(8, 197, 3, 12, -1);  qkv_6 = None
    qkv_7 = reshape_6.permute(2, 0, 3, 1, 4);  reshape_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_3 = qkv_7.unbind(0);  qkv_7 = None
    q_3 = unbind_3[0]
    k_3 = unbind_3[1]
    v_3 = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:148, code: if self.relative_position_bias_table is not None:
    l__mod___blocks_3_attn_relative_position_bias_table = self.L__mod___blocks_3_attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    l__mod___blocks_3_attn_relative_position_index = self.L__mod___blocks_3_attn_relative_position_index
    view_6 = l__mod___blocks_3_attn_relative_position_index.view(-1);  l__mod___blocks_3_attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    getitem_15 = l__mod___blocks_3_attn_relative_position_bias_table[view_6];  l__mod___blocks_3_attn_relative_position_bias_table = view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    relative_position_bias_6 = getitem_15.view(197, 197, -1);  getitem_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_7 = relative_position_bias_6.permute(2, 0, 1);  relative_position_bias_6 = None
    relative_position_bias_7 = permute_7.contiguous();  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    rel_pos_bias_3 = relative_position_bias_7.unsqueeze(0);  relative_position_bias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    x_52 = torch._C._nn.scaled_dot_product_attention(q_3, k_3, v_3, attn_mask = rel_pos_bias_3, dropout_p = 0.0);  q_3 = k_3 = v_3 = rel_pos_bias_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_4 = x_52.transpose(1, 2);  x_52 = None
    x_53 = transpose_4.reshape(8, 197, 768);  transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    x_54 = self.L__mod___blocks_3_attn_proj(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    x_55 = self.L__mod___blocks_3_attn_proj_drop(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_6 = l__mod___blocks_3_gamma_1 * x_55;  l__mod___blocks_3_gamma_1 = x_55 = None
    l__mod___blocks_3_drop_path1 = self.L__mod___blocks_3_drop_path1(mul_6);  mul_6 = None
    x_56 = x_50 + l__mod___blocks_3_drop_path1;  x_50 = l__mod___blocks_3_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    l__mod___blocks_3_gamma_2 = self.L__mod___blocks_3_gamma_2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_3_norm2_weight = self.L__mod___blocks_3_norm2_weight
    l__mod___blocks_3_norm2_bias = self.L__mod___blocks_3_norm2_bias
    x_57 = torch.nn.functional.layer_norm(x_56, (768,), l__mod___blocks_3_norm2_weight, l__mod___blocks_3_norm2_bias, 1e-06);  l__mod___blocks_3_norm2_weight = l__mod___blocks_3_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_58 = self.L__mod___blocks_3_mlp_fc1(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_59 = self.L__mod___blocks_3_mlp_act(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_60 = self.L__mod___blocks_3_mlp_drop1(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_61 = self.L__mod___blocks_3_mlp_norm(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_62 = self.L__mod___blocks_3_mlp_fc2(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_63 = self.L__mod___blocks_3_mlp_drop2(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_7 = l__mod___blocks_3_gamma_2 * x_63;  l__mod___blocks_3_gamma_2 = x_63 = None
    l__mod___blocks_3_drop_path2 = self.L__mod___blocks_3_drop_path2(mul_7);  mul_7 = None
    x_65 = x_56 + l__mod___blocks_3_drop_path2;  x_56 = l__mod___blocks_3_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:237, code: if self.gamma_1 is None:
    l__mod___blocks_4_gamma_1 = self.L__mod___blocks_4_gamma_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_4_norm1_weight = self.L__mod___blocks_4_norm1_weight
    l__mod___blocks_4_norm1_bias = self.L__mod___blocks_4_norm1_bias
    x_66 = torch.nn.functional.layer_norm(x_65, (768,), l__mod___blocks_4_norm1_weight, l__mod___blocks_4_norm1_bias, 1e-06);  l__mod___blocks_4_norm1_weight = l__mod___blocks_4_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    l__mod___blocks_4_attn_q_bias = self.L__mod___blocks_4_attn_q_bias
    l__mod___blocks_4_attn_k_bias = self.L__mod___blocks_4_attn_k_bias
    l__mod___blocks_4_attn_v_bias = self.L__mod___blocks_4_attn_v_bias
    qkv_bias_4 = torch.cat((l__mod___blocks_4_attn_q_bias, l__mod___blocks_4_attn_k_bias, l__mod___blocks_4_attn_v_bias));  l__mod___blocks_4_attn_q_bias = l__mod___blocks_4_attn_k_bias = l__mod___blocks_4_attn_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    l__mod___blocks_4_attn_qkv_weight = self.L__mod___blocks_4_attn_qkv_weight
    qkv_8 = torch._C._nn.linear(input = x_66, weight = l__mod___blocks_4_attn_qkv_weight, bias = qkv_bias_4);  x_66 = l__mod___blocks_4_attn_qkv_weight = qkv_bias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    reshape_8 = qkv_8.reshape(8, 197, 3, 12, -1);  qkv_8 = None
    qkv_9 = reshape_8.permute(2, 0, 3, 1, 4);  reshape_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_4 = qkv_9.unbind(0);  qkv_9 = None
    q_4 = unbind_4[0]
    k_4 = unbind_4[1]
    v_4 = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:148, code: if self.relative_position_bias_table is not None:
    l__mod___blocks_4_attn_relative_position_bias_table = self.L__mod___blocks_4_attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    l__mod___blocks_4_attn_relative_position_index = self.L__mod___blocks_4_attn_relative_position_index
    view_8 = l__mod___blocks_4_attn_relative_position_index.view(-1);  l__mod___blocks_4_attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    getitem_19 = l__mod___blocks_4_attn_relative_position_bias_table[view_8];  l__mod___blocks_4_attn_relative_position_bias_table = view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    relative_position_bias_8 = getitem_19.view(197, 197, -1);  getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_9 = relative_position_bias_8.permute(2, 0, 1);  relative_position_bias_8 = None
    relative_position_bias_9 = permute_9.contiguous();  permute_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    rel_pos_bias_4 = relative_position_bias_9.unsqueeze(0);  relative_position_bias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    x_67 = torch._C._nn.scaled_dot_product_attention(q_4, k_4, v_4, attn_mask = rel_pos_bias_4, dropout_p = 0.0);  q_4 = k_4 = v_4 = rel_pos_bias_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_5 = x_67.transpose(1, 2);  x_67 = None
    x_68 = transpose_5.reshape(8, 197, 768);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    x_69 = self.L__mod___blocks_4_attn_proj(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    x_70 = self.L__mod___blocks_4_attn_proj_drop(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_8 = l__mod___blocks_4_gamma_1 * x_70;  l__mod___blocks_4_gamma_1 = x_70 = None
    l__mod___blocks_4_drop_path1 = self.L__mod___blocks_4_drop_path1(mul_8);  mul_8 = None
    x_71 = x_65 + l__mod___blocks_4_drop_path1;  x_65 = l__mod___blocks_4_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    l__mod___blocks_4_gamma_2 = self.L__mod___blocks_4_gamma_2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_4_norm2_weight = self.L__mod___blocks_4_norm2_weight
    l__mod___blocks_4_norm2_bias = self.L__mod___blocks_4_norm2_bias
    x_72 = torch.nn.functional.layer_norm(x_71, (768,), l__mod___blocks_4_norm2_weight, l__mod___blocks_4_norm2_bias, 1e-06);  l__mod___blocks_4_norm2_weight = l__mod___blocks_4_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_73 = self.L__mod___blocks_4_mlp_fc1(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_74 = self.L__mod___blocks_4_mlp_act(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_75 = self.L__mod___blocks_4_mlp_drop1(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_76 = self.L__mod___blocks_4_mlp_norm(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_77 = self.L__mod___blocks_4_mlp_fc2(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_78 = self.L__mod___blocks_4_mlp_drop2(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_9 = l__mod___blocks_4_gamma_2 * x_78;  l__mod___blocks_4_gamma_2 = x_78 = None
    l__mod___blocks_4_drop_path2 = self.L__mod___blocks_4_drop_path2(mul_9);  mul_9 = None
    x_80 = x_71 + l__mod___blocks_4_drop_path2;  x_71 = l__mod___blocks_4_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:237, code: if self.gamma_1 is None:
    l__mod___blocks_5_gamma_1 = self.L__mod___blocks_5_gamma_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_5_norm1_weight = self.L__mod___blocks_5_norm1_weight
    l__mod___blocks_5_norm1_bias = self.L__mod___blocks_5_norm1_bias
    x_81 = torch.nn.functional.layer_norm(x_80, (768,), l__mod___blocks_5_norm1_weight, l__mod___blocks_5_norm1_bias, 1e-06);  l__mod___blocks_5_norm1_weight = l__mod___blocks_5_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    l__mod___blocks_5_attn_q_bias = self.L__mod___blocks_5_attn_q_bias
    l__mod___blocks_5_attn_k_bias = self.L__mod___blocks_5_attn_k_bias
    l__mod___blocks_5_attn_v_bias = self.L__mod___blocks_5_attn_v_bias
    qkv_bias_5 = torch.cat((l__mod___blocks_5_attn_q_bias, l__mod___blocks_5_attn_k_bias, l__mod___blocks_5_attn_v_bias));  l__mod___blocks_5_attn_q_bias = l__mod___blocks_5_attn_k_bias = l__mod___blocks_5_attn_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    l__mod___blocks_5_attn_qkv_weight = self.L__mod___blocks_5_attn_qkv_weight
    qkv_10 = torch._C._nn.linear(input = x_81, weight = l__mod___blocks_5_attn_qkv_weight, bias = qkv_bias_5);  x_81 = l__mod___blocks_5_attn_qkv_weight = qkv_bias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    reshape_10 = qkv_10.reshape(8, 197, 3, 12, -1);  qkv_10 = None
    qkv_11 = reshape_10.permute(2, 0, 3, 1, 4);  reshape_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_5 = qkv_11.unbind(0);  qkv_11 = None
    q_5 = unbind_5[0]
    k_5 = unbind_5[1]
    v_5 = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:148, code: if self.relative_position_bias_table is not None:
    l__mod___blocks_5_attn_relative_position_bias_table = self.L__mod___blocks_5_attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    l__mod___blocks_5_attn_relative_position_index = self.L__mod___blocks_5_attn_relative_position_index
    view_10 = l__mod___blocks_5_attn_relative_position_index.view(-1);  l__mod___blocks_5_attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    getitem_23 = l__mod___blocks_5_attn_relative_position_bias_table[view_10];  l__mod___blocks_5_attn_relative_position_bias_table = view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    relative_position_bias_10 = getitem_23.view(197, 197, -1);  getitem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_11 = relative_position_bias_10.permute(2, 0, 1);  relative_position_bias_10 = None
    relative_position_bias_11 = permute_11.contiguous();  permute_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    rel_pos_bias_5 = relative_position_bias_11.unsqueeze(0);  relative_position_bias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    x_82 = torch._C._nn.scaled_dot_product_attention(q_5, k_5, v_5, attn_mask = rel_pos_bias_5, dropout_p = 0.0);  q_5 = k_5 = v_5 = rel_pos_bias_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_6 = x_82.transpose(1, 2);  x_82 = None
    x_83 = transpose_6.reshape(8, 197, 768);  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    x_84 = self.L__mod___blocks_5_attn_proj(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    x_85 = self.L__mod___blocks_5_attn_proj_drop(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_10 = l__mod___blocks_5_gamma_1 * x_85;  l__mod___blocks_5_gamma_1 = x_85 = None
    l__mod___blocks_5_drop_path1 = self.L__mod___blocks_5_drop_path1(mul_10);  mul_10 = None
    x_86 = x_80 + l__mod___blocks_5_drop_path1;  x_80 = l__mod___blocks_5_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    l__mod___blocks_5_gamma_2 = self.L__mod___blocks_5_gamma_2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_5_norm2_weight = self.L__mod___blocks_5_norm2_weight
    l__mod___blocks_5_norm2_bias = self.L__mod___blocks_5_norm2_bias
    x_87 = torch.nn.functional.layer_norm(x_86, (768,), l__mod___blocks_5_norm2_weight, l__mod___blocks_5_norm2_bias, 1e-06);  l__mod___blocks_5_norm2_weight = l__mod___blocks_5_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_88 = self.L__mod___blocks_5_mlp_fc1(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_89 = self.L__mod___blocks_5_mlp_act(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_90 = self.L__mod___blocks_5_mlp_drop1(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_91 = self.L__mod___blocks_5_mlp_norm(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_92 = self.L__mod___blocks_5_mlp_fc2(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_93 = self.L__mod___blocks_5_mlp_drop2(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_11 = l__mod___blocks_5_gamma_2 * x_93;  l__mod___blocks_5_gamma_2 = x_93 = None
    l__mod___blocks_5_drop_path2 = self.L__mod___blocks_5_drop_path2(mul_11);  mul_11 = None
    x_95 = x_86 + l__mod___blocks_5_drop_path2;  x_86 = l__mod___blocks_5_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:237, code: if self.gamma_1 is None:
    l__mod___blocks_6_gamma_1 = self.L__mod___blocks_6_gamma_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_6_norm1_weight = self.L__mod___blocks_6_norm1_weight
    l__mod___blocks_6_norm1_bias = self.L__mod___blocks_6_norm1_bias
    x_96 = torch.nn.functional.layer_norm(x_95, (768,), l__mod___blocks_6_norm1_weight, l__mod___blocks_6_norm1_bias, 1e-06);  l__mod___blocks_6_norm1_weight = l__mod___blocks_6_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    l__mod___blocks_6_attn_q_bias = self.L__mod___blocks_6_attn_q_bias
    l__mod___blocks_6_attn_k_bias = self.L__mod___blocks_6_attn_k_bias
    l__mod___blocks_6_attn_v_bias = self.L__mod___blocks_6_attn_v_bias
    qkv_bias_6 = torch.cat((l__mod___blocks_6_attn_q_bias, l__mod___blocks_6_attn_k_bias, l__mod___blocks_6_attn_v_bias));  l__mod___blocks_6_attn_q_bias = l__mod___blocks_6_attn_k_bias = l__mod___blocks_6_attn_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    l__mod___blocks_6_attn_qkv_weight = self.L__mod___blocks_6_attn_qkv_weight
    qkv_12 = torch._C._nn.linear(input = x_96, weight = l__mod___blocks_6_attn_qkv_weight, bias = qkv_bias_6);  x_96 = l__mod___blocks_6_attn_qkv_weight = qkv_bias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    reshape_12 = qkv_12.reshape(8, 197, 3, 12, -1);  qkv_12 = None
    qkv_13 = reshape_12.permute(2, 0, 3, 1, 4);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_6 = qkv_13.unbind(0);  qkv_13 = None
    q_6 = unbind_6[0]
    k_6 = unbind_6[1]
    v_6 = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:148, code: if self.relative_position_bias_table is not None:
    l__mod___blocks_6_attn_relative_position_bias_table = self.L__mod___blocks_6_attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    l__mod___blocks_6_attn_relative_position_index = self.L__mod___blocks_6_attn_relative_position_index
    view_12 = l__mod___blocks_6_attn_relative_position_index.view(-1);  l__mod___blocks_6_attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    getitem_27 = l__mod___blocks_6_attn_relative_position_bias_table[view_12];  l__mod___blocks_6_attn_relative_position_bias_table = view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    relative_position_bias_12 = getitem_27.view(197, 197, -1);  getitem_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_13 = relative_position_bias_12.permute(2, 0, 1);  relative_position_bias_12 = None
    relative_position_bias_13 = permute_13.contiguous();  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    rel_pos_bias_6 = relative_position_bias_13.unsqueeze(0);  relative_position_bias_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    x_97 = torch._C._nn.scaled_dot_product_attention(q_6, k_6, v_6, attn_mask = rel_pos_bias_6, dropout_p = 0.0);  q_6 = k_6 = v_6 = rel_pos_bias_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_7 = x_97.transpose(1, 2);  x_97 = None
    x_98 = transpose_7.reshape(8, 197, 768);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    x_99 = self.L__mod___blocks_6_attn_proj(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    x_100 = self.L__mod___blocks_6_attn_proj_drop(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_12 = l__mod___blocks_6_gamma_1 * x_100;  l__mod___blocks_6_gamma_1 = x_100 = None
    l__mod___blocks_6_drop_path1 = self.L__mod___blocks_6_drop_path1(mul_12);  mul_12 = None
    x_101 = x_95 + l__mod___blocks_6_drop_path1;  x_95 = l__mod___blocks_6_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    l__mod___blocks_6_gamma_2 = self.L__mod___blocks_6_gamma_2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_6_norm2_weight = self.L__mod___blocks_6_norm2_weight
    l__mod___blocks_6_norm2_bias = self.L__mod___blocks_6_norm2_bias
    x_102 = torch.nn.functional.layer_norm(x_101, (768,), l__mod___blocks_6_norm2_weight, l__mod___blocks_6_norm2_bias, 1e-06);  l__mod___blocks_6_norm2_weight = l__mod___blocks_6_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_103 = self.L__mod___blocks_6_mlp_fc1(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_104 = self.L__mod___blocks_6_mlp_act(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_105 = self.L__mod___blocks_6_mlp_drop1(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_106 = self.L__mod___blocks_6_mlp_norm(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_107 = self.L__mod___blocks_6_mlp_fc2(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_108 = self.L__mod___blocks_6_mlp_drop2(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_13 = l__mod___blocks_6_gamma_2 * x_108;  l__mod___blocks_6_gamma_2 = x_108 = None
    l__mod___blocks_6_drop_path2 = self.L__mod___blocks_6_drop_path2(mul_13);  mul_13 = None
    x_110 = x_101 + l__mod___blocks_6_drop_path2;  x_101 = l__mod___blocks_6_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:237, code: if self.gamma_1 is None:
    l__mod___blocks_7_gamma_1 = self.L__mod___blocks_7_gamma_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_7_norm1_weight = self.L__mod___blocks_7_norm1_weight
    l__mod___blocks_7_norm1_bias = self.L__mod___blocks_7_norm1_bias
    x_111 = torch.nn.functional.layer_norm(x_110, (768,), l__mod___blocks_7_norm1_weight, l__mod___blocks_7_norm1_bias, 1e-06);  l__mod___blocks_7_norm1_weight = l__mod___blocks_7_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    l__mod___blocks_7_attn_q_bias = self.L__mod___blocks_7_attn_q_bias
    l__mod___blocks_7_attn_k_bias = self.L__mod___blocks_7_attn_k_bias
    l__mod___blocks_7_attn_v_bias = self.L__mod___blocks_7_attn_v_bias
    qkv_bias_7 = torch.cat((l__mod___blocks_7_attn_q_bias, l__mod___blocks_7_attn_k_bias, l__mod___blocks_7_attn_v_bias));  l__mod___blocks_7_attn_q_bias = l__mod___blocks_7_attn_k_bias = l__mod___blocks_7_attn_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    l__mod___blocks_7_attn_qkv_weight = self.L__mod___blocks_7_attn_qkv_weight
    qkv_14 = torch._C._nn.linear(input = x_111, weight = l__mod___blocks_7_attn_qkv_weight, bias = qkv_bias_7);  x_111 = l__mod___blocks_7_attn_qkv_weight = qkv_bias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    reshape_14 = qkv_14.reshape(8, 197, 3, 12, -1);  qkv_14 = None
    qkv_15 = reshape_14.permute(2, 0, 3, 1, 4);  reshape_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_7 = qkv_15.unbind(0);  qkv_15 = None
    q_7 = unbind_7[0]
    k_7 = unbind_7[1]
    v_7 = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:148, code: if self.relative_position_bias_table is not None:
    l__mod___blocks_7_attn_relative_position_bias_table = self.L__mod___blocks_7_attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    l__mod___blocks_7_attn_relative_position_index = self.L__mod___blocks_7_attn_relative_position_index
    view_14 = l__mod___blocks_7_attn_relative_position_index.view(-1);  l__mod___blocks_7_attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    getitem_31 = l__mod___blocks_7_attn_relative_position_bias_table[view_14];  l__mod___blocks_7_attn_relative_position_bias_table = view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    relative_position_bias_14 = getitem_31.view(197, 197, -1);  getitem_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_15 = relative_position_bias_14.permute(2, 0, 1);  relative_position_bias_14 = None
    relative_position_bias_15 = permute_15.contiguous();  permute_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    rel_pos_bias_7 = relative_position_bias_15.unsqueeze(0);  relative_position_bias_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    x_112 = torch._C._nn.scaled_dot_product_attention(q_7, k_7, v_7, attn_mask = rel_pos_bias_7, dropout_p = 0.0);  q_7 = k_7 = v_7 = rel_pos_bias_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_8 = x_112.transpose(1, 2);  x_112 = None
    x_113 = transpose_8.reshape(8, 197, 768);  transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    x_114 = self.L__mod___blocks_7_attn_proj(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    x_115 = self.L__mod___blocks_7_attn_proj_drop(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_14 = l__mod___blocks_7_gamma_1 * x_115;  l__mod___blocks_7_gamma_1 = x_115 = None
    l__mod___blocks_7_drop_path1 = self.L__mod___blocks_7_drop_path1(mul_14);  mul_14 = None
    x_116 = x_110 + l__mod___blocks_7_drop_path1;  x_110 = l__mod___blocks_7_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    l__mod___blocks_7_gamma_2 = self.L__mod___blocks_7_gamma_2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_7_norm2_weight = self.L__mod___blocks_7_norm2_weight
    l__mod___blocks_7_norm2_bias = self.L__mod___blocks_7_norm2_bias
    x_117 = torch.nn.functional.layer_norm(x_116, (768,), l__mod___blocks_7_norm2_weight, l__mod___blocks_7_norm2_bias, 1e-06);  l__mod___blocks_7_norm2_weight = l__mod___blocks_7_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_118 = self.L__mod___blocks_7_mlp_fc1(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_119 = self.L__mod___blocks_7_mlp_act(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_120 = self.L__mod___blocks_7_mlp_drop1(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_121 = self.L__mod___blocks_7_mlp_norm(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_122 = self.L__mod___blocks_7_mlp_fc2(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_123 = self.L__mod___blocks_7_mlp_drop2(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_15 = l__mod___blocks_7_gamma_2 * x_123;  l__mod___blocks_7_gamma_2 = x_123 = None
    l__mod___blocks_7_drop_path2 = self.L__mod___blocks_7_drop_path2(mul_15);  mul_15 = None
    x_125 = x_116 + l__mod___blocks_7_drop_path2;  x_116 = l__mod___blocks_7_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:237, code: if self.gamma_1 is None:
    l__mod___blocks_8_gamma_1 = self.L__mod___blocks_8_gamma_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_8_norm1_weight = self.L__mod___blocks_8_norm1_weight
    l__mod___blocks_8_norm1_bias = self.L__mod___blocks_8_norm1_bias
    x_126 = torch.nn.functional.layer_norm(x_125, (768,), l__mod___blocks_8_norm1_weight, l__mod___blocks_8_norm1_bias, 1e-06);  l__mod___blocks_8_norm1_weight = l__mod___blocks_8_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    l__mod___blocks_8_attn_q_bias = self.L__mod___blocks_8_attn_q_bias
    l__mod___blocks_8_attn_k_bias = self.L__mod___blocks_8_attn_k_bias
    l__mod___blocks_8_attn_v_bias = self.L__mod___blocks_8_attn_v_bias
    qkv_bias_8 = torch.cat((l__mod___blocks_8_attn_q_bias, l__mod___blocks_8_attn_k_bias, l__mod___blocks_8_attn_v_bias));  l__mod___blocks_8_attn_q_bias = l__mod___blocks_8_attn_k_bias = l__mod___blocks_8_attn_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    l__mod___blocks_8_attn_qkv_weight = self.L__mod___blocks_8_attn_qkv_weight
    qkv_16 = torch._C._nn.linear(input = x_126, weight = l__mod___blocks_8_attn_qkv_weight, bias = qkv_bias_8);  x_126 = l__mod___blocks_8_attn_qkv_weight = qkv_bias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    reshape_16 = qkv_16.reshape(8, 197, 3, 12, -1);  qkv_16 = None
    qkv_17 = reshape_16.permute(2, 0, 3, 1, 4);  reshape_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_8 = qkv_17.unbind(0);  qkv_17 = None
    q_8 = unbind_8[0]
    k_8 = unbind_8[1]
    v_8 = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:148, code: if self.relative_position_bias_table is not None:
    l__mod___blocks_8_attn_relative_position_bias_table = self.L__mod___blocks_8_attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    l__mod___blocks_8_attn_relative_position_index = self.L__mod___blocks_8_attn_relative_position_index
    view_16 = l__mod___blocks_8_attn_relative_position_index.view(-1);  l__mod___blocks_8_attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    getitem_35 = l__mod___blocks_8_attn_relative_position_bias_table[view_16];  l__mod___blocks_8_attn_relative_position_bias_table = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    relative_position_bias_16 = getitem_35.view(197, 197, -1);  getitem_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_17 = relative_position_bias_16.permute(2, 0, 1);  relative_position_bias_16 = None
    relative_position_bias_17 = permute_17.contiguous();  permute_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    rel_pos_bias_8 = relative_position_bias_17.unsqueeze(0);  relative_position_bias_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    x_127 = torch._C._nn.scaled_dot_product_attention(q_8, k_8, v_8, attn_mask = rel_pos_bias_8, dropout_p = 0.0);  q_8 = k_8 = v_8 = rel_pos_bias_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_9 = x_127.transpose(1, 2);  x_127 = None
    x_128 = transpose_9.reshape(8, 197, 768);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    x_129 = self.L__mod___blocks_8_attn_proj(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    x_130 = self.L__mod___blocks_8_attn_proj_drop(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_16 = l__mod___blocks_8_gamma_1 * x_130;  l__mod___blocks_8_gamma_1 = x_130 = None
    l__mod___blocks_8_drop_path1 = self.L__mod___blocks_8_drop_path1(mul_16);  mul_16 = None
    x_131 = x_125 + l__mod___blocks_8_drop_path1;  x_125 = l__mod___blocks_8_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    l__mod___blocks_8_gamma_2 = self.L__mod___blocks_8_gamma_2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_8_norm2_weight = self.L__mod___blocks_8_norm2_weight
    l__mod___blocks_8_norm2_bias = self.L__mod___blocks_8_norm2_bias
    x_132 = torch.nn.functional.layer_norm(x_131, (768,), l__mod___blocks_8_norm2_weight, l__mod___blocks_8_norm2_bias, 1e-06);  l__mod___blocks_8_norm2_weight = l__mod___blocks_8_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_133 = self.L__mod___blocks_8_mlp_fc1(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_134 = self.L__mod___blocks_8_mlp_act(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_135 = self.L__mod___blocks_8_mlp_drop1(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_136 = self.L__mod___blocks_8_mlp_norm(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_137 = self.L__mod___blocks_8_mlp_fc2(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_138 = self.L__mod___blocks_8_mlp_drop2(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_17 = l__mod___blocks_8_gamma_2 * x_138;  l__mod___blocks_8_gamma_2 = x_138 = None
    l__mod___blocks_8_drop_path2 = self.L__mod___blocks_8_drop_path2(mul_17);  mul_17 = None
    x_140 = x_131 + l__mod___blocks_8_drop_path2;  x_131 = l__mod___blocks_8_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:237, code: if self.gamma_1 is None:
    l__mod___blocks_9_gamma_1 = self.L__mod___blocks_9_gamma_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_9_norm1_weight = self.L__mod___blocks_9_norm1_weight
    l__mod___blocks_9_norm1_bias = self.L__mod___blocks_9_norm1_bias
    x_141 = torch.nn.functional.layer_norm(x_140, (768,), l__mod___blocks_9_norm1_weight, l__mod___blocks_9_norm1_bias, 1e-06);  l__mod___blocks_9_norm1_weight = l__mod___blocks_9_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    l__mod___blocks_9_attn_q_bias = self.L__mod___blocks_9_attn_q_bias
    l__mod___blocks_9_attn_k_bias = self.L__mod___blocks_9_attn_k_bias
    l__mod___blocks_9_attn_v_bias = self.L__mod___blocks_9_attn_v_bias
    qkv_bias_9 = torch.cat((l__mod___blocks_9_attn_q_bias, l__mod___blocks_9_attn_k_bias, l__mod___blocks_9_attn_v_bias));  l__mod___blocks_9_attn_q_bias = l__mod___blocks_9_attn_k_bias = l__mod___blocks_9_attn_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    l__mod___blocks_9_attn_qkv_weight = self.L__mod___blocks_9_attn_qkv_weight
    qkv_18 = torch._C._nn.linear(input = x_141, weight = l__mod___blocks_9_attn_qkv_weight, bias = qkv_bias_9);  x_141 = l__mod___blocks_9_attn_qkv_weight = qkv_bias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    reshape_18 = qkv_18.reshape(8, 197, 3, 12, -1);  qkv_18 = None
    qkv_19 = reshape_18.permute(2, 0, 3, 1, 4);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_9 = qkv_19.unbind(0);  qkv_19 = None
    q_9 = unbind_9[0]
    k_9 = unbind_9[1]
    v_9 = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:148, code: if self.relative_position_bias_table is not None:
    l__mod___blocks_9_attn_relative_position_bias_table = self.L__mod___blocks_9_attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    l__mod___blocks_9_attn_relative_position_index = self.L__mod___blocks_9_attn_relative_position_index
    view_18 = l__mod___blocks_9_attn_relative_position_index.view(-1);  l__mod___blocks_9_attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    getitem_39 = l__mod___blocks_9_attn_relative_position_bias_table[view_18];  l__mod___blocks_9_attn_relative_position_bias_table = view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    relative_position_bias_18 = getitem_39.view(197, 197, -1);  getitem_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_19 = relative_position_bias_18.permute(2, 0, 1);  relative_position_bias_18 = None
    relative_position_bias_19 = permute_19.contiguous();  permute_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    rel_pos_bias_9 = relative_position_bias_19.unsqueeze(0);  relative_position_bias_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    x_142 = torch._C._nn.scaled_dot_product_attention(q_9, k_9, v_9, attn_mask = rel_pos_bias_9, dropout_p = 0.0);  q_9 = k_9 = v_9 = rel_pos_bias_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_10 = x_142.transpose(1, 2);  x_142 = None
    x_143 = transpose_10.reshape(8, 197, 768);  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    x_144 = self.L__mod___blocks_9_attn_proj(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    x_145 = self.L__mod___blocks_9_attn_proj_drop(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_18 = l__mod___blocks_9_gamma_1 * x_145;  l__mod___blocks_9_gamma_1 = x_145 = None
    l__mod___blocks_9_drop_path1 = self.L__mod___blocks_9_drop_path1(mul_18);  mul_18 = None
    x_146 = x_140 + l__mod___blocks_9_drop_path1;  x_140 = l__mod___blocks_9_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    l__mod___blocks_9_gamma_2 = self.L__mod___blocks_9_gamma_2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_9_norm2_weight = self.L__mod___blocks_9_norm2_weight
    l__mod___blocks_9_norm2_bias = self.L__mod___blocks_9_norm2_bias
    x_147 = torch.nn.functional.layer_norm(x_146, (768,), l__mod___blocks_9_norm2_weight, l__mod___blocks_9_norm2_bias, 1e-06);  l__mod___blocks_9_norm2_weight = l__mod___blocks_9_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_148 = self.L__mod___blocks_9_mlp_fc1(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_149 = self.L__mod___blocks_9_mlp_act(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_150 = self.L__mod___blocks_9_mlp_drop1(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_151 = self.L__mod___blocks_9_mlp_norm(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_152 = self.L__mod___blocks_9_mlp_fc2(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_153 = self.L__mod___blocks_9_mlp_drop2(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_19 = l__mod___blocks_9_gamma_2 * x_153;  l__mod___blocks_9_gamma_2 = x_153 = None
    l__mod___blocks_9_drop_path2 = self.L__mod___blocks_9_drop_path2(mul_19);  mul_19 = None
    x_155 = x_146 + l__mod___blocks_9_drop_path2;  x_146 = l__mod___blocks_9_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:237, code: if self.gamma_1 is None:
    l__mod___blocks_10_gamma_1 = self.L__mod___blocks_10_gamma_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_10_norm1_weight = self.L__mod___blocks_10_norm1_weight
    l__mod___blocks_10_norm1_bias = self.L__mod___blocks_10_norm1_bias
    x_156 = torch.nn.functional.layer_norm(x_155, (768,), l__mod___blocks_10_norm1_weight, l__mod___blocks_10_norm1_bias, 1e-06);  l__mod___blocks_10_norm1_weight = l__mod___blocks_10_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    l__mod___blocks_10_attn_q_bias = self.L__mod___blocks_10_attn_q_bias
    l__mod___blocks_10_attn_k_bias = self.L__mod___blocks_10_attn_k_bias
    l__mod___blocks_10_attn_v_bias = self.L__mod___blocks_10_attn_v_bias
    qkv_bias_10 = torch.cat((l__mod___blocks_10_attn_q_bias, l__mod___blocks_10_attn_k_bias, l__mod___blocks_10_attn_v_bias));  l__mod___blocks_10_attn_q_bias = l__mod___blocks_10_attn_k_bias = l__mod___blocks_10_attn_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    l__mod___blocks_10_attn_qkv_weight = self.L__mod___blocks_10_attn_qkv_weight
    qkv_20 = torch._C._nn.linear(input = x_156, weight = l__mod___blocks_10_attn_qkv_weight, bias = qkv_bias_10);  x_156 = l__mod___blocks_10_attn_qkv_weight = qkv_bias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    reshape_20 = qkv_20.reshape(8, 197, 3, 12, -1);  qkv_20 = None
    qkv_21 = reshape_20.permute(2, 0, 3, 1, 4);  reshape_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_10 = qkv_21.unbind(0);  qkv_21 = None
    q_10 = unbind_10[0]
    k_10 = unbind_10[1]
    v_10 = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:148, code: if self.relative_position_bias_table is not None:
    l__mod___blocks_10_attn_relative_position_bias_table = self.L__mod___blocks_10_attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    l__mod___blocks_10_attn_relative_position_index = self.L__mod___blocks_10_attn_relative_position_index
    view_20 = l__mod___blocks_10_attn_relative_position_index.view(-1);  l__mod___blocks_10_attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    getitem_43 = l__mod___blocks_10_attn_relative_position_bias_table[view_20];  l__mod___blocks_10_attn_relative_position_bias_table = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    relative_position_bias_20 = getitem_43.view(197, 197, -1);  getitem_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_21 = relative_position_bias_20.permute(2, 0, 1);  relative_position_bias_20 = None
    relative_position_bias_21 = permute_21.contiguous();  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    rel_pos_bias_10 = relative_position_bias_21.unsqueeze(0);  relative_position_bias_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    x_157 = torch._C._nn.scaled_dot_product_attention(q_10, k_10, v_10, attn_mask = rel_pos_bias_10, dropout_p = 0.0);  q_10 = k_10 = v_10 = rel_pos_bias_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_11 = x_157.transpose(1, 2);  x_157 = None
    x_158 = transpose_11.reshape(8, 197, 768);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    x_159 = self.L__mod___blocks_10_attn_proj(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    x_160 = self.L__mod___blocks_10_attn_proj_drop(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_20 = l__mod___blocks_10_gamma_1 * x_160;  l__mod___blocks_10_gamma_1 = x_160 = None
    l__mod___blocks_10_drop_path1 = self.L__mod___blocks_10_drop_path1(mul_20);  mul_20 = None
    x_161 = x_155 + l__mod___blocks_10_drop_path1;  x_155 = l__mod___blocks_10_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    l__mod___blocks_10_gamma_2 = self.L__mod___blocks_10_gamma_2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_10_norm2_weight = self.L__mod___blocks_10_norm2_weight
    l__mod___blocks_10_norm2_bias = self.L__mod___blocks_10_norm2_bias
    x_162 = torch.nn.functional.layer_norm(x_161, (768,), l__mod___blocks_10_norm2_weight, l__mod___blocks_10_norm2_bias, 1e-06);  l__mod___blocks_10_norm2_weight = l__mod___blocks_10_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_163 = self.L__mod___blocks_10_mlp_fc1(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_164 = self.L__mod___blocks_10_mlp_act(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_165 = self.L__mod___blocks_10_mlp_drop1(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_166 = self.L__mod___blocks_10_mlp_norm(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_167 = self.L__mod___blocks_10_mlp_fc2(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_168 = self.L__mod___blocks_10_mlp_drop2(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_21 = l__mod___blocks_10_gamma_2 * x_168;  l__mod___blocks_10_gamma_2 = x_168 = None
    l__mod___blocks_10_drop_path2 = self.L__mod___blocks_10_drop_path2(mul_21);  mul_21 = None
    x_170 = x_161 + l__mod___blocks_10_drop_path2;  x_161 = l__mod___blocks_10_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:237, code: if self.gamma_1 is None:
    l__mod___blocks_11_gamma_1 = self.L__mod___blocks_11_gamma_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_11_norm1_weight = self.L__mod___blocks_11_norm1_weight
    l__mod___blocks_11_norm1_bias = self.L__mod___blocks_11_norm1_bias
    x_171 = torch.nn.functional.layer_norm(x_170, (768,), l__mod___blocks_11_norm1_weight, l__mod___blocks_11_norm1_bias, 1e-06);  l__mod___blocks_11_norm1_weight = l__mod___blocks_11_norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:141, code: qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
    l__mod___blocks_11_attn_q_bias = self.L__mod___blocks_11_attn_q_bias
    l__mod___blocks_11_attn_k_bias = self.L__mod___blocks_11_attn_k_bias
    l__mod___blocks_11_attn_v_bias = self.L__mod___blocks_11_attn_v_bias
    qkv_bias_11 = torch.cat((l__mod___blocks_11_attn_q_bias, l__mod___blocks_11_attn_k_bias, l__mod___blocks_11_attn_v_bias));  l__mod___blocks_11_attn_q_bias = l__mod___blocks_11_attn_k_bias = l__mod___blocks_11_attn_v_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:142, code: qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
    l__mod___blocks_11_attn_qkv_weight = self.L__mod___blocks_11_attn_qkv_weight
    qkv_22 = torch._C._nn.linear(input = x_171, weight = l__mod___blocks_11_attn_qkv_weight, bias = qkv_bias_11);  x_171 = l__mod___blocks_11_attn_qkv_weight = qkv_bias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:143, code: qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    reshape_22 = qkv_22.reshape(8, 197, 3, 12, -1);  qkv_22 = None
    qkv_23 = reshape_22.permute(2, 0, 3, 1, 4);  reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:144, code: q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
    unbind_11 = qkv_23.unbind(0);  qkv_23 = None
    q_11 = unbind_11[0]
    k_11 = unbind_11[1]
    v_11 = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:148, code: if self.relative_position_bias_table is not None:
    l__mod___blocks_11_attn_relative_position_bias_table = self.L__mod___blocks_11_attn_relative_position_bias_table
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    l__mod___blocks_11_attn_relative_position_index = self.L__mod___blocks_11_attn_relative_position_index
    view_22 = l__mod___blocks_11_attn_relative_position_index.view(-1);  l__mod___blocks_11_attn_relative_position_index = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:131, code: relative_position_bias = self.relative_position_bias_table[
    getitem_47 = l__mod___blocks_11_attn_relative_position_bias_table[view_22];  l__mod___blocks_11_attn_relative_position_bias_table = view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:132, code: self.relative_position_index.view(-1)].view(
    relative_position_bias_22 = getitem_47.view(197, 197, -1);  getitem_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:135, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
    permute_23 = relative_position_bias_22.permute(2, 0, 1);  relative_position_bias_22 = None
    relative_position_bias_23 = permute_23.contiguous();  permute_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:136, code: return relative_position_bias.unsqueeze(0)
    rel_pos_bias_11 = relative_position_bias_23.unsqueeze(0);  relative_position_bias_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:155, code: x = F.scaled_dot_product_attention(
    x_172 = torch._C._nn.scaled_dot_product_attention(q_11, k_11, v_11, attn_mask = rel_pos_bias_11, dropout_p = 0.0);  q_11 = k_11 = v_11 = rel_pos_bias_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:173, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_12 = x_172.transpose(1, 2);  x_172 = None
    x_173 = transpose_12.reshape(8, 197, 768);  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:174, code: x = self.proj(x)
    x_174 = self.L__mod___blocks_11_attn_proj(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:175, code: x = self.proj_drop(x)
    x_175 = self.L__mod___blocks_11_attn_proj_drop(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:241, code: x = x + self.drop_path1(self.gamma_1 * self.attn(self.norm1(x), shared_rel_pos_bias=shared_rel_pos_bias))
    mul_22 = l__mod___blocks_11_gamma_1 * x_175;  l__mod___blocks_11_gamma_1 = x_175 = None
    l__mod___blocks_11_drop_path1 = self.L__mod___blocks_11_drop_path1(mul_22);  mul_22 = None
    x_176 = x_170 + l__mod___blocks_11_drop_path1;  x_170 = l__mod___blocks_11_drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    l__mod___blocks_11_gamma_2 = self.L__mod___blocks_11_gamma_2
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___blocks_11_norm2_weight = self.L__mod___blocks_11_norm2_weight
    l__mod___blocks_11_norm2_bias = self.L__mod___blocks_11_norm2_bias
    x_177 = torch.nn.functional.layer_norm(x_176, (768,), l__mod___blocks_11_norm2_weight, l__mod___blocks_11_norm2_bias, 1e-06);  l__mod___blocks_11_norm2_weight = l__mod___blocks_11_norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_178 = self.L__mod___blocks_11_mlp_fc1(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_179 = self.L__mod___blocks_11_mlp_act(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_180 = self.L__mod___blocks_11_mlp_drop1(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_181 = self.L__mod___blocks_11_mlp_norm(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_182 = self.L__mod___blocks_11_mlp_fc2(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_183 = self.L__mod___blocks_11_mlp_drop2(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:242, code: x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_23 = l__mod___blocks_11_gamma_2 * x_183;  l__mod___blocks_11_gamma_2 = x_183 = None
    l__mod___blocks_11_drop_path2 = self.L__mod___blocks_11_drop_path2(mul_23);  mul_23 = None
    x_185 = x_176 + l__mod___blocks_11_drop_path2;  x_176 = l__mod___blocks_11_drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:416, code: x = self.norm(x)
    x_187 = self.L__mod___norm(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:421, code: x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    getitem_48 = x_187[(slice(None, None, None), slice(1, None, None))];  x_187 = None
    x_188 = getitem_48.mean(dim = 1);  getitem_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:57, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___fc_norm_weight = self.L__mod___fc_norm_weight
    l__mod___fc_norm_bias = self.L__mod___fc_norm_bias
    x_190 = torch.nn.functional.layer_norm(x_188, (768,), l__mod___fc_norm_weight, l__mod___fc_norm_bias, 1e-06);  x_188 = l__mod___fc_norm_weight = l__mod___fc_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:423, code: x = self.head_drop(x)
    x_191 = self.L__mod___head_drop(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/beit.py:424, code: return x if pre_logits else self.head(x)
    x_192 = self.L__mod___head(x_191);  x_191 = None
    return (x_192,)
    