from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:87, code: x = self.proj(x)
    x = self.L__mod___patch_embed_proj(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:89, code: x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
    flatten = x.flatten(2);  x = None
    x_1 = flatten.transpose(1, 2);  flatten = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/patch_embed.py:92, code: x = self.norm(x)
    x_3 = self.L__mod___patch_embed_norm(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:86, code: pos_embed = self.pos_embed
    pos_embed = self.L__mod___pos_embed
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:100, code: self.cls_token.expand(x.shape[0], -1, -1),
    l__mod___cls_token = self.L__mod___cls_token
    expand = l__mod___cls_token.expand(8, -1, -1);  l__mod___cls_token = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:101, code: self.dist_token.expand(x.shape[0], -1, -1),
    l__mod___dist_token = self.L__mod___dist_token
    expand_1 = l__mod___dist_token.expand(8, -1, -1);  l__mod___dist_token = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:99, code: x = torch.cat((
    x_4 = torch.cat((expand, expand_1, x_3), dim = 1);  expand = expand_1 = x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:104, code: x = x + pos_embed
    x_5 = x_4 + pos_embed;  x_4 = pos_embed = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:105, code: return self.pos_drop(x)
    x_6 = self.L__mod___pos_drop(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:635, code: x = self.patch_drop(x)
    x_7 = self.L__mod___patch_drop(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:636, code: x = self.norm_pre(x)
    x_8 = self.L__mod___norm_pre(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___0___norm1 = self.getattr_L__mod___blocks___0___norm1(x_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___0___attn_qkv = self.getattr_L__mod___blocks___0___attn_qkv(getattr_l__mod___blocks___0___norm1);  getattr_l__mod___blocks___0___norm1 = None
    reshape = getattr_l__mod___blocks___0___attn_qkv.reshape(8, 198, 3, 12, 64);  getattr_l__mod___blocks___0___attn_qkv = None
    qkv = reshape.permute(2, 0, 3, 1, 4);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = qkv.unbind(0);  qkv = None
    q = unbind[0]
    k = unbind[1]
    v = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_1 = self.getattr_L__mod___blocks___0___attn_q_norm(q);  q = None
    k_1 = self.getattr_L__mod___blocks___0___attn_k_norm(k);  k = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_9 = torch._C._nn.scaled_dot_product_attention(q_1, k_1, v, dropout_p = 0.0);  q_1 = k_1 = v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_1 = x_9.transpose(1, 2);  x_9 = None
    x_10 = transpose_1.reshape(8, 198, 768);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_11 = self.getattr_L__mod___blocks___0___attn_proj(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_12 = self.getattr_L__mod___blocks___0___attn_proj_drop(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___0___ls1 = self.getattr_L__mod___blocks___0___ls1(x_12);  x_12 = None
    getattr_l__mod___blocks___0___drop_path1 = self.getattr_L__mod___blocks___0___drop_path1(getattr_l__mod___blocks___0___ls1);  getattr_l__mod___blocks___0___ls1 = None
    x_13 = x_8 + getattr_l__mod___blocks___0___drop_path1;  x_8 = getattr_l__mod___blocks___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___0___norm2 = self.getattr_L__mod___blocks___0___norm2(x_13)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_14 = self.getattr_L__mod___blocks___0___mlp_fc1(getattr_l__mod___blocks___0___norm2);  getattr_l__mod___blocks___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_15 = self.getattr_L__mod___blocks___0___mlp_act(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_16 = self.getattr_L__mod___blocks___0___mlp_drop1(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_17 = self.getattr_L__mod___blocks___0___mlp_norm(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_18 = self.getattr_L__mod___blocks___0___mlp_fc2(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_19 = self.getattr_L__mod___blocks___0___mlp_drop2(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___0___ls2 = self.getattr_L__mod___blocks___0___ls2(x_19);  x_19 = None
    getattr_l__mod___blocks___0___drop_path2 = self.getattr_L__mod___blocks___0___drop_path2(getattr_l__mod___blocks___0___ls2);  getattr_l__mod___blocks___0___ls2 = None
    x_20 = x_13 + getattr_l__mod___blocks___0___drop_path2;  x_13 = getattr_l__mod___blocks___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___1___norm1 = self.getattr_L__mod___blocks___1___norm1(x_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___1___attn_qkv = self.getattr_L__mod___blocks___1___attn_qkv(getattr_l__mod___blocks___1___norm1);  getattr_l__mod___blocks___1___norm1 = None
    reshape_2 = getattr_l__mod___blocks___1___attn_qkv.reshape(8, 198, 3, 12, 64);  getattr_l__mod___blocks___1___attn_qkv = None
    qkv_1 = reshape_2.permute(2, 0, 3, 1, 4);  reshape_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = qkv_1.unbind(0);  qkv_1 = None
    q_2 = unbind_1[0]
    k_2 = unbind_1[1]
    v_1 = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_3 = self.getattr_L__mod___blocks___1___attn_q_norm(q_2);  q_2 = None
    k_3 = self.getattr_L__mod___blocks___1___attn_k_norm(k_2);  k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_21 = torch._C._nn.scaled_dot_product_attention(q_3, k_3, v_1, dropout_p = 0.0);  q_3 = k_3 = v_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_2 = x_21.transpose(1, 2);  x_21 = None
    x_22 = transpose_2.reshape(8, 198, 768);  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_23 = self.getattr_L__mod___blocks___1___attn_proj(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_24 = self.getattr_L__mod___blocks___1___attn_proj_drop(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___1___ls1 = self.getattr_L__mod___blocks___1___ls1(x_24);  x_24 = None
    getattr_l__mod___blocks___1___drop_path1 = self.getattr_L__mod___blocks___1___drop_path1(getattr_l__mod___blocks___1___ls1);  getattr_l__mod___blocks___1___ls1 = None
    x_25 = x_20 + getattr_l__mod___blocks___1___drop_path1;  x_20 = getattr_l__mod___blocks___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___1___norm2 = self.getattr_L__mod___blocks___1___norm2(x_25)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_26 = self.getattr_L__mod___blocks___1___mlp_fc1(getattr_l__mod___blocks___1___norm2);  getattr_l__mod___blocks___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_27 = self.getattr_L__mod___blocks___1___mlp_act(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_28 = self.getattr_L__mod___blocks___1___mlp_drop1(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_29 = self.getattr_L__mod___blocks___1___mlp_norm(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_30 = self.getattr_L__mod___blocks___1___mlp_fc2(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_31 = self.getattr_L__mod___blocks___1___mlp_drop2(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___1___ls2 = self.getattr_L__mod___blocks___1___ls2(x_31);  x_31 = None
    getattr_l__mod___blocks___1___drop_path2 = self.getattr_L__mod___blocks___1___drop_path2(getattr_l__mod___blocks___1___ls2);  getattr_l__mod___blocks___1___ls2 = None
    x_32 = x_25 + getattr_l__mod___blocks___1___drop_path2;  x_25 = getattr_l__mod___blocks___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___2___norm1 = self.getattr_L__mod___blocks___2___norm1(x_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___2___attn_qkv = self.getattr_L__mod___blocks___2___attn_qkv(getattr_l__mod___blocks___2___norm1);  getattr_l__mod___blocks___2___norm1 = None
    reshape_4 = getattr_l__mod___blocks___2___attn_qkv.reshape(8, 198, 3, 12, 64);  getattr_l__mod___blocks___2___attn_qkv = None
    qkv_2 = reshape_4.permute(2, 0, 3, 1, 4);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = qkv_2.unbind(0);  qkv_2 = None
    q_4 = unbind_2[0]
    k_4 = unbind_2[1]
    v_2 = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_5 = self.getattr_L__mod___blocks___2___attn_q_norm(q_4);  q_4 = None
    k_5 = self.getattr_L__mod___blocks___2___attn_k_norm(k_4);  k_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_33 = torch._C._nn.scaled_dot_product_attention(q_5, k_5, v_2, dropout_p = 0.0);  q_5 = k_5 = v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_3 = x_33.transpose(1, 2);  x_33 = None
    x_34 = transpose_3.reshape(8, 198, 768);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_35 = self.getattr_L__mod___blocks___2___attn_proj(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_36 = self.getattr_L__mod___blocks___2___attn_proj_drop(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___2___ls1 = self.getattr_L__mod___blocks___2___ls1(x_36);  x_36 = None
    getattr_l__mod___blocks___2___drop_path1 = self.getattr_L__mod___blocks___2___drop_path1(getattr_l__mod___blocks___2___ls1);  getattr_l__mod___blocks___2___ls1 = None
    x_37 = x_32 + getattr_l__mod___blocks___2___drop_path1;  x_32 = getattr_l__mod___blocks___2___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___2___norm2 = self.getattr_L__mod___blocks___2___norm2(x_37)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_38 = self.getattr_L__mod___blocks___2___mlp_fc1(getattr_l__mod___blocks___2___norm2);  getattr_l__mod___blocks___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_39 = self.getattr_L__mod___blocks___2___mlp_act(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_40 = self.getattr_L__mod___blocks___2___mlp_drop1(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_41 = self.getattr_L__mod___blocks___2___mlp_norm(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_42 = self.getattr_L__mod___blocks___2___mlp_fc2(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_43 = self.getattr_L__mod___blocks___2___mlp_drop2(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___2___ls2 = self.getattr_L__mod___blocks___2___ls2(x_43);  x_43 = None
    getattr_l__mod___blocks___2___drop_path2 = self.getattr_L__mod___blocks___2___drop_path2(getattr_l__mod___blocks___2___ls2);  getattr_l__mod___blocks___2___ls2 = None
    x_44 = x_37 + getattr_l__mod___blocks___2___drop_path2;  x_37 = getattr_l__mod___blocks___2___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___3___norm1 = self.getattr_L__mod___blocks___3___norm1(x_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___3___attn_qkv = self.getattr_L__mod___blocks___3___attn_qkv(getattr_l__mod___blocks___3___norm1);  getattr_l__mod___blocks___3___norm1 = None
    reshape_6 = getattr_l__mod___blocks___3___attn_qkv.reshape(8, 198, 3, 12, 64);  getattr_l__mod___blocks___3___attn_qkv = None
    qkv_3 = reshape_6.permute(2, 0, 3, 1, 4);  reshape_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = qkv_3.unbind(0);  qkv_3 = None
    q_6 = unbind_3[0]
    k_6 = unbind_3[1]
    v_3 = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_7 = self.getattr_L__mod___blocks___3___attn_q_norm(q_6);  q_6 = None
    k_7 = self.getattr_L__mod___blocks___3___attn_k_norm(k_6);  k_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_45 = torch._C._nn.scaled_dot_product_attention(q_7, k_7, v_3, dropout_p = 0.0);  q_7 = k_7 = v_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_4 = x_45.transpose(1, 2);  x_45 = None
    x_46 = transpose_4.reshape(8, 198, 768);  transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_47 = self.getattr_L__mod___blocks___3___attn_proj(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_48 = self.getattr_L__mod___blocks___3___attn_proj_drop(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___3___ls1 = self.getattr_L__mod___blocks___3___ls1(x_48);  x_48 = None
    getattr_l__mod___blocks___3___drop_path1 = self.getattr_L__mod___blocks___3___drop_path1(getattr_l__mod___blocks___3___ls1);  getattr_l__mod___blocks___3___ls1 = None
    x_49 = x_44 + getattr_l__mod___blocks___3___drop_path1;  x_44 = getattr_l__mod___blocks___3___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___3___norm2 = self.getattr_L__mod___blocks___3___norm2(x_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_50 = self.getattr_L__mod___blocks___3___mlp_fc1(getattr_l__mod___blocks___3___norm2);  getattr_l__mod___blocks___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_51 = self.getattr_L__mod___blocks___3___mlp_act(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_52 = self.getattr_L__mod___blocks___3___mlp_drop1(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_53 = self.getattr_L__mod___blocks___3___mlp_norm(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_54 = self.getattr_L__mod___blocks___3___mlp_fc2(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_55 = self.getattr_L__mod___blocks___3___mlp_drop2(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___3___ls2 = self.getattr_L__mod___blocks___3___ls2(x_55);  x_55 = None
    getattr_l__mod___blocks___3___drop_path2 = self.getattr_L__mod___blocks___3___drop_path2(getattr_l__mod___blocks___3___ls2);  getattr_l__mod___blocks___3___ls2 = None
    x_56 = x_49 + getattr_l__mod___blocks___3___drop_path2;  x_49 = getattr_l__mod___blocks___3___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___4___norm1 = self.getattr_L__mod___blocks___4___norm1(x_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___4___attn_qkv = self.getattr_L__mod___blocks___4___attn_qkv(getattr_l__mod___blocks___4___norm1);  getattr_l__mod___blocks___4___norm1 = None
    reshape_8 = getattr_l__mod___blocks___4___attn_qkv.reshape(8, 198, 3, 12, 64);  getattr_l__mod___blocks___4___attn_qkv = None
    qkv_4 = reshape_8.permute(2, 0, 3, 1, 4);  reshape_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = qkv_4.unbind(0);  qkv_4 = None
    q_8 = unbind_4[0]
    k_8 = unbind_4[1]
    v_4 = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_9 = self.getattr_L__mod___blocks___4___attn_q_norm(q_8);  q_8 = None
    k_9 = self.getattr_L__mod___blocks___4___attn_k_norm(k_8);  k_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_57 = torch._C._nn.scaled_dot_product_attention(q_9, k_9, v_4, dropout_p = 0.0);  q_9 = k_9 = v_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_5 = x_57.transpose(1, 2);  x_57 = None
    x_58 = transpose_5.reshape(8, 198, 768);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_59 = self.getattr_L__mod___blocks___4___attn_proj(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_60 = self.getattr_L__mod___blocks___4___attn_proj_drop(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___4___ls1 = self.getattr_L__mod___blocks___4___ls1(x_60);  x_60 = None
    getattr_l__mod___blocks___4___drop_path1 = self.getattr_L__mod___blocks___4___drop_path1(getattr_l__mod___blocks___4___ls1);  getattr_l__mod___blocks___4___ls1 = None
    x_61 = x_56 + getattr_l__mod___blocks___4___drop_path1;  x_56 = getattr_l__mod___blocks___4___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___4___norm2 = self.getattr_L__mod___blocks___4___norm2(x_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_62 = self.getattr_L__mod___blocks___4___mlp_fc1(getattr_l__mod___blocks___4___norm2);  getattr_l__mod___blocks___4___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_63 = self.getattr_L__mod___blocks___4___mlp_act(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_64 = self.getattr_L__mod___blocks___4___mlp_drop1(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_65 = self.getattr_L__mod___blocks___4___mlp_norm(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_66 = self.getattr_L__mod___blocks___4___mlp_fc2(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_67 = self.getattr_L__mod___blocks___4___mlp_drop2(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___4___ls2 = self.getattr_L__mod___blocks___4___ls2(x_67);  x_67 = None
    getattr_l__mod___blocks___4___drop_path2 = self.getattr_L__mod___blocks___4___drop_path2(getattr_l__mod___blocks___4___ls2);  getattr_l__mod___blocks___4___ls2 = None
    x_68 = x_61 + getattr_l__mod___blocks___4___drop_path2;  x_61 = getattr_l__mod___blocks___4___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___5___norm1 = self.getattr_L__mod___blocks___5___norm1(x_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___5___attn_qkv = self.getattr_L__mod___blocks___5___attn_qkv(getattr_l__mod___blocks___5___norm1);  getattr_l__mod___blocks___5___norm1 = None
    reshape_10 = getattr_l__mod___blocks___5___attn_qkv.reshape(8, 198, 3, 12, 64);  getattr_l__mod___blocks___5___attn_qkv = None
    qkv_5 = reshape_10.permute(2, 0, 3, 1, 4);  reshape_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = qkv_5.unbind(0);  qkv_5 = None
    q_10 = unbind_5[0]
    k_10 = unbind_5[1]
    v_5 = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_11 = self.getattr_L__mod___blocks___5___attn_q_norm(q_10);  q_10 = None
    k_11 = self.getattr_L__mod___blocks___5___attn_k_norm(k_10);  k_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_69 = torch._C._nn.scaled_dot_product_attention(q_11, k_11, v_5, dropout_p = 0.0);  q_11 = k_11 = v_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_6 = x_69.transpose(1, 2);  x_69 = None
    x_70 = transpose_6.reshape(8, 198, 768);  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_71 = self.getattr_L__mod___blocks___5___attn_proj(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_72 = self.getattr_L__mod___blocks___5___attn_proj_drop(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___5___ls1 = self.getattr_L__mod___blocks___5___ls1(x_72);  x_72 = None
    getattr_l__mod___blocks___5___drop_path1 = self.getattr_L__mod___blocks___5___drop_path1(getattr_l__mod___blocks___5___ls1);  getattr_l__mod___blocks___5___ls1 = None
    x_73 = x_68 + getattr_l__mod___blocks___5___drop_path1;  x_68 = getattr_l__mod___blocks___5___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___5___norm2 = self.getattr_L__mod___blocks___5___norm2(x_73)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_74 = self.getattr_L__mod___blocks___5___mlp_fc1(getattr_l__mod___blocks___5___norm2);  getattr_l__mod___blocks___5___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_75 = self.getattr_L__mod___blocks___5___mlp_act(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_76 = self.getattr_L__mod___blocks___5___mlp_drop1(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_77 = self.getattr_L__mod___blocks___5___mlp_norm(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_78 = self.getattr_L__mod___blocks___5___mlp_fc2(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_79 = self.getattr_L__mod___blocks___5___mlp_drop2(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___5___ls2 = self.getattr_L__mod___blocks___5___ls2(x_79);  x_79 = None
    getattr_l__mod___blocks___5___drop_path2 = self.getattr_L__mod___blocks___5___drop_path2(getattr_l__mod___blocks___5___ls2);  getattr_l__mod___blocks___5___ls2 = None
    x_80 = x_73 + getattr_l__mod___blocks___5___drop_path2;  x_73 = getattr_l__mod___blocks___5___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___6___norm1 = self.getattr_L__mod___blocks___6___norm1(x_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___6___attn_qkv = self.getattr_L__mod___blocks___6___attn_qkv(getattr_l__mod___blocks___6___norm1);  getattr_l__mod___blocks___6___norm1 = None
    reshape_12 = getattr_l__mod___blocks___6___attn_qkv.reshape(8, 198, 3, 12, 64);  getattr_l__mod___blocks___6___attn_qkv = None
    qkv_6 = reshape_12.permute(2, 0, 3, 1, 4);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = qkv_6.unbind(0);  qkv_6 = None
    q_12 = unbind_6[0]
    k_12 = unbind_6[1]
    v_6 = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_13 = self.getattr_L__mod___blocks___6___attn_q_norm(q_12);  q_12 = None
    k_13 = self.getattr_L__mod___blocks___6___attn_k_norm(k_12);  k_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_81 = torch._C._nn.scaled_dot_product_attention(q_13, k_13, v_6, dropout_p = 0.0);  q_13 = k_13 = v_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_7 = x_81.transpose(1, 2);  x_81 = None
    x_82 = transpose_7.reshape(8, 198, 768);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_83 = self.getattr_L__mod___blocks___6___attn_proj(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_84 = self.getattr_L__mod___blocks___6___attn_proj_drop(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___6___ls1 = self.getattr_L__mod___blocks___6___ls1(x_84);  x_84 = None
    getattr_l__mod___blocks___6___drop_path1 = self.getattr_L__mod___blocks___6___drop_path1(getattr_l__mod___blocks___6___ls1);  getattr_l__mod___blocks___6___ls1 = None
    x_85 = x_80 + getattr_l__mod___blocks___6___drop_path1;  x_80 = getattr_l__mod___blocks___6___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___6___norm2 = self.getattr_L__mod___blocks___6___norm2(x_85)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_86 = self.getattr_L__mod___blocks___6___mlp_fc1(getattr_l__mod___blocks___6___norm2);  getattr_l__mod___blocks___6___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_87 = self.getattr_L__mod___blocks___6___mlp_act(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_88 = self.getattr_L__mod___blocks___6___mlp_drop1(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_89 = self.getattr_L__mod___blocks___6___mlp_norm(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_90 = self.getattr_L__mod___blocks___6___mlp_fc2(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_91 = self.getattr_L__mod___blocks___6___mlp_drop2(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___6___ls2 = self.getattr_L__mod___blocks___6___ls2(x_91);  x_91 = None
    getattr_l__mod___blocks___6___drop_path2 = self.getattr_L__mod___blocks___6___drop_path2(getattr_l__mod___blocks___6___ls2);  getattr_l__mod___blocks___6___ls2 = None
    x_92 = x_85 + getattr_l__mod___blocks___6___drop_path2;  x_85 = getattr_l__mod___blocks___6___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___7___norm1 = self.getattr_L__mod___blocks___7___norm1(x_92)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___7___attn_qkv = self.getattr_L__mod___blocks___7___attn_qkv(getattr_l__mod___blocks___7___norm1);  getattr_l__mod___blocks___7___norm1 = None
    reshape_14 = getattr_l__mod___blocks___7___attn_qkv.reshape(8, 198, 3, 12, 64);  getattr_l__mod___blocks___7___attn_qkv = None
    qkv_7 = reshape_14.permute(2, 0, 3, 1, 4);  reshape_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = qkv_7.unbind(0);  qkv_7 = None
    q_14 = unbind_7[0]
    k_14 = unbind_7[1]
    v_7 = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_15 = self.getattr_L__mod___blocks___7___attn_q_norm(q_14);  q_14 = None
    k_15 = self.getattr_L__mod___blocks___7___attn_k_norm(k_14);  k_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_93 = torch._C._nn.scaled_dot_product_attention(q_15, k_15, v_7, dropout_p = 0.0);  q_15 = k_15 = v_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_8 = x_93.transpose(1, 2);  x_93 = None
    x_94 = transpose_8.reshape(8, 198, 768);  transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_95 = self.getattr_L__mod___blocks___7___attn_proj(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_96 = self.getattr_L__mod___blocks___7___attn_proj_drop(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___7___ls1 = self.getattr_L__mod___blocks___7___ls1(x_96);  x_96 = None
    getattr_l__mod___blocks___7___drop_path1 = self.getattr_L__mod___blocks___7___drop_path1(getattr_l__mod___blocks___7___ls1);  getattr_l__mod___blocks___7___ls1 = None
    x_97 = x_92 + getattr_l__mod___blocks___7___drop_path1;  x_92 = getattr_l__mod___blocks___7___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___7___norm2 = self.getattr_L__mod___blocks___7___norm2(x_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_98 = self.getattr_L__mod___blocks___7___mlp_fc1(getattr_l__mod___blocks___7___norm2);  getattr_l__mod___blocks___7___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_99 = self.getattr_L__mod___blocks___7___mlp_act(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_100 = self.getattr_L__mod___blocks___7___mlp_drop1(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_101 = self.getattr_L__mod___blocks___7___mlp_norm(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_102 = self.getattr_L__mod___blocks___7___mlp_fc2(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_103 = self.getattr_L__mod___blocks___7___mlp_drop2(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___7___ls2 = self.getattr_L__mod___blocks___7___ls2(x_103);  x_103 = None
    getattr_l__mod___blocks___7___drop_path2 = self.getattr_L__mod___blocks___7___drop_path2(getattr_l__mod___blocks___7___ls2);  getattr_l__mod___blocks___7___ls2 = None
    x_104 = x_97 + getattr_l__mod___blocks___7___drop_path2;  x_97 = getattr_l__mod___blocks___7___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___8___norm1 = self.getattr_L__mod___blocks___8___norm1(x_104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___8___attn_qkv = self.getattr_L__mod___blocks___8___attn_qkv(getattr_l__mod___blocks___8___norm1);  getattr_l__mod___blocks___8___norm1 = None
    reshape_16 = getattr_l__mod___blocks___8___attn_qkv.reshape(8, 198, 3, 12, 64);  getattr_l__mod___blocks___8___attn_qkv = None
    qkv_8 = reshape_16.permute(2, 0, 3, 1, 4);  reshape_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = qkv_8.unbind(0);  qkv_8 = None
    q_16 = unbind_8[0]
    k_16 = unbind_8[1]
    v_8 = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_17 = self.getattr_L__mod___blocks___8___attn_q_norm(q_16);  q_16 = None
    k_17 = self.getattr_L__mod___blocks___8___attn_k_norm(k_16);  k_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_105 = torch._C._nn.scaled_dot_product_attention(q_17, k_17, v_8, dropout_p = 0.0);  q_17 = k_17 = v_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_9 = x_105.transpose(1, 2);  x_105 = None
    x_106 = transpose_9.reshape(8, 198, 768);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_107 = self.getattr_L__mod___blocks___8___attn_proj(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_108 = self.getattr_L__mod___blocks___8___attn_proj_drop(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___8___ls1 = self.getattr_L__mod___blocks___8___ls1(x_108);  x_108 = None
    getattr_l__mod___blocks___8___drop_path1 = self.getattr_L__mod___blocks___8___drop_path1(getattr_l__mod___blocks___8___ls1);  getattr_l__mod___blocks___8___ls1 = None
    x_109 = x_104 + getattr_l__mod___blocks___8___drop_path1;  x_104 = getattr_l__mod___blocks___8___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___8___norm2 = self.getattr_L__mod___blocks___8___norm2(x_109)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_110 = self.getattr_L__mod___blocks___8___mlp_fc1(getattr_l__mod___blocks___8___norm2);  getattr_l__mod___blocks___8___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_111 = self.getattr_L__mod___blocks___8___mlp_act(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_112 = self.getattr_L__mod___blocks___8___mlp_drop1(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_113 = self.getattr_L__mod___blocks___8___mlp_norm(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_114 = self.getattr_L__mod___blocks___8___mlp_fc2(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_115 = self.getattr_L__mod___blocks___8___mlp_drop2(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___8___ls2 = self.getattr_L__mod___blocks___8___ls2(x_115);  x_115 = None
    getattr_l__mod___blocks___8___drop_path2 = self.getattr_L__mod___blocks___8___drop_path2(getattr_l__mod___blocks___8___ls2);  getattr_l__mod___blocks___8___ls2 = None
    x_116 = x_109 + getattr_l__mod___blocks___8___drop_path2;  x_109 = getattr_l__mod___blocks___8___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___9___norm1 = self.getattr_L__mod___blocks___9___norm1(x_116)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___9___attn_qkv = self.getattr_L__mod___blocks___9___attn_qkv(getattr_l__mod___blocks___9___norm1);  getattr_l__mod___blocks___9___norm1 = None
    reshape_18 = getattr_l__mod___blocks___9___attn_qkv.reshape(8, 198, 3, 12, 64);  getattr_l__mod___blocks___9___attn_qkv = None
    qkv_9 = reshape_18.permute(2, 0, 3, 1, 4);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_9 = qkv_9.unbind(0);  qkv_9 = None
    q_18 = unbind_9[0]
    k_18 = unbind_9[1]
    v_9 = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_19 = self.getattr_L__mod___blocks___9___attn_q_norm(q_18);  q_18 = None
    k_19 = self.getattr_L__mod___blocks___9___attn_k_norm(k_18);  k_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_117 = torch._C._nn.scaled_dot_product_attention(q_19, k_19, v_9, dropout_p = 0.0);  q_19 = k_19 = v_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_10 = x_117.transpose(1, 2);  x_117 = None
    x_118 = transpose_10.reshape(8, 198, 768);  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_119 = self.getattr_L__mod___blocks___9___attn_proj(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_120 = self.getattr_L__mod___blocks___9___attn_proj_drop(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___9___ls1 = self.getattr_L__mod___blocks___9___ls1(x_120);  x_120 = None
    getattr_l__mod___blocks___9___drop_path1 = self.getattr_L__mod___blocks___9___drop_path1(getattr_l__mod___blocks___9___ls1);  getattr_l__mod___blocks___9___ls1 = None
    x_121 = x_116 + getattr_l__mod___blocks___9___drop_path1;  x_116 = getattr_l__mod___blocks___9___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___9___norm2 = self.getattr_L__mod___blocks___9___norm2(x_121)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_122 = self.getattr_L__mod___blocks___9___mlp_fc1(getattr_l__mod___blocks___9___norm2);  getattr_l__mod___blocks___9___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_123 = self.getattr_L__mod___blocks___9___mlp_act(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_124 = self.getattr_L__mod___blocks___9___mlp_drop1(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_125 = self.getattr_L__mod___blocks___9___mlp_norm(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_126 = self.getattr_L__mod___blocks___9___mlp_fc2(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_127 = self.getattr_L__mod___blocks___9___mlp_drop2(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___9___ls2 = self.getattr_L__mod___blocks___9___ls2(x_127);  x_127 = None
    getattr_l__mod___blocks___9___drop_path2 = self.getattr_L__mod___blocks___9___drop_path2(getattr_l__mod___blocks___9___ls2);  getattr_l__mod___blocks___9___ls2 = None
    x_128 = x_121 + getattr_l__mod___blocks___9___drop_path2;  x_121 = getattr_l__mod___blocks___9___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___10___norm1 = self.getattr_L__mod___blocks___10___norm1(x_128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___10___attn_qkv = self.getattr_L__mod___blocks___10___attn_qkv(getattr_l__mod___blocks___10___norm1);  getattr_l__mod___blocks___10___norm1 = None
    reshape_20 = getattr_l__mod___blocks___10___attn_qkv.reshape(8, 198, 3, 12, 64);  getattr_l__mod___blocks___10___attn_qkv = None
    qkv_10 = reshape_20.permute(2, 0, 3, 1, 4);  reshape_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_10 = qkv_10.unbind(0);  qkv_10 = None
    q_20 = unbind_10[0]
    k_20 = unbind_10[1]
    v_10 = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_21 = self.getattr_L__mod___blocks___10___attn_q_norm(q_20);  q_20 = None
    k_21 = self.getattr_L__mod___blocks___10___attn_k_norm(k_20);  k_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_129 = torch._C._nn.scaled_dot_product_attention(q_21, k_21, v_10, dropout_p = 0.0);  q_21 = k_21 = v_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_11 = x_129.transpose(1, 2);  x_129 = None
    x_130 = transpose_11.reshape(8, 198, 768);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_131 = self.getattr_L__mod___blocks___10___attn_proj(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_132 = self.getattr_L__mod___blocks___10___attn_proj_drop(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___10___ls1 = self.getattr_L__mod___blocks___10___ls1(x_132);  x_132 = None
    getattr_l__mod___blocks___10___drop_path1 = self.getattr_L__mod___blocks___10___drop_path1(getattr_l__mod___blocks___10___ls1);  getattr_l__mod___blocks___10___ls1 = None
    x_133 = x_128 + getattr_l__mod___blocks___10___drop_path1;  x_128 = getattr_l__mod___blocks___10___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___10___norm2 = self.getattr_L__mod___blocks___10___norm2(x_133)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_134 = self.getattr_L__mod___blocks___10___mlp_fc1(getattr_l__mod___blocks___10___norm2);  getattr_l__mod___blocks___10___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_135 = self.getattr_L__mod___blocks___10___mlp_act(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_136 = self.getattr_L__mod___blocks___10___mlp_drop1(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_137 = self.getattr_L__mod___blocks___10___mlp_norm(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_138 = self.getattr_L__mod___blocks___10___mlp_fc2(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_139 = self.getattr_L__mod___blocks___10___mlp_drop2(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___10___ls2 = self.getattr_L__mod___blocks___10___ls2(x_139);  x_139 = None
    getattr_l__mod___blocks___10___drop_path2 = self.getattr_L__mod___blocks___10___drop_path2(getattr_l__mod___blocks___10___ls2);  getattr_l__mod___blocks___10___ls2 = None
    x_140 = x_133 + getattr_l__mod___blocks___10___drop_path2;  x_133 = getattr_l__mod___blocks___10___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___11___norm1 = self.getattr_L__mod___blocks___11___norm1(x_140)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___11___attn_qkv = self.getattr_L__mod___blocks___11___attn_qkv(getattr_l__mod___blocks___11___norm1);  getattr_l__mod___blocks___11___norm1 = None
    reshape_22 = getattr_l__mod___blocks___11___attn_qkv.reshape(8, 198, 3, 12, 64);  getattr_l__mod___blocks___11___attn_qkv = None
    qkv_11 = reshape_22.permute(2, 0, 3, 1, 4);  reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_11 = qkv_11.unbind(0);  qkv_11 = None
    q_22 = unbind_11[0]
    k_22 = unbind_11[1]
    v_11 = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_23 = self.getattr_L__mod___blocks___11___attn_q_norm(q_22);  q_22 = None
    k_23 = self.getattr_L__mod___blocks___11___attn_k_norm(k_22);  k_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_141 = torch._C._nn.scaled_dot_product_attention(q_23, k_23, v_11, dropout_p = 0.0);  q_23 = k_23 = v_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_12 = x_141.transpose(1, 2);  x_141 = None
    x_142 = transpose_12.reshape(8, 198, 768);  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_143 = self.getattr_L__mod___blocks___11___attn_proj(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_144 = self.getattr_L__mod___blocks___11___attn_proj_drop(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks___11___ls1 = self.getattr_L__mod___blocks___11___ls1(x_144);  x_144 = None
    getattr_l__mod___blocks___11___drop_path1 = self.getattr_L__mod___blocks___11___drop_path1(getattr_l__mod___blocks___11___ls1);  getattr_l__mod___blocks___11___ls1 = None
    x_145 = x_140 + getattr_l__mod___blocks___11___drop_path1;  x_140 = getattr_l__mod___blocks___11___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___11___norm2 = self.getattr_L__mod___blocks___11___norm2(x_145)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_146 = self.getattr_L__mod___blocks___11___mlp_fc1(getattr_l__mod___blocks___11___norm2);  getattr_l__mod___blocks___11___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_147 = self.getattr_L__mod___blocks___11___mlp_act(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_148 = self.getattr_L__mod___blocks___11___mlp_drop1(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_149 = self.getattr_L__mod___blocks___11___mlp_norm(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_150 = self.getattr_L__mod___blocks___11___mlp_fc2(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_151 = self.getattr_L__mod___blocks___11___mlp_drop2(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks___11___ls2 = self.getattr_L__mod___blocks___11___ls2(x_151);  x_151 = None
    getattr_l__mod___blocks___11___drop_path2 = self.getattr_L__mod___blocks___11___drop_path2(getattr_l__mod___blocks___11___ls2);  getattr_l__mod___blocks___11___ls2 = None
    x_153 = x_145 + getattr_l__mod___blocks___11___drop_path2;  x_145 = getattr_l__mod___blocks___11___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:641, code: x = self.norm(x)
    x_155 = self.L__mod___norm(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:108, code: x, x_dist = x[:, 0], x[:, 1]
    x_156 = x_155[(slice(None, None, None), 0)]
    x_dist = x_155[(slice(None, None, None), 1)];  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:111, code: x = self.head(x)
    x_157 = self.L__mod___head(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:112, code: x_dist = self.head_dist(x_dist)
    x_dist_1 = self.L__mod___head_dist(x_dist);  x_dist = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/deit.py:118, code: return (x + x_dist) / 2
    add_25 = x_157 + x_dist_1;  x_157 = x_dist_1 = None
    pred = add_25 / 2;  add_25 = None
    return (pred,)
    