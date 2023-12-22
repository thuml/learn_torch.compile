from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:138, code: x = self.conv(x)
    x = self.L__mod___patch_embed_conv(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:257, code: x = self.pos_drop(x + self.pos_embed)
    l__mod___pos_embed = self.L__mod___pos_embed
    add = x + l__mod___pos_embed;  x = l__mod___pos_embed = None
    x_2 = self.L__mod___pos_drop(add);  add = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:258, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    l__mod___cls_token = self.L__mod___cls_token
    cls_tokens = l__mod___cls_token.expand(8, -1, -1);  l__mod___cls_token = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:81, code: x = x.flatten(2).transpose(1, 2)
    flatten = x_2.flatten(2);  x_2 = None
    x_3 = flatten.transpose(1, 2);  flatten = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:82, code: x = torch.cat((cls_tokens, x), dim=1)
    x_4 = torch.cat((cls_tokens, x_3), dim = 1);  cls_tokens = x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:84, code: x = self.norm(x)
    x_5 = self.L__mod___transformers_0_norm(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_0_blocks___0___norm1 = self.getattr_L__mod___transformers_0_blocks___0___norm1(x_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_0_blocks___0___attn_qkv = self.getattr_L__mod___transformers_0_blocks___0___attn_qkv(getattr_l__mod___transformers_0_blocks___0___norm1);  getattr_l__mod___transformers_0_blocks___0___norm1 = None
    reshape = getattr_l__mod___transformers_0_blocks___0___attn_qkv.reshape(8, 962, 3, 4, 64);  getattr_l__mod___transformers_0_blocks___0___attn_qkv = None
    qkv = reshape.permute(2, 0, 3, 1, 4);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = qkv.unbind(0);  qkv = None
    q = unbind[0]
    k = unbind[1]
    v = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_1 = self.getattr_L__mod___transformers_0_blocks___0___attn_q_norm(q);  q = None
    k_1 = self.getattr_L__mod___transformers_0_blocks___0___attn_k_norm(k);  k = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_6 = torch._C._nn.scaled_dot_product_attention(q_1, k_1, v, dropout_p = 0.0);  q_1 = k_1 = v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_1 = x_6.transpose(1, 2);  x_6 = None
    x_7 = transpose_1.reshape(8, 962, 256);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_8 = self.getattr_L__mod___transformers_0_blocks___0___attn_proj(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_9 = self.getattr_L__mod___transformers_0_blocks___0___attn_proj_drop(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_0_blocks___0___ls1 = self.getattr_L__mod___transformers_0_blocks___0___ls1(x_9);  x_9 = None
    getattr_l__mod___transformers_0_blocks___0___drop_path1 = self.getattr_L__mod___transformers_0_blocks___0___drop_path1(getattr_l__mod___transformers_0_blocks___0___ls1);  getattr_l__mod___transformers_0_blocks___0___ls1 = None
    x_10 = x_5 + getattr_l__mod___transformers_0_blocks___0___drop_path1;  x_5 = getattr_l__mod___transformers_0_blocks___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_0_blocks___0___norm2 = self.getattr_L__mod___transformers_0_blocks___0___norm2(x_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_11 = self.getattr_L__mod___transformers_0_blocks___0___mlp_fc1(getattr_l__mod___transformers_0_blocks___0___norm2);  getattr_l__mod___transformers_0_blocks___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_12 = self.getattr_L__mod___transformers_0_blocks___0___mlp_act(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_13 = self.getattr_L__mod___transformers_0_blocks___0___mlp_drop1(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_14 = self.getattr_L__mod___transformers_0_blocks___0___mlp_norm(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_15 = self.getattr_L__mod___transformers_0_blocks___0___mlp_fc2(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_16 = self.getattr_L__mod___transformers_0_blocks___0___mlp_drop2(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_0_blocks___0___ls2 = self.getattr_L__mod___transformers_0_blocks___0___ls2(x_16);  x_16 = None
    getattr_l__mod___transformers_0_blocks___0___drop_path2 = self.getattr_L__mod___transformers_0_blocks___0___drop_path2(getattr_l__mod___transformers_0_blocks___0___ls2);  getattr_l__mod___transformers_0_blocks___0___ls2 = None
    x_17 = x_10 + getattr_l__mod___transformers_0_blocks___0___drop_path2;  x_10 = getattr_l__mod___transformers_0_blocks___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_0_blocks___1___norm1 = self.getattr_L__mod___transformers_0_blocks___1___norm1(x_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_0_blocks___1___attn_qkv = self.getattr_L__mod___transformers_0_blocks___1___attn_qkv(getattr_l__mod___transformers_0_blocks___1___norm1);  getattr_l__mod___transformers_0_blocks___1___norm1 = None
    reshape_2 = getattr_l__mod___transformers_0_blocks___1___attn_qkv.reshape(8, 962, 3, 4, 64);  getattr_l__mod___transformers_0_blocks___1___attn_qkv = None
    qkv_1 = reshape_2.permute(2, 0, 3, 1, 4);  reshape_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = qkv_1.unbind(0);  qkv_1 = None
    q_2 = unbind_1[0]
    k_2 = unbind_1[1]
    v_1 = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_3 = self.getattr_L__mod___transformers_0_blocks___1___attn_q_norm(q_2);  q_2 = None
    k_3 = self.getattr_L__mod___transformers_0_blocks___1___attn_k_norm(k_2);  k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_18 = torch._C._nn.scaled_dot_product_attention(q_3, k_3, v_1, dropout_p = 0.0);  q_3 = k_3 = v_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_2 = x_18.transpose(1, 2);  x_18 = None
    x_19 = transpose_2.reshape(8, 962, 256);  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_20 = self.getattr_L__mod___transformers_0_blocks___1___attn_proj(x_19);  x_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_21 = self.getattr_L__mod___transformers_0_blocks___1___attn_proj_drop(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_0_blocks___1___ls1 = self.getattr_L__mod___transformers_0_blocks___1___ls1(x_21);  x_21 = None
    getattr_l__mod___transformers_0_blocks___1___drop_path1 = self.getattr_L__mod___transformers_0_blocks___1___drop_path1(getattr_l__mod___transformers_0_blocks___1___ls1);  getattr_l__mod___transformers_0_blocks___1___ls1 = None
    x_22 = x_17 + getattr_l__mod___transformers_0_blocks___1___drop_path1;  x_17 = getattr_l__mod___transformers_0_blocks___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_0_blocks___1___norm2 = self.getattr_L__mod___transformers_0_blocks___1___norm2(x_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_23 = self.getattr_L__mod___transformers_0_blocks___1___mlp_fc1(getattr_l__mod___transformers_0_blocks___1___norm2);  getattr_l__mod___transformers_0_blocks___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_24 = self.getattr_L__mod___transformers_0_blocks___1___mlp_act(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_25 = self.getattr_L__mod___transformers_0_blocks___1___mlp_drop1(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_26 = self.getattr_L__mod___transformers_0_blocks___1___mlp_norm(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_27 = self.getattr_L__mod___transformers_0_blocks___1___mlp_fc2(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_28 = self.getattr_L__mod___transformers_0_blocks___1___mlp_drop2(x_27);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_0_blocks___1___ls2 = self.getattr_L__mod___transformers_0_blocks___1___ls2(x_28);  x_28 = None
    getattr_l__mod___transformers_0_blocks___1___drop_path2 = self.getattr_L__mod___transformers_0_blocks___1___drop_path2(getattr_l__mod___transformers_0_blocks___1___ls2);  getattr_l__mod___transformers_0_blocks___1___ls2 = None
    x_29 = x_22 + getattr_l__mod___transformers_0_blocks___1___drop_path2;  x_22 = getattr_l__mod___transformers_0_blocks___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_0_blocks___2___norm1 = self.getattr_L__mod___transformers_0_blocks___2___norm1(x_29)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_0_blocks___2___attn_qkv = self.getattr_L__mod___transformers_0_blocks___2___attn_qkv(getattr_l__mod___transformers_0_blocks___2___norm1);  getattr_l__mod___transformers_0_blocks___2___norm1 = None
    reshape_4 = getattr_l__mod___transformers_0_blocks___2___attn_qkv.reshape(8, 962, 3, 4, 64);  getattr_l__mod___transformers_0_blocks___2___attn_qkv = None
    qkv_2 = reshape_4.permute(2, 0, 3, 1, 4);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = qkv_2.unbind(0);  qkv_2 = None
    q_4 = unbind_2[0]
    k_4 = unbind_2[1]
    v_2 = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_5 = self.getattr_L__mod___transformers_0_blocks___2___attn_q_norm(q_4);  q_4 = None
    k_5 = self.getattr_L__mod___transformers_0_blocks___2___attn_k_norm(k_4);  k_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_30 = torch._C._nn.scaled_dot_product_attention(q_5, k_5, v_2, dropout_p = 0.0);  q_5 = k_5 = v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_3 = x_30.transpose(1, 2);  x_30 = None
    x_31 = transpose_3.reshape(8, 962, 256);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_32 = self.getattr_L__mod___transformers_0_blocks___2___attn_proj(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_33 = self.getattr_L__mod___transformers_0_blocks___2___attn_proj_drop(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_0_blocks___2___ls1 = self.getattr_L__mod___transformers_0_blocks___2___ls1(x_33);  x_33 = None
    getattr_l__mod___transformers_0_blocks___2___drop_path1 = self.getattr_L__mod___transformers_0_blocks___2___drop_path1(getattr_l__mod___transformers_0_blocks___2___ls1);  getattr_l__mod___transformers_0_blocks___2___ls1 = None
    x_34 = x_29 + getattr_l__mod___transformers_0_blocks___2___drop_path1;  x_29 = getattr_l__mod___transformers_0_blocks___2___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_0_blocks___2___norm2 = self.getattr_L__mod___transformers_0_blocks___2___norm2(x_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_35 = self.getattr_L__mod___transformers_0_blocks___2___mlp_fc1(getattr_l__mod___transformers_0_blocks___2___norm2);  getattr_l__mod___transformers_0_blocks___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_36 = self.getattr_L__mod___transformers_0_blocks___2___mlp_act(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_37 = self.getattr_L__mod___transformers_0_blocks___2___mlp_drop1(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_38 = self.getattr_L__mod___transformers_0_blocks___2___mlp_norm(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_39 = self.getattr_L__mod___transformers_0_blocks___2___mlp_fc2(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_40 = self.getattr_L__mod___transformers_0_blocks___2___mlp_drop2(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_0_blocks___2___ls2 = self.getattr_L__mod___transformers_0_blocks___2___ls2(x_40);  x_40 = None
    getattr_l__mod___transformers_0_blocks___2___drop_path2 = self.getattr_L__mod___transformers_0_blocks___2___drop_path2(getattr_l__mod___transformers_0_blocks___2___ls2);  getattr_l__mod___transformers_0_blocks___2___ls2 = None
    x_42 = x_34 + getattr_l__mod___transformers_0_blocks___2___drop_path2;  x_34 = getattr_l__mod___transformers_0_blocks___2___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    cls_tokens_2 = x_42[(slice(None, None, None), slice(None, 1, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:88, code: x = x[:, token_length:]
    x_43 = x_42[(slice(None, None, None), slice(1, None, None))];  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:89, code: x = x.transpose(1, 2).reshape(B, C, H, W)
    transpose_4 = x_43.transpose(1, 2);  x_43 = None
    x_45 = transpose_4.reshape(8, 256, 31, 31);  transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:110, code: x = self.conv(x)
    x_47 = self.L__mod___transformers_1_pool_conv(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:111, code: cls_token = self.fc(cls_token)
    cls_tokens_3 = self.L__mod___transformers_1_pool_fc(cls_tokens_2);  cls_tokens_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:81, code: x = x.flatten(2).transpose(1, 2)
    flatten_1 = x_47.flatten(2);  x_47 = None
    x_48 = flatten_1.transpose(1, 2);  flatten_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:82, code: x = torch.cat((cls_tokens, x), dim=1)
    x_49 = torch.cat((cls_tokens_3, x_48), dim = 1);  cls_tokens_3 = x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:84, code: x = self.norm(x)
    x_50 = self.L__mod___transformers_1_norm(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_1_blocks___0___norm1 = self.getattr_L__mod___transformers_1_blocks___0___norm1(x_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_1_blocks___0___attn_qkv = self.getattr_L__mod___transformers_1_blocks___0___attn_qkv(getattr_l__mod___transformers_1_blocks___0___norm1);  getattr_l__mod___transformers_1_blocks___0___norm1 = None
    reshape_7 = getattr_l__mod___transformers_1_blocks___0___attn_qkv.reshape(8, 257, 3, 8, 64);  getattr_l__mod___transformers_1_blocks___0___attn_qkv = None
    qkv_3 = reshape_7.permute(2, 0, 3, 1, 4);  reshape_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = qkv_3.unbind(0);  qkv_3 = None
    q_6 = unbind_3[0]
    k_6 = unbind_3[1]
    v_3 = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_7 = self.getattr_L__mod___transformers_1_blocks___0___attn_q_norm(q_6);  q_6 = None
    k_7 = self.getattr_L__mod___transformers_1_blocks___0___attn_k_norm(k_6);  k_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_51 = torch._C._nn.scaled_dot_product_attention(q_7, k_7, v_3, dropout_p = 0.0);  q_7 = k_7 = v_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_6 = x_51.transpose(1, 2);  x_51 = None
    x_52 = transpose_6.reshape(8, 257, 512);  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_53 = self.getattr_L__mod___transformers_1_blocks___0___attn_proj(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_54 = self.getattr_L__mod___transformers_1_blocks___0___attn_proj_drop(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_1_blocks___0___ls1 = self.getattr_L__mod___transformers_1_blocks___0___ls1(x_54);  x_54 = None
    getattr_l__mod___transformers_1_blocks___0___drop_path1 = self.getattr_L__mod___transformers_1_blocks___0___drop_path1(getattr_l__mod___transformers_1_blocks___0___ls1);  getattr_l__mod___transformers_1_blocks___0___ls1 = None
    x_55 = x_50 + getattr_l__mod___transformers_1_blocks___0___drop_path1;  x_50 = getattr_l__mod___transformers_1_blocks___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_1_blocks___0___norm2 = self.getattr_L__mod___transformers_1_blocks___0___norm2(x_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_56 = self.getattr_L__mod___transformers_1_blocks___0___mlp_fc1(getattr_l__mod___transformers_1_blocks___0___norm2);  getattr_l__mod___transformers_1_blocks___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_57 = self.getattr_L__mod___transformers_1_blocks___0___mlp_act(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_58 = self.getattr_L__mod___transformers_1_blocks___0___mlp_drop1(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_59 = self.getattr_L__mod___transformers_1_blocks___0___mlp_norm(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_60 = self.getattr_L__mod___transformers_1_blocks___0___mlp_fc2(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_61 = self.getattr_L__mod___transformers_1_blocks___0___mlp_drop2(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_1_blocks___0___ls2 = self.getattr_L__mod___transformers_1_blocks___0___ls2(x_61);  x_61 = None
    getattr_l__mod___transformers_1_blocks___0___drop_path2 = self.getattr_L__mod___transformers_1_blocks___0___drop_path2(getattr_l__mod___transformers_1_blocks___0___ls2);  getattr_l__mod___transformers_1_blocks___0___ls2 = None
    x_62 = x_55 + getattr_l__mod___transformers_1_blocks___0___drop_path2;  x_55 = getattr_l__mod___transformers_1_blocks___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_1_blocks___1___norm1 = self.getattr_L__mod___transformers_1_blocks___1___norm1(x_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_1_blocks___1___attn_qkv = self.getattr_L__mod___transformers_1_blocks___1___attn_qkv(getattr_l__mod___transformers_1_blocks___1___norm1);  getattr_l__mod___transformers_1_blocks___1___norm1 = None
    reshape_9 = getattr_l__mod___transformers_1_blocks___1___attn_qkv.reshape(8, 257, 3, 8, 64);  getattr_l__mod___transformers_1_blocks___1___attn_qkv = None
    qkv_4 = reshape_9.permute(2, 0, 3, 1, 4);  reshape_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = qkv_4.unbind(0);  qkv_4 = None
    q_8 = unbind_4[0]
    k_8 = unbind_4[1]
    v_4 = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_9 = self.getattr_L__mod___transformers_1_blocks___1___attn_q_norm(q_8);  q_8 = None
    k_9 = self.getattr_L__mod___transformers_1_blocks___1___attn_k_norm(k_8);  k_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_63 = torch._C._nn.scaled_dot_product_attention(q_9, k_9, v_4, dropout_p = 0.0);  q_9 = k_9 = v_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_7 = x_63.transpose(1, 2);  x_63 = None
    x_64 = transpose_7.reshape(8, 257, 512);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_65 = self.getattr_L__mod___transformers_1_blocks___1___attn_proj(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_66 = self.getattr_L__mod___transformers_1_blocks___1___attn_proj_drop(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_1_blocks___1___ls1 = self.getattr_L__mod___transformers_1_blocks___1___ls1(x_66);  x_66 = None
    getattr_l__mod___transformers_1_blocks___1___drop_path1 = self.getattr_L__mod___transformers_1_blocks___1___drop_path1(getattr_l__mod___transformers_1_blocks___1___ls1);  getattr_l__mod___transformers_1_blocks___1___ls1 = None
    x_67 = x_62 + getattr_l__mod___transformers_1_blocks___1___drop_path1;  x_62 = getattr_l__mod___transformers_1_blocks___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_1_blocks___1___norm2 = self.getattr_L__mod___transformers_1_blocks___1___norm2(x_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_68 = self.getattr_L__mod___transformers_1_blocks___1___mlp_fc1(getattr_l__mod___transformers_1_blocks___1___norm2);  getattr_l__mod___transformers_1_blocks___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_69 = self.getattr_L__mod___transformers_1_blocks___1___mlp_act(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_70 = self.getattr_L__mod___transformers_1_blocks___1___mlp_drop1(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_71 = self.getattr_L__mod___transformers_1_blocks___1___mlp_norm(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_72 = self.getattr_L__mod___transformers_1_blocks___1___mlp_fc2(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_73 = self.getattr_L__mod___transformers_1_blocks___1___mlp_drop2(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_1_blocks___1___ls2 = self.getattr_L__mod___transformers_1_blocks___1___ls2(x_73);  x_73 = None
    getattr_l__mod___transformers_1_blocks___1___drop_path2 = self.getattr_L__mod___transformers_1_blocks___1___drop_path2(getattr_l__mod___transformers_1_blocks___1___ls2);  getattr_l__mod___transformers_1_blocks___1___ls2 = None
    x_74 = x_67 + getattr_l__mod___transformers_1_blocks___1___drop_path2;  x_67 = getattr_l__mod___transformers_1_blocks___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_1_blocks___2___norm1 = self.getattr_L__mod___transformers_1_blocks___2___norm1(x_74)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_1_blocks___2___attn_qkv = self.getattr_L__mod___transformers_1_blocks___2___attn_qkv(getattr_l__mod___transformers_1_blocks___2___norm1);  getattr_l__mod___transformers_1_blocks___2___norm1 = None
    reshape_11 = getattr_l__mod___transformers_1_blocks___2___attn_qkv.reshape(8, 257, 3, 8, 64);  getattr_l__mod___transformers_1_blocks___2___attn_qkv = None
    qkv_5 = reshape_11.permute(2, 0, 3, 1, 4);  reshape_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = qkv_5.unbind(0);  qkv_5 = None
    q_10 = unbind_5[0]
    k_10 = unbind_5[1]
    v_5 = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_11 = self.getattr_L__mod___transformers_1_blocks___2___attn_q_norm(q_10);  q_10 = None
    k_11 = self.getattr_L__mod___transformers_1_blocks___2___attn_k_norm(k_10);  k_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_75 = torch._C._nn.scaled_dot_product_attention(q_11, k_11, v_5, dropout_p = 0.0);  q_11 = k_11 = v_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_8 = x_75.transpose(1, 2);  x_75 = None
    x_76 = transpose_8.reshape(8, 257, 512);  transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_77 = self.getattr_L__mod___transformers_1_blocks___2___attn_proj(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_78 = self.getattr_L__mod___transformers_1_blocks___2___attn_proj_drop(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_1_blocks___2___ls1 = self.getattr_L__mod___transformers_1_blocks___2___ls1(x_78);  x_78 = None
    getattr_l__mod___transformers_1_blocks___2___drop_path1 = self.getattr_L__mod___transformers_1_blocks___2___drop_path1(getattr_l__mod___transformers_1_blocks___2___ls1);  getattr_l__mod___transformers_1_blocks___2___ls1 = None
    x_79 = x_74 + getattr_l__mod___transformers_1_blocks___2___drop_path1;  x_74 = getattr_l__mod___transformers_1_blocks___2___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_1_blocks___2___norm2 = self.getattr_L__mod___transformers_1_blocks___2___norm2(x_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_80 = self.getattr_L__mod___transformers_1_blocks___2___mlp_fc1(getattr_l__mod___transformers_1_blocks___2___norm2);  getattr_l__mod___transformers_1_blocks___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_81 = self.getattr_L__mod___transformers_1_blocks___2___mlp_act(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_82 = self.getattr_L__mod___transformers_1_blocks___2___mlp_drop1(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_83 = self.getattr_L__mod___transformers_1_blocks___2___mlp_norm(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_84 = self.getattr_L__mod___transformers_1_blocks___2___mlp_fc2(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_85 = self.getattr_L__mod___transformers_1_blocks___2___mlp_drop2(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_1_blocks___2___ls2 = self.getattr_L__mod___transformers_1_blocks___2___ls2(x_85);  x_85 = None
    getattr_l__mod___transformers_1_blocks___2___drop_path2 = self.getattr_L__mod___transformers_1_blocks___2___drop_path2(getattr_l__mod___transformers_1_blocks___2___ls2);  getattr_l__mod___transformers_1_blocks___2___ls2 = None
    x_86 = x_79 + getattr_l__mod___transformers_1_blocks___2___drop_path2;  x_79 = getattr_l__mod___transformers_1_blocks___2___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_1_blocks___3___norm1 = self.getattr_L__mod___transformers_1_blocks___3___norm1(x_86)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_1_blocks___3___attn_qkv = self.getattr_L__mod___transformers_1_blocks___3___attn_qkv(getattr_l__mod___transformers_1_blocks___3___norm1);  getattr_l__mod___transformers_1_blocks___3___norm1 = None
    reshape_13 = getattr_l__mod___transformers_1_blocks___3___attn_qkv.reshape(8, 257, 3, 8, 64);  getattr_l__mod___transformers_1_blocks___3___attn_qkv = None
    qkv_6 = reshape_13.permute(2, 0, 3, 1, 4);  reshape_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = qkv_6.unbind(0);  qkv_6 = None
    q_12 = unbind_6[0]
    k_12 = unbind_6[1]
    v_6 = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_13 = self.getattr_L__mod___transformers_1_blocks___3___attn_q_norm(q_12);  q_12 = None
    k_13 = self.getattr_L__mod___transformers_1_blocks___3___attn_k_norm(k_12);  k_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_87 = torch._C._nn.scaled_dot_product_attention(q_13, k_13, v_6, dropout_p = 0.0);  q_13 = k_13 = v_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_9 = x_87.transpose(1, 2);  x_87 = None
    x_88 = transpose_9.reshape(8, 257, 512);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_89 = self.getattr_L__mod___transformers_1_blocks___3___attn_proj(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_90 = self.getattr_L__mod___transformers_1_blocks___3___attn_proj_drop(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_1_blocks___3___ls1 = self.getattr_L__mod___transformers_1_blocks___3___ls1(x_90);  x_90 = None
    getattr_l__mod___transformers_1_blocks___3___drop_path1 = self.getattr_L__mod___transformers_1_blocks___3___drop_path1(getattr_l__mod___transformers_1_blocks___3___ls1);  getattr_l__mod___transformers_1_blocks___3___ls1 = None
    x_91 = x_86 + getattr_l__mod___transformers_1_blocks___3___drop_path1;  x_86 = getattr_l__mod___transformers_1_blocks___3___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_1_blocks___3___norm2 = self.getattr_L__mod___transformers_1_blocks___3___norm2(x_91)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_92 = self.getattr_L__mod___transformers_1_blocks___3___mlp_fc1(getattr_l__mod___transformers_1_blocks___3___norm2);  getattr_l__mod___transformers_1_blocks___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_93 = self.getattr_L__mod___transformers_1_blocks___3___mlp_act(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_94 = self.getattr_L__mod___transformers_1_blocks___3___mlp_drop1(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_95 = self.getattr_L__mod___transformers_1_blocks___3___mlp_norm(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_96 = self.getattr_L__mod___transformers_1_blocks___3___mlp_fc2(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_97 = self.getattr_L__mod___transformers_1_blocks___3___mlp_drop2(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_1_blocks___3___ls2 = self.getattr_L__mod___transformers_1_blocks___3___ls2(x_97);  x_97 = None
    getattr_l__mod___transformers_1_blocks___3___drop_path2 = self.getattr_L__mod___transformers_1_blocks___3___drop_path2(getattr_l__mod___transformers_1_blocks___3___ls2);  getattr_l__mod___transformers_1_blocks___3___ls2 = None
    x_98 = x_91 + getattr_l__mod___transformers_1_blocks___3___drop_path2;  x_91 = getattr_l__mod___transformers_1_blocks___3___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_1_blocks___4___norm1 = self.getattr_L__mod___transformers_1_blocks___4___norm1(x_98)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_1_blocks___4___attn_qkv = self.getattr_L__mod___transformers_1_blocks___4___attn_qkv(getattr_l__mod___transformers_1_blocks___4___norm1);  getattr_l__mod___transformers_1_blocks___4___norm1 = None
    reshape_15 = getattr_l__mod___transformers_1_blocks___4___attn_qkv.reshape(8, 257, 3, 8, 64);  getattr_l__mod___transformers_1_blocks___4___attn_qkv = None
    qkv_7 = reshape_15.permute(2, 0, 3, 1, 4);  reshape_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = qkv_7.unbind(0);  qkv_7 = None
    q_14 = unbind_7[0]
    k_14 = unbind_7[1]
    v_7 = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_15 = self.getattr_L__mod___transformers_1_blocks___4___attn_q_norm(q_14);  q_14 = None
    k_15 = self.getattr_L__mod___transformers_1_blocks___4___attn_k_norm(k_14);  k_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_99 = torch._C._nn.scaled_dot_product_attention(q_15, k_15, v_7, dropout_p = 0.0);  q_15 = k_15 = v_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_10 = x_99.transpose(1, 2);  x_99 = None
    x_100 = transpose_10.reshape(8, 257, 512);  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_101 = self.getattr_L__mod___transformers_1_blocks___4___attn_proj(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_102 = self.getattr_L__mod___transformers_1_blocks___4___attn_proj_drop(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_1_blocks___4___ls1 = self.getattr_L__mod___transformers_1_blocks___4___ls1(x_102);  x_102 = None
    getattr_l__mod___transformers_1_blocks___4___drop_path1 = self.getattr_L__mod___transformers_1_blocks___4___drop_path1(getattr_l__mod___transformers_1_blocks___4___ls1);  getattr_l__mod___transformers_1_blocks___4___ls1 = None
    x_103 = x_98 + getattr_l__mod___transformers_1_blocks___4___drop_path1;  x_98 = getattr_l__mod___transformers_1_blocks___4___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_1_blocks___4___norm2 = self.getattr_L__mod___transformers_1_blocks___4___norm2(x_103)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_104 = self.getattr_L__mod___transformers_1_blocks___4___mlp_fc1(getattr_l__mod___transformers_1_blocks___4___norm2);  getattr_l__mod___transformers_1_blocks___4___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_105 = self.getattr_L__mod___transformers_1_blocks___4___mlp_act(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_106 = self.getattr_L__mod___transformers_1_blocks___4___mlp_drop1(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_107 = self.getattr_L__mod___transformers_1_blocks___4___mlp_norm(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_108 = self.getattr_L__mod___transformers_1_blocks___4___mlp_fc2(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_109 = self.getattr_L__mod___transformers_1_blocks___4___mlp_drop2(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_1_blocks___4___ls2 = self.getattr_L__mod___transformers_1_blocks___4___ls2(x_109);  x_109 = None
    getattr_l__mod___transformers_1_blocks___4___drop_path2 = self.getattr_L__mod___transformers_1_blocks___4___drop_path2(getattr_l__mod___transformers_1_blocks___4___ls2);  getattr_l__mod___transformers_1_blocks___4___ls2 = None
    x_110 = x_103 + getattr_l__mod___transformers_1_blocks___4___drop_path2;  x_103 = getattr_l__mod___transformers_1_blocks___4___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_1_blocks___5___norm1 = self.getattr_L__mod___transformers_1_blocks___5___norm1(x_110)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_1_blocks___5___attn_qkv = self.getattr_L__mod___transformers_1_blocks___5___attn_qkv(getattr_l__mod___transformers_1_blocks___5___norm1);  getattr_l__mod___transformers_1_blocks___5___norm1 = None
    reshape_17 = getattr_l__mod___transformers_1_blocks___5___attn_qkv.reshape(8, 257, 3, 8, 64);  getattr_l__mod___transformers_1_blocks___5___attn_qkv = None
    qkv_8 = reshape_17.permute(2, 0, 3, 1, 4);  reshape_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = qkv_8.unbind(0);  qkv_8 = None
    q_16 = unbind_8[0]
    k_16 = unbind_8[1]
    v_8 = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_17 = self.getattr_L__mod___transformers_1_blocks___5___attn_q_norm(q_16);  q_16 = None
    k_17 = self.getattr_L__mod___transformers_1_blocks___5___attn_k_norm(k_16);  k_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_111 = torch._C._nn.scaled_dot_product_attention(q_17, k_17, v_8, dropout_p = 0.0);  q_17 = k_17 = v_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_11 = x_111.transpose(1, 2);  x_111 = None
    x_112 = transpose_11.reshape(8, 257, 512);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_113 = self.getattr_L__mod___transformers_1_blocks___5___attn_proj(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_114 = self.getattr_L__mod___transformers_1_blocks___5___attn_proj_drop(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_1_blocks___5___ls1 = self.getattr_L__mod___transformers_1_blocks___5___ls1(x_114);  x_114 = None
    getattr_l__mod___transformers_1_blocks___5___drop_path1 = self.getattr_L__mod___transformers_1_blocks___5___drop_path1(getattr_l__mod___transformers_1_blocks___5___ls1);  getattr_l__mod___transformers_1_blocks___5___ls1 = None
    x_115 = x_110 + getattr_l__mod___transformers_1_blocks___5___drop_path1;  x_110 = getattr_l__mod___transformers_1_blocks___5___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_1_blocks___5___norm2 = self.getattr_L__mod___transformers_1_blocks___5___norm2(x_115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_116 = self.getattr_L__mod___transformers_1_blocks___5___mlp_fc1(getattr_l__mod___transformers_1_blocks___5___norm2);  getattr_l__mod___transformers_1_blocks___5___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_117 = self.getattr_L__mod___transformers_1_blocks___5___mlp_act(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_118 = self.getattr_L__mod___transformers_1_blocks___5___mlp_drop1(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_119 = self.getattr_L__mod___transformers_1_blocks___5___mlp_norm(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_120 = self.getattr_L__mod___transformers_1_blocks___5___mlp_fc2(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_121 = self.getattr_L__mod___transformers_1_blocks___5___mlp_drop2(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_1_blocks___5___ls2 = self.getattr_L__mod___transformers_1_blocks___5___ls2(x_121);  x_121 = None
    getattr_l__mod___transformers_1_blocks___5___drop_path2 = self.getattr_L__mod___transformers_1_blocks___5___drop_path2(getattr_l__mod___transformers_1_blocks___5___ls2);  getattr_l__mod___transformers_1_blocks___5___ls2 = None
    x_123 = x_115 + getattr_l__mod___transformers_1_blocks___5___drop_path2;  x_115 = getattr_l__mod___transformers_1_blocks___5___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    cls_tokens_5 = x_123[(slice(None, None, None), slice(None, 1, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:88, code: x = x[:, token_length:]
    x_124 = x_123[(slice(None, None, None), slice(1, None, None))];  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:89, code: x = x.transpose(1, 2).reshape(B, C, H, W)
    transpose_12 = x_124.transpose(1, 2);  x_124 = None
    x_126 = transpose_12.reshape(8, 512, 16, 16);  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:110, code: x = self.conv(x)
    x_128 = self.L__mod___transformers_2_pool_conv(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:111, code: cls_token = self.fc(cls_token)
    cls_tokens_6 = self.L__mod___transformers_2_pool_fc(cls_tokens_5);  cls_tokens_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:81, code: x = x.flatten(2).transpose(1, 2)
    flatten_2 = x_128.flatten(2);  x_128 = None
    x_129 = flatten_2.transpose(1, 2);  flatten_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:82, code: x = torch.cat((cls_tokens, x), dim=1)
    x_130 = torch.cat((cls_tokens_6, x_129), dim = 1);  cls_tokens_6 = x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:84, code: x = self.norm(x)
    x_131 = self.L__mod___transformers_2_norm(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_2_blocks___0___norm1 = self.getattr_L__mod___transformers_2_blocks___0___norm1(x_131)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_2_blocks___0___attn_qkv = self.getattr_L__mod___transformers_2_blocks___0___attn_qkv(getattr_l__mod___transformers_2_blocks___0___norm1);  getattr_l__mod___transformers_2_blocks___0___norm1 = None
    reshape_20 = getattr_l__mod___transformers_2_blocks___0___attn_qkv.reshape(8, 65, 3, 16, 64);  getattr_l__mod___transformers_2_blocks___0___attn_qkv = None
    qkv_9 = reshape_20.permute(2, 0, 3, 1, 4);  reshape_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_9 = qkv_9.unbind(0);  qkv_9 = None
    q_18 = unbind_9[0]
    k_18 = unbind_9[1]
    v_9 = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_19 = self.getattr_L__mod___transformers_2_blocks___0___attn_q_norm(q_18);  q_18 = None
    k_19 = self.getattr_L__mod___transformers_2_blocks___0___attn_k_norm(k_18);  k_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_132 = torch._C._nn.scaled_dot_product_attention(q_19, k_19, v_9, dropout_p = 0.0);  q_19 = k_19 = v_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_14 = x_132.transpose(1, 2);  x_132 = None
    x_133 = transpose_14.reshape(8, 65, 1024);  transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_134 = self.getattr_L__mod___transformers_2_blocks___0___attn_proj(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_135 = self.getattr_L__mod___transformers_2_blocks___0___attn_proj_drop(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_2_blocks___0___ls1 = self.getattr_L__mod___transformers_2_blocks___0___ls1(x_135);  x_135 = None
    getattr_l__mod___transformers_2_blocks___0___drop_path1 = self.getattr_L__mod___transformers_2_blocks___0___drop_path1(getattr_l__mod___transformers_2_blocks___0___ls1);  getattr_l__mod___transformers_2_blocks___0___ls1 = None
    x_136 = x_131 + getattr_l__mod___transformers_2_blocks___0___drop_path1;  x_131 = getattr_l__mod___transformers_2_blocks___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_2_blocks___0___norm2 = self.getattr_L__mod___transformers_2_blocks___0___norm2(x_136)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_137 = self.getattr_L__mod___transformers_2_blocks___0___mlp_fc1(getattr_l__mod___transformers_2_blocks___0___norm2);  getattr_l__mod___transformers_2_blocks___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_138 = self.getattr_L__mod___transformers_2_blocks___0___mlp_act(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_139 = self.getattr_L__mod___transformers_2_blocks___0___mlp_drop1(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_140 = self.getattr_L__mod___transformers_2_blocks___0___mlp_norm(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_141 = self.getattr_L__mod___transformers_2_blocks___0___mlp_fc2(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_142 = self.getattr_L__mod___transformers_2_blocks___0___mlp_drop2(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_2_blocks___0___ls2 = self.getattr_L__mod___transformers_2_blocks___0___ls2(x_142);  x_142 = None
    getattr_l__mod___transformers_2_blocks___0___drop_path2 = self.getattr_L__mod___transformers_2_blocks___0___drop_path2(getattr_l__mod___transformers_2_blocks___0___ls2);  getattr_l__mod___transformers_2_blocks___0___ls2 = None
    x_143 = x_136 + getattr_l__mod___transformers_2_blocks___0___drop_path2;  x_136 = getattr_l__mod___transformers_2_blocks___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_2_blocks___1___norm1 = self.getattr_L__mod___transformers_2_blocks___1___norm1(x_143)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_2_blocks___1___attn_qkv = self.getattr_L__mod___transformers_2_blocks___1___attn_qkv(getattr_l__mod___transformers_2_blocks___1___norm1);  getattr_l__mod___transformers_2_blocks___1___norm1 = None
    reshape_22 = getattr_l__mod___transformers_2_blocks___1___attn_qkv.reshape(8, 65, 3, 16, 64);  getattr_l__mod___transformers_2_blocks___1___attn_qkv = None
    qkv_10 = reshape_22.permute(2, 0, 3, 1, 4);  reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_10 = qkv_10.unbind(0);  qkv_10 = None
    q_20 = unbind_10[0]
    k_20 = unbind_10[1]
    v_10 = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_21 = self.getattr_L__mod___transformers_2_blocks___1___attn_q_norm(q_20);  q_20 = None
    k_21 = self.getattr_L__mod___transformers_2_blocks___1___attn_k_norm(k_20);  k_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_144 = torch._C._nn.scaled_dot_product_attention(q_21, k_21, v_10, dropout_p = 0.0);  q_21 = k_21 = v_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_15 = x_144.transpose(1, 2);  x_144 = None
    x_145 = transpose_15.reshape(8, 65, 1024);  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_146 = self.getattr_L__mod___transformers_2_blocks___1___attn_proj(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_147 = self.getattr_L__mod___transformers_2_blocks___1___attn_proj_drop(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_2_blocks___1___ls1 = self.getattr_L__mod___transformers_2_blocks___1___ls1(x_147);  x_147 = None
    getattr_l__mod___transformers_2_blocks___1___drop_path1 = self.getattr_L__mod___transformers_2_blocks___1___drop_path1(getattr_l__mod___transformers_2_blocks___1___ls1);  getattr_l__mod___transformers_2_blocks___1___ls1 = None
    x_148 = x_143 + getattr_l__mod___transformers_2_blocks___1___drop_path1;  x_143 = getattr_l__mod___transformers_2_blocks___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_2_blocks___1___norm2 = self.getattr_L__mod___transformers_2_blocks___1___norm2(x_148)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_149 = self.getattr_L__mod___transformers_2_blocks___1___mlp_fc1(getattr_l__mod___transformers_2_blocks___1___norm2);  getattr_l__mod___transformers_2_blocks___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_150 = self.getattr_L__mod___transformers_2_blocks___1___mlp_act(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_151 = self.getattr_L__mod___transformers_2_blocks___1___mlp_drop1(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_152 = self.getattr_L__mod___transformers_2_blocks___1___mlp_norm(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_153 = self.getattr_L__mod___transformers_2_blocks___1___mlp_fc2(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_154 = self.getattr_L__mod___transformers_2_blocks___1___mlp_drop2(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_2_blocks___1___ls2 = self.getattr_L__mod___transformers_2_blocks___1___ls2(x_154);  x_154 = None
    getattr_l__mod___transformers_2_blocks___1___drop_path2 = self.getattr_L__mod___transformers_2_blocks___1___drop_path2(getattr_l__mod___transformers_2_blocks___1___ls2);  getattr_l__mod___transformers_2_blocks___1___ls2 = None
    x_155 = x_148 + getattr_l__mod___transformers_2_blocks___1___drop_path2;  x_148 = getattr_l__mod___transformers_2_blocks___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_2_blocks___2___norm1 = self.getattr_L__mod___transformers_2_blocks___2___norm1(x_155)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_2_blocks___2___attn_qkv = self.getattr_L__mod___transformers_2_blocks___2___attn_qkv(getattr_l__mod___transformers_2_blocks___2___norm1);  getattr_l__mod___transformers_2_blocks___2___norm1 = None
    reshape_24 = getattr_l__mod___transformers_2_blocks___2___attn_qkv.reshape(8, 65, 3, 16, 64);  getattr_l__mod___transformers_2_blocks___2___attn_qkv = None
    qkv_11 = reshape_24.permute(2, 0, 3, 1, 4);  reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_11 = qkv_11.unbind(0);  qkv_11 = None
    q_22 = unbind_11[0]
    k_22 = unbind_11[1]
    v_11 = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_23 = self.getattr_L__mod___transformers_2_blocks___2___attn_q_norm(q_22);  q_22 = None
    k_23 = self.getattr_L__mod___transformers_2_blocks___2___attn_k_norm(k_22);  k_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_156 = torch._C._nn.scaled_dot_product_attention(q_23, k_23, v_11, dropout_p = 0.0);  q_23 = k_23 = v_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_16 = x_156.transpose(1, 2);  x_156 = None
    x_157 = transpose_16.reshape(8, 65, 1024);  transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_158 = self.getattr_L__mod___transformers_2_blocks___2___attn_proj(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_159 = self.getattr_L__mod___transformers_2_blocks___2___attn_proj_drop(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_2_blocks___2___ls1 = self.getattr_L__mod___transformers_2_blocks___2___ls1(x_159);  x_159 = None
    getattr_l__mod___transformers_2_blocks___2___drop_path1 = self.getattr_L__mod___transformers_2_blocks___2___drop_path1(getattr_l__mod___transformers_2_blocks___2___ls1);  getattr_l__mod___transformers_2_blocks___2___ls1 = None
    x_160 = x_155 + getattr_l__mod___transformers_2_blocks___2___drop_path1;  x_155 = getattr_l__mod___transformers_2_blocks___2___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_2_blocks___2___norm2 = self.getattr_L__mod___transformers_2_blocks___2___norm2(x_160)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_161 = self.getattr_L__mod___transformers_2_blocks___2___mlp_fc1(getattr_l__mod___transformers_2_blocks___2___norm2);  getattr_l__mod___transformers_2_blocks___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_162 = self.getattr_L__mod___transformers_2_blocks___2___mlp_act(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_163 = self.getattr_L__mod___transformers_2_blocks___2___mlp_drop1(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_164 = self.getattr_L__mod___transformers_2_blocks___2___mlp_norm(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_165 = self.getattr_L__mod___transformers_2_blocks___2___mlp_fc2(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_166 = self.getattr_L__mod___transformers_2_blocks___2___mlp_drop2(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_2_blocks___2___ls2 = self.getattr_L__mod___transformers_2_blocks___2___ls2(x_166);  x_166 = None
    getattr_l__mod___transformers_2_blocks___2___drop_path2 = self.getattr_L__mod___transformers_2_blocks___2___drop_path2(getattr_l__mod___transformers_2_blocks___2___ls2);  getattr_l__mod___transformers_2_blocks___2___ls2 = None
    x_167 = x_160 + getattr_l__mod___transformers_2_blocks___2___drop_path2;  x_160 = getattr_l__mod___transformers_2_blocks___2___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_2_blocks___3___norm1 = self.getattr_L__mod___transformers_2_blocks___3___norm1(x_167)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___transformers_2_blocks___3___attn_qkv = self.getattr_L__mod___transformers_2_blocks___3___attn_qkv(getattr_l__mod___transformers_2_blocks___3___norm1);  getattr_l__mod___transformers_2_blocks___3___norm1 = None
    reshape_26 = getattr_l__mod___transformers_2_blocks___3___attn_qkv.reshape(8, 65, 3, 16, 64);  getattr_l__mod___transformers_2_blocks___3___attn_qkv = None
    qkv_12 = reshape_26.permute(2, 0, 3, 1, 4);  reshape_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_12 = qkv_12.unbind(0);  qkv_12 = None
    q_24 = unbind_12[0]
    k_24 = unbind_12[1]
    v_12 = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_25 = self.getattr_L__mod___transformers_2_blocks___3___attn_q_norm(q_24);  q_24 = None
    k_25 = self.getattr_L__mod___transformers_2_blocks___3___attn_k_norm(k_24);  k_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_168 = torch._C._nn.scaled_dot_product_attention(q_25, k_25, v_12, dropout_p = 0.0);  q_25 = k_25 = v_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_17 = x_168.transpose(1, 2);  x_168 = None
    x_169 = transpose_17.reshape(8, 65, 1024);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_170 = self.getattr_L__mod___transformers_2_blocks___3___attn_proj(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_171 = self.getattr_L__mod___transformers_2_blocks___3___attn_proj_drop(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___transformers_2_blocks___3___ls1 = self.getattr_L__mod___transformers_2_blocks___3___ls1(x_171);  x_171 = None
    getattr_l__mod___transformers_2_blocks___3___drop_path1 = self.getattr_L__mod___transformers_2_blocks___3___drop_path1(getattr_l__mod___transformers_2_blocks___3___ls1);  getattr_l__mod___transformers_2_blocks___3___ls1 = None
    x_172 = x_167 + getattr_l__mod___transformers_2_blocks___3___drop_path1;  x_167 = getattr_l__mod___transformers_2_blocks___3___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_2_blocks___3___norm2 = self.getattr_L__mod___transformers_2_blocks___3___norm2(x_172)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_173 = self.getattr_L__mod___transformers_2_blocks___3___mlp_fc1(getattr_l__mod___transformers_2_blocks___3___norm2);  getattr_l__mod___transformers_2_blocks___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_174 = self.getattr_L__mod___transformers_2_blocks___3___mlp_act(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_175 = self.getattr_L__mod___transformers_2_blocks___3___mlp_drop1(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_176 = self.getattr_L__mod___transformers_2_blocks___3___mlp_norm(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_177 = self.getattr_L__mod___transformers_2_blocks___3___mlp_fc2(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_178 = self.getattr_L__mod___transformers_2_blocks___3___mlp_drop2(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___transformers_2_blocks___3___ls2 = self.getattr_L__mod___transformers_2_blocks___3___ls2(x_178);  x_178 = None
    getattr_l__mod___transformers_2_blocks___3___drop_path2 = self.getattr_L__mod___transformers_2_blocks___3___drop_path2(getattr_l__mod___transformers_2_blocks___3___ls2);  getattr_l__mod___transformers_2_blocks___3___ls2 = None
    x_180 = x_172 + getattr_l__mod___transformers_2_blocks___3___drop_path2;  x_172 = getattr_l__mod___transformers_2_blocks___3___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:87, code: cls_tokens = x[:, :token_length]
    cls_tokens_8 = x_180[(slice(None, None, None), slice(None, 1, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:88, code: x = x[:, token_length:]
    x_181 = x_180[(slice(None, None, None), slice(1, None, None))];  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:89, code: x = x.transpose(1, 2).reshape(B, C, H, W)
    transpose_18 = x_181.transpose(1, 2);  x_181 = None
    x_183 = transpose_18.reshape(8, 1024, 8, 8);  transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:260, code: cls_tokens = self.norm(cls_tokens)
    x_184 = self.L__mod___norm(cls_tokens_8);  cls_tokens_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:280, code: x = x[:, 0]
    x_185 = x_184[(slice(None, None, None), 0)];  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:281, code: x = self.head_drop(x)
    x_186 = self.L__mod___head_drop(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/pit.py:283, code: x = self.head(x)
    pred = self.L__mod___head(x_186);  x_186 = None
    return (pred,)
    