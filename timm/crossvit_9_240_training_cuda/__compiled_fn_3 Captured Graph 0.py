from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    x_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:84, code: x = self.proj(x).flatten(2).transpose(1, 2)
    l__mod___patch_embed_0_proj = self.L__mod___patch_embed_0_proj(x_)
    flatten = l__mod___patch_embed_0_proj.flatten(2);  l__mod___patch_embed_0_proj = None
    x__1 = flatten.transpose(1, 2);  flatten = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:439, code: cls_tokens = self.cls_token_0 if i == 0 else self.cls_token_1  # hard-coded for torch jit script
    cls_tokens = self.L__mod___cls_token_0
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:440, code: cls_tokens = cls_tokens.expand(B, -1, -1)
    cls_tokens_1 = cls_tokens.expand(8, -1, -1);  cls_tokens = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:441, code: x_ = torch.cat((cls_tokens, x_), dim=1)
    x__2 = torch.cat((cls_tokens_1, x__1), dim = 1);  cls_tokens_1 = x__1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:442, code: pos_embed = self.pos_embed_0 if i == 0 else self.pos_embed_1  # hard-coded for torch jit script
    pos_embed = self.L__mod___pos_embed_0
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:443, code: x_ = x_ + pos_embed
    x__3 = x__2 + pos_embed;  x__2 = pos_embed = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:444, code: x_ = self.pos_drop(x_)
    x__4 = self.L__mod___pos_drop(x__3);  x__3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:292, code: x = torch.nn.functional.interpolate(x, size=ss, mode='bicubic', align_corners=False)
    x__5 = torch.nn.functional.interpolate(x_, size = (224, 224), mode = 'bicubic', align_corners = False);  x_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:84, code: x = self.proj(x).flatten(2).transpose(1, 2)
    l__mod___patch_embed_1_proj = self.L__mod___patch_embed_1_proj(x__5);  x__5 = None
    flatten_1 = l__mod___patch_embed_1_proj.flatten(2);  l__mod___patch_embed_1_proj = None
    x__6 = flatten_1.transpose(1, 2);  flatten_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:439, code: cls_tokens = self.cls_token_0 if i == 0 else self.cls_token_1  # hard-coded for torch jit script
    cls_tokens_2 = self.L__mod___cls_token_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:440, code: cls_tokens = cls_tokens.expand(B, -1, -1)
    cls_tokens_3 = cls_tokens_2.expand(8, -1, -1);  cls_tokens_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:441, code: x_ = torch.cat((cls_tokens, x_), dim=1)
    x__7 = torch.cat((cls_tokens_3, x__6), dim = 1);  cls_tokens_3 = x__6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:442, code: pos_embed = self.pos_embed_0 if i == 0 else self.pos_embed_1  # hard-coded for torch jit script
    pos_embed_1 = self.L__mod___pos_embed_1
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:443, code: x_ = x_ + pos_embed
    x__8 = x__7 + pos_embed_1;  x__7 = pos_embed_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:444, code: x_ = self.pos_drop(x_)
    x__9 = self.L__mod___pos_drop(x__8);  x__8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_0_blocks_0___0___norm1 = self.getattr_L__mod___blocks_0_blocks_0___0___norm1(x__4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks_0_blocks_0___0___attn_qkv = self.getattr_L__mod___blocks_0_blocks_0___0___attn_qkv(getattr_l__mod___blocks_0_blocks_0___0___norm1);  getattr_l__mod___blocks_0_blocks_0___0___norm1 = None
    reshape = getattr_l__mod___blocks_0_blocks_0___0___attn_qkv.reshape(8, 401, 3, 4, 32);  getattr_l__mod___blocks_0_blocks_0___0___attn_qkv = None
    qkv = reshape.permute(2, 0, 3, 1, 4);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind = qkv.unbind(0);  qkv = None
    q = unbind[0]
    k = unbind[1]
    v = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_1 = self.getattr_L__mod___blocks_0_blocks_0___0___attn_q_norm(q);  q = None
    k_1 = self.getattr_L__mod___blocks_0_blocks_0___0___attn_k_norm(k);  k = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_3 = torch._C._nn.scaled_dot_product_attention(q_1, k_1, v, dropout_p = 0.0);  q_1 = k_1 = v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_2 = x_3.transpose(1, 2);  x_3 = None
    x_4 = transpose_2.reshape(8, 401, 128);  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_5 = self.getattr_L__mod___blocks_0_blocks_0___0___attn_proj(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_6 = self.getattr_L__mod___blocks_0_blocks_0___0___attn_proj_drop(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_0_blocks_0___0___ls1 = self.getattr_L__mod___blocks_0_blocks_0___0___ls1(x_6);  x_6 = None
    getattr_l__mod___blocks_0_blocks_0___0___drop_path1 = self.getattr_L__mod___blocks_0_blocks_0___0___drop_path1(getattr_l__mod___blocks_0_blocks_0___0___ls1);  getattr_l__mod___blocks_0_blocks_0___0___ls1 = None
    x_7 = x__4 + getattr_l__mod___blocks_0_blocks_0___0___drop_path1;  x__4 = getattr_l__mod___blocks_0_blocks_0___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_0_blocks_0___0___norm2 = self.getattr_L__mod___blocks_0_blocks_0___0___norm2(x_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_8 = self.getattr_L__mod___blocks_0_blocks_0___0___mlp_fc1(getattr_l__mod___blocks_0_blocks_0___0___norm2);  getattr_l__mod___blocks_0_blocks_0___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_9 = self.getattr_L__mod___blocks_0_blocks_0___0___mlp_act(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_10 = self.getattr_L__mod___blocks_0_blocks_0___0___mlp_drop1(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_11 = self.getattr_L__mod___blocks_0_blocks_0___0___mlp_norm(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_12 = self.getattr_L__mod___blocks_0_blocks_0___0___mlp_fc2(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_13 = self.getattr_L__mod___blocks_0_blocks_0___0___mlp_drop2(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_0_blocks_0___0___ls2 = self.getattr_L__mod___blocks_0_blocks_0___0___ls2(x_13);  x_13 = None
    getattr_l__mod___blocks_0_blocks_0___0___drop_path2 = self.getattr_L__mod___blocks_0_blocks_0___0___drop_path2(getattr_l__mod___blocks_0_blocks_0___0___ls2);  getattr_l__mod___blocks_0_blocks_0___0___ls2 = None
    x_14 = x_7 + getattr_l__mod___blocks_0_blocks_0___0___drop_path2;  x_7 = getattr_l__mod___blocks_0_blocks_0___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_0_blocks_1___0___norm1 = self.getattr_L__mod___blocks_0_blocks_1___0___norm1(x__9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks_0_blocks_1___0___attn_qkv = self.getattr_L__mod___blocks_0_blocks_1___0___attn_qkv(getattr_l__mod___blocks_0_blocks_1___0___norm1);  getattr_l__mod___blocks_0_blocks_1___0___norm1 = None
    reshape_2 = getattr_l__mod___blocks_0_blocks_1___0___attn_qkv.reshape(8, 197, 3, 4, 64);  getattr_l__mod___blocks_0_blocks_1___0___attn_qkv = None
    qkv_1 = reshape_2.permute(2, 0, 3, 1, 4);  reshape_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_1 = qkv_1.unbind(0);  qkv_1 = None
    q_2 = unbind_1[0]
    k_2 = unbind_1[1]
    v_1 = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_3 = self.getattr_L__mod___blocks_0_blocks_1___0___attn_q_norm(q_2);  q_2 = None
    k_3 = self.getattr_L__mod___blocks_0_blocks_1___0___attn_k_norm(k_2);  k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_15 = torch._C._nn.scaled_dot_product_attention(q_3, k_3, v_1, dropout_p = 0.0);  q_3 = k_3 = v_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_3 = x_15.transpose(1, 2);  x_15 = None
    x_16 = transpose_3.reshape(8, 197, 256);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_17 = self.getattr_L__mod___blocks_0_blocks_1___0___attn_proj(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_18 = self.getattr_L__mod___blocks_0_blocks_1___0___attn_proj_drop(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_0_blocks_1___0___ls1 = self.getattr_L__mod___blocks_0_blocks_1___0___ls1(x_18);  x_18 = None
    getattr_l__mod___blocks_0_blocks_1___0___drop_path1 = self.getattr_L__mod___blocks_0_blocks_1___0___drop_path1(getattr_l__mod___blocks_0_blocks_1___0___ls1);  getattr_l__mod___blocks_0_blocks_1___0___ls1 = None
    x_19 = x__9 + getattr_l__mod___blocks_0_blocks_1___0___drop_path1;  x__9 = getattr_l__mod___blocks_0_blocks_1___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_0_blocks_1___0___norm2 = self.getattr_L__mod___blocks_0_blocks_1___0___norm2(x_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_20 = self.getattr_L__mod___blocks_0_blocks_1___0___mlp_fc1(getattr_l__mod___blocks_0_blocks_1___0___norm2);  getattr_l__mod___blocks_0_blocks_1___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_21 = self.getattr_L__mod___blocks_0_blocks_1___0___mlp_act(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_22 = self.getattr_L__mod___blocks_0_blocks_1___0___mlp_drop1(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_23 = self.getattr_L__mod___blocks_0_blocks_1___0___mlp_norm(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_24 = self.getattr_L__mod___blocks_0_blocks_1___0___mlp_fc2(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_25 = self.getattr_L__mod___blocks_0_blocks_1___0___mlp_drop2(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_0_blocks_1___0___ls2 = self.getattr_L__mod___blocks_0_blocks_1___0___ls2(x_25);  x_25 = None
    getattr_l__mod___blocks_0_blocks_1___0___drop_path2 = self.getattr_L__mod___blocks_0_blocks_1___0___drop_path2(getattr_l__mod___blocks_0_blocks_1___0___ls2);  getattr_l__mod___blocks_0_blocks_1___0___ls2 = None
    x_26 = x_19 + getattr_l__mod___blocks_0_blocks_1___0___drop_path2;  x_19 = getattr_l__mod___blocks_0_blocks_1___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_0_blocks_1___1___norm1 = self.getattr_L__mod___blocks_0_blocks_1___1___norm1(x_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks_0_blocks_1___1___attn_qkv = self.getattr_L__mod___blocks_0_blocks_1___1___attn_qkv(getattr_l__mod___blocks_0_blocks_1___1___norm1);  getattr_l__mod___blocks_0_blocks_1___1___norm1 = None
    reshape_4 = getattr_l__mod___blocks_0_blocks_1___1___attn_qkv.reshape(8, 197, 3, 4, 64);  getattr_l__mod___blocks_0_blocks_1___1___attn_qkv = None
    qkv_2 = reshape_4.permute(2, 0, 3, 1, 4);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_2 = qkv_2.unbind(0);  qkv_2 = None
    q_4 = unbind_2[0]
    k_4 = unbind_2[1]
    v_2 = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_5 = self.getattr_L__mod___blocks_0_blocks_1___1___attn_q_norm(q_4);  q_4 = None
    k_5 = self.getattr_L__mod___blocks_0_blocks_1___1___attn_k_norm(k_4);  k_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_27 = torch._C._nn.scaled_dot_product_attention(q_5, k_5, v_2, dropout_p = 0.0);  q_5 = k_5 = v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_4 = x_27.transpose(1, 2);  x_27 = None
    x_28 = transpose_4.reshape(8, 197, 256);  transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_29 = self.getattr_L__mod___blocks_0_blocks_1___1___attn_proj(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_30 = self.getattr_L__mod___blocks_0_blocks_1___1___attn_proj_drop(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_0_blocks_1___1___ls1 = self.getattr_L__mod___blocks_0_blocks_1___1___ls1(x_30);  x_30 = None
    getattr_l__mod___blocks_0_blocks_1___1___drop_path1 = self.getattr_L__mod___blocks_0_blocks_1___1___drop_path1(getattr_l__mod___blocks_0_blocks_1___1___ls1);  getattr_l__mod___blocks_0_blocks_1___1___ls1 = None
    x_31 = x_26 + getattr_l__mod___blocks_0_blocks_1___1___drop_path1;  x_26 = getattr_l__mod___blocks_0_blocks_1___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_0_blocks_1___1___norm2 = self.getattr_L__mod___blocks_0_blocks_1___1___norm2(x_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_32 = self.getattr_L__mod___blocks_0_blocks_1___1___mlp_fc1(getattr_l__mod___blocks_0_blocks_1___1___norm2);  getattr_l__mod___blocks_0_blocks_1___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_33 = self.getattr_L__mod___blocks_0_blocks_1___1___mlp_act(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_34 = self.getattr_L__mod___blocks_0_blocks_1___1___mlp_drop1(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_35 = self.getattr_L__mod___blocks_0_blocks_1___1___mlp_norm(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_36 = self.getattr_L__mod___blocks_0_blocks_1___1___mlp_fc2(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_37 = self.getattr_L__mod___blocks_0_blocks_1___1___mlp_drop2(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_0_blocks_1___1___ls2 = self.getattr_L__mod___blocks_0_blocks_1___1___ls2(x_37);  x_37 = None
    getattr_l__mod___blocks_0_blocks_1___1___drop_path2 = self.getattr_L__mod___blocks_0_blocks_1___1___drop_path2(getattr_l__mod___blocks_0_blocks_1___1___ls2);  getattr_l__mod___blocks_0_blocks_1___1___ls2 = None
    x_38 = x_31 + getattr_l__mod___blocks_0_blocks_1___1___drop_path2;  x_31 = getattr_l__mod___blocks_0_blocks_1___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_0_blocks_1___2___norm1 = self.getattr_L__mod___blocks_0_blocks_1___2___norm1(x_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks_0_blocks_1___2___attn_qkv = self.getattr_L__mod___blocks_0_blocks_1___2___attn_qkv(getattr_l__mod___blocks_0_blocks_1___2___norm1);  getattr_l__mod___blocks_0_blocks_1___2___norm1 = None
    reshape_6 = getattr_l__mod___blocks_0_blocks_1___2___attn_qkv.reshape(8, 197, 3, 4, 64);  getattr_l__mod___blocks_0_blocks_1___2___attn_qkv = None
    qkv_3 = reshape_6.permute(2, 0, 3, 1, 4);  reshape_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_3 = qkv_3.unbind(0);  qkv_3 = None
    q_6 = unbind_3[0]
    k_6 = unbind_3[1]
    v_3 = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_7 = self.getattr_L__mod___blocks_0_blocks_1___2___attn_q_norm(q_6);  q_6 = None
    k_7 = self.getattr_L__mod___blocks_0_blocks_1___2___attn_k_norm(k_6);  k_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_39 = torch._C._nn.scaled_dot_product_attention(q_7, k_7, v_3, dropout_p = 0.0);  q_7 = k_7 = v_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_5 = x_39.transpose(1, 2);  x_39 = None
    x_40 = transpose_5.reshape(8, 197, 256);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_41 = self.getattr_L__mod___blocks_0_blocks_1___2___attn_proj(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_42 = self.getattr_L__mod___blocks_0_blocks_1___2___attn_proj_drop(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_0_blocks_1___2___ls1 = self.getattr_L__mod___blocks_0_blocks_1___2___ls1(x_42);  x_42 = None
    getattr_l__mod___blocks_0_blocks_1___2___drop_path1 = self.getattr_L__mod___blocks_0_blocks_1___2___drop_path1(getattr_l__mod___blocks_0_blocks_1___2___ls1);  getattr_l__mod___blocks_0_blocks_1___2___ls1 = None
    x_43 = x_38 + getattr_l__mod___blocks_0_blocks_1___2___drop_path1;  x_38 = getattr_l__mod___blocks_0_blocks_1___2___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_0_blocks_1___2___norm2 = self.getattr_L__mod___blocks_0_blocks_1___2___norm2(x_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_44 = self.getattr_L__mod___blocks_0_blocks_1___2___mlp_fc1(getattr_l__mod___blocks_0_blocks_1___2___norm2);  getattr_l__mod___blocks_0_blocks_1___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_45 = self.getattr_L__mod___blocks_0_blocks_1___2___mlp_act(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_46 = self.getattr_L__mod___blocks_0_blocks_1___2___mlp_drop1(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_47 = self.getattr_L__mod___blocks_0_blocks_1___2___mlp_norm(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_48 = self.getattr_L__mod___blocks_0_blocks_1___2___mlp_fc2(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_49 = self.getattr_L__mod___blocks_0_blocks_1___2___mlp_drop2(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_0_blocks_1___2___ls2 = self.getattr_L__mod___blocks_0_blocks_1___2___ls2(x_49);  x_49 = None
    getattr_l__mod___blocks_0_blocks_1___2___drop_path2 = self.getattr_L__mod___blocks_0_blocks_1___2___drop_path2(getattr_l__mod___blocks_0_blocks_1___2___ls2);  getattr_l__mod___blocks_0_blocks_1___2___ls2 = None
    x_50 = x_43 + getattr_l__mod___blocks_0_blocks_1___2___drop_path2;  x_43 = getattr_l__mod___blocks_0_blocks_1___2___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    getitem_12 = x_14[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_0_projs_0_0 = self.L__mod___blocks_0_projs_0_0(getitem_12);  getitem_12 = None
    l__mod___blocks_0_projs_0_1 = self.L__mod___blocks_0_projs_0_1(l__mod___blocks_0_projs_0_0);  l__mod___blocks_0_projs_0_0 = None
    l__mod___blocks_0_projs_0_2 = self.L__mod___blocks_0_projs_0_2(l__mod___blocks_0_projs_0_1);  l__mod___blocks_0_projs_0_1 = None
    getitem_13 = x_50[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_0_projs_1_0 = self.L__mod___blocks_0_projs_1_0(getitem_13);  getitem_13 = None
    l__mod___blocks_0_projs_1_1 = self.L__mod___blocks_0_projs_1_1(l__mod___blocks_0_projs_1_0);  l__mod___blocks_0_projs_1_0 = None
    l__mod___blocks_0_projs_1_2 = self.L__mod___blocks_0_projs_1_2(l__mod___blocks_0_projs_1_1);  l__mod___blocks_0_projs_1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    getitem_14 = x_50[(slice(None, None, None), slice(1, None, None), Ellipsis)]
    tmp = torch.cat((l__mod___blocks_0_projs_0_2, getitem_14), dim = 1);  l__mod___blocks_0_projs_0_2 = getitem_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    getitem_15 = tmp[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_0_fusion_0_norm1 = self.L__mod___blocks_0_fusion_0_norm1(tmp);  tmp = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    getitem_16 = l__mod___blocks_0_fusion_0_norm1[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_0_fusion_0_attn_wq = self.L__mod___blocks_0_fusion_0_attn_wq(getitem_16);  getitem_16 = None
    reshape_8 = l__mod___blocks_0_fusion_0_attn_wq.reshape(8, 1, 4, 64);  l__mod___blocks_0_fusion_0_attn_wq = None
    q_8 = reshape_8.permute(0, 2, 1, 3);  reshape_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_0_fusion_0_attn_wk = self.L__mod___blocks_0_fusion_0_attn_wk(l__mod___blocks_0_fusion_0_norm1)
    reshape_9 = l__mod___blocks_0_fusion_0_attn_wk.reshape(8, 197, 4, 64);  l__mod___blocks_0_fusion_0_attn_wk = None
    k_8 = reshape_9.permute(0, 2, 1, 3);  reshape_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_0_fusion_0_attn_wv = self.L__mod___blocks_0_fusion_0_attn_wv(l__mod___blocks_0_fusion_0_norm1);  l__mod___blocks_0_fusion_0_norm1 = None
    reshape_10 = l__mod___blocks_0_fusion_0_attn_wv.reshape(8, 197, 4, 64);  l__mod___blocks_0_fusion_0_attn_wv = None
    v_4 = reshape_10.permute(0, 2, 1, 3);  reshape_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    transpose_6 = k_8.transpose(-2, -1);  k_8 = None
    matmul = q_8 @ transpose_6;  q_8 = transpose_6 = None
    attn = matmul * 0.125;  matmul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    attn_1 = attn.softmax(dim = -1);  attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    attn_2 = self.L__mod___blocks_0_fusion_0_attn_attn_drop(attn_1);  attn_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    matmul_1 = attn_2 @ v_4;  attn_2 = v_4 = None
    transpose_7 = matmul_1.transpose(1, 2);  matmul_1 = None
    x_51 = transpose_7.reshape(8, 1, 256);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    x_52 = self.L__mod___blocks_0_fusion_0_attn_proj(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    x_53 = self.L__mod___blocks_0_fusion_0_attn_proj_drop(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_0_fusion_0_drop_path = self.L__mod___blocks_0_fusion_0_drop_path(x_53);  x_53 = None
    tmp_1 = getitem_15 + l__mod___blocks_0_fusion_0_drop_path;  getitem_15 = l__mod___blocks_0_fusion_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    getitem_17 = tmp_1[(slice(None, None, None), slice(0, 1, None), Ellipsis)];  tmp_1 = None
    l__mod___blocks_0_revert_projs_0_0 = self.L__mod___blocks_0_revert_projs_0_0(getitem_17);  getitem_17 = None
    l__mod___blocks_0_revert_projs_0_1 = self.L__mod___blocks_0_revert_projs_0_1(l__mod___blocks_0_revert_projs_0_0);  l__mod___blocks_0_revert_projs_0_0 = None
    reverted_proj_cls_token = self.L__mod___blocks_0_revert_projs_0_2(l__mod___blocks_0_revert_projs_0_1);  l__mod___blocks_0_revert_projs_0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    getitem_18 = x_14[(slice(None, None, None), slice(1, None, None), Ellipsis)]
    tmp_2 = torch.cat((reverted_proj_cls_token, getitem_18), dim = 1);  reverted_proj_cls_token = getitem_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    getitem_19 = x_14[(slice(None, None, None), slice(1, None, None), Ellipsis)];  x_14 = None
    tmp_3 = torch.cat((l__mod___blocks_0_projs_1_2, getitem_19), dim = 1);  l__mod___blocks_0_projs_1_2 = getitem_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    getitem_20 = tmp_3[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_0_fusion_1_norm1 = self.L__mod___blocks_0_fusion_1_norm1(tmp_3);  tmp_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    getitem_21 = l__mod___blocks_0_fusion_1_norm1[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_0_fusion_1_attn_wq = self.L__mod___blocks_0_fusion_1_attn_wq(getitem_21);  getitem_21 = None
    reshape_12 = l__mod___blocks_0_fusion_1_attn_wq.reshape(8, 1, 4, 32);  l__mod___blocks_0_fusion_1_attn_wq = None
    q_9 = reshape_12.permute(0, 2, 1, 3);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_0_fusion_1_attn_wk = self.L__mod___blocks_0_fusion_1_attn_wk(l__mod___blocks_0_fusion_1_norm1)
    reshape_13 = l__mod___blocks_0_fusion_1_attn_wk.reshape(8, 401, 4, 32);  l__mod___blocks_0_fusion_1_attn_wk = None
    k_9 = reshape_13.permute(0, 2, 1, 3);  reshape_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_0_fusion_1_attn_wv = self.L__mod___blocks_0_fusion_1_attn_wv(l__mod___blocks_0_fusion_1_norm1);  l__mod___blocks_0_fusion_1_norm1 = None
    reshape_14 = l__mod___blocks_0_fusion_1_attn_wv.reshape(8, 401, 4, 32);  l__mod___blocks_0_fusion_1_attn_wv = None
    v_5 = reshape_14.permute(0, 2, 1, 3);  reshape_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    transpose_8 = k_9.transpose(-2, -1);  k_9 = None
    matmul_2 = q_9 @ transpose_8;  q_9 = transpose_8 = None
    attn_3 = matmul_2 * 0.1767766952966369;  matmul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    attn_4 = attn_3.softmax(dim = -1);  attn_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    attn_5 = self.L__mod___blocks_0_fusion_1_attn_attn_drop(attn_4);  attn_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    matmul_3 = attn_5 @ v_5;  attn_5 = v_5 = None
    transpose_9 = matmul_3.transpose(1, 2);  matmul_3 = None
    x_55 = transpose_9.reshape(8, 1, 128);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    x_56 = self.L__mod___blocks_0_fusion_1_attn_proj(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    x_57 = self.L__mod___blocks_0_fusion_1_attn_proj_drop(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_0_fusion_1_drop_path = self.L__mod___blocks_0_fusion_1_drop_path(x_57);  x_57 = None
    tmp_4 = getitem_20 + l__mod___blocks_0_fusion_1_drop_path;  getitem_20 = l__mod___blocks_0_fusion_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    getitem_22 = tmp_4[(slice(None, None, None), slice(0, 1, None), Ellipsis)];  tmp_4 = None
    l__mod___blocks_0_revert_projs_1_0 = self.L__mod___blocks_0_revert_projs_1_0(getitem_22);  getitem_22 = None
    l__mod___blocks_0_revert_projs_1_1 = self.L__mod___blocks_0_revert_projs_1_1(l__mod___blocks_0_revert_projs_1_0);  l__mod___blocks_0_revert_projs_1_0 = None
    reverted_proj_cls_token_1 = self.L__mod___blocks_0_revert_projs_1_2(l__mod___blocks_0_revert_projs_1_1);  l__mod___blocks_0_revert_projs_1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    getitem_23 = x_50[(slice(None, None, None), slice(1, None, None), Ellipsis)];  x_50 = None
    tmp_5 = torch.cat((reverted_proj_cls_token_1, getitem_23), dim = 1);  reverted_proj_cls_token_1 = getitem_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_1_blocks_0___0___norm1 = self.getattr_L__mod___blocks_1_blocks_0___0___norm1(tmp_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks_1_blocks_0___0___attn_qkv = self.getattr_L__mod___blocks_1_blocks_0___0___attn_qkv(getattr_l__mod___blocks_1_blocks_0___0___norm1);  getattr_l__mod___blocks_1_blocks_0___0___norm1 = None
    reshape_16 = getattr_l__mod___blocks_1_blocks_0___0___attn_qkv.reshape(8, 401, 3, 4, 32);  getattr_l__mod___blocks_1_blocks_0___0___attn_qkv = None
    qkv_4 = reshape_16.permute(2, 0, 3, 1, 4);  reshape_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_4 = qkv_4.unbind(0);  qkv_4 = None
    q_10 = unbind_4[0]
    k_10 = unbind_4[1]
    v_6 = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_11 = self.getattr_L__mod___blocks_1_blocks_0___0___attn_q_norm(q_10);  q_10 = None
    k_11 = self.getattr_L__mod___blocks_1_blocks_0___0___attn_k_norm(k_10);  k_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_59 = torch._C._nn.scaled_dot_product_attention(q_11, k_11, v_6, dropout_p = 0.0);  q_11 = k_11 = v_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_10 = x_59.transpose(1, 2);  x_59 = None
    x_60 = transpose_10.reshape(8, 401, 128);  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_61 = self.getattr_L__mod___blocks_1_blocks_0___0___attn_proj(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_62 = self.getattr_L__mod___blocks_1_blocks_0___0___attn_proj_drop(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_1_blocks_0___0___ls1 = self.getattr_L__mod___blocks_1_blocks_0___0___ls1(x_62);  x_62 = None
    getattr_l__mod___blocks_1_blocks_0___0___drop_path1 = self.getattr_L__mod___blocks_1_blocks_0___0___drop_path1(getattr_l__mod___blocks_1_blocks_0___0___ls1);  getattr_l__mod___blocks_1_blocks_0___0___ls1 = None
    x_63 = tmp_2 + getattr_l__mod___blocks_1_blocks_0___0___drop_path1;  tmp_2 = getattr_l__mod___blocks_1_blocks_0___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_1_blocks_0___0___norm2 = self.getattr_L__mod___blocks_1_blocks_0___0___norm2(x_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_64 = self.getattr_L__mod___blocks_1_blocks_0___0___mlp_fc1(getattr_l__mod___blocks_1_blocks_0___0___norm2);  getattr_l__mod___blocks_1_blocks_0___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_65 = self.getattr_L__mod___blocks_1_blocks_0___0___mlp_act(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_66 = self.getattr_L__mod___blocks_1_blocks_0___0___mlp_drop1(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_67 = self.getattr_L__mod___blocks_1_blocks_0___0___mlp_norm(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_68 = self.getattr_L__mod___blocks_1_blocks_0___0___mlp_fc2(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_69 = self.getattr_L__mod___blocks_1_blocks_0___0___mlp_drop2(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_1_blocks_0___0___ls2 = self.getattr_L__mod___blocks_1_blocks_0___0___ls2(x_69);  x_69 = None
    getattr_l__mod___blocks_1_blocks_0___0___drop_path2 = self.getattr_L__mod___blocks_1_blocks_0___0___drop_path2(getattr_l__mod___blocks_1_blocks_0___0___ls2);  getattr_l__mod___blocks_1_blocks_0___0___ls2 = None
    x_70 = x_63 + getattr_l__mod___blocks_1_blocks_0___0___drop_path2;  x_63 = getattr_l__mod___blocks_1_blocks_0___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_1_blocks_1___0___norm1 = self.getattr_L__mod___blocks_1_blocks_1___0___norm1(tmp_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks_1_blocks_1___0___attn_qkv = self.getattr_L__mod___blocks_1_blocks_1___0___attn_qkv(getattr_l__mod___blocks_1_blocks_1___0___norm1);  getattr_l__mod___blocks_1_blocks_1___0___norm1 = None
    reshape_18 = getattr_l__mod___blocks_1_blocks_1___0___attn_qkv.reshape(8, 197, 3, 4, 64);  getattr_l__mod___blocks_1_blocks_1___0___attn_qkv = None
    qkv_5 = reshape_18.permute(2, 0, 3, 1, 4);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_5 = qkv_5.unbind(0);  qkv_5 = None
    q_12 = unbind_5[0]
    k_12 = unbind_5[1]
    v_7 = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_13 = self.getattr_L__mod___blocks_1_blocks_1___0___attn_q_norm(q_12);  q_12 = None
    k_13 = self.getattr_L__mod___blocks_1_blocks_1___0___attn_k_norm(k_12);  k_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_71 = torch._C._nn.scaled_dot_product_attention(q_13, k_13, v_7, dropout_p = 0.0);  q_13 = k_13 = v_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_11 = x_71.transpose(1, 2);  x_71 = None
    x_72 = transpose_11.reshape(8, 197, 256);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_73 = self.getattr_L__mod___blocks_1_blocks_1___0___attn_proj(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_74 = self.getattr_L__mod___blocks_1_blocks_1___0___attn_proj_drop(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_1_blocks_1___0___ls1 = self.getattr_L__mod___blocks_1_blocks_1___0___ls1(x_74);  x_74 = None
    getattr_l__mod___blocks_1_blocks_1___0___drop_path1 = self.getattr_L__mod___blocks_1_blocks_1___0___drop_path1(getattr_l__mod___blocks_1_blocks_1___0___ls1);  getattr_l__mod___blocks_1_blocks_1___0___ls1 = None
    x_75 = tmp_5 + getattr_l__mod___blocks_1_blocks_1___0___drop_path1;  tmp_5 = getattr_l__mod___blocks_1_blocks_1___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_1_blocks_1___0___norm2 = self.getattr_L__mod___blocks_1_blocks_1___0___norm2(x_75)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_76 = self.getattr_L__mod___blocks_1_blocks_1___0___mlp_fc1(getattr_l__mod___blocks_1_blocks_1___0___norm2);  getattr_l__mod___blocks_1_blocks_1___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_77 = self.getattr_L__mod___blocks_1_blocks_1___0___mlp_act(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_78 = self.getattr_L__mod___blocks_1_blocks_1___0___mlp_drop1(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_79 = self.getattr_L__mod___blocks_1_blocks_1___0___mlp_norm(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_80 = self.getattr_L__mod___blocks_1_blocks_1___0___mlp_fc2(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_81 = self.getattr_L__mod___blocks_1_blocks_1___0___mlp_drop2(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_1_blocks_1___0___ls2 = self.getattr_L__mod___blocks_1_blocks_1___0___ls2(x_81);  x_81 = None
    getattr_l__mod___blocks_1_blocks_1___0___drop_path2 = self.getattr_L__mod___blocks_1_blocks_1___0___drop_path2(getattr_l__mod___blocks_1_blocks_1___0___ls2);  getattr_l__mod___blocks_1_blocks_1___0___ls2 = None
    x_82 = x_75 + getattr_l__mod___blocks_1_blocks_1___0___drop_path2;  x_75 = getattr_l__mod___blocks_1_blocks_1___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_1_blocks_1___1___norm1 = self.getattr_L__mod___blocks_1_blocks_1___1___norm1(x_82)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks_1_blocks_1___1___attn_qkv = self.getattr_L__mod___blocks_1_blocks_1___1___attn_qkv(getattr_l__mod___blocks_1_blocks_1___1___norm1);  getattr_l__mod___blocks_1_blocks_1___1___norm1 = None
    reshape_20 = getattr_l__mod___blocks_1_blocks_1___1___attn_qkv.reshape(8, 197, 3, 4, 64);  getattr_l__mod___blocks_1_blocks_1___1___attn_qkv = None
    qkv_6 = reshape_20.permute(2, 0, 3, 1, 4);  reshape_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_6 = qkv_6.unbind(0);  qkv_6 = None
    q_14 = unbind_6[0]
    k_14 = unbind_6[1]
    v_8 = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_15 = self.getattr_L__mod___blocks_1_blocks_1___1___attn_q_norm(q_14);  q_14 = None
    k_15 = self.getattr_L__mod___blocks_1_blocks_1___1___attn_k_norm(k_14);  k_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_83 = torch._C._nn.scaled_dot_product_attention(q_15, k_15, v_8, dropout_p = 0.0);  q_15 = k_15 = v_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_12 = x_83.transpose(1, 2);  x_83 = None
    x_84 = transpose_12.reshape(8, 197, 256);  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_85 = self.getattr_L__mod___blocks_1_blocks_1___1___attn_proj(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_86 = self.getattr_L__mod___blocks_1_blocks_1___1___attn_proj_drop(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_1_blocks_1___1___ls1 = self.getattr_L__mod___blocks_1_blocks_1___1___ls1(x_86);  x_86 = None
    getattr_l__mod___blocks_1_blocks_1___1___drop_path1 = self.getattr_L__mod___blocks_1_blocks_1___1___drop_path1(getattr_l__mod___blocks_1_blocks_1___1___ls1);  getattr_l__mod___blocks_1_blocks_1___1___ls1 = None
    x_87 = x_82 + getattr_l__mod___blocks_1_blocks_1___1___drop_path1;  x_82 = getattr_l__mod___blocks_1_blocks_1___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_1_blocks_1___1___norm2 = self.getattr_L__mod___blocks_1_blocks_1___1___norm2(x_87)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_88 = self.getattr_L__mod___blocks_1_blocks_1___1___mlp_fc1(getattr_l__mod___blocks_1_blocks_1___1___norm2);  getattr_l__mod___blocks_1_blocks_1___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_89 = self.getattr_L__mod___blocks_1_blocks_1___1___mlp_act(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_90 = self.getattr_L__mod___blocks_1_blocks_1___1___mlp_drop1(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_91 = self.getattr_L__mod___blocks_1_blocks_1___1___mlp_norm(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_92 = self.getattr_L__mod___blocks_1_blocks_1___1___mlp_fc2(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_93 = self.getattr_L__mod___blocks_1_blocks_1___1___mlp_drop2(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_1_blocks_1___1___ls2 = self.getattr_L__mod___blocks_1_blocks_1___1___ls2(x_93);  x_93 = None
    getattr_l__mod___blocks_1_blocks_1___1___drop_path2 = self.getattr_L__mod___blocks_1_blocks_1___1___drop_path2(getattr_l__mod___blocks_1_blocks_1___1___ls2);  getattr_l__mod___blocks_1_blocks_1___1___ls2 = None
    x_94 = x_87 + getattr_l__mod___blocks_1_blocks_1___1___drop_path2;  x_87 = getattr_l__mod___blocks_1_blocks_1___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_1_blocks_1___2___norm1 = self.getattr_L__mod___blocks_1_blocks_1___2___norm1(x_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks_1_blocks_1___2___attn_qkv = self.getattr_L__mod___blocks_1_blocks_1___2___attn_qkv(getattr_l__mod___blocks_1_blocks_1___2___norm1);  getattr_l__mod___blocks_1_blocks_1___2___norm1 = None
    reshape_22 = getattr_l__mod___blocks_1_blocks_1___2___attn_qkv.reshape(8, 197, 3, 4, 64);  getattr_l__mod___blocks_1_blocks_1___2___attn_qkv = None
    qkv_7 = reshape_22.permute(2, 0, 3, 1, 4);  reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_7 = qkv_7.unbind(0);  qkv_7 = None
    q_16 = unbind_7[0]
    k_16 = unbind_7[1]
    v_9 = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_17 = self.getattr_L__mod___blocks_1_blocks_1___2___attn_q_norm(q_16);  q_16 = None
    k_17 = self.getattr_L__mod___blocks_1_blocks_1___2___attn_k_norm(k_16);  k_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_95 = torch._C._nn.scaled_dot_product_attention(q_17, k_17, v_9, dropout_p = 0.0);  q_17 = k_17 = v_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_13 = x_95.transpose(1, 2);  x_95 = None
    x_96 = transpose_13.reshape(8, 197, 256);  transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_97 = self.getattr_L__mod___blocks_1_blocks_1___2___attn_proj(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_98 = self.getattr_L__mod___blocks_1_blocks_1___2___attn_proj_drop(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_1_blocks_1___2___ls1 = self.getattr_L__mod___blocks_1_blocks_1___2___ls1(x_98);  x_98 = None
    getattr_l__mod___blocks_1_blocks_1___2___drop_path1 = self.getattr_L__mod___blocks_1_blocks_1___2___drop_path1(getattr_l__mod___blocks_1_blocks_1___2___ls1);  getattr_l__mod___blocks_1_blocks_1___2___ls1 = None
    x_99 = x_94 + getattr_l__mod___blocks_1_blocks_1___2___drop_path1;  x_94 = getattr_l__mod___blocks_1_blocks_1___2___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_1_blocks_1___2___norm2 = self.getattr_L__mod___blocks_1_blocks_1___2___norm2(x_99)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_100 = self.getattr_L__mod___blocks_1_blocks_1___2___mlp_fc1(getattr_l__mod___blocks_1_blocks_1___2___norm2);  getattr_l__mod___blocks_1_blocks_1___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_101 = self.getattr_L__mod___blocks_1_blocks_1___2___mlp_act(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_102 = self.getattr_L__mod___blocks_1_blocks_1___2___mlp_drop1(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_103 = self.getattr_L__mod___blocks_1_blocks_1___2___mlp_norm(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_104 = self.getattr_L__mod___blocks_1_blocks_1___2___mlp_fc2(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_105 = self.getattr_L__mod___blocks_1_blocks_1___2___mlp_drop2(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_1_blocks_1___2___ls2 = self.getattr_L__mod___blocks_1_blocks_1___2___ls2(x_105);  x_105 = None
    getattr_l__mod___blocks_1_blocks_1___2___drop_path2 = self.getattr_L__mod___blocks_1_blocks_1___2___drop_path2(getattr_l__mod___blocks_1_blocks_1___2___ls2);  getattr_l__mod___blocks_1_blocks_1___2___ls2 = None
    x_106 = x_99 + getattr_l__mod___blocks_1_blocks_1___2___drop_path2;  x_99 = getattr_l__mod___blocks_1_blocks_1___2___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    getitem_36 = x_70[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_1_projs_0_0 = self.L__mod___blocks_1_projs_0_0(getitem_36);  getitem_36 = None
    l__mod___blocks_1_projs_0_1 = self.L__mod___blocks_1_projs_0_1(l__mod___blocks_1_projs_0_0);  l__mod___blocks_1_projs_0_0 = None
    l__mod___blocks_1_projs_0_2 = self.L__mod___blocks_1_projs_0_2(l__mod___blocks_1_projs_0_1);  l__mod___blocks_1_projs_0_1 = None
    getitem_37 = x_106[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_1_projs_1_0 = self.L__mod___blocks_1_projs_1_0(getitem_37);  getitem_37 = None
    l__mod___blocks_1_projs_1_1 = self.L__mod___blocks_1_projs_1_1(l__mod___blocks_1_projs_1_0);  l__mod___blocks_1_projs_1_0 = None
    l__mod___blocks_1_projs_1_2 = self.L__mod___blocks_1_projs_1_2(l__mod___blocks_1_projs_1_1);  l__mod___blocks_1_projs_1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    getitem_38 = x_106[(slice(None, None, None), slice(1, None, None), Ellipsis)]
    tmp_6 = torch.cat((l__mod___blocks_1_projs_0_2, getitem_38), dim = 1);  l__mod___blocks_1_projs_0_2 = getitem_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    getitem_39 = tmp_6[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_1_fusion_0_norm1 = self.L__mod___blocks_1_fusion_0_norm1(tmp_6);  tmp_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    getitem_40 = l__mod___blocks_1_fusion_0_norm1[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_1_fusion_0_attn_wq = self.L__mod___blocks_1_fusion_0_attn_wq(getitem_40);  getitem_40 = None
    reshape_24 = l__mod___blocks_1_fusion_0_attn_wq.reshape(8, 1, 4, 64);  l__mod___blocks_1_fusion_0_attn_wq = None
    q_18 = reshape_24.permute(0, 2, 1, 3);  reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_1_fusion_0_attn_wk = self.L__mod___blocks_1_fusion_0_attn_wk(l__mod___blocks_1_fusion_0_norm1)
    reshape_25 = l__mod___blocks_1_fusion_0_attn_wk.reshape(8, 197, 4, 64);  l__mod___blocks_1_fusion_0_attn_wk = None
    k_18 = reshape_25.permute(0, 2, 1, 3);  reshape_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_1_fusion_0_attn_wv = self.L__mod___blocks_1_fusion_0_attn_wv(l__mod___blocks_1_fusion_0_norm1);  l__mod___blocks_1_fusion_0_norm1 = None
    reshape_26 = l__mod___blocks_1_fusion_0_attn_wv.reshape(8, 197, 4, 64);  l__mod___blocks_1_fusion_0_attn_wv = None
    v_10 = reshape_26.permute(0, 2, 1, 3);  reshape_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    transpose_14 = k_18.transpose(-2, -1);  k_18 = None
    matmul_4 = q_18 @ transpose_14;  q_18 = transpose_14 = None
    attn_6 = matmul_4 * 0.125;  matmul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    attn_7 = attn_6.softmax(dim = -1);  attn_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    attn_8 = self.L__mod___blocks_1_fusion_0_attn_attn_drop(attn_7);  attn_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    matmul_5 = attn_8 @ v_10;  attn_8 = v_10 = None
    transpose_15 = matmul_5.transpose(1, 2);  matmul_5 = None
    x_107 = transpose_15.reshape(8, 1, 256);  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    x_108 = self.L__mod___blocks_1_fusion_0_attn_proj(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    x_109 = self.L__mod___blocks_1_fusion_0_attn_proj_drop(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_1_fusion_0_drop_path = self.L__mod___blocks_1_fusion_0_drop_path(x_109);  x_109 = None
    tmp_7 = getitem_39 + l__mod___blocks_1_fusion_0_drop_path;  getitem_39 = l__mod___blocks_1_fusion_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    getitem_41 = tmp_7[(slice(None, None, None), slice(0, 1, None), Ellipsis)];  tmp_7 = None
    l__mod___blocks_1_revert_projs_0_0 = self.L__mod___blocks_1_revert_projs_0_0(getitem_41);  getitem_41 = None
    l__mod___blocks_1_revert_projs_0_1 = self.L__mod___blocks_1_revert_projs_0_1(l__mod___blocks_1_revert_projs_0_0);  l__mod___blocks_1_revert_projs_0_0 = None
    reverted_proj_cls_token_2 = self.L__mod___blocks_1_revert_projs_0_2(l__mod___blocks_1_revert_projs_0_1);  l__mod___blocks_1_revert_projs_0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    getitem_42 = x_70[(slice(None, None, None), slice(1, None, None), Ellipsis)]
    tmp_8 = torch.cat((reverted_proj_cls_token_2, getitem_42), dim = 1);  reverted_proj_cls_token_2 = getitem_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    getitem_43 = x_70[(slice(None, None, None), slice(1, None, None), Ellipsis)];  x_70 = None
    tmp_9 = torch.cat((l__mod___blocks_1_projs_1_2, getitem_43), dim = 1);  l__mod___blocks_1_projs_1_2 = getitem_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    getitem_44 = tmp_9[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_1_fusion_1_norm1 = self.L__mod___blocks_1_fusion_1_norm1(tmp_9);  tmp_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    getitem_45 = l__mod___blocks_1_fusion_1_norm1[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_1_fusion_1_attn_wq = self.L__mod___blocks_1_fusion_1_attn_wq(getitem_45);  getitem_45 = None
    reshape_28 = l__mod___blocks_1_fusion_1_attn_wq.reshape(8, 1, 4, 32);  l__mod___blocks_1_fusion_1_attn_wq = None
    q_19 = reshape_28.permute(0, 2, 1, 3);  reshape_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_1_fusion_1_attn_wk = self.L__mod___blocks_1_fusion_1_attn_wk(l__mod___blocks_1_fusion_1_norm1)
    reshape_29 = l__mod___blocks_1_fusion_1_attn_wk.reshape(8, 401, 4, 32);  l__mod___blocks_1_fusion_1_attn_wk = None
    k_19 = reshape_29.permute(0, 2, 1, 3);  reshape_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_1_fusion_1_attn_wv = self.L__mod___blocks_1_fusion_1_attn_wv(l__mod___blocks_1_fusion_1_norm1);  l__mod___blocks_1_fusion_1_norm1 = None
    reshape_30 = l__mod___blocks_1_fusion_1_attn_wv.reshape(8, 401, 4, 32);  l__mod___blocks_1_fusion_1_attn_wv = None
    v_11 = reshape_30.permute(0, 2, 1, 3);  reshape_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    transpose_16 = k_19.transpose(-2, -1);  k_19 = None
    matmul_6 = q_19 @ transpose_16;  q_19 = transpose_16 = None
    attn_9 = matmul_6 * 0.1767766952966369;  matmul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    attn_10 = attn_9.softmax(dim = -1);  attn_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    attn_11 = self.L__mod___blocks_1_fusion_1_attn_attn_drop(attn_10);  attn_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    matmul_7 = attn_11 @ v_11;  attn_11 = v_11 = None
    transpose_17 = matmul_7.transpose(1, 2);  matmul_7 = None
    x_111 = transpose_17.reshape(8, 1, 128);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    x_112 = self.L__mod___blocks_1_fusion_1_attn_proj(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    x_113 = self.L__mod___blocks_1_fusion_1_attn_proj_drop(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_1_fusion_1_drop_path = self.L__mod___blocks_1_fusion_1_drop_path(x_113);  x_113 = None
    tmp_10 = getitem_44 + l__mod___blocks_1_fusion_1_drop_path;  getitem_44 = l__mod___blocks_1_fusion_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    getitem_46 = tmp_10[(slice(None, None, None), slice(0, 1, None), Ellipsis)];  tmp_10 = None
    l__mod___blocks_1_revert_projs_1_0 = self.L__mod___blocks_1_revert_projs_1_0(getitem_46);  getitem_46 = None
    l__mod___blocks_1_revert_projs_1_1 = self.L__mod___blocks_1_revert_projs_1_1(l__mod___blocks_1_revert_projs_1_0);  l__mod___blocks_1_revert_projs_1_0 = None
    reverted_proj_cls_token_3 = self.L__mod___blocks_1_revert_projs_1_2(l__mod___blocks_1_revert_projs_1_1);  l__mod___blocks_1_revert_projs_1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    getitem_47 = x_106[(slice(None, None, None), slice(1, None, None), Ellipsis)];  x_106 = None
    tmp_11 = torch.cat((reverted_proj_cls_token_3, getitem_47), dim = 1);  reverted_proj_cls_token_3 = getitem_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_2_blocks_0___0___norm1 = self.getattr_L__mod___blocks_2_blocks_0___0___norm1(tmp_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks_2_blocks_0___0___attn_qkv = self.getattr_L__mod___blocks_2_blocks_0___0___attn_qkv(getattr_l__mod___blocks_2_blocks_0___0___norm1);  getattr_l__mod___blocks_2_blocks_0___0___norm1 = None
    reshape_32 = getattr_l__mod___blocks_2_blocks_0___0___attn_qkv.reshape(8, 401, 3, 4, 32);  getattr_l__mod___blocks_2_blocks_0___0___attn_qkv = None
    qkv_8 = reshape_32.permute(2, 0, 3, 1, 4);  reshape_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_8 = qkv_8.unbind(0);  qkv_8 = None
    q_20 = unbind_8[0]
    k_20 = unbind_8[1]
    v_12 = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_21 = self.getattr_L__mod___blocks_2_blocks_0___0___attn_q_norm(q_20);  q_20 = None
    k_21 = self.getattr_L__mod___blocks_2_blocks_0___0___attn_k_norm(k_20);  k_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_115 = torch._C._nn.scaled_dot_product_attention(q_21, k_21, v_12, dropout_p = 0.0);  q_21 = k_21 = v_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_18 = x_115.transpose(1, 2);  x_115 = None
    x_116 = transpose_18.reshape(8, 401, 128);  transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_117 = self.getattr_L__mod___blocks_2_blocks_0___0___attn_proj(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_118 = self.getattr_L__mod___blocks_2_blocks_0___0___attn_proj_drop(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_2_blocks_0___0___ls1 = self.getattr_L__mod___blocks_2_blocks_0___0___ls1(x_118);  x_118 = None
    getattr_l__mod___blocks_2_blocks_0___0___drop_path1 = self.getattr_L__mod___blocks_2_blocks_0___0___drop_path1(getattr_l__mod___blocks_2_blocks_0___0___ls1);  getattr_l__mod___blocks_2_blocks_0___0___ls1 = None
    x_119 = tmp_8 + getattr_l__mod___blocks_2_blocks_0___0___drop_path1;  tmp_8 = getattr_l__mod___blocks_2_blocks_0___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_2_blocks_0___0___norm2 = self.getattr_L__mod___blocks_2_blocks_0___0___norm2(x_119)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_120 = self.getattr_L__mod___blocks_2_blocks_0___0___mlp_fc1(getattr_l__mod___blocks_2_blocks_0___0___norm2);  getattr_l__mod___blocks_2_blocks_0___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_121 = self.getattr_L__mod___blocks_2_blocks_0___0___mlp_act(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_122 = self.getattr_L__mod___blocks_2_blocks_0___0___mlp_drop1(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_123 = self.getattr_L__mod___blocks_2_blocks_0___0___mlp_norm(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_124 = self.getattr_L__mod___blocks_2_blocks_0___0___mlp_fc2(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_125 = self.getattr_L__mod___blocks_2_blocks_0___0___mlp_drop2(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_2_blocks_0___0___ls2 = self.getattr_L__mod___blocks_2_blocks_0___0___ls2(x_125);  x_125 = None
    getattr_l__mod___blocks_2_blocks_0___0___drop_path2 = self.getattr_L__mod___blocks_2_blocks_0___0___drop_path2(getattr_l__mod___blocks_2_blocks_0___0___ls2);  getattr_l__mod___blocks_2_blocks_0___0___ls2 = None
    x_126 = x_119 + getattr_l__mod___blocks_2_blocks_0___0___drop_path2;  x_119 = getattr_l__mod___blocks_2_blocks_0___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_2_blocks_1___0___norm1 = self.getattr_L__mod___blocks_2_blocks_1___0___norm1(tmp_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks_2_blocks_1___0___attn_qkv = self.getattr_L__mod___blocks_2_blocks_1___0___attn_qkv(getattr_l__mod___blocks_2_blocks_1___0___norm1);  getattr_l__mod___blocks_2_blocks_1___0___norm1 = None
    reshape_34 = getattr_l__mod___blocks_2_blocks_1___0___attn_qkv.reshape(8, 197, 3, 4, 64);  getattr_l__mod___blocks_2_blocks_1___0___attn_qkv = None
    qkv_9 = reshape_34.permute(2, 0, 3, 1, 4);  reshape_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_9 = qkv_9.unbind(0);  qkv_9 = None
    q_22 = unbind_9[0]
    k_22 = unbind_9[1]
    v_13 = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_23 = self.getattr_L__mod___blocks_2_blocks_1___0___attn_q_norm(q_22);  q_22 = None
    k_23 = self.getattr_L__mod___blocks_2_blocks_1___0___attn_k_norm(k_22);  k_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_127 = torch._C._nn.scaled_dot_product_attention(q_23, k_23, v_13, dropout_p = 0.0);  q_23 = k_23 = v_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_19 = x_127.transpose(1, 2);  x_127 = None
    x_128 = transpose_19.reshape(8, 197, 256);  transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_129 = self.getattr_L__mod___blocks_2_blocks_1___0___attn_proj(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_130 = self.getattr_L__mod___blocks_2_blocks_1___0___attn_proj_drop(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_2_blocks_1___0___ls1 = self.getattr_L__mod___blocks_2_blocks_1___0___ls1(x_130);  x_130 = None
    getattr_l__mod___blocks_2_blocks_1___0___drop_path1 = self.getattr_L__mod___blocks_2_blocks_1___0___drop_path1(getattr_l__mod___blocks_2_blocks_1___0___ls1);  getattr_l__mod___blocks_2_blocks_1___0___ls1 = None
    x_131 = tmp_11 + getattr_l__mod___blocks_2_blocks_1___0___drop_path1;  tmp_11 = getattr_l__mod___blocks_2_blocks_1___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_2_blocks_1___0___norm2 = self.getattr_L__mod___blocks_2_blocks_1___0___norm2(x_131)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_132 = self.getattr_L__mod___blocks_2_blocks_1___0___mlp_fc1(getattr_l__mod___blocks_2_blocks_1___0___norm2);  getattr_l__mod___blocks_2_blocks_1___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_133 = self.getattr_L__mod___blocks_2_blocks_1___0___mlp_act(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_134 = self.getattr_L__mod___blocks_2_blocks_1___0___mlp_drop1(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_135 = self.getattr_L__mod___blocks_2_blocks_1___0___mlp_norm(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_136 = self.getattr_L__mod___blocks_2_blocks_1___0___mlp_fc2(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_137 = self.getattr_L__mod___blocks_2_blocks_1___0___mlp_drop2(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_2_blocks_1___0___ls2 = self.getattr_L__mod___blocks_2_blocks_1___0___ls2(x_137);  x_137 = None
    getattr_l__mod___blocks_2_blocks_1___0___drop_path2 = self.getattr_L__mod___blocks_2_blocks_1___0___drop_path2(getattr_l__mod___blocks_2_blocks_1___0___ls2);  getattr_l__mod___blocks_2_blocks_1___0___ls2 = None
    x_138 = x_131 + getattr_l__mod___blocks_2_blocks_1___0___drop_path2;  x_131 = getattr_l__mod___blocks_2_blocks_1___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_2_blocks_1___1___norm1 = self.getattr_L__mod___blocks_2_blocks_1___1___norm1(x_138)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks_2_blocks_1___1___attn_qkv = self.getattr_L__mod___blocks_2_blocks_1___1___attn_qkv(getattr_l__mod___blocks_2_blocks_1___1___norm1);  getattr_l__mod___blocks_2_blocks_1___1___norm1 = None
    reshape_36 = getattr_l__mod___blocks_2_blocks_1___1___attn_qkv.reshape(8, 197, 3, 4, 64);  getattr_l__mod___blocks_2_blocks_1___1___attn_qkv = None
    qkv_10 = reshape_36.permute(2, 0, 3, 1, 4);  reshape_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_10 = qkv_10.unbind(0);  qkv_10 = None
    q_24 = unbind_10[0]
    k_24 = unbind_10[1]
    v_14 = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_25 = self.getattr_L__mod___blocks_2_blocks_1___1___attn_q_norm(q_24);  q_24 = None
    k_25 = self.getattr_L__mod___blocks_2_blocks_1___1___attn_k_norm(k_24);  k_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_139 = torch._C._nn.scaled_dot_product_attention(q_25, k_25, v_14, dropout_p = 0.0);  q_25 = k_25 = v_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_20 = x_139.transpose(1, 2);  x_139 = None
    x_140 = transpose_20.reshape(8, 197, 256);  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_141 = self.getattr_L__mod___blocks_2_blocks_1___1___attn_proj(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_142 = self.getattr_L__mod___blocks_2_blocks_1___1___attn_proj_drop(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_2_blocks_1___1___ls1 = self.getattr_L__mod___blocks_2_blocks_1___1___ls1(x_142);  x_142 = None
    getattr_l__mod___blocks_2_blocks_1___1___drop_path1 = self.getattr_L__mod___blocks_2_blocks_1___1___drop_path1(getattr_l__mod___blocks_2_blocks_1___1___ls1);  getattr_l__mod___blocks_2_blocks_1___1___ls1 = None
    x_143 = x_138 + getattr_l__mod___blocks_2_blocks_1___1___drop_path1;  x_138 = getattr_l__mod___blocks_2_blocks_1___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_2_blocks_1___1___norm2 = self.getattr_L__mod___blocks_2_blocks_1___1___norm2(x_143)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_144 = self.getattr_L__mod___blocks_2_blocks_1___1___mlp_fc1(getattr_l__mod___blocks_2_blocks_1___1___norm2);  getattr_l__mod___blocks_2_blocks_1___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_145 = self.getattr_L__mod___blocks_2_blocks_1___1___mlp_act(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_146 = self.getattr_L__mod___blocks_2_blocks_1___1___mlp_drop1(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_147 = self.getattr_L__mod___blocks_2_blocks_1___1___mlp_norm(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_148 = self.getattr_L__mod___blocks_2_blocks_1___1___mlp_fc2(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_149 = self.getattr_L__mod___blocks_2_blocks_1___1___mlp_drop2(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_2_blocks_1___1___ls2 = self.getattr_L__mod___blocks_2_blocks_1___1___ls2(x_149);  x_149 = None
    getattr_l__mod___blocks_2_blocks_1___1___drop_path2 = self.getattr_L__mod___blocks_2_blocks_1___1___drop_path2(getattr_l__mod___blocks_2_blocks_1___1___ls2);  getattr_l__mod___blocks_2_blocks_1___1___ls2 = None
    x_150 = x_143 + getattr_l__mod___blocks_2_blocks_1___1___drop_path2;  x_143 = getattr_l__mod___blocks_2_blocks_1___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_2_blocks_1___2___norm1 = self.getattr_L__mod___blocks_2_blocks_1___2___norm1(x_150)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:81, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks_2_blocks_1___2___attn_qkv = self.getattr_L__mod___blocks_2_blocks_1___2___attn_qkv(getattr_l__mod___blocks_2_blocks_1___2___norm1);  getattr_l__mod___blocks_2_blocks_1___2___norm1 = None
    reshape_38 = getattr_l__mod___blocks_2_blocks_1___2___attn_qkv.reshape(8, 197, 3, 4, 64);  getattr_l__mod___blocks_2_blocks_1___2___attn_qkv = None
    qkv_11 = reshape_38.permute(2, 0, 3, 1, 4);  reshape_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:82, code: q, k, v = qkv.unbind(0)
    unbind_11 = qkv_11.unbind(0);  qkv_11 = None
    q_26 = unbind_11[0]
    k_26 = unbind_11[1]
    v_15 = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:83, code: q, k = self.q_norm(q), self.k_norm(k)
    q_27 = self.getattr_L__mod___blocks_2_blocks_1___2___attn_q_norm(q_26);  q_26 = None
    k_27 = self.getattr_L__mod___blocks_2_blocks_1___2___attn_k_norm(k_26);  k_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:86, code: x = F.scaled_dot_product_attention(
    x_151 = torch._C._nn.scaled_dot_product_attention(q_27, k_27, v_15, dropout_p = 0.0);  q_27 = k_27 = v_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:97, code: x = x.transpose(1, 2).reshape(B, N, C)
    transpose_21 = x_151.transpose(1, 2);  x_151 = None
    x_152 = transpose_21.reshape(8, 197, 256);  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:98, code: x = self.proj(x)
    x_153 = self.getattr_L__mod___blocks_2_blocks_1___2___attn_proj(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:99, code: x = self.proj_drop(x)
    x_154 = self.getattr_L__mod___blocks_2_blocks_1___2___attn_proj_drop(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:155, code: x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
    getattr_l__mod___blocks_2_blocks_1___2___ls1 = self.getattr_L__mod___blocks_2_blocks_1___2___ls1(x_154);  x_154 = None
    getattr_l__mod___blocks_2_blocks_1___2___drop_path1 = self.getattr_L__mod___blocks_2_blocks_1___2___drop_path1(getattr_l__mod___blocks_2_blocks_1___2___ls1);  getattr_l__mod___blocks_2_blocks_1___2___ls1 = None
    x_155 = x_150 + getattr_l__mod___blocks_2_blocks_1___2___drop_path1;  x_150 = getattr_l__mod___blocks_2_blocks_1___2___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_2_blocks_1___2___norm2 = self.getattr_L__mod___blocks_2_blocks_1___2___norm2(x_155)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_156 = self.getattr_L__mod___blocks_2_blocks_1___2___mlp_fc1(getattr_l__mod___blocks_2_blocks_1___2___norm2);  getattr_l__mod___blocks_2_blocks_1___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_157 = self.getattr_L__mod___blocks_2_blocks_1___2___mlp_act(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_158 = self.getattr_L__mod___blocks_2_blocks_1___2___mlp_drop1(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_159 = self.getattr_L__mod___blocks_2_blocks_1___2___mlp_norm(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_160 = self.getattr_L__mod___blocks_2_blocks_1___2___mlp_fc2(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_161 = self.getattr_L__mod___blocks_2_blocks_1___2___mlp_drop2(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/vision_transformer.py:156, code: x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
    getattr_l__mod___blocks_2_blocks_1___2___ls2 = self.getattr_L__mod___blocks_2_blocks_1___2___ls2(x_161);  x_161 = None
    getattr_l__mod___blocks_2_blocks_1___2___drop_path2 = self.getattr_L__mod___blocks_2_blocks_1___2___drop_path2(getattr_l__mod___blocks_2_blocks_1___2___ls2);  getattr_l__mod___blocks_2_blocks_1___2___ls2 = None
    x_162 = x_155 + getattr_l__mod___blocks_2_blocks_1___2___drop_path2;  x_155 = getattr_l__mod___blocks_2_blocks_1___2___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:258, code: proj_cls_token.append(proj(outs_b[i][:, 0:1, ...]))
    getitem_60 = x_126[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_2_projs_0_0 = self.L__mod___blocks_2_projs_0_0(getitem_60);  getitem_60 = None
    l__mod___blocks_2_projs_0_1 = self.L__mod___blocks_2_projs_0_1(l__mod___blocks_2_projs_0_0);  l__mod___blocks_2_projs_0_0 = None
    l__mod___blocks_2_projs_0_2 = self.L__mod___blocks_2_projs_0_2(l__mod___blocks_2_projs_0_1);  l__mod___blocks_2_projs_0_1 = None
    getitem_61 = x_162[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_2_projs_1_0 = self.L__mod___blocks_2_projs_1_0(getitem_61);  getitem_61 = None
    l__mod___blocks_2_projs_1_1 = self.L__mod___blocks_2_projs_1_1(l__mod___blocks_2_projs_1_0);  l__mod___blocks_2_projs_1_0 = None
    l__mod___blocks_2_projs_1_2 = self.L__mod___blocks_2_projs_1_2(l__mod___blocks_2_projs_1_1);  l__mod___blocks_2_projs_1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    getitem_62 = x_162[(slice(None, None, None), slice(1, None, None), Ellipsis)]
    tmp_12 = torch.cat((l__mod___blocks_2_projs_0_2, getitem_62), dim = 1);  l__mod___blocks_2_projs_0_2 = getitem_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    getitem_63 = tmp_12[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_2_fusion_0_norm1 = self.L__mod___blocks_2_fusion_0_norm1(tmp_12);  tmp_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    getitem_64 = l__mod___blocks_2_fusion_0_norm1[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_2_fusion_0_attn_wq = self.L__mod___blocks_2_fusion_0_attn_wq(getitem_64);  getitem_64 = None
    reshape_40 = l__mod___blocks_2_fusion_0_attn_wq.reshape(8, 1, 4, 64);  l__mod___blocks_2_fusion_0_attn_wq = None
    q_28 = reshape_40.permute(0, 2, 1, 3);  reshape_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_fusion_0_attn_wk = self.L__mod___blocks_2_fusion_0_attn_wk(l__mod___blocks_2_fusion_0_norm1)
    reshape_41 = l__mod___blocks_2_fusion_0_attn_wk.reshape(8, 197, 4, 64);  l__mod___blocks_2_fusion_0_attn_wk = None
    k_28 = reshape_41.permute(0, 2, 1, 3);  reshape_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_fusion_0_attn_wv = self.L__mod___blocks_2_fusion_0_attn_wv(l__mod___blocks_2_fusion_0_norm1);  l__mod___blocks_2_fusion_0_norm1 = None
    reshape_42 = l__mod___blocks_2_fusion_0_attn_wv.reshape(8, 197, 4, 64);  l__mod___blocks_2_fusion_0_attn_wv = None
    v_16 = reshape_42.permute(0, 2, 1, 3);  reshape_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    transpose_22 = k_28.transpose(-2, -1);  k_28 = None
    matmul_8 = q_28 @ transpose_22;  q_28 = transpose_22 = None
    attn_12 = matmul_8 * 0.125;  matmul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    attn_13 = attn_12.softmax(dim = -1);  attn_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    attn_14 = self.L__mod___blocks_2_fusion_0_attn_attn_drop(attn_13);  attn_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    matmul_9 = attn_14 @ v_16;  attn_14 = v_16 = None
    transpose_23 = matmul_9.transpose(1, 2);  matmul_9 = None
    x_163 = transpose_23.reshape(8, 1, 256);  transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    x_164 = self.L__mod___blocks_2_fusion_0_attn_proj(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    x_165 = self.L__mod___blocks_2_fusion_0_attn_proj_drop(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_2_fusion_0_drop_path = self.L__mod___blocks_2_fusion_0_drop_path(x_165);  x_165 = None
    tmp_13 = getitem_63 + l__mod___blocks_2_fusion_0_drop_path;  getitem_63 = l__mod___blocks_2_fusion_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    getitem_65 = tmp_13[(slice(None, None, None), slice(0, 1, None), Ellipsis)];  tmp_13 = None
    l__mod___blocks_2_revert_projs_0_0 = self.L__mod___blocks_2_revert_projs_0_0(getitem_65);  getitem_65 = None
    l__mod___blocks_2_revert_projs_0_1 = self.L__mod___blocks_2_revert_projs_0_1(l__mod___blocks_2_revert_projs_0_0);  l__mod___blocks_2_revert_projs_0_0 = None
    reverted_proj_cls_token_4 = self.L__mod___blocks_2_revert_projs_0_2(l__mod___blocks_2_revert_projs_0_1);  l__mod___blocks_2_revert_projs_0_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    getitem_66 = x_126[(slice(None, None, None), slice(1, None, None), Ellipsis)]
    tmp_14 = torch.cat((reverted_proj_cls_token_4, getitem_66), dim = 1);  reverted_proj_cls_token_4 = getitem_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:263, code: tmp = torch.cat((proj_cls_token[i], outs_b[(i + 1) % self.num_branches][:, 1:, ...]), dim=1)
    getitem_67 = x_126[(slice(None, None, None), slice(1, None, None), Ellipsis)];  x_126 = None
    tmp_15 = torch.cat((l__mod___blocks_2_projs_1_2, getitem_67), dim = 1);  l__mod___blocks_2_projs_1_2 = getitem_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    getitem_68 = tmp_15[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_2_fusion_1_norm1 = self.L__mod___blocks_2_fusion_1_norm1(tmp_15);  tmp_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:113, code: q = self.wq(x[:, 0:1, ...]).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    getitem_69 = l__mod___blocks_2_fusion_1_norm1[(slice(None, None, None), slice(0, 1, None), Ellipsis)]
    l__mod___blocks_2_fusion_1_attn_wq = self.L__mod___blocks_2_fusion_1_attn_wq(getitem_69);  getitem_69 = None
    reshape_44 = l__mod___blocks_2_fusion_1_attn_wq.reshape(8, 1, 4, 32);  l__mod___blocks_2_fusion_1_attn_wq = None
    q_29 = reshape_44.permute(0, 2, 1, 3);  reshape_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:115, code: k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_fusion_1_attn_wk = self.L__mod___blocks_2_fusion_1_attn_wk(l__mod___blocks_2_fusion_1_norm1)
    reshape_45 = l__mod___blocks_2_fusion_1_attn_wk.reshape(8, 401, 4, 32);  l__mod___blocks_2_fusion_1_attn_wk = None
    k_29 = reshape_45.permute(0, 2, 1, 3);  reshape_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:117, code: v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_2_fusion_1_attn_wv = self.L__mod___blocks_2_fusion_1_attn_wv(l__mod___blocks_2_fusion_1_norm1);  l__mod___blocks_2_fusion_1_norm1 = None
    reshape_46 = l__mod___blocks_2_fusion_1_attn_wv.reshape(8, 401, 4, 32);  l__mod___blocks_2_fusion_1_attn_wv = None
    v_17 = reshape_46.permute(0, 2, 1, 3);  reshape_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:119, code: attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
    transpose_24 = k_29.transpose(-2, -1);  k_29 = None
    matmul_10 = q_29 @ transpose_24;  q_29 = transpose_24 = None
    attn_15 = matmul_10 * 0.1767766952966369;  matmul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:120, code: attn = attn.softmax(dim=-1)
    attn_16 = attn_15.softmax(dim = -1);  attn_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:121, code: attn = self.attn_drop(attn)
    attn_17 = self.L__mod___blocks_2_fusion_1_attn_attn_drop(attn_16);  attn_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:123, code: x = (attn @ v).transpose(1, 2).reshape(B, 1, C)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
    matmul_11 = attn_17 @ v_17;  attn_17 = v_17 = None
    transpose_25 = matmul_11.transpose(1, 2);  matmul_11 = None
    x_167 = transpose_25.reshape(8, 1, 128);  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:124, code: x = self.proj(x)
    x_168 = self.L__mod___blocks_2_fusion_1_attn_proj(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:125, code: x = self.proj_drop(x)
    x_169 = self.L__mod___blocks_2_fusion_1_attn_proj_drop(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:156, code: x = x[:, 0:1, ...] + self.drop_path(self.attn(self.norm1(x)))
    l__mod___blocks_2_fusion_1_drop_path = self.L__mod___blocks_2_fusion_1_drop_path(x_169);  x_169 = None
    tmp_16 = getitem_68 + l__mod___blocks_2_fusion_1_drop_path;  getitem_68 = l__mod___blocks_2_fusion_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:265, code: reverted_proj_cls_token = revert_proj(tmp[:, 0:1, ...])
    getitem_70 = tmp_16[(slice(None, None, None), slice(0, 1, None), Ellipsis)];  tmp_16 = None
    l__mod___blocks_2_revert_projs_1_0 = self.L__mod___blocks_2_revert_projs_1_0(getitem_70);  getitem_70 = None
    l__mod___blocks_2_revert_projs_1_1 = self.L__mod___blocks_2_revert_projs_1_1(l__mod___blocks_2_revert_projs_1_0);  l__mod___blocks_2_revert_projs_1_0 = None
    reverted_proj_cls_token_5 = self.L__mod___blocks_2_revert_projs_1_2(l__mod___blocks_2_revert_projs_1_1);  l__mod___blocks_2_revert_projs_1_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:266, code: tmp = torch.cat((reverted_proj_cls_token, outs_b[i][:, 1:, ...]), dim=1)
    getitem_71 = x_162[(slice(None, None, None), slice(1, None, None), Ellipsis)];  x_162 = None
    tmp_17 = torch.cat((reverted_proj_cls_token_5, getitem_71), dim = 1);  reverted_proj_cls_token_5 = getitem_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:451, code: xs = [norm(xs[i]) for i, norm in enumerate(self.norm)]
    x_171 = self.L__mod___norm_0(tmp_14);  tmp_14 = None
    x_172 = self.L__mod___norm_1(tmp_17);  tmp_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:455, code: xs = [x[:, 1:].mean(dim=1) for x in xs] if self.global_pool == 'avg' else [x[:, 0] for x in xs]
    x_173 = x_171[(slice(None, None, None), 0)];  x_171 = None
    x_174 = x_172[(slice(None, None, None), 0)];  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:456, code: xs = [self.head_drop(x) for x in xs]
    l__mod___head_drop = self.L__mod___head_drop(x_173);  x_173 = None
    l__mod___head_drop_1 = self.L__mod___head_drop(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:459, code: return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)
    l__mod___head_0 = self.L__mod___head_0(l__mod___head_drop);  l__mod___head_drop = None
    l__mod___head_1 = self.L__mod___head_1(l__mod___head_drop_1);  l__mod___head_drop_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/crossvit.py:459, code: return torch.mean(torch.stack([head(xs[i]) for i, head in enumerate(self.head)], dim=0), dim=0)
    stack = torch.stack([l__mod___head_0, l__mod___head_1], dim = 0);  l__mod___head_0 = l__mod___head_1 = None
    pred = torch.mean(stack, dim = 0);  stack = None
    return (pred,)
    