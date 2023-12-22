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
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:341, code: x = x + self.pos_embed
    l__mod___pos_embed = self.L__mod___pos_embed
    x_4 = x_3 + l__mod___pos_embed;  x_3 = l__mod___pos_embed = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:342, code: x = self.pos_drop(x)
    x_5 = self.L__mod___pos_drop(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___0___gamma_1 = self.getattr_L__mod___blocks___0___gamma_1
    getattr_l__mod___blocks___0___norm1 = self.getattr_L__mod___blocks___0___norm1(x_5)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___0___attn_qkv = self.getattr_L__mod___blocks___0___attn_qkv(getattr_l__mod___blocks___0___norm1);  getattr_l__mod___blocks___0___norm1 = None
    reshape = getattr_l__mod___blocks___0___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___0___attn_qkv = None
    qkv = reshape.permute(2, 0, 3, 1, 4);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem = qkv[0]
    q = getitem * 0.14433756729740643;  getitem = None
    k = qkv[1]
    v = qkv[2];  qkv = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_1 = k.transpose(-2, -1);  k = None
    attn = q @ transpose_1;  q = transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_1 = attn.permute(0, 2, 3, 1);  attn = None
    getattr_l__mod___blocks___0___attn_proj_l = self.getattr_L__mod___blocks___0___attn_proj_l(permute_1);  permute_1 = None
    attn_1 = getattr_l__mod___blocks___0___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___0___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_2 = attn_1.softmax(dim = -1);  attn_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_3 = attn_2.permute(0, 2, 3, 1);  attn_2 = None
    getattr_l__mod___blocks___0___attn_proj_w = self.getattr_L__mod___blocks___0___attn_proj_w(permute_3);  permute_3 = None
    attn_3 = getattr_l__mod___blocks___0___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___0___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_4 = self.getattr_L__mod___blocks___0___attn_attn_drop(attn_3);  attn_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_1 = attn_4 @ v;  attn_4 = v = None
    transpose_2 = matmul_1.transpose(1, 2);  matmul_1 = None
    x_6 = transpose_2.reshape(8, 576, 768);  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_7 = self.getattr_L__mod___blocks___0___attn_proj(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_8 = self.getattr_L__mod___blocks___0___attn_proj_drop(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_1 = getattr_l__mod___blocks___0___gamma_1 * x_8;  getattr_l__mod___blocks___0___gamma_1 = x_8 = None
    getattr_l__mod___blocks___0___drop_path = self.getattr_L__mod___blocks___0___drop_path(mul_1);  mul_1 = None
    x_9 = x_5 + getattr_l__mod___blocks___0___drop_path;  x_5 = getattr_l__mod___blocks___0___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___0___gamma_2 = self.getattr_L__mod___blocks___0___gamma_2
    getattr_l__mod___blocks___0___norm2 = self.getattr_L__mod___blocks___0___norm2(x_9)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_10 = self.getattr_L__mod___blocks___0___mlp_fc1(getattr_l__mod___blocks___0___norm2);  getattr_l__mod___blocks___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_11 = self.getattr_L__mod___blocks___0___mlp_act(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_12 = self.getattr_L__mod___blocks___0___mlp_drop1(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_13 = self.getattr_L__mod___blocks___0___mlp_norm(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_14 = self.getattr_L__mod___blocks___0___mlp_fc2(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_15 = self.getattr_L__mod___blocks___0___mlp_drop2(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_2 = getattr_l__mod___blocks___0___gamma_2 * x_15;  getattr_l__mod___blocks___0___gamma_2 = x_15 = None
    getattr_l__mod___blocks___0___drop_path_1 = self.getattr_L__mod___blocks___0___drop_path(mul_2);  mul_2 = None
    x_16 = x_9 + getattr_l__mod___blocks___0___drop_path_1;  x_9 = getattr_l__mod___blocks___0___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___1___gamma_1 = self.getattr_L__mod___blocks___1___gamma_1
    getattr_l__mod___blocks___1___norm1 = self.getattr_L__mod___blocks___1___norm1(x_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___1___attn_qkv = self.getattr_L__mod___blocks___1___attn_qkv(getattr_l__mod___blocks___1___norm1);  getattr_l__mod___blocks___1___norm1 = None
    reshape_2 = getattr_l__mod___blocks___1___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___1___attn_qkv = None
    qkv_1 = reshape_2.permute(2, 0, 3, 1, 4);  reshape_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_3 = qkv_1[0]
    q_1 = getitem_3 * 0.14433756729740643;  getitem_3 = None
    k_1 = qkv_1[1]
    v_1 = qkv_1[2];  qkv_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_3 = k_1.transpose(-2, -1);  k_1 = None
    attn_5 = q_1 @ transpose_3;  q_1 = transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_6 = attn_5.permute(0, 2, 3, 1);  attn_5 = None
    getattr_l__mod___blocks___1___attn_proj_l = self.getattr_L__mod___blocks___1___attn_proj_l(permute_6);  permute_6 = None
    attn_6 = getattr_l__mod___blocks___1___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___1___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_7 = attn_6.softmax(dim = -1);  attn_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_8 = attn_7.permute(0, 2, 3, 1);  attn_7 = None
    getattr_l__mod___blocks___1___attn_proj_w = self.getattr_L__mod___blocks___1___attn_proj_w(permute_8);  permute_8 = None
    attn_8 = getattr_l__mod___blocks___1___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___1___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_9 = self.getattr_L__mod___blocks___1___attn_attn_drop(attn_8);  attn_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_3 = attn_9 @ v_1;  attn_9 = v_1 = None
    transpose_4 = matmul_3.transpose(1, 2);  matmul_3 = None
    x_17 = transpose_4.reshape(8, 576, 768);  transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_18 = self.getattr_L__mod___blocks___1___attn_proj(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_19 = self.getattr_L__mod___blocks___1___attn_proj_drop(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_4 = getattr_l__mod___blocks___1___gamma_1 * x_19;  getattr_l__mod___blocks___1___gamma_1 = x_19 = None
    getattr_l__mod___blocks___1___drop_path = self.getattr_L__mod___blocks___1___drop_path(mul_4);  mul_4 = None
    x_20 = x_16 + getattr_l__mod___blocks___1___drop_path;  x_16 = getattr_l__mod___blocks___1___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___1___gamma_2 = self.getattr_L__mod___blocks___1___gamma_2
    getattr_l__mod___blocks___1___norm2 = self.getattr_L__mod___blocks___1___norm2(x_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_21 = self.getattr_L__mod___blocks___1___mlp_fc1(getattr_l__mod___blocks___1___norm2);  getattr_l__mod___blocks___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_22 = self.getattr_L__mod___blocks___1___mlp_act(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_23 = self.getattr_L__mod___blocks___1___mlp_drop1(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_24 = self.getattr_L__mod___blocks___1___mlp_norm(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_25 = self.getattr_L__mod___blocks___1___mlp_fc2(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_26 = self.getattr_L__mod___blocks___1___mlp_drop2(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_5 = getattr_l__mod___blocks___1___gamma_2 * x_26;  getattr_l__mod___blocks___1___gamma_2 = x_26 = None
    getattr_l__mod___blocks___1___drop_path_1 = self.getattr_L__mod___blocks___1___drop_path(mul_5);  mul_5 = None
    x_27 = x_20 + getattr_l__mod___blocks___1___drop_path_1;  x_20 = getattr_l__mod___blocks___1___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___2___gamma_1 = self.getattr_L__mod___blocks___2___gamma_1
    getattr_l__mod___blocks___2___norm1 = self.getattr_L__mod___blocks___2___norm1(x_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___2___attn_qkv = self.getattr_L__mod___blocks___2___attn_qkv(getattr_l__mod___blocks___2___norm1);  getattr_l__mod___blocks___2___norm1 = None
    reshape_4 = getattr_l__mod___blocks___2___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___2___attn_qkv = None
    qkv_2 = reshape_4.permute(2, 0, 3, 1, 4);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_6 = qkv_2[0]
    q_2 = getitem_6 * 0.14433756729740643;  getitem_6 = None
    k_2 = qkv_2[1]
    v_2 = qkv_2[2];  qkv_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_5 = k_2.transpose(-2, -1);  k_2 = None
    attn_10 = q_2 @ transpose_5;  q_2 = transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_11 = attn_10.permute(0, 2, 3, 1);  attn_10 = None
    getattr_l__mod___blocks___2___attn_proj_l = self.getattr_L__mod___blocks___2___attn_proj_l(permute_11);  permute_11 = None
    attn_11 = getattr_l__mod___blocks___2___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___2___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_12 = attn_11.softmax(dim = -1);  attn_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_13 = attn_12.permute(0, 2, 3, 1);  attn_12 = None
    getattr_l__mod___blocks___2___attn_proj_w = self.getattr_L__mod___blocks___2___attn_proj_w(permute_13);  permute_13 = None
    attn_13 = getattr_l__mod___blocks___2___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___2___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_14 = self.getattr_L__mod___blocks___2___attn_attn_drop(attn_13);  attn_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_5 = attn_14 @ v_2;  attn_14 = v_2 = None
    transpose_6 = matmul_5.transpose(1, 2);  matmul_5 = None
    x_28 = transpose_6.reshape(8, 576, 768);  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_29 = self.getattr_L__mod___blocks___2___attn_proj(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_30 = self.getattr_L__mod___blocks___2___attn_proj_drop(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_7 = getattr_l__mod___blocks___2___gamma_1 * x_30;  getattr_l__mod___blocks___2___gamma_1 = x_30 = None
    getattr_l__mod___blocks___2___drop_path = self.getattr_L__mod___blocks___2___drop_path(mul_7);  mul_7 = None
    x_31 = x_27 + getattr_l__mod___blocks___2___drop_path;  x_27 = getattr_l__mod___blocks___2___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___2___gamma_2 = self.getattr_L__mod___blocks___2___gamma_2
    getattr_l__mod___blocks___2___norm2 = self.getattr_L__mod___blocks___2___norm2(x_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_32 = self.getattr_L__mod___blocks___2___mlp_fc1(getattr_l__mod___blocks___2___norm2);  getattr_l__mod___blocks___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_33 = self.getattr_L__mod___blocks___2___mlp_act(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_34 = self.getattr_L__mod___blocks___2___mlp_drop1(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_35 = self.getattr_L__mod___blocks___2___mlp_norm(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_36 = self.getattr_L__mod___blocks___2___mlp_fc2(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_37 = self.getattr_L__mod___blocks___2___mlp_drop2(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_8 = getattr_l__mod___blocks___2___gamma_2 * x_37;  getattr_l__mod___blocks___2___gamma_2 = x_37 = None
    getattr_l__mod___blocks___2___drop_path_1 = self.getattr_L__mod___blocks___2___drop_path(mul_8);  mul_8 = None
    x_38 = x_31 + getattr_l__mod___blocks___2___drop_path_1;  x_31 = getattr_l__mod___blocks___2___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___3___gamma_1 = self.getattr_L__mod___blocks___3___gamma_1
    getattr_l__mod___blocks___3___norm1 = self.getattr_L__mod___blocks___3___norm1(x_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___3___attn_qkv = self.getattr_L__mod___blocks___3___attn_qkv(getattr_l__mod___blocks___3___norm1);  getattr_l__mod___blocks___3___norm1 = None
    reshape_6 = getattr_l__mod___blocks___3___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___3___attn_qkv = None
    qkv_3 = reshape_6.permute(2, 0, 3, 1, 4);  reshape_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_9 = qkv_3[0]
    q_3 = getitem_9 * 0.14433756729740643;  getitem_9 = None
    k_3 = qkv_3[1]
    v_3 = qkv_3[2];  qkv_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_7 = k_3.transpose(-2, -1);  k_3 = None
    attn_15 = q_3 @ transpose_7;  q_3 = transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_16 = attn_15.permute(0, 2, 3, 1);  attn_15 = None
    getattr_l__mod___blocks___3___attn_proj_l = self.getattr_L__mod___blocks___3___attn_proj_l(permute_16);  permute_16 = None
    attn_16 = getattr_l__mod___blocks___3___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___3___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_17 = attn_16.softmax(dim = -1);  attn_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_18 = attn_17.permute(0, 2, 3, 1);  attn_17 = None
    getattr_l__mod___blocks___3___attn_proj_w = self.getattr_L__mod___blocks___3___attn_proj_w(permute_18);  permute_18 = None
    attn_18 = getattr_l__mod___blocks___3___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___3___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_19 = self.getattr_L__mod___blocks___3___attn_attn_drop(attn_18);  attn_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_7 = attn_19 @ v_3;  attn_19 = v_3 = None
    transpose_8 = matmul_7.transpose(1, 2);  matmul_7 = None
    x_39 = transpose_8.reshape(8, 576, 768);  transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_40 = self.getattr_L__mod___blocks___3___attn_proj(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_41 = self.getattr_L__mod___blocks___3___attn_proj_drop(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_10 = getattr_l__mod___blocks___3___gamma_1 * x_41;  getattr_l__mod___blocks___3___gamma_1 = x_41 = None
    getattr_l__mod___blocks___3___drop_path = self.getattr_L__mod___blocks___3___drop_path(mul_10);  mul_10 = None
    x_42 = x_38 + getattr_l__mod___blocks___3___drop_path;  x_38 = getattr_l__mod___blocks___3___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___3___gamma_2 = self.getattr_L__mod___blocks___3___gamma_2
    getattr_l__mod___blocks___3___norm2 = self.getattr_L__mod___blocks___3___norm2(x_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_43 = self.getattr_L__mod___blocks___3___mlp_fc1(getattr_l__mod___blocks___3___norm2);  getattr_l__mod___blocks___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_44 = self.getattr_L__mod___blocks___3___mlp_act(x_43);  x_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_45 = self.getattr_L__mod___blocks___3___mlp_drop1(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_46 = self.getattr_L__mod___blocks___3___mlp_norm(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_47 = self.getattr_L__mod___blocks___3___mlp_fc2(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_48 = self.getattr_L__mod___blocks___3___mlp_drop2(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_11 = getattr_l__mod___blocks___3___gamma_2 * x_48;  getattr_l__mod___blocks___3___gamma_2 = x_48 = None
    getattr_l__mod___blocks___3___drop_path_1 = self.getattr_L__mod___blocks___3___drop_path(mul_11);  mul_11 = None
    x_49 = x_42 + getattr_l__mod___blocks___3___drop_path_1;  x_42 = getattr_l__mod___blocks___3___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___4___gamma_1 = self.getattr_L__mod___blocks___4___gamma_1
    getattr_l__mod___blocks___4___norm1 = self.getattr_L__mod___blocks___4___norm1(x_49)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___4___attn_qkv = self.getattr_L__mod___blocks___4___attn_qkv(getattr_l__mod___blocks___4___norm1);  getattr_l__mod___blocks___4___norm1 = None
    reshape_8 = getattr_l__mod___blocks___4___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___4___attn_qkv = None
    qkv_4 = reshape_8.permute(2, 0, 3, 1, 4);  reshape_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_12 = qkv_4[0]
    q_4 = getitem_12 * 0.14433756729740643;  getitem_12 = None
    k_4 = qkv_4[1]
    v_4 = qkv_4[2];  qkv_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_9 = k_4.transpose(-2, -1);  k_4 = None
    attn_20 = q_4 @ transpose_9;  q_4 = transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_21 = attn_20.permute(0, 2, 3, 1);  attn_20 = None
    getattr_l__mod___blocks___4___attn_proj_l = self.getattr_L__mod___blocks___4___attn_proj_l(permute_21);  permute_21 = None
    attn_21 = getattr_l__mod___blocks___4___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___4___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_22 = attn_21.softmax(dim = -1);  attn_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_23 = attn_22.permute(0, 2, 3, 1);  attn_22 = None
    getattr_l__mod___blocks___4___attn_proj_w = self.getattr_L__mod___blocks___4___attn_proj_w(permute_23);  permute_23 = None
    attn_23 = getattr_l__mod___blocks___4___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___4___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_24 = self.getattr_L__mod___blocks___4___attn_attn_drop(attn_23);  attn_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_9 = attn_24 @ v_4;  attn_24 = v_4 = None
    transpose_10 = matmul_9.transpose(1, 2);  matmul_9 = None
    x_50 = transpose_10.reshape(8, 576, 768);  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_51 = self.getattr_L__mod___blocks___4___attn_proj(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_52 = self.getattr_L__mod___blocks___4___attn_proj_drop(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_13 = getattr_l__mod___blocks___4___gamma_1 * x_52;  getattr_l__mod___blocks___4___gamma_1 = x_52 = None
    getattr_l__mod___blocks___4___drop_path = self.getattr_L__mod___blocks___4___drop_path(mul_13);  mul_13 = None
    x_53 = x_49 + getattr_l__mod___blocks___4___drop_path;  x_49 = getattr_l__mod___blocks___4___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___4___gamma_2 = self.getattr_L__mod___blocks___4___gamma_2
    getattr_l__mod___blocks___4___norm2 = self.getattr_L__mod___blocks___4___norm2(x_53)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_54 = self.getattr_L__mod___blocks___4___mlp_fc1(getattr_l__mod___blocks___4___norm2);  getattr_l__mod___blocks___4___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_55 = self.getattr_L__mod___blocks___4___mlp_act(x_54);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_56 = self.getattr_L__mod___blocks___4___mlp_drop1(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_57 = self.getattr_L__mod___blocks___4___mlp_norm(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_58 = self.getattr_L__mod___blocks___4___mlp_fc2(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_59 = self.getattr_L__mod___blocks___4___mlp_drop2(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_14 = getattr_l__mod___blocks___4___gamma_2 * x_59;  getattr_l__mod___blocks___4___gamma_2 = x_59 = None
    getattr_l__mod___blocks___4___drop_path_1 = self.getattr_L__mod___blocks___4___drop_path(mul_14);  mul_14 = None
    x_60 = x_53 + getattr_l__mod___blocks___4___drop_path_1;  x_53 = getattr_l__mod___blocks___4___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___5___gamma_1 = self.getattr_L__mod___blocks___5___gamma_1
    getattr_l__mod___blocks___5___norm1 = self.getattr_L__mod___blocks___5___norm1(x_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___5___attn_qkv = self.getattr_L__mod___blocks___5___attn_qkv(getattr_l__mod___blocks___5___norm1);  getattr_l__mod___blocks___5___norm1 = None
    reshape_10 = getattr_l__mod___blocks___5___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___5___attn_qkv = None
    qkv_5 = reshape_10.permute(2, 0, 3, 1, 4);  reshape_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_15 = qkv_5[0]
    q_5 = getitem_15 * 0.14433756729740643;  getitem_15 = None
    k_5 = qkv_5[1]
    v_5 = qkv_5[2];  qkv_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_11 = k_5.transpose(-2, -1);  k_5 = None
    attn_25 = q_5 @ transpose_11;  q_5 = transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_26 = attn_25.permute(0, 2, 3, 1);  attn_25 = None
    getattr_l__mod___blocks___5___attn_proj_l = self.getattr_L__mod___blocks___5___attn_proj_l(permute_26);  permute_26 = None
    attn_26 = getattr_l__mod___blocks___5___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___5___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_27 = attn_26.softmax(dim = -1);  attn_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_28 = attn_27.permute(0, 2, 3, 1);  attn_27 = None
    getattr_l__mod___blocks___5___attn_proj_w = self.getattr_L__mod___blocks___5___attn_proj_w(permute_28);  permute_28 = None
    attn_28 = getattr_l__mod___blocks___5___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___5___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_29 = self.getattr_L__mod___blocks___5___attn_attn_drop(attn_28);  attn_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_11 = attn_29 @ v_5;  attn_29 = v_5 = None
    transpose_12 = matmul_11.transpose(1, 2);  matmul_11 = None
    x_61 = transpose_12.reshape(8, 576, 768);  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_62 = self.getattr_L__mod___blocks___5___attn_proj(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_63 = self.getattr_L__mod___blocks___5___attn_proj_drop(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_16 = getattr_l__mod___blocks___5___gamma_1 * x_63;  getattr_l__mod___blocks___5___gamma_1 = x_63 = None
    getattr_l__mod___blocks___5___drop_path = self.getattr_L__mod___blocks___5___drop_path(mul_16);  mul_16 = None
    x_64 = x_60 + getattr_l__mod___blocks___5___drop_path;  x_60 = getattr_l__mod___blocks___5___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___5___gamma_2 = self.getattr_L__mod___blocks___5___gamma_2
    getattr_l__mod___blocks___5___norm2 = self.getattr_L__mod___blocks___5___norm2(x_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_65 = self.getattr_L__mod___blocks___5___mlp_fc1(getattr_l__mod___blocks___5___norm2);  getattr_l__mod___blocks___5___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_66 = self.getattr_L__mod___blocks___5___mlp_act(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_67 = self.getattr_L__mod___blocks___5___mlp_drop1(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_68 = self.getattr_L__mod___blocks___5___mlp_norm(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_69 = self.getattr_L__mod___blocks___5___mlp_fc2(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_70 = self.getattr_L__mod___blocks___5___mlp_drop2(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_17 = getattr_l__mod___blocks___5___gamma_2 * x_70;  getattr_l__mod___blocks___5___gamma_2 = x_70 = None
    getattr_l__mod___blocks___5___drop_path_1 = self.getattr_L__mod___blocks___5___drop_path(mul_17);  mul_17 = None
    x_71 = x_64 + getattr_l__mod___blocks___5___drop_path_1;  x_64 = getattr_l__mod___blocks___5___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___6___gamma_1 = self.getattr_L__mod___blocks___6___gamma_1
    getattr_l__mod___blocks___6___norm1 = self.getattr_L__mod___blocks___6___norm1(x_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___6___attn_qkv = self.getattr_L__mod___blocks___6___attn_qkv(getattr_l__mod___blocks___6___norm1);  getattr_l__mod___blocks___6___norm1 = None
    reshape_12 = getattr_l__mod___blocks___6___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___6___attn_qkv = None
    qkv_6 = reshape_12.permute(2, 0, 3, 1, 4);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_18 = qkv_6[0]
    q_6 = getitem_18 * 0.14433756729740643;  getitem_18 = None
    k_6 = qkv_6[1]
    v_6 = qkv_6[2];  qkv_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_13 = k_6.transpose(-2, -1);  k_6 = None
    attn_30 = q_6 @ transpose_13;  q_6 = transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_31 = attn_30.permute(0, 2, 3, 1);  attn_30 = None
    getattr_l__mod___blocks___6___attn_proj_l = self.getattr_L__mod___blocks___6___attn_proj_l(permute_31);  permute_31 = None
    attn_31 = getattr_l__mod___blocks___6___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___6___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_32 = attn_31.softmax(dim = -1);  attn_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_33 = attn_32.permute(0, 2, 3, 1);  attn_32 = None
    getattr_l__mod___blocks___6___attn_proj_w = self.getattr_L__mod___blocks___6___attn_proj_w(permute_33);  permute_33 = None
    attn_33 = getattr_l__mod___blocks___6___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___6___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_34 = self.getattr_L__mod___blocks___6___attn_attn_drop(attn_33);  attn_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_13 = attn_34 @ v_6;  attn_34 = v_6 = None
    transpose_14 = matmul_13.transpose(1, 2);  matmul_13 = None
    x_72 = transpose_14.reshape(8, 576, 768);  transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_73 = self.getattr_L__mod___blocks___6___attn_proj(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_74 = self.getattr_L__mod___blocks___6___attn_proj_drop(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_19 = getattr_l__mod___blocks___6___gamma_1 * x_74;  getattr_l__mod___blocks___6___gamma_1 = x_74 = None
    getattr_l__mod___blocks___6___drop_path = self.getattr_L__mod___blocks___6___drop_path(mul_19);  mul_19 = None
    x_75 = x_71 + getattr_l__mod___blocks___6___drop_path;  x_71 = getattr_l__mod___blocks___6___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___6___gamma_2 = self.getattr_L__mod___blocks___6___gamma_2
    getattr_l__mod___blocks___6___norm2 = self.getattr_L__mod___blocks___6___norm2(x_75)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_76 = self.getattr_L__mod___blocks___6___mlp_fc1(getattr_l__mod___blocks___6___norm2);  getattr_l__mod___blocks___6___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_77 = self.getattr_L__mod___blocks___6___mlp_act(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_78 = self.getattr_L__mod___blocks___6___mlp_drop1(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_79 = self.getattr_L__mod___blocks___6___mlp_norm(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_80 = self.getattr_L__mod___blocks___6___mlp_fc2(x_79);  x_79 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_81 = self.getattr_L__mod___blocks___6___mlp_drop2(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_20 = getattr_l__mod___blocks___6___gamma_2 * x_81;  getattr_l__mod___blocks___6___gamma_2 = x_81 = None
    getattr_l__mod___blocks___6___drop_path_1 = self.getattr_L__mod___blocks___6___drop_path(mul_20);  mul_20 = None
    x_82 = x_75 + getattr_l__mod___blocks___6___drop_path_1;  x_75 = getattr_l__mod___blocks___6___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___7___gamma_1 = self.getattr_L__mod___blocks___7___gamma_1
    getattr_l__mod___blocks___7___norm1 = self.getattr_L__mod___blocks___7___norm1(x_82)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___7___attn_qkv = self.getattr_L__mod___blocks___7___attn_qkv(getattr_l__mod___blocks___7___norm1);  getattr_l__mod___blocks___7___norm1 = None
    reshape_14 = getattr_l__mod___blocks___7___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___7___attn_qkv = None
    qkv_7 = reshape_14.permute(2, 0, 3, 1, 4);  reshape_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_21 = qkv_7[0]
    q_7 = getitem_21 * 0.14433756729740643;  getitem_21 = None
    k_7 = qkv_7[1]
    v_7 = qkv_7[2];  qkv_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_15 = k_7.transpose(-2, -1);  k_7 = None
    attn_35 = q_7 @ transpose_15;  q_7 = transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_36 = attn_35.permute(0, 2, 3, 1);  attn_35 = None
    getattr_l__mod___blocks___7___attn_proj_l = self.getattr_L__mod___blocks___7___attn_proj_l(permute_36);  permute_36 = None
    attn_36 = getattr_l__mod___blocks___7___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___7___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_37 = attn_36.softmax(dim = -1);  attn_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_38 = attn_37.permute(0, 2, 3, 1);  attn_37 = None
    getattr_l__mod___blocks___7___attn_proj_w = self.getattr_L__mod___blocks___7___attn_proj_w(permute_38);  permute_38 = None
    attn_38 = getattr_l__mod___blocks___7___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___7___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_39 = self.getattr_L__mod___blocks___7___attn_attn_drop(attn_38);  attn_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_15 = attn_39 @ v_7;  attn_39 = v_7 = None
    transpose_16 = matmul_15.transpose(1, 2);  matmul_15 = None
    x_83 = transpose_16.reshape(8, 576, 768);  transpose_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_84 = self.getattr_L__mod___blocks___7___attn_proj(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_85 = self.getattr_L__mod___blocks___7___attn_proj_drop(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_22 = getattr_l__mod___blocks___7___gamma_1 * x_85;  getattr_l__mod___blocks___7___gamma_1 = x_85 = None
    getattr_l__mod___blocks___7___drop_path = self.getattr_L__mod___blocks___7___drop_path(mul_22);  mul_22 = None
    x_86 = x_82 + getattr_l__mod___blocks___7___drop_path;  x_82 = getattr_l__mod___blocks___7___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___7___gamma_2 = self.getattr_L__mod___blocks___7___gamma_2
    getattr_l__mod___blocks___7___norm2 = self.getattr_L__mod___blocks___7___norm2(x_86)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_87 = self.getattr_L__mod___blocks___7___mlp_fc1(getattr_l__mod___blocks___7___norm2);  getattr_l__mod___blocks___7___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_88 = self.getattr_L__mod___blocks___7___mlp_act(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_89 = self.getattr_L__mod___blocks___7___mlp_drop1(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_90 = self.getattr_L__mod___blocks___7___mlp_norm(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_91 = self.getattr_L__mod___blocks___7___mlp_fc2(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_92 = self.getattr_L__mod___blocks___7___mlp_drop2(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_23 = getattr_l__mod___blocks___7___gamma_2 * x_92;  getattr_l__mod___blocks___7___gamma_2 = x_92 = None
    getattr_l__mod___blocks___7___drop_path_1 = self.getattr_L__mod___blocks___7___drop_path(mul_23);  mul_23 = None
    x_93 = x_86 + getattr_l__mod___blocks___7___drop_path_1;  x_86 = getattr_l__mod___blocks___7___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___8___gamma_1 = self.getattr_L__mod___blocks___8___gamma_1
    getattr_l__mod___blocks___8___norm1 = self.getattr_L__mod___blocks___8___norm1(x_93)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___8___attn_qkv = self.getattr_L__mod___blocks___8___attn_qkv(getattr_l__mod___blocks___8___norm1);  getattr_l__mod___blocks___8___norm1 = None
    reshape_16 = getattr_l__mod___blocks___8___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___8___attn_qkv = None
    qkv_8 = reshape_16.permute(2, 0, 3, 1, 4);  reshape_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_24 = qkv_8[0]
    q_8 = getitem_24 * 0.14433756729740643;  getitem_24 = None
    k_8 = qkv_8[1]
    v_8 = qkv_8[2];  qkv_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_17 = k_8.transpose(-2, -1);  k_8 = None
    attn_40 = q_8 @ transpose_17;  q_8 = transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_41 = attn_40.permute(0, 2, 3, 1);  attn_40 = None
    getattr_l__mod___blocks___8___attn_proj_l = self.getattr_L__mod___blocks___8___attn_proj_l(permute_41);  permute_41 = None
    attn_41 = getattr_l__mod___blocks___8___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___8___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_42 = attn_41.softmax(dim = -1);  attn_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_43 = attn_42.permute(0, 2, 3, 1);  attn_42 = None
    getattr_l__mod___blocks___8___attn_proj_w = self.getattr_L__mod___blocks___8___attn_proj_w(permute_43);  permute_43 = None
    attn_43 = getattr_l__mod___blocks___8___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___8___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_44 = self.getattr_L__mod___blocks___8___attn_attn_drop(attn_43);  attn_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_17 = attn_44 @ v_8;  attn_44 = v_8 = None
    transpose_18 = matmul_17.transpose(1, 2);  matmul_17 = None
    x_94 = transpose_18.reshape(8, 576, 768);  transpose_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_95 = self.getattr_L__mod___blocks___8___attn_proj(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_96 = self.getattr_L__mod___blocks___8___attn_proj_drop(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_25 = getattr_l__mod___blocks___8___gamma_1 * x_96;  getattr_l__mod___blocks___8___gamma_1 = x_96 = None
    getattr_l__mod___blocks___8___drop_path = self.getattr_L__mod___blocks___8___drop_path(mul_25);  mul_25 = None
    x_97 = x_93 + getattr_l__mod___blocks___8___drop_path;  x_93 = getattr_l__mod___blocks___8___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___8___gamma_2 = self.getattr_L__mod___blocks___8___gamma_2
    getattr_l__mod___blocks___8___norm2 = self.getattr_L__mod___blocks___8___norm2(x_97)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_98 = self.getattr_L__mod___blocks___8___mlp_fc1(getattr_l__mod___blocks___8___norm2);  getattr_l__mod___blocks___8___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_99 = self.getattr_L__mod___blocks___8___mlp_act(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_100 = self.getattr_L__mod___blocks___8___mlp_drop1(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_101 = self.getattr_L__mod___blocks___8___mlp_norm(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_102 = self.getattr_L__mod___blocks___8___mlp_fc2(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_103 = self.getattr_L__mod___blocks___8___mlp_drop2(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_26 = getattr_l__mod___blocks___8___gamma_2 * x_103;  getattr_l__mod___blocks___8___gamma_2 = x_103 = None
    getattr_l__mod___blocks___8___drop_path_1 = self.getattr_L__mod___blocks___8___drop_path(mul_26);  mul_26 = None
    x_104 = x_97 + getattr_l__mod___blocks___8___drop_path_1;  x_97 = getattr_l__mod___blocks___8___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___9___gamma_1 = self.getattr_L__mod___blocks___9___gamma_1
    getattr_l__mod___blocks___9___norm1 = self.getattr_L__mod___blocks___9___norm1(x_104)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___9___attn_qkv = self.getattr_L__mod___blocks___9___attn_qkv(getattr_l__mod___blocks___9___norm1);  getattr_l__mod___blocks___9___norm1 = None
    reshape_18 = getattr_l__mod___blocks___9___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___9___attn_qkv = None
    qkv_9 = reshape_18.permute(2, 0, 3, 1, 4);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_27 = qkv_9[0]
    q_9 = getitem_27 * 0.14433756729740643;  getitem_27 = None
    k_9 = qkv_9[1]
    v_9 = qkv_9[2];  qkv_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_19 = k_9.transpose(-2, -1);  k_9 = None
    attn_45 = q_9 @ transpose_19;  q_9 = transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_46 = attn_45.permute(0, 2, 3, 1);  attn_45 = None
    getattr_l__mod___blocks___9___attn_proj_l = self.getattr_L__mod___blocks___9___attn_proj_l(permute_46);  permute_46 = None
    attn_46 = getattr_l__mod___blocks___9___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___9___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_47 = attn_46.softmax(dim = -1);  attn_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_48 = attn_47.permute(0, 2, 3, 1);  attn_47 = None
    getattr_l__mod___blocks___9___attn_proj_w = self.getattr_L__mod___blocks___9___attn_proj_w(permute_48);  permute_48 = None
    attn_48 = getattr_l__mod___blocks___9___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___9___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_49 = self.getattr_L__mod___blocks___9___attn_attn_drop(attn_48);  attn_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_19 = attn_49 @ v_9;  attn_49 = v_9 = None
    transpose_20 = matmul_19.transpose(1, 2);  matmul_19 = None
    x_105 = transpose_20.reshape(8, 576, 768);  transpose_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_106 = self.getattr_L__mod___blocks___9___attn_proj(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_107 = self.getattr_L__mod___blocks___9___attn_proj_drop(x_106);  x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_28 = getattr_l__mod___blocks___9___gamma_1 * x_107;  getattr_l__mod___blocks___9___gamma_1 = x_107 = None
    getattr_l__mod___blocks___9___drop_path = self.getattr_L__mod___blocks___9___drop_path(mul_28);  mul_28 = None
    x_108 = x_104 + getattr_l__mod___blocks___9___drop_path;  x_104 = getattr_l__mod___blocks___9___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___9___gamma_2 = self.getattr_L__mod___blocks___9___gamma_2
    getattr_l__mod___blocks___9___norm2 = self.getattr_L__mod___blocks___9___norm2(x_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_109 = self.getattr_L__mod___blocks___9___mlp_fc1(getattr_l__mod___blocks___9___norm2);  getattr_l__mod___blocks___9___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_110 = self.getattr_L__mod___blocks___9___mlp_act(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_111 = self.getattr_L__mod___blocks___9___mlp_drop1(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_112 = self.getattr_L__mod___blocks___9___mlp_norm(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_113 = self.getattr_L__mod___blocks___9___mlp_fc2(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_114 = self.getattr_L__mod___blocks___9___mlp_drop2(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_29 = getattr_l__mod___blocks___9___gamma_2 * x_114;  getattr_l__mod___blocks___9___gamma_2 = x_114 = None
    getattr_l__mod___blocks___9___drop_path_1 = self.getattr_L__mod___blocks___9___drop_path(mul_29);  mul_29 = None
    x_115 = x_108 + getattr_l__mod___blocks___9___drop_path_1;  x_108 = getattr_l__mod___blocks___9___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___10___gamma_1 = self.getattr_L__mod___blocks___10___gamma_1
    getattr_l__mod___blocks___10___norm1 = self.getattr_L__mod___blocks___10___norm1(x_115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___10___attn_qkv = self.getattr_L__mod___blocks___10___attn_qkv(getattr_l__mod___blocks___10___norm1);  getattr_l__mod___blocks___10___norm1 = None
    reshape_20 = getattr_l__mod___blocks___10___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___10___attn_qkv = None
    qkv_10 = reshape_20.permute(2, 0, 3, 1, 4);  reshape_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_30 = qkv_10[0]
    q_10 = getitem_30 * 0.14433756729740643;  getitem_30 = None
    k_10 = qkv_10[1]
    v_10 = qkv_10[2];  qkv_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_21 = k_10.transpose(-2, -1);  k_10 = None
    attn_50 = q_10 @ transpose_21;  q_10 = transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_51 = attn_50.permute(0, 2, 3, 1);  attn_50 = None
    getattr_l__mod___blocks___10___attn_proj_l = self.getattr_L__mod___blocks___10___attn_proj_l(permute_51);  permute_51 = None
    attn_51 = getattr_l__mod___blocks___10___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___10___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_52 = attn_51.softmax(dim = -1);  attn_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_53 = attn_52.permute(0, 2, 3, 1);  attn_52 = None
    getattr_l__mod___blocks___10___attn_proj_w = self.getattr_L__mod___blocks___10___attn_proj_w(permute_53);  permute_53 = None
    attn_53 = getattr_l__mod___blocks___10___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___10___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_54 = self.getattr_L__mod___blocks___10___attn_attn_drop(attn_53);  attn_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_21 = attn_54 @ v_10;  attn_54 = v_10 = None
    transpose_22 = matmul_21.transpose(1, 2);  matmul_21 = None
    x_116 = transpose_22.reshape(8, 576, 768);  transpose_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_117 = self.getattr_L__mod___blocks___10___attn_proj(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_118 = self.getattr_L__mod___blocks___10___attn_proj_drop(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_31 = getattr_l__mod___blocks___10___gamma_1 * x_118;  getattr_l__mod___blocks___10___gamma_1 = x_118 = None
    getattr_l__mod___blocks___10___drop_path = self.getattr_L__mod___blocks___10___drop_path(mul_31);  mul_31 = None
    x_119 = x_115 + getattr_l__mod___blocks___10___drop_path;  x_115 = getattr_l__mod___blocks___10___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___10___gamma_2 = self.getattr_L__mod___blocks___10___gamma_2
    getattr_l__mod___blocks___10___norm2 = self.getattr_L__mod___blocks___10___norm2(x_119)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_120 = self.getattr_L__mod___blocks___10___mlp_fc1(getattr_l__mod___blocks___10___norm2);  getattr_l__mod___blocks___10___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_121 = self.getattr_L__mod___blocks___10___mlp_act(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_122 = self.getattr_L__mod___blocks___10___mlp_drop1(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_123 = self.getattr_L__mod___blocks___10___mlp_norm(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_124 = self.getattr_L__mod___blocks___10___mlp_fc2(x_123);  x_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_125 = self.getattr_L__mod___blocks___10___mlp_drop2(x_124);  x_124 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_32 = getattr_l__mod___blocks___10___gamma_2 * x_125;  getattr_l__mod___blocks___10___gamma_2 = x_125 = None
    getattr_l__mod___blocks___10___drop_path_1 = self.getattr_L__mod___blocks___10___drop_path(mul_32);  mul_32 = None
    x_126 = x_119 + getattr_l__mod___blocks___10___drop_path_1;  x_119 = getattr_l__mod___blocks___10___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___11___gamma_1 = self.getattr_L__mod___blocks___11___gamma_1
    getattr_l__mod___blocks___11___norm1 = self.getattr_L__mod___blocks___11___norm1(x_126)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___11___attn_qkv = self.getattr_L__mod___blocks___11___attn_qkv(getattr_l__mod___blocks___11___norm1);  getattr_l__mod___blocks___11___norm1 = None
    reshape_22 = getattr_l__mod___blocks___11___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___11___attn_qkv = None
    qkv_11 = reshape_22.permute(2, 0, 3, 1, 4);  reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_33 = qkv_11[0]
    q_11 = getitem_33 * 0.14433756729740643;  getitem_33 = None
    k_11 = qkv_11[1]
    v_11 = qkv_11[2];  qkv_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_23 = k_11.transpose(-2, -1);  k_11 = None
    attn_55 = q_11 @ transpose_23;  q_11 = transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_56 = attn_55.permute(0, 2, 3, 1);  attn_55 = None
    getattr_l__mod___blocks___11___attn_proj_l = self.getattr_L__mod___blocks___11___attn_proj_l(permute_56);  permute_56 = None
    attn_56 = getattr_l__mod___blocks___11___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___11___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_57 = attn_56.softmax(dim = -1);  attn_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_58 = attn_57.permute(0, 2, 3, 1);  attn_57 = None
    getattr_l__mod___blocks___11___attn_proj_w = self.getattr_L__mod___blocks___11___attn_proj_w(permute_58);  permute_58 = None
    attn_58 = getattr_l__mod___blocks___11___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___11___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_59 = self.getattr_L__mod___blocks___11___attn_attn_drop(attn_58);  attn_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_23 = attn_59 @ v_11;  attn_59 = v_11 = None
    transpose_24 = matmul_23.transpose(1, 2);  matmul_23 = None
    x_127 = transpose_24.reshape(8, 576, 768);  transpose_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_128 = self.getattr_L__mod___blocks___11___attn_proj(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_129 = self.getattr_L__mod___blocks___11___attn_proj_drop(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_34 = getattr_l__mod___blocks___11___gamma_1 * x_129;  getattr_l__mod___blocks___11___gamma_1 = x_129 = None
    getattr_l__mod___blocks___11___drop_path = self.getattr_L__mod___blocks___11___drop_path(mul_34);  mul_34 = None
    x_130 = x_126 + getattr_l__mod___blocks___11___drop_path;  x_126 = getattr_l__mod___blocks___11___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___11___gamma_2 = self.getattr_L__mod___blocks___11___gamma_2
    getattr_l__mod___blocks___11___norm2 = self.getattr_L__mod___blocks___11___norm2(x_130)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_131 = self.getattr_L__mod___blocks___11___mlp_fc1(getattr_l__mod___blocks___11___norm2);  getattr_l__mod___blocks___11___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_132 = self.getattr_L__mod___blocks___11___mlp_act(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_133 = self.getattr_L__mod___blocks___11___mlp_drop1(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_134 = self.getattr_L__mod___blocks___11___mlp_norm(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_135 = self.getattr_L__mod___blocks___11___mlp_fc2(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_136 = self.getattr_L__mod___blocks___11___mlp_drop2(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_35 = getattr_l__mod___blocks___11___gamma_2 * x_136;  getattr_l__mod___blocks___11___gamma_2 = x_136 = None
    getattr_l__mod___blocks___11___drop_path_1 = self.getattr_L__mod___blocks___11___drop_path(mul_35);  mul_35 = None
    x_137 = x_130 + getattr_l__mod___blocks___11___drop_path_1;  x_130 = getattr_l__mod___blocks___11___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___12___gamma_1 = self.getattr_L__mod___blocks___12___gamma_1
    getattr_l__mod___blocks___12___norm1 = self.getattr_L__mod___blocks___12___norm1(x_137)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___12___attn_qkv = self.getattr_L__mod___blocks___12___attn_qkv(getattr_l__mod___blocks___12___norm1);  getattr_l__mod___blocks___12___norm1 = None
    reshape_24 = getattr_l__mod___blocks___12___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___12___attn_qkv = None
    qkv_12 = reshape_24.permute(2, 0, 3, 1, 4);  reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_36 = qkv_12[0]
    q_12 = getitem_36 * 0.14433756729740643;  getitem_36 = None
    k_12 = qkv_12[1]
    v_12 = qkv_12[2];  qkv_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_25 = k_12.transpose(-2, -1);  k_12 = None
    attn_60 = q_12 @ transpose_25;  q_12 = transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_61 = attn_60.permute(0, 2, 3, 1);  attn_60 = None
    getattr_l__mod___blocks___12___attn_proj_l = self.getattr_L__mod___blocks___12___attn_proj_l(permute_61);  permute_61 = None
    attn_61 = getattr_l__mod___blocks___12___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___12___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_62 = attn_61.softmax(dim = -1);  attn_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_63 = attn_62.permute(0, 2, 3, 1);  attn_62 = None
    getattr_l__mod___blocks___12___attn_proj_w = self.getattr_L__mod___blocks___12___attn_proj_w(permute_63);  permute_63 = None
    attn_63 = getattr_l__mod___blocks___12___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___12___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_64 = self.getattr_L__mod___blocks___12___attn_attn_drop(attn_63);  attn_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_25 = attn_64 @ v_12;  attn_64 = v_12 = None
    transpose_26 = matmul_25.transpose(1, 2);  matmul_25 = None
    x_138 = transpose_26.reshape(8, 576, 768);  transpose_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_139 = self.getattr_L__mod___blocks___12___attn_proj(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_140 = self.getattr_L__mod___blocks___12___attn_proj_drop(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_37 = getattr_l__mod___blocks___12___gamma_1 * x_140;  getattr_l__mod___blocks___12___gamma_1 = x_140 = None
    getattr_l__mod___blocks___12___drop_path = self.getattr_L__mod___blocks___12___drop_path(mul_37);  mul_37 = None
    x_141 = x_137 + getattr_l__mod___blocks___12___drop_path;  x_137 = getattr_l__mod___blocks___12___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___12___gamma_2 = self.getattr_L__mod___blocks___12___gamma_2
    getattr_l__mod___blocks___12___norm2 = self.getattr_L__mod___blocks___12___norm2(x_141)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_142 = self.getattr_L__mod___blocks___12___mlp_fc1(getattr_l__mod___blocks___12___norm2);  getattr_l__mod___blocks___12___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_143 = self.getattr_L__mod___blocks___12___mlp_act(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_144 = self.getattr_L__mod___blocks___12___mlp_drop1(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_145 = self.getattr_L__mod___blocks___12___mlp_norm(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_146 = self.getattr_L__mod___blocks___12___mlp_fc2(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_147 = self.getattr_L__mod___blocks___12___mlp_drop2(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_38 = getattr_l__mod___blocks___12___gamma_2 * x_147;  getattr_l__mod___blocks___12___gamma_2 = x_147 = None
    getattr_l__mod___blocks___12___drop_path_1 = self.getattr_L__mod___blocks___12___drop_path(mul_38);  mul_38 = None
    x_148 = x_141 + getattr_l__mod___blocks___12___drop_path_1;  x_141 = getattr_l__mod___blocks___12___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___13___gamma_1 = self.getattr_L__mod___blocks___13___gamma_1
    getattr_l__mod___blocks___13___norm1 = self.getattr_L__mod___blocks___13___norm1(x_148)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___13___attn_qkv = self.getattr_L__mod___blocks___13___attn_qkv(getattr_l__mod___blocks___13___norm1);  getattr_l__mod___blocks___13___norm1 = None
    reshape_26 = getattr_l__mod___blocks___13___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___13___attn_qkv = None
    qkv_13 = reshape_26.permute(2, 0, 3, 1, 4);  reshape_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_39 = qkv_13[0]
    q_13 = getitem_39 * 0.14433756729740643;  getitem_39 = None
    k_13 = qkv_13[1]
    v_13 = qkv_13[2];  qkv_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_27 = k_13.transpose(-2, -1);  k_13 = None
    attn_65 = q_13 @ transpose_27;  q_13 = transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_66 = attn_65.permute(0, 2, 3, 1);  attn_65 = None
    getattr_l__mod___blocks___13___attn_proj_l = self.getattr_L__mod___blocks___13___attn_proj_l(permute_66);  permute_66 = None
    attn_66 = getattr_l__mod___blocks___13___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___13___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_67 = attn_66.softmax(dim = -1);  attn_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_68 = attn_67.permute(0, 2, 3, 1);  attn_67 = None
    getattr_l__mod___blocks___13___attn_proj_w = self.getattr_L__mod___blocks___13___attn_proj_w(permute_68);  permute_68 = None
    attn_68 = getattr_l__mod___blocks___13___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___13___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_69 = self.getattr_L__mod___blocks___13___attn_attn_drop(attn_68);  attn_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_27 = attn_69 @ v_13;  attn_69 = v_13 = None
    transpose_28 = matmul_27.transpose(1, 2);  matmul_27 = None
    x_149 = transpose_28.reshape(8, 576, 768);  transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_150 = self.getattr_L__mod___blocks___13___attn_proj(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_151 = self.getattr_L__mod___blocks___13___attn_proj_drop(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_40 = getattr_l__mod___blocks___13___gamma_1 * x_151;  getattr_l__mod___blocks___13___gamma_1 = x_151 = None
    getattr_l__mod___blocks___13___drop_path = self.getattr_L__mod___blocks___13___drop_path(mul_40);  mul_40 = None
    x_152 = x_148 + getattr_l__mod___blocks___13___drop_path;  x_148 = getattr_l__mod___blocks___13___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___13___gamma_2 = self.getattr_L__mod___blocks___13___gamma_2
    getattr_l__mod___blocks___13___norm2 = self.getattr_L__mod___blocks___13___norm2(x_152)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_153 = self.getattr_L__mod___blocks___13___mlp_fc1(getattr_l__mod___blocks___13___norm2);  getattr_l__mod___blocks___13___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_154 = self.getattr_L__mod___blocks___13___mlp_act(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_155 = self.getattr_L__mod___blocks___13___mlp_drop1(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_156 = self.getattr_L__mod___blocks___13___mlp_norm(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_157 = self.getattr_L__mod___blocks___13___mlp_fc2(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_158 = self.getattr_L__mod___blocks___13___mlp_drop2(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_41 = getattr_l__mod___blocks___13___gamma_2 * x_158;  getattr_l__mod___blocks___13___gamma_2 = x_158 = None
    getattr_l__mod___blocks___13___drop_path_1 = self.getattr_L__mod___blocks___13___drop_path(mul_41);  mul_41 = None
    x_159 = x_152 + getattr_l__mod___blocks___13___drop_path_1;  x_152 = getattr_l__mod___blocks___13___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___14___gamma_1 = self.getattr_L__mod___blocks___14___gamma_1
    getattr_l__mod___blocks___14___norm1 = self.getattr_L__mod___blocks___14___norm1(x_159)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___14___attn_qkv = self.getattr_L__mod___blocks___14___attn_qkv(getattr_l__mod___blocks___14___norm1);  getattr_l__mod___blocks___14___norm1 = None
    reshape_28 = getattr_l__mod___blocks___14___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___14___attn_qkv = None
    qkv_14 = reshape_28.permute(2, 0, 3, 1, 4);  reshape_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_42 = qkv_14[0]
    q_14 = getitem_42 * 0.14433756729740643;  getitem_42 = None
    k_14 = qkv_14[1]
    v_14 = qkv_14[2];  qkv_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_29 = k_14.transpose(-2, -1);  k_14 = None
    attn_70 = q_14 @ transpose_29;  q_14 = transpose_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_71 = attn_70.permute(0, 2, 3, 1);  attn_70 = None
    getattr_l__mod___blocks___14___attn_proj_l = self.getattr_L__mod___blocks___14___attn_proj_l(permute_71);  permute_71 = None
    attn_71 = getattr_l__mod___blocks___14___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___14___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_72 = attn_71.softmax(dim = -1);  attn_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_73 = attn_72.permute(0, 2, 3, 1);  attn_72 = None
    getattr_l__mod___blocks___14___attn_proj_w = self.getattr_L__mod___blocks___14___attn_proj_w(permute_73);  permute_73 = None
    attn_73 = getattr_l__mod___blocks___14___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___14___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_74 = self.getattr_L__mod___blocks___14___attn_attn_drop(attn_73);  attn_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_29 = attn_74 @ v_14;  attn_74 = v_14 = None
    transpose_30 = matmul_29.transpose(1, 2);  matmul_29 = None
    x_160 = transpose_30.reshape(8, 576, 768);  transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_161 = self.getattr_L__mod___blocks___14___attn_proj(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_162 = self.getattr_L__mod___blocks___14___attn_proj_drop(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_43 = getattr_l__mod___blocks___14___gamma_1 * x_162;  getattr_l__mod___blocks___14___gamma_1 = x_162 = None
    getattr_l__mod___blocks___14___drop_path = self.getattr_L__mod___blocks___14___drop_path(mul_43);  mul_43 = None
    x_163 = x_159 + getattr_l__mod___blocks___14___drop_path;  x_159 = getattr_l__mod___blocks___14___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___14___gamma_2 = self.getattr_L__mod___blocks___14___gamma_2
    getattr_l__mod___blocks___14___norm2 = self.getattr_L__mod___blocks___14___norm2(x_163)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_164 = self.getattr_L__mod___blocks___14___mlp_fc1(getattr_l__mod___blocks___14___norm2);  getattr_l__mod___blocks___14___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_165 = self.getattr_L__mod___blocks___14___mlp_act(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_166 = self.getattr_L__mod___blocks___14___mlp_drop1(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_167 = self.getattr_L__mod___blocks___14___mlp_norm(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_168 = self.getattr_L__mod___blocks___14___mlp_fc2(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_169 = self.getattr_L__mod___blocks___14___mlp_drop2(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_44 = getattr_l__mod___blocks___14___gamma_2 * x_169;  getattr_l__mod___blocks___14___gamma_2 = x_169 = None
    getattr_l__mod___blocks___14___drop_path_1 = self.getattr_L__mod___blocks___14___drop_path(mul_44);  mul_44 = None
    x_170 = x_163 + getattr_l__mod___blocks___14___drop_path_1;  x_163 = getattr_l__mod___blocks___14___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___15___gamma_1 = self.getattr_L__mod___blocks___15___gamma_1
    getattr_l__mod___blocks___15___norm1 = self.getattr_L__mod___blocks___15___norm1(x_170)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___15___attn_qkv = self.getattr_L__mod___blocks___15___attn_qkv(getattr_l__mod___blocks___15___norm1);  getattr_l__mod___blocks___15___norm1 = None
    reshape_30 = getattr_l__mod___blocks___15___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___15___attn_qkv = None
    qkv_15 = reshape_30.permute(2, 0, 3, 1, 4);  reshape_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_45 = qkv_15[0]
    q_15 = getitem_45 * 0.14433756729740643;  getitem_45 = None
    k_15 = qkv_15[1]
    v_15 = qkv_15[2];  qkv_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_31 = k_15.transpose(-2, -1);  k_15 = None
    attn_75 = q_15 @ transpose_31;  q_15 = transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_76 = attn_75.permute(0, 2, 3, 1);  attn_75 = None
    getattr_l__mod___blocks___15___attn_proj_l = self.getattr_L__mod___blocks___15___attn_proj_l(permute_76);  permute_76 = None
    attn_76 = getattr_l__mod___blocks___15___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___15___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_77 = attn_76.softmax(dim = -1);  attn_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_78 = attn_77.permute(0, 2, 3, 1);  attn_77 = None
    getattr_l__mod___blocks___15___attn_proj_w = self.getattr_L__mod___blocks___15___attn_proj_w(permute_78);  permute_78 = None
    attn_78 = getattr_l__mod___blocks___15___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___15___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_79 = self.getattr_L__mod___blocks___15___attn_attn_drop(attn_78);  attn_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_31 = attn_79 @ v_15;  attn_79 = v_15 = None
    transpose_32 = matmul_31.transpose(1, 2);  matmul_31 = None
    x_171 = transpose_32.reshape(8, 576, 768);  transpose_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_172 = self.getattr_L__mod___blocks___15___attn_proj(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_173 = self.getattr_L__mod___blocks___15___attn_proj_drop(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_46 = getattr_l__mod___blocks___15___gamma_1 * x_173;  getattr_l__mod___blocks___15___gamma_1 = x_173 = None
    getattr_l__mod___blocks___15___drop_path = self.getattr_L__mod___blocks___15___drop_path(mul_46);  mul_46 = None
    x_174 = x_170 + getattr_l__mod___blocks___15___drop_path;  x_170 = getattr_l__mod___blocks___15___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___15___gamma_2 = self.getattr_L__mod___blocks___15___gamma_2
    getattr_l__mod___blocks___15___norm2 = self.getattr_L__mod___blocks___15___norm2(x_174)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_175 = self.getattr_L__mod___blocks___15___mlp_fc1(getattr_l__mod___blocks___15___norm2);  getattr_l__mod___blocks___15___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_176 = self.getattr_L__mod___blocks___15___mlp_act(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_177 = self.getattr_L__mod___blocks___15___mlp_drop1(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_178 = self.getattr_L__mod___blocks___15___mlp_norm(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_179 = self.getattr_L__mod___blocks___15___mlp_fc2(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_180 = self.getattr_L__mod___blocks___15___mlp_drop2(x_179);  x_179 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_47 = getattr_l__mod___blocks___15___gamma_2 * x_180;  getattr_l__mod___blocks___15___gamma_2 = x_180 = None
    getattr_l__mod___blocks___15___drop_path_1 = self.getattr_L__mod___blocks___15___drop_path(mul_47);  mul_47 = None
    x_181 = x_174 + getattr_l__mod___blocks___15___drop_path_1;  x_174 = getattr_l__mod___blocks___15___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___16___gamma_1 = self.getattr_L__mod___blocks___16___gamma_1
    getattr_l__mod___blocks___16___norm1 = self.getattr_L__mod___blocks___16___norm1(x_181)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___16___attn_qkv = self.getattr_L__mod___blocks___16___attn_qkv(getattr_l__mod___blocks___16___norm1);  getattr_l__mod___blocks___16___norm1 = None
    reshape_32 = getattr_l__mod___blocks___16___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___16___attn_qkv = None
    qkv_16 = reshape_32.permute(2, 0, 3, 1, 4);  reshape_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_48 = qkv_16[0]
    q_16 = getitem_48 * 0.14433756729740643;  getitem_48 = None
    k_16 = qkv_16[1]
    v_16 = qkv_16[2];  qkv_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_33 = k_16.transpose(-2, -1);  k_16 = None
    attn_80 = q_16 @ transpose_33;  q_16 = transpose_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_81 = attn_80.permute(0, 2, 3, 1);  attn_80 = None
    getattr_l__mod___blocks___16___attn_proj_l = self.getattr_L__mod___blocks___16___attn_proj_l(permute_81);  permute_81 = None
    attn_81 = getattr_l__mod___blocks___16___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___16___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_82 = attn_81.softmax(dim = -1);  attn_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_83 = attn_82.permute(0, 2, 3, 1);  attn_82 = None
    getattr_l__mod___blocks___16___attn_proj_w = self.getattr_L__mod___blocks___16___attn_proj_w(permute_83);  permute_83 = None
    attn_83 = getattr_l__mod___blocks___16___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___16___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_84 = self.getattr_L__mod___blocks___16___attn_attn_drop(attn_83);  attn_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_33 = attn_84 @ v_16;  attn_84 = v_16 = None
    transpose_34 = matmul_33.transpose(1, 2);  matmul_33 = None
    x_182 = transpose_34.reshape(8, 576, 768);  transpose_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_183 = self.getattr_L__mod___blocks___16___attn_proj(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_184 = self.getattr_L__mod___blocks___16___attn_proj_drop(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_49 = getattr_l__mod___blocks___16___gamma_1 * x_184;  getattr_l__mod___blocks___16___gamma_1 = x_184 = None
    getattr_l__mod___blocks___16___drop_path = self.getattr_L__mod___blocks___16___drop_path(mul_49);  mul_49 = None
    x_185 = x_181 + getattr_l__mod___blocks___16___drop_path;  x_181 = getattr_l__mod___blocks___16___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___16___gamma_2 = self.getattr_L__mod___blocks___16___gamma_2
    getattr_l__mod___blocks___16___norm2 = self.getattr_L__mod___blocks___16___norm2(x_185)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_186 = self.getattr_L__mod___blocks___16___mlp_fc1(getattr_l__mod___blocks___16___norm2);  getattr_l__mod___blocks___16___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_187 = self.getattr_L__mod___blocks___16___mlp_act(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_188 = self.getattr_L__mod___blocks___16___mlp_drop1(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_189 = self.getattr_L__mod___blocks___16___mlp_norm(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_190 = self.getattr_L__mod___blocks___16___mlp_fc2(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_191 = self.getattr_L__mod___blocks___16___mlp_drop2(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_50 = getattr_l__mod___blocks___16___gamma_2 * x_191;  getattr_l__mod___blocks___16___gamma_2 = x_191 = None
    getattr_l__mod___blocks___16___drop_path_1 = self.getattr_L__mod___blocks___16___drop_path(mul_50);  mul_50 = None
    x_192 = x_185 + getattr_l__mod___blocks___16___drop_path_1;  x_185 = getattr_l__mod___blocks___16___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___17___gamma_1 = self.getattr_L__mod___blocks___17___gamma_1
    getattr_l__mod___blocks___17___norm1 = self.getattr_L__mod___blocks___17___norm1(x_192)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___17___attn_qkv = self.getattr_L__mod___blocks___17___attn_qkv(getattr_l__mod___blocks___17___norm1);  getattr_l__mod___blocks___17___norm1 = None
    reshape_34 = getattr_l__mod___blocks___17___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___17___attn_qkv = None
    qkv_17 = reshape_34.permute(2, 0, 3, 1, 4);  reshape_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_51 = qkv_17[0]
    q_17 = getitem_51 * 0.14433756729740643;  getitem_51 = None
    k_17 = qkv_17[1]
    v_17 = qkv_17[2];  qkv_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_35 = k_17.transpose(-2, -1);  k_17 = None
    attn_85 = q_17 @ transpose_35;  q_17 = transpose_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_86 = attn_85.permute(0, 2, 3, 1);  attn_85 = None
    getattr_l__mod___blocks___17___attn_proj_l = self.getattr_L__mod___blocks___17___attn_proj_l(permute_86);  permute_86 = None
    attn_86 = getattr_l__mod___blocks___17___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___17___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_87 = attn_86.softmax(dim = -1);  attn_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_88 = attn_87.permute(0, 2, 3, 1);  attn_87 = None
    getattr_l__mod___blocks___17___attn_proj_w = self.getattr_L__mod___blocks___17___attn_proj_w(permute_88);  permute_88 = None
    attn_88 = getattr_l__mod___blocks___17___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___17___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_89 = self.getattr_L__mod___blocks___17___attn_attn_drop(attn_88);  attn_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_35 = attn_89 @ v_17;  attn_89 = v_17 = None
    transpose_36 = matmul_35.transpose(1, 2);  matmul_35 = None
    x_193 = transpose_36.reshape(8, 576, 768);  transpose_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_194 = self.getattr_L__mod___blocks___17___attn_proj(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_195 = self.getattr_L__mod___blocks___17___attn_proj_drop(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_52 = getattr_l__mod___blocks___17___gamma_1 * x_195;  getattr_l__mod___blocks___17___gamma_1 = x_195 = None
    getattr_l__mod___blocks___17___drop_path = self.getattr_L__mod___blocks___17___drop_path(mul_52);  mul_52 = None
    x_196 = x_192 + getattr_l__mod___blocks___17___drop_path;  x_192 = getattr_l__mod___blocks___17___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___17___gamma_2 = self.getattr_L__mod___blocks___17___gamma_2
    getattr_l__mod___blocks___17___norm2 = self.getattr_L__mod___blocks___17___norm2(x_196)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_197 = self.getattr_L__mod___blocks___17___mlp_fc1(getattr_l__mod___blocks___17___norm2);  getattr_l__mod___blocks___17___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_198 = self.getattr_L__mod___blocks___17___mlp_act(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_199 = self.getattr_L__mod___blocks___17___mlp_drop1(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_200 = self.getattr_L__mod___blocks___17___mlp_norm(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_201 = self.getattr_L__mod___blocks___17___mlp_fc2(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_202 = self.getattr_L__mod___blocks___17___mlp_drop2(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_53 = getattr_l__mod___blocks___17___gamma_2 * x_202;  getattr_l__mod___blocks___17___gamma_2 = x_202 = None
    getattr_l__mod___blocks___17___drop_path_1 = self.getattr_L__mod___blocks___17___drop_path(mul_53);  mul_53 = None
    x_203 = x_196 + getattr_l__mod___blocks___17___drop_path_1;  x_196 = getattr_l__mod___blocks___17___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___18___gamma_1 = self.getattr_L__mod___blocks___18___gamma_1
    getattr_l__mod___blocks___18___norm1 = self.getattr_L__mod___blocks___18___norm1(x_203)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___18___attn_qkv = self.getattr_L__mod___blocks___18___attn_qkv(getattr_l__mod___blocks___18___norm1);  getattr_l__mod___blocks___18___norm1 = None
    reshape_36 = getattr_l__mod___blocks___18___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___18___attn_qkv = None
    qkv_18 = reshape_36.permute(2, 0, 3, 1, 4);  reshape_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_54 = qkv_18[0]
    q_18 = getitem_54 * 0.14433756729740643;  getitem_54 = None
    k_18 = qkv_18[1]
    v_18 = qkv_18[2];  qkv_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_37 = k_18.transpose(-2, -1);  k_18 = None
    attn_90 = q_18 @ transpose_37;  q_18 = transpose_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_91 = attn_90.permute(0, 2, 3, 1);  attn_90 = None
    getattr_l__mod___blocks___18___attn_proj_l = self.getattr_L__mod___blocks___18___attn_proj_l(permute_91);  permute_91 = None
    attn_91 = getattr_l__mod___blocks___18___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___18___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_92 = attn_91.softmax(dim = -1);  attn_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_93 = attn_92.permute(0, 2, 3, 1);  attn_92 = None
    getattr_l__mod___blocks___18___attn_proj_w = self.getattr_L__mod___blocks___18___attn_proj_w(permute_93);  permute_93 = None
    attn_93 = getattr_l__mod___blocks___18___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___18___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_94 = self.getattr_L__mod___blocks___18___attn_attn_drop(attn_93);  attn_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_37 = attn_94 @ v_18;  attn_94 = v_18 = None
    transpose_38 = matmul_37.transpose(1, 2);  matmul_37 = None
    x_204 = transpose_38.reshape(8, 576, 768);  transpose_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_205 = self.getattr_L__mod___blocks___18___attn_proj(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_206 = self.getattr_L__mod___blocks___18___attn_proj_drop(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_55 = getattr_l__mod___blocks___18___gamma_1 * x_206;  getattr_l__mod___blocks___18___gamma_1 = x_206 = None
    getattr_l__mod___blocks___18___drop_path = self.getattr_L__mod___blocks___18___drop_path(mul_55);  mul_55 = None
    x_207 = x_203 + getattr_l__mod___blocks___18___drop_path;  x_203 = getattr_l__mod___blocks___18___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___18___gamma_2 = self.getattr_L__mod___blocks___18___gamma_2
    getattr_l__mod___blocks___18___norm2 = self.getattr_L__mod___blocks___18___norm2(x_207)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_208 = self.getattr_L__mod___blocks___18___mlp_fc1(getattr_l__mod___blocks___18___norm2);  getattr_l__mod___blocks___18___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_209 = self.getattr_L__mod___blocks___18___mlp_act(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_210 = self.getattr_L__mod___blocks___18___mlp_drop1(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_211 = self.getattr_L__mod___blocks___18___mlp_norm(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_212 = self.getattr_L__mod___blocks___18___mlp_fc2(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_213 = self.getattr_L__mod___blocks___18___mlp_drop2(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_56 = getattr_l__mod___blocks___18___gamma_2 * x_213;  getattr_l__mod___blocks___18___gamma_2 = x_213 = None
    getattr_l__mod___blocks___18___drop_path_1 = self.getattr_L__mod___blocks___18___drop_path(mul_56);  mul_56 = None
    x_214 = x_207 + getattr_l__mod___blocks___18___drop_path_1;  x_207 = getattr_l__mod___blocks___18___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___19___gamma_1 = self.getattr_L__mod___blocks___19___gamma_1
    getattr_l__mod___blocks___19___norm1 = self.getattr_L__mod___blocks___19___norm1(x_214)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___19___attn_qkv = self.getattr_L__mod___blocks___19___attn_qkv(getattr_l__mod___blocks___19___norm1);  getattr_l__mod___blocks___19___norm1 = None
    reshape_38 = getattr_l__mod___blocks___19___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___19___attn_qkv = None
    qkv_19 = reshape_38.permute(2, 0, 3, 1, 4);  reshape_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_57 = qkv_19[0]
    q_19 = getitem_57 * 0.14433756729740643;  getitem_57 = None
    k_19 = qkv_19[1]
    v_19 = qkv_19[2];  qkv_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_39 = k_19.transpose(-2, -1);  k_19 = None
    attn_95 = q_19 @ transpose_39;  q_19 = transpose_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_96 = attn_95.permute(0, 2, 3, 1);  attn_95 = None
    getattr_l__mod___blocks___19___attn_proj_l = self.getattr_L__mod___blocks___19___attn_proj_l(permute_96);  permute_96 = None
    attn_96 = getattr_l__mod___blocks___19___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___19___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_97 = attn_96.softmax(dim = -1);  attn_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_98 = attn_97.permute(0, 2, 3, 1);  attn_97 = None
    getattr_l__mod___blocks___19___attn_proj_w = self.getattr_L__mod___blocks___19___attn_proj_w(permute_98);  permute_98 = None
    attn_98 = getattr_l__mod___blocks___19___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___19___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_99 = self.getattr_L__mod___blocks___19___attn_attn_drop(attn_98);  attn_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_39 = attn_99 @ v_19;  attn_99 = v_19 = None
    transpose_40 = matmul_39.transpose(1, 2);  matmul_39 = None
    x_215 = transpose_40.reshape(8, 576, 768);  transpose_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_216 = self.getattr_L__mod___blocks___19___attn_proj(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_217 = self.getattr_L__mod___blocks___19___attn_proj_drop(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_58 = getattr_l__mod___blocks___19___gamma_1 * x_217;  getattr_l__mod___blocks___19___gamma_1 = x_217 = None
    getattr_l__mod___blocks___19___drop_path = self.getattr_L__mod___blocks___19___drop_path(mul_58);  mul_58 = None
    x_218 = x_214 + getattr_l__mod___blocks___19___drop_path;  x_214 = getattr_l__mod___blocks___19___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___19___gamma_2 = self.getattr_L__mod___blocks___19___gamma_2
    getattr_l__mod___blocks___19___norm2 = self.getattr_L__mod___blocks___19___norm2(x_218)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_219 = self.getattr_L__mod___blocks___19___mlp_fc1(getattr_l__mod___blocks___19___norm2);  getattr_l__mod___blocks___19___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_220 = self.getattr_L__mod___blocks___19___mlp_act(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_221 = self.getattr_L__mod___blocks___19___mlp_drop1(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_222 = self.getattr_L__mod___blocks___19___mlp_norm(x_221);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_223 = self.getattr_L__mod___blocks___19___mlp_fc2(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_224 = self.getattr_L__mod___blocks___19___mlp_drop2(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_59 = getattr_l__mod___blocks___19___gamma_2 * x_224;  getattr_l__mod___blocks___19___gamma_2 = x_224 = None
    getattr_l__mod___blocks___19___drop_path_1 = self.getattr_L__mod___blocks___19___drop_path(mul_59);  mul_59 = None
    x_225 = x_218 + getattr_l__mod___blocks___19___drop_path_1;  x_218 = getattr_l__mod___blocks___19___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___20___gamma_1 = self.getattr_L__mod___blocks___20___gamma_1
    getattr_l__mod___blocks___20___norm1 = self.getattr_L__mod___blocks___20___norm1(x_225)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___20___attn_qkv = self.getattr_L__mod___blocks___20___attn_qkv(getattr_l__mod___blocks___20___norm1);  getattr_l__mod___blocks___20___norm1 = None
    reshape_40 = getattr_l__mod___blocks___20___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___20___attn_qkv = None
    qkv_20 = reshape_40.permute(2, 0, 3, 1, 4);  reshape_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_60 = qkv_20[0]
    q_20 = getitem_60 * 0.14433756729740643;  getitem_60 = None
    k_20 = qkv_20[1]
    v_20 = qkv_20[2];  qkv_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_41 = k_20.transpose(-2, -1);  k_20 = None
    attn_100 = q_20 @ transpose_41;  q_20 = transpose_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_101 = attn_100.permute(0, 2, 3, 1);  attn_100 = None
    getattr_l__mod___blocks___20___attn_proj_l = self.getattr_L__mod___blocks___20___attn_proj_l(permute_101);  permute_101 = None
    attn_101 = getattr_l__mod___blocks___20___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___20___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_102 = attn_101.softmax(dim = -1);  attn_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_103 = attn_102.permute(0, 2, 3, 1);  attn_102 = None
    getattr_l__mod___blocks___20___attn_proj_w = self.getattr_L__mod___blocks___20___attn_proj_w(permute_103);  permute_103 = None
    attn_103 = getattr_l__mod___blocks___20___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___20___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_104 = self.getattr_L__mod___blocks___20___attn_attn_drop(attn_103);  attn_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_41 = attn_104 @ v_20;  attn_104 = v_20 = None
    transpose_42 = matmul_41.transpose(1, 2);  matmul_41 = None
    x_226 = transpose_42.reshape(8, 576, 768);  transpose_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_227 = self.getattr_L__mod___blocks___20___attn_proj(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_228 = self.getattr_L__mod___blocks___20___attn_proj_drop(x_227);  x_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_61 = getattr_l__mod___blocks___20___gamma_1 * x_228;  getattr_l__mod___blocks___20___gamma_1 = x_228 = None
    getattr_l__mod___blocks___20___drop_path = self.getattr_L__mod___blocks___20___drop_path(mul_61);  mul_61 = None
    x_229 = x_225 + getattr_l__mod___blocks___20___drop_path;  x_225 = getattr_l__mod___blocks___20___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___20___gamma_2 = self.getattr_L__mod___blocks___20___gamma_2
    getattr_l__mod___blocks___20___norm2 = self.getattr_L__mod___blocks___20___norm2(x_229)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_230 = self.getattr_L__mod___blocks___20___mlp_fc1(getattr_l__mod___blocks___20___norm2);  getattr_l__mod___blocks___20___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_231 = self.getattr_L__mod___blocks___20___mlp_act(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_232 = self.getattr_L__mod___blocks___20___mlp_drop1(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_233 = self.getattr_L__mod___blocks___20___mlp_norm(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_234 = self.getattr_L__mod___blocks___20___mlp_fc2(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_235 = self.getattr_L__mod___blocks___20___mlp_drop2(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_62 = getattr_l__mod___blocks___20___gamma_2 * x_235;  getattr_l__mod___blocks___20___gamma_2 = x_235 = None
    getattr_l__mod___blocks___20___drop_path_1 = self.getattr_L__mod___blocks___20___drop_path(mul_62);  mul_62 = None
    x_236 = x_229 + getattr_l__mod___blocks___20___drop_path_1;  x_229 = getattr_l__mod___blocks___20___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___21___gamma_1 = self.getattr_L__mod___blocks___21___gamma_1
    getattr_l__mod___blocks___21___norm1 = self.getattr_L__mod___blocks___21___norm1(x_236)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___21___attn_qkv = self.getattr_L__mod___blocks___21___attn_qkv(getattr_l__mod___blocks___21___norm1);  getattr_l__mod___blocks___21___norm1 = None
    reshape_42 = getattr_l__mod___blocks___21___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___21___attn_qkv = None
    qkv_21 = reshape_42.permute(2, 0, 3, 1, 4);  reshape_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_63 = qkv_21[0]
    q_21 = getitem_63 * 0.14433756729740643;  getitem_63 = None
    k_21 = qkv_21[1]
    v_21 = qkv_21[2];  qkv_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_43 = k_21.transpose(-2, -1);  k_21 = None
    attn_105 = q_21 @ transpose_43;  q_21 = transpose_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_106 = attn_105.permute(0, 2, 3, 1);  attn_105 = None
    getattr_l__mod___blocks___21___attn_proj_l = self.getattr_L__mod___blocks___21___attn_proj_l(permute_106);  permute_106 = None
    attn_106 = getattr_l__mod___blocks___21___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___21___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_107 = attn_106.softmax(dim = -1);  attn_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_108 = attn_107.permute(0, 2, 3, 1);  attn_107 = None
    getattr_l__mod___blocks___21___attn_proj_w = self.getattr_L__mod___blocks___21___attn_proj_w(permute_108);  permute_108 = None
    attn_108 = getattr_l__mod___blocks___21___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___21___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_109 = self.getattr_L__mod___blocks___21___attn_attn_drop(attn_108);  attn_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_43 = attn_109 @ v_21;  attn_109 = v_21 = None
    transpose_44 = matmul_43.transpose(1, 2);  matmul_43 = None
    x_237 = transpose_44.reshape(8, 576, 768);  transpose_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_238 = self.getattr_L__mod___blocks___21___attn_proj(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_239 = self.getattr_L__mod___blocks___21___attn_proj_drop(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_64 = getattr_l__mod___blocks___21___gamma_1 * x_239;  getattr_l__mod___blocks___21___gamma_1 = x_239 = None
    getattr_l__mod___blocks___21___drop_path = self.getattr_L__mod___blocks___21___drop_path(mul_64);  mul_64 = None
    x_240 = x_236 + getattr_l__mod___blocks___21___drop_path;  x_236 = getattr_l__mod___blocks___21___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___21___gamma_2 = self.getattr_L__mod___blocks___21___gamma_2
    getattr_l__mod___blocks___21___norm2 = self.getattr_L__mod___blocks___21___norm2(x_240)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_241 = self.getattr_L__mod___blocks___21___mlp_fc1(getattr_l__mod___blocks___21___norm2);  getattr_l__mod___blocks___21___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_242 = self.getattr_L__mod___blocks___21___mlp_act(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_243 = self.getattr_L__mod___blocks___21___mlp_drop1(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_244 = self.getattr_L__mod___blocks___21___mlp_norm(x_243);  x_243 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_245 = self.getattr_L__mod___blocks___21___mlp_fc2(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_246 = self.getattr_L__mod___blocks___21___mlp_drop2(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_65 = getattr_l__mod___blocks___21___gamma_2 * x_246;  getattr_l__mod___blocks___21___gamma_2 = x_246 = None
    getattr_l__mod___blocks___21___drop_path_1 = self.getattr_L__mod___blocks___21___drop_path(mul_65);  mul_65 = None
    x_247 = x_240 + getattr_l__mod___blocks___21___drop_path_1;  x_240 = getattr_l__mod___blocks___21___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___22___gamma_1 = self.getattr_L__mod___blocks___22___gamma_1
    getattr_l__mod___blocks___22___norm1 = self.getattr_L__mod___blocks___22___norm1(x_247)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___22___attn_qkv = self.getattr_L__mod___blocks___22___attn_qkv(getattr_l__mod___blocks___22___norm1);  getattr_l__mod___blocks___22___norm1 = None
    reshape_44 = getattr_l__mod___blocks___22___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___22___attn_qkv = None
    qkv_22 = reshape_44.permute(2, 0, 3, 1, 4);  reshape_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_66 = qkv_22[0]
    q_22 = getitem_66 * 0.14433756729740643;  getitem_66 = None
    k_22 = qkv_22[1]
    v_22 = qkv_22[2];  qkv_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_45 = k_22.transpose(-2, -1);  k_22 = None
    attn_110 = q_22 @ transpose_45;  q_22 = transpose_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_111 = attn_110.permute(0, 2, 3, 1);  attn_110 = None
    getattr_l__mod___blocks___22___attn_proj_l = self.getattr_L__mod___blocks___22___attn_proj_l(permute_111);  permute_111 = None
    attn_111 = getattr_l__mod___blocks___22___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___22___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_112 = attn_111.softmax(dim = -1);  attn_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_113 = attn_112.permute(0, 2, 3, 1);  attn_112 = None
    getattr_l__mod___blocks___22___attn_proj_w = self.getattr_L__mod___blocks___22___attn_proj_w(permute_113);  permute_113 = None
    attn_113 = getattr_l__mod___blocks___22___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___22___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_114 = self.getattr_L__mod___blocks___22___attn_attn_drop(attn_113);  attn_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_45 = attn_114 @ v_22;  attn_114 = v_22 = None
    transpose_46 = matmul_45.transpose(1, 2);  matmul_45 = None
    x_248 = transpose_46.reshape(8, 576, 768);  transpose_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_249 = self.getattr_L__mod___blocks___22___attn_proj(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_250 = self.getattr_L__mod___blocks___22___attn_proj_drop(x_249);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_67 = getattr_l__mod___blocks___22___gamma_1 * x_250;  getattr_l__mod___blocks___22___gamma_1 = x_250 = None
    getattr_l__mod___blocks___22___drop_path = self.getattr_L__mod___blocks___22___drop_path(mul_67);  mul_67 = None
    x_251 = x_247 + getattr_l__mod___blocks___22___drop_path;  x_247 = getattr_l__mod___blocks___22___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___22___gamma_2 = self.getattr_L__mod___blocks___22___gamma_2
    getattr_l__mod___blocks___22___norm2 = self.getattr_L__mod___blocks___22___norm2(x_251)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_252 = self.getattr_L__mod___blocks___22___mlp_fc1(getattr_l__mod___blocks___22___norm2);  getattr_l__mod___blocks___22___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_253 = self.getattr_L__mod___blocks___22___mlp_act(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_254 = self.getattr_L__mod___blocks___22___mlp_drop1(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_255 = self.getattr_L__mod___blocks___22___mlp_norm(x_254);  x_254 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_256 = self.getattr_L__mod___blocks___22___mlp_fc2(x_255);  x_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_257 = self.getattr_L__mod___blocks___22___mlp_drop2(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_68 = getattr_l__mod___blocks___22___gamma_2 * x_257;  getattr_l__mod___blocks___22___gamma_2 = x_257 = None
    getattr_l__mod___blocks___22___drop_path_1 = self.getattr_L__mod___blocks___22___drop_path(mul_68);  mul_68 = None
    x_258 = x_251 + getattr_l__mod___blocks___22___drop_path_1;  x_251 = getattr_l__mod___blocks___22___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___23___gamma_1 = self.getattr_L__mod___blocks___23___gamma_1
    getattr_l__mod___blocks___23___norm1 = self.getattr_L__mod___blocks___23___norm1(x_258)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___23___attn_qkv = self.getattr_L__mod___blocks___23___attn_qkv(getattr_l__mod___blocks___23___norm1);  getattr_l__mod___blocks___23___norm1 = None
    reshape_46 = getattr_l__mod___blocks___23___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___23___attn_qkv = None
    qkv_23 = reshape_46.permute(2, 0, 3, 1, 4);  reshape_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_69 = qkv_23[0]
    q_23 = getitem_69 * 0.14433756729740643;  getitem_69 = None
    k_23 = qkv_23[1]
    v_23 = qkv_23[2];  qkv_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_47 = k_23.transpose(-2, -1);  k_23 = None
    attn_115 = q_23 @ transpose_47;  q_23 = transpose_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_116 = attn_115.permute(0, 2, 3, 1);  attn_115 = None
    getattr_l__mod___blocks___23___attn_proj_l = self.getattr_L__mod___blocks___23___attn_proj_l(permute_116);  permute_116 = None
    attn_116 = getattr_l__mod___blocks___23___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___23___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_117 = attn_116.softmax(dim = -1);  attn_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_118 = attn_117.permute(0, 2, 3, 1);  attn_117 = None
    getattr_l__mod___blocks___23___attn_proj_w = self.getattr_L__mod___blocks___23___attn_proj_w(permute_118);  permute_118 = None
    attn_118 = getattr_l__mod___blocks___23___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___23___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_119 = self.getattr_L__mod___blocks___23___attn_attn_drop(attn_118);  attn_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_47 = attn_119 @ v_23;  attn_119 = v_23 = None
    transpose_48 = matmul_47.transpose(1, 2);  matmul_47 = None
    x_259 = transpose_48.reshape(8, 576, 768);  transpose_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_260 = self.getattr_L__mod___blocks___23___attn_proj(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_261 = self.getattr_L__mod___blocks___23___attn_proj_drop(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_70 = getattr_l__mod___blocks___23___gamma_1 * x_261;  getattr_l__mod___blocks___23___gamma_1 = x_261 = None
    getattr_l__mod___blocks___23___drop_path = self.getattr_L__mod___blocks___23___drop_path(mul_70);  mul_70 = None
    x_262 = x_258 + getattr_l__mod___blocks___23___drop_path;  x_258 = getattr_l__mod___blocks___23___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___23___gamma_2 = self.getattr_L__mod___blocks___23___gamma_2
    getattr_l__mod___blocks___23___norm2 = self.getattr_L__mod___blocks___23___norm2(x_262)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_263 = self.getattr_L__mod___blocks___23___mlp_fc1(getattr_l__mod___blocks___23___norm2);  getattr_l__mod___blocks___23___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_264 = self.getattr_L__mod___blocks___23___mlp_act(x_263);  x_263 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_265 = self.getattr_L__mod___blocks___23___mlp_drop1(x_264);  x_264 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_266 = self.getattr_L__mod___blocks___23___mlp_norm(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_267 = self.getattr_L__mod___blocks___23___mlp_fc2(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_268 = self.getattr_L__mod___blocks___23___mlp_drop2(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_71 = getattr_l__mod___blocks___23___gamma_2 * x_268;  getattr_l__mod___blocks___23___gamma_2 = x_268 = None
    getattr_l__mod___blocks___23___drop_path_1 = self.getattr_L__mod___blocks___23___drop_path(mul_71);  mul_71 = None
    x_269 = x_262 + getattr_l__mod___blocks___23___drop_path_1;  x_262 = getattr_l__mod___blocks___23___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___24___gamma_1 = self.getattr_L__mod___blocks___24___gamma_1
    getattr_l__mod___blocks___24___norm1 = self.getattr_L__mod___blocks___24___norm1(x_269)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___24___attn_qkv = self.getattr_L__mod___blocks___24___attn_qkv(getattr_l__mod___blocks___24___norm1);  getattr_l__mod___blocks___24___norm1 = None
    reshape_48 = getattr_l__mod___blocks___24___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___24___attn_qkv = None
    qkv_24 = reshape_48.permute(2, 0, 3, 1, 4);  reshape_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_72 = qkv_24[0]
    q_24 = getitem_72 * 0.14433756729740643;  getitem_72 = None
    k_24 = qkv_24[1]
    v_24 = qkv_24[2];  qkv_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_49 = k_24.transpose(-2, -1);  k_24 = None
    attn_120 = q_24 @ transpose_49;  q_24 = transpose_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_121 = attn_120.permute(0, 2, 3, 1);  attn_120 = None
    getattr_l__mod___blocks___24___attn_proj_l = self.getattr_L__mod___blocks___24___attn_proj_l(permute_121);  permute_121 = None
    attn_121 = getattr_l__mod___blocks___24___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___24___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_122 = attn_121.softmax(dim = -1);  attn_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_123 = attn_122.permute(0, 2, 3, 1);  attn_122 = None
    getattr_l__mod___blocks___24___attn_proj_w = self.getattr_L__mod___blocks___24___attn_proj_w(permute_123);  permute_123 = None
    attn_123 = getattr_l__mod___blocks___24___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___24___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_124 = self.getattr_L__mod___blocks___24___attn_attn_drop(attn_123);  attn_123 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_49 = attn_124 @ v_24;  attn_124 = v_24 = None
    transpose_50 = matmul_49.transpose(1, 2);  matmul_49 = None
    x_270 = transpose_50.reshape(8, 576, 768);  transpose_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_271 = self.getattr_L__mod___blocks___24___attn_proj(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_272 = self.getattr_L__mod___blocks___24___attn_proj_drop(x_271);  x_271 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_73 = getattr_l__mod___blocks___24___gamma_1 * x_272;  getattr_l__mod___blocks___24___gamma_1 = x_272 = None
    getattr_l__mod___blocks___24___drop_path = self.getattr_L__mod___blocks___24___drop_path(mul_73);  mul_73 = None
    x_273 = x_269 + getattr_l__mod___blocks___24___drop_path;  x_269 = getattr_l__mod___blocks___24___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___24___gamma_2 = self.getattr_L__mod___blocks___24___gamma_2
    getattr_l__mod___blocks___24___norm2 = self.getattr_L__mod___blocks___24___norm2(x_273)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_274 = self.getattr_L__mod___blocks___24___mlp_fc1(getattr_l__mod___blocks___24___norm2);  getattr_l__mod___blocks___24___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_275 = self.getattr_L__mod___blocks___24___mlp_act(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_276 = self.getattr_L__mod___blocks___24___mlp_drop1(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_277 = self.getattr_L__mod___blocks___24___mlp_norm(x_276);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_278 = self.getattr_L__mod___blocks___24___mlp_fc2(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_279 = self.getattr_L__mod___blocks___24___mlp_drop2(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_74 = getattr_l__mod___blocks___24___gamma_2 * x_279;  getattr_l__mod___blocks___24___gamma_2 = x_279 = None
    getattr_l__mod___blocks___24___drop_path_1 = self.getattr_L__mod___blocks___24___drop_path(mul_74);  mul_74 = None
    x_280 = x_273 + getattr_l__mod___blocks___24___drop_path_1;  x_273 = getattr_l__mod___blocks___24___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___25___gamma_1 = self.getattr_L__mod___blocks___25___gamma_1
    getattr_l__mod___blocks___25___norm1 = self.getattr_L__mod___blocks___25___norm1(x_280)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___25___attn_qkv = self.getattr_L__mod___blocks___25___attn_qkv(getattr_l__mod___blocks___25___norm1);  getattr_l__mod___blocks___25___norm1 = None
    reshape_50 = getattr_l__mod___blocks___25___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___25___attn_qkv = None
    qkv_25 = reshape_50.permute(2, 0, 3, 1, 4);  reshape_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_75 = qkv_25[0]
    q_25 = getitem_75 * 0.14433756729740643;  getitem_75 = None
    k_25 = qkv_25[1]
    v_25 = qkv_25[2];  qkv_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_51 = k_25.transpose(-2, -1);  k_25 = None
    attn_125 = q_25 @ transpose_51;  q_25 = transpose_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_126 = attn_125.permute(0, 2, 3, 1);  attn_125 = None
    getattr_l__mod___blocks___25___attn_proj_l = self.getattr_L__mod___blocks___25___attn_proj_l(permute_126);  permute_126 = None
    attn_126 = getattr_l__mod___blocks___25___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___25___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_127 = attn_126.softmax(dim = -1);  attn_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_128 = attn_127.permute(0, 2, 3, 1);  attn_127 = None
    getattr_l__mod___blocks___25___attn_proj_w = self.getattr_L__mod___blocks___25___attn_proj_w(permute_128);  permute_128 = None
    attn_128 = getattr_l__mod___blocks___25___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___25___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_129 = self.getattr_L__mod___blocks___25___attn_attn_drop(attn_128);  attn_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_51 = attn_129 @ v_25;  attn_129 = v_25 = None
    transpose_52 = matmul_51.transpose(1, 2);  matmul_51 = None
    x_281 = transpose_52.reshape(8, 576, 768);  transpose_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_282 = self.getattr_L__mod___blocks___25___attn_proj(x_281);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_283 = self.getattr_L__mod___blocks___25___attn_proj_drop(x_282);  x_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_76 = getattr_l__mod___blocks___25___gamma_1 * x_283;  getattr_l__mod___blocks___25___gamma_1 = x_283 = None
    getattr_l__mod___blocks___25___drop_path = self.getattr_L__mod___blocks___25___drop_path(mul_76);  mul_76 = None
    x_284 = x_280 + getattr_l__mod___blocks___25___drop_path;  x_280 = getattr_l__mod___blocks___25___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___25___gamma_2 = self.getattr_L__mod___blocks___25___gamma_2
    getattr_l__mod___blocks___25___norm2 = self.getattr_L__mod___blocks___25___norm2(x_284)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_285 = self.getattr_L__mod___blocks___25___mlp_fc1(getattr_l__mod___blocks___25___norm2);  getattr_l__mod___blocks___25___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_286 = self.getattr_L__mod___blocks___25___mlp_act(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_287 = self.getattr_L__mod___blocks___25___mlp_drop1(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_288 = self.getattr_L__mod___blocks___25___mlp_norm(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_289 = self.getattr_L__mod___blocks___25___mlp_fc2(x_288);  x_288 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_290 = self.getattr_L__mod___blocks___25___mlp_drop2(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_77 = getattr_l__mod___blocks___25___gamma_2 * x_290;  getattr_l__mod___blocks___25___gamma_2 = x_290 = None
    getattr_l__mod___blocks___25___drop_path_1 = self.getattr_L__mod___blocks___25___drop_path(mul_77);  mul_77 = None
    x_291 = x_284 + getattr_l__mod___blocks___25___drop_path_1;  x_284 = getattr_l__mod___blocks___25___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___26___gamma_1 = self.getattr_L__mod___blocks___26___gamma_1
    getattr_l__mod___blocks___26___norm1 = self.getattr_L__mod___blocks___26___norm1(x_291)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___26___attn_qkv = self.getattr_L__mod___blocks___26___attn_qkv(getattr_l__mod___blocks___26___norm1);  getattr_l__mod___blocks___26___norm1 = None
    reshape_52 = getattr_l__mod___blocks___26___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___26___attn_qkv = None
    qkv_26 = reshape_52.permute(2, 0, 3, 1, 4);  reshape_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_78 = qkv_26[0]
    q_26 = getitem_78 * 0.14433756729740643;  getitem_78 = None
    k_26 = qkv_26[1]
    v_26 = qkv_26[2];  qkv_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_53 = k_26.transpose(-2, -1);  k_26 = None
    attn_130 = q_26 @ transpose_53;  q_26 = transpose_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_131 = attn_130.permute(0, 2, 3, 1);  attn_130 = None
    getattr_l__mod___blocks___26___attn_proj_l = self.getattr_L__mod___blocks___26___attn_proj_l(permute_131);  permute_131 = None
    attn_131 = getattr_l__mod___blocks___26___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___26___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_132 = attn_131.softmax(dim = -1);  attn_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_133 = attn_132.permute(0, 2, 3, 1);  attn_132 = None
    getattr_l__mod___blocks___26___attn_proj_w = self.getattr_L__mod___blocks___26___attn_proj_w(permute_133);  permute_133 = None
    attn_133 = getattr_l__mod___blocks___26___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___26___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_134 = self.getattr_L__mod___blocks___26___attn_attn_drop(attn_133);  attn_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_53 = attn_134 @ v_26;  attn_134 = v_26 = None
    transpose_54 = matmul_53.transpose(1, 2);  matmul_53 = None
    x_292 = transpose_54.reshape(8, 576, 768);  transpose_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_293 = self.getattr_L__mod___blocks___26___attn_proj(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_294 = self.getattr_L__mod___blocks___26___attn_proj_drop(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_79 = getattr_l__mod___blocks___26___gamma_1 * x_294;  getattr_l__mod___blocks___26___gamma_1 = x_294 = None
    getattr_l__mod___blocks___26___drop_path = self.getattr_L__mod___blocks___26___drop_path(mul_79);  mul_79 = None
    x_295 = x_291 + getattr_l__mod___blocks___26___drop_path;  x_291 = getattr_l__mod___blocks___26___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___26___gamma_2 = self.getattr_L__mod___blocks___26___gamma_2
    getattr_l__mod___blocks___26___norm2 = self.getattr_L__mod___blocks___26___norm2(x_295)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_296 = self.getattr_L__mod___blocks___26___mlp_fc1(getattr_l__mod___blocks___26___norm2);  getattr_l__mod___blocks___26___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_297 = self.getattr_L__mod___blocks___26___mlp_act(x_296);  x_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_298 = self.getattr_L__mod___blocks___26___mlp_drop1(x_297);  x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_299 = self.getattr_L__mod___blocks___26___mlp_norm(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_300 = self.getattr_L__mod___blocks___26___mlp_fc2(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_301 = self.getattr_L__mod___blocks___26___mlp_drop2(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_80 = getattr_l__mod___blocks___26___gamma_2 * x_301;  getattr_l__mod___blocks___26___gamma_2 = x_301 = None
    getattr_l__mod___blocks___26___drop_path_1 = self.getattr_L__mod___blocks___26___drop_path(mul_80);  mul_80 = None
    x_302 = x_295 + getattr_l__mod___blocks___26___drop_path_1;  x_295 = getattr_l__mod___blocks___26___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___27___gamma_1 = self.getattr_L__mod___blocks___27___gamma_1
    getattr_l__mod___blocks___27___norm1 = self.getattr_L__mod___blocks___27___norm1(x_302)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___27___attn_qkv = self.getattr_L__mod___blocks___27___attn_qkv(getattr_l__mod___blocks___27___norm1);  getattr_l__mod___blocks___27___norm1 = None
    reshape_54 = getattr_l__mod___blocks___27___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___27___attn_qkv = None
    qkv_27 = reshape_54.permute(2, 0, 3, 1, 4);  reshape_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_81 = qkv_27[0]
    q_27 = getitem_81 * 0.14433756729740643;  getitem_81 = None
    k_27 = qkv_27[1]
    v_27 = qkv_27[2];  qkv_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_55 = k_27.transpose(-2, -1);  k_27 = None
    attn_135 = q_27 @ transpose_55;  q_27 = transpose_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_136 = attn_135.permute(0, 2, 3, 1);  attn_135 = None
    getattr_l__mod___blocks___27___attn_proj_l = self.getattr_L__mod___blocks___27___attn_proj_l(permute_136);  permute_136 = None
    attn_136 = getattr_l__mod___blocks___27___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___27___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_137 = attn_136.softmax(dim = -1);  attn_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_138 = attn_137.permute(0, 2, 3, 1);  attn_137 = None
    getattr_l__mod___blocks___27___attn_proj_w = self.getattr_L__mod___blocks___27___attn_proj_w(permute_138);  permute_138 = None
    attn_138 = getattr_l__mod___blocks___27___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___27___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_139 = self.getattr_L__mod___blocks___27___attn_attn_drop(attn_138);  attn_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_55 = attn_139 @ v_27;  attn_139 = v_27 = None
    transpose_56 = matmul_55.transpose(1, 2);  matmul_55 = None
    x_303 = transpose_56.reshape(8, 576, 768);  transpose_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_304 = self.getattr_L__mod___blocks___27___attn_proj(x_303);  x_303 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_305 = self.getattr_L__mod___blocks___27___attn_proj_drop(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_82 = getattr_l__mod___blocks___27___gamma_1 * x_305;  getattr_l__mod___blocks___27___gamma_1 = x_305 = None
    getattr_l__mod___blocks___27___drop_path = self.getattr_L__mod___blocks___27___drop_path(mul_82);  mul_82 = None
    x_306 = x_302 + getattr_l__mod___blocks___27___drop_path;  x_302 = getattr_l__mod___blocks___27___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___27___gamma_2 = self.getattr_L__mod___blocks___27___gamma_2
    getattr_l__mod___blocks___27___norm2 = self.getattr_L__mod___blocks___27___norm2(x_306)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_307 = self.getattr_L__mod___blocks___27___mlp_fc1(getattr_l__mod___blocks___27___norm2);  getattr_l__mod___blocks___27___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_308 = self.getattr_L__mod___blocks___27___mlp_act(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_309 = self.getattr_L__mod___blocks___27___mlp_drop1(x_308);  x_308 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_310 = self.getattr_L__mod___blocks___27___mlp_norm(x_309);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_311 = self.getattr_L__mod___blocks___27___mlp_fc2(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_312 = self.getattr_L__mod___blocks___27___mlp_drop2(x_311);  x_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_83 = getattr_l__mod___blocks___27___gamma_2 * x_312;  getattr_l__mod___blocks___27___gamma_2 = x_312 = None
    getattr_l__mod___blocks___27___drop_path_1 = self.getattr_L__mod___blocks___27___drop_path(mul_83);  mul_83 = None
    x_313 = x_306 + getattr_l__mod___blocks___27___drop_path_1;  x_306 = getattr_l__mod___blocks___27___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___28___gamma_1 = self.getattr_L__mod___blocks___28___gamma_1
    getattr_l__mod___blocks___28___norm1 = self.getattr_L__mod___blocks___28___norm1(x_313)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___28___attn_qkv = self.getattr_L__mod___blocks___28___attn_qkv(getattr_l__mod___blocks___28___norm1);  getattr_l__mod___blocks___28___norm1 = None
    reshape_56 = getattr_l__mod___blocks___28___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___28___attn_qkv = None
    qkv_28 = reshape_56.permute(2, 0, 3, 1, 4);  reshape_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_84 = qkv_28[0]
    q_28 = getitem_84 * 0.14433756729740643;  getitem_84 = None
    k_28 = qkv_28[1]
    v_28 = qkv_28[2];  qkv_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_57 = k_28.transpose(-2, -1);  k_28 = None
    attn_140 = q_28 @ transpose_57;  q_28 = transpose_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_141 = attn_140.permute(0, 2, 3, 1);  attn_140 = None
    getattr_l__mod___blocks___28___attn_proj_l = self.getattr_L__mod___blocks___28___attn_proj_l(permute_141);  permute_141 = None
    attn_141 = getattr_l__mod___blocks___28___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___28___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_142 = attn_141.softmax(dim = -1);  attn_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_143 = attn_142.permute(0, 2, 3, 1);  attn_142 = None
    getattr_l__mod___blocks___28___attn_proj_w = self.getattr_L__mod___blocks___28___attn_proj_w(permute_143);  permute_143 = None
    attn_143 = getattr_l__mod___blocks___28___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___28___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_144 = self.getattr_L__mod___blocks___28___attn_attn_drop(attn_143);  attn_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_57 = attn_144 @ v_28;  attn_144 = v_28 = None
    transpose_58 = matmul_57.transpose(1, 2);  matmul_57 = None
    x_314 = transpose_58.reshape(8, 576, 768);  transpose_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_315 = self.getattr_L__mod___blocks___28___attn_proj(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_316 = self.getattr_L__mod___blocks___28___attn_proj_drop(x_315);  x_315 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_85 = getattr_l__mod___blocks___28___gamma_1 * x_316;  getattr_l__mod___blocks___28___gamma_1 = x_316 = None
    getattr_l__mod___blocks___28___drop_path = self.getattr_L__mod___blocks___28___drop_path(mul_85);  mul_85 = None
    x_317 = x_313 + getattr_l__mod___blocks___28___drop_path;  x_313 = getattr_l__mod___blocks___28___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___28___gamma_2 = self.getattr_L__mod___blocks___28___gamma_2
    getattr_l__mod___blocks___28___norm2 = self.getattr_L__mod___blocks___28___norm2(x_317)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_318 = self.getattr_L__mod___blocks___28___mlp_fc1(getattr_l__mod___blocks___28___norm2);  getattr_l__mod___blocks___28___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_319 = self.getattr_L__mod___blocks___28___mlp_act(x_318);  x_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_320 = self.getattr_L__mod___blocks___28___mlp_drop1(x_319);  x_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_321 = self.getattr_L__mod___blocks___28___mlp_norm(x_320);  x_320 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_322 = self.getattr_L__mod___blocks___28___mlp_fc2(x_321);  x_321 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_323 = self.getattr_L__mod___blocks___28___mlp_drop2(x_322);  x_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_86 = getattr_l__mod___blocks___28___gamma_2 * x_323;  getattr_l__mod___blocks___28___gamma_2 = x_323 = None
    getattr_l__mod___blocks___28___drop_path_1 = self.getattr_L__mod___blocks___28___drop_path(mul_86);  mul_86 = None
    x_324 = x_317 + getattr_l__mod___blocks___28___drop_path_1;  x_317 = getattr_l__mod___blocks___28___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___29___gamma_1 = self.getattr_L__mod___blocks___29___gamma_1
    getattr_l__mod___blocks___29___norm1 = self.getattr_L__mod___blocks___29___norm1(x_324)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___29___attn_qkv = self.getattr_L__mod___blocks___29___attn_qkv(getattr_l__mod___blocks___29___norm1);  getattr_l__mod___blocks___29___norm1 = None
    reshape_58 = getattr_l__mod___blocks___29___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___29___attn_qkv = None
    qkv_29 = reshape_58.permute(2, 0, 3, 1, 4);  reshape_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_87 = qkv_29[0]
    q_29 = getitem_87 * 0.14433756729740643;  getitem_87 = None
    k_29 = qkv_29[1]
    v_29 = qkv_29[2];  qkv_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_59 = k_29.transpose(-2, -1);  k_29 = None
    attn_145 = q_29 @ transpose_59;  q_29 = transpose_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_146 = attn_145.permute(0, 2, 3, 1);  attn_145 = None
    getattr_l__mod___blocks___29___attn_proj_l = self.getattr_L__mod___blocks___29___attn_proj_l(permute_146);  permute_146 = None
    attn_146 = getattr_l__mod___blocks___29___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___29___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_147 = attn_146.softmax(dim = -1);  attn_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_148 = attn_147.permute(0, 2, 3, 1);  attn_147 = None
    getattr_l__mod___blocks___29___attn_proj_w = self.getattr_L__mod___blocks___29___attn_proj_w(permute_148);  permute_148 = None
    attn_148 = getattr_l__mod___blocks___29___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___29___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_149 = self.getattr_L__mod___blocks___29___attn_attn_drop(attn_148);  attn_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_59 = attn_149 @ v_29;  attn_149 = v_29 = None
    transpose_60 = matmul_59.transpose(1, 2);  matmul_59 = None
    x_325 = transpose_60.reshape(8, 576, 768);  transpose_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_326 = self.getattr_L__mod___blocks___29___attn_proj(x_325);  x_325 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_327 = self.getattr_L__mod___blocks___29___attn_proj_drop(x_326);  x_326 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_88 = getattr_l__mod___blocks___29___gamma_1 * x_327;  getattr_l__mod___blocks___29___gamma_1 = x_327 = None
    getattr_l__mod___blocks___29___drop_path = self.getattr_L__mod___blocks___29___drop_path(mul_88);  mul_88 = None
    x_328 = x_324 + getattr_l__mod___blocks___29___drop_path;  x_324 = getattr_l__mod___blocks___29___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___29___gamma_2 = self.getattr_L__mod___blocks___29___gamma_2
    getattr_l__mod___blocks___29___norm2 = self.getattr_L__mod___blocks___29___norm2(x_328)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_329 = self.getattr_L__mod___blocks___29___mlp_fc1(getattr_l__mod___blocks___29___norm2);  getattr_l__mod___blocks___29___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_330 = self.getattr_L__mod___blocks___29___mlp_act(x_329);  x_329 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_331 = self.getattr_L__mod___blocks___29___mlp_drop1(x_330);  x_330 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_332 = self.getattr_L__mod___blocks___29___mlp_norm(x_331);  x_331 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_333 = self.getattr_L__mod___blocks___29___mlp_fc2(x_332);  x_332 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_334 = self.getattr_L__mod___blocks___29___mlp_drop2(x_333);  x_333 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_89 = getattr_l__mod___blocks___29___gamma_2 * x_334;  getattr_l__mod___blocks___29___gamma_2 = x_334 = None
    getattr_l__mod___blocks___29___drop_path_1 = self.getattr_L__mod___blocks___29___drop_path(mul_89);  mul_89 = None
    x_335 = x_328 + getattr_l__mod___blocks___29___drop_path_1;  x_328 = getattr_l__mod___blocks___29___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___30___gamma_1 = self.getattr_L__mod___blocks___30___gamma_1
    getattr_l__mod___blocks___30___norm1 = self.getattr_L__mod___blocks___30___norm1(x_335)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___30___attn_qkv = self.getattr_L__mod___blocks___30___attn_qkv(getattr_l__mod___blocks___30___norm1);  getattr_l__mod___blocks___30___norm1 = None
    reshape_60 = getattr_l__mod___blocks___30___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___30___attn_qkv = None
    qkv_30 = reshape_60.permute(2, 0, 3, 1, 4);  reshape_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_90 = qkv_30[0]
    q_30 = getitem_90 * 0.14433756729740643;  getitem_90 = None
    k_30 = qkv_30[1]
    v_30 = qkv_30[2];  qkv_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_61 = k_30.transpose(-2, -1);  k_30 = None
    attn_150 = q_30 @ transpose_61;  q_30 = transpose_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_151 = attn_150.permute(0, 2, 3, 1);  attn_150 = None
    getattr_l__mod___blocks___30___attn_proj_l = self.getattr_L__mod___blocks___30___attn_proj_l(permute_151);  permute_151 = None
    attn_151 = getattr_l__mod___blocks___30___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___30___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_152 = attn_151.softmax(dim = -1);  attn_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_153 = attn_152.permute(0, 2, 3, 1);  attn_152 = None
    getattr_l__mod___blocks___30___attn_proj_w = self.getattr_L__mod___blocks___30___attn_proj_w(permute_153);  permute_153 = None
    attn_153 = getattr_l__mod___blocks___30___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___30___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_154 = self.getattr_L__mod___blocks___30___attn_attn_drop(attn_153);  attn_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_61 = attn_154 @ v_30;  attn_154 = v_30 = None
    transpose_62 = matmul_61.transpose(1, 2);  matmul_61 = None
    x_336 = transpose_62.reshape(8, 576, 768);  transpose_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_337 = self.getattr_L__mod___blocks___30___attn_proj(x_336);  x_336 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_338 = self.getattr_L__mod___blocks___30___attn_proj_drop(x_337);  x_337 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_91 = getattr_l__mod___blocks___30___gamma_1 * x_338;  getattr_l__mod___blocks___30___gamma_1 = x_338 = None
    getattr_l__mod___blocks___30___drop_path = self.getattr_L__mod___blocks___30___drop_path(mul_91);  mul_91 = None
    x_339 = x_335 + getattr_l__mod___blocks___30___drop_path;  x_335 = getattr_l__mod___blocks___30___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___30___gamma_2 = self.getattr_L__mod___blocks___30___gamma_2
    getattr_l__mod___blocks___30___norm2 = self.getattr_L__mod___blocks___30___norm2(x_339)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_340 = self.getattr_L__mod___blocks___30___mlp_fc1(getattr_l__mod___blocks___30___norm2);  getattr_l__mod___blocks___30___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_341 = self.getattr_L__mod___blocks___30___mlp_act(x_340);  x_340 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_342 = self.getattr_L__mod___blocks___30___mlp_drop1(x_341);  x_341 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_343 = self.getattr_L__mod___blocks___30___mlp_norm(x_342);  x_342 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_344 = self.getattr_L__mod___blocks___30___mlp_fc2(x_343);  x_343 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_345 = self.getattr_L__mod___blocks___30___mlp_drop2(x_344);  x_344 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_92 = getattr_l__mod___blocks___30___gamma_2 * x_345;  getattr_l__mod___blocks___30___gamma_2 = x_345 = None
    getattr_l__mod___blocks___30___drop_path_1 = self.getattr_L__mod___blocks___30___drop_path(mul_92);  mul_92 = None
    x_346 = x_339 + getattr_l__mod___blocks___30___drop_path_1;  x_339 = getattr_l__mod___blocks___30___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___31___gamma_1 = self.getattr_L__mod___blocks___31___gamma_1
    getattr_l__mod___blocks___31___norm1 = self.getattr_L__mod___blocks___31___norm1(x_346)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___31___attn_qkv = self.getattr_L__mod___blocks___31___attn_qkv(getattr_l__mod___blocks___31___norm1);  getattr_l__mod___blocks___31___norm1 = None
    reshape_62 = getattr_l__mod___blocks___31___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___31___attn_qkv = None
    qkv_31 = reshape_62.permute(2, 0, 3, 1, 4);  reshape_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_93 = qkv_31[0]
    q_31 = getitem_93 * 0.14433756729740643;  getitem_93 = None
    k_31 = qkv_31[1]
    v_31 = qkv_31[2];  qkv_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_63 = k_31.transpose(-2, -1);  k_31 = None
    attn_155 = q_31 @ transpose_63;  q_31 = transpose_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_156 = attn_155.permute(0, 2, 3, 1);  attn_155 = None
    getattr_l__mod___blocks___31___attn_proj_l = self.getattr_L__mod___blocks___31___attn_proj_l(permute_156);  permute_156 = None
    attn_156 = getattr_l__mod___blocks___31___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___31___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_157 = attn_156.softmax(dim = -1);  attn_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_158 = attn_157.permute(0, 2, 3, 1);  attn_157 = None
    getattr_l__mod___blocks___31___attn_proj_w = self.getattr_L__mod___blocks___31___attn_proj_w(permute_158);  permute_158 = None
    attn_158 = getattr_l__mod___blocks___31___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___31___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_159 = self.getattr_L__mod___blocks___31___attn_attn_drop(attn_158);  attn_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_63 = attn_159 @ v_31;  attn_159 = v_31 = None
    transpose_64 = matmul_63.transpose(1, 2);  matmul_63 = None
    x_347 = transpose_64.reshape(8, 576, 768);  transpose_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_348 = self.getattr_L__mod___blocks___31___attn_proj(x_347);  x_347 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_349 = self.getattr_L__mod___blocks___31___attn_proj_drop(x_348);  x_348 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_94 = getattr_l__mod___blocks___31___gamma_1 * x_349;  getattr_l__mod___blocks___31___gamma_1 = x_349 = None
    getattr_l__mod___blocks___31___drop_path = self.getattr_L__mod___blocks___31___drop_path(mul_94);  mul_94 = None
    x_350 = x_346 + getattr_l__mod___blocks___31___drop_path;  x_346 = getattr_l__mod___blocks___31___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___31___gamma_2 = self.getattr_L__mod___blocks___31___gamma_2
    getattr_l__mod___blocks___31___norm2 = self.getattr_L__mod___blocks___31___norm2(x_350)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_351 = self.getattr_L__mod___blocks___31___mlp_fc1(getattr_l__mod___blocks___31___norm2);  getattr_l__mod___blocks___31___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_352 = self.getattr_L__mod___blocks___31___mlp_act(x_351);  x_351 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_353 = self.getattr_L__mod___blocks___31___mlp_drop1(x_352);  x_352 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_354 = self.getattr_L__mod___blocks___31___mlp_norm(x_353);  x_353 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_355 = self.getattr_L__mod___blocks___31___mlp_fc2(x_354);  x_354 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_356 = self.getattr_L__mod___blocks___31___mlp_drop2(x_355);  x_355 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_95 = getattr_l__mod___blocks___31___gamma_2 * x_356;  getattr_l__mod___blocks___31___gamma_2 = x_356 = None
    getattr_l__mod___blocks___31___drop_path_1 = self.getattr_L__mod___blocks___31___drop_path(mul_95);  mul_95 = None
    x_357 = x_350 + getattr_l__mod___blocks___31___drop_path_1;  x_350 = getattr_l__mod___blocks___31___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___32___gamma_1 = self.getattr_L__mod___blocks___32___gamma_1
    getattr_l__mod___blocks___32___norm1 = self.getattr_L__mod___blocks___32___norm1(x_357)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___32___attn_qkv = self.getattr_L__mod___blocks___32___attn_qkv(getattr_l__mod___blocks___32___norm1);  getattr_l__mod___blocks___32___norm1 = None
    reshape_64 = getattr_l__mod___blocks___32___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___32___attn_qkv = None
    qkv_32 = reshape_64.permute(2, 0, 3, 1, 4);  reshape_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_96 = qkv_32[0]
    q_32 = getitem_96 * 0.14433756729740643;  getitem_96 = None
    k_32 = qkv_32[1]
    v_32 = qkv_32[2];  qkv_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_65 = k_32.transpose(-2, -1);  k_32 = None
    attn_160 = q_32 @ transpose_65;  q_32 = transpose_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_161 = attn_160.permute(0, 2, 3, 1);  attn_160 = None
    getattr_l__mod___blocks___32___attn_proj_l = self.getattr_L__mod___blocks___32___attn_proj_l(permute_161);  permute_161 = None
    attn_161 = getattr_l__mod___blocks___32___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___32___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_162 = attn_161.softmax(dim = -1);  attn_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_163 = attn_162.permute(0, 2, 3, 1);  attn_162 = None
    getattr_l__mod___blocks___32___attn_proj_w = self.getattr_L__mod___blocks___32___attn_proj_w(permute_163);  permute_163 = None
    attn_163 = getattr_l__mod___blocks___32___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___32___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_164 = self.getattr_L__mod___blocks___32___attn_attn_drop(attn_163);  attn_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_65 = attn_164 @ v_32;  attn_164 = v_32 = None
    transpose_66 = matmul_65.transpose(1, 2);  matmul_65 = None
    x_358 = transpose_66.reshape(8, 576, 768);  transpose_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_359 = self.getattr_L__mod___blocks___32___attn_proj(x_358);  x_358 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_360 = self.getattr_L__mod___blocks___32___attn_proj_drop(x_359);  x_359 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_97 = getattr_l__mod___blocks___32___gamma_1 * x_360;  getattr_l__mod___blocks___32___gamma_1 = x_360 = None
    getattr_l__mod___blocks___32___drop_path = self.getattr_L__mod___blocks___32___drop_path(mul_97);  mul_97 = None
    x_361 = x_357 + getattr_l__mod___blocks___32___drop_path;  x_357 = getattr_l__mod___blocks___32___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___32___gamma_2 = self.getattr_L__mod___blocks___32___gamma_2
    getattr_l__mod___blocks___32___norm2 = self.getattr_L__mod___blocks___32___norm2(x_361)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_362 = self.getattr_L__mod___blocks___32___mlp_fc1(getattr_l__mod___blocks___32___norm2);  getattr_l__mod___blocks___32___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_363 = self.getattr_L__mod___blocks___32___mlp_act(x_362);  x_362 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_364 = self.getattr_L__mod___blocks___32___mlp_drop1(x_363);  x_363 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_365 = self.getattr_L__mod___blocks___32___mlp_norm(x_364);  x_364 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_366 = self.getattr_L__mod___blocks___32___mlp_fc2(x_365);  x_365 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_367 = self.getattr_L__mod___blocks___32___mlp_drop2(x_366);  x_366 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_98 = getattr_l__mod___blocks___32___gamma_2 * x_367;  getattr_l__mod___blocks___32___gamma_2 = x_367 = None
    getattr_l__mod___blocks___32___drop_path_1 = self.getattr_L__mod___blocks___32___drop_path(mul_98);  mul_98 = None
    x_368 = x_361 + getattr_l__mod___blocks___32___drop_path_1;  x_361 = getattr_l__mod___blocks___32___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___33___gamma_1 = self.getattr_L__mod___blocks___33___gamma_1
    getattr_l__mod___blocks___33___norm1 = self.getattr_L__mod___blocks___33___norm1(x_368)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___33___attn_qkv = self.getattr_L__mod___blocks___33___attn_qkv(getattr_l__mod___blocks___33___norm1);  getattr_l__mod___blocks___33___norm1 = None
    reshape_66 = getattr_l__mod___blocks___33___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___33___attn_qkv = None
    qkv_33 = reshape_66.permute(2, 0, 3, 1, 4);  reshape_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_99 = qkv_33[0]
    q_33 = getitem_99 * 0.14433756729740643;  getitem_99 = None
    k_33 = qkv_33[1]
    v_33 = qkv_33[2];  qkv_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_67 = k_33.transpose(-2, -1);  k_33 = None
    attn_165 = q_33 @ transpose_67;  q_33 = transpose_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_166 = attn_165.permute(0, 2, 3, 1);  attn_165 = None
    getattr_l__mod___blocks___33___attn_proj_l = self.getattr_L__mod___blocks___33___attn_proj_l(permute_166);  permute_166 = None
    attn_166 = getattr_l__mod___blocks___33___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___33___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_167 = attn_166.softmax(dim = -1);  attn_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_168 = attn_167.permute(0, 2, 3, 1);  attn_167 = None
    getattr_l__mod___blocks___33___attn_proj_w = self.getattr_L__mod___blocks___33___attn_proj_w(permute_168);  permute_168 = None
    attn_168 = getattr_l__mod___blocks___33___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___33___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_169 = self.getattr_L__mod___blocks___33___attn_attn_drop(attn_168);  attn_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_67 = attn_169 @ v_33;  attn_169 = v_33 = None
    transpose_68 = matmul_67.transpose(1, 2);  matmul_67 = None
    x_369 = transpose_68.reshape(8, 576, 768);  transpose_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_370 = self.getattr_L__mod___blocks___33___attn_proj(x_369);  x_369 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_371 = self.getattr_L__mod___blocks___33___attn_proj_drop(x_370);  x_370 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_100 = getattr_l__mod___blocks___33___gamma_1 * x_371;  getattr_l__mod___blocks___33___gamma_1 = x_371 = None
    getattr_l__mod___blocks___33___drop_path = self.getattr_L__mod___blocks___33___drop_path(mul_100);  mul_100 = None
    x_372 = x_368 + getattr_l__mod___blocks___33___drop_path;  x_368 = getattr_l__mod___blocks___33___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___33___gamma_2 = self.getattr_L__mod___blocks___33___gamma_2
    getattr_l__mod___blocks___33___norm2 = self.getattr_L__mod___blocks___33___norm2(x_372)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_373 = self.getattr_L__mod___blocks___33___mlp_fc1(getattr_l__mod___blocks___33___norm2);  getattr_l__mod___blocks___33___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_374 = self.getattr_L__mod___blocks___33___mlp_act(x_373);  x_373 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_375 = self.getattr_L__mod___blocks___33___mlp_drop1(x_374);  x_374 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_376 = self.getattr_L__mod___blocks___33___mlp_norm(x_375);  x_375 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_377 = self.getattr_L__mod___blocks___33___mlp_fc2(x_376);  x_376 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_378 = self.getattr_L__mod___blocks___33___mlp_drop2(x_377);  x_377 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_101 = getattr_l__mod___blocks___33___gamma_2 * x_378;  getattr_l__mod___blocks___33___gamma_2 = x_378 = None
    getattr_l__mod___blocks___33___drop_path_1 = self.getattr_L__mod___blocks___33___drop_path(mul_101);  mul_101 = None
    x_379 = x_372 + getattr_l__mod___blocks___33___drop_path_1;  x_372 = getattr_l__mod___blocks___33___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___34___gamma_1 = self.getattr_L__mod___blocks___34___gamma_1
    getattr_l__mod___blocks___34___norm1 = self.getattr_L__mod___blocks___34___norm1(x_379)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___34___attn_qkv = self.getattr_L__mod___blocks___34___attn_qkv(getattr_l__mod___blocks___34___norm1);  getattr_l__mod___blocks___34___norm1 = None
    reshape_68 = getattr_l__mod___blocks___34___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___34___attn_qkv = None
    qkv_34 = reshape_68.permute(2, 0, 3, 1, 4);  reshape_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_102 = qkv_34[0]
    q_34 = getitem_102 * 0.14433756729740643;  getitem_102 = None
    k_34 = qkv_34[1]
    v_34 = qkv_34[2];  qkv_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_69 = k_34.transpose(-2, -1);  k_34 = None
    attn_170 = q_34 @ transpose_69;  q_34 = transpose_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_171 = attn_170.permute(0, 2, 3, 1);  attn_170 = None
    getattr_l__mod___blocks___34___attn_proj_l = self.getattr_L__mod___blocks___34___attn_proj_l(permute_171);  permute_171 = None
    attn_171 = getattr_l__mod___blocks___34___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___34___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_172 = attn_171.softmax(dim = -1);  attn_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_173 = attn_172.permute(0, 2, 3, 1);  attn_172 = None
    getattr_l__mod___blocks___34___attn_proj_w = self.getattr_L__mod___blocks___34___attn_proj_w(permute_173);  permute_173 = None
    attn_173 = getattr_l__mod___blocks___34___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___34___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_174 = self.getattr_L__mod___blocks___34___attn_attn_drop(attn_173);  attn_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_69 = attn_174 @ v_34;  attn_174 = v_34 = None
    transpose_70 = matmul_69.transpose(1, 2);  matmul_69 = None
    x_380 = transpose_70.reshape(8, 576, 768);  transpose_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_381 = self.getattr_L__mod___blocks___34___attn_proj(x_380);  x_380 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_382 = self.getattr_L__mod___blocks___34___attn_proj_drop(x_381);  x_381 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_103 = getattr_l__mod___blocks___34___gamma_1 * x_382;  getattr_l__mod___blocks___34___gamma_1 = x_382 = None
    getattr_l__mod___blocks___34___drop_path = self.getattr_L__mod___blocks___34___drop_path(mul_103);  mul_103 = None
    x_383 = x_379 + getattr_l__mod___blocks___34___drop_path;  x_379 = getattr_l__mod___blocks___34___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___34___gamma_2 = self.getattr_L__mod___blocks___34___gamma_2
    getattr_l__mod___blocks___34___norm2 = self.getattr_L__mod___blocks___34___norm2(x_383)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_384 = self.getattr_L__mod___blocks___34___mlp_fc1(getattr_l__mod___blocks___34___norm2);  getattr_l__mod___blocks___34___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_385 = self.getattr_L__mod___blocks___34___mlp_act(x_384);  x_384 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_386 = self.getattr_L__mod___blocks___34___mlp_drop1(x_385);  x_385 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_387 = self.getattr_L__mod___blocks___34___mlp_norm(x_386);  x_386 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_388 = self.getattr_L__mod___blocks___34___mlp_fc2(x_387);  x_387 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_389 = self.getattr_L__mod___blocks___34___mlp_drop2(x_388);  x_388 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_104 = getattr_l__mod___blocks___34___gamma_2 * x_389;  getattr_l__mod___blocks___34___gamma_2 = x_389 = None
    getattr_l__mod___blocks___34___drop_path_1 = self.getattr_L__mod___blocks___34___drop_path(mul_104);  mul_104 = None
    x_390 = x_383 + getattr_l__mod___blocks___34___drop_path_1;  x_383 = getattr_l__mod___blocks___34___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    getattr_l__mod___blocks___35___gamma_1 = self.getattr_L__mod___blocks___35___gamma_1
    getattr_l__mod___blocks___35___norm1 = self.getattr_L__mod___blocks___35___norm1(x_390)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:139, code: qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___blocks___35___attn_qkv = self.getattr_L__mod___blocks___35___attn_qkv(getattr_l__mod___blocks___35___norm1);  getattr_l__mod___blocks___35___norm1 = None
    reshape_70 = getattr_l__mod___blocks___35___attn_qkv.reshape(8, 576, 3, 16, 48);  getattr_l__mod___blocks___35___attn_qkv = None
    qkv_35 = reshape_70.permute(2, 0, 3, 1, 4);  reshape_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:140, code: q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
    getitem_105 = qkv_35[0]
    q_35 = getitem_105 * 0.14433756729740643;  getitem_105 = None
    k_35 = qkv_35[1]
    v_35 = qkv_35[2];  qkv_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:142, code: attn = q @ k.transpose(-2, -1)
    transpose_71 = k_35.transpose(-2, -1);  k_35 = None
    attn_175 = q_35 @ transpose_71;  q_35 = transpose_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:144, code: attn = self.proj_l(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_176 = attn_175.permute(0, 2, 3, 1);  attn_175 = None
    getattr_l__mod___blocks___35___attn_proj_l = self.getattr_L__mod___blocks___35___attn_proj_l(permute_176);  permute_176 = None
    attn_176 = getattr_l__mod___blocks___35___attn_proj_l.permute(0, 3, 1, 2);  getattr_l__mod___blocks___35___attn_proj_l = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:146, code: attn = attn.softmax(dim=-1)
    attn_177 = attn_176.softmax(dim = -1);  attn_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:148, code: attn = self.proj_w(attn.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    permute_178 = attn_177.permute(0, 2, 3, 1);  attn_177 = None
    getattr_l__mod___blocks___35___attn_proj_w = self.getattr_L__mod___blocks___35___attn_proj_w(permute_178);  permute_178 = None
    attn_178 = getattr_l__mod___blocks___35___attn_proj_w.permute(0, 3, 1, 2);  getattr_l__mod___blocks___35___attn_proj_w = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:149, code: attn = self.attn_drop(attn)
    attn_179 = self.getattr_L__mod___blocks___35___attn_attn_drop(attn_178);  attn_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:151, code: x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    matmul_71 = attn_179 @ v_35;  attn_179 = v_35 = None
    transpose_72 = matmul_71.transpose(1, 2);  matmul_71 = None
    x_391 = transpose_72.reshape(8, 576, 768);  transpose_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:152, code: x = self.proj(x)
    x_392 = self.getattr_L__mod___blocks___35___attn_proj(x_391);  x_391 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:153, code: x = self.proj_drop(x)
    x_393 = self.getattr_L__mod___blocks___35___attn_proj_drop(x_392);  x_392 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:197, code: x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
    mul_106 = getattr_l__mod___blocks___35___gamma_1 * x_393;  getattr_l__mod___blocks___35___gamma_1 = x_393 = None
    getattr_l__mod___blocks___35___drop_path = self.getattr_L__mod___blocks___35___drop_path(mul_106);  mul_106 = None
    x_394 = x_390 + getattr_l__mod___blocks___35___drop_path;  x_390 = getattr_l__mod___blocks___35___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    getattr_l__mod___blocks___35___gamma_2 = self.getattr_L__mod___blocks___35___gamma_2
    getattr_l__mod___blocks___35___norm2 = self.getattr_L__mod___blocks___35___norm2(x_394)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_395 = self.getattr_L__mod___blocks___35___mlp_fc1(getattr_l__mod___blocks___35___norm2);  getattr_l__mod___blocks___35___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_396 = self.getattr_L__mod___blocks___35___mlp_act(x_395);  x_395 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_397 = self.getattr_L__mod___blocks___35___mlp_drop1(x_396);  x_396 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_398 = self.getattr_L__mod___blocks___35___mlp_norm(x_397);  x_397 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_399 = self.getattr_L__mod___blocks___35___mlp_fc2(x_398);  x_398 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_400 = self.getattr_L__mod___blocks___35___mlp_drop2(x_399);  x_399 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:198, code: x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
    mul_107 = getattr_l__mod___blocks___35___gamma_2 * x_400;  getattr_l__mod___blocks___35___gamma_2 = x_400 = None
    getattr_l__mod___blocks___35___drop_path_1 = self.getattr_L__mod___blocks___35___drop_path(mul_107);  mul_107 = None
    x_402 = x_394 + getattr_l__mod___blocks___35___drop_path_1;  x_394 = getattr_l__mod___blocks___35___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:347, code: cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    l__mod___cls_token = self.L__mod___cls_token
    cls_tokens = l__mod___cls_token.expand(8, -1, -1);  l__mod___cls_token = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:109, code: u = torch.cat((x_cls, x), dim=1)
    u = torch.cat((cls_tokens, x_402), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    l__mod___blocks_token_only_0_gamma_1 = self.L__mod___blocks_token_only_0_gamma_1
    l__mod___blocks_token_only_0_norm1 = self.L__mod___blocks_token_only_0_norm1(u);  u = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    getitem_108 = l__mod___blocks_token_only_0_norm1[(slice(None, None, None), 0)]
    l__mod___blocks_token_only_0_attn_q = self.L__mod___blocks_token_only_0_attn_q(getitem_108);  getitem_108 = None
    unsqueeze = l__mod___blocks_token_only_0_attn_q.unsqueeze(1);  l__mod___blocks_token_only_0_attn_q = None
    reshape_72 = unsqueeze.reshape(8, 1, 16, 48);  unsqueeze = None
    q_36 = reshape_72.permute(0, 2, 1, 3);  reshape_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_token_only_0_attn_k = self.L__mod___blocks_token_only_0_attn_k(l__mod___blocks_token_only_0_norm1)
    reshape_73 = l__mod___blocks_token_only_0_attn_k.reshape(8, 577, 16, 48);  l__mod___blocks_token_only_0_attn_k = None
    k_36 = reshape_73.permute(0, 2, 1, 3);  reshape_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_token_only_0_attn_v = self.L__mod___blocks_token_only_0_attn_v(l__mod___blocks_token_only_0_norm1);  l__mod___blocks_token_only_0_norm1 = None
    reshape_74 = l__mod___blocks_token_only_0_attn_v.reshape(8, 577, 16, 48);  l__mod___blocks_token_only_0_attn_v = None
    v_36 = reshape_74.permute(0, 2, 1, 3);  reshape_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    x_cls = torch._C._nn.scaled_dot_product_attention(q_36, k_36, v_36, dropout_p = 0.0);  q_36 = k_36 = v_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    transpose_73 = x_cls.transpose(1, 2);  x_cls = None
    x_cls_1 = transpose_73.reshape(8, 1, 768);  transpose_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    x_cls_2 = self.L__mod___blocks_token_only_0_attn_proj(x_cls_1);  x_cls_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:64, code: x_cls = self.proj_drop(x_cls)
    x_cls_3 = self.L__mod___blocks_token_only_0_attn_proj_drop(x_cls_2);  x_cls_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    mul_108 = l__mod___blocks_token_only_0_gamma_1 * x_cls_3;  l__mod___blocks_token_only_0_gamma_1 = x_cls_3 = None
    l__mod___blocks_token_only_0_drop_path = self.L__mod___blocks_token_only_0_drop_path(mul_108);  mul_108 = None
    x_cls_4 = cls_tokens + l__mod___blocks_token_only_0_drop_path;  cls_tokens = l__mod___blocks_token_only_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    l__mod___blocks_token_only_0_gamma_2 = self.L__mod___blocks_token_only_0_gamma_2
    l__mod___blocks_token_only_0_norm2 = self.L__mod___blocks_token_only_0_norm2(x_cls_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_403 = self.L__mod___blocks_token_only_0_mlp_fc1(l__mod___blocks_token_only_0_norm2);  l__mod___blocks_token_only_0_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_404 = self.L__mod___blocks_token_only_0_mlp_act(x_403);  x_403 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_405 = self.L__mod___blocks_token_only_0_mlp_drop1(x_404);  x_404 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_406 = self.L__mod___blocks_token_only_0_mlp_norm(x_405);  x_405 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_407 = self.L__mod___blocks_token_only_0_mlp_fc2(x_406);  x_406 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_408 = self.L__mod___blocks_token_only_0_mlp_drop2(x_407);  x_407 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    mul_109 = l__mod___blocks_token_only_0_gamma_2 * x_408;  l__mod___blocks_token_only_0_gamma_2 = x_408 = None
    l__mod___blocks_token_only_0_drop_path_1 = self.L__mod___blocks_token_only_0_drop_path(mul_109);  mul_109 = None
    cls_tokens_1 = x_cls_4 + l__mod___blocks_token_only_0_drop_path_1;  x_cls_4 = l__mod___blocks_token_only_0_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:109, code: u = torch.cat((x_cls, x), dim=1)
    u_1 = torch.cat((cls_tokens_1, x_402), dim = 1)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    l__mod___blocks_token_only_1_gamma_1 = self.L__mod___blocks_token_only_1_gamma_1
    l__mod___blocks_token_only_1_norm1 = self.L__mod___blocks_token_only_1_norm1(u_1);  u_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:46, code: q = self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    getitem_109 = l__mod___blocks_token_only_1_norm1[(slice(None, None, None), 0)]
    l__mod___blocks_token_only_1_attn_q = self.L__mod___blocks_token_only_1_attn_q(getitem_109);  getitem_109 = None
    unsqueeze_1 = l__mod___blocks_token_only_1_attn_q.unsqueeze(1);  l__mod___blocks_token_only_1_attn_q = None
    reshape_76 = unsqueeze_1.reshape(8, 1, 16, 48);  unsqueeze_1 = None
    q_37 = reshape_76.permute(0, 2, 1, 3);  reshape_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:47, code: k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_token_only_1_attn_k = self.L__mod___blocks_token_only_1_attn_k(l__mod___blocks_token_only_1_norm1)
    reshape_77 = l__mod___blocks_token_only_1_attn_k.reshape(8, 577, 16, 48);  l__mod___blocks_token_only_1_attn_k = None
    k_37 = reshape_77.permute(0, 2, 1, 3);  reshape_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:48, code: v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    l__mod___blocks_token_only_1_attn_v = self.L__mod___blocks_token_only_1_attn_v(l__mod___blocks_token_only_1_norm1);  l__mod___blocks_token_only_1_norm1 = None
    reshape_78 = l__mod___blocks_token_only_1_attn_v.reshape(8, 577, 16, 48);  l__mod___blocks_token_only_1_attn_v = None
    v_37 = reshape_78.permute(0, 2, 1, 3);  reshape_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:51, code: x_cls = torch.nn.functional.scaled_dot_product_attention(
    x_cls_6 = torch._C._nn.scaled_dot_product_attention(q_37, k_37, v_37, dropout_p = 0.0);  q_37 = k_37 = v_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:62, code: x_cls = x_cls.transpose(1, 2).reshape(B, 1, C)
    transpose_74 = x_cls_6.transpose(1, 2);  x_cls_6 = None
    x_cls_7 = transpose_74.reshape(8, 1, 768);  transpose_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:63, code: x_cls = self.proj(x_cls)
    x_cls_8 = self.L__mod___blocks_token_only_1_attn_proj(x_cls_7);  x_cls_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:64, code: x_cls = self.proj_drop(x_cls)
    x_cls_9 = self.L__mod___blocks_token_only_1_attn_proj_drop(x_cls_8);  x_cls_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:110, code: x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))
    mul_110 = l__mod___blocks_token_only_1_gamma_1 * x_cls_9;  l__mod___blocks_token_only_1_gamma_1 = x_cls_9 = None
    l__mod___blocks_token_only_1_drop_path = self.L__mod___blocks_token_only_1_drop_path(mul_110);  mul_110 = None
    x_cls_10 = cls_tokens_1 + l__mod___blocks_token_only_1_drop_path;  cls_tokens_1 = l__mod___blocks_token_only_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    l__mod___blocks_token_only_1_gamma_2 = self.L__mod___blocks_token_only_1_gamma_2
    l__mod___blocks_token_only_1_norm2 = self.L__mod___blocks_token_only_1_norm2(x_cls_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_409 = self.L__mod___blocks_token_only_1_mlp_fc1(l__mod___blocks_token_only_1_norm2);  l__mod___blocks_token_only_1_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_410 = self.L__mod___blocks_token_only_1_mlp_act(x_409);  x_409 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_411 = self.L__mod___blocks_token_only_1_mlp_drop1(x_410);  x_410 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_412 = self.L__mod___blocks_token_only_1_mlp_norm(x_411);  x_411 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_413 = self.L__mod___blocks_token_only_1_mlp_fc2(x_412);  x_412 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_414 = self.L__mod___blocks_token_only_1_mlp_drop2(x_413);  x_413 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:111, code: x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
    mul_111 = l__mod___blocks_token_only_1_gamma_2 * x_414;  l__mod___blocks_token_only_1_gamma_2 = x_414 = None
    l__mod___blocks_token_only_1_drop_path_1 = self.L__mod___blocks_token_only_1_drop_path(mul_111);  mul_111 = None
    cls_tokens_2 = x_cls_10 + l__mod___blocks_token_only_1_drop_path_1;  x_cls_10 = l__mod___blocks_token_only_1_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:350, code: x = torch.cat((cls_tokens, x), dim=1)
    x_415 = torch.cat((cls_tokens_2, x_402), dim = 1);  cls_tokens_2 = x_402 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:351, code: x = self.norm(x)
    x_417 = self.L__mod___norm(x_415);  x_415 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:356, code: x = x[:, 1:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
    x_418 = x_417[(slice(None, None, None), 0)];  x_417 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:357, code: x = self.head_drop(x)
    x_419 = self.L__mod___head_drop(x_418);  x_418 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/cait.py:358, code: return x if pre_logits else self.head(x)
    x_420 = self.L__mod___head(x_419);  x_419 = None
    return (x_420,)
    