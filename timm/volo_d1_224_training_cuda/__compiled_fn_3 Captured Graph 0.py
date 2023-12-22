from __future__ import annotations



def forward(self, L_cloned_inputs_0_ : torch.Tensor):
    l_cloned_inputs_0_ = L_cloned_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:357, code: x = self.conv(x)
    l__mod___patch_embed_conv_0 = self.L__mod___patch_embed_conv_0(l_cloned_inputs_0_);  l_cloned_inputs_0_ = None
    l__mod___patch_embed_conv_1 = self.L__mod___patch_embed_conv_1(l__mod___patch_embed_conv_0);  l__mod___patch_embed_conv_0 = None
    l__mod___patch_embed_conv_2 = self.L__mod___patch_embed_conv_2(l__mod___patch_embed_conv_1);  l__mod___patch_embed_conv_1 = None
    l__mod___patch_embed_conv_3 = self.L__mod___patch_embed_conv_3(l__mod___patch_embed_conv_2);  l__mod___patch_embed_conv_2 = None
    l__mod___patch_embed_conv_4 = self.L__mod___patch_embed_conv_4(l__mod___patch_embed_conv_3);  l__mod___patch_embed_conv_3 = None
    l__mod___patch_embed_conv_5 = self.L__mod___patch_embed_conv_5(l__mod___patch_embed_conv_4);  l__mod___patch_embed_conv_4 = None
    l__mod___patch_embed_conv_6 = self.L__mod___patch_embed_conv_6(l__mod___patch_embed_conv_5);  l__mod___patch_embed_conv_5 = None
    l__mod___patch_embed_conv_7 = self.L__mod___patch_embed_conv_7(l__mod___patch_embed_conv_6);  l__mod___patch_embed_conv_6 = None
    x = self.L__mod___patch_embed_conv_8(l__mod___patch_embed_conv_7);  l__mod___patch_embed_conv_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:358, code: x = self.proj(x)  # B, C, H, W
    x_1 = self.L__mod___patch_embed_proj(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:695, code: x = self.patch_embed(x).permute(0, 2, 3, 1)  # B,C,H,W-> B,H,W,C
    x_2 = x_1.permute(0, 2, 3, 1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_0___0___norm1 = self.getattr_L__mod___network_0___0___norm1(x_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    getattr_l__mod___network_0___0___attn_v = self.getattr_L__mod___network_0___0___attn_v(getattr_l__mod___network_0___0___norm1)
    v = getattr_l__mod___network_0___0___attn_v.permute(0, 3, 1, 2);  getattr_l__mod___network_0___0___attn_v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    getattr_l__mod___network_0___0___attn_unfold = self.getattr_L__mod___network_0___0___attn_unfold(v);  v = None
    reshape = getattr_l__mod___network_0___0___attn_unfold.reshape(8, 6, 32, 9, 196);  getattr_l__mod___network_0___0___attn_unfold = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    v_1 = reshape.permute(0, 1, 4, 3, 2);  reshape = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_3 = getattr_l__mod___network_0___0___norm1.permute(0, 3, 1, 2);  getattr_l__mod___network_0___0___norm1 = None
    getattr_l__mod___network_0___0___attn_pool = self.getattr_L__mod___network_0___0___attn_pool(permute_3);  permute_3 = None
    attn = getattr_l__mod___network_0___0___attn_pool.permute(0, 2, 3, 1);  getattr_l__mod___network_0___0___attn_pool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    getattr_l__mod___network_0___0___attn_attn = self.getattr_L__mod___network_0___0___attn_attn(attn);  attn = None
    reshape_1 = getattr_l__mod___network_0___0___attn_attn.reshape(8, 196, 6, 9, 9);  getattr_l__mod___network_0___0___attn_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    attn_1 = reshape_1.permute(0, 2, 1, 3, 4);  reshape_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    attn_2 = attn_1 * 0.1767766952966369;  attn_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    attn_3 = attn_2.softmax(dim = -1);  attn_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:85, code: attn = self.attn_drop(attn)
    attn_4 = self.getattr_L__mod___network_0___0___attn_attn_drop(attn_3);  attn_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    matmul = attn_4 @ v_1;  attn_4 = v_1 = None
    permute_6 = matmul.permute(0, 1, 4, 3, 2);  matmul = None
    x_3 = permute_6.reshape(8, 1728, 196);  permute_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    x_4 = torch.nn.functional.fold(x_3, output_size = (28, 28), kernel_size = 3, padding = 1, stride = 2);  x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_7 = x_4.permute(0, 2, 3, 1);  x_4 = None
    x_5 = self.getattr_L__mod___network_0___0___attn_proj(permute_7);  permute_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:91, code: x = self.proj_drop(x)
    x_6 = self.getattr_L__mod___network_0___0___attn_proj_drop(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_0___0___drop_path = self.getattr_L__mod___network_0___0___drop_path(x_6);  x_6 = None
    x_7 = x_2 + getattr_l__mod___network_0___0___drop_path;  x_2 = getattr_l__mod___network_0___0___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_0___0___norm2 = self.getattr_L__mod___network_0___0___norm2(x_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_8 = self.getattr_L__mod___network_0___0___mlp_fc1(getattr_l__mod___network_0___0___norm2);  getattr_l__mod___network_0___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_9 = self.getattr_L__mod___network_0___0___mlp_act(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_10 = self.getattr_L__mod___network_0___0___mlp_drop1(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_11 = self.getattr_L__mod___network_0___0___mlp_norm(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_12 = self.getattr_L__mod___network_0___0___mlp_fc2(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_13 = self.getattr_L__mod___network_0___0___mlp_drop2(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_0___0___drop_path_1 = self.getattr_L__mod___network_0___0___drop_path(x_13);  x_13 = None
    x_14 = x_7 + getattr_l__mod___network_0___0___drop_path_1;  x_7 = getattr_l__mod___network_0___0___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_0___1___norm1 = self.getattr_L__mod___network_0___1___norm1(x_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    getattr_l__mod___network_0___1___attn_v = self.getattr_L__mod___network_0___1___attn_v(getattr_l__mod___network_0___1___norm1)
    v_2 = getattr_l__mod___network_0___1___attn_v.permute(0, 3, 1, 2);  getattr_l__mod___network_0___1___attn_v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    getattr_l__mod___network_0___1___attn_unfold = self.getattr_L__mod___network_0___1___attn_unfold(v_2);  v_2 = None
    reshape_3 = getattr_l__mod___network_0___1___attn_unfold.reshape(8, 6, 32, 9, 196);  getattr_l__mod___network_0___1___attn_unfold = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    v_3 = reshape_3.permute(0, 1, 4, 3, 2);  reshape_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_10 = getattr_l__mod___network_0___1___norm1.permute(0, 3, 1, 2);  getattr_l__mod___network_0___1___norm1 = None
    getattr_l__mod___network_0___1___attn_pool = self.getattr_L__mod___network_0___1___attn_pool(permute_10);  permute_10 = None
    attn_5 = getattr_l__mod___network_0___1___attn_pool.permute(0, 2, 3, 1);  getattr_l__mod___network_0___1___attn_pool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    getattr_l__mod___network_0___1___attn_attn = self.getattr_L__mod___network_0___1___attn_attn(attn_5);  attn_5 = None
    reshape_4 = getattr_l__mod___network_0___1___attn_attn.reshape(8, 196, 6, 9, 9);  getattr_l__mod___network_0___1___attn_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    attn_6 = reshape_4.permute(0, 2, 1, 3, 4);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    attn_7 = attn_6 * 0.1767766952966369;  attn_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    attn_8 = attn_7.softmax(dim = -1);  attn_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:85, code: attn = self.attn_drop(attn)
    attn_9 = self.getattr_L__mod___network_0___1___attn_attn_drop(attn_8);  attn_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    matmul_1 = attn_9 @ v_3;  attn_9 = v_3 = None
    permute_13 = matmul_1.permute(0, 1, 4, 3, 2);  matmul_1 = None
    x_15 = permute_13.reshape(8, 1728, 196);  permute_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    x_16 = torch.nn.functional.fold(x_15, output_size = (28, 28), kernel_size = 3, padding = 1, stride = 2);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_14 = x_16.permute(0, 2, 3, 1);  x_16 = None
    x_17 = self.getattr_L__mod___network_0___1___attn_proj(permute_14);  permute_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:91, code: x = self.proj_drop(x)
    x_18 = self.getattr_L__mod___network_0___1___attn_proj_drop(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_0___1___drop_path = self.getattr_L__mod___network_0___1___drop_path(x_18);  x_18 = None
    x_19 = x_14 + getattr_l__mod___network_0___1___drop_path;  x_14 = getattr_l__mod___network_0___1___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_0___1___norm2 = self.getattr_L__mod___network_0___1___norm2(x_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_20 = self.getattr_L__mod___network_0___1___mlp_fc1(getattr_l__mod___network_0___1___norm2);  getattr_l__mod___network_0___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_21 = self.getattr_L__mod___network_0___1___mlp_act(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_22 = self.getattr_L__mod___network_0___1___mlp_drop1(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_23 = self.getattr_L__mod___network_0___1___mlp_norm(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_24 = self.getattr_L__mod___network_0___1___mlp_fc2(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_25 = self.getattr_L__mod___network_0___1___mlp_drop2(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_0___1___drop_path_1 = self.getattr_L__mod___network_0___1___drop_path(x_25);  x_25 = None
    x_26 = x_19 + getattr_l__mod___network_0___1___drop_path_1;  x_19 = getattr_l__mod___network_0___1___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_0___2___norm1 = self.getattr_L__mod___network_0___2___norm1(x_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    getattr_l__mod___network_0___2___attn_v = self.getattr_L__mod___network_0___2___attn_v(getattr_l__mod___network_0___2___norm1)
    v_4 = getattr_l__mod___network_0___2___attn_v.permute(0, 3, 1, 2);  getattr_l__mod___network_0___2___attn_v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    getattr_l__mod___network_0___2___attn_unfold = self.getattr_L__mod___network_0___2___attn_unfold(v_4);  v_4 = None
    reshape_6 = getattr_l__mod___network_0___2___attn_unfold.reshape(8, 6, 32, 9, 196);  getattr_l__mod___network_0___2___attn_unfold = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    v_5 = reshape_6.permute(0, 1, 4, 3, 2);  reshape_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_17 = getattr_l__mod___network_0___2___norm1.permute(0, 3, 1, 2);  getattr_l__mod___network_0___2___norm1 = None
    getattr_l__mod___network_0___2___attn_pool = self.getattr_L__mod___network_0___2___attn_pool(permute_17);  permute_17 = None
    attn_10 = getattr_l__mod___network_0___2___attn_pool.permute(0, 2, 3, 1);  getattr_l__mod___network_0___2___attn_pool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    getattr_l__mod___network_0___2___attn_attn = self.getattr_L__mod___network_0___2___attn_attn(attn_10);  attn_10 = None
    reshape_7 = getattr_l__mod___network_0___2___attn_attn.reshape(8, 196, 6, 9, 9);  getattr_l__mod___network_0___2___attn_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    attn_11 = reshape_7.permute(0, 2, 1, 3, 4);  reshape_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    attn_12 = attn_11 * 0.1767766952966369;  attn_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    attn_13 = attn_12.softmax(dim = -1);  attn_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:85, code: attn = self.attn_drop(attn)
    attn_14 = self.getattr_L__mod___network_0___2___attn_attn_drop(attn_13);  attn_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    matmul_2 = attn_14 @ v_5;  attn_14 = v_5 = None
    permute_20 = matmul_2.permute(0, 1, 4, 3, 2);  matmul_2 = None
    x_27 = permute_20.reshape(8, 1728, 196);  permute_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    x_28 = torch.nn.functional.fold(x_27, output_size = (28, 28), kernel_size = 3, padding = 1, stride = 2);  x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_21 = x_28.permute(0, 2, 3, 1);  x_28 = None
    x_29 = self.getattr_L__mod___network_0___2___attn_proj(permute_21);  permute_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:91, code: x = self.proj_drop(x)
    x_30 = self.getattr_L__mod___network_0___2___attn_proj_drop(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_0___2___drop_path = self.getattr_L__mod___network_0___2___drop_path(x_30);  x_30 = None
    x_31 = x_26 + getattr_l__mod___network_0___2___drop_path;  x_26 = getattr_l__mod___network_0___2___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_0___2___norm2 = self.getattr_L__mod___network_0___2___norm2(x_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_32 = self.getattr_L__mod___network_0___2___mlp_fc1(getattr_l__mod___network_0___2___norm2);  getattr_l__mod___network_0___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_33 = self.getattr_L__mod___network_0___2___mlp_act(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_34 = self.getattr_L__mod___network_0___2___mlp_drop1(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_35 = self.getattr_L__mod___network_0___2___mlp_norm(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_36 = self.getattr_L__mod___network_0___2___mlp_fc2(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_37 = self.getattr_L__mod___network_0___2___mlp_drop2(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_0___2___drop_path_1 = self.getattr_L__mod___network_0___2___drop_path(x_37);  x_37 = None
    x_38 = x_31 + getattr_l__mod___network_0___2___drop_path_1;  x_31 = getattr_l__mod___network_0___2___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_0___3___norm1 = self.getattr_L__mod___network_0___3___norm1(x_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:72, code: v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
    getattr_l__mod___network_0___3___attn_v = self.getattr_L__mod___network_0___3___attn_v(getattr_l__mod___network_0___3___norm1)
    v_6 = getattr_l__mod___network_0___3___attn_v.permute(0, 3, 1, 2);  getattr_l__mod___network_0___3___attn_v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:75, code: v = self.unfold(v).reshape(
    getattr_l__mod___network_0___3___attn_unfold = self.getattr_L__mod___network_0___3___attn_unfold(v_6);  v_6 = None
    reshape_9 = getattr_l__mod___network_0___3___attn_unfold.reshape(8, 6, 32, 9, 196);  getattr_l__mod___network_0___3___attn_unfold = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:77, code: self.kernel_size * self.kernel_size, h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
    v_7 = reshape_9.permute(0, 1, 4, 3, 2);  reshape_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:79, code: attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    permute_24 = getattr_l__mod___network_0___3___norm1.permute(0, 3, 1, 2);  getattr_l__mod___network_0___3___norm1 = None
    getattr_l__mod___network_0___3___attn_pool = self.getattr_L__mod___network_0___3___attn_pool(permute_24);  permute_24 = None
    attn_15 = getattr_l__mod___network_0___3___attn_pool.permute(0, 2, 3, 1);  getattr_l__mod___network_0___3___attn_pool = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:80, code: attn = self.attn(attn).reshape(
    getattr_l__mod___network_0___3___attn_attn = self.getattr_L__mod___network_0___3___attn_attn(attn_15);  attn_15 = None
    reshape_10 = getattr_l__mod___network_0___3___attn_attn.reshape(8, 196, 6, 9, 9);  getattr_l__mod___network_0___3___attn_attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:82, code: self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
    attn_16 = reshape_10.permute(0, 2, 1, 3, 4);  reshape_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:83, code: attn = attn * self.scale
    attn_17 = attn_16 * 0.1767766952966369;  attn_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:84, code: attn = attn.softmax(dim=-1)
    attn_18 = attn_17.softmax(dim = -1);  attn_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:85, code: attn = self.attn_drop(attn)
    attn_19 = self.getattr_L__mod___network_0___3___attn_attn_drop(attn_18);  attn_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:87, code: x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(B, C * self.kernel_size * self.kernel_size, h * w)
    matmul_3 = attn_19 @ v_7;  attn_19 = v_7 = None
    permute_27 = matmul_3.permute(0, 1, 4, 3, 2);  matmul_3 = None
    x_39 = permute_27.reshape(8, 1728, 196);  permute_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:88, code: x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)
    x_40 = torch.nn.functional.fold(x_39, output_size = (28, 28), kernel_size = 3, padding = 1, stride = 2);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:90, code: x = self.proj(x.permute(0, 2, 3, 1))
    permute_28 = x_40.permute(0, 2, 3, 1);  x_40 = None
    x_41 = self.getattr_L__mod___network_0___3___attn_proj(permute_28);  permute_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:91, code: x = self.proj_drop(x)
    x_42 = self.getattr_L__mod___network_0___3___attn_proj_drop(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:134, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_0___3___drop_path = self.getattr_L__mod___network_0___3___drop_path(x_42);  x_42 = None
    x_43 = x_38 + getattr_l__mod___network_0___3___drop_path;  x_38 = getattr_l__mod___network_0___3___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_0___3___norm2 = self.getattr_L__mod___network_0___3___norm2(x_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_44 = self.getattr_L__mod___network_0___3___mlp_fc1(getattr_l__mod___network_0___3___norm2);  getattr_l__mod___network_0___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_45 = self.getattr_L__mod___network_0___3___mlp_act(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_46 = self.getattr_L__mod___network_0___3___mlp_drop1(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_47 = self.getattr_L__mod___network_0___3___mlp_norm(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_48 = self.getattr_L__mod___network_0___3___mlp_fc2(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_49 = self.getattr_L__mod___network_0___3___mlp_drop2(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:135, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_0___3___drop_path_1 = self.getattr_L__mod___network_0___3___drop_path(x_49);  x_49 = None
    x_51 = x_43 + getattr_l__mod___network_0___3___drop_path_1;  x_43 = getattr_l__mod___network_0___3___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:371, code: x = x.permute(0, 3, 1, 2)
    x_52 = x_51.permute(0, 3, 1, 2);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:372, code: x = self.proj(x)  # B, C, H, W
    x_53 = self.L__mod___network_1_proj(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:373, code: x = x.permute(0, 2, 3, 1)
    x_55 = x_53.permute(0, 2, 3, 1);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:620, code: x = x + self.pos_embed
    l__mod___pos_embed = self.L__mod___pos_embed
    x_56 = x_55 + l__mod___pos_embed;  x_55 = l__mod___pos_embed = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:621, code: x = self.pos_drop(x)
    x_57 = self.L__mod___pos_drop(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_2___0___norm1 = self.getattr_L__mod___network_2___0___norm1(x_57)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_2___0___attn_qkv = self.getattr_L__mod___network_2___0___attn_qkv(getattr_l__mod___network_2___0___norm1);  getattr_l__mod___network_2___0___norm1 = None
    reshape_12 = getattr_l__mod___network_2___0___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_2___0___attn_qkv = None
    qkv = reshape_12.permute(2, 0, 3, 1, 4);  reshape_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind = qkv.unbind(0);  qkv = None
    q = unbind[0]
    k = unbind[1]
    v_8 = unbind[2];  unbind = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose = k.transpose(-2, -1);  k = None
    matmul_4 = q @ transpose;  q = transpose = None
    attn_20 = matmul_4 * 0.1767766952966369;  matmul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_21 = attn_20.softmax(dim = -1);  attn_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_22 = self.getattr_L__mod___network_2___0___attn_attn_drop(attn_21);  attn_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_5 = attn_22 @ v_8;  attn_22 = v_8 = None
    transpose_1 = matmul_5.transpose(1, 2);  matmul_5 = None
    x_58 = transpose_1.reshape(8, 14, 14, 384);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_59 = self.getattr_L__mod___network_2___0___attn_proj(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_60 = self.getattr_L__mod___network_2___0___attn_proj_drop(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_2___0___drop_path = self.getattr_L__mod___network_2___0___drop_path(x_60);  x_60 = None
    x_61 = x_57 + getattr_l__mod___network_2___0___drop_path;  x_57 = getattr_l__mod___network_2___0___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_2___0___norm2 = self.getattr_L__mod___network_2___0___norm2(x_61)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_62 = self.getattr_L__mod___network_2___0___mlp_fc1(getattr_l__mod___network_2___0___norm2);  getattr_l__mod___network_2___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_63 = self.getattr_L__mod___network_2___0___mlp_act(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_64 = self.getattr_L__mod___network_2___0___mlp_drop1(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_65 = self.getattr_L__mod___network_2___0___mlp_norm(x_64);  x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_66 = self.getattr_L__mod___network_2___0___mlp_fc2(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_67 = self.getattr_L__mod___network_2___0___mlp_drop2(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_2___0___drop_path_1 = self.getattr_L__mod___network_2___0___drop_path(x_67);  x_67 = None
    x_68 = x_61 + getattr_l__mod___network_2___0___drop_path_1;  x_61 = getattr_l__mod___network_2___0___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_2___1___norm1 = self.getattr_L__mod___network_2___1___norm1(x_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_2___1___attn_qkv = self.getattr_L__mod___network_2___1___attn_qkv(getattr_l__mod___network_2___1___norm1);  getattr_l__mod___network_2___1___norm1 = None
    reshape_14 = getattr_l__mod___network_2___1___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_2___1___attn_qkv = None
    qkv_1 = reshape_14.permute(2, 0, 3, 1, 4);  reshape_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_1 = qkv_1.unbind(0);  qkv_1 = None
    q_1 = unbind_1[0]
    k_1 = unbind_1[1]
    v_9 = unbind_1[2];  unbind_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_2 = k_1.transpose(-2, -1);  k_1 = None
    matmul_6 = q_1 @ transpose_2;  q_1 = transpose_2 = None
    attn_23 = matmul_6 * 0.1767766952966369;  matmul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_24 = attn_23.softmax(dim = -1);  attn_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_25 = self.getattr_L__mod___network_2___1___attn_attn_drop(attn_24);  attn_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_7 = attn_25 @ v_9;  attn_25 = v_9 = None
    transpose_3 = matmul_7.transpose(1, 2);  matmul_7 = None
    x_69 = transpose_3.reshape(8, 14, 14, 384);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_70 = self.getattr_L__mod___network_2___1___attn_proj(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_71 = self.getattr_L__mod___network_2___1___attn_proj_drop(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_2___1___drop_path = self.getattr_L__mod___network_2___1___drop_path(x_71);  x_71 = None
    x_72 = x_68 + getattr_l__mod___network_2___1___drop_path;  x_68 = getattr_l__mod___network_2___1___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_2___1___norm2 = self.getattr_L__mod___network_2___1___norm2(x_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_73 = self.getattr_L__mod___network_2___1___mlp_fc1(getattr_l__mod___network_2___1___norm2);  getattr_l__mod___network_2___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_74 = self.getattr_L__mod___network_2___1___mlp_act(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_75 = self.getattr_L__mod___network_2___1___mlp_drop1(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_76 = self.getattr_L__mod___network_2___1___mlp_norm(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_77 = self.getattr_L__mod___network_2___1___mlp_fc2(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_78 = self.getattr_L__mod___network_2___1___mlp_drop2(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_2___1___drop_path_1 = self.getattr_L__mod___network_2___1___drop_path(x_78);  x_78 = None
    x_79 = x_72 + getattr_l__mod___network_2___1___drop_path_1;  x_72 = getattr_l__mod___network_2___1___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_2___2___norm1 = self.getattr_L__mod___network_2___2___norm1(x_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_2___2___attn_qkv = self.getattr_L__mod___network_2___2___attn_qkv(getattr_l__mod___network_2___2___norm1);  getattr_l__mod___network_2___2___norm1 = None
    reshape_16 = getattr_l__mod___network_2___2___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_2___2___attn_qkv = None
    qkv_2 = reshape_16.permute(2, 0, 3, 1, 4);  reshape_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_2 = qkv_2.unbind(0);  qkv_2 = None
    q_2 = unbind_2[0]
    k_2 = unbind_2[1]
    v_10 = unbind_2[2];  unbind_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_4 = k_2.transpose(-2, -1);  k_2 = None
    matmul_8 = q_2 @ transpose_4;  q_2 = transpose_4 = None
    attn_26 = matmul_8 * 0.1767766952966369;  matmul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_27 = attn_26.softmax(dim = -1);  attn_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_28 = self.getattr_L__mod___network_2___2___attn_attn_drop(attn_27);  attn_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_9 = attn_28 @ v_10;  attn_28 = v_10 = None
    transpose_5 = matmul_9.transpose(1, 2);  matmul_9 = None
    x_80 = transpose_5.reshape(8, 14, 14, 384);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_81 = self.getattr_L__mod___network_2___2___attn_proj(x_80);  x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_82 = self.getattr_L__mod___network_2___2___attn_proj_drop(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_2___2___drop_path = self.getattr_L__mod___network_2___2___drop_path(x_82);  x_82 = None
    x_83 = x_79 + getattr_l__mod___network_2___2___drop_path;  x_79 = getattr_l__mod___network_2___2___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_2___2___norm2 = self.getattr_L__mod___network_2___2___norm2(x_83)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_84 = self.getattr_L__mod___network_2___2___mlp_fc1(getattr_l__mod___network_2___2___norm2);  getattr_l__mod___network_2___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_85 = self.getattr_L__mod___network_2___2___mlp_act(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_86 = self.getattr_L__mod___network_2___2___mlp_drop1(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_87 = self.getattr_L__mod___network_2___2___mlp_norm(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_88 = self.getattr_L__mod___network_2___2___mlp_fc2(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_89 = self.getattr_L__mod___network_2___2___mlp_drop2(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_2___2___drop_path_1 = self.getattr_L__mod___network_2___2___drop_path(x_89);  x_89 = None
    x_90 = x_83 + getattr_l__mod___network_2___2___drop_path_1;  x_83 = getattr_l__mod___network_2___2___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_2___3___norm1 = self.getattr_L__mod___network_2___3___norm1(x_90)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_2___3___attn_qkv = self.getattr_L__mod___network_2___3___attn_qkv(getattr_l__mod___network_2___3___norm1);  getattr_l__mod___network_2___3___norm1 = None
    reshape_18 = getattr_l__mod___network_2___3___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_2___3___attn_qkv = None
    qkv_3 = reshape_18.permute(2, 0, 3, 1, 4);  reshape_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_3 = qkv_3.unbind(0);  qkv_3 = None
    q_3 = unbind_3[0]
    k_3 = unbind_3[1]
    v_11 = unbind_3[2];  unbind_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_6 = k_3.transpose(-2, -1);  k_3 = None
    matmul_10 = q_3 @ transpose_6;  q_3 = transpose_6 = None
    attn_29 = matmul_10 * 0.1767766952966369;  matmul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_30 = attn_29.softmax(dim = -1);  attn_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_31 = self.getattr_L__mod___network_2___3___attn_attn_drop(attn_30);  attn_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_11 = attn_31 @ v_11;  attn_31 = v_11 = None
    transpose_7 = matmul_11.transpose(1, 2);  matmul_11 = None
    x_91 = transpose_7.reshape(8, 14, 14, 384);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_92 = self.getattr_L__mod___network_2___3___attn_proj(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_93 = self.getattr_L__mod___network_2___3___attn_proj_drop(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_2___3___drop_path = self.getattr_L__mod___network_2___3___drop_path(x_93);  x_93 = None
    x_94 = x_90 + getattr_l__mod___network_2___3___drop_path;  x_90 = getattr_l__mod___network_2___3___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_2___3___norm2 = self.getattr_L__mod___network_2___3___norm2(x_94)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_95 = self.getattr_L__mod___network_2___3___mlp_fc1(getattr_l__mod___network_2___3___norm2);  getattr_l__mod___network_2___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_96 = self.getattr_L__mod___network_2___3___mlp_act(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_97 = self.getattr_L__mod___network_2___3___mlp_drop1(x_96);  x_96 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_98 = self.getattr_L__mod___network_2___3___mlp_norm(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_99 = self.getattr_L__mod___network_2___3___mlp_fc2(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_100 = self.getattr_L__mod___network_2___3___mlp_drop2(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_2___3___drop_path_1 = self.getattr_L__mod___network_2___3___drop_path(x_100);  x_100 = None
    x_102 = x_94 + getattr_l__mod___network_2___3___drop_path_1;  x_94 = getattr_l__mod___network_2___3___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___0___norm1 = self.getattr_L__mod___network_3___0___norm1(x_102)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_3___0___attn_qkv = self.getattr_L__mod___network_3___0___attn_qkv(getattr_l__mod___network_3___0___norm1);  getattr_l__mod___network_3___0___norm1 = None
    reshape_20 = getattr_l__mod___network_3___0___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_3___0___attn_qkv = None
    qkv_4 = reshape_20.permute(2, 0, 3, 1, 4);  reshape_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_4 = qkv_4.unbind(0);  qkv_4 = None
    q_4 = unbind_4[0]
    k_4 = unbind_4[1]
    v_12 = unbind_4[2];  unbind_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_8 = k_4.transpose(-2, -1);  k_4 = None
    matmul_12 = q_4 @ transpose_8;  q_4 = transpose_8 = None
    attn_32 = matmul_12 * 0.1767766952966369;  matmul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_33 = attn_32.softmax(dim = -1);  attn_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_34 = self.getattr_L__mod___network_3___0___attn_attn_drop(attn_33);  attn_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_13 = attn_34 @ v_12;  attn_34 = v_12 = None
    transpose_9 = matmul_13.transpose(1, 2);  matmul_13 = None
    x_103 = transpose_9.reshape(8, 14, 14, 384);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_104 = self.getattr_L__mod___network_3___0___attn_proj(x_103);  x_103 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_105 = self.getattr_L__mod___network_3___0___attn_proj_drop(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___0___drop_path = self.getattr_L__mod___network_3___0___drop_path(x_105);  x_105 = None
    x_106 = x_102 + getattr_l__mod___network_3___0___drop_path;  x_102 = getattr_l__mod___network_3___0___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___0___norm2 = self.getattr_L__mod___network_3___0___norm2(x_106)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_107 = self.getattr_L__mod___network_3___0___mlp_fc1(getattr_l__mod___network_3___0___norm2);  getattr_l__mod___network_3___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_108 = self.getattr_L__mod___network_3___0___mlp_act(x_107);  x_107 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_109 = self.getattr_L__mod___network_3___0___mlp_drop1(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_110 = self.getattr_L__mod___network_3___0___mlp_norm(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_111 = self.getattr_L__mod___network_3___0___mlp_fc2(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_112 = self.getattr_L__mod___network_3___0___mlp_drop2(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___0___drop_path_1 = self.getattr_L__mod___network_3___0___drop_path(x_112);  x_112 = None
    x_113 = x_106 + getattr_l__mod___network_3___0___drop_path_1;  x_106 = getattr_l__mod___network_3___0___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___1___norm1 = self.getattr_L__mod___network_3___1___norm1(x_113)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_3___1___attn_qkv = self.getattr_L__mod___network_3___1___attn_qkv(getattr_l__mod___network_3___1___norm1);  getattr_l__mod___network_3___1___norm1 = None
    reshape_22 = getattr_l__mod___network_3___1___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_3___1___attn_qkv = None
    qkv_5 = reshape_22.permute(2, 0, 3, 1, 4);  reshape_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_5 = qkv_5.unbind(0);  qkv_5 = None
    q_5 = unbind_5[0]
    k_5 = unbind_5[1]
    v_13 = unbind_5[2];  unbind_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_10 = k_5.transpose(-2, -1);  k_5 = None
    matmul_14 = q_5 @ transpose_10;  q_5 = transpose_10 = None
    attn_35 = matmul_14 * 0.1767766952966369;  matmul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_36 = attn_35.softmax(dim = -1);  attn_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_37 = self.getattr_L__mod___network_3___1___attn_attn_drop(attn_36);  attn_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_15 = attn_37 @ v_13;  attn_37 = v_13 = None
    transpose_11 = matmul_15.transpose(1, 2);  matmul_15 = None
    x_114 = transpose_11.reshape(8, 14, 14, 384);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_115 = self.getattr_L__mod___network_3___1___attn_proj(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_116 = self.getattr_L__mod___network_3___1___attn_proj_drop(x_115);  x_115 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___1___drop_path = self.getattr_L__mod___network_3___1___drop_path(x_116);  x_116 = None
    x_117 = x_113 + getattr_l__mod___network_3___1___drop_path;  x_113 = getattr_l__mod___network_3___1___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___1___norm2 = self.getattr_L__mod___network_3___1___norm2(x_117)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_118 = self.getattr_L__mod___network_3___1___mlp_fc1(getattr_l__mod___network_3___1___norm2);  getattr_l__mod___network_3___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_119 = self.getattr_L__mod___network_3___1___mlp_act(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_120 = self.getattr_L__mod___network_3___1___mlp_drop1(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_121 = self.getattr_L__mod___network_3___1___mlp_norm(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_122 = self.getattr_L__mod___network_3___1___mlp_fc2(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_123 = self.getattr_L__mod___network_3___1___mlp_drop2(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___1___drop_path_1 = self.getattr_L__mod___network_3___1___drop_path(x_123);  x_123 = None
    x_124 = x_117 + getattr_l__mod___network_3___1___drop_path_1;  x_117 = getattr_l__mod___network_3___1___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___2___norm1 = self.getattr_L__mod___network_3___2___norm1(x_124)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_3___2___attn_qkv = self.getattr_L__mod___network_3___2___attn_qkv(getattr_l__mod___network_3___2___norm1);  getattr_l__mod___network_3___2___norm1 = None
    reshape_24 = getattr_l__mod___network_3___2___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_3___2___attn_qkv = None
    qkv_6 = reshape_24.permute(2, 0, 3, 1, 4);  reshape_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_6 = qkv_6.unbind(0);  qkv_6 = None
    q_6 = unbind_6[0]
    k_6 = unbind_6[1]
    v_14 = unbind_6[2];  unbind_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_12 = k_6.transpose(-2, -1);  k_6 = None
    matmul_16 = q_6 @ transpose_12;  q_6 = transpose_12 = None
    attn_38 = matmul_16 * 0.1767766952966369;  matmul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_39 = attn_38.softmax(dim = -1);  attn_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_40 = self.getattr_L__mod___network_3___2___attn_attn_drop(attn_39);  attn_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_17 = attn_40 @ v_14;  attn_40 = v_14 = None
    transpose_13 = matmul_17.transpose(1, 2);  matmul_17 = None
    x_125 = transpose_13.reshape(8, 14, 14, 384);  transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_126 = self.getattr_L__mod___network_3___2___attn_proj(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_127 = self.getattr_L__mod___network_3___2___attn_proj_drop(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___2___drop_path = self.getattr_L__mod___network_3___2___drop_path(x_127);  x_127 = None
    x_128 = x_124 + getattr_l__mod___network_3___2___drop_path;  x_124 = getattr_l__mod___network_3___2___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___2___norm2 = self.getattr_L__mod___network_3___2___norm2(x_128)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_129 = self.getattr_L__mod___network_3___2___mlp_fc1(getattr_l__mod___network_3___2___norm2);  getattr_l__mod___network_3___2___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_130 = self.getattr_L__mod___network_3___2___mlp_act(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_131 = self.getattr_L__mod___network_3___2___mlp_drop1(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_132 = self.getattr_L__mod___network_3___2___mlp_norm(x_131);  x_131 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_133 = self.getattr_L__mod___network_3___2___mlp_fc2(x_132);  x_132 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_134 = self.getattr_L__mod___network_3___2___mlp_drop2(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___2___drop_path_1 = self.getattr_L__mod___network_3___2___drop_path(x_134);  x_134 = None
    x_135 = x_128 + getattr_l__mod___network_3___2___drop_path_1;  x_128 = getattr_l__mod___network_3___2___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___3___norm1 = self.getattr_L__mod___network_3___3___norm1(x_135)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_3___3___attn_qkv = self.getattr_L__mod___network_3___3___attn_qkv(getattr_l__mod___network_3___3___norm1);  getattr_l__mod___network_3___3___norm1 = None
    reshape_26 = getattr_l__mod___network_3___3___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_3___3___attn_qkv = None
    qkv_7 = reshape_26.permute(2, 0, 3, 1, 4);  reshape_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_7 = qkv_7.unbind(0);  qkv_7 = None
    q_7 = unbind_7[0]
    k_7 = unbind_7[1]
    v_15 = unbind_7[2];  unbind_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_14 = k_7.transpose(-2, -1);  k_7 = None
    matmul_18 = q_7 @ transpose_14;  q_7 = transpose_14 = None
    attn_41 = matmul_18 * 0.1767766952966369;  matmul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_42 = attn_41.softmax(dim = -1);  attn_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_43 = self.getattr_L__mod___network_3___3___attn_attn_drop(attn_42);  attn_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_19 = attn_43 @ v_15;  attn_43 = v_15 = None
    transpose_15 = matmul_19.transpose(1, 2);  matmul_19 = None
    x_136 = transpose_15.reshape(8, 14, 14, 384);  transpose_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_137 = self.getattr_L__mod___network_3___3___attn_proj(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_138 = self.getattr_L__mod___network_3___3___attn_proj_drop(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___3___drop_path = self.getattr_L__mod___network_3___3___drop_path(x_138);  x_138 = None
    x_139 = x_135 + getattr_l__mod___network_3___3___drop_path;  x_135 = getattr_l__mod___network_3___3___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___3___norm2 = self.getattr_L__mod___network_3___3___norm2(x_139)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_140 = self.getattr_L__mod___network_3___3___mlp_fc1(getattr_l__mod___network_3___3___norm2);  getattr_l__mod___network_3___3___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_141 = self.getattr_L__mod___network_3___3___mlp_act(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_142 = self.getattr_L__mod___network_3___3___mlp_drop1(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_143 = self.getattr_L__mod___network_3___3___mlp_norm(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_144 = self.getattr_L__mod___network_3___3___mlp_fc2(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_145 = self.getattr_L__mod___network_3___3___mlp_drop2(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___3___drop_path_1 = self.getattr_L__mod___network_3___3___drop_path(x_145);  x_145 = None
    x_146 = x_139 + getattr_l__mod___network_3___3___drop_path_1;  x_139 = getattr_l__mod___network_3___3___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___4___norm1 = self.getattr_L__mod___network_3___4___norm1(x_146)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_3___4___attn_qkv = self.getattr_L__mod___network_3___4___attn_qkv(getattr_l__mod___network_3___4___norm1);  getattr_l__mod___network_3___4___norm1 = None
    reshape_28 = getattr_l__mod___network_3___4___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_3___4___attn_qkv = None
    qkv_8 = reshape_28.permute(2, 0, 3, 1, 4);  reshape_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_8 = qkv_8.unbind(0);  qkv_8 = None
    q_8 = unbind_8[0]
    k_8 = unbind_8[1]
    v_16 = unbind_8[2];  unbind_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_16 = k_8.transpose(-2, -1);  k_8 = None
    matmul_20 = q_8 @ transpose_16;  q_8 = transpose_16 = None
    attn_44 = matmul_20 * 0.1767766952966369;  matmul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_45 = attn_44.softmax(dim = -1);  attn_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_46 = self.getattr_L__mod___network_3___4___attn_attn_drop(attn_45);  attn_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_21 = attn_46 @ v_16;  attn_46 = v_16 = None
    transpose_17 = matmul_21.transpose(1, 2);  matmul_21 = None
    x_147 = transpose_17.reshape(8, 14, 14, 384);  transpose_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_148 = self.getattr_L__mod___network_3___4___attn_proj(x_147);  x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_149 = self.getattr_L__mod___network_3___4___attn_proj_drop(x_148);  x_148 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___4___drop_path = self.getattr_L__mod___network_3___4___drop_path(x_149);  x_149 = None
    x_150 = x_146 + getattr_l__mod___network_3___4___drop_path;  x_146 = getattr_l__mod___network_3___4___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___4___norm2 = self.getattr_L__mod___network_3___4___norm2(x_150)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_151 = self.getattr_L__mod___network_3___4___mlp_fc1(getattr_l__mod___network_3___4___norm2);  getattr_l__mod___network_3___4___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_152 = self.getattr_L__mod___network_3___4___mlp_act(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_153 = self.getattr_L__mod___network_3___4___mlp_drop1(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_154 = self.getattr_L__mod___network_3___4___mlp_norm(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_155 = self.getattr_L__mod___network_3___4___mlp_fc2(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_156 = self.getattr_L__mod___network_3___4___mlp_drop2(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___4___drop_path_1 = self.getattr_L__mod___network_3___4___drop_path(x_156);  x_156 = None
    x_157 = x_150 + getattr_l__mod___network_3___4___drop_path_1;  x_150 = getattr_l__mod___network_3___4___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___5___norm1 = self.getattr_L__mod___network_3___5___norm1(x_157)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_3___5___attn_qkv = self.getattr_L__mod___network_3___5___attn_qkv(getattr_l__mod___network_3___5___norm1);  getattr_l__mod___network_3___5___norm1 = None
    reshape_30 = getattr_l__mod___network_3___5___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_3___5___attn_qkv = None
    qkv_9 = reshape_30.permute(2, 0, 3, 1, 4);  reshape_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_9 = qkv_9.unbind(0);  qkv_9 = None
    q_9 = unbind_9[0]
    k_9 = unbind_9[1]
    v_17 = unbind_9[2];  unbind_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_18 = k_9.transpose(-2, -1);  k_9 = None
    matmul_22 = q_9 @ transpose_18;  q_9 = transpose_18 = None
    attn_47 = matmul_22 * 0.1767766952966369;  matmul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_48 = attn_47.softmax(dim = -1);  attn_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_49 = self.getattr_L__mod___network_3___5___attn_attn_drop(attn_48);  attn_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_23 = attn_49 @ v_17;  attn_49 = v_17 = None
    transpose_19 = matmul_23.transpose(1, 2);  matmul_23 = None
    x_158 = transpose_19.reshape(8, 14, 14, 384);  transpose_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_159 = self.getattr_L__mod___network_3___5___attn_proj(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_160 = self.getattr_L__mod___network_3___5___attn_proj_drop(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___5___drop_path = self.getattr_L__mod___network_3___5___drop_path(x_160);  x_160 = None
    x_161 = x_157 + getattr_l__mod___network_3___5___drop_path;  x_157 = getattr_l__mod___network_3___5___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___5___norm2 = self.getattr_L__mod___network_3___5___norm2(x_161)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_162 = self.getattr_L__mod___network_3___5___mlp_fc1(getattr_l__mod___network_3___5___norm2);  getattr_l__mod___network_3___5___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_163 = self.getattr_L__mod___network_3___5___mlp_act(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_164 = self.getattr_L__mod___network_3___5___mlp_drop1(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_165 = self.getattr_L__mod___network_3___5___mlp_norm(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_166 = self.getattr_L__mod___network_3___5___mlp_fc2(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_167 = self.getattr_L__mod___network_3___5___mlp_drop2(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___5___drop_path_1 = self.getattr_L__mod___network_3___5___drop_path(x_167);  x_167 = None
    x_168 = x_161 + getattr_l__mod___network_3___5___drop_path_1;  x_161 = getattr_l__mod___network_3___5___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___6___norm1 = self.getattr_L__mod___network_3___6___norm1(x_168)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_3___6___attn_qkv = self.getattr_L__mod___network_3___6___attn_qkv(getattr_l__mod___network_3___6___norm1);  getattr_l__mod___network_3___6___norm1 = None
    reshape_32 = getattr_l__mod___network_3___6___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_3___6___attn_qkv = None
    qkv_10 = reshape_32.permute(2, 0, 3, 1, 4);  reshape_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_10 = qkv_10.unbind(0);  qkv_10 = None
    q_10 = unbind_10[0]
    k_10 = unbind_10[1]
    v_18 = unbind_10[2];  unbind_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_20 = k_10.transpose(-2, -1);  k_10 = None
    matmul_24 = q_10 @ transpose_20;  q_10 = transpose_20 = None
    attn_50 = matmul_24 * 0.1767766952966369;  matmul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_51 = attn_50.softmax(dim = -1);  attn_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_52 = self.getattr_L__mod___network_3___6___attn_attn_drop(attn_51);  attn_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_25 = attn_52 @ v_18;  attn_52 = v_18 = None
    transpose_21 = matmul_25.transpose(1, 2);  matmul_25 = None
    x_169 = transpose_21.reshape(8, 14, 14, 384);  transpose_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_170 = self.getattr_L__mod___network_3___6___attn_proj(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_171 = self.getattr_L__mod___network_3___6___attn_proj_drop(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___6___drop_path = self.getattr_L__mod___network_3___6___drop_path(x_171);  x_171 = None
    x_172 = x_168 + getattr_l__mod___network_3___6___drop_path;  x_168 = getattr_l__mod___network_3___6___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___6___norm2 = self.getattr_L__mod___network_3___6___norm2(x_172)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_173 = self.getattr_L__mod___network_3___6___mlp_fc1(getattr_l__mod___network_3___6___norm2);  getattr_l__mod___network_3___6___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_174 = self.getattr_L__mod___network_3___6___mlp_act(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_175 = self.getattr_L__mod___network_3___6___mlp_drop1(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_176 = self.getattr_L__mod___network_3___6___mlp_norm(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_177 = self.getattr_L__mod___network_3___6___mlp_fc2(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_178 = self.getattr_L__mod___network_3___6___mlp_drop2(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___6___drop_path_1 = self.getattr_L__mod___network_3___6___drop_path(x_178);  x_178 = None
    x_179 = x_172 + getattr_l__mod___network_3___6___drop_path_1;  x_172 = getattr_l__mod___network_3___6___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___7___norm1 = self.getattr_L__mod___network_3___7___norm1(x_179)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_3___7___attn_qkv = self.getattr_L__mod___network_3___7___attn_qkv(getattr_l__mod___network_3___7___norm1);  getattr_l__mod___network_3___7___norm1 = None
    reshape_34 = getattr_l__mod___network_3___7___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_3___7___attn_qkv = None
    qkv_11 = reshape_34.permute(2, 0, 3, 1, 4);  reshape_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_11 = qkv_11.unbind(0);  qkv_11 = None
    q_11 = unbind_11[0]
    k_11 = unbind_11[1]
    v_19 = unbind_11[2];  unbind_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_22 = k_11.transpose(-2, -1);  k_11 = None
    matmul_26 = q_11 @ transpose_22;  q_11 = transpose_22 = None
    attn_53 = matmul_26 * 0.1767766952966369;  matmul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_54 = attn_53.softmax(dim = -1);  attn_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_55 = self.getattr_L__mod___network_3___7___attn_attn_drop(attn_54);  attn_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_27 = attn_55 @ v_19;  attn_55 = v_19 = None
    transpose_23 = matmul_27.transpose(1, 2);  matmul_27 = None
    x_180 = transpose_23.reshape(8, 14, 14, 384);  transpose_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_181 = self.getattr_L__mod___network_3___7___attn_proj(x_180);  x_180 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_182 = self.getattr_L__mod___network_3___7___attn_proj_drop(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_3___7___drop_path = self.getattr_L__mod___network_3___7___drop_path(x_182);  x_182 = None
    x_183 = x_179 + getattr_l__mod___network_3___7___drop_path;  x_179 = getattr_l__mod___network_3___7___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___7___norm2 = self.getattr_L__mod___network_3___7___norm2(x_183)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_184 = self.getattr_L__mod___network_3___7___mlp_fc1(getattr_l__mod___network_3___7___norm2);  getattr_l__mod___network_3___7___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_185 = self.getattr_L__mod___network_3___7___mlp_act(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_186 = self.getattr_L__mod___network_3___7___mlp_drop1(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_187 = self.getattr_L__mod___network_3___7___mlp_norm(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_188 = self.getattr_L__mod___network_3___7___mlp_fc2(x_187);  x_187 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_189 = self.getattr_L__mod___network_3___7___mlp_drop2(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_3___7___drop_path_1 = self.getattr_L__mod___network_3___7___drop_path(x_189);  x_189 = None
    x_191 = x_183 + getattr_l__mod___network_3___7___drop_path_1;  x_183 = getattr_l__mod___network_3___7___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_4___0___norm1 = self.getattr_L__mod___network_4___0___norm1(x_191)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_4___0___attn_qkv = self.getattr_L__mod___network_4___0___attn_qkv(getattr_l__mod___network_4___0___norm1);  getattr_l__mod___network_4___0___norm1 = None
    reshape_36 = getattr_l__mod___network_4___0___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_4___0___attn_qkv = None
    qkv_12 = reshape_36.permute(2, 0, 3, 1, 4);  reshape_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_12 = qkv_12.unbind(0);  qkv_12 = None
    q_12 = unbind_12[0]
    k_12 = unbind_12[1]
    v_20 = unbind_12[2];  unbind_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_24 = k_12.transpose(-2, -1);  k_12 = None
    matmul_28 = q_12 @ transpose_24;  q_12 = transpose_24 = None
    attn_56 = matmul_28 * 0.1767766952966369;  matmul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_57 = attn_56.softmax(dim = -1);  attn_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_58 = self.getattr_L__mod___network_4___0___attn_attn_drop(attn_57);  attn_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_29 = attn_58 @ v_20;  attn_58 = v_20 = None
    transpose_25 = matmul_29.transpose(1, 2);  matmul_29 = None
    x_192 = transpose_25.reshape(8, 14, 14, 384);  transpose_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_193 = self.getattr_L__mod___network_4___0___attn_proj(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_194 = self.getattr_L__mod___network_4___0___attn_proj_drop(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_4___0___drop_path = self.getattr_L__mod___network_4___0___drop_path(x_194);  x_194 = None
    x_195 = x_191 + getattr_l__mod___network_4___0___drop_path;  x_191 = getattr_l__mod___network_4___0___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_4___0___norm2 = self.getattr_L__mod___network_4___0___norm2(x_195)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_196 = self.getattr_L__mod___network_4___0___mlp_fc1(getattr_l__mod___network_4___0___norm2);  getattr_l__mod___network_4___0___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_197 = self.getattr_L__mod___network_4___0___mlp_act(x_196);  x_196 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_198 = self.getattr_L__mod___network_4___0___mlp_drop1(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_199 = self.getattr_L__mod___network_4___0___mlp_norm(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_200 = self.getattr_L__mod___network_4___0___mlp_fc2(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_201 = self.getattr_L__mod___network_4___0___mlp_drop2(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_4___0___drop_path_1 = self.getattr_L__mod___network_4___0___drop_path(x_201);  x_201 = None
    x_202 = x_195 + getattr_l__mod___network_4___0___drop_path_1;  x_195 = getattr_l__mod___network_4___0___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_4___1___norm1 = self.getattr_L__mod___network_4___1___norm1(x_202)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:162, code: qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    getattr_l__mod___network_4___1___attn_qkv = self.getattr_L__mod___network_4___1___attn_qkv(getattr_l__mod___network_4___1___norm1);  getattr_l__mod___network_4___1___norm1 = None
    reshape_38 = getattr_l__mod___network_4___1___attn_qkv.reshape(8, 196, 3, 12, 32);  getattr_l__mod___network_4___1___attn_qkv = None
    qkv_13 = reshape_38.permute(2, 0, 3, 1, 4);  reshape_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:163, code: q, k, v = qkv.unbind(0)
    unbind_13 = qkv_13.unbind(0);  qkv_13 = None
    q_13 = unbind_13[0]
    k_13 = unbind_13[1]
    v_21 = unbind_13[2];  unbind_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:165, code: attn = (q @ k.transpose(-2, -1)) * self.scale
    transpose_26 = k_13.transpose(-2, -1);  k_13 = None
    matmul_30 = q_13 @ transpose_26;  q_13 = transpose_26 = None
    attn_59 = matmul_30 * 0.1767766952966369;  matmul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:166, code: attn = attn.softmax(dim=-1)
    attn_60 = attn_59.softmax(dim = -1);  attn_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:167, code: attn = self.attn_drop(attn)
    attn_61 = self.getattr_L__mod___network_4___1___attn_attn_drop(attn_60);  attn_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:169, code: x = (attn @ v).transpose(1, 2).reshape(B, H, W, C)
    matmul_31 = attn_61 @ v_21;  attn_61 = v_21 = None
    transpose_27 = matmul_31.transpose(1, 2);  matmul_31 = None
    x_203 = transpose_27.reshape(8, 14, 14, 384);  transpose_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:170, code: x = self.proj(x)
    x_204 = self.getattr_L__mod___network_4___1___attn_proj(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:171, code: x = self.proj_drop(x)
    x_205 = self.getattr_L__mod___network_4___1___attn_proj_drop(x_204);  x_204 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:201, code: x = x + self.drop_path(self.attn(self.norm1(x)))
    getattr_l__mod___network_4___1___drop_path = self.getattr_L__mod___network_4___1___drop_path(x_205);  x_205 = None
    x_206 = x_202 + getattr_l__mod___network_4___1___drop_path;  x_202 = getattr_l__mod___network_4___1___drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_4___1___norm2 = self.getattr_L__mod___network_4___1___norm2(x_206)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_207 = self.getattr_L__mod___network_4___1___mlp_fc1(getattr_l__mod___network_4___1___norm2);  getattr_l__mod___network_4___1___norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_208 = self.getattr_L__mod___network_4___1___mlp_act(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_209 = self.getattr_L__mod___network_4___1___mlp_drop1(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_210 = self.getattr_L__mod___network_4___1___mlp_norm(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_211 = self.getattr_L__mod___network_4___1___mlp_fc2(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_212 = self.getattr_L__mod___network_4___1___mlp_drop2(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:202, code: x = x + self.drop_path(self.mlp(self.norm2(x)))
    getattr_l__mod___network_4___1___drop_path_1 = self.getattr_L__mod___network_4___1___drop_path(x_212);  x_212 = None
    x_214 = x_206 + getattr_l__mod___network_4___1___drop_path_1;  x_206 = getattr_l__mod___network_4___1___drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:628, code: x = x.reshape(B, -1, C)
    x_216 = x_214.reshape(8, -1, 384);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:633, code: cls_tokens = self.cls_token.expand(B, -1, -1)
    l__mod___cls_token = self.L__mod___cls_token
    cls_tokens = l__mod___cls_token.expand(8, -1, -1);  l__mod___cls_token = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:634, code: x = torch.cat([cls_tokens, x], dim=1)
    x_217 = torch.cat([cls_tokens, x_216], dim = 1);  cls_tokens = x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    cls_embed = x_217[(slice(None, None, None), slice(None, 1, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    l__mod___post_network_0_norm1 = self.L__mod___post_network_0_norm1(x_217)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___post_network_0_attn_kv = self.L__mod___post_network_0_attn_kv(l__mod___post_network_0_norm1)
    reshape_41 = l__mod___post_network_0_attn_kv.reshape(8, 197, 2, 12, 32);  l__mod___post_network_0_attn_kv = None
    kv = reshape_41.permute(2, 0, 3, 1, 4);  reshape_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:236, code: k, v = kv.unbind(0)
    unbind_14 = kv.unbind(0);  kv = None
    k_14 = unbind_14[0]
    v_22 = unbind_14[1];  unbind_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    getitem_45 = l__mod___post_network_0_norm1[(slice(None, None, None), slice(None, 1, None), slice(None, None, None))];  l__mod___post_network_0_norm1 = None
    l__mod___post_network_0_attn_q = self.L__mod___post_network_0_attn_q(getitem_45);  getitem_45 = None
    q_14 = l__mod___post_network_0_attn_q.reshape(8, 12, 1, 32);  l__mod___post_network_0_attn_q = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    mul_18 = q_14 * 0.1767766952966369;  q_14 = None
    transpose_28 = k_14.transpose(-2, -1);  k_14 = None
    attn_62 = mul_18 @ transpose_28;  mul_18 = transpose_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    attn_63 = attn_62.softmax(dim = -1);  attn_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:240, code: attn = self.attn_drop(attn)
    attn_64 = self.L__mod___post_network_0_attn_attn_drop(attn_63);  attn_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    matmul_33 = attn_64 @ v_22;  attn_64 = v_22 = None
    transpose_29 = matmul_33.transpose(1, 2);  matmul_33 = None
    cls_embed_1 = transpose_29.reshape(8, 1, 384);  transpose_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    cls_embed_2 = self.L__mod___post_network_0_attn_proj(cls_embed_1);  cls_embed_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:244, code: cls_embed = self.proj_drop(cls_embed)
    cls_embed_3 = self.L__mod___post_network_0_attn_proj_drop(cls_embed_2);  cls_embed_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    l__mod___post_network_0_drop_path = self.L__mod___post_network_0_drop_path(cls_embed_3);  cls_embed_3 = None
    cls_embed_4 = cls_embed + l__mod___post_network_0_drop_path;  cls_embed = l__mod___post_network_0_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    l__mod___post_network_0_norm2 = self.L__mod___post_network_0_norm2(cls_embed_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_218 = self.L__mod___post_network_0_mlp_fc1(l__mod___post_network_0_norm2);  l__mod___post_network_0_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_219 = self.L__mod___post_network_0_mlp_act(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_220 = self.L__mod___post_network_0_mlp_drop1(x_219);  x_219 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_221 = self.L__mod___post_network_0_mlp_norm(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_222 = self.L__mod___post_network_0_mlp_fc2(x_221);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_223 = self.L__mod___post_network_0_mlp_drop2(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    l__mod___post_network_0_drop_path_1 = self.L__mod___post_network_0_drop_path(x_223);  x_223 = None
    cls_embed_5 = cls_embed_4 + l__mod___post_network_0_drop_path_1;  cls_embed_4 = l__mod___post_network_0_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:288, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
    getitem_46 = x_217[(slice(None, None, None), slice(1, None, None))];  x_217 = None
    x_224 = torch.cat([cls_embed_5, getitem_46], dim = 1);  cls_embed_5 = getitem_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:285, code: cls_embed = x[:, :1]
    cls_embed_6 = x_224[(slice(None, None, None), slice(None, 1, None))]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    l__mod___post_network_1_norm1 = self.L__mod___post_network_1_norm1(x_224)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:235, code: kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    l__mod___post_network_1_attn_kv = self.L__mod___post_network_1_attn_kv(l__mod___post_network_1_norm1)
    reshape_44 = l__mod___post_network_1_attn_kv.reshape(8, 197, 2, 12, 32);  l__mod___post_network_1_attn_kv = None
    kv_1 = reshape_44.permute(2, 0, 3, 1, 4);  reshape_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:236, code: k, v = kv.unbind(0)
    unbind_15 = kv_1.unbind(0);  kv_1 = None
    k_15 = unbind_15[0]
    v_23 = unbind_15[1];  unbind_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:237, code: q = self.q(x[:, :1, :]).reshape(B, self.num_heads, 1, self.head_dim)
    getitem_50 = l__mod___post_network_1_norm1[(slice(None, None, None), slice(None, 1, None), slice(None, None, None))];  l__mod___post_network_1_norm1 = None
    l__mod___post_network_1_attn_q = self.L__mod___post_network_1_attn_q(getitem_50);  getitem_50 = None
    q_15 = l__mod___post_network_1_attn_q.reshape(8, 12, 1, 32);  l__mod___post_network_1_attn_q = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:238, code: attn = ((q * self.scale) @ k.transpose(-2, -1))
    mul_19 = q_15 * 0.1767766952966369;  q_15 = None
    transpose_30 = k_15.transpose(-2, -1);  k_15 = None
    attn_65 = mul_19 @ transpose_30;  mul_19 = transpose_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:239, code: attn = attn.softmax(dim=-1)
    attn_66 = attn_65.softmax(dim = -1);  attn_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:240, code: attn = self.attn_drop(attn)
    attn_67 = self.L__mod___post_network_1_attn_attn_drop(attn_66);  attn_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:242, code: cls_embed = (attn @ v).transpose(1, 2).reshape(B, 1, self.head_dim * self.num_heads)
    matmul_35 = attn_67 @ v_23;  attn_67 = v_23 = None
    transpose_31 = matmul_35.transpose(1, 2);  matmul_35 = None
    cls_embed_7 = transpose_31.reshape(8, 1, 384);  transpose_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:243, code: cls_embed = self.proj(cls_embed)
    cls_embed_8 = self.L__mod___post_network_1_attn_proj(cls_embed_7);  cls_embed_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:244, code: cls_embed = self.proj_drop(cls_embed)
    cls_embed_9 = self.L__mod___post_network_1_attn_proj_drop(cls_embed_8);  cls_embed_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:286, code: cls_embed = cls_embed + self.drop_path(self.attn(self.norm1(x)))
    l__mod___post_network_1_drop_path = self.L__mod___post_network_1_drop_path(cls_embed_9);  cls_embed_9 = None
    cls_embed_10 = cls_embed_6 + l__mod___post_network_1_drop_path;  cls_embed_6 = l__mod___post_network_1_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    l__mod___post_network_1_norm2 = self.L__mod___post_network_1_norm2(cls_embed_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_225 = self.L__mod___post_network_1_mlp_fc1(l__mod___post_network_1_norm2);  l__mod___post_network_1_norm2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_226 = self.L__mod___post_network_1_mlp_act(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_227 = self.L__mod___post_network_1_mlp_drop1(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_228 = self.L__mod___post_network_1_mlp_norm(x_227);  x_227 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_229 = self.L__mod___post_network_1_mlp_fc2(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_230 = self.L__mod___post_network_1_mlp_drop2(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:287, code: cls_embed = cls_embed + self.drop_path(self.mlp(self.norm2(cls_embed)))
    l__mod___post_network_1_drop_path_1 = self.L__mod___post_network_1_drop_path(x_230);  x_230 = None
    cls_embed_11 = cls_embed_10 + l__mod___post_network_1_drop_path_1;  cls_embed_10 = l__mod___post_network_1_drop_path_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:288, code: return torch.cat([cls_embed, x[:, 1:]], dim=1)
    getitem_51 = x_224[(slice(None, None, None), slice(1, None, None))];  x_224 = None
    x_232 = torch.cat([cls_embed_11, getitem_51], dim = 1);  cls_embed_11 = getitem_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:703, code: x = self.norm(x)
    x_234 = self.L__mod___norm(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:710, code: out = x[:, 0]
    out = x_234[(slice(None, None, None), 0)]
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:713, code: x = self.head_drop(x)
    x_235 = self.L__mod___head_drop(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:716, code: out = self.head(out)
    out_1 = self.L__mod___head(out);  out = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:719, code: aux = self.aux_head(x[:, 1:])
    getitem_53 = x_235[(slice(None, None, None), slice(1, None, None))];  x_235 = None
    aux = self.L__mod___aux_head(getitem_53);  getitem_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/volo.py:720, code: out = out + 0.5 * aux.max(1)[0]
    max_1 = aux.max(1);  aux = None
    getitem_54 = max_1[0];  max_1 = None
    mul_20 = 0.5 * getitem_54;  getitem_54 = None
    pred = out_1 + mul_20;  out_1 = mul_20 = None
    return (pred,)
    