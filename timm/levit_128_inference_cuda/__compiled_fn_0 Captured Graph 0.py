from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    l__mod___stem_conv1_linear = self.L__mod___stem_conv1_linear(l_inputs_0_);  l_inputs_0_ = None
    l__mod___stem_conv1_bn = self.L__mod___stem_conv1_bn(l__mod___stem_conv1_linear);  l__mod___stem_conv1_linear = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    l__mod___stem_act1 = self.L__mod___stem_act1(l__mod___stem_conv1_bn);  l__mod___stem_conv1_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    l__mod___stem_conv2_linear = self.L__mod___stem_conv2_linear(l__mod___stem_act1);  l__mod___stem_act1 = None
    l__mod___stem_conv2_bn = self.L__mod___stem_conv2_bn(l__mod___stem_conv2_linear);  l__mod___stem_conv2_linear = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    l__mod___stem_act2 = self.L__mod___stem_act2(l__mod___stem_conv2_bn);  l__mod___stem_conv2_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    l__mod___stem_conv3_linear = self.L__mod___stem_conv3_linear(l__mod___stem_act2);  l__mod___stem_act2 = None
    l__mod___stem_conv3_bn = self.L__mod___stem_conv3_bn(l__mod___stem_conv3_linear);  l__mod___stem_conv3_linear = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:637, code: x = self.stem(x)
    l__mod___stem_act3 = self.L__mod___stem_act3(l__mod___stem_conv3_bn);  l__mod___stem_conv3_bn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:65, code: return self.bn(self.linear(x))
    l__mod___stem_conv4_linear = self.L__mod___stem_conv4_linear(l__mod___stem_act3);  l__mod___stem_act3 = None
    x = self.L__mod___stem_conv4_bn(l__mod___stem_conv4_linear);  l__mod___stem_conv4_linear = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:639, code: x = x.flatten(2).transpose(1, 2)
    flatten = x.flatten(2);  x = None
    x_1 = flatten.transpose(1, 2);  flatten = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:517, code: x = self.downsample(x)
    x_2 = self.getattr_L__mod___stages___0___downsample(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_3 = self.getattr_getattr_L__mod___stages___0___blocks___0___attn_qkv_linear(x_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_1 = x_3.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___0___attn_qkv_bn = self.getattr_getattr_L__mod___stages___0___blocks___0___attn_qkv_bn(flatten_1);  flatten_1 = None
    reshape_as = getattr_getattr_l__mod___stages___0___blocks___0___attn_qkv_bn.reshape_as(x_3);  getattr_getattr_l__mod___stages___0___blocks___0___attn_qkv_bn = x_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view = reshape_as.view(8, 196, 4, -1);  reshape_as = None
    split = view.split([16, 16, 32], dim = 3);  view = None
    q = split[0]
    k = split[1]
    v = split[2];  split = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    q_1 = q.permute(0, 2, 1, 3);  q = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    k_1 = k.permute(0, 2, 3, 1);  k = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    v_1 = v.permute(0, 2, 1, 3);  v = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul = q_1 @ k_1;  q_1 = k_1 = None
    mul = matmul * 0.25;  matmul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_getattr_l__mod___stages___0___blocks___0___attn_attention_biases = self.getattr_getattr_L__mod___stages___0___blocks___0___attn_attention_biases
    getattr_getattr_l__mod___stages___0___blocks___0___attn_attention_bias_idxs = self.getattr_getattr_L__mod___stages___0___blocks___0___attn_attention_bias_idxs
    getitem_3 = getattr_getattr_l__mod___stages___0___blocks___0___attn_attention_biases[(slice(None, None, None), getattr_getattr_l__mod___stages___0___blocks___0___attn_attention_bias_idxs)];  getattr_getattr_l__mod___stages___0___blocks___0___attn_attention_biases = getattr_getattr_l__mod___stages___0___blocks___0___attn_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn = mul + getitem_3;  mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    attn_1 = attn.softmax(dim = -1);  attn = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    matmul_1 = attn_1 @ v_1;  attn_1 = v_1 = None
    transpose_1 = matmul_1.transpose(1, 2);  matmul_1 = None
    x_4 = transpose_1.reshape(8, 196, 128);  transpose_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_act = self.getattr_getattr_L__mod___stages___0___blocks___0___attn_proj_act(x_4);  x_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_5 = self.getattr_getattr_L__mod___stages___0___blocks___0___attn_proj_ln_linear(getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_act);  getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_2 = x_5.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_ln_bn = self.getattr_getattr_L__mod___stages___0___blocks___0___attn_proj_ln_bn(flatten_2);  flatten_2 = None
    x_6 = getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_ln_bn.reshape_as(x_5);  getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_ln_bn = x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    getattr_getattr_l__mod___stages___0___blocks___0___drop_path1 = self.getattr_getattr_L__mod___stages___0___blocks___0___drop_path1(x_6);  x_6 = None
    x_7 = x_2 + getattr_getattr_l__mod___stages___0___blocks___0___drop_path1;  x_2 = getattr_getattr_l__mod___stages___0___blocks___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_8 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_ln1_linear(x_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_3 = x_8.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_ln1_bn(flatten_3);  flatten_3 = None
    x_9 = getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn.reshape_as(x_8);  getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn = x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_10 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_act(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_11 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_drop(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_12 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_ln2_linear(x_11);  x_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_4 = x_12.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln2_bn = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_ln2_bn(flatten_4);  flatten_4 = None
    x_13 = getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln2_bn.reshape_as(x_12);  getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln2_bn = x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    getattr_getattr_l__mod___stages___0___blocks___0___drop_path2 = self.getattr_getattr_L__mod___stages___0___blocks___0___drop_path2(x_13);  x_13 = None
    x_14 = x_7 + getattr_getattr_l__mod___stages___0___blocks___0___drop_path2;  x_7 = getattr_getattr_l__mod___stages___0___blocks___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_15 = self.getattr_getattr_L__mod___stages___0___blocks___1___attn_qkv_linear(x_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_5 = x_15.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___1___attn_qkv_bn = self.getattr_getattr_L__mod___stages___0___blocks___1___attn_qkv_bn(flatten_5);  flatten_5 = None
    reshape_as_4 = getattr_getattr_l__mod___stages___0___blocks___1___attn_qkv_bn.reshape_as(x_15);  getattr_getattr_l__mod___stages___0___blocks___1___attn_qkv_bn = x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_1 = reshape_as_4.view(8, 196, 4, -1);  reshape_as_4 = None
    split_1 = view_1.split([16, 16, 32], dim = 3);  view_1 = None
    q_2 = split_1[0]
    k_2 = split_1[1]
    v_2 = split_1[2];  split_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    q_3 = q_2.permute(0, 2, 1, 3);  q_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    k_3 = k_2.permute(0, 2, 3, 1);  k_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    v_3 = v_2.permute(0, 2, 1, 3);  v_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_2 = q_3 @ k_3;  q_3 = k_3 = None
    mul_1 = matmul_2 * 0.25;  matmul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_getattr_l__mod___stages___0___blocks___1___attn_attention_biases = self.getattr_getattr_L__mod___stages___0___blocks___1___attn_attention_biases
    getattr_getattr_l__mod___stages___0___blocks___1___attn_attention_bias_idxs = self.getattr_getattr_L__mod___stages___0___blocks___1___attn_attention_bias_idxs
    getitem_7 = getattr_getattr_l__mod___stages___0___blocks___1___attn_attention_biases[(slice(None, None, None), getattr_getattr_l__mod___stages___0___blocks___1___attn_attention_bias_idxs)];  getattr_getattr_l__mod___stages___0___blocks___1___attn_attention_biases = getattr_getattr_l__mod___stages___0___blocks___1___attn_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_2 = mul_1 + getitem_7;  mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    attn_3 = attn_2.softmax(dim = -1);  attn_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    matmul_3 = attn_3 @ v_3;  attn_3 = v_3 = None
    transpose_2 = matmul_3.transpose(1, 2);  matmul_3 = None
    x_16 = transpose_2.reshape(8, 196, 128);  transpose_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    getattr_getattr_l__mod___stages___0___blocks___1___attn_proj_act = self.getattr_getattr_L__mod___stages___0___blocks___1___attn_proj_act(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_17 = self.getattr_getattr_L__mod___stages___0___blocks___1___attn_proj_ln_linear(getattr_getattr_l__mod___stages___0___blocks___1___attn_proj_act);  getattr_getattr_l__mod___stages___0___blocks___1___attn_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_6 = x_17.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___1___attn_proj_ln_bn = self.getattr_getattr_L__mod___stages___0___blocks___1___attn_proj_ln_bn(flatten_6);  flatten_6 = None
    x_18 = getattr_getattr_l__mod___stages___0___blocks___1___attn_proj_ln_bn.reshape_as(x_17);  getattr_getattr_l__mod___stages___0___blocks___1___attn_proj_ln_bn = x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    getattr_getattr_l__mod___stages___0___blocks___1___drop_path1 = self.getattr_getattr_L__mod___stages___0___blocks___1___drop_path1(x_18);  x_18 = None
    x_19 = x_14 + getattr_getattr_l__mod___stages___0___blocks___1___drop_path1;  x_14 = getattr_getattr_l__mod___stages___0___blocks___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_20 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_ln1_linear(x_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_7 = x_20.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln1_bn = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_ln1_bn(flatten_7);  flatten_7 = None
    x_21 = getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln1_bn.reshape_as(x_20);  getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln1_bn = x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_22 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_act(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_23 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_drop(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_24 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_ln2_linear(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_8 = x_24.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln2_bn = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_ln2_bn(flatten_8);  flatten_8 = None
    x_25 = getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln2_bn.reshape_as(x_24);  getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln2_bn = x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    getattr_getattr_l__mod___stages___0___blocks___1___drop_path2 = self.getattr_getattr_L__mod___stages___0___blocks___1___drop_path2(x_25);  x_25 = None
    x_26 = x_19 + getattr_getattr_l__mod___stages___0___blocks___1___drop_path2;  x_19 = getattr_getattr_l__mod___stages___0___blocks___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_27 = self.getattr_getattr_L__mod___stages___0___blocks___2___attn_qkv_linear(x_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_9 = x_27.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___2___attn_qkv_bn = self.getattr_getattr_L__mod___stages___0___blocks___2___attn_qkv_bn(flatten_9);  flatten_9 = None
    reshape_as_8 = getattr_getattr_l__mod___stages___0___blocks___2___attn_qkv_bn.reshape_as(x_27);  getattr_getattr_l__mod___stages___0___blocks___2___attn_qkv_bn = x_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_2 = reshape_as_8.view(8, 196, 4, -1);  reshape_as_8 = None
    split_2 = view_2.split([16, 16, 32], dim = 3);  view_2 = None
    q_4 = split_2[0]
    k_4 = split_2[1]
    v_4 = split_2[2];  split_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    q_5 = q_4.permute(0, 2, 1, 3);  q_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    k_5 = k_4.permute(0, 2, 3, 1);  k_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    v_5 = v_4.permute(0, 2, 1, 3);  v_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_4 = q_5 @ k_5;  q_5 = k_5 = None
    mul_2 = matmul_4 * 0.25;  matmul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_getattr_l__mod___stages___0___blocks___2___attn_attention_biases = self.getattr_getattr_L__mod___stages___0___blocks___2___attn_attention_biases
    getattr_getattr_l__mod___stages___0___blocks___2___attn_attention_bias_idxs = self.getattr_getattr_L__mod___stages___0___blocks___2___attn_attention_bias_idxs
    getitem_11 = getattr_getattr_l__mod___stages___0___blocks___2___attn_attention_biases[(slice(None, None, None), getattr_getattr_l__mod___stages___0___blocks___2___attn_attention_bias_idxs)];  getattr_getattr_l__mod___stages___0___blocks___2___attn_attention_biases = getattr_getattr_l__mod___stages___0___blocks___2___attn_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_4 = mul_2 + getitem_11;  mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    attn_5 = attn_4.softmax(dim = -1);  attn_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    matmul_5 = attn_5 @ v_5;  attn_5 = v_5 = None
    transpose_3 = matmul_5.transpose(1, 2);  matmul_5 = None
    x_28 = transpose_3.reshape(8, 196, 128);  transpose_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    getattr_getattr_l__mod___stages___0___blocks___2___attn_proj_act = self.getattr_getattr_L__mod___stages___0___blocks___2___attn_proj_act(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_29 = self.getattr_getattr_L__mod___stages___0___blocks___2___attn_proj_ln_linear(getattr_getattr_l__mod___stages___0___blocks___2___attn_proj_act);  getattr_getattr_l__mod___stages___0___blocks___2___attn_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_10 = x_29.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___2___attn_proj_ln_bn = self.getattr_getattr_L__mod___stages___0___blocks___2___attn_proj_ln_bn(flatten_10);  flatten_10 = None
    x_30 = getattr_getattr_l__mod___stages___0___blocks___2___attn_proj_ln_bn.reshape_as(x_29);  getattr_getattr_l__mod___stages___0___blocks___2___attn_proj_ln_bn = x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    getattr_getattr_l__mod___stages___0___blocks___2___drop_path1 = self.getattr_getattr_L__mod___stages___0___blocks___2___drop_path1(x_30);  x_30 = None
    x_31 = x_26 + getattr_getattr_l__mod___stages___0___blocks___2___drop_path1;  x_26 = getattr_getattr_l__mod___stages___0___blocks___2___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_32 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_ln1_linear(x_31)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_11 = x_32.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln1_bn = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_ln1_bn(flatten_11);  flatten_11 = None
    x_33 = getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln1_bn.reshape_as(x_32);  getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln1_bn = x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_34 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_act(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_35 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_drop(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_36 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_ln2_linear(x_35);  x_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_12 = x_36.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln2_bn = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_ln2_bn(flatten_12);  flatten_12 = None
    x_37 = getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln2_bn.reshape_as(x_36);  getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln2_bn = x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    getattr_getattr_l__mod___stages___0___blocks___2___drop_path2 = self.getattr_getattr_L__mod___stages___0___blocks___2___drop_path2(x_37);  x_37 = None
    x_38 = x_31 + getattr_getattr_l__mod___stages___0___blocks___2___drop_path2;  x_31 = getattr_getattr_l__mod___stages___0___blocks___2___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_39 = self.getattr_getattr_L__mod___stages___0___blocks___3___attn_qkv_linear(x_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_13 = x_39.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___3___attn_qkv_bn = self.getattr_getattr_L__mod___stages___0___blocks___3___attn_qkv_bn(flatten_13);  flatten_13 = None
    reshape_as_12 = getattr_getattr_l__mod___stages___0___blocks___3___attn_qkv_bn.reshape_as(x_39);  getattr_getattr_l__mod___stages___0___blocks___3___attn_qkv_bn = x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_3 = reshape_as_12.view(8, 196, 4, -1);  reshape_as_12 = None
    split_3 = view_3.split([16, 16, 32], dim = 3);  view_3 = None
    q_6 = split_3[0]
    k_6 = split_3[1]
    v_6 = split_3[2];  split_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    q_7 = q_6.permute(0, 2, 1, 3);  q_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    k_7 = k_6.permute(0, 2, 3, 1);  k_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    v_7 = v_6.permute(0, 2, 1, 3);  v_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_6 = q_7 @ k_7;  q_7 = k_7 = None
    mul_3 = matmul_6 * 0.25;  matmul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_getattr_l__mod___stages___0___blocks___3___attn_attention_biases = self.getattr_getattr_L__mod___stages___0___blocks___3___attn_attention_biases
    getattr_getattr_l__mod___stages___0___blocks___3___attn_attention_bias_idxs = self.getattr_getattr_L__mod___stages___0___blocks___3___attn_attention_bias_idxs
    getitem_15 = getattr_getattr_l__mod___stages___0___blocks___3___attn_attention_biases[(slice(None, None, None), getattr_getattr_l__mod___stages___0___blocks___3___attn_attention_bias_idxs)];  getattr_getattr_l__mod___stages___0___blocks___3___attn_attention_biases = getattr_getattr_l__mod___stages___0___blocks___3___attn_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_6 = mul_3 + getitem_15;  mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    attn_7 = attn_6.softmax(dim = -1);  attn_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    matmul_7 = attn_7 @ v_7;  attn_7 = v_7 = None
    transpose_4 = matmul_7.transpose(1, 2);  matmul_7 = None
    x_40 = transpose_4.reshape(8, 196, 128);  transpose_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    getattr_getattr_l__mod___stages___0___blocks___3___attn_proj_act = self.getattr_getattr_L__mod___stages___0___blocks___3___attn_proj_act(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_41 = self.getattr_getattr_L__mod___stages___0___blocks___3___attn_proj_ln_linear(getattr_getattr_l__mod___stages___0___blocks___3___attn_proj_act);  getattr_getattr_l__mod___stages___0___blocks___3___attn_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_14 = x_41.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___3___attn_proj_ln_bn = self.getattr_getattr_L__mod___stages___0___blocks___3___attn_proj_ln_bn(flatten_14);  flatten_14 = None
    x_42 = getattr_getattr_l__mod___stages___0___blocks___3___attn_proj_ln_bn.reshape_as(x_41);  getattr_getattr_l__mod___stages___0___blocks___3___attn_proj_ln_bn = x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    getattr_getattr_l__mod___stages___0___blocks___3___drop_path1 = self.getattr_getattr_L__mod___stages___0___blocks___3___drop_path1(x_42);  x_42 = None
    x_43 = x_38 + getattr_getattr_l__mod___stages___0___blocks___3___drop_path1;  x_38 = getattr_getattr_l__mod___stages___0___blocks___3___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_44 = self.getattr_getattr_L__mod___stages___0___blocks___3___mlp_ln1_linear(x_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_15 = x_44.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln1_bn = self.getattr_getattr_L__mod___stages___0___blocks___3___mlp_ln1_bn(flatten_15);  flatten_15 = None
    x_45 = getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln1_bn.reshape_as(x_44);  getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln1_bn = x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_46 = self.getattr_getattr_L__mod___stages___0___blocks___3___mlp_act(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_47 = self.getattr_getattr_L__mod___stages___0___blocks___3___mlp_drop(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_48 = self.getattr_getattr_L__mod___stages___0___blocks___3___mlp_ln2_linear(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_16 = x_48.flatten(0, 1)
    getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln2_bn = self.getattr_getattr_L__mod___stages___0___blocks___3___mlp_ln2_bn(flatten_16);  flatten_16 = None
    x_49 = getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln2_bn.reshape_as(x_48);  getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln2_bn = x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    getattr_getattr_l__mod___stages___0___blocks___3___drop_path2 = self.getattr_getattr_L__mod___stages___0___blocks___3___drop_path2(x_49);  x_49 = None
    x_51 = x_43 + getattr_getattr_l__mod___stages___0___blocks___3___drop_path2;  x_43 = getattr_getattr_l__mod___stages___0___blocks___3___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_52 = self.getattr_L__mod___stages___1___downsample_attn_downsample_kv_linear(x_51)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_17 = x_52.flatten(0, 1)
    getattr_l__mod___stages___1___downsample_attn_downsample_kv_bn = self.getattr_L__mod___stages___1___downsample_attn_downsample_kv_bn(flatten_17);  flatten_17 = None
    reshape_as_16 = getattr_l__mod___stages___1___downsample_attn_downsample_kv_bn.reshape_as(x_52);  getattr_l__mod___stages___1___downsample_attn_downsample_kv_bn = x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:331, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
    view_4 = reshape_as_16.view(8, 196, 8, -1);  reshape_as_16 = None
    split_4 = view_4.split([16, 64], dim = 3);  view_4 = None
    k_8 = split_4[0]
    v_8 = split_4[1];  split_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:332, code: k = k.permute(0, 2, 3, 1)  # BHCN
    k_9 = k_8.permute(0, 2, 3, 1);  k_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:333, code: v = v.permute(0, 2, 1, 3)  # BHNC
    v_9 = v_8.permute(0, 2, 1, 3);  v_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:157, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
    x_53 = x_51.view(8, 14, 14, 128);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:161, code: x = x[:, ::self.stride, ::self.stride]
    x_54 = x_53[(slice(None, None, None), slice(None, None, 2), slice(None, None, 2))];  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:162, code: return x.reshape(B, -1, C)
    reshape_4 = x_54.reshape(8, -1, 128);  x_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_55 = self.getattr_L__mod___stages___1___downsample_attn_downsample_q_ln_linear(reshape_4);  reshape_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_18 = x_55.flatten(0, 1)
    getattr_l__mod___stages___1___downsample_attn_downsample_q_ln_bn = self.getattr_L__mod___stages___1___downsample_attn_downsample_q_ln_bn(flatten_18);  flatten_18 = None
    reshape_as_17 = getattr_l__mod___stages___1___downsample_attn_downsample_q_ln_bn.reshape_as(x_55);  getattr_l__mod___stages___1___downsample_attn_downsample_q_ln_bn = x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:334, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    view_6 = reshape_as_17.view(8, -1, 8, 16);  reshape_as_17 = None
    q_8 = view_6.permute(0, 2, 1, 3);  view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_8 = q_8 @ k_9;  q_8 = k_9 = None
    mul_4 = matmul_8 * 0.25;  matmul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:315, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_l__mod___stages___1___downsample_attn_downsample_attention_biases = self.getattr_L__mod___stages___1___downsample_attn_downsample_attention_biases
    getattr_l__mod___stages___1___downsample_attn_downsample_attention_bias_idxs = self.getattr_L__mod___stages___1___downsample_attn_downsample_attention_bias_idxs
    getitem_19 = getattr_l__mod___stages___1___downsample_attn_downsample_attention_biases[(slice(None, None, None), getattr_l__mod___stages___1___downsample_attn_downsample_attention_bias_idxs)];  getattr_l__mod___stages___1___downsample_attn_downsample_attention_biases = getattr_l__mod___stages___1___downsample_attn_downsample_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_8 = mul_4 + getitem_19;  mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    attn_9 = attn_8.softmax(dim = -1);  attn_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    matmul_9 = attn_9 @ v_9;  attn_9 = v_9 = None
    transpose_5 = matmul_9.transpose(1, 2);  matmul_9 = None
    x_56 = transpose_5.reshape(8, -1, 512);  transpose_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:340, code: x = self.proj(x)
    getattr_l__mod___stages___1___downsample_attn_downsample_proj_act = self.getattr_L__mod___stages___1___downsample_attn_downsample_proj_act(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_57 = self.getattr_L__mod___stages___1___downsample_attn_downsample_proj_ln_linear(getattr_l__mod___stages___1___downsample_attn_downsample_proj_act);  getattr_l__mod___stages___1___downsample_attn_downsample_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_19 = x_57.flatten(0, 1)
    getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn = self.getattr_L__mod___stages___1___downsample_attn_downsample_proj_ln_bn(flatten_19);  flatten_19 = None
    x_59 = getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn.reshape_as(x_57);  getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn = x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_60 = self.getattr_L__mod___stages___1___downsample_mlp_ln1_linear(x_59)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_20 = x_60.flatten(0, 1)
    getattr_l__mod___stages___1___downsample_mlp_ln1_bn = self.getattr_L__mod___stages___1___downsample_mlp_ln1_bn(flatten_20);  flatten_20 = None
    x_61 = getattr_l__mod___stages___1___downsample_mlp_ln1_bn.reshape_as(x_60);  getattr_l__mod___stages___1___downsample_mlp_ln1_bn = x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_62 = self.getattr_L__mod___stages___1___downsample_mlp_act(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_63 = self.getattr_L__mod___stages___1___downsample_mlp_drop(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_64 = self.getattr_L__mod___stages___1___downsample_mlp_ln2_linear(x_63);  x_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_21 = x_64.flatten(0, 1)
    getattr_l__mod___stages___1___downsample_mlp_ln2_bn = self.getattr_L__mod___stages___1___downsample_mlp_ln2_bn(flatten_21);  flatten_21 = None
    x_65 = getattr_l__mod___stages___1___downsample_mlp_ln2_bn.reshape_as(x_64);  getattr_l__mod___stages___1___downsample_mlp_ln2_bn = x_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:415, code: x = x + self.drop_path(self.mlp(x))
    getattr_l__mod___stages___1___downsample_drop_path = self.getattr_L__mod___stages___1___downsample_drop_path(x_65);  x_65 = None
    x_67 = x_59 + getattr_l__mod___stages___1___downsample_drop_path;  x_59 = getattr_l__mod___stages___1___downsample_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_68 = self.getattr_getattr_L__mod___stages___1___blocks___0___attn_qkv_linear(x_67)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_22 = x_68.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___0___attn_qkv_bn = self.getattr_getattr_L__mod___stages___1___blocks___0___attn_qkv_bn(flatten_22);  flatten_22 = None
    reshape_as_21 = getattr_getattr_l__mod___stages___1___blocks___0___attn_qkv_bn.reshape_as(x_68);  getattr_getattr_l__mod___stages___1___blocks___0___attn_qkv_bn = x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_7 = reshape_as_21.view(8, 49, 8, -1);  reshape_as_21 = None
    split_5 = view_7.split([16, 16, 32], dim = 3);  view_7 = None
    q_9 = split_5[0]
    k_10 = split_5[1]
    v_10 = split_5[2];  split_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    q_10 = q_9.permute(0, 2, 1, 3);  q_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    k_11 = k_10.permute(0, 2, 3, 1);  k_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    v_11 = v_10.permute(0, 2, 1, 3);  v_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_10 = q_10 @ k_11;  q_10 = k_11 = None
    mul_5 = matmul_10 * 0.25;  matmul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_getattr_l__mod___stages___1___blocks___0___attn_attention_biases = self.getattr_getattr_L__mod___stages___1___blocks___0___attn_attention_biases
    getattr_getattr_l__mod___stages___1___blocks___0___attn_attention_bias_idxs = self.getattr_getattr_L__mod___stages___1___blocks___0___attn_attention_bias_idxs
    getitem_23 = getattr_getattr_l__mod___stages___1___blocks___0___attn_attention_biases[(slice(None, None, None), getattr_getattr_l__mod___stages___1___blocks___0___attn_attention_bias_idxs)];  getattr_getattr_l__mod___stages___1___blocks___0___attn_attention_biases = getattr_getattr_l__mod___stages___1___blocks___0___attn_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_10 = mul_5 + getitem_23;  mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    attn_11 = attn_10.softmax(dim = -1);  attn_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    matmul_11 = attn_11 @ v_11;  attn_11 = v_11 = None
    transpose_6 = matmul_11.transpose(1, 2);  matmul_11 = None
    x_69 = transpose_6.reshape(8, 49, 256);  transpose_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_act = self.getattr_getattr_L__mod___stages___1___blocks___0___attn_proj_act(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_70 = self.getattr_getattr_L__mod___stages___1___blocks___0___attn_proj_ln_linear(getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_act);  getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_23 = x_70.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_ln_bn = self.getattr_getattr_L__mod___stages___1___blocks___0___attn_proj_ln_bn(flatten_23);  flatten_23 = None
    x_71 = getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_ln_bn.reshape_as(x_70);  getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_ln_bn = x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    getattr_getattr_l__mod___stages___1___blocks___0___drop_path1 = self.getattr_getattr_L__mod___stages___1___blocks___0___drop_path1(x_71);  x_71 = None
    x_72 = x_67 + getattr_getattr_l__mod___stages___1___blocks___0___drop_path1;  x_67 = getattr_getattr_l__mod___stages___1___blocks___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_73 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_ln1_linear(x_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_24 = x_73.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln1_bn = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_ln1_bn(flatten_24);  flatten_24 = None
    x_74 = getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln1_bn.reshape_as(x_73);  getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln1_bn = x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_75 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_act(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_76 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_drop(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_77 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_ln2_linear(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_25 = x_77.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln2_bn = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_ln2_bn(flatten_25);  flatten_25 = None
    x_78 = getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln2_bn.reshape_as(x_77);  getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln2_bn = x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    getattr_getattr_l__mod___stages___1___blocks___0___drop_path2 = self.getattr_getattr_L__mod___stages___1___blocks___0___drop_path2(x_78);  x_78 = None
    x_79 = x_72 + getattr_getattr_l__mod___stages___1___blocks___0___drop_path2;  x_72 = getattr_getattr_l__mod___stages___1___blocks___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_80 = self.getattr_getattr_L__mod___stages___1___blocks___1___attn_qkv_linear(x_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_26 = x_80.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___1___attn_qkv_bn = self.getattr_getattr_L__mod___stages___1___blocks___1___attn_qkv_bn(flatten_26);  flatten_26 = None
    reshape_as_25 = getattr_getattr_l__mod___stages___1___blocks___1___attn_qkv_bn.reshape_as(x_80);  getattr_getattr_l__mod___stages___1___blocks___1___attn_qkv_bn = x_80 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_8 = reshape_as_25.view(8, 49, 8, -1);  reshape_as_25 = None
    split_6 = view_8.split([16, 16, 32], dim = 3);  view_8 = None
    q_11 = split_6[0]
    k_12 = split_6[1]
    v_12 = split_6[2];  split_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    q_12 = q_11.permute(0, 2, 1, 3);  q_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    k_13 = k_12.permute(0, 2, 3, 1);  k_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    v_13 = v_12.permute(0, 2, 1, 3);  v_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_12 = q_12 @ k_13;  q_12 = k_13 = None
    mul_6 = matmul_12 * 0.25;  matmul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_getattr_l__mod___stages___1___blocks___1___attn_attention_biases = self.getattr_getattr_L__mod___stages___1___blocks___1___attn_attention_biases
    getattr_getattr_l__mod___stages___1___blocks___1___attn_attention_bias_idxs = self.getattr_getattr_L__mod___stages___1___blocks___1___attn_attention_bias_idxs
    getitem_27 = getattr_getattr_l__mod___stages___1___blocks___1___attn_attention_biases[(slice(None, None, None), getattr_getattr_l__mod___stages___1___blocks___1___attn_attention_bias_idxs)];  getattr_getattr_l__mod___stages___1___blocks___1___attn_attention_biases = getattr_getattr_l__mod___stages___1___blocks___1___attn_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_12 = mul_6 + getitem_27;  mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    attn_13 = attn_12.softmax(dim = -1);  attn_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    matmul_13 = attn_13 @ v_13;  attn_13 = v_13 = None
    transpose_7 = matmul_13.transpose(1, 2);  matmul_13 = None
    x_81 = transpose_7.reshape(8, 49, 256);  transpose_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    getattr_getattr_l__mod___stages___1___blocks___1___attn_proj_act = self.getattr_getattr_L__mod___stages___1___blocks___1___attn_proj_act(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_82 = self.getattr_getattr_L__mod___stages___1___blocks___1___attn_proj_ln_linear(getattr_getattr_l__mod___stages___1___blocks___1___attn_proj_act);  getattr_getattr_l__mod___stages___1___blocks___1___attn_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_27 = x_82.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___1___attn_proj_ln_bn = self.getattr_getattr_L__mod___stages___1___blocks___1___attn_proj_ln_bn(flatten_27);  flatten_27 = None
    x_83 = getattr_getattr_l__mod___stages___1___blocks___1___attn_proj_ln_bn.reshape_as(x_82);  getattr_getattr_l__mod___stages___1___blocks___1___attn_proj_ln_bn = x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    getattr_getattr_l__mod___stages___1___blocks___1___drop_path1 = self.getattr_getattr_L__mod___stages___1___blocks___1___drop_path1(x_83);  x_83 = None
    x_84 = x_79 + getattr_getattr_l__mod___stages___1___blocks___1___drop_path1;  x_79 = getattr_getattr_l__mod___stages___1___blocks___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_85 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_ln1_linear(x_84)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_28 = x_85.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln1_bn = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_ln1_bn(flatten_28);  flatten_28 = None
    x_86 = getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln1_bn.reshape_as(x_85);  getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln1_bn = x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_87 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_act(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_88 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_drop(x_87);  x_87 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_89 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_ln2_linear(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_29 = x_89.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln2_bn = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_ln2_bn(flatten_29);  flatten_29 = None
    x_90 = getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln2_bn.reshape_as(x_89);  getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln2_bn = x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    getattr_getattr_l__mod___stages___1___blocks___1___drop_path2 = self.getattr_getattr_L__mod___stages___1___blocks___1___drop_path2(x_90);  x_90 = None
    x_91 = x_84 + getattr_getattr_l__mod___stages___1___blocks___1___drop_path2;  x_84 = getattr_getattr_l__mod___stages___1___blocks___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_92 = self.getattr_getattr_L__mod___stages___1___blocks___2___attn_qkv_linear(x_91)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_30 = x_92.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___2___attn_qkv_bn = self.getattr_getattr_L__mod___stages___1___blocks___2___attn_qkv_bn(flatten_30);  flatten_30 = None
    reshape_as_29 = getattr_getattr_l__mod___stages___1___blocks___2___attn_qkv_bn.reshape_as(x_92);  getattr_getattr_l__mod___stages___1___blocks___2___attn_qkv_bn = x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_9 = reshape_as_29.view(8, 49, 8, -1);  reshape_as_29 = None
    split_7 = view_9.split([16, 16, 32], dim = 3);  view_9 = None
    q_13 = split_7[0]
    k_14 = split_7[1]
    v_14 = split_7[2];  split_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    q_14 = q_13.permute(0, 2, 1, 3);  q_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    k_15 = k_14.permute(0, 2, 3, 1);  k_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    v_15 = v_14.permute(0, 2, 1, 3);  v_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_14 = q_14 @ k_15;  q_14 = k_15 = None
    mul_7 = matmul_14 * 0.25;  matmul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_getattr_l__mod___stages___1___blocks___2___attn_attention_biases = self.getattr_getattr_L__mod___stages___1___blocks___2___attn_attention_biases
    getattr_getattr_l__mod___stages___1___blocks___2___attn_attention_bias_idxs = self.getattr_getattr_L__mod___stages___1___blocks___2___attn_attention_bias_idxs
    getitem_31 = getattr_getattr_l__mod___stages___1___blocks___2___attn_attention_biases[(slice(None, None, None), getattr_getattr_l__mod___stages___1___blocks___2___attn_attention_bias_idxs)];  getattr_getattr_l__mod___stages___1___blocks___2___attn_attention_biases = getattr_getattr_l__mod___stages___1___blocks___2___attn_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_14 = mul_7 + getitem_31;  mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    attn_15 = attn_14.softmax(dim = -1);  attn_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    matmul_15 = attn_15 @ v_15;  attn_15 = v_15 = None
    transpose_8 = matmul_15.transpose(1, 2);  matmul_15 = None
    x_93 = transpose_8.reshape(8, 49, 256);  transpose_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    getattr_getattr_l__mod___stages___1___blocks___2___attn_proj_act = self.getattr_getattr_L__mod___stages___1___blocks___2___attn_proj_act(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_94 = self.getattr_getattr_L__mod___stages___1___blocks___2___attn_proj_ln_linear(getattr_getattr_l__mod___stages___1___blocks___2___attn_proj_act);  getattr_getattr_l__mod___stages___1___blocks___2___attn_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_31 = x_94.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___2___attn_proj_ln_bn = self.getattr_getattr_L__mod___stages___1___blocks___2___attn_proj_ln_bn(flatten_31);  flatten_31 = None
    x_95 = getattr_getattr_l__mod___stages___1___blocks___2___attn_proj_ln_bn.reshape_as(x_94);  getattr_getattr_l__mod___stages___1___blocks___2___attn_proj_ln_bn = x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    getattr_getattr_l__mod___stages___1___blocks___2___drop_path1 = self.getattr_getattr_L__mod___stages___1___blocks___2___drop_path1(x_95);  x_95 = None
    x_96 = x_91 + getattr_getattr_l__mod___stages___1___blocks___2___drop_path1;  x_91 = getattr_getattr_l__mod___stages___1___blocks___2___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_97 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_ln1_linear(x_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_32 = x_97.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln1_bn = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_ln1_bn(flatten_32);  flatten_32 = None
    x_98 = getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln1_bn.reshape_as(x_97);  getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln1_bn = x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_99 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_act(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_100 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_drop(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_101 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_ln2_linear(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_33 = x_101.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln2_bn = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_ln2_bn(flatten_33);  flatten_33 = None
    x_102 = getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln2_bn.reshape_as(x_101);  getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln2_bn = x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    getattr_getattr_l__mod___stages___1___blocks___2___drop_path2 = self.getattr_getattr_L__mod___stages___1___blocks___2___drop_path2(x_102);  x_102 = None
    x_103 = x_96 + getattr_getattr_l__mod___stages___1___blocks___2___drop_path2;  x_96 = getattr_getattr_l__mod___stages___1___blocks___2___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_104 = self.getattr_getattr_L__mod___stages___1___blocks___3___attn_qkv_linear(x_103)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_34 = x_104.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___3___attn_qkv_bn = self.getattr_getattr_L__mod___stages___1___blocks___3___attn_qkv_bn(flatten_34);  flatten_34 = None
    reshape_as_33 = getattr_getattr_l__mod___stages___1___blocks___3___attn_qkv_bn.reshape_as(x_104);  getattr_getattr_l__mod___stages___1___blocks___3___attn_qkv_bn = x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_10 = reshape_as_33.view(8, 49, 8, -1);  reshape_as_33 = None
    split_8 = view_10.split([16, 16, 32], dim = 3);  view_10 = None
    q_15 = split_8[0]
    k_16 = split_8[1]
    v_16 = split_8[2];  split_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    q_16 = q_15.permute(0, 2, 1, 3);  q_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    k_17 = k_16.permute(0, 2, 3, 1);  k_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    v_17 = v_16.permute(0, 2, 1, 3);  v_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_16 = q_16 @ k_17;  q_16 = k_17 = None
    mul_8 = matmul_16 * 0.25;  matmul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_getattr_l__mod___stages___1___blocks___3___attn_attention_biases = self.getattr_getattr_L__mod___stages___1___blocks___3___attn_attention_biases
    getattr_getattr_l__mod___stages___1___blocks___3___attn_attention_bias_idxs = self.getattr_getattr_L__mod___stages___1___blocks___3___attn_attention_bias_idxs
    getitem_35 = getattr_getattr_l__mod___stages___1___blocks___3___attn_attention_biases[(slice(None, None, None), getattr_getattr_l__mod___stages___1___blocks___3___attn_attention_bias_idxs)];  getattr_getattr_l__mod___stages___1___blocks___3___attn_attention_biases = getattr_getattr_l__mod___stages___1___blocks___3___attn_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_16 = mul_8 + getitem_35;  mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    attn_17 = attn_16.softmax(dim = -1);  attn_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    matmul_17 = attn_17 @ v_17;  attn_17 = v_17 = None
    transpose_9 = matmul_17.transpose(1, 2);  matmul_17 = None
    x_105 = transpose_9.reshape(8, 49, 256);  transpose_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    getattr_getattr_l__mod___stages___1___blocks___3___attn_proj_act = self.getattr_getattr_L__mod___stages___1___blocks___3___attn_proj_act(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_106 = self.getattr_getattr_L__mod___stages___1___blocks___3___attn_proj_ln_linear(getattr_getattr_l__mod___stages___1___blocks___3___attn_proj_act);  getattr_getattr_l__mod___stages___1___blocks___3___attn_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_35 = x_106.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___3___attn_proj_ln_bn = self.getattr_getattr_L__mod___stages___1___blocks___3___attn_proj_ln_bn(flatten_35);  flatten_35 = None
    x_107 = getattr_getattr_l__mod___stages___1___blocks___3___attn_proj_ln_bn.reshape_as(x_106);  getattr_getattr_l__mod___stages___1___blocks___3___attn_proj_ln_bn = x_106 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    getattr_getattr_l__mod___stages___1___blocks___3___drop_path1 = self.getattr_getattr_L__mod___stages___1___blocks___3___drop_path1(x_107);  x_107 = None
    x_108 = x_103 + getattr_getattr_l__mod___stages___1___blocks___3___drop_path1;  x_103 = getattr_getattr_l__mod___stages___1___blocks___3___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_109 = self.getattr_getattr_L__mod___stages___1___blocks___3___mlp_ln1_linear(x_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_36 = x_109.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln1_bn = self.getattr_getattr_L__mod___stages___1___blocks___3___mlp_ln1_bn(flatten_36);  flatten_36 = None
    x_110 = getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln1_bn.reshape_as(x_109);  getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln1_bn = x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_111 = self.getattr_getattr_L__mod___stages___1___blocks___3___mlp_act(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_112 = self.getattr_getattr_L__mod___stages___1___blocks___3___mlp_drop(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_113 = self.getattr_getattr_L__mod___stages___1___blocks___3___mlp_ln2_linear(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_37 = x_113.flatten(0, 1)
    getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln2_bn = self.getattr_getattr_L__mod___stages___1___blocks___3___mlp_ln2_bn(flatten_37);  flatten_37 = None
    x_114 = getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln2_bn.reshape_as(x_113);  getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln2_bn = x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    getattr_getattr_l__mod___stages___1___blocks___3___drop_path2 = self.getattr_getattr_L__mod___stages___1___blocks___3___drop_path2(x_114);  x_114 = None
    x_116 = x_108 + getattr_getattr_l__mod___stages___1___blocks___3___drop_path2;  x_108 = getattr_getattr_l__mod___stages___1___blocks___3___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_117 = self.getattr_L__mod___stages___2___downsample_attn_downsample_kv_linear(x_116)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_38 = x_117.flatten(0, 1)
    getattr_l__mod___stages___2___downsample_attn_downsample_kv_bn = self.getattr_L__mod___stages___2___downsample_attn_downsample_kv_bn(flatten_38);  flatten_38 = None
    reshape_as_37 = getattr_l__mod___stages___2___downsample_attn_downsample_kv_bn.reshape_as(x_117);  getattr_l__mod___stages___2___downsample_attn_downsample_kv_bn = x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:331, code: k, v = self.kv(x).view(B, N, self.num_heads, -1).split([self.key_dim, self.val_dim], dim=3)
    view_11 = reshape_as_37.view(8, 49, 16, -1);  reshape_as_37 = None
    split_9 = view_11.split([16, 64], dim = 3);  view_11 = None
    k_18 = split_9[0]
    v_18 = split_9[1];  split_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:332, code: k = k.permute(0, 2, 3, 1)  # BHCN
    k_19 = k_18.permute(0, 2, 3, 1);  k_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:333, code: v = v.permute(0, 2, 1, 3)  # BHNC
    v_19 = v_18.permute(0, 2, 1, 3);  v_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:157, code: x = x.view(B, self.resolution[0], self.resolution[1], C)
    x_118 = x_116.view(8, 7, 7, 256);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:161, code: x = x[:, ::self.stride, ::self.stride]
    x_119 = x_118[(slice(None, None, None), slice(None, None, 2), slice(None, None, 2))];  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:162, code: return x.reshape(B, -1, C)
    reshape_10 = x_119.reshape(8, -1, 256);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_120 = self.getattr_L__mod___stages___2___downsample_attn_downsample_q_ln_linear(reshape_10);  reshape_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_39 = x_120.flatten(0, 1)
    getattr_l__mod___stages___2___downsample_attn_downsample_q_ln_bn = self.getattr_L__mod___stages___2___downsample_attn_downsample_q_ln_bn(flatten_39);  flatten_39 = None
    reshape_as_38 = getattr_l__mod___stages___2___downsample_attn_downsample_q_ln_bn.reshape_as(x_120);  getattr_l__mod___stages___2___downsample_attn_downsample_q_ln_bn = x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:334, code: q = self.q(x).view(B, -1, self.num_heads, self.key_dim).permute(0, 2, 1, 3)
    view_13 = reshape_as_38.view(8, -1, 16, 16);  reshape_as_38 = None
    q_17 = view_13.permute(0, 2, 1, 3);  view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_18 = q_17 @ k_19;  q_17 = k_19 = None
    mul_9 = matmul_18 * 0.25;  matmul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:315, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_l__mod___stages___2___downsample_attn_downsample_attention_biases = self.getattr_L__mod___stages___2___downsample_attn_downsample_attention_biases
    getattr_l__mod___stages___2___downsample_attn_downsample_attention_bias_idxs = self.getattr_L__mod___stages___2___downsample_attn_downsample_attention_bias_idxs
    getitem_39 = getattr_l__mod___stages___2___downsample_attn_downsample_attention_biases[(slice(None, None, None), getattr_l__mod___stages___2___downsample_attn_downsample_attention_bias_idxs)];  getattr_l__mod___stages___2___downsample_attn_downsample_attention_biases = getattr_l__mod___stages___2___downsample_attn_downsample_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:336, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_18 = mul_9 + getitem_39;  mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:337, code: attn = attn.softmax(dim=-1)
    attn_19 = attn_18.softmax(dim = -1);  attn_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:339, code: x = (attn @ v).transpose(1, 2).reshape(B, -1, self.val_attn_dim)
    matmul_19 = attn_19 @ v_19;  attn_19 = v_19 = None
    transpose_10 = matmul_19.transpose(1, 2);  matmul_19 = None
    x_121 = transpose_10.reshape(8, -1, 1024);  transpose_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:340, code: x = self.proj(x)
    getattr_l__mod___stages___2___downsample_attn_downsample_proj_act = self.getattr_L__mod___stages___2___downsample_attn_downsample_proj_act(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_122 = self.getattr_L__mod___stages___2___downsample_attn_downsample_proj_ln_linear(getattr_l__mod___stages___2___downsample_attn_downsample_proj_act);  getattr_l__mod___stages___2___downsample_attn_downsample_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_40 = x_122.flatten(0, 1)
    getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn = self.getattr_L__mod___stages___2___downsample_attn_downsample_proj_ln_bn(flatten_40);  flatten_40 = None
    x_124 = getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn.reshape_as(x_122);  getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn = x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_125 = self.getattr_L__mod___stages___2___downsample_mlp_ln1_linear(x_124)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_41 = x_125.flatten(0, 1)
    getattr_l__mod___stages___2___downsample_mlp_ln1_bn = self.getattr_L__mod___stages___2___downsample_mlp_ln1_bn(flatten_41);  flatten_41 = None
    x_126 = getattr_l__mod___stages___2___downsample_mlp_ln1_bn.reshape_as(x_125);  getattr_l__mod___stages___2___downsample_mlp_ln1_bn = x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_127 = self.getattr_L__mod___stages___2___downsample_mlp_act(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_128 = self.getattr_L__mod___stages___2___downsample_mlp_drop(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_129 = self.getattr_L__mod___stages___2___downsample_mlp_ln2_linear(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_42 = x_129.flatten(0, 1)
    getattr_l__mod___stages___2___downsample_mlp_ln2_bn = self.getattr_L__mod___stages___2___downsample_mlp_ln2_bn(flatten_42);  flatten_42 = None
    x_130 = getattr_l__mod___stages___2___downsample_mlp_ln2_bn.reshape_as(x_129);  getattr_l__mod___stages___2___downsample_mlp_ln2_bn = x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:415, code: x = x + self.drop_path(self.mlp(x))
    getattr_l__mod___stages___2___downsample_drop_path = self.getattr_L__mod___stages___2___downsample_drop_path(x_130);  x_130 = None
    x_132 = x_124 + getattr_l__mod___stages___2___downsample_drop_path;  x_124 = getattr_l__mod___stages___2___downsample_drop_path = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_133 = self.getattr_getattr_L__mod___stages___2___blocks___0___attn_qkv_linear(x_132)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_43 = x_133.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___0___attn_qkv_bn = self.getattr_getattr_L__mod___stages___2___blocks___0___attn_qkv_bn(flatten_43);  flatten_43 = None
    reshape_as_42 = getattr_getattr_l__mod___stages___2___blocks___0___attn_qkv_bn.reshape_as(x_133);  getattr_getattr_l__mod___stages___2___blocks___0___attn_qkv_bn = x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_14 = reshape_as_42.view(8, 16, 12, -1);  reshape_as_42 = None
    split_10 = view_14.split([16, 16, 32], dim = 3);  view_14 = None
    q_18 = split_10[0]
    k_20 = split_10[1]
    v_20 = split_10[2];  split_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    q_19 = q_18.permute(0, 2, 1, 3);  q_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    k_21 = k_20.permute(0, 2, 3, 1);  k_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    v_21 = v_20.permute(0, 2, 1, 3);  v_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_20 = q_19 @ k_21;  q_19 = k_21 = None
    mul_10 = matmul_20 * 0.25;  matmul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_getattr_l__mod___stages___2___blocks___0___attn_attention_biases = self.getattr_getattr_L__mod___stages___2___blocks___0___attn_attention_biases
    getattr_getattr_l__mod___stages___2___blocks___0___attn_attention_bias_idxs = self.getattr_getattr_L__mod___stages___2___blocks___0___attn_attention_bias_idxs
    getitem_43 = getattr_getattr_l__mod___stages___2___blocks___0___attn_attention_biases[(slice(None, None, None), getattr_getattr_l__mod___stages___2___blocks___0___attn_attention_bias_idxs)];  getattr_getattr_l__mod___stages___2___blocks___0___attn_attention_biases = getattr_getattr_l__mod___stages___2___blocks___0___attn_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_20 = mul_10 + getitem_43;  mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    attn_21 = attn_20.softmax(dim = -1);  attn_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    matmul_21 = attn_21 @ v_21;  attn_21 = v_21 = None
    transpose_11 = matmul_21.transpose(1, 2);  matmul_21 = None
    x_134 = transpose_11.reshape(8, 16, 384);  transpose_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_act = self.getattr_getattr_L__mod___stages___2___blocks___0___attn_proj_act(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_135 = self.getattr_getattr_L__mod___stages___2___blocks___0___attn_proj_ln_linear(getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_act);  getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_44 = x_135.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_ln_bn = self.getattr_getattr_L__mod___stages___2___blocks___0___attn_proj_ln_bn(flatten_44);  flatten_44 = None
    x_136 = getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_ln_bn.reshape_as(x_135);  getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_ln_bn = x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    getattr_getattr_l__mod___stages___2___blocks___0___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___0___drop_path1(x_136);  x_136 = None
    x_137 = x_132 + getattr_getattr_l__mod___stages___2___blocks___0___drop_path1;  x_132 = getattr_getattr_l__mod___stages___2___blocks___0___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_138 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_ln1_linear(x_137)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_45 = x_138.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___0___mlp_ln1_bn = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_ln1_bn(flatten_45);  flatten_45 = None
    x_139 = getattr_getattr_l__mod___stages___2___blocks___0___mlp_ln1_bn.reshape_as(x_138);  getattr_getattr_l__mod___stages___2___blocks___0___mlp_ln1_bn = x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_140 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_act(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_141 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_drop(x_140);  x_140 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_142 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_ln2_linear(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_46 = x_142.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___0___mlp_ln2_bn = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_ln2_bn(flatten_46);  flatten_46 = None
    x_143 = getattr_getattr_l__mod___stages___2___blocks___0___mlp_ln2_bn.reshape_as(x_142);  getattr_getattr_l__mod___stages___2___blocks___0___mlp_ln2_bn = x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    getattr_getattr_l__mod___stages___2___blocks___0___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___0___drop_path2(x_143);  x_143 = None
    x_144 = x_137 + getattr_getattr_l__mod___stages___2___blocks___0___drop_path2;  x_137 = getattr_getattr_l__mod___stages___2___blocks___0___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_145 = self.getattr_getattr_L__mod___stages___2___blocks___1___attn_qkv_linear(x_144)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_47 = x_145.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___1___attn_qkv_bn = self.getattr_getattr_L__mod___stages___2___blocks___1___attn_qkv_bn(flatten_47);  flatten_47 = None
    reshape_as_46 = getattr_getattr_l__mod___stages___2___blocks___1___attn_qkv_bn.reshape_as(x_145);  getattr_getattr_l__mod___stages___2___blocks___1___attn_qkv_bn = x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_15 = reshape_as_46.view(8, 16, 12, -1);  reshape_as_46 = None
    split_11 = view_15.split([16, 16, 32], dim = 3);  view_15 = None
    q_20 = split_11[0]
    k_22 = split_11[1]
    v_22 = split_11[2];  split_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    q_21 = q_20.permute(0, 2, 1, 3);  q_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    k_23 = k_22.permute(0, 2, 3, 1);  k_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    v_23 = v_22.permute(0, 2, 1, 3);  v_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_22 = q_21 @ k_23;  q_21 = k_23 = None
    mul_11 = matmul_22 * 0.25;  matmul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_getattr_l__mod___stages___2___blocks___1___attn_attention_biases = self.getattr_getattr_L__mod___stages___2___blocks___1___attn_attention_biases
    getattr_getattr_l__mod___stages___2___blocks___1___attn_attention_bias_idxs = self.getattr_getattr_L__mod___stages___2___blocks___1___attn_attention_bias_idxs
    getitem_47 = getattr_getattr_l__mod___stages___2___blocks___1___attn_attention_biases[(slice(None, None, None), getattr_getattr_l__mod___stages___2___blocks___1___attn_attention_bias_idxs)];  getattr_getattr_l__mod___stages___2___blocks___1___attn_attention_biases = getattr_getattr_l__mod___stages___2___blocks___1___attn_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_22 = mul_11 + getitem_47;  mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    attn_23 = attn_22.softmax(dim = -1);  attn_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    matmul_23 = attn_23 @ v_23;  attn_23 = v_23 = None
    transpose_12 = matmul_23.transpose(1, 2);  matmul_23 = None
    x_146 = transpose_12.reshape(8, 16, 384);  transpose_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    getattr_getattr_l__mod___stages___2___blocks___1___attn_proj_act = self.getattr_getattr_L__mod___stages___2___blocks___1___attn_proj_act(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_147 = self.getattr_getattr_L__mod___stages___2___blocks___1___attn_proj_ln_linear(getattr_getattr_l__mod___stages___2___blocks___1___attn_proj_act);  getattr_getattr_l__mod___stages___2___blocks___1___attn_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_48 = x_147.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___1___attn_proj_ln_bn = self.getattr_getattr_L__mod___stages___2___blocks___1___attn_proj_ln_bn(flatten_48);  flatten_48 = None
    x_148 = getattr_getattr_l__mod___stages___2___blocks___1___attn_proj_ln_bn.reshape_as(x_147);  getattr_getattr_l__mod___stages___2___blocks___1___attn_proj_ln_bn = x_147 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    getattr_getattr_l__mod___stages___2___blocks___1___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___1___drop_path1(x_148);  x_148 = None
    x_149 = x_144 + getattr_getattr_l__mod___stages___2___blocks___1___drop_path1;  x_144 = getattr_getattr_l__mod___stages___2___blocks___1___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_150 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_ln1_linear(x_149)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_49 = x_150.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___1___mlp_ln1_bn = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_ln1_bn(flatten_49);  flatten_49 = None
    x_151 = getattr_getattr_l__mod___stages___2___blocks___1___mlp_ln1_bn.reshape_as(x_150);  getattr_getattr_l__mod___stages___2___blocks___1___mlp_ln1_bn = x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_152 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_act(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_153 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_drop(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_154 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_ln2_linear(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_50 = x_154.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___1___mlp_ln2_bn = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_ln2_bn(flatten_50);  flatten_50 = None
    x_155 = getattr_getattr_l__mod___stages___2___blocks___1___mlp_ln2_bn.reshape_as(x_154);  getattr_getattr_l__mod___stages___2___blocks___1___mlp_ln2_bn = x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    getattr_getattr_l__mod___stages___2___blocks___1___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___1___drop_path2(x_155);  x_155 = None
    x_156 = x_149 + getattr_getattr_l__mod___stages___2___blocks___1___drop_path2;  x_149 = getattr_getattr_l__mod___stages___2___blocks___1___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_157 = self.getattr_getattr_L__mod___stages___2___blocks___2___attn_qkv_linear(x_156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_51 = x_157.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___2___attn_qkv_bn = self.getattr_getattr_L__mod___stages___2___blocks___2___attn_qkv_bn(flatten_51);  flatten_51 = None
    reshape_as_50 = getattr_getattr_l__mod___stages___2___blocks___2___attn_qkv_bn.reshape_as(x_157);  getattr_getattr_l__mod___stages___2___blocks___2___attn_qkv_bn = x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_16 = reshape_as_50.view(8, 16, 12, -1);  reshape_as_50 = None
    split_12 = view_16.split([16, 16, 32], dim = 3);  view_16 = None
    q_22 = split_12[0]
    k_24 = split_12[1]
    v_24 = split_12[2];  split_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    q_23 = q_22.permute(0, 2, 1, 3);  q_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    k_25 = k_24.permute(0, 2, 3, 1);  k_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    v_25 = v_24.permute(0, 2, 1, 3);  v_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_24 = q_23 @ k_25;  q_23 = k_25 = None
    mul_12 = matmul_24 * 0.25;  matmul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_getattr_l__mod___stages___2___blocks___2___attn_attention_biases = self.getattr_getattr_L__mod___stages___2___blocks___2___attn_attention_biases
    getattr_getattr_l__mod___stages___2___blocks___2___attn_attention_bias_idxs = self.getattr_getattr_L__mod___stages___2___blocks___2___attn_attention_bias_idxs
    getitem_51 = getattr_getattr_l__mod___stages___2___blocks___2___attn_attention_biases[(slice(None, None, None), getattr_getattr_l__mod___stages___2___blocks___2___attn_attention_bias_idxs)];  getattr_getattr_l__mod___stages___2___blocks___2___attn_attention_biases = getattr_getattr_l__mod___stages___2___blocks___2___attn_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_24 = mul_12 + getitem_51;  mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    attn_25 = attn_24.softmax(dim = -1);  attn_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    matmul_25 = attn_25 @ v_25;  attn_25 = v_25 = None
    transpose_13 = matmul_25.transpose(1, 2);  matmul_25 = None
    x_158 = transpose_13.reshape(8, 16, 384);  transpose_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    getattr_getattr_l__mod___stages___2___blocks___2___attn_proj_act = self.getattr_getattr_L__mod___stages___2___blocks___2___attn_proj_act(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_159 = self.getattr_getattr_L__mod___stages___2___blocks___2___attn_proj_ln_linear(getattr_getattr_l__mod___stages___2___blocks___2___attn_proj_act);  getattr_getattr_l__mod___stages___2___blocks___2___attn_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_52 = x_159.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___2___attn_proj_ln_bn = self.getattr_getattr_L__mod___stages___2___blocks___2___attn_proj_ln_bn(flatten_52);  flatten_52 = None
    x_160 = getattr_getattr_l__mod___stages___2___blocks___2___attn_proj_ln_bn.reshape_as(x_159);  getattr_getattr_l__mod___stages___2___blocks___2___attn_proj_ln_bn = x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    getattr_getattr_l__mod___stages___2___blocks___2___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___2___drop_path1(x_160);  x_160 = None
    x_161 = x_156 + getattr_getattr_l__mod___stages___2___blocks___2___drop_path1;  x_156 = getattr_getattr_l__mod___stages___2___blocks___2___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_162 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_ln1_linear(x_161)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_53 = x_162.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___2___mlp_ln1_bn = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_ln1_bn(flatten_53);  flatten_53 = None
    x_163 = getattr_getattr_l__mod___stages___2___blocks___2___mlp_ln1_bn.reshape_as(x_162);  getattr_getattr_l__mod___stages___2___blocks___2___mlp_ln1_bn = x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_164 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_act(x_163);  x_163 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_165 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_drop(x_164);  x_164 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_166 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_ln2_linear(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_54 = x_166.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___2___mlp_ln2_bn = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_ln2_bn(flatten_54);  flatten_54 = None
    x_167 = getattr_getattr_l__mod___stages___2___blocks___2___mlp_ln2_bn.reshape_as(x_166);  getattr_getattr_l__mod___stages___2___blocks___2___mlp_ln2_bn = x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    getattr_getattr_l__mod___stages___2___blocks___2___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___2___drop_path2(x_167);  x_167 = None
    x_168 = x_161 + getattr_getattr_l__mod___stages___2___blocks___2___drop_path2;  x_161 = getattr_getattr_l__mod___stages___2___blocks___2___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_169 = self.getattr_getattr_L__mod___stages___2___blocks___3___attn_qkv_linear(x_168)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_55 = x_169.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___3___attn_qkv_bn = self.getattr_getattr_L__mod___stages___2___blocks___3___attn_qkv_bn(flatten_55);  flatten_55 = None
    reshape_as_54 = getattr_getattr_l__mod___stages___2___blocks___3___attn_qkv_bn.reshape_as(x_169);  getattr_getattr_l__mod___stages___2___blocks___3___attn_qkv_bn = x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:230, code: q, k, v = self.qkv(x).view(
    view_17 = reshape_as_54.view(8, 16, 12, -1);  reshape_as_54 = None
    split_13 = view_17.split([16, 16, 32], dim = 3);  view_17 = None
    q_24 = split_13[0]
    k_26 = split_13[1]
    v_26 = split_13[2];  split_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:232, code: q = q.permute(0, 2, 1, 3)
    q_25 = q_24.permute(0, 2, 1, 3);  q_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:233, code: k = k.permute(0, 2, 3, 1)
    k_27 = k_26.permute(0, 2, 3, 1);  k_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:234, code: v = v.permute(0, 2, 1, 3)
    v_27 = v_26.permute(0, 2, 1, 3);  v_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    matmul_26 = q_25 @ k_27;  q_25 = k_27 = None
    mul_13 = matmul_26 * 0.25;  matmul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:215, code: self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
    getattr_getattr_l__mod___stages___2___blocks___3___attn_attention_biases = self.getattr_getattr_L__mod___stages___2___blocks___3___attn_attention_biases
    getattr_getattr_l__mod___stages___2___blocks___3___attn_attention_bias_idxs = self.getattr_getattr_L__mod___stages___2___blocks___3___attn_attention_bias_idxs
    getitem_55 = getattr_getattr_l__mod___stages___2___blocks___3___attn_attention_biases[(slice(None, None, None), getattr_getattr_l__mod___stages___2___blocks___3___attn_attention_bias_idxs)];  getattr_getattr_l__mod___stages___2___blocks___3___attn_attention_biases = getattr_getattr_l__mod___stages___2___blocks___3___attn_attention_bias_idxs = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:236, code: attn = q @ k * self.scale + self.get_attention_biases(x.device)
    attn_26 = mul_13 + getitem_55;  mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:237, code: attn = attn.softmax(dim=-1)
    attn_27 = attn_26.softmax(dim = -1);  attn_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:239, code: x = (attn @ v).transpose(1, 2).reshape(B, N, self.val_attn_dim)
    matmul_27 = attn_27 @ v_27;  attn_27 = v_27 = None
    transpose_14 = matmul_27.transpose(1, 2);  matmul_27 = None
    x_170 = transpose_14.reshape(8, 16, 384);  transpose_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:240, code: x = self.proj(x)
    getattr_getattr_l__mod___stages___2___blocks___3___attn_proj_act = self.getattr_getattr_L__mod___stages___2___blocks___3___attn_proj_act(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_171 = self.getattr_getattr_L__mod___stages___2___blocks___3___attn_proj_ln_linear(getattr_getattr_l__mod___stages___2___blocks___3___attn_proj_act);  getattr_getattr_l__mod___stages___2___blocks___3___attn_proj_act = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_56 = x_171.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___3___attn_proj_ln_bn = self.getattr_getattr_L__mod___stages___2___blocks___3___attn_proj_ln_bn(flatten_56);  flatten_56 = None
    x_172 = getattr_getattr_l__mod___stages___2___blocks___3___attn_proj_ln_bn.reshape_as(x_171);  getattr_getattr_l__mod___stages___2___blocks___3___attn_proj_ln_bn = x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:456, code: x = x + self.drop_path1(self.attn(x))
    getattr_getattr_l__mod___stages___2___blocks___3___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___3___drop_path1(x_172);  x_172 = None
    x_173 = x_168 + getattr_getattr_l__mod___stages___2___blocks___3___drop_path1;  x_168 = getattr_getattr_l__mod___stages___2___blocks___3___drop_path1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_174 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_ln1_linear(x_173)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_57 = x_174.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___3___mlp_ln1_bn = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_ln1_bn(flatten_57);  flatten_57 = None
    x_175 = getattr_getattr_l__mod___stages___2___blocks___3___mlp_ln1_bn.reshape_as(x_174);  getattr_getattr_l__mod___stages___2___blocks___3___mlp_ln1_bn = x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:368, code: x = self.act(x)
    x_176 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_act(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:369, code: x = self.drop(x)
    x_177 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_drop(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:88, code: x = self.linear(x)
    x_178 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_ln2_linear(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:89, code: return self.bn(x.flatten(0, 1)).reshape_as(x)
    flatten_58 = x_178.flatten(0, 1)
    getattr_getattr_l__mod___stages___2___blocks___3___mlp_ln2_bn = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_ln2_bn(flatten_58);  flatten_58 = None
    x_179 = getattr_getattr_l__mod___stages___2___blocks___3___mlp_ln2_bn.reshape_as(x_178);  getattr_getattr_l__mod___stages___2___blocks___3___mlp_ln2_bn = x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:457, code: x = x + self.drop_path2(self.mlp(x))
    getattr_getattr_l__mod___stages___2___blocks___3___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___3___drop_path2(x_179);  x_179 = None
    x_183 = x_173 + getattr_getattr_l__mod___stages___2___blocks___3___drop_path2;  x_173 = getattr_getattr_l__mod___stages___2___blocks___3___drop_path2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:681, code: x = x.mean(dim=(-2, -1)) if self.use_conv else x.mean(dim=1)
    x_184 = x_183.mean(dim = 1);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:119, code: return self.linear(self.drop(self.bn(x)))
    l__mod___head_bn = self.L__mod___head_bn(x_184)
    l__mod___head_drop = self.L__mod___head_drop(l__mod___head_bn);  l__mod___head_bn = None
    x_185 = self.L__mod___head_linear(l__mod___head_drop);  l__mod___head_drop = None
    l__mod___head_dist_bn = self.L__mod___head_dist_bn(x_184);  x_184 = None
    l__mod___head_dist_drop = self.L__mod___head_dist_drop(l__mod___head_dist_bn);  l__mod___head_dist_bn = None
    x_dist = self.L__mod___head_dist_linear(l__mod___head_dist_drop);  l__mod___head_dist_drop = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/levit.py:690, code: return (x + x_dist) / 2
    add_40 = x_185 + x_dist;  x_185 = x_dist = None
    x_186 = add_40 / 2;  add_40 = None
    return (x_186, getitem_3, getitem_7, getitem_11, getitem_15, getitem_19, getitem_23, getitem_27, getitem_31, getitem_35, getitem_39, getitem_43, getitem_47, getitem_51, getitem_55)
    