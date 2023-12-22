from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:72, code: x = self.conv(x)
    x = self.L__mod___stem_conv(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:73, code: x = self.norm(x)
    x_2 = self.L__mod___stem_norm(x);  x = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:430, code: x = self.downsample(x)
    x_3 = self.getattr_L__mod___stages___0___downsample(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___0___blocks___0___res_scale1 = self.getattr_getattr_L__mod___stages___0___blocks___0___res_scale1(x_3)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___0___norm1_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___norm1_weight
    getattr_getattr_l__mod___stages___0___blocks___0___norm1_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___norm1_bias
    group_norm = torch.nn.functional.group_norm(x_3, 1, getattr_getattr_l__mod___stages___0___blocks___0___norm1_weight, getattr_getattr_l__mod___stages___0___blocks___0___norm1_bias, 1e-05);  x_3 = getattr_getattr_l__mod___stages___0___blocks___0___norm1_weight = getattr_getattr_l__mod___stages___0___blocks___0___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y = self.getattr_getattr_L__mod___stages___0___blocks___0___token_mixer_pool(group_norm)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub = y - group_norm;  y = group_norm = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___0___blocks___0___drop_path1 = self.getattr_getattr_L__mod___stages___0___blocks___0___drop_path1(sub);  sub = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___0___blocks___0___layer_scale1_scale = self.getattr_getattr_L__mod___stages___0___blocks___0___layer_scale1_scale
    view = getattr_getattr_l__mod___stages___0___blocks___0___layer_scale1_scale.view((96, 1, 1));  getattr_getattr_l__mod___stages___0___blocks___0___layer_scale1_scale = None
    mul = getattr_getattr_l__mod___stages___0___blocks___0___drop_path1 * view;  getattr_getattr_l__mod___stages___0___blocks___0___drop_path1 = view = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_4 = getattr_getattr_l__mod___stages___0___blocks___0___res_scale1 + mul;  getattr_getattr_l__mod___stages___0___blocks___0___res_scale1 = mul = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___0___blocks___0___res_scale2 = self.getattr_getattr_L__mod___stages___0___blocks___0___res_scale2(x_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___0___norm2_weight = self.getattr_getattr_L__mod___stages___0___blocks___0___norm2_weight
    getattr_getattr_l__mod___stages___0___blocks___0___norm2_bias = self.getattr_getattr_L__mod___stages___0___blocks___0___norm2_bias
    group_norm_1 = torch.nn.functional.group_norm(x_4, 1, getattr_getattr_l__mod___stages___0___blocks___0___norm2_weight, getattr_getattr_l__mod___stages___0___blocks___0___norm2_bias, 1e-05);  x_4 = getattr_getattr_l__mod___stages___0___blocks___0___norm2_weight = getattr_getattr_l__mod___stages___0___blocks___0___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_5 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_fc1(group_norm_1);  group_norm_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_6 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_act(x_5);  x_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_7 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_drop1(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_8 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_norm(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_9 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_fc2(x_8);  x_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_10 = self.getattr_getattr_L__mod___stages___0___blocks___0___mlp_drop2(x_9);  x_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___0___blocks___0___drop_path2 = self.getattr_getattr_L__mod___stages___0___blocks___0___drop_path2(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___0___blocks___0___layer_scale2_scale = self.getattr_getattr_L__mod___stages___0___blocks___0___layer_scale2_scale
    view_1 = getattr_getattr_l__mod___stages___0___blocks___0___layer_scale2_scale.view((96, 1, 1));  getattr_getattr_l__mod___stages___0___blocks___0___layer_scale2_scale = None
    mul_1 = getattr_getattr_l__mod___stages___0___blocks___0___drop_path2 * view_1;  getattr_getattr_l__mod___stages___0___blocks___0___drop_path2 = view_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_11 = getattr_getattr_l__mod___stages___0___blocks___0___res_scale2 + mul_1;  getattr_getattr_l__mod___stages___0___blocks___0___res_scale2 = mul_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___0___blocks___1___res_scale1 = self.getattr_getattr_L__mod___stages___0___blocks___1___res_scale1(x_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___1___norm1_weight = self.getattr_getattr_L__mod___stages___0___blocks___1___norm1_weight
    getattr_getattr_l__mod___stages___0___blocks___1___norm1_bias = self.getattr_getattr_L__mod___stages___0___blocks___1___norm1_bias
    group_norm_2 = torch.nn.functional.group_norm(x_11, 1, getattr_getattr_l__mod___stages___0___blocks___1___norm1_weight, getattr_getattr_l__mod___stages___0___blocks___1___norm1_bias, 1e-05);  x_11 = getattr_getattr_l__mod___stages___0___blocks___1___norm1_weight = getattr_getattr_l__mod___stages___0___blocks___1___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_1 = self.getattr_getattr_L__mod___stages___0___blocks___1___token_mixer_pool(group_norm_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_1 = y_1 - group_norm_2;  y_1 = group_norm_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___0___blocks___1___drop_path1 = self.getattr_getattr_L__mod___stages___0___blocks___1___drop_path1(sub_1);  sub_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___0___blocks___1___layer_scale1_scale = self.getattr_getattr_L__mod___stages___0___blocks___1___layer_scale1_scale
    view_2 = getattr_getattr_l__mod___stages___0___blocks___1___layer_scale1_scale.view((96, 1, 1));  getattr_getattr_l__mod___stages___0___blocks___1___layer_scale1_scale = None
    mul_2 = getattr_getattr_l__mod___stages___0___blocks___1___drop_path1 * view_2;  getattr_getattr_l__mod___stages___0___blocks___1___drop_path1 = view_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_12 = getattr_getattr_l__mod___stages___0___blocks___1___res_scale1 + mul_2;  getattr_getattr_l__mod___stages___0___blocks___1___res_scale1 = mul_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___0___blocks___1___res_scale2 = self.getattr_getattr_L__mod___stages___0___blocks___1___res_scale2(x_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___1___norm2_weight = self.getattr_getattr_L__mod___stages___0___blocks___1___norm2_weight
    getattr_getattr_l__mod___stages___0___blocks___1___norm2_bias = self.getattr_getattr_L__mod___stages___0___blocks___1___norm2_bias
    group_norm_3 = torch.nn.functional.group_norm(x_12, 1, getattr_getattr_l__mod___stages___0___blocks___1___norm2_weight, getattr_getattr_l__mod___stages___0___blocks___1___norm2_bias, 1e-05);  x_12 = getattr_getattr_l__mod___stages___0___blocks___1___norm2_weight = getattr_getattr_l__mod___stages___0___blocks___1___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_13 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_fc1(group_norm_3);  group_norm_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_14 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_act(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_15 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_drop1(x_14);  x_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_16 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_norm(x_15);  x_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_17 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_fc2(x_16);  x_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_18 = self.getattr_getattr_L__mod___stages___0___blocks___1___mlp_drop2(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___0___blocks___1___drop_path2 = self.getattr_getattr_L__mod___stages___0___blocks___1___drop_path2(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___0___blocks___1___layer_scale2_scale = self.getattr_getattr_L__mod___stages___0___blocks___1___layer_scale2_scale
    view_3 = getattr_getattr_l__mod___stages___0___blocks___1___layer_scale2_scale.view((96, 1, 1));  getattr_getattr_l__mod___stages___0___blocks___1___layer_scale2_scale = None
    mul_3 = getattr_getattr_l__mod___stages___0___blocks___1___drop_path2 * view_3;  getattr_getattr_l__mod___stages___0___blocks___1___drop_path2 = view_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_19 = getattr_getattr_l__mod___stages___0___blocks___1___res_scale2 + mul_3;  getattr_getattr_l__mod___stages___0___blocks___1___res_scale2 = mul_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___0___blocks___2___res_scale1 = self.getattr_getattr_L__mod___stages___0___blocks___2___res_scale1(x_19)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___2___norm1_weight = self.getattr_getattr_L__mod___stages___0___blocks___2___norm1_weight
    getattr_getattr_l__mod___stages___0___blocks___2___norm1_bias = self.getattr_getattr_L__mod___stages___0___blocks___2___norm1_bias
    group_norm_4 = torch.nn.functional.group_norm(x_19, 1, getattr_getattr_l__mod___stages___0___blocks___2___norm1_weight, getattr_getattr_l__mod___stages___0___blocks___2___norm1_bias, 1e-05);  x_19 = getattr_getattr_l__mod___stages___0___blocks___2___norm1_weight = getattr_getattr_l__mod___stages___0___blocks___2___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_2 = self.getattr_getattr_L__mod___stages___0___blocks___2___token_mixer_pool(group_norm_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_2 = y_2 - group_norm_4;  y_2 = group_norm_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___0___blocks___2___drop_path1 = self.getattr_getattr_L__mod___stages___0___blocks___2___drop_path1(sub_2);  sub_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___0___blocks___2___layer_scale1_scale = self.getattr_getattr_L__mod___stages___0___blocks___2___layer_scale1_scale
    view_4 = getattr_getattr_l__mod___stages___0___blocks___2___layer_scale1_scale.view((96, 1, 1));  getattr_getattr_l__mod___stages___0___blocks___2___layer_scale1_scale = None
    mul_4 = getattr_getattr_l__mod___stages___0___blocks___2___drop_path1 * view_4;  getattr_getattr_l__mod___stages___0___blocks___2___drop_path1 = view_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_20 = getattr_getattr_l__mod___stages___0___blocks___2___res_scale1 + mul_4;  getattr_getattr_l__mod___stages___0___blocks___2___res_scale1 = mul_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___0___blocks___2___res_scale2 = self.getattr_getattr_L__mod___stages___0___blocks___2___res_scale2(x_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___2___norm2_weight = self.getattr_getattr_L__mod___stages___0___blocks___2___norm2_weight
    getattr_getattr_l__mod___stages___0___blocks___2___norm2_bias = self.getattr_getattr_L__mod___stages___0___blocks___2___norm2_bias
    group_norm_5 = torch.nn.functional.group_norm(x_20, 1, getattr_getattr_l__mod___stages___0___blocks___2___norm2_weight, getattr_getattr_l__mod___stages___0___blocks___2___norm2_bias, 1e-05);  x_20 = getattr_getattr_l__mod___stages___0___blocks___2___norm2_weight = getattr_getattr_l__mod___stages___0___blocks___2___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_21 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_fc1(group_norm_5);  group_norm_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_22 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_act(x_21);  x_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_23 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_drop1(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_24 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_norm(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_25 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_fc2(x_24);  x_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_26 = self.getattr_getattr_L__mod___stages___0___blocks___2___mlp_drop2(x_25);  x_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___0___blocks___2___drop_path2 = self.getattr_getattr_L__mod___stages___0___blocks___2___drop_path2(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___0___blocks___2___layer_scale2_scale = self.getattr_getattr_L__mod___stages___0___blocks___2___layer_scale2_scale
    view_5 = getattr_getattr_l__mod___stages___0___blocks___2___layer_scale2_scale.view((96, 1, 1));  getattr_getattr_l__mod___stages___0___blocks___2___layer_scale2_scale = None
    mul_5 = getattr_getattr_l__mod___stages___0___blocks___2___drop_path2 * view_5;  getattr_getattr_l__mod___stages___0___blocks___2___drop_path2 = view_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_27 = getattr_getattr_l__mod___stages___0___blocks___2___res_scale2 + mul_5;  getattr_getattr_l__mod___stages___0___blocks___2___res_scale2 = mul_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___0___blocks___3___res_scale1 = self.getattr_getattr_L__mod___stages___0___blocks___3___res_scale1(x_27)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___3___norm1_weight = self.getattr_getattr_L__mod___stages___0___blocks___3___norm1_weight
    getattr_getattr_l__mod___stages___0___blocks___3___norm1_bias = self.getattr_getattr_L__mod___stages___0___blocks___3___norm1_bias
    group_norm_6 = torch.nn.functional.group_norm(x_27, 1, getattr_getattr_l__mod___stages___0___blocks___3___norm1_weight, getattr_getattr_l__mod___stages___0___blocks___3___norm1_bias, 1e-05);  x_27 = getattr_getattr_l__mod___stages___0___blocks___3___norm1_weight = getattr_getattr_l__mod___stages___0___blocks___3___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_3 = self.getattr_getattr_L__mod___stages___0___blocks___3___token_mixer_pool(group_norm_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_3 = y_3 - group_norm_6;  y_3 = group_norm_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___0___blocks___3___drop_path1 = self.getattr_getattr_L__mod___stages___0___blocks___3___drop_path1(sub_3);  sub_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___0___blocks___3___layer_scale1_scale = self.getattr_getattr_L__mod___stages___0___blocks___3___layer_scale1_scale
    view_6 = getattr_getattr_l__mod___stages___0___blocks___3___layer_scale1_scale.view((96, 1, 1));  getattr_getattr_l__mod___stages___0___blocks___3___layer_scale1_scale = None
    mul_6 = getattr_getattr_l__mod___stages___0___blocks___3___drop_path1 * view_6;  getattr_getattr_l__mod___stages___0___blocks___3___drop_path1 = view_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_28 = getattr_getattr_l__mod___stages___0___blocks___3___res_scale1 + mul_6;  getattr_getattr_l__mod___stages___0___blocks___3___res_scale1 = mul_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___0___blocks___3___res_scale2 = self.getattr_getattr_L__mod___stages___0___blocks___3___res_scale2(x_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___3___norm2_weight = self.getattr_getattr_L__mod___stages___0___blocks___3___norm2_weight
    getattr_getattr_l__mod___stages___0___blocks___3___norm2_bias = self.getattr_getattr_L__mod___stages___0___blocks___3___norm2_bias
    group_norm_7 = torch.nn.functional.group_norm(x_28, 1, getattr_getattr_l__mod___stages___0___blocks___3___norm2_weight, getattr_getattr_l__mod___stages___0___blocks___3___norm2_bias, 1e-05);  x_28 = getattr_getattr_l__mod___stages___0___blocks___3___norm2_weight = getattr_getattr_l__mod___stages___0___blocks___3___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_29 = self.getattr_getattr_L__mod___stages___0___blocks___3___mlp_fc1(group_norm_7);  group_norm_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_30 = self.getattr_getattr_L__mod___stages___0___blocks___3___mlp_act(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_31 = self.getattr_getattr_L__mod___stages___0___blocks___3___mlp_drop1(x_30);  x_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_32 = self.getattr_getattr_L__mod___stages___0___blocks___3___mlp_norm(x_31);  x_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_33 = self.getattr_getattr_L__mod___stages___0___blocks___3___mlp_fc2(x_32);  x_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_34 = self.getattr_getattr_L__mod___stages___0___blocks___3___mlp_drop2(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___0___blocks___3___drop_path2 = self.getattr_getattr_L__mod___stages___0___blocks___3___drop_path2(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___0___blocks___3___layer_scale2_scale = self.getattr_getattr_L__mod___stages___0___blocks___3___layer_scale2_scale
    view_7 = getattr_getattr_l__mod___stages___0___blocks___3___layer_scale2_scale.view((96, 1, 1));  getattr_getattr_l__mod___stages___0___blocks___3___layer_scale2_scale = None
    mul_7 = getattr_getattr_l__mod___stages___0___blocks___3___drop_path2 * view_7;  getattr_getattr_l__mod___stages___0___blocks___3___drop_path2 = view_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_35 = getattr_getattr_l__mod___stages___0___blocks___3___res_scale2 + mul_7;  getattr_getattr_l__mod___stages___0___blocks___3___res_scale2 = mul_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___0___blocks___4___res_scale1 = self.getattr_getattr_L__mod___stages___0___blocks___4___res_scale1(x_35)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___4___norm1_weight = self.getattr_getattr_L__mod___stages___0___blocks___4___norm1_weight
    getattr_getattr_l__mod___stages___0___blocks___4___norm1_bias = self.getattr_getattr_L__mod___stages___0___blocks___4___norm1_bias
    group_norm_8 = torch.nn.functional.group_norm(x_35, 1, getattr_getattr_l__mod___stages___0___blocks___4___norm1_weight, getattr_getattr_l__mod___stages___0___blocks___4___norm1_bias, 1e-05);  x_35 = getattr_getattr_l__mod___stages___0___blocks___4___norm1_weight = getattr_getattr_l__mod___stages___0___blocks___4___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_4 = self.getattr_getattr_L__mod___stages___0___blocks___4___token_mixer_pool(group_norm_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_4 = y_4 - group_norm_8;  y_4 = group_norm_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___0___blocks___4___drop_path1 = self.getattr_getattr_L__mod___stages___0___blocks___4___drop_path1(sub_4);  sub_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___0___blocks___4___layer_scale1_scale = self.getattr_getattr_L__mod___stages___0___blocks___4___layer_scale1_scale
    view_8 = getattr_getattr_l__mod___stages___0___blocks___4___layer_scale1_scale.view((96, 1, 1));  getattr_getattr_l__mod___stages___0___blocks___4___layer_scale1_scale = None
    mul_8 = getattr_getattr_l__mod___stages___0___blocks___4___drop_path1 * view_8;  getattr_getattr_l__mod___stages___0___blocks___4___drop_path1 = view_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_36 = getattr_getattr_l__mod___stages___0___blocks___4___res_scale1 + mul_8;  getattr_getattr_l__mod___stages___0___blocks___4___res_scale1 = mul_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___0___blocks___4___res_scale2 = self.getattr_getattr_L__mod___stages___0___blocks___4___res_scale2(x_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___4___norm2_weight = self.getattr_getattr_L__mod___stages___0___blocks___4___norm2_weight
    getattr_getattr_l__mod___stages___0___blocks___4___norm2_bias = self.getattr_getattr_L__mod___stages___0___blocks___4___norm2_bias
    group_norm_9 = torch.nn.functional.group_norm(x_36, 1, getattr_getattr_l__mod___stages___0___blocks___4___norm2_weight, getattr_getattr_l__mod___stages___0___blocks___4___norm2_bias, 1e-05);  x_36 = getattr_getattr_l__mod___stages___0___blocks___4___norm2_weight = getattr_getattr_l__mod___stages___0___blocks___4___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_37 = self.getattr_getattr_L__mod___stages___0___blocks___4___mlp_fc1(group_norm_9);  group_norm_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_38 = self.getattr_getattr_L__mod___stages___0___blocks___4___mlp_act(x_37);  x_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_39 = self.getattr_getattr_L__mod___stages___0___blocks___4___mlp_drop1(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_40 = self.getattr_getattr_L__mod___stages___0___blocks___4___mlp_norm(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_41 = self.getattr_getattr_L__mod___stages___0___blocks___4___mlp_fc2(x_40);  x_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_42 = self.getattr_getattr_L__mod___stages___0___blocks___4___mlp_drop2(x_41);  x_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___0___blocks___4___drop_path2 = self.getattr_getattr_L__mod___stages___0___blocks___4___drop_path2(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___0___blocks___4___layer_scale2_scale = self.getattr_getattr_L__mod___stages___0___blocks___4___layer_scale2_scale
    view_9 = getattr_getattr_l__mod___stages___0___blocks___4___layer_scale2_scale.view((96, 1, 1));  getattr_getattr_l__mod___stages___0___blocks___4___layer_scale2_scale = None
    mul_9 = getattr_getattr_l__mod___stages___0___blocks___4___drop_path2 * view_9;  getattr_getattr_l__mod___stages___0___blocks___4___drop_path2 = view_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_43 = getattr_getattr_l__mod___stages___0___blocks___4___res_scale2 + mul_9;  getattr_getattr_l__mod___stages___0___blocks___4___res_scale2 = mul_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___0___blocks___5___res_scale1 = self.getattr_getattr_L__mod___stages___0___blocks___5___res_scale1(x_43)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___5___norm1_weight = self.getattr_getattr_L__mod___stages___0___blocks___5___norm1_weight
    getattr_getattr_l__mod___stages___0___blocks___5___norm1_bias = self.getattr_getattr_L__mod___stages___0___blocks___5___norm1_bias
    group_norm_10 = torch.nn.functional.group_norm(x_43, 1, getattr_getattr_l__mod___stages___0___blocks___5___norm1_weight, getattr_getattr_l__mod___stages___0___blocks___5___norm1_bias, 1e-05);  x_43 = getattr_getattr_l__mod___stages___0___blocks___5___norm1_weight = getattr_getattr_l__mod___stages___0___blocks___5___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_5 = self.getattr_getattr_L__mod___stages___0___blocks___5___token_mixer_pool(group_norm_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_5 = y_5 - group_norm_10;  y_5 = group_norm_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___0___blocks___5___drop_path1 = self.getattr_getattr_L__mod___stages___0___blocks___5___drop_path1(sub_5);  sub_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___0___blocks___5___layer_scale1_scale = self.getattr_getattr_L__mod___stages___0___blocks___5___layer_scale1_scale
    view_10 = getattr_getattr_l__mod___stages___0___blocks___5___layer_scale1_scale.view((96, 1, 1));  getattr_getattr_l__mod___stages___0___blocks___5___layer_scale1_scale = None
    mul_10 = getattr_getattr_l__mod___stages___0___blocks___5___drop_path1 * view_10;  getattr_getattr_l__mod___stages___0___blocks___5___drop_path1 = view_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_44 = getattr_getattr_l__mod___stages___0___blocks___5___res_scale1 + mul_10;  getattr_getattr_l__mod___stages___0___blocks___5___res_scale1 = mul_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___0___blocks___5___res_scale2 = self.getattr_getattr_L__mod___stages___0___blocks___5___res_scale2(x_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___0___blocks___5___norm2_weight = self.getattr_getattr_L__mod___stages___0___blocks___5___norm2_weight
    getattr_getattr_l__mod___stages___0___blocks___5___norm2_bias = self.getattr_getattr_L__mod___stages___0___blocks___5___norm2_bias
    group_norm_11 = torch.nn.functional.group_norm(x_44, 1, getattr_getattr_l__mod___stages___0___blocks___5___norm2_weight, getattr_getattr_l__mod___stages___0___blocks___5___norm2_bias, 1e-05);  x_44 = getattr_getattr_l__mod___stages___0___blocks___5___norm2_weight = getattr_getattr_l__mod___stages___0___blocks___5___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_45 = self.getattr_getattr_L__mod___stages___0___blocks___5___mlp_fc1(group_norm_11);  group_norm_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_46 = self.getattr_getattr_L__mod___stages___0___blocks___5___mlp_act(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_47 = self.getattr_getattr_L__mod___stages___0___blocks___5___mlp_drop1(x_46);  x_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_48 = self.getattr_getattr_L__mod___stages___0___blocks___5___mlp_norm(x_47);  x_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_49 = self.getattr_getattr_L__mod___stages___0___blocks___5___mlp_fc2(x_48);  x_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_50 = self.getattr_getattr_L__mod___stages___0___blocks___5___mlp_drop2(x_49);  x_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___0___blocks___5___drop_path2 = self.getattr_getattr_L__mod___stages___0___blocks___5___drop_path2(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___0___blocks___5___layer_scale2_scale = self.getattr_getattr_L__mod___stages___0___blocks___5___layer_scale2_scale
    view_11 = getattr_getattr_l__mod___stages___0___blocks___5___layer_scale2_scale.view((96, 1, 1));  getattr_getattr_l__mod___stages___0___blocks___5___layer_scale2_scale = None
    mul_11 = getattr_getattr_l__mod___stages___0___blocks___5___drop_path2 * view_11;  getattr_getattr_l__mod___stages___0___blocks___5___drop_path2 = view_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_52 = getattr_getattr_l__mod___stages___0___blocks___5___res_scale2 + mul_11;  getattr_getattr_l__mod___stages___0___blocks___5___res_scale2 = mul_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:102, code: x = self.norm(x)
    x_53 = self.getattr_L__mod___stages___1___downsample_norm(x_52);  x_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:103, code: x = self.conv(x)
    x_55 = self.getattr_L__mod___stages___1___downsample_conv(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___1___blocks___0___res_scale1 = self.getattr_getattr_L__mod___stages___1___blocks___0___res_scale1(x_55)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___0___norm1_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___norm1_weight
    getattr_getattr_l__mod___stages___1___blocks___0___norm1_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___norm1_bias
    group_norm_12 = torch.nn.functional.group_norm(x_55, 1, getattr_getattr_l__mod___stages___1___blocks___0___norm1_weight, getattr_getattr_l__mod___stages___1___blocks___0___norm1_bias, 1e-05);  x_55 = getattr_getattr_l__mod___stages___1___blocks___0___norm1_weight = getattr_getattr_l__mod___stages___1___blocks___0___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_6 = self.getattr_getattr_L__mod___stages___1___blocks___0___token_mixer_pool(group_norm_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_6 = y_6 - group_norm_12;  y_6 = group_norm_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___1___blocks___0___drop_path1 = self.getattr_getattr_L__mod___stages___1___blocks___0___drop_path1(sub_6);  sub_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___1___blocks___0___layer_scale1_scale = self.getattr_getattr_L__mod___stages___1___blocks___0___layer_scale1_scale
    view_12 = getattr_getattr_l__mod___stages___1___blocks___0___layer_scale1_scale.view((192, 1, 1));  getattr_getattr_l__mod___stages___1___blocks___0___layer_scale1_scale = None
    mul_12 = getattr_getattr_l__mod___stages___1___blocks___0___drop_path1 * view_12;  getattr_getattr_l__mod___stages___1___blocks___0___drop_path1 = view_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_56 = getattr_getattr_l__mod___stages___1___blocks___0___res_scale1 + mul_12;  getattr_getattr_l__mod___stages___1___blocks___0___res_scale1 = mul_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___1___blocks___0___res_scale2 = self.getattr_getattr_L__mod___stages___1___blocks___0___res_scale2(x_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___0___norm2_weight = self.getattr_getattr_L__mod___stages___1___blocks___0___norm2_weight
    getattr_getattr_l__mod___stages___1___blocks___0___norm2_bias = self.getattr_getattr_L__mod___stages___1___blocks___0___norm2_bias
    group_norm_13 = torch.nn.functional.group_norm(x_56, 1, getattr_getattr_l__mod___stages___1___blocks___0___norm2_weight, getattr_getattr_l__mod___stages___1___blocks___0___norm2_bias, 1e-05);  x_56 = getattr_getattr_l__mod___stages___1___blocks___0___norm2_weight = getattr_getattr_l__mod___stages___1___blocks___0___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_57 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_fc1(group_norm_13);  group_norm_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_58 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_act(x_57);  x_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_59 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_drop1(x_58);  x_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_60 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_norm(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_61 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_fc2(x_60);  x_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_62 = self.getattr_getattr_L__mod___stages___1___blocks___0___mlp_drop2(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___1___blocks___0___drop_path2 = self.getattr_getattr_L__mod___stages___1___blocks___0___drop_path2(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___1___blocks___0___layer_scale2_scale = self.getattr_getattr_L__mod___stages___1___blocks___0___layer_scale2_scale
    view_13 = getattr_getattr_l__mod___stages___1___blocks___0___layer_scale2_scale.view((192, 1, 1));  getattr_getattr_l__mod___stages___1___blocks___0___layer_scale2_scale = None
    mul_13 = getattr_getattr_l__mod___stages___1___blocks___0___drop_path2 * view_13;  getattr_getattr_l__mod___stages___1___blocks___0___drop_path2 = view_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_63 = getattr_getattr_l__mod___stages___1___blocks___0___res_scale2 + mul_13;  getattr_getattr_l__mod___stages___1___blocks___0___res_scale2 = mul_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___1___blocks___1___res_scale1 = self.getattr_getattr_L__mod___stages___1___blocks___1___res_scale1(x_63)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___1___norm1_weight = self.getattr_getattr_L__mod___stages___1___blocks___1___norm1_weight
    getattr_getattr_l__mod___stages___1___blocks___1___norm1_bias = self.getattr_getattr_L__mod___stages___1___blocks___1___norm1_bias
    group_norm_14 = torch.nn.functional.group_norm(x_63, 1, getattr_getattr_l__mod___stages___1___blocks___1___norm1_weight, getattr_getattr_l__mod___stages___1___blocks___1___norm1_bias, 1e-05);  x_63 = getattr_getattr_l__mod___stages___1___blocks___1___norm1_weight = getattr_getattr_l__mod___stages___1___blocks___1___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_7 = self.getattr_getattr_L__mod___stages___1___blocks___1___token_mixer_pool(group_norm_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_7 = y_7 - group_norm_14;  y_7 = group_norm_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___1___blocks___1___drop_path1 = self.getattr_getattr_L__mod___stages___1___blocks___1___drop_path1(sub_7);  sub_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___1___blocks___1___layer_scale1_scale = self.getattr_getattr_L__mod___stages___1___blocks___1___layer_scale1_scale
    view_14 = getattr_getattr_l__mod___stages___1___blocks___1___layer_scale1_scale.view((192, 1, 1));  getattr_getattr_l__mod___stages___1___blocks___1___layer_scale1_scale = None
    mul_14 = getattr_getattr_l__mod___stages___1___blocks___1___drop_path1 * view_14;  getattr_getattr_l__mod___stages___1___blocks___1___drop_path1 = view_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_64 = getattr_getattr_l__mod___stages___1___blocks___1___res_scale1 + mul_14;  getattr_getattr_l__mod___stages___1___blocks___1___res_scale1 = mul_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___1___blocks___1___res_scale2 = self.getattr_getattr_L__mod___stages___1___blocks___1___res_scale2(x_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___1___norm2_weight = self.getattr_getattr_L__mod___stages___1___blocks___1___norm2_weight
    getattr_getattr_l__mod___stages___1___blocks___1___norm2_bias = self.getattr_getattr_L__mod___stages___1___blocks___1___norm2_bias
    group_norm_15 = torch.nn.functional.group_norm(x_64, 1, getattr_getattr_l__mod___stages___1___blocks___1___norm2_weight, getattr_getattr_l__mod___stages___1___blocks___1___norm2_bias, 1e-05);  x_64 = getattr_getattr_l__mod___stages___1___blocks___1___norm2_weight = getattr_getattr_l__mod___stages___1___blocks___1___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_65 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_fc1(group_norm_15);  group_norm_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_66 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_act(x_65);  x_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_67 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_drop1(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_68 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_norm(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_69 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_fc2(x_68);  x_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_70 = self.getattr_getattr_L__mod___stages___1___blocks___1___mlp_drop2(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___1___blocks___1___drop_path2 = self.getattr_getattr_L__mod___stages___1___blocks___1___drop_path2(x_70);  x_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___1___blocks___1___layer_scale2_scale = self.getattr_getattr_L__mod___stages___1___blocks___1___layer_scale2_scale
    view_15 = getattr_getattr_l__mod___stages___1___blocks___1___layer_scale2_scale.view((192, 1, 1));  getattr_getattr_l__mod___stages___1___blocks___1___layer_scale2_scale = None
    mul_15 = getattr_getattr_l__mod___stages___1___blocks___1___drop_path2 * view_15;  getattr_getattr_l__mod___stages___1___blocks___1___drop_path2 = view_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_71 = getattr_getattr_l__mod___stages___1___blocks___1___res_scale2 + mul_15;  getattr_getattr_l__mod___stages___1___blocks___1___res_scale2 = mul_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___1___blocks___2___res_scale1 = self.getattr_getattr_L__mod___stages___1___blocks___2___res_scale1(x_71)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___2___norm1_weight = self.getattr_getattr_L__mod___stages___1___blocks___2___norm1_weight
    getattr_getattr_l__mod___stages___1___blocks___2___norm1_bias = self.getattr_getattr_L__mod___stages___1___blocks___2___norm1_bias
    group_norm_16 = torch.nn.functional.group_norm(x_71, 1, getattr_getattr_l__mod___stages___1___blocks___2___norm1_weight, getattr_getattr_l__mod___stages___1___blocks___2___norm1_bias, 1e-05);  x_71 = getattr_getattr_l__mod___stages___1___blocks___2___norm1_weight = getattr_getattr_l__mod___stages___1___blocks___2___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_8 = self.getattr_getattr_L__mod___stages___1___blocks___2___token_mixer_pool(group_norm_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_8 = y_8 - group_norm_16;  y_8 = group_norm_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___1___blocks___2___drop_path1 = self.getattr_getattr_L__mod___stages___1___blocks___2___drop_path1(sub_8);  sub_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___1___blocks___2___layer_scale1_scale = self.getattr_getattr_L__mod___stages___1___blocks___2___layer_scale1_scale
    view_16 = getattr_getattr_l__mod___stages___1___blocks___2___layer_scale1_scale.view((192, 1, 1));  getattr_getattr_l__mod___stages___1___blocks___2___layer_scale1_scale = None
    mul_16 = getattr_getattr_l__mod___stages___1___blocks___2___drop_path1 * view_16;  getattr_getattr_l__mod___stages___1___blocks___2___drop_path1 = view_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_72 = getattr_getattr_l__mod___stages___1___blocks___2___res_scale1 + mul_16;  getattr_getattr_l__mod___stages___1___blocks___2___res_scale1 = mul_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___1___blocks___2___res_scale2 = self.getattr_getattr_L__mod___stages___1___blocks___2___res_scale2(x_72)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___2___norm2_weight = self.getattr_getattr_L__mod___stages___1___blocks___2___norm2_weight
    getattr_getattr_l__mod___stages___1___blocks___2___norm2_bias = self.getattr_getattr_L__mod___stages___1___blocks___2___norm2_bias
    group_norm_17 = torch.nn.functional.group_norm(x_72, 1, getattr_getattr_l__mod___stages___1___blocks___2___norm2_weight, getattr_getattr_l__mod___stages___1___blocks___2___norm2_bias, 1e-05);  x_72 = getattr_getattr_l__mod___stages___1___blocks___2___norm2_weight = getattr_getattr_l__mod___stages___1___blocks___2___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_73 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_fc1(group_norm_17);  group_norm_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_74 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_act(x_73);  x_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_75 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_drop1(x_74);  x_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_76 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_norm(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_77 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_fc2(x_76);  x_76 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_78 = self.getattr_getattr_L__mod___stages___1___blocks___2___mlp_drop2(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___1___blocks___2___drop_path2 = self.getattr_getattr_L__mod___stages___1___blocks___2___drop_path2(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___1___blocks___2___layer_scale2_scale = self.getattr_getattr_L__mod___stages___1___blocks___2___layer_scale2_scale
    view_17 = getattr_getattr_l__mod___stages___1___blocks___2___layer_scale2_scale.view((192, 1, 1));  getattr_getattr_l__mod___stages___1___blocks___2___layer_scale2_scale = None
    mul_17 = getattr_getattr_l__mod___stages___1___blocks___2___drop_path2 * view_17;  getattr_getattr_l__mod___stages___1___blocks___2___drop_path2 = view_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_79 = getattr_getattr_l__mod___stages___1___blocks___2___res_scale2 + mul_17;  getattr_getattr_l__mod___stages___1___blocks___2___res_scale2 = mul_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___1___blocks___3___res_scale1 = self.getattr_getattr_L__mod___stages___1___blocks___3___res_scale1(x_79)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___3___norm1_weight = self.getattr_getattr_L__mod___stages___1___blocks___3___norm1_weight
    getattr_getattr_l__mod___stages___1___blocks___3___norm1_bias = self.getattr_getattr_L__mod___stages___1___blocks___3___norm1_bias
    group_norm_18 = torch.nn.functional.group_norm(x_79, 1, getattr_getattr_l__mod___stages___1___blocks___3___norm1_weight, getattr_getattr_l__mod___stages___1___blocks___3___norm1_bias, 1e-05);  x_79 = getattr_getattr_l__mod___stages___1___blocks___3___norm1_weight = getattr_getattr_l__mod___stages___1___blocks___3___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_9 = self.getattr_getattr_L__mod___stages___1___blocks___3___token_mixer_pool(group_norm_18)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_9 = y_9 - group_norm_18;  y_9 = group_norm_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___1___blocks___3___drop_path1 = self.getattr_getattr_L__mod___stages___1___blocks___3___drop_path1(sub_9);  sub_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___1___blocks___3___layer_scale1_scale = self.getattr_getattr_L__mod___stages___1___blocks___3___layer_scale1_scale
    view_18 = getattr_getattr_l__mod___stages___1___blocks___3___layer_scale1_scale.view((192, 1, 1));  getattr_getattr_l__mod___stages___1___blocks___3___layer_scale1_scale = None
    mul_18 = getattr_getattr_l__mod___stages___1___blocks___3___drop_path1 * view_18;  getattr_getattr_l__mod___stages___1___blocks___3___drop_path1 = view_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_80 = getattr_getattr_l__mod___stages___1___blocks___3___res_scale1 + mul_18;  getattr_getattr_l__mod___stages___1___blocks___3___res_scale1 = mul_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___1___blocks___3___res_scale2 = self.getattr_getattr_L__mod___stages___1___blocks___3___res_scale2(x_80)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___3___norm2_weight = self.getattr_getattr_L__mod___stages___1___blocks___3___norm2_weight
    getattr_getattr_l__mod___stages___1___blocks___3___norm2_bias = self.getattr_getattr_L__mod___stages___1___blocks___3___norm2_bias
    group_norm_19 = torch.nn.functional.group_norm(x_80, 1, getattr_getattr_l__mod___stages___1___blocks___3___norm2_weight, getattr_getattr_l__mod___stages___1___blocks___3___norm2_bias, 1e-05);  x_80 = getattr_getattr_l__mod___stages___1___blocks___3___norm2_weight = getattr_getattr_l__mod___stages___1___blocks___3___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_81 = self.getattr_getattr_L__mod___stages___1___blocks___3___mlp_fc1(group_norm_19);  group_norm_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_82 = self.getattr_getattr_L__mod___stages___1___blocks___3___mlp_act(x_81);  x_81 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_83 = self.getattr_getattr_L__mod___stages___1___blocks___3___mlp_drop1(x_82);  x_82 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_84 = self.getattr_getattr_L__mod___stages___1___blocks___3___mlp_norm(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_85 = self.getattr_getattr_L__mod___stages___1___blocks___3___mlp_fc2(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_86 = self.getattr_getattr_L__mod___stages___1___blocks___3___mlp_drop2(x_85);  x_85 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___1___blocks___3___drop_path2 = self.getattr_getattr_L__mod___stages___1___blocks___3___drop_path2(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___1___blocks___3___layer_scale2_scale = self.getattr_getattr_L__mod___stages___1___blocks___3___layer_scale2_scale
    view_19 = getattr_getattr_l__mod___stages___1___blocks___3___layer_scale2_scale.view((192, 1, 1));  getattr_getattr_l__mod___stages___1___blocks___3___layer_scale2_scale = None
    mul_19 = getattr_getattr_l__mod___stages___1___blocks___3___drop_path2 * view_19;  getattr_getattr_l__mod___stages___1___blocks___3___drop_path2 = view_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_87 = getattr_getattr_l__mod___stages___1___blocks___3___res_scale2 + mul_19;  getattr_getattr_l__mod___stages___1___blocks___3___res_scale2 = mul_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___1___blocks___4___res_scale1 = self.getattr_getattr_L__mod___stages___1___blocks___4___res_scale1(x_87)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___4___norm1_weight = self.getattr_getattr_L__mod___stages___1___blocks___4___norm1_weight
    getattr_getattr_l__mod___stages___1___blocks___4___norm1_bias = self.getattr_getattr_L__mod___stages___1___blocks___4___norm1_bias
    group_norm_20 = torch.nn.functional.group_norm(x_87, 1, getattr_getattr_l__mod___stages___1___blocks___4___norm1_weight, getattr_getattr_l__mod___stages___1___blocks___4___norm1_bias, 1e-05);  x_87 = getattr_getattr_l__mod___stages___1___blocks___4___norm1_weight = getattr_getattr_l__mod___stages___1___blocks___4___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_10 = self.getattr_getattr_L__mod___stages___1___blocks___4___token_mixer_pool(group_norm_20)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_10 = y_10 - group_norm_20;  y_10 = group_norm_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___1___blocks___4___drop_path1 = self.getattr_getattr_L__mod___stages___1___blocks___4___drop_path1(sub_10);  sub_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___1___blocks___4___layer_scale1_scale = self.getattr_getattr_L__mod___stages___1___blocks___4___layer_scale1_scale
    view_20 = getattr_getattr_l__mod___stages___1___blocks___4___layer_scale1_scale.view((192, 1, 1));  getattr_getattr_l__mod___stages___1___blocks___4___layer_scale1_scale = None
    mul_20 = getattr_getattr_l__mod___stages___1___blocks___4___drop_path1 * view_20;  getattr_getattr_l__mod___stages___1___blocks___4___drop_path1 = view_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_88 = getattr_getattr_l__mod___stages___1___blocks___4___res_scale1 + mul_20;  getattr_getattr_l__mod___stages___1___blocks___4___res_scale1 = mul_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___1___blocks___4___res_scale2 = self.getattr_getattr_L__mod___stages___1___blocks___4___res_scale2(x_88)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___4___norm2_weight = self.getattr_getattr_L__mod___stages___1___blocks___4___norm2_weight
    getattr_getattr_l__mod___stages___1___blocks___4___norm2_bias = self.getattr_getattr_L__mod___stages___1___blocks___4___norm2_bias
    group_norm_21 = torch.nn.functional.group_norm(x_88, 1, getattr_getattr_l__mod___stages___1___blocks___4___norm2_weight, getattr_getattr_l__mod___stages___1___blocks___4___norm2_bias, 1e-05);  x_88 = getattr_getattr_l__mod___stages___1___blocks___4___norm2_weight = getattr_getattr_l__mod___stages___1___blocks___4___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_89 = self.getattr_getattr_L__mod___stages___1___blocks___4___mlp_fc1(group_norm_21);  group_norm_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_90 = self.getattr_getattr_L__mod___stages___1___blocks___4___mlp_act(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_91 = self.getattr_getattr_L__mod___stages___1___blocks___4___mlp_drop1(x_90);  x_90 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_92 = self.getattr_getattr_L__mod___stages___1___blocks___4___mlp_norm(x_91);  x_91 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_93 = self.getattr_getattr_L__mod___stages___1___blocks___4___mlp_fc2(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_94 = self.getattr_getattr_L__mod___stages___1___blocks___4___mlp_drop2(x_93);  x_93 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___1___blocks___4___drop_path2 = self.getattr_getattr_L__mod___stages___1___blocks___4___drop_path2(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___1___blocks___4___layer_scale2_scale = self.getattr_getattr_L__mod___stages___1___blocks___4___layer_scale2_scale
    view_21 = getattr_getattr_l__mod___stages___1___blocks___4___layer_scale2_scale.view((192, 1, 1));  getattr_getattr_l__mod___stages___1___blocks___4___layer_scale2_scale = None
    mul_21 = getattr_getattr_l__mod___stages___1___blocks___4___drop_path2 * view_21;  getattr_getattr_l__mod___stages___1___blocks___4___drop_path2 = view_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_95 = getattr_getattr_l__mod___stages___1___blocks___4___res_scale2 + mul_21;  getattr_getattr_l__mod___stages___1___blocks___4___res_scale2 = mul_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___1___blocks___5___res_scale1 = self.getattr_getattr_L__mod___stages___1___blocks___5___res_scale1(x_95)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___5___norm1_weight = self.getattr_getattr_L__mod___stages___1___blocks___5___norm1_weight
    getattr_getattr_l__mod___stages___1___blocks___5___norm1_bias = self.getattr_getattr_L__mod___stages___1___blocks___5___norm1_bias
    group_norm_22 = torch.nn.functional.group_norm(x_95, 1, getattr_getattr_l__mod___stages___1___blocks___5___norm1_weight, getattr_getattr_l__mod___stages___1___blocks___5___norm1_bias, 1e-05);  x_95 = getattr_getattr_l__mod___stages___1___blocks___5___norm1_weight = getattr_getattr_l__mod___stages___1___blocks___5___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_11 = self.getattr_getattr_L__mod___stages___1___blocks___5___token_mixer_pool(group_norm_22)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_11 = y_11 - group_norm_22;  y_11 = group_norm_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___1___blocks___5___drop_path1 = self.getattr_getattr_L__mod___stages___1___blocks___5___drop_path1(sub_11);  sub_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___1___blocks___5___layer_scale1_scale = self.getattr_getattr_L__mod___stages___1___blocks___5___layer_scale1_scale
    view_22 = getattr_getattr_l__mod___stages___1___blocks___5___layer_scale1_scale.view((192, 1, 1));  getattr_getattr_l__mod___stages___1___blocks___5___layer_scale1_scale = None
    mul_22 = getattr_getattr_l__mod___stages___1___blocks___5___drop_path1 * view_22;  getattr_getattr_l__mod___stages___1___blocks___5___drop_path1 = view_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_96 = getattr_getattr_l__mod___stages___1___blocks___5___res_scale1 + mul_22;  getattr_getattr_l__mod___stages___1___blocks___5___res_scale1 = mul_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___1___blocks___5___res_scale2 = self.getattr_getattr_L__mod___stages___1___blocks___5___res_scale2(x_96)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___1___blocks___5___norm2_weight = self.getattr_getattr_L__mod___stages___1___blocks___5___norm2_weight
    getattr_getattr_l__mod___stages___1___blocks___5___norm2_bias = self.getattr_getattr_L__mod___stages___1___blocks___5___norm2_bias
    group_norm_23 = torch.nn.functional.group_norm(x_96, 1, getattr_getattr_l__mod___stages___1___blocks___5___norm2_weight, getattr_getattr_l__mod___stages___1___blocks___5___norm2_bias, 1e-05);  x_96 = getattr_getattr_l__mod___stages___1___blocks___5___norm2_weight = getattr_getattr_l__mod___stages___1___blocks___5___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_97 = self.getattr_getattr_L__mod___stages___1___blocks___5___mlp_fc1(group_norm_23);  group_norm_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_98 = self.getattr_getattr_L__mod___stages___1___blocks___5___mlp_act(x_97);  x_97 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_99 = self.getattr_getattr_L__mod___stages___1___blocks___5___mlp_drop1(x_98);  x_98 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_100 = self.getattr_getattr_L__mod___stages___1___blocks___5___mlp_norm(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_101 = self.getattr_getattr_L__mod___stages___1___blocks___5___mlp_fc2(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_102 = self.getattr_getattr_L__mod___stages___1___blocks___5___mlp_drop2(x_101);  x_101 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___1___blocks___5___drop_path2 = self.getattr_getattr_L__mod___stages___1___blocks___5___drop_path2(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___1___blocks___5___layer_scale2_scale = self.getattr_getattr_L__mod___stages___1___blocks___5___layer_scale2_scale
    view_23 = getattr_getattr_l__mod___stages___1___blocks___5___layer_scale2_scale.view((192, 1, 1));  getattr_getattr_l__mod___stages___1___blocks___5___layer_scale2_scale = None
    mul_23 = getattr_getattr_l__mod___stages___1___blocks___5___drop_path2 * view_23;  getattr_getattr_l__mod___stages___1___blocks___5___drop_path2 = view_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_104 = getattr_getattr_l__mod___stages___1___blocks___5___res_scale2 + mul_23;  getattr_getattr_l__mod___stages___1___blocks___5___res_scale2 = mul_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:102, code: x = self.norm(x)
    x_105 = self.getattr_L__mod___stages___2___downsample_norm(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:103, code: x = self.conv(x)
    x_107 = self.getattr_L__mod___stages___2___downsample_conv(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___0___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___0___res_scale1(x_107)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___0___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___0___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___norm1_bias
    group_norm_24 = torch.nn.functional.group_norm(x_107, 1, getattr_getattr_l__mod___stages___2___blocks___0___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___0___norm1_bias, 1e-05);  x_107 = getattr_getattr_l__mod___stages___2___blocks___0___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___0___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_12 = self.getattr_getattr_L__mod___stages___2___blocks___0___token_mixer_pool(group_norm_24)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_12 = y_12 - group_norm_24;  y_12 = group_norm_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___0___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___0___drop_path1(sub_12);  sub_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___0___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___0___layer_scale1_scale
    view_24 = getattr_getattr_l__mod___stages___2___blocks___0___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___0___layer_scale1_scale = None
    mul_24 = getattr_getattr_l__mod___stages___2___blocks___0___drop_path1 * view_24;  getattr_getattr_l__mod___stages___2___blocks___0___drop_path1 = view_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_108 = getattr_getattr_l__mod___stages___2___blocks___0___res_scale1 + mul_24;  getattr_getattr_l__mod___stages___2___blocks___0___res_scale1 = mul_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___0___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___0___res_scale2(x_108)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___0___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___0___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___0___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___0___norm2_bias
    group_norm_25 = torch.nn.functional.group_norm(x_108, 1, getattr_getattr_l__mod___stages___2___blocks___0___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___0___norm2_bias, 1e-05);  x_108 = getattr_getattr_l__mod___stages___2___blocks___0___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___0___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_109 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_fc1(group_norm_25);  group_norm_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_110 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_act(x_109);  x_109 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_111 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_drop1(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_112 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_norm(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_113 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_fc2(x_112);  x_112 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_114 = self.getattr_getattr_L__mod___stages___2___blocks___0___mlp_drop2(x_113);  x_113 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___0___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___0___drop_path2(x_114);  x_114 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___0___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___0___layer_scale2_scale
    view_25 = getattr_getattr_l__mod___stages___2___blocks___0___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___0___layer_scale2_scale = None
    mul_25 = getattr_getattr_l__mod___stages___2___blocks___0___drop_path2 * view_25;  getattr_getattr_l__mod___stages___2___blocks___0___drop_path2 = view_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_115 = getattr_getattr_l__mod___stages___2___blocks___0___res_scale2 + mul_25;  getattr_getattr_l__mod___stages___2___blocks___0___res_scale2 = mul_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___1___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___1___res_scale1(x_115)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___1___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___1___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___1___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___1___norm1_bias
    group_norm_26 = torch.nn.functional.group_norm(x_115, 1, getattr_getattr_l__mod___stages___2___blocks___1___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___1___norm1_bias, 1e-05);  x_115 = getattr_getattr_l__mod___stages___2___blocks___1___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___1___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_13 = self.getattr_getattr_L__mod___stages___2___blocks___1___token_mixer_pool(group_norm_26)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_13 = y_13 - group_norm_26;  y_13 = group_norm_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___1___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___1___drop_path1(sub_13);  sub_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___1___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___1___layer_scale1_scale
    view_26 = getattr_getattr_l__mod___stages___2___blocks___1___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___1___layer_scale1_scale = None
    mul_26 = getattr_getattr_l__mod___stages___2___blocks___1___drop_path1 * view_26;  getattr_getattr_l__mod___stages___2___blocks___1___drop_path1 = view_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_116 = getattr_getattr_l__mod___stages___2___blocks___1___res_scale1 + mul_26;  getattr_getattr_l__mod___stages___2___blocks___1___res_scale1 = mul_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___1___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___1___res_scale2(x_116)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___1___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___1___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___1___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___1___norm2_bias
    group_norm_27 = torch.nn.functional.group_norm(x_116, 1, getattr_getattr_l__mod___stages___2___blocks___1___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___1___norm2_bias, 1e-05);  x_116 = getattr_getattr_l__mod___stages___2___blocks___1___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___1___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_117 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_fc1(group_norm_27);  group_norm_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_118 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_act(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_119 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_drop1(x_118);  x_118 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_120 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_norm(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_121 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_fc2(x_120);  x_120 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_122 = self.getattr_getattr_L__mod___stages___2___blocks___1___mlp_drop2(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___1___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___1___drop_path2(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___1___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___1___layer_scale2_scale
    view_27 = getattr_getattr_l__mod___stages___2___blocks___1___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___1___layer_scale2_scale = None
    mul_27 = getattr_getattr_l__mod___stages___2___blocks___1___drop_path2 * view_27;  getattr_getattr_l__mod___stages___2___blocks___1___drop_path2 = view_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_123 = getattr_getattr_l__mod___stages___2___blocks___1___res_scale2 + mul_27;  getattr_getattr_l__mod___stages___2___blocks___1___res_scale2 = mul_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___2___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___2___res_scale1(x_123)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___2___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___2___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___2___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___2___norm1_bias
    group_norm_28 = torch.nn.functional.group_norm(x_123, 1, getattr_getattr_l__mod___stages___2___blocks___2___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___2___norm1_bias, 1e-05);  x_123 = getattr_getattr_l__mod___stages___2___blocks___2___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___2___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_14 = self.getattr_getattr_L__mod___stages___2___blocks___2___token_mixer_pool(group_norm_28)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_14 = y_14 - group_norm_28;  y_14 = group_norm_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___2___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___2___drop_path1(sub_14);  sub_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___2___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___2___layer_scale1_scale
    view_28 = getattr_getattr_l__mod___stages___2___blocks___2___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___2___layer_scale1_scale = None
    mul_28 = getattr_getattr_l__mod___stages___2___blocks___2___drop_path1 * view_28;  getattr_getattr_l__mod___stages___2___blocks___2___drop_path1 = view_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_124 = getattr_getattr_l__mod___stages___2___blocks___2___res_scale1 + mul_28;  getattr_getattr_l__mod___stages___2___blocks___2___res_scale1 = mul_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___2___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___2___res_scale2(x_124)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___2___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___2___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___2___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___2___norm2_bias
    group_norm_29 = torch.nn.functional.group_norm(x_124, 1, getattr_getattr_l__mod___stages___2___blocks___2___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___2___norm2_bias, 1e-05);  x_124 = getattr_getattr_l__mod___stages___2___blocks___2___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___2___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_125 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_fc1(group_norm_29);  group_norm_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_126 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_act(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_127 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_drop1(x_126);  x_126 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_128 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_norm(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_129 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_fc2(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_130 = self.getattr_getattr_L__mod___stages___2___blocks___2___mlp_drop2(x_129);  x_129 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___2___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___2___drop_path2(x_130);  x_130 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___2___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___2___layer_scale2_scale
    view_29 = getattr_getattr_l__mod___stages___2___blocks___2___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___2___layer_scale2_scale = None
    mul_29 = getattr_getattr_l__mod___stages___2___blocks___2___drop_path2 * view_29;  getattr_getattr_l__mod___stages___2___blocks___2___drop_path2 = view_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_131 = getattr_getattr_l__mod___stages___2___blocks___2___res_scale2 + mul_29;  getattr_getattr_l__mod___stages___2___blocks___2___res_scale2 = mul_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___3___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___3___res_scale1(x_131)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___3___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___3___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___3___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___3___norm1_bias
    group_norm_30 = torch.nn.functional.group_norm(x_131, 1, getattr_getattr_l__mod___stages___2___blocks___3___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___3___norm1_bias, 1e-05);  x_131 = getattr_getattr_l__mod___stages___2___blocks___3___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___3___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_15 = self.getattr_getattr_L__mod___stages___2___blocks___3___token_mixer_pool(group_norm_30)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_15 = y_15 - group_norm_30;  y_15 = group_norm_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___3___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___3___drop_path1(sub_15);  sub_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___3___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___3___layer_scale1_scale
    view_30 = getattr_getattr_l__mod___stages___2___blocks___3___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___3___layer_scale1_scale = None
    mul_30 = getattr_getattr_l__mod___stages___2___blocks___3___drop_path1 * view_30;  getattr_getattr_l__mod___stages___2___blocks___3___drop_path1 = view_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_132 = getattr_getattr_l__mod___stages___2___blocks___3___res_scale1 + mul_30;  getattr_getattr_l__mod___stages___2___blocks___3___res_scale1 = mul_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___3___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___3___res_scale2(x_132)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___3___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___3___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___3___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___3___norm2_bias
    group_norm_31 = torch.nn.functional.group_norm(x_132, 1, getattr_getattr_l__mod___stages___2___blocks___3___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___3___norm2_bias, 1e-05);  x_132 = getattr_getattr_l__mod___stages___2___blocks___3___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___3___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_133 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_fc1(group_norm_31);  group_norm_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_134 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_act(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_135 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_drop1(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_136 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_norm(x_135);  x_135 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_137 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_fc2(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_138 = self.getattr_getattr_L__mod___stages___2___blocks___3___mlp_drop2(x_137);  x_137 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___3___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___3___drop_path2(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___3___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___3___layer_scale2_scale
    view_31 = getattr_getattr_l__mod___stages___2___blocks___3___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___3___layer_scale2_scale = None
    mul_31 = getattr_getattr_l__mod___stages___2___blocks___3___drop_path2 * view_31;  getattr_getattr_l__mod___stages___2___blocks___3___drop_path2 = view_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_139 = getattr_getattr_l__mod___stages___2___blocks___3___res_scale2 + mul_31;  getattr_getattr_l__mod___stages___2___blocks___3___res_scale2 = mul_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___4___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___4___res_scale1(x_139)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___4___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___4___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___4___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___4___norm1_bias
    group_norm_32 = torch.nn.functional.group_norm(x_139, 1, getattr_getattr_l__mod___stages___2___blocks___4___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___4___norm1_bias, 1e-05);  x_139 = getattr_getattr_l__mod___stages___2___blocks___4___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___4___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_16 = self.getattr_getattr_L__mod___stages___2___blocks___4___token_mixer_pool(group_norm_32)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_16 = y_16 - group_norm_32;  y_16 = group_norm_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___4___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___4___drop_path1(sub_16);  sub_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___4___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___4___layer_scale1_scale
    view_32 = getattr_getattr_l__mod___stages___2___blocks___4___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___4___layer_scale1_scale = None
    mul_32 = getattr_getattr_l__mod___stages___2___blocks___4___drop_path1 * view_32;  getattr_getattr_l__mod___stages___2___blocks___4___drop_path1 = view_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_140 = getattr_getattr_l__mod___stages___2___blocks___4___res_scale1 + mul_32;  getattr_getattr_l__mod___stages___2___blocks___4___res_scale1 = mul_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___4___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___4___res_scale2(x_140)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___4___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___4___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___4___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___4___norm2_bias
    group_norm_33 = torch.nn.functional.group_norm(x_140, 1, getattr_getattr_l__mod___stages___2___blocks___4___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___4___norm2_bias, 1e-05);  x_140 = getattr_getattr_l__mod___stages___2___blocks___4___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___4___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_141 = self.getattr_getattr_L__mod___stages___2___blocks___4___mlp_fc1(group_norm_33);  group_norm_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_142 = self.getattr_getattr_L__mod___stages___2___blocks___4___mlp_act(x_141);  x_141 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_143 = self.getattr_getattr_L__mod___stages___2___blocks___4___mlp_drop1(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_144 = self.getattr_getattr_L__mod___stages___2___blocks___4___mlp_norm(x_143);  x_143 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_145 = self.getattr_getattr_L__mod___stages___2___blocks___4___mlp_fc2(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_146 = self.getattr_getattr_L__mod___stages___2___blocks___4___mlp_drop2(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___4___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___4___drop_path2(x_146);  x_146 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___4___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___4___layer_scale2_scale
    view_33 = getattr_getattr_l__mod___stages___2___blocks___4___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___4___layer_scale2_scale = None
    mul_33 = getattr_getattr_l__mod___stages___2___blocks___4___drop_path2 * view_33;  getattr_getattr_l__mod___stages___2___blocks___4___drop_path2 = view_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_147 = getattr_getattr_l__mod___stages___2___blocks___4___res_scale2 + mul_33;  getattr_getattr_l__mod___stages___2___blocks___4___res_scale2 = mul_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___5___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___5___res_scale1(x_147)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___5___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___5___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___5___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___5___norm1_bias
    group_norm_34 = torch.nn.functional.group_norm(x_147, 1, getattr_getattr_l__mod___stages___2___blocks___5___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___5___norm1_bias, 1e-05);  x_147 = getattr_getattr_l__mod___stages___2___blocks___5___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___5___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_17 = self.getattr_getattr_L__mod___stages___2___blocks___5___token_mixer_pool(group_norm_34)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_17 = y_17 - group_norm_34;  y_17 = group_norm_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___5___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___5___drop_path1(sub_17);  sub_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___5___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___5___layer_scale1_scale
    view_34 = getattr_getattr_l__mod___stages___2___blocks___5___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___5___layer_scale1_scale = None
    mul_34 = getattr_getattr_l__mod___stages___2___blocks___5___drop_path1 * view_34;  getattr_getattr_l__mod___stages___2___blocks___5___drop_path1 = view_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_148 = getattr_getattr_l__mod___stages___2___blocks___5___res_scale1 + mul_34;  getattr_getattr_l__mod___stages___2___blocks___5___res_scale1 = mul_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___5___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___5___res_scale2(x_148)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___5___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___5___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___5___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___5___norm2_bias
    group_norm_35 = torch.nn.functional.group_norm(x_148, 1, getattr_getattr_l__mod___stages___2___blocks___5___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___5___norm2_bias, 1e-05);  x_148 = getattr_getattr_l__mod___stages___2___blocks___5___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___5___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_149 = self.getattr_getattr_L__mod___stages___2___blocks___5___mlp_fc1(group_norm_35);  group_norm_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_150 = self.getattr_getattr_L__mod___stages___2___blocks___5___mlp_act(x_149);  x_149 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_151 = self.getattr_getattr_L__mod___stages___2___blocks___5___mlp_drop1(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_152 = self.getattr_getattr_L__mod___stages___2___blocks___5___mlp_norm(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_153 = self.getattr_getattr_L__mod___stages___2___blocks___5___mlp_fc2(x_152);  x_152 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_154 = self.getattr_getattr_L__mod___stages___2___blocks___5___mlp_drop2(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___5___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___5___drop_path2(x_154);  x_154 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___5___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___5___layer_scale2_scale
    view_35 = getattr_getattr_l__mod___stages___2___blocks___5___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___5___layer_scale2_scale = None
    mul_35 = getattr_getattr_l__mod___stages___2___blocks___5___drop_path2 * view_35;  getattr_getattr_l__mod___stages___2___blocks___5___drop_path2 = view_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_155 = getattr_getattr_l__mod___stages___2___blocks___5___res_scale2 + mul_35;  getattr_getattr_l__mod___stages___2___blocks___5___res_scale2 = mul_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___6___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___6___res_scale1(x_155)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___6___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___6___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___6___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___6___norm1_bias
    group_norm_36 = torch.nn.functional.group_norm(x_155, 1, getattr_getattr_l__mod___stages___2___blocks___6___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___6___norm1_bias, 1e-05);  x_155 = getattr_getattr_l__mod___stages___2___blocks___6___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___6___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_18 = self.getattr_getattr_L__mod___stages___2___blocks___6___token_mixer_pool(group_norm_36)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_18 = y_18 - group_norm_36;  y_18 = group_norm_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___6___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___6___drop_path1(sub_18);  sub_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___6___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___6___layer_scale1_scale
    view_36 = getattr_getattr_l__mod___stages___2___blocks___6___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___6___layer_scale1_scale = None
    mul_36 = getattr_getattr_l__mod___stages___2___blocks___6___drop_path1 * view_36;  getattr_getattr_l__mod___stages___2___blocks___6___drop_path1 = view_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_156 = getattr_getattr_l__mod___stages___2___blocks___6___res_scale1 + mul_36;  getattr_getattr_l__mod___stages___2___blocks___6___res_scale1 = mul_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___6___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___6___res_scale2(x_156)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___6___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___6___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___6___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___6___norm2_bias
    group_norm_37 = torch.nn.functional.group_norm(x_156, 1, getattr_getattr_l__mod___stages___2___blocks___6___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___6___norm2_bias, 1e-05);  x_156 = getattr_getattr_l__mod___stages___2___blocks___6___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___6___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_157 = self.getattr_getattr_L__mod___stages___2___blocks___6___mlp_fc1(group_norm_37);  group_norm_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_158 = self.getattr_getattr_L__mod___stages___2___blocks___6___mlp_act(x_157);  x_157 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_159 = self.getattr_getattr_L__mod___stages___2___blocks___6___mlp_drop1(x_158);  x_158 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_160 = self.getattr_getattr_L__mod___stages___2___blocks___6___mlp_norm(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_161 = self.getattr_getattr_L__mod___stages___2___blocks___6___mlp_fc2(x_160);  x_160 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_162 = self.getattr_getattr_L__mod___stages___2___blocks___6___mlp_drop2(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___6___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___6___drop_path2(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___6___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___6___layer_scale2_scale
    view_37 = getattr_getattr_l__mod___stages___2___blocks___6___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___6___layer_scale2_scale = None
    mul_37 = getattr_getattr_l__mod___stages___2___blocks___6___drop_path2 * view_37;  getattr_getattr_l__mod___stages___2___blocks___6___drop_path2 = view_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_163 = getattr_getattr_l__mod___stages___2___blocks___6___res_scale2 + mul_37;  getattr_getattr_l__mod___stages___2___blocks___6___res_scale2 = mul_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___7___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___7___res_scale1(x_163)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___7___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___7___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___7___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___7___norm1_bias
    group_norm_38 = torch.nn.functional.group_norm(x_163, 1, getattr_getattr_l__mod___stages___2___blocks___7___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___7___norm1_bias, 1e-05);  x_163 = getattr_getattr_l__mod___stages___2___blocks___7___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___7___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_19 = self.getattr_getattr_L__mod___stages___2___blocks___7___token_mixer_pool(group_norm_38)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_19 = y_19 - group_norm_38;  y_19 = group_norm_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___7___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___7___drop_path1(sub_19);  sub_19 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___7___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___7___layer_scale1_scale
    view_38 = getattr_getattr_l__mod___stages___2___blocks___7___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___7___layer_scale1_scale = None
    mul_38 = getattr_getattr_l__mod___stages___2___blocks___7___drop_path1 * view_38;  getattr_getattr_l__mod___stages___2___blocks___7___drop_path1 = view_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_164 = getattr_getattr_l__mod___stages___2___blocks___7___res_scale1 + mul_38;  getattr_getattr_l__mod___stages___2___blocks___7___res_scale1 = mul_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___7___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___7___res_scale2(x_164)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___7___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___7___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___7___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___7___norm2_bias
    group_norm_39 = torch.nn.functional.group_norm(x_164, 1, getattr_getattr_l__mod___stages___2___blocks___7___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___7___norm2_bias, 1e-05);  x_164 = getattr_getattr_l__mod___stages___2___blocks___7___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___7___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_165 = self.getattr_getattr_L__mod___stages___2___blocks___7___mlp_fc1(group_norm_39);  group_norm_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_166 = self.getattr_getattr_L__mod___stages___2___blocks___7___mlp_act(x_165);  x_165 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_167 = self.getattr_getattr_L__mod___stages___2___blocks___7___mlp_drop1(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_168 = self.getattr_getattr_L__mod___stages___2___blocks___7___mlp_norm(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_169 = self.getattr_getattr_L__mod___stages___2___blocks___7___mlp_fc2(x_168);  x_168 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_170 = self.getattr_getattr_L__mod___stages___2___blocks___7___mlp_drop2(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___7___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___7___drop_path2(x_170);  x_170 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___7___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___7___layer_scale2_scale
    view_39 = getattr_getattr_l__mod___stages___2___blocks___7___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___7___layer_scale2_scale = None
    mul_39 = getattr_getattr_l__mod___stages___2___blocks___7___drop_path2 * view_39;  getattr_getattr_l__mod___stages___2___blocks___7___drop_path2 = view_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_171 = getattr_getattr_l__mod___stages___2___blocks___7___res_scale2 + mul_39;  getattr_getattr_l__mod___stages___2___blocks___7___res_scale2 = mul_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___8___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___8___res_scale1(x_171)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___8___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___8___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___8___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___8___norm1_bias
    group_norm_40 = torch.nn.functional.group_norm(x_171, 1, getattr_getattr_l__mod___stages___2___blocks___8___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___8___norm1_bias, 1e-05);  x_171 = getattr_getattr_l__mod___stages___2___blocks___8___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___8___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_20 = self.getattr_getattr_L__mod___stages___2___blocks___8___token_mixer_pool(group_norm_40)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_20 = y_20 - group_norm_40;  y_20 = group_norm_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___8___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___8___drop_path1(sub_20);  sub_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___8___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___8___layer_scale1_scale
    view_40 = getattr_getattr_l__mod___stages___2___blocks___8___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___8___layer_scale1_scale = None
    mul_40 = getattr_getattr_l__mod___stages___2___blocks___8___drop_path1 * view_40;  getattr_getattr_l__mod___stages___2___blocks___8___drop_path1 = view_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_172 = getattr_getattr_l__mod___stages___2___blocks___8___res_scale1 + mul_40;  getattr_getattr_l__mod___stages___2___blocks___8___res_scale1 = mul_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___8___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___8___res_scale2(x_172)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___8___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___8___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___8___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___8___norm2_bias
    group_norm_41 = torch.nn.functional.group_norm(x_172, 1, getattr_getattr_l__mod___stages___2___blocks___8___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___8___norm2_bias, 1e-05);  x_172 = getattr_getattr_l__mod___stages___2___blocks___8___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___8___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_173 = self.getattr_getattr_L__mod___stages___2___blocks___8___mlp_fc1(group_norm_41);  group_norm_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_174 = self.getattr_getattr_L__mod___stages___2___blocks___8___mlp_act(x_173);  x_173 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_175 = self.getattr_getattr_L__mod___stages___2___blocks___8___mlp_drop1(x_174);  x_174 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_176 = self.getattr_getattr_L__mod___stages___2___blocks___8___mlp_norm(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_177 = self.getattr_getattr_L__mod___stages___2___blocks___8___mlp_fc2(x_176);  x_176 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_178 = self.getattr_getattr_L__mod___stages___2___blocks___8___mlp_drop2(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___8___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___8___drop_path2(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___8___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___8___layer_scale2_scale
    view_41 = getattr_getattr_l__mod___stages___2___blocks___8___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___8___layer_scale2_scale = None
    mul_41 = getattr_getattr_l__mod___stages___2___blocks___8___drop_path2 * view_41;  getattr_getattr_l__mod___stages___2___blocks___8___drop_path2 = view_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_179 = getattr_getattr_l__mod___stages___2___blocks___8___res_scale2 + mul_41;  getattr_getattr_l__mod___stages___2___blocks___8___res_scale2 = mul_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___9___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___9___res_scale1(x_179)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___9___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___9___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___9___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___9___norm1_bias
    group_norm_42 = torch.nn.functional.group_norm(x_179, 1, getattr_getattr_l__mod___stages___2___blocks___9___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___9___norm1_bias, 1e-05);  x_179 = getattr_getattr_l__mod___stages___2___blocks___9___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___9___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_21 = self.getattr_getattr_L__mod___stages___2___blocks___9___token_mixer_pool(group_norm_42)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_21 = y_21 - group_norm_42;  y_21 = group_norm_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___9___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___9___drop_path1(sub_21);  sub_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___9___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___9___layer_scale1_scale
    view_42 = getattr_getattr_l__mod___stages___2___blocks___9___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___9___layer_scale1_scale = None
    mul_42 = getattr_getattr_l__mod___stages___2___blocks___9___drop_path1 * view_42;  getattr_getattr_l__mod___stages___2___blocks___9___drop_path1 = view_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_180 = getattr_getattr_l__mod___stages___2___blocks___9___res_scale1 + mul_42;  getattr_getattr_l__mod___stages___2___blocks___9___res_scale1 = mul_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___9___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___9___res_scale2(x_180)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___9___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___9___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___9___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___9___norm2_bias
    group_norm_43 = torch.nn.functional.group_norm(x_180, 1, getattr_getattr_l__mod___stages___2___blocks___9___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___9___norm2_bias, 1e-05);  x_180 = getattr_getattr_l__mod___stages___2___blocks___9___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___9___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_181 = self.getattr_getattr_L__mod___stages___2___blocks___9___mlp_fc1(group_norm_43);  group_norm_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_182 = self.getattr_getattr_L__mod___stages___2___blocks___9___mlp_act(x_181);  x_181 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_183 = self.getattr_getattr_L__mod___stages___2___blocks___9___mlp_drop1(x_182);  x_182 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_184 = self.getattr_getattr_L__mod___stages___2___blocks___9___mlp_norm(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_185 = self.getattr_getattr_L__mod___stages___2___blocks___9___mlp_fc2(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_186 = self.getattr_getattr_L__mod___stages___2___blocks___9___mlp_drop2(x_185);  x_185 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___9___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___9___drop_path2(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___9___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___9___layer_scale2_scale
    view_43 = getattr_getattr_l__mod___stages___2___blocks___9___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___9___layer_scale2_scale = None
    mul_43 = getattr_getattr_l__mod___stages___2___blocks___9___drop_path2 * view_43;  getattr_getattr_l__mod___stages___2___blocks___9___drop_path2 = view_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_187 = getattr_getattr_l__mod___stages___2___blocks___9___res_scale2 + mul_43;  getattr_getattr_l__mod___stages___2___blocks___9___res_scale2 = mul_43 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___10___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___10___res_scale1(x_187)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___10___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___10___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___10___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___10___norm1_bias
    group_norm_44 = torch.nn.functional.group_norm(x_187, 1, getattr_getattr_l__mod___stages___2___blocks___10___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___10___norm1_bias, 1e-05);  x_187 = getattr_getattr_l__mod___stages___2___blocks___10___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___10___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_22 = self.getattr_getattr_L__mod___stages___2___blocks___10___token_mixer_pool(group_norm_44)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_22 = y_22 - group_norm_44;  y_22 = group_norm_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___10___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___10___drop_path1(sub_22);  sub_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___10___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___10___layer_scale1_scale
    view_44 = getattr_getattr_l__mod___stages___2___blocks___10___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___10___layer_scale1_scale = None
    mul_44 = getattr_getattr_l__mod___stages___2___blocks___10___drop_path1 * view_44;  getattr_getattr_l__mod___stages___2___blocks___10___drop_path1 = view_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_188 = getattr_getattr_l__mod___stages___2___blocks___10___res_scale1 + mul_44;  getattr_getattr_l__mod___stages___2___blocks___10___res_scale1 = mul_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___10___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___10___res_scale2(x_188)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___10___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___10___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___10___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___10___norm2_bias
    group_norm_45 = torch.nn.functional.group_norm(x_188, 1, getattr_getattr_l__mod___stages___2___blocks___10___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___10___norm2_bias, 1e-05);  x_188 = getattr_getattr_l__mod___stages___2___blocks___10___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___10___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_189 = self.getattr_getattr_L__mod___stages___2___blocks___10___mlp_fc1(group_norm_45);  group_norm_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_190 = self.getattr_getattr_L__mod___stages___2___blocks___10___mlp_act(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_191 = self.getattr_getattr_L__mod___stages___2___blocks___10___mlp_drop1(x_190);  x_190 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_192 = self.getattr_getattr_L__mod___stages___2___blocks___10___mlp_norm(x_191);  x_191 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_193 = self.getattr_getattr_L__mod___stages___2___blocks___10___mlp_fc2(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_194 = self.getattr_getattr_L__mod___stages___2___blocks___10___mlp_drop2(x_193);  x_193 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___10___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___10___drop_path2(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___10___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___10___layer_scale2_scale
    view_45 = getattr_getattr_l__mod___stages___2___blocks___10___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___10___layer_scale2_scale = None
    mul_45 = getattr_getattr_l__mod___stages___2___blocks___10___drop_path2 * view_45;  getattr_getattr_l__mod___stages___2___blocks___10___drop_path2 = view_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_195 = getattr_getattr_l__mod___stages___2___blocks___10___res_scale2 + mul_45;  getattr_getattr_l__mod___stages___2___blocks___10___res_scale2 = mul_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___11___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___11___res_scale1(x_195)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___11___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___11___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___11___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___11___norm1_bias
    group_norm_46 = torch.nn.functional.group_norm(x_195, 1, getattr_getattr_l__mod___stages___2___blocks___11___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___11___norm1_bias, 1e-05);  x_195 = getattr_getattr_l__mod___stages___2___blocks___11___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___11___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_23 = self.getattr_getattr_L__mod___stages___2___blocks___11___token_mixer_pool(group_norm_46)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_23 = y_23 - group_norm_46;  y_23 = group_norm_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___11___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___11___drop_path1(sub_23);  sub_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___11___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___11___layer_scale1_scale
    view_46 = getattr_getattr_l__mod___stages___2___blocks___11___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___11___layer_scale1_scale = None
    mul_46 = getattr_getattr_l__mod___stages___2___blocks___11___drop_path1 * view_46;  getattr_getattr_l__mod___stages___2___blocks___11___drop_path1 = view_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_196 = getattr_getattr_l__mod___stages___2___blocks___11___res_scale1 + mul_46;  getattr_getattr_l__mod___stages___2___blocks___11___res_scale1 = mul_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___11___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___11___res_scale2(x_196)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___11___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___11___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___11___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___11___norm2_bias
    group_norm_47 = torch.nn.functional.group_norm(x_196, 1, getattr_getattr_l__mod___stages___2___blocks___11___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___11___norm2_bias, 1e-05);  x_196 = getattr_getattr_l__mod___stages___2___blocks___11___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___11___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_197 = self.getattr_getattr_L__mod___stages___2___blocks___11___mlp_fc1(group_norm_47);  group_norm_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_198 = self.getattr_getattr_L__mod___stages___2___blocks___11___mlp_act(x_197);  x_197 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_199 = self.getattr_getattr_L__mod___stages___2___blocks___11___mlp_drop1(x_198);  x_198 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_200 = self.getattr_getattr_L__mod___stages___2___blocks___11___mlp_norm(x_199);  x_199 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_201 = self.getattr_getattr_L__mod___stages___2___blocks___11___mlp_fc2(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_202 = self.getattr_getattr_L__mod___stages___2___blocks___11___mlp_drop2(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___11___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___11___drop_path2(x_202);  x_202 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___11___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___11___layer_scale2_scale
    view_47 = getattr_getattr_l__mod___stages___2___blocks___11___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___11___layer_scale2_scale = None
    mul_47 = getattr_getattr_l__mod___stages___2___blocks___11___drop_path2 * view_47;  getattr_getattr_l__mod___stages___2___blocks___11___drop_path2 = view_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_203 = getattr_getattr_l__mod___stages___2___blocks___11___res_scale2 + mul_47;  getattr_getattr_l__mod___stages___2___blocks___11___res_scale2 = mul_47 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___12___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___12___res_scale1(x_203)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___12___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___12___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___12___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___12___norm1_bias
    group_norm_48 = torch.nn.functional.group_norm(x_203, 1, getattr_getattr_l__mod___stages___2___blocks___12___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___12___norm1_bias, 1e-05);  x_203 = getattr_getattr_l__mod___stages___2___blocks___12___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___12___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_24 = self.getattr_getattr_L__mod___stages___2___blocks___12___token_mixer_pool(group_norm_48)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_24 = y_24 - group_norm_48;  y_24 = group_norm_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___12___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___12___drop_path1(sub_24);  sub_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___12___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___12___layer_scale1_scale
    view_48 = getattr_getattr_l__mod___stages___2___blocks___12___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___12___layer_scale1_scale = None
    mul_48 = getattr_getattr_l__mod___stages___2___blocks___12___drop_path1 * view_48;  getattr_getattr_l__mod___stages___2___blocks___12___drop_path1 = view_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_204 = getattr_getattr_l__mod___stages___2___blocks___12___res_scale1 + mul_48;  getattr_getattr_l__mod___stages___2___blocks___12___res_scale1 = mul_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___12___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___12___res_scale2(x_204)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___12___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___12___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___12___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___12___norm2_bias
    group_norm_49 = torch.nn.functional.group_norm(x_204, 1, getattr_getattr_l__mod___stages___2___blocks___12___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___12___norm2_bias, 1e-05);  x_204 = getattr_getattr_l__mod___stages___2___blocks___12___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___12___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_205 = self.getattr_getattr_L__mod___stages___2___blocks___12___mlp_fc1(group_norm_49);  group_norm_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_206 = self.getattr_getattr_L__mod___stages___2___blocks___12___mlp_act(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_207 = self.getattr_getattr_L__mod___stages___2___blocks___12___mlp_drop1(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_208 = self.getattr_getattr_L__mod___stages___2___blocks___12___mlp_norm(x_207);  x_207 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_209 = self.getattr_getattr_L__mod___stages___2___blocks___12___mlp_fc2(x_208);  x_208 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_210 = self.getattr_getattr_L__mod___stages___2___blocks___12___mlp_drop2(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___12___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___12___drop_path2(x_210);  x_210 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___12___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___12___layer_scale2_scale
    view_49 = getattr_getattr_l__mod___stages___2___blocks___12___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___12___layer_scale2_scale = None
    mul_49 = getattr_getattr_l__mod___stages___2___blocks___12___drop_path2 * view_49;  getattr_getattr_l__mod___stages___2___blocks___12___drop_path2 = view_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_211 = getattr_getattr_l__mod___stages___2___blocks___12___res_scale2 + mul_49;  getattr_getattr_l__mod___stages___2___blocks___12___res_scale2 = mul_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___13___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___13___res_scale1(x_211)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___13___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___13___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___13___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___13___norm1_bias
    group_norm_50 = torch.nn.functional.group_norm(x_211, 1, getattr_getattr_l__mod___stages___2___blocks___13___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___13___norm1_bias, 1e-05);  x_211 = getattr_getattr_l__mod___stages___2___blocks___13___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___13___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_25 = self.getattr_getattr_L__mod___stages___2___blocks___13___token_mixer_pool(group_norm_50)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_25 = y_25 - group_norm_50;  y_25 = group_norm_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___13___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___13___drop_path1(sub_25);  sub_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___13___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___13___layer_scale1_scale
    view_50 = getattr_getattr_l__mod___stages___2___blocks___13___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___13___layer_scale1_scale = None
    mul_50 = getattr_getattr_l__mod___stages___2___blocks___13___drop_path1 * view_50;  getattr_getattr_l__mod___stages___2___blocks___13___drop_path1 = view_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_212 = getattr_getattr_l__mod___stages___2___blocks___13___res_scale1 + mul_50;  getattr_getattr_l__mod___stages___2___blocks___13___res_scale1 = mul_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___13___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___13___res_scale2(x_212)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___13___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___13___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___13___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___13___norm2_bias
    group_norm_51 = torch.nn.functional.group_norm(x_212, 1, getattr_getattr_l__mod___stages___2___blocks___13___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___13___norm2_bias, 1e-05);  x_212 = getattr_getattr_l__mod___stages___2___blocks___13___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___13___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_213 = self.getattr_getattr_L__mod___stages___2___blocks___13___mlp_fc1(group_norm_51);  group_norm_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_214 = self.getattr_getattr_L__mod___stages___2___blocks___13___mlp_act(x_213);  x_213 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_215 = self.getattr_getattr_L__mod___stages___2___blocks___13___mlp_drop1(x_214);  x_214 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_216 = self.getattr_getattr_L__mod___stages___2___blocks___13___mlp_norm(x_215);  x_215 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_217 = self.getattr_getattr_L__mod___stages___2___blocks___13___mlp_fc2(x_216);  x_216 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_218 = self.getattr_getattr_L__mod___stages___2___blocks___13___mlp_drop2(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___13___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___13___drop_path2(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___13___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___13___layer_scale2_scale
    view_51 = getattr_getattr_l__mod___stages___2___blocks___13___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___13___layer_scale2_scale = None
    mul_51 = getattr_getattr_l__mod___stages___2___blocks___13___drop_path2 * view_51;  getattr_getattr_l__mod___stages___2___blocks___13___drop_path2 = view_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_219 = getattr_getattr_l__mod___stages___2___blocks___13___res_scale2 + mul_51;  getattr_getattr_l__mod___stages___2___blocks___13___res_scale2 = mul_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___14___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___14___res_scale1(x_219)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___14___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___14___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___14___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___14___norm1_bias
    group_norm_52 = torch.nn.functional.group_norm(x_219, 1, getattr_getattr_l__mod___stages___2___blocks___14___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___14___norm1_bias, 1e-05);  x_219 = getattr_getattr_l__mod___stages___2___blocks___14___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___14___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_26 = self.getattr_getattr_L__mod___stages___2___blocks___14___token_mixer_pool(group_norm_52)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_26 = y_26 - group_norm_52;  y_26 = group_norm_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___14___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___14___drop_path1(sub_26);  sub_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___14___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___14___layer_scale1_scale
    view_52 = getattr_getattr_l__mod___stages___2___blocks___14___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___14___layer_scale1_scale = None
    mul_52 = getattr_getattr_l__mod___stages___2___blocks___14___drop_path1 * view_52;  getattr_getattr_l__mod___stages___2___blocks___14___drop_path1 = view_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_220 = getattr_getattr_l__mod___stages___2___blocks___14___res_scale1 + mul_52;  getattr_getattr_l__mod___stages___2___blocks___14___res_scale1 = mul_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___14___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___14___res_scale2(x_220)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___14___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___14___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___14___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___14___norm2_bias
    group_norm_53 = torch.nn.functional.group_norm(x_220, 1, getattr_getattr_l__mod___stages___2___blocks___14___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___14___norm2_bias, 1e-05);  x_220 = getattr_getattr_l__mod___stages___2___blocks___14___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___14___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_221 = self.getattr_getattr_L__mod___stages___2___blocks___14___mlp_fc1(group_norm_53);  group_norm_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_222 = self.getattr_getattr_L__mod___stages___2___blocks___14___mlp_act(x_221);  x_221 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_223 = self.getattr_getattr_L__mod___stages___2___blocks___14___mlp_drop1(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_224 = self.getattr_getattr_L__mod___stages___2___blocks___14___mlp_norm(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_225 = self.getattr_getattr_L__mod___stages___2___blocks___14___mlp_fc2(x_224);  x_224 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_226 = self.getattr_getattr_L__mod___stages___2___blocks___14___mlp_drop2(x_225);  x_225 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___14___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___14___drop_path2(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___14___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___14___layer_scale2_scale
    view_53 = getattr_getattr_l__mod___stages___2___blocks___14___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___14___layer_scale2_scale = None
    mul_53 = getattr_getattr_l__mod___stages___2___blocks___14___drop_path2 * view_53;  getattr_getattr_l__mod___stages___2___blocks___14___drop_path2 = view_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_227 = getattr_getattr_l__mod___stages___2___blocks___14___res_scale2 + mul_53;  getattr_getattr_l__mod___stages___2___blocks___14___res_scale2 = mul_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___15___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___15___res_scale1(x_227)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___15___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___15___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___15___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___15___norm1_bias
    group_norm_54 = torch.nn.functional.group_norm(x_227, 1, getattr_getattr_l__mod___stages___2___blocks___15___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___15___norm1_bias, 1e-05);  x_227 = getattr_getattr_l__mod___stages___2___blocks___15___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___15___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_27 = self.getattr_getattr_L__mod___stages___2___blocks___15___token_mixer_pool(group_norm_54)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_27 = y_27 - group_norm_54;  y_27 = group_norm_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___15___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___15___drop_path1(sub_27);  sub_27 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___15___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___15___layer_scale1_scale
    view_54 = getattr_getattr_l__mod___stages___2___blocks___15___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___15___layer_scale1_scale = None
    mul_54 = getattr_getattr_l__mod___stages___2___blocks___15___drop_path1 * view_54;  getattr_getattr_l__mod___stages___2___blocks___15___drop_path1 = view_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_228 = getattr_getattr_l__mod___stages___2___blocks___15___res_scale1 + mul_54;  getattr_getattr_l__mod___stages___2___blocks___15___res_scale1 = mul_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___15___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___15___res_scale2(x_228)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___15___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___15___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___15___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___15___norm2_bias
    group_norm_55 = torch.nn.functional.group_norm(x_228, 1, getattr_getattr_l__mod___stages___2___blocks___15___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___15___norm2_bias, 1e-05);  x_228 = getattr_getattr_l__mod___stages___2___blocks___15___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___15___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_229 = self.getattr_getattr_L__mod___stages___2___blocks___15___mlp_fc1(group_norm_55);  group_norm_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_230 = self.getattr_getattr_L__mod___stages___2___blocks___15___mlp_act(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_231 = self.getattr_getattr_L__mod___stages___2___blocks___15___mlp_drop1(x_230);  x_230 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_232 = self.getattr_getattr_L__mod___stages___2___blocks___15___mlp_norm(x_231);  x_231 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_233 = self.getattr_getattr_L__mod___stages___2___blocks___15___mlp_fc2(x_232);  x_232 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_234 = self.getattr_getattr_L__mod___stages___2___blocks___15___mlp_drop2(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___15___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___15___drop_path2(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___15___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___15___layer_scale2_scale
    view_55 = getattr_getattr_l__mod___stages___2___blocks___15___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___15___layer_scale2_scale = None
    mul_55 = getattr_getattr_l__mod___stages___2___blocks___15___drop_path2 * view_55;  getattr_getattr_l__mod___stages___2___blocks___15___drop_path2 = view_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_235 = getattr_getattr_l__mod___stages___2___blocks___15___res_scale2 + mul_55;  getattr_getattr_l__mod___stages___2___blocks___15___res_scale2 = mul_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___16___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___16___res_scale1(x_235)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___16___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___16___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___16___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___16___norm1_bias
    group_norm_56 = torch.nn.functional.group_norm(x_235, 1, getattr_getattr_l__mod___stages___2___blocks___16___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___16___norm1_bias, 1e-05);  x_235 = getattr_getattr_l__mod___stages___2___blocks___16___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___16___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_28 = self.getattr_getattr_L__mod___stages___2___blocks___16___token_mixer_pool(group_norm_56)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_28 = y_28 - group_norm_56;  y_28 = group_norm_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___16___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___16___drop_path1(sub_28);  sub_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___16___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___16___layer_scale1_scale
    view_56 = getattr_getattr_l__mod___stages___2___blocks___16___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___16___layer_scale1_scale = None
    mul_56 = getattr_getattr_l__mod___stages___2___blocks___16___drop_path1 * view_56;  getattr_getattr_l__mod___stages___2___blocks___16___drop_path1 = view_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_236 = getattr_getattr_l__mod___stages___2___blocks___16___res_scale1 + mul_56;  getattr_getattr_l__mod___stages___2___blocks___16___res_scale1 = mul_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___16___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___16___res_scale2(x_236)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___16___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___16___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___16___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___16___norm2_bias
    group_norm_57 = torch.nn.functional.group_norm(x_236, 1, getattr_getattr_l__mod___stages___2___blocks___16___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___16___norm2_bias, 1e-05);  x_236 = getattr_getattr_l__mod___stages___2___blocks___16___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___16___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_237 = self.getattr_getattr_L__mod___stages___2___blocks___16___mlp_fc1(group_norm_57);  group_norm_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_238 = self.getattr_getattr_L__mod___stages___2___blocks___16___mlp_act(x_237);  x_237 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_239 = self.getattr_getattr_L__mod___stages___2___blocks___16___mlp_drop1(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_240 = self.getattr_getattr_L__mod___stages___2___blocks___16___mlp_norm(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_241 = self.getattr_getattr_L__mod___stages___2___blocks___16___mlp_fc2(x_240);  x_240 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_242 = self.getattr_getattr_L__mod___stages___2___blocks___16___mlp_drop2(x_241);  x_241 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___16___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___16___drop_path2(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___16___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___16___layer_scale2_scale
    view_57 = getattr_getattr_l__mod___stages___2___blocks___16___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___16___layer_scale2_scale = None
    mul_57 = getattr_getattr_l__mod___stages___2___blocks___16___drop_path2 * view_57;  getattr_getattr_l__mod___stages___2___blocks___16___drop_path2 = view_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_243 = getattr_getattr_l__mod___stages___2___blocks___16___res_scale2 + mul_57;  getattr_getattr_l__mod___stages___2___blocks___16___res_scale2 = mul_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___2___blocks___17___res_scale1 = self.getattr_getattr_L__mod___stages___2___blocks___17___res_scale1(x_243)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___17___norm1_weight = self.getattr_getattr_L__mod___stages___2___blocks___17___norm1_weight
    getattr_getattr_l__mod___stages___2___blocks___17___norm1_bias = self.getattr_getattr_L__mod___stages___2___blocks___17___norm1_bias
    group_norm_58 = torch.nn.functional.group_norm(x_243, 1, getattr_getattr_l__mod___stages___2___blocks___17___norm1_weight, getattr_getattr_l__mod___stages___2___blocks___17___norm1_bias, 1e-05);  x_243 = getattr_getattr_l__mod___stages___2___blocks___17___norm1_weight = getattr_getattr_l__mod___stages___2___blocks___17___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_29 = self.getattr_getattr_L__mod___stages___2___blocks___17___token_mixer_pool(group_norm_58)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_29 = y_29 - group_norm_58;  y_29 = group_norm_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___2___blocks___17___drop_path1 = self.getattr_getattr_L__mod___stages___2___blocks___17___drop_path1(sub_29);  sub_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___17___layer_scale1_scale = self.getattr_getattr_L__mod___stages___2___blocks___17___layer_scale1_scale
    view_58 = getattr_getattr_l__mod___stages___2___blocks___17___layer_scale1_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___17___layer_scale1_scale = None
    mul_58 = getattr_getattr_l__mod___stages___2___blocks___17___drop_path1 * view_58;  getattr_getattr_l__mod___stages___2___blocks___17___drop_path1 = view_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_244 = getattr_getattr_l__mod___stages___2___blocks___17___res_scale1 + mul_58;  getattr_getattr_l__mod___stages___2___blocks___17___res_scale1 = mul_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___2___blocks___17___res_scale2 = self.getattr_getattr_L__mod___stages___2___blocks___17___res_scale2(x_244)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___2___blocks___17___norm2_weight = self.getattr_getattr_L__mod___stages___2___blocks___17___norm2_weight
    getattr_getattr_l__mod___stages___2___blocks___17___norm2_bias = self.getattr_getattr_L__mod___stages___2___blocks___17___norm2_bias
    group_norm_59 = torch.nn.functional.group_norm(x_244, 1, getattr_getattr_l__mod___stages___2___blocks___17___norm2_weight, getattr_getattr_l__mod___stages___2___blocks___17___norm2_bias, 1e-05);  x_244 = getattr_getattr_l__mod___stages___2___blocks___17___norm2_weight = getattr_getattr_l__mod___stages___2___blocks___17___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_245 = self.getattr_getattr_L__mod___stages___2___blocks___17___mlp_fc1(group_norm_59);  group_norm_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_246 = self.getattr_getattr_L__mod___stages___2___blocks___17___mlp_act(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_247 = self.getattr_getattr_L__mod___stages___2___blocks___17___mlp_drop1(x_246);  x_246 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_248 = self.getattr_getattr_L__mod___stages___2___blocks___17___mlp_norm(x_247);  x_247 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_249 = self.getattr_getattr_L__mod___stages___2___blocks___17___mlp_fc2(x_248);  x_248 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_250 = self.getattr_getattr_L__mod___stages___2___blocks___17___mlp_drop2(x_249);  x_249 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___2___blocks___17___drop_path2 = self.getattr_getattr_L__mod___stages___2___blocks___17___drop_path2(x_250);  x_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___2___blocks___17___layer_scale2_scale = self.getattr_getattr_L__mod___stages___2___blocks___17___layer_scale2_scale
    view_59 = getattr_getattr_l__mod___stages___2___blocks___17___layer_scale2_scale.view((384, 1, 1));  getattr_getattr_l__mod___stages___2___blocks___17___layer_scale2_scale = None
    mul_59 = getattr_getattr_l__mod___stages___2___blocks___17___drop_path2 * view_59;  getattr_getattr_l__mod___stages___2___blocks___17___drop_path2 = view_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_252 = getattr_getattr_l__mod___stages___2___blocks___17___res_scale2 + mul_59;  getattr_getattr_l__mod___stages___2___blocks___17___res_scale2 = mul_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:102, code: x = self.norm(x)
    x_253 = self.getattr_L__mod___stages___3___downsample_norm(x_252);  x_252 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:103, code: x = self.conv(x)
    x_255 = self.getattr_L__mod___stages___3___downsample_conv(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___3___blocks___0___res_scale1 = self.getattr_getattr_L__mod___stages___3___blocks___0___res_scale1(x_255)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___0___norm1_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___norm1_weight
    getattr_getattr_l__mod___stages___3___blocks___0___norm1_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___norm1_bias
    group_norm_60 = torch.nn.functional.group_norm(x_255, 1, getattr_getattr_l__mod___stages___3___blocks___0___norm1_weight, getattr_getattr_l__mod___stages___3___blocks___0___norm1_bias, 1e-05);  x_255 = getattr_getattr_l__mod___stages___3___blocks___0___norm1_weight = getattr_getattr_l__mod___stages___3___blocks___0___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_30 = self.getattr_getattr_L__mod___stages___3___blocks___0___token_mixer_pool(group_norm_60)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_30 = y_30 - group_norm_60;  y_30 = group_norm_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___3___blocks___0___drop_path1 = self.getattr_getattr_L__mod___stages___3___blocks___0___drop_path1(sub_30);  sub_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___3___blocks___0___layer_scale1_scale = self.getattr_getattr_L__mod___stages___3___blocks___0___layer_scale1_scale
    view_60 = getattr_getattr_l__mod___stages___3___blocks___0___layer_scale1_scale.view((768, 1, 1));  getattr_getattr_l__mod___stages___3___blocks___0___layer_scale1_scale = None
    mul_60 = getattr_getattr_l__mod___stages___3___blocks___0___drop_path1 * view_60;  getattr_getattr_l__mod___stages___3___blocks___0___drop_path1 = view_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_256 = getattr_getattr_l__mod___stages___3___blocks___0___res_scale1 + mul_60;  getattr_getattr_l__mod___stages___3___blocks___0___res_scale1 = mul_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___3___blocks___0___res_scale2 = self.getattr_getattr_L__mod___stages___3___blocks___0___res_scale2(x_256)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___0___norm2_weight = self.getattr_getattr_L__mod___stages___3___blocks___0___norm2_weight
    getattr_getattr_l__mod___stages___3___blocks___0___norm2_bias = self.getattr_getattr_L__mod___stages___3___blocks___0___norm2_bias
    group_norm_61 = torch.nn.functional.group_norm(x_256, 1, getattr_getattr_l__mod___stages___3___blocks___0___norm2_weight, getattr_getattr_l__mod___stages___3___blocks___0___norm2_bias, 1e-05);  x_256 = getattr_getattr_l__mod___stages___3___blocks___0___norm2_weight = getattr_getattr_l__mod___stages___3___blocks___0___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_257 = self.getattr_getattr_L__mod___stages___3___blocks___0___mlp_fc1(group_norm_61);  group_norm_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_258 = self.getattr_getattr_L__mod___stages___3___blocks___0___mlp_act(x_257);  x_257 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_259 = self.getattr_getattr_L__mod___stages___3___blocks___0___mlp_drop1(x_258);  x_258 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_260 = self.getattr_getattr_L__mod___stages___3___blocks___0___mlp_norm(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_261 = self.getattr_getattr_L__mod___stages___3___blocks___0___mlp_fc2(x_260);  x_260 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_262 = self.getattr_getattr_L__mod___stages___3___blocks___0___mlp_drop2(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___3___blocks___0___drop_path2 = self.getattr_getattr_L__mod___stages___3___blocks___0___drop_path2(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___3___blocks___0___layer_scale2_scale = self.getattr_getattr_L__mod___stages___3___blocks___0___layer_scale2_scale
    view_61 = getattr_getattr_l__mod___stages___3___blocks___0___layer_scale2_scale.view((768, 1, 1));  getattr_getattr_l__mod___stages___3___blocks___0___layer_scale2_scale = None
    mul_61 = getattr_getattr_l__mod___stages___3___blocks___0___drop_path2 * view_61;  getattr_getattr_l__mod___stages___3___blocks___0___drop_path2 = view_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_263 = getattr_getattr_l__mod___stages___3___blocks___0___res_scale2 + mul_61;  getattr_getattr_l__mod___stages___3___blocks___0___res_scale2 = mul_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___3___blocks___1___res_scale1 = self.getattr_getattr_L__mod___stages___3___blocks___1___res_scale1(x_263)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___1___norm1_weight = self.getattr_getattr_L__mod___stages___3___blocks___1___norm1_weight
    getattr_getattr_l__mod___stages___3___blocks___1___norm1_bias = self.getattr_getattr_L__mod___stages___3___blocks___1___norm1_bias
    group_norm_62 = torch.nn.functional.group_norm(x_263, 1, getattr_getattr_l__mod___stages___3___blocks___1___norm1_weight, getattr_getattr_l__mod___stages___3___blocks___1___norm1_bias, 1e-05);  x_263 = getattr_getattr_l__mod___stages___3___blocks___1___norm1_weight = getattr_getattr_l__mod___stages___3___blocks___1___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_31 = self.getattr_getattr_L__mod___stages___3___blocks___1___token_mixer_pool(group_norm_62)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_31 = y_31 - group_norm_62;  y_31 = group_norm_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___3___blocks___1___drop_path1 = self.getattr_getattr_L__mod___stages___3___blocks___1___drop_path1(sub_31);  sub_31 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___3___blocks___1___layer_scale1_scale = self.getattr_getattr_L__mod___stages___3___blocks___1___layer_scale1_scale
    view_62 = getattr_getattr_l__mod___stages___3___blocks___1___layer_scale1_scale.view((768, 1, 1));  getattr_getattr_l__mod___stages___3___blocks___1___layer_scale1_scale = None
    mul_62 = getattr_getattr_l__mod___stages___3___blocks___1___drop_path1 * view_62;  getattr_getattr_l__mod___stages___3___blocks___1___drop_path1 = view_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_264 = getattr_getattr_l__mod___stages___3___blocks___1___res_scale1 + mul_62;  getattr_getattr_l__mod___stages___3___blocks___1___res_scale1 = mul_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___3___blocks___1___res_scale2 = self.getattr_getattr_L__mod___stages___3___blocks___1___res_scale2(x_264)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___1___norm2_weight = self.getattr_getattr_L__mod___stages___3___blocks___1___norm2_weight
    getattr_getattr_l__mod___stages___3___blocks___1___norm2_bias = self.getattr_getattr_L__mod___stages___3___blocks___1___norm2_bias
    group_norm_63 = torch.nn.functional.group_norm(x_264, 1, getattr_getattr_l__mod___stages___3___blocks___1___norm2_weight, getattr_getattr_l__mod___stages___3___blocks___1___norm2_bias, 1e-05);  x_264 = getattr_getattr_l__mod___stages___3___blocks___1___norm2_weight = getattr_getattr_l__mod___stages___3___blocks___1___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_265 = self.getattr_getattr_L__mod___stages___3___blocks___1___mlp_fc1(group_norm_63);  group_norm_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_266 = self.getattr_getattr_L__mod___stages___3___blocks___1___mlp_act(x_265);  x_265 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_267 = self.getattr_getattr_L__mod___stages___3___blocks___1___mlp_drop1(x_266);  x_266 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_268 = self.getattr_getattr_L__mod___stages___3___blocks___1___mlp_norm(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_269 = self.getattr_getattr_L__mod___stages___3___blocks___1___mlp_fc2(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_270 = self.getattr_getattr_L__mod___stages___3___blocks___1___mlp_drop2(x_269);  x_269 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___3___blocks___1___drop_path2 = self.getattr_getattr_L__mod___stages___3___blocks___1___drop_path2(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___3___blocks___1___layer_scale2_scale = self.getattr_getattr_L__mod___stages___3___blocks___1___layer_scale2_scale
    view_63 = getattr_getattr_l__mod___stages___3___blocks___1___layer_scale2_scale.view((768, 1, 1));  getattr_getattr_l__mod___stages___3___blocks___1___layer_scale2_scale = None
    mul_63 = getattr_getattr_l__mod___stages___3___blocks___1___drop_path2 * view_63;  getattr_getattr_l__mod___stages___3___blocks___1___drop_path2 = view_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_271 = getattr_getattr_l__mod___stages___3___blocks___1___res_scale2 + mul_63;  getattr_getattr_l__mod___stages___3___blocks___1___res_scale2 = mul_63 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___3___blocks___2___res_scale1 = self.getattr_getattr_L__mod___stages___3___blocks___2___res_scale1(x_271)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___2___norm1_weight = self.getattr_getattr_L__mod___stages___3___blocks___2___norm1_weight
    getattr_getattr_l__mod___stages___3___blocks___2___norm1_bias = self.getattr_getattr_L__mod___stages___3___blocks___2___norm1_bias
    group_norm_64 = torch.nn.functional.group_norm(x_271, 1, getattr_getattr_l__mod___stages___3___blocks___2___norm1_weight, getattr_getattr_l__mod___stages___3___blocks___2___norm1_bias, 1e-05);  x_271 = getattr_getattr_l__mod___stages___3___blocks___2___norm1_weight = getattr_getattr_l__mod___stages___3___blocks___2___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_32 = self.getattr_getattr_L__mod___stages___3___blocks___2___token_mixer_pool(group_norm_64)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_32 = y_32 - group_norm_64;  y_32 = group_norm_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___3___blocks___2___drop_path1 = self.getattr_getattr_L__mod___stages___3___blocks___2___drop_path1(sub_32);  sub_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___3___blocks___2___layer_scale1_scale = self.getattr_getattr_L__mod___stages___3___blocks___2___layer_scale1_scale
    view_64 = getattr_getattr_l__mod___stages___3___blocks___2___layer_scale1_scale.view((768, 1, 1));  getattr_getattr_l__mod___stages___3___blocks___2___layer_scale1_scale = None
    mul_64 = getattr_getattr_l__mod___stages___3___blocks___2___drop_path1 * view_64;  getattr_getattr_l__mod___stages___3___blocks___2___drop_path1 = view_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_272 = getattr_getattr_l__mod___stages___3___blocks___2___res_scale1 + mul_64;  getattr_getattr_l__mod___stages___3___blocks___2___res_scale1 = mul_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___3___blocks___2___res_scale2 = self.getattr_getattr_L__mod___stages___3___blocks___2___res_scale2(x_272)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___2___norm2_weight = self.getattr_getattr_L__mod___stages___3___blocks___2___norm2_weight
    getattr_getattr_l__mod___stages___3___blocks___2___norm2_bias = self.getattr_getattr_L__mod___stages___3___blocks___2___norm2_bias
    group_norm_65 = torch.nn.functional.group_norm(x_272, 1, getattr_getattr_l__mod___stages___3___blocks___2___norm2_weight, getattr_getattr_l__mod___stages___3___blocks___2___norm2_bias, 1e-05);  x_272 = getattr_getattr_l__mod___stages___3___blocks___2___norm2_weight = getattr_getattr_l__mod___stages___3___blocks___2___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_273 = self.getattr_getattr_L__mod___stages___3___blocks___2___mlp_fc1(group_norm_65);  group_norm_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_274 = self.getattr_getattr_L__mod___stages___3___blocks___2___mlp_act(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_275 = self.getattr_getattr_L__mod___stages___3___blocks___2___mlp_drop1(x_274);  x_274 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_276 = self.getattr_getattr_L__mod___stages___3___blocks___2___mlp_norm(x_275);  x_275 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_277 = self.getattr_getattr_L__mod___stages___3___blocks___2___mlp_fc2(x_276);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_278 = self.getattr_getattr_L__mod___stages___3___blocks___2___mlp_drop2(x_277);  x_277 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___3___blocks___2___drop_path2 = self.getattr_getattr_L__mod___stages___3___blocks___2___drop_path2(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___3___blocks___2___layer_scale2_scale = self.getattr_getattr_L__mod___stages___3___blocks___2___layer_scale2_scale
    view_65 = getattr_getattr_l__mod___stages___3___blocks___2___layer_scale2_scale.view((768, 1, 1));  getattr_getattr_l__mod___stages___3___blocks___2___layer_scale2_scale = None
    mul_65 = getattr_getattr_l__mod___stages___3___blocks___2___drop_path2 * view_65;  getattr_getattr_l__mod___stages___3___blocks___2___drop_path2 = view_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_279 = getattr_getattr_l__mod___stages___3___blocks___2___res_scale2 + mul_65;  getattr_getattr_l__mod___stages___3___blocks___2___res_scale2 = mul_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___3___blocks___3___res_scale1 = self.getattr_getattr_L__mod___stages___3___blocks___3___res_scale1(x_279)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___3___norm1_weight = self.getattr_getattr_L__mod___stages___3___blocks___3___norm1_weight
    getattr_getattr_l__mod___stages___3___blocks___3___norm1_bias = self.getattr_getattr_L__mod___stages___3___blocks___3___norm1_bias
    group_norm_66 = torch.nn.functional.group_norm(x_279, 1, getattr_getattr_l__mod___stages___3___blocks___3___norm1_weight, getattr_getattr_l__mod___stages___3___blocks___3___norm1_bias, 1e-05);  x_279 = getattr_getattr_l__mod___stages___3___blocks___3___norm1_weight = getattr_getattr_l__mod___stages___3___blocks___3___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_33 = self.getattr_getattr_L__mod___stages___3___blocks___3___token_mixer_pool(group_norm_66)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_33 = y_33 - group_norm_66;  y_33 = group_norm_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___3___blocks___3___drop_path1 = self.getattr_getattr_L__mod___stages___3___blocks___3___drop_path1(sub_33);  sub_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___3___blocks___3___layer_scale1_scale = self.getattr_getattr_L__mod___stages___3___blocks___3___layer_scale1_scale
    view_66 = getattr_getattr_l__mod___stages___3___blocks___3___layer_scale1_scale.view((768, 1, 1));  getattr_getattr_l__mod___stages___3___blocks___3___layer_scale1_scale = None
    mul_66 = getattr_getattr_l__mod___stages___3___blocks___3___drop_path1 * view_66;  getattr_getattr_l__mod___stages___3___blocks___3___drop_path1 = view_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_280 = getattr_getattr_l__mod___stages___3___blocks___3___res_scale1 + mul_66;  getattr_getattr_l__mod___stages___3___blocks___3___res_scale1 = mul_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___3___blocks___3___res_scale2 = self.getattr_getattr_L__mod___stages___3___blocks___3___res_scale2(x_280)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___3___norm2_weight = self.getattr_getattr_L__mod___stages___3___blocks___3___norm2_weight
    getattr_getattr_l__mod___stages___3___blocks___3___norm2_bias = self.getattr_getattr_L__mod___stages___3___blocks___3___norm2_bias
    group_norm_67 = torch.nn.functional.group_norm(x_280, 1, getattr_getattr_l__mod___stages___3___blocks___3___norm2_weight, getattr_getattr_l__mod___stages___3___blocks___3___norm2_bias, 1e-05);  x_280 = getattr_getattr_l__mod___stages___3___blocks___3___norm2_weight = getattr_getattr_l__mod___stages___3___blocks___3___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_281 = self.getattr_getattr_L__mod___stages___3___blocks___3___mlp_fc1(group_norm_67);  group_norm_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_282 = self.getattr_getattr_L__mod___stages___3___blocks___3___mlp_act(x_281);  x_281 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_283 = self.getattr_getattr_L__mod___stages___3___blocks___3___mlp_drop1(x_282);  x_282 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_284 = self.getattr_getattr_L__mod___stages___3___blocks___3___mlp_norm(x_283);  x_283 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_285 = self.getattr_getattr_L__mod___stages___3___blocks___3___mlp_fc2(x_284);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_286 = self.getattr_getattr_L__mod___stages___3___blocks___3___mlp_drop2(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___3___blocks___3___drop_path2 = self.getattr_getattr_L__mod___stages___3___blocks___3___drop_path2(x_286);  x_286 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___3___blocks___3___layer_scale2_scale = self.getattr_getattr_L__mod___stages___3___blocks___3___layer_scale2_scale
    view_67 = getattr_getattr_l__mod___stages___3___blocks___3___layer_scale2_scale.view((768, 1, 1));  getattr_getattr_l__mod___stages___3___blocks___3___layer_scale2_scale = None
    mul_67 = getattr_getattr_l__mod___stages___3___blocks___3___drop_path2 * view_67;  getattr_getattr_l__mod___stages___3___blocks___3___drop_path2 = view_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_287 = getattr_getattr_l__mod___stages___3___blocks___3___res_scale2 + mul_67;  getattr_getattr_l__mod___stages___3___blocks___3___res_scale2 = mul_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___3___blocks___4___res_scale1 = self.getattr_getattr_L__mod___stages___3___blocks___4___res_scale1(x_287)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___4___norm1_weight = self.getattr_getattr_L__mod___stages___3___blocks___4___norm1_weight
    getattr_getattr_l__mod___stages___3___blocks___4___norm1_bias = self.getattr_getattr_L__mod___stages___3___blocks___4___norm1_bias
    group_norm_68 = torch.nn.functional.group_norm(x_287, 1, getattr_getattr_l__mod___stages___3___blocks___4___norm1_weight, getattr_getattr_l__mod___stages___3___blocks___4___norm1_bias, 1e-05);  x_287 = getattr_getattr_l__mod___stages___3___blocks___4___norm1_weight = getattr_getattr_l__mod___stages___3___blocks___4___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_34 = self.getattr_getattr_L__mod___stages___3___blocks___4___token_mixer_pool(group_norm_68)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_34 = y_34 - group_norm_68;  y_34 = group_norm_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___3___blocks___4___drop_path1 = self.getattr_getattr_L__mod___stages___3___blocks___4___drop_path1(sub_34);  sub_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___3___blocks___4___layer_scale1_scale = self.getattr_getattr_L__mod___stages___3___blocks___4___layer_scale1_scale
    view_68 = getattr_getattr_l__mod___stages___3___blocks___4___layer_scale1_scale.view((768, 1, 1));  getattr_getattr_l__mod___stages___3___blocks___4___layer_scale1_scale = None
    mul_68 = getattr_getattr_l__mod___stages___3___blocks___4___drop_path1 * view_68;  getattr_getattr_l__mod___stages___3___blocks___4___drop_path1 = view_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_288 = getattr_getattr_l__mod___stages___3___blocks___4___res_scale1 + mul_68;  getattr_getattr_l__mod___stages___3___blocks___4___res_scale1 = mul_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___3___blocks___4___res_scale2 = self.getattr_getattr_L__mod___stages___3___blocks___4___res_scale2(x_288)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___4___norm2_weight = self.getattr_getattr_L__mod___stages___3___blocks___4___norm2_weight
    getattr_getattr_l__mod___stages___3___blocks___4___norm2_bias = self.getattr_getattr_L__mod___stages___3___blocks___4___norm2_bias
    group_norm_69 = torch.nn.functional.group_norm(x_288, 1, getattr_getattr_l__mod___stages___3___blocks___4___norm2_weight, getattr_getattr_l__mod___stages___3___blocks___4___norm2_bias, 1e-05);  x_288 = getattr_getattr_l__mod___stages___3___blocks___4___norm2_weight = getattr_getattr_l__mod___stages___3___blocks___4___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_289 = self.getattr_getattr_L__mod___stages___3___blocks___4___mlp_fc1(group_norm_69);  group_norm_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_290 = self.getattr_getattr_L__mod___stages___3___blocks___4___mlp_act(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_291 = self.getattr_getattr_L__mod___stages___3___blocks___4___mlp_drop1(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_292 = self.getattr_getattr_L__mod___stages___3___blocks___4___mlp_norm(x_291);  x_291 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_293 = self.getattr_getattr_L__mod___stages___3___blocks___4___mlp_fc2(x_292);  x_292 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_294 = self.getattr_getattr_L__mod___stages___3___blocks___4___mlp_drop2(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___3___blocks___4___drop_path2 = self.getattr_getattr_L__mod___stages___3___blocks___4___drop_path2(x_294);  x_294 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___3___blocks___4___layer_scale2_scale = self.getattr_getattr_L__mod___stages___3___blocks___4___layer_scale2_scale
    view_69 = getattr_getattr_l__mod___stages___3___blocks___4___layer_scale2_scale.view((768, 1, 1));  getattr_getattr_l__mod___stages___3___blocks___4___layer_scale2_scale = None
    mul_69 = getattr_getattr_l__mod___stages___3___blocks___4___drop_path2 * view_69;  getattr_getattr_l__mod___stages___3___blocks___4___drop_path2 = view_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_295 = getattr_getattr_l__mod___stages___3___blocks___4___res_scale2 + mul_69;  getattr_getattr_l__mod___stages___3___blocks___4___res_scale2 = mul_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    getattr_getattr_l__mod___stages___3___blocks___5___res_scale1 = self.getattr_getattr_L__mod___stages___3___blocks___5___res_scale1(x_295)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___5___norm1_weight = self.getattr_getattr_L__mod___stages___3___blocks___5___norm1_weight
    getattr_getattr_l__mod___stages___3___blocks___5___norm1_bias = self.getattr_getattr_L__mod___stages___3___blocks___5___norm1_bias
    group_norm_70 = torch.nn.functional.group_norm(x_295, 1, getattr_getattr_l__mod___stages___3___blocks___5___norm1_weight, getattr_getattr_l__mod___stages___3___blocks___5___norm1_bias, 1e-05);  x_295 = getattr_getattr_l__mod___stages___3___blocks___5___norm1_weight = getattr_getattr_l__mod___stages___3___blocks___5___norm1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:285, code: y = self.pool(x)
    y_35 = self.getattr_getattr_L__mod___stages___3___blocks___5___token_mixer_pool(group_norm_70)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:286, code: return y - x
    sub_35 = y_35 - group_norm_70;  y_35 = group_norm_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:365, code: self.drop_path1(
    getattr_getattr_l__mod___stages___3___blocks___5___drop_path1 = self.getattr_getattr_L__mod___stages___3___blocks___5___drop_path1(sub_35);  sub_35 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___3___blocks___5___layer_scale1_scale = self.getattr_getattr_L__mod___stages___3___blocks___5___layer_scale1_scale
    view_70 = getattr_getattr_l__mod___stages___3___blocks___5___layer_scale1_scale.view((768, 1, 1));  getattr_getattr_l__mod___stages___3___blocks___5___layer_scale1_scale = None
    mul_70 = getattr_getattr_l__mod___stages___3___blocks___5___drop_path1 * view_70;  getattr_getattr_l__mod___stages___3___blocks___5___drop_path1 = view_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:363, code: x = self.res_scale1(x) + \
    x_296 = getattr_getattr_l__mod___stages___3___blocks___5___res_scale1 + mul_70;  getattr_getattr_l__mod___stages___3___blocks___5___res_scale1 = mul_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    getattr_getattr_l__mod___stages___3___blocks___5___res_scale2 = self.getattr_getattr_L__mod___stages___3___blocks___5___res_scale2(x_296)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:43, code: return F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
    getattr_getattr_l__mod___stages___3___blocks___5___norm2_weight = self.getattr_getattr_L__mod___stages___3___blocks___5___norm2_weight
    getattr_getattr_l__mod___stages___3___blocks___5___norm2_bias = self.getattr_getattr_L__mod___stages___3___blocks___5___norm2_bias
    group_norm_71 = torch.nn.functional.group_norm(x_296, 1, getattr_getattr_l__mod___stages___3___blocks___5___norm2_weight, getattr_getattr_l__mod___stages___3___blocks___5___norm2_bias, 1e-05);  x_296 = getattr_getattr_l__mod___stages___3___blocks___5___norm2_weight = getattr_getattr_l__mod___stages___3___blocks___5___norm2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:42, code: x = self.fc1(x)
    x_297 = self.getattr_getattr_L__mod___stages___3___blocks___5___mlp_fc1(group_norm_71);  group_norm_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:43, code: x = self.act(x)
    x_298 = self.getattr_getattr_L__mod___stages___3___blocks___5___mlp_act(x_297);  x_297 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:44, code: x = self.drop1(x)
    x_299 = self.getattr_getattr_L__mod___stages___3___blocks___5___mlp_drop1(x_298);  x_298 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:45, code: x = self.norm(x)
    x_300 = self.getattr_getattr_L__mod___stages___3___blocks___5___mlp_norm(x_299);  x_299 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:46, code: x = self.fc2(x)
    x_301 = self.getattr_getattr_L__mod___stages___3___blocks___5___mlp_fc2(x_300);  x_300 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/mlp.py:47, code: x = self.drop2(x)
    x_302 = self.getattr_getattr_L__mod___stages___3___blocks___5___mlp_drop2(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:371, code: self.drop_path2(
    getattr_getattr_l__mod___stages___3___blocks___5___drop_path2 = self.getattr_getattr_L__mod___stages___3___blocks___5___drop_path2(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:118, code: return x * self.scale.view(self.shape)
    getattr_getattr_l__mod___stages___3___blocks___5___layer_scale2_scale = self.getattr_getattr_L__mod___stages___3___blocks___5___layer_scale2_scale
    view_71 = getattr_getattr_l__mod___stages___3___blocks___5___layer_scale2_scale.view((768, 1, 1));  getattr_getattr_l__mod___stages___3___blocks___5___layer_scale2_scale = None
    mul_71 = getattr_getattr_l__mod___stages___3___blocks___5___drop_path2 * view_71;  getattr_getattr_l__mod___stages___3___blocks___5___drop_path2 = view_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:369, code: x = self.res_scale2(x) + \
    x_306 = getattr_getattr_l__mod___stages___3___blocks___5___res_scale2 + mul_71;  getattr_getattr_l__mod___stages___3___blocks___5___res_scale2 = mul_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_307 = self.L__mod___head_global_pool_pool(x_306);  x_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_309 = self.L__mod___head_global_pool_flatten(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:68, code: x = x.permute(0, 2, 3, 1)
    x_310 = x_309.permute(0, 2, 3, 1);  x_309 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:72, code: x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
    l__mod___head_norm_weight = self.L__mod___head_norm_weight
    l__mod___head_norm_bias = self.L__mod___head_norm_bias
    x_311 = torch.nn.functional.layer_norm(x_310, (768,), l__mod___head_norm_weight, l__mod___head_norm_bias, 1e-06);  x_310 = l__mod___head_norm_weight = l__mod___head_norm_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm.py:73, code: x = x.permute(0, 3, 1, 2)
    x_313 = x_311.permute(0, 3, 1, 2);  x_311 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:600, code: x = self.head.flatten(x)
    x_314 = self.L__mod___head_flatten(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:601, code: x = self.head.drop(x)
    x_315 = self.L__mod___head_drop(x_314);  x_314 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/metaformer.py:602, code: return x if pre_logits else self.head.fc(x)
    x_316 = self.L__mod___head_fc(x_315);  x_315 = None
    return (x_316,)
    