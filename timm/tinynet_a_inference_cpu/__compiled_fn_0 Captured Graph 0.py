from __future__ import annotations



def forward(self, L_inputs_0_ : torch.Tensor):
    l_inputs_0_ = L_inputs_0_
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:162, code: x = self.conv_stem(x)
    x = self.L__mod___conv_stem(l_inputs_0_);  l_inputs_0_ = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___bn1_running_mean = self.L__mod___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___bn1_running_var = self.L__mod___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___bn1_weight = self.L__mod___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___bn1_bias = self.L__mod___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_1 = torch.nn.functional.batch_norm(x, l__mod___bn1_running_mean, l__mod___bn1_running_var, l__mod___bn1_weight, l__mod___bn1_bias, False, 0.1, 1e-05);  x = l__mod___bn1_running_mean = l__mod___bn1_running_var = l__mod___bn1_weight = l__mod___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_2 = self.L__mod___bn1_drop(x_1);  x_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut = self.L__mod___bn1_act(x_2);  x_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:123, code: x = self.conv_dw(x)
    x_5 = self.getattr_getattr_L__mod___blocks___0_____0___conv_dw(shortcut);  shortcut = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___0_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___0_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___0_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___0_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___0_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___0_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___0_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___0_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_6 = torch.nn.functional.batch_norm(x_5, getattr_getattr_l__mod___blocks___0_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___0_____0___bn1_running_var, getattr_getattr_l__mod___blocks___0_____0___bn1_weight, getattr_getattr_l__mod___blocks___0_____0___bn1_bias, False, 0.1, 1e-05);  x_5 = getattr_getattr_l__mod___blocks___0_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___0_____0___bn1_running_var = getattr_getattr_l__mod___blocks___0_____0___bn1_weight = getattr_getattr_l__mod___blocks___0_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_7 = self.getattr_getattr_L__mod___blocks___0_____0___bn1_drop(x_6);  x_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_9 = self.getattr_getattr_L__mod___blocks___0_____0___bn1_act(x_7);  x_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se = x_9.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_1 = self.getattr_getattr_L__mod___blocks___0_____0___se_conv_reduce(x_se);  x_se = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_2 = self.getattr_getattr_L__mod___blocks___0_____0___se_act1(x_se_1);  x_se_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_3 = self.getattr_getattr_L__mod___blocks___0_____0___se_conv_expand(x_se_2);  x_se_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___0_____0___se_gate = self.getattr_getattr_L__mod___blocks___0_____0___se_gate(x_se_3);  x_se_3 = None
    x_10 = x_9 * getattr_getattr_l__mod___blocks___0_____0___se_gate;  x_9 = getattr_getattr_l__mod___blocks___0_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:126, code: x = self.conv_pw(x)
    x_11 = self.getattr_getattr_L__mod___blocks___0_____0___conv_pw(x_10);  x_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___0_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___0_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___0_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___0_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___0_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___0_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___0_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___0_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_12 = torch.nn.functional.batch_norm(x_11, getattr_getattr_l__mod___blocks___0_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___0_____0___bn2_running_var, getattr_getattr_l__mod___blocks___0_____0___bn2_weight, getattr_getattr_l__mod___blocks___0_____0___bn2_bias, False, 0.1, 1e-05);  x_11 = getattr_getattr_l__mod___blocks___0_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___0_____0___bn2_running_var = getattr_getattr_l__mod___blocks___0_____0___bn2_weight = getattr_getattr_l__mod___blocks___0_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_13 = self.getattr_getattr_L__mod___blocks___0_____0___bn2_drop(x_12);  x_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_1 = self.getattr_getattr_L__mod___blocks___0_____0___bn2_act(x_13);  x_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_16 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pw(shortcut_1);  shortcut_1 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___1_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___1_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___1_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___1_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_17 = torch.nn.functional.batch_norm(x_16, getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn1_running_var, getattr_getattr_l__mod___blocks___1_____0___bn1_weight, getattr_getattr_l__mod___blocks___1_____0___bn1_bias, False, 0.1, 1e-05);  x_16 = getattr_getattr_l__mod___blocks___1_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn1_running_var = getattr_getattr_l__mod___blocks___1_____0___bn1_weight = getattr_getattr_l__mod___blocks___1_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_18 = self.getattr_getattr_L__mod___blocks___1_____0___bn1_drop(x_17);  x_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_20 = self.getattr_getattr_L__mod___blocks___1_____0___bn1_act(x_18);  x_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_21 = self.getattr_getattr_L__mod___blocks___1_____0___conv_dw(x_20);  x_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___1_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___1_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___1_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___1_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_22 = torch.nn.functional.batch_norm(x_21, getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn2_running_var, getattr_getattr_l__mod___blocks___1_____0___bn2_weight, getattr_getattr_l__mod___blocks___1_____0___bn2_bias, False, 0.1, 1e-05);  x_21 = getattr_getattr_l__mod___blocks___1_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn2_running_var = getattr_getattr_l__mod___blocks___1_____0___bn2_weight = getattr_getattr_l__mod___blocks___1_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_23 = self.getattr_getattr_L__mod___blocks___1_____0___bn2_drop(x_22);  x_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_25 = self.getattr_getattr_L__mod___blocks___1_____0___bn2_act(x_23);  x_23 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_4 = x_25.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_5 = self.getattr_getattr_L__mod___blocks___1_____0___se_conv_reduce(x_se_4);  x_se_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_6 = self.getattr_getattr_L__mod___blocks___1_____0___se_act1(x_se_5);  x_se_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_7 = self.getattr_getattr_L__mod___blocks___1_____0___se_conv_expand(x_se_6);  x_se_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___1_____0___se_gate = self.getattr_getattr_L__mod___blocks___1_____0___se_gate(x_se_7);  x_se_7 = None
    x_26 = x_25 * getattr_getattr_l__mod___blocks___1_____0___se_gate;  x_25 = getattr_getattr_l__mod___blocks___1_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_27 = self.getattr_getattr_L__mod___blocks___1_____0___conv_pwl(x_26);  x_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___1_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___1_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___1_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___1_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_28 = torch.nn.functional.batch_norm(x_27, getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____0___bn3_running_var, getattr_getattr_l__mod___blocks___1_____0___bn3_weight, getattr_getattr_l__mod___blocks___1_____0___bn3_bias, False, 0.1, 1e-05);  x_27 = getattr_getattr_l__mod___blocks___1_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____0___bn3_running_var = getattr_getattr_l__mod___blocks___1_____0___bn3_weight = getattr_getattr_l__mod___blocks___1_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_29 = self.getattr_getattr_L__mod___blocks___1_____0___bn3_drop(x_28);  x_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_2 = self.getattr_getattr_L__mod___blocks___1_____0___bn3_act(x_29);  x_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_32 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pw(shortcut_2)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___1_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___1_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___1_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___1_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_33 = torch.nn.functional.batch_norm(x_32, getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn1_running_var, getattr_getattr_l__mod___blocks___1_____1___bn1_weight, getattr_getattr_l__mod___blocks___1_____1___bn1_bias, False, 0.1, 1e-05);  x_32 = getattr_getattr_l__mod___blocks___1_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn1_running_var = getattr_getattr_l__mod___blocks___1_____1___bn1_weight = getattr_getattr_l__mod___blocks___1_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_34 = self.getattr_getattr_L__mod___blocks___1_____1___bn1_drop(x_33);  x_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_36 = self.getattr_getattr_L__mod___blocks___1_____1___bn1_act(x_34);  x_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_37 = self.getattr_getattr_L__mod___blocks___1_____1___conv_dw(x_36);  x_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___1_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___1_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___1_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___1_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_38 = torch.nn.functional.batch_norm(x_37, getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn2_running_var, getattr_getattr_l__mod___blocks___1_____1___bn2_weight, getattr_getattr_l__mod___blocks___1_____1___bn2_bias, False, 0.1, 1e-05);  x_37 = getattr_getattr_l__mod___blocks___1_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn2_running_var = getattr_getattr_l__mod___blocks___1_____1___bn2_weight = getattr_getattr_l__mod___blocks___1_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_39 = self.getattr_getattr_L__mod___blocks___1_____1___bn2_drop(x_38);  x_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_41 = self.getattr_getattr_L__mod___blocks___1_____1___bn2_act(x_39);  x_39 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_8 = x_41.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_9 = self.getattr_getattr_L__mod___blocks___1_____1___se_conv_reduce(x_se_8);  x_se_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_10 = self.getattr_getattr_L__mod___blocks___1_____1___se_act1(x_se_9);  x_se_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_11 = self.getattr_getattr_L__mod___blocks___1_____1___se_conv_expand(x_se_10);  x_se_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___1_____1___se_gate = self.getattr_getattr_L__mod___blocks___1_____1___se_gate(x_se_11);  x_se_11 = None
    x_42 = x_41 * getattr_getattr_l__mod___blocks___1_____1___se_gate;  x_41 = getattr_getattr_l__mod___blocks___1_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_43 = self.getattr_getattr_L__mod___blocks___1_____1___conv_pwl(x_42);  x_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___1_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___1_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___1_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___1_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___1_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___1_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___1_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_44 = torch.nn.functional.batch_norm(x_43, getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___1_____1___bn3_running_var, getattr_getattr_l__mod___blocks___1_____1___bn3_weight, getattr_getattr_l__mod___blocks___1_____1___bn3_bias, False, 0.1, 1e-05);  x_43 = getattr_getattr_l__mod___blocks___1_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___1_____1___bn3_running_var = getattr_getattr_l__mod___blocks___1_____1___bn3_weight = getattr_getattr_l__mod___blocks___1_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_45 = self.getattr_getattr_L__mod___blocks___1_____1___bn3_drop(x_44);  x_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_47 = self.getattr_getattr_L__mod___blocks___1_____1___bn3_act(x_45);  x_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___1_____1___drop_path = self.getattr_getattr_L__mod___blocks___1_____1___drop_path(x_47);  x_47 = None
    shortcut_3 = getattr_getattr_l__mod___blocks___1_____1___drop_path + shortcut_2;  getattr_getattr_l__mod___blocks___1_____1___drop_path = shortcut_2 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_49 = self.getattr_getattr_L__mod___blocks___2_____0___conv_pw(shortcut_3);  shortcut_3 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_50 = torch.nn.functional.batch_norm(x_49, getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn1_running_var, getattr_getattr_l__mod___blocks___2_____0___bn1_weight, getattr_getattr_l__mod___blocks___2_____0___bn1_bias, False, 0.1, 1e-05);  x_49 = getattr_getattr_l__mod___blocks___2_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn1_running_var = getattr_getattr_l__mod___blocks___2_____0___bn1_weight = getattr_getattr_l__mod___blocks___2_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_51 = self.getattr_getattr_L__mod___blocks___2_____0___bn1_drop(x_50);  x_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_53 = self.getattr_getattr_L__mod___blocks___2_____0___bn1_act(x_51);  x_51 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_54 = self.getattr_getattr_L__mod___blocks___2_____0___conv_dw(x_53);  x_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_55 = torch.nn.functional.batch_norm(x_54, getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn2_running_var, getattr_getattr_l__mod___blocks___2_____0___bn2_weight, getattr_getattr_l__mod___blocks___2_____0___bn2_bias, False, 0.1, 1e-05);  x_54 = getattr_getattr_l__mod___blocks___2_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn2_running_var = getattr_getattr_l__mod___blocks___2_____0___bn2_weight = getattr_getattr_l__mod___blocks___2_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_56 = self.getattr_getattr_L__mod___blocks___2_____0___bn2_drop(x_55);  x_55 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_58 = self.getattr_getattr_L__mod___blocks___2_____0___bn2_act(x_56);  x_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_12 = x_58.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_13 = self.getattr_getattr_L__mod___blocks___2_____0___se_conv_reduce(x_se_12);  x_se_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_14 = self.getattr_getattr_L__mod___blocks___2_____0___se_act1(x_se_13);  x_se_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_15 = self.getattr_getattr_L__mod___blocks___2_____0___se_conv_expand(x_se_14);  x_se_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____0___se_gate = self.getattr_getattr_L__mod___blocks___2_____0___se_gate(x_se_15);  x_se_15 = None
    x_59 = x_58 * getattr_getattr_l__mod___blocks___2_____0___se_gate;  x_58 = getattr_getattr_l__mod___blocks___2_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_60 = self.getattr_getattr_L__mod___blocks___2_____0___conv_pwl(x_59);  x_59 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_61 = torch.nn.functional.batch_norm(x_60, getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____0___bn3_running_var, getattr_getattr_l__mod___blocks___2_____0___bn3_weight, getattr_getattr_l__mod___blocks___2_____0___bn3_bias, False, 0.1, 1e-05);  x_60 = getattr_getattr_l__mod___blocks___2_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____0___bn3_running_var = getattr_getattr_l__mod___blocks___2_____0___bn3_weight = getattr_getattr_l__mod___blocks___2_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_62 = self.getattr_getattr_L__mod___blocks___2_____0___bn3_drop(x_61);  x_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_4 = self.getattr_getattr_L__mod___blocks___2_____0___bn3_act(x_62);  x_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_65 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pw(shortcut_4)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___2_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___2_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___2_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___2_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_66 = torch.nn.functional.batch_norm(x_65, getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn1_running_var, getattr_getattr_l__mod___blocks___2_____1___bn1_weight, getattr_getattr_l__mod___blocks___2_____1___bn1_bias, False, 0.1, 1e-05);  x_65 = getattr_getattr_l__mod___blocks___2_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn1_running_var = getattr_getattr_l__mod___blocks___2_____1___bn1_weight = getattr_getattr_l__mod___blocks___2_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_67 = self.getattr_getattr_L__mod___blocks___2_____1___bn1_drop(x_66);  x_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_69 = self.getattr_getattr_L__mod___blocks___2_____1___bn1_act(x_67);  x_67 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_70 = self.getattr_getattr_L__mod___blocks___2_____1___conv_dw(x_69);  x_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___2_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___2_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___2_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___2_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_71 = torch.nn.functional.batch_norm(x_70, getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn2_running_var, getattr_getattr_l__mod___blocks___2_____1___bn2_weight, getattr_getattr_l__mod___blocks___2_____1___bn2_bias, False, 0.1, 1e-05);  x_70 = getattr_getattr_l__mod___blocks___2_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn2_running_var = getattr_getattr_l__mod___blocks___2_____1___bn2_weight = getattr_getattr_l__mod___blocks___2_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_72 = self.getattr_getattr_L__mod___blocks___2_____1___bn2_drop(x_71);  x_71 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_74 = self.getattr_getattr_L__mod___blocks___2_____1___bn2_act(x_72);  x_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_16 = x_74.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_17 = self.getattr_getattr_L__mod___blocks___2_____1___se_conv_reduce(x_se_16);  x_se_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_18 = self.getattr_getattr_L__mod___blocks___2_____1___se_act1(x_se_17);  x_se_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_19 = self.getattr_getattr_L__mod___blocks___2_____1___se_conv_expand(x_se_18);  x_se_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___2_____1___se_gate = self.getattr_getattr_L__mod___blocks___2_____1___se_gate(x_se_19);  x_se_19 = None
    x_75 = x_74 * getattr_getattr_l__mod___blocks___2_____1___se_gate;  x_74 = getattr_getattr_l__mod___blocks___2_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_76 = self.getattr_getattr_L__mod___blocks___2_____1___conv_pwl(x_75);  x_75 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___2_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___2_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___2_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___2_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___2_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___2_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___2_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_77 = torch.nn.functional.batch_norm(x_76, getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___2_____1___bn3_running_var, getattr_getattr_l__mod___blocks___2_____1___bn3_weight, getattr_getattr_l__mod___blocks___2_____1___bn3_bias, False, 0.1, 1e-05);  x_76 = getattr_getattr_l__mod___blocks___2_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___2_____1___bn3_running_var = getattr_getattr_l__mod___blocks___2_____1___bn3_weight = getattr_getattr_l__mod___blocks___2_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_78 = self.getattr_getattr_L__mod___blocks___2_____1___bn3_drop(x_77);  x_77 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_80 = self.getattr_getattr_L__mod___blocks___2_____1___bn3_act(x_78);  x_78 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___2_____1___drop_path = self.getattr_getattr_L__mod___blocks___2_____1___drop_path(x_80);  x_80 = None
    shortcut_5 = getattr_getattr_l__mod___blocks___2_____1___drop_path + shortcut_4;  getattr_getattr_l__mod___blocks___2_____1___drop_path = shortcut_4 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_82 = self.getattr_getattr_L__mod___blocks___3_____0___conv_pw(shortcut_5);  shortcut_5 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_83 = torch.nn.functional.batch_norm(x_82, getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn1_running_var, getattr_getattr_l__mod___blocks___3_____0___bn1_weight, getattr_getattr_l__mod___blocks___3_____0___bn1_bias, False, 0.1, 1e-05);  x_82 = getattr_getattr_l__mod___blocks___3_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn1_running_var = getattr_getattr_l__mod___blocks___3_____0___bn1_weight = getattr_getattr_l__mod___blocks___3_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_84 = self.getattr_getattr_L__mod___blocks___3_____0___bn1_drop(x_83);  x_83 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_86 = self.getattr_getattr_L__mod___blocks___3_____0___bn1_act(x_84);  x_84 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_87 = self.getattr_getattr_L__mod___blocks___3_____0___conv_dw(x_86);  x_86 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_88 = torch.nn.functional.batch_norm(x_87, getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn2_running_var, getattr_getattr_l__mod___blocks___3_____0___bn2_weight, getattr_getattr_l__mod___blocks___3_____0___bn2_bias, False, 0.1, 1e-05);  x_87 = getattr_getattr_l__mod___blocks___3_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn2_running_var = getattr_getattr_l__mod___blocks___3_____0___bn2_weight = getattr_getattr_l__mod___blocks___3_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_89 = self.getattr_getattr_L__mod___blocks___3_____0___bn2_drop(x_88);  x_88 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_91 = self.getattr_getattr_L__mod___blocks___3_____0___bn2_act(x_89);  x_89 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_20 = x_91.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_21 = self.getattr_getattr_L__mod___blocks___3_____0___se_conv_reduce(x_se_20);  x_se_20 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_22 = self.getattr_getattr_L__mod___blocks___3_____0___se_act1(x_se_21);  x_se_21 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_23 = self.getattr_getattr_L__mod___blocks___3_____0___se_conv_expand(x_se_22);  x_se_22 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____0___se_gate = self.getattr_getattr_L__mod___blocks___3_____0___se_gate(x_se_23);  x_se_23 = None
    x_92 = x_91 * getattr_getattr_l__mod___blocks___3_____0___se_gate;  x_91 = getattr_getattr_l__mod___blocks___3_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_93 = self.getattr_getattr_L__mod___blocks___3_____0___conv_pwl(x_92);  x_92 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_94 = torch.nn.functional.batch_norm(x_93, getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____0___bn3_running_var, getattr_getattr_l__mod___blocks___3_____0___bn3_weight, getattr_getattr_l__mod___blocks___3_____0___bn3_bias, False, 0.1, 1e-05);  x_93 = getattr_getattr_l__mod___blocks___3_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____0___bn3_running_var = getattr_getattr_l__mod___blocks___3_____0___bn3_weight = getattr_getattr_l__mod___blocks___3_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_95 = self.getattr_getattr_L__mod___blocks___3_____0___bn3_drop(x_94);  x_94 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_6 = self.getattr_getattr_L__mod___blocks___3_____0___bn3_act(x_95);  x_95 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_98 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pw(shortcut_6)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_99 = torch.nn.functional.batch_norm(x_98, getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn1_running_var, getattr_getattr_l__mod___blocks___3_____1___bn1_weight, getattr_getattr_l__mod___blocks___3_____1___bn1_bias, False, 0.1, 1e-05);  x_98 = getattr_getattr_l__mod___blocks___3_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn1_running_var = getattr_getattr_l__mod___blocks___3_____1___bn1_weight = getattr_getattr_l__mod___blocks___3_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_100 = self.getattr_getattr_L__mod___blocks___3_____1___bn1_drop(x_99);  x_99 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_102 = self.getattr_getattr_L__mod___blocks___3_____1___bn1_act(x_100);  x_100 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_103 = self.getattr_getattr_L__mod___blocks___3_____1___conv_dw(x_102);  x_102 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_104 = torch.nn.functional.batch_norm(x_103, getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn2_running_var, getattr_getattr_l__mod___blocks___3_____1___bn2_weight, getattr_getattr_l__mod___blocks___3_____1___bn2_bias, False, 0.1, 1e-05);  x_103 = getattr_getattr_l__mod___blocks___3_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn2_running_var = getattr_getattr_l__mod___blocks___3_____1___bn2_weight = getattr_getattr_l__mod___blocks___3_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_105 = self.getattr_getattr_L__mod___blocks___3_____1___bn2_drop(x_104);  x_104 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_107 = self.getattr_getattr_L__mod___blocks___3_____1___bn2_act(x_105);  x_105 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_24 = x_107.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_25 = self.getattr_getattr_L__mod___blocks___3_____1___se_conv_reduce(x_se_24);  x_se_24 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_26 = self.getattr_getattr_L__mod___blocks___3_____1___se_act1(x_se_25);  x_se_25 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_27 = self.getattr_getattr_L__mod___blocks___3_____1___se_conv_expand(x_se_26);  x_se_26 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____1___se_gate = self.getattr_getattr_L__mod___blocks___3_____1___se_gate(x_se_27);  x_se_27 = None
    x_108 = x_107 * getattr_getattr_l__mod___blocks___3_____1___se_gate;  x_107 = getattr_getattr_l__mod___blocks___3_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_109 = self.getattr_getattr_L__mod___blocks___3_____1___conv_pwl(x_108);  x_108 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_110 = torch.nn.functional.batch_norm(x_109, getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____1___bn3_running_var, getattr_getattr_l__mod___blocks___3_____1___bn3_weight, getattr_getattr_l__mod___blocks___3_____1___bn3_bias, False, 0.1, 1e-05);  x_109 = getattr_getattr_l__mod___blocks___3_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____1___bn3_running_var = getattr_getattr_l__mod___blocks___3_____1___bn3_weight = getattr_getattr_l__mod___blocks___3_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_111 = self.getattr_getattr_L__mod___blocks___3_____1___bn3_drop(x_110);  x_110 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_113 = self.getattr_getattr_L__mod___blocks___3_____1___bn3_act(x_111);  x_111 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____1___drop_path = self.getattr_getattr_L__mod___blocks___3_____1___drop_path(x_113);  x_113 = None
    shortcut_7 = getattr_getattr_l__mod___blocks___3_____1___drop_path + shortcut_6;  getattr_getattr_l__mod___blocks___3_____1___drop_path = shortcut_6 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_115 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pw(shortcut_7)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_116 = torch.nn.functional.batch_norm(x_115, getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn1_running_var, getattr_getattr_l__mod___blocks___3_____2___bn1_weight, getattr_getattr_l__mod___blocks___3_____2___bn1_bias, False, 0.1, 1e-05);  x_115 = getattr_getattr_l__mod___blocks___3_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn1_running_var = getattr_getattr_l__mod___blocks___3_____2___bn1_weight = getattr_getattr_l__mod___blocks___3_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_117 = self.getattr_getattr_L__mod___blocks___3_____2___bn1_drop(x_116);  x_116 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_119 = self.getattr_getattr_L__mod___blocks___3_____2___bn1_act(x_117);  x_117 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_120 = self.getattr_getattr_L__mod___blocks___3_____2___conv_dw(x_119);  x_119 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_121 = torch.nn.functional.batch_norm(x_120, getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn2_running_var, getattr_getattr_l__mod___blocks___3_____2___bn2_weight, getattr_getattr_l__mod___blocks___3_____2___bn2_bias, False, 0.1, 1e-05);  x_120 = getattr_getattr_l__mod___blocks___3_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn2_running_var = getattr_getattr_l__mod___blocks___3_____2___bn2_weight = getattr_getattr_l__mod___blocks___3_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_122 = self.getattr_getattr_L__mod___blocks___3_____2___bn2_drop(x_121);  x_121 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_124 = self.getattr_getattr_L__mod___blocks___3_____2___bn2_act(x_122);  x_122 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_28 = x_124.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_29 = self.getattr_getattr_L__mod___blocks___3_____2___se_conv_reduce(x_se_28);  x_se_28 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_30 = self.getattr_getattr_L__mod___blocks___3_____2___se_act1(x_se_29);  x_se_29 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_31 = self.getattr_getattr_L__mod___blocks___3_____2___se_conv_expand(x_se_30);  x_se_30 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____2___se_gate = self.getattr_getattr_L__mod___blocks___3_____2___se_gate(x_se_31);  x_se_31 = None
    x_125 = x_124 * getattr_getattr_l__mod___blocks___3_____2___se_gate;  x_124 = getattr_getattr_l__mod___blocks___3_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_126 = self.getattr_getattr_L__mod___blocks___3_____2___conv_pwl(x_125);  x_125 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_127 = torch.nn.functional.batch_norm(x_126, getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____2___bn3_running_var, getattr_getattr_l__mod___blocks___3_____2___bn3_weight, getattr_getattr_l__mod___blocks___3_____2___bn3_bias, False, 0.1, 1e-05);  x_126 = getattr_getattr_l__mod___blocks___3_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____2___bn3_running_var = getattr_getattr_l__mod___blocks___3_____2___bn3_weight = getattr_getattr_l__mod___blocks___3_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_128 = self.getattr_getattr_L__mod___blocks___3_____2___bn3_drop(x_127);  x_127 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_130 = self.getattr_getattr_L__mod___blocks___3_____2___bn3_act(x_128);  x_128 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____2___drop_path = self.getattr_getattr_L__mod___blocks___3_____2___drop_path(x_130);  x_130 = None
    shortcut_8 = getattr_getattr_l__mod___blocks___3_____2___drop_path + shortcut_7;  getattr_getattr_l__mod___blocks___3_____2___drop_path = shortcut_7 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_132 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pw(shortcut_8)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___3_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___3_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___3_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___3_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_133 = torch.nn.functional.batch_norm(x_132, getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn1_running_var, getattr_getattr_l__mod___blocks___3_____3___bn1_weight, getattr_getattr_l__mod___blocks___3_____3___bn1_bias, False, 0.1, 1e-05);  x_132 = getattr_getattr_l__mod___blocks___3_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn1_running_var = getattr_getattr_l__mod___blocks___3_____3___bn1_weight = getattr_getattr_l__mod___blocks___3_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_134 = self.getattr_getattr_L__mod___blocks___3_____3___bn1_drop(x_133);  x_133 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_136 = self.getattr_getattr_L__mod___blocks___3_____3___bn1_act(x_134);  x_134 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_137 = self.getattr_getattr_L__mod___blocks___3_____3___conv_dw(x_136);  x_136 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___3_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___3_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___3_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___3_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_138 = torch.nn.functional.batch_norm(x_137, getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn2_running_var, getattr_getattr_l__mod___blocks___3_____3___bn2_weight, getattr_getattr_l__mod___blocks___3_____3___bn2_bias, False, 0.1, 1e-05);  x_137 = getattr_getattr_l__mod___blocks___3_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn2_running_var = getattr_getattr_l__mod___blocks___3_____3___bn2_weight = getattr_getattr_l__mod___blocks___3_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_139 = self.getattr_getattr_L__mod___blocks___3_____3___bn2_drop(x_138);  x_138 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_141 = self.getattr_getattr_L__mod___blocks___3_____3___bn2_act(x_139);  x_139 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_32 = x_141.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_33 = self.getattr_getattr_L__mod___blocks___3_____3___se_conv_reduce(x_se_32);  x_se_32 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_34 = self.getattr_getattr_L__mod___blocks___3_____3___se_act1(x_se_33);  x_se_33 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_35 = self.getattr_getattr_L__mod___blocks___3_____3___se_conv_expand(x_se_34);  x_se_34 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___3_____3___se_gate = self.getattr_getattr_L__mod___blocks___3_____3___se_gate(x_se_35);  x_se_35 = None
    x_142 = x_141 * getattr_getattr_l__mod___blocks___3_____3___se_gate;  x_141 = getattr_getattr_l__mod___blocks___3_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_143 = self.getattr_getattr_L__mod___blocks___3_____3___conv_pwl(x_142);  x_142 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___3_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___3_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___3_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___3_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___3_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___3_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___3_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_144 = torch.nn.functional.batch_norm(x_143, getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___3_____3___bn3_running_var, getattr_getattr_l__mod___blocks___3_____3___bn3_weight, getattr_getattr_l__mod___blocks___3_____3___bn3_bias, False, 0.1, 1e-05);  x_143 = getattr_getattr_l__mod___blocks___3_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___3_____3___bn3_running_var = getattr_getattr_l__mod___blocks___3_____3___bn3_weight = getattr_getattr_l__mod___blocks___3_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_145 = self.getattr_getattr_L__mod___blocks___3_____3___bn3_drop(x_144);  x_144 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_147 = self.getattr_getattr_L__mod___blocks___3_____3___bn3_act(x_145);  x_145 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___3_____3___drop_path = self.getattr_getattr_L__mod___blocks___3_____3___drop_path(x_147);  x_147 = None
    shortcut_9 = getattr_getattr_l__mod___blocks___3_____3___drop_path + shortcut_8;  getattr_getattr_l__mod___blocks___3_____3___drop_path = shortcut_8 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_149 = self.getattr_getattr_L__mod___blocks___4_____0___conv_pw(shortcut_9);  shortcut_9 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_150 = torch.nn.functional.batch_norm(x_149, getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn1_running_var, getattr_getattr_l__mod___blocks___4_____0___bn1_weight, getattr_getattr_l__mod___blocks___4_____0___bn1_bias, False, 0.1, 1e-05);  x_149 = getattr_getattr_l__mod___blocks___4_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn1_running_var = getattr_getattr_l__mod___blocks___4_____0___bn1_weight = getattr_getattr_l__mod___blocks___4_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_151 = self.getattr_getattr_L__mod___blocks___4_____0___bn1_drop(x_150);  x_150 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_153 = self.getattr_getattr_L__mod___blocks___4_____0___bn1_act(x_151);  x_151 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_154 = self.getattr_getattr_L__mod___blocks___4_____0___conv_dw(x_153);  x_153 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_155 = torch.nn.functional.batch_norm(x_154, getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn2_running_var, getattr_getattr_l__mod___blocks___4_____0___bn2_weight, getattr_getattr_l__mod___blocks___4_____0___bn2_bias, False, 0.1, 1e-05);  x_154 = getattr_getattr_l__mod___blocks___4_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn2_running_var = getattr_getattr_l__mod___blocks___4_____0___bn2_weight = getattr_getattr_l__mod___blocks___4_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_156 = self.getattr_getattr_L__mod___blocks___4_____0___bn2_drop(x_155);  x_155 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_158 = self.getattr_getattr_L__mod___blocks___4_____0___bn2_act(x_156);  x_156 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_36 = x_158.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_37 = self.getattr_getattr_L__mod___blocks___4_____0___se_conv_reduce(x_se_36);  x_se_36 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_38 = self.getattr_getattr_L__mod___blocks___4_____0___se_act1(x_se_37);  x_se_37 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_39 = self.getattr_getattr_L__mod___blocks___4_____0___se_conv_expand(x_se_38);  x_se_38 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____0___se_gate = self.getattr_getattr_L__mod___blocks___4_____0___se_gate(x_se_39);  x_se_39 = None
    x_159 = x_158 * getattr_getattr_l__mod___blocks___4_____0___se_gate;  x_158 = getattr_getattr_l__mod___blocks___4_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_160 = self.getattr_getattr_L__mod___blocks___4_____0___conv_pwl(x_159);  x_159 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_161 = torch.nn.functional.batch_norm(x_160, getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____0___bn3_running_var, getattr_getattr_l__mod___blocks___4_____0___bn3_weight, getattr_getattr_l__mod___blocks___4_____0___bn3_bias, False, 0.1, 1e-05);  x_160 = getattr_getattr_l__mod___blocks___4_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____0___bn3_running_var = getattr_getattr_l__mod___blocks___4_____0___bn3_weight = getattr_getattr_l__mod___blocks___4_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_162 = self.getattr_getattr_L__mod___blocks___4_____0___bn3_drop(x_161);  x_161 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_10 = self.getattr_getattr_L__mod___blocks___4_____0___bn3_act(x_162);  x_162 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_165 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pw(shortcut_10)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_166 = torch.nn.functional.batch_norm(x_165, getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn1_running_var, getattr_getattr_l__mod___blocks___4_____1___bn1_weight, getattr_getattr_l__mod___blocks___4_____1___bn1_bias, False, 0.1, 1e-05);  x_165 = getattr_getattr_l__mod___blocks___4_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn1_running_var = getattr_getattr_l__mod___blocks___4_____1___bn1_weight = getattr_getattr_l__mod___blocks___4_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_167 = self.getattr_getattr_L__mod___blocks___4_____1___bn1_drop(x_166);  x_166 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_169 = self.getattr_getattr_L__mod___blocks___4_____1___bn1_act(x_167);  x_167 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_170 = self.getattr_getattr_L__mod___blocks___4_____1___conv_dw(x_169);  x_169 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_171 = torch.nn.functional.batch_norm(x_170, getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn2_running_var, getattr_getattr_l__mod___blocks___4_____1___bn2_weight, getattr_getattr_l__mod___blocks___4_____1___bn2_bias, False, 0.1, 1e-05);  x_170 = getattr_getattr_l__mod___blocks___4_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn2_running_var = getattr_getattr_l__mod___blocks___4_____1___bn2_weight = getattr_getattr_l__mod___blocks___4_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_172 = self.getattr_getattr_L__mod___blocks___4_____1___bn2_drop(x_171);  x_171 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_174 = self.getattr_getattr_L__mod___blocks___4_____1___bn2_act(x_172);  x_172 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_40 = x_174.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_41 = self.getattr_getattr_L__mod___blocks___4_____1___se_conv_reduce(x_se_40);  x_se_40 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_42 = self.getattr_getattr_L__mod___blocks___4_____1___se_act1(x_se_41);  x_se_41 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_43 = self.getattr_getattr_L__mod___blocks___4_____1___se_conv_expand(x_se_42);  x_se_42 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____1___se_gate = self.getattr_getattr_L__mod___blocks___4_____1___se_gate(x_se_43);  x_se_43 = None
    x_175 = x_174 * getattr_getattr_l__mod___blocks___4_____1___se_gate;  x_174 = getattr_getattr_l__mod___blocks___4_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_176 = self.getattr_getattr_L__mod___blocks___4_____1___conv_pwl(x_175);  x_175 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_177 = torch.nn.functional.batch_norm(x_176, getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____1___bn3_running_var, getattr_getattr_l__mod___blocks___4_____1___bn3_weight, getattr_getattr_l__mod___blocks___4_____1___bn3_bias, False, 0.1, 1e-05);  x_176 = getattr_getattr_l__mod___blocks___4_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____1___bn3_running_var = getattr_getattr_l__mod___blocks___4_____1___bn3_weight = getattr_getattr_l__mod___blocks___4_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_178 = self.getattr_getattr_L__mod___blocks___4_____1___bn3_drop(x_177);  x_177 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_180 = self.getattr_getattr_L__mod___blocks___4_____1___bn3_act(x_178);  x_178 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____1___drop_path = self.getattr_getattr_L__mod___blocks___4_____1___drop_path(x_180);  x_180 = None
    shortcut_11 = getattr_getattr_l__mod___blocks___4_____1___drop_path + shortcut_10;  getattr_getattr_l__mod___blocks___4_____1___drop_path = shortcut_10 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_182 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pw(shortcut_11)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_183 = torch.nn.functional.batch_norm(x_182, getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn1_running_var, getattr_getattr_l__mod___blocks___4_____2___bn1_weight, getattr_getattr_l__mod___blocks___4_____2___bn1_bias, False, 0.1, 1e-05);  x_182 = getattr_getattr_l__mod___blocks___4_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn1_running_var = getattr_getattr_l__mod___blocks___4_____2___bn1_weight = getattr_getattr_l__mod___blocks___4_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_184 = self.getattr_getattr_L__mod___blocks___4_____2___bn1_drop(x_183);  x_183 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_186 = self.getattr_getattr_L__mod___blocks___4_____2___bn1_act(x_184);  x_184 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_187 = self.getattr_getattr_L__mod___blocks___4_____2___conv_dw(x_186);  x_186 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_188 = torch.nn.functional.batch_norm(x_187, getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn2_running_var, getattr_getattr_l__mod___blocks___4_____2___bn2_weight, getattr_getattr_l__mod___blocks___4_____2___bn2_bias, False, 0.1, 1e-05);  x_187 = getattr_getattr_l__mod___blocks___4_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn2_running_var = getattr_getattr_l__mod___blocks___4_____2___bn2_weight = getattr_getattr_l__mod___blocks___4_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_189 = self.getattr_getattr_L__mod___blocks___4_____2___bn2_drop(x_188);  x_188 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_191 = self.getattr_getattr_L__mod___blocks___4_____2___bn2_act(x_189);  x_189 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_44 = x_191.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_45 = self.getattr_getattr_L__mod___blocks___4_____2___se_conv_reduce(x_se_44);  x_se_44 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_46 = self.getattr_getattr_L__mod___blocks___4_____2___se_act1(x_se_45);  x_se_45 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_47 = self.getattr_getattr_L__mod___blocks___4_____2___se_conv_expand(x_se_46);  x_se_46 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____2___se_gate = self.getattr_getattr_L__mod___blocks___4_____2___se_gate(x_se_47);  x_se_47 = None
    x_192 = x_191 * getattr_getattr_l__mod___blocks___4_____2___se_gate;  x_191 = getattr_getattr_l__mod___blocks___4_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_193 = self.getattr_getattr_L__mod___blocks___4_____2___conv_pwl(x_192);  x_192 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_194 = torch.nn.functional.batch_norm(x_193, getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____2___bn3_running_var, getattr_getattr_l__mod___blocks___4_____2___bn3_weight, getattr_getattr_l__mod___blocks___4_____2___bn3_bias, False, 0.1, 1e-05);  x_193 = getattr_getattr_l__mod___blocks___4_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____2___bn3_running_var = getattr_getattr_l__mod___blocks___4_____2___bn3_weight = getattr_getattr_l__mod___blocks___4_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_195 = self.getattr_getattr_L__mod___blocks___4_____2___bn3_drop(x_194);  x_194 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_197 = self.getattr_getattr_L__mod___blocks___4_____2___bn3_act(x_195);  x_195 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____2___drop_path = self.getattr_getattr_L__mod___blocks___4_____2___drop_path(x_197);  x_197 = None
    shortcut_12 = getattr_getattr_l__mod___blocks___4_____2___drop_path + shortcut_11;  getattr_getattr_l__mod___blocks___4_____2___drop_path = shortcut_11 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_199 = self.getattr_getattr_L__mod___blocks___4_____3___conv_pw(shortcut_12)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___4_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___4_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___4_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___4_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_200 = torch.nn.functional.batch_norm(x_199, getattr_getattr_l__mod___blocks___4_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___4_____3___bn1_running_var, getattr_getattr_l__mod___blocks___4_____3___bn1_weight, getattr_getattr_l__mod___blocks___4_____3___bn1_bias, False, 0.1, 1e-05);  x_199 = getattr_getattr_l__mod___blocks___4_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___4_____3___bn1_running_var = getattr_getattr_l__mod___blocks___4_____3___bn1_weight = getattr_getattr_l__mod___blocks___4_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_201 = self.getattr_getattr_L__mod___blocks___4_____3___bn1_drop(x_200);  x_200 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_203 = self.getattr_getattr_L__mod___blocks___4_____3___bn1_act(x_201);  x_201 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_204 = self.getattr_getattr_L__mod___blocks___4_____3___conv_dw(x_203);  x_203 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___4_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___4_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___4_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___4_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_205 = torch.nn.functional.batch_norm(x_204, getattr_getattr_l__mod___blocks___4_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___4_____3___bn2_running_var, getattr_getattr_l__mod___blocks___4_____3___bn2_weight, getattr_getattr_l__mod___blocks___4_____3___bn2_bias, False, 0.1, 1e-05);  x_204 = getattr_getattr_l__mod___blocks___4_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___4_____3___bn2_running_var = getattr_getattr_l__mod___blocks___4_____3___bn2_weight = getattr_getattr_l__mod___blocks___4_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_206 = self.getattr_getattr_L__mod___blocks___4_____3___bn2_drop(x_205);  x_205 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_208 = self.getattr_getattr_L__mod___blocks___4_____3___bn2_act(x_206);  x_206 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_48 = x_208.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_49 = self.getattr_getattr_L__mod___blocks___4_____3___se_conv_reduce(x_se_48);  x_se_48 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_50 = self.getattr_getattr_L__mod___blocks___4_____3___se_act1(x_se_49);  x_se_49 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_51 = self.getattr_getattr_L__mod___blocks___4_____3___se_conv_expand(x_se_50);  x_se_50 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___4_____3___se_gate = self.getattr_getattr_L__mod___blocks___4_____3___se_gate(x_se_51);  x_se_51 = None
    x_209 = x_208 * getattr_getattr_l__mod___blocks___4_____3___se_gate;  x_208 = getattr_getattr_l__mod___blocks___4_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_210 = self.getattr_getattr_L__mod___blocks___4_____3___conv_pwl(x_209);  x_209 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___4_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___4_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___4_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___4_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___4_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___4_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___4_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___4_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_211 = torch.nn.functional.batch_norm(x_210, getattr_getattr_l__mod___blocks___4_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___4_____3___bn3_running_var, getattr_getattr_l__mod___blocks___4_____3___bn3_weight, getattr_getattr_l__mod___blocks___4_____3___bn3_bias, False, 0.1, 1e-05);  x_210 = getattr_getattr_l__mod___blocks___4_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___4_____3___bn3_running_var = getattr_getattr_l__mod___blocks___4_____3___bn3_weight = getattr_getattr_l__mod___blocks___4_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_212 = self.getattr_getattr_L__mod___blocks___4_____3___bn3_drop(x_211);  x_211 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_214 = self.getattr_getattr_L__mod___blocks___4_____3___bn3_act(x_212);  x_212 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___4_____3___drop_path = self.getattr_getattr_L__mod___blocks___4_____3___drop_path(x_214);  x_214 = None
    shortcut_13 = getattr_getattr_l__mod___blocks___4_____3___drop_path + shortcut_12;  getattr_getattr_l__mod___blocks___4_____3___drop_path = shortcut_12 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_216 = self.getattr_getattr_L__mod___blocks___5_____0___conv_pw(shortcut_13);  shortcut_13 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_217 = torch.nn.functional.batch_norm(x_216, getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn1_running_var, getattr_getattr_l__mod___blocks___5_____0___bn1_weight, getattr_getattr_l__mod___blocks___5_____0___bn1_bias, False, 0.1, 1e-05);  x_216 = getattr_getattr_l__mod___blocks___5_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn1_running_var = getattr_getattr_l__mod___blocks___5_____0___bn1_weight = getattr_getattr_l__mod___blocks___5_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_218 = self.getattr_getattr_L__mod___blocks___5_____0___bn1_drop(x_217);  x_217 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_220 = self.getattr_getattr_L__mod___blocks___5_____0___bn1_act(x_218);  x_218 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_221 = self.getattr_getattr_L__mod___blocks___5_____0___conv_dw(x_220);  x_220 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_222 = torch.nn.functional.batch_norm(x_221, getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn2_running_var, getattr_getattr_l__mod___blocks___5_____0___bn2_weight, getattr_getattr_l__mod___blocks___5_____0___bn2_bias, False, 0.1, 1e-05);  x_221 = getattr_getattr_l__mod___blocks___5_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn2_running_var = getattr_getattr_l__mod___blocks___5_____0___bn2_weight = getattr_getattr_l__mod___blocks___5_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_223 = self.getattr_getattr_L__mod___blocks___5_____0___bn2_drop(x_222);  x_222 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_225 = self.getattr_getattr_L__mod___blocks___5_____0___bn2_act(x_223);  x_223 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_52 = x_225.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_53 = self.getattr_getattr_L__mod___blocks___5_____0___se_conv_reduce(x_se_52);  x_se_52 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_54 = self.getattr_getattr_L__mod___blocks___5_____0___se_act1(x_se_53);  x_se_53 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_55 = self.getattr_getattr_L__mod___blocks___5_____0___se_conv_expand(x_se_54);  x_se_54 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____0___se_gate = self.getattr_getattr_L__mod___blocks___5_____0___se_gate(x_se_55);  x_se_55 = None
    x_226 = x_225 * getattr_getattr_l__mod___blocks___5_____0___se_gate;  x_225 = getattr_getattr_l__mod___blocks___5_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_227 = self.getattr_getattr_L__mod___blocks___5_____0___conv_pwl(x_226);  x_226 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_228 = torch.nn.functional.batch_norm(x_227, getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____0___bn3_running_var, getattr_getattr_l__mod___blocks___5_____0___bn3_weight, getattr_getattr_l__mod___blocks___5_____0___bn3_bias, False, 0.1, 1e-05);  x_227 = getattr_getattr_l__mod___blocks___5_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____0___bn3_running_var = getattr_getattr_l__mod___blocks___5_____0___bn3_weight = getattr_getattr_l__mod___blocks___5_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_229 = self.getattr_getattr_L__mod___blocks___5_____0___bn3_drop(x_228);  x_228 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    shortcut_14 = self.getattr_getattr_L__mod___blocks___5_____0___bn3_act(x_229);  x_229 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_232 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pw(shortcut_14)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_233 = torch.nn.functional.batch_norm(x_232, getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn1_running_var, getattr_getattr_l__mod___blocks___5_____1___bn1_weight, getattr_getattr_l__mod___blocks___5_____1___bn1_bias, False, 0.1, 1e-05);  x_232 = getattr_getattr_l__mod___blocks___5_____1___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn1_running_var = getattr_getattr_l__mod___blocks___5_____1___bn1_weight = getattr_getattr_l__mod___blocks___5_____1___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_234 = self.getattr_getattr_L__mod___blocks___5_____1___bn1_drop(x_233);  x_233 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_236 = self.getattr_getattr_L__mod___blocks___5_____1___bn1_act(x_234);  x_234 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_237 = self.getattr_getattr_L__mod___blocks___5_____1___conv_dw(x_236);  x_236 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_238 = torch.nn.functional.batch_norm(x_237, getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn2_running_var, getattr_getattr_l__mod___blocks___5_____1___bn2_weight, getattr_getattr_l__mod___blocks___5_____1___bn2_bias, False, 0.1, 1e-05);  x_237 = getattr_getattr_l__mod___blocks___5_____1___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn2_running_var = getattr_getattr_l__mod___blocks___5_____1___bn2_weight = getattr_getattr_l__mod___blocks___5_____1___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_239 = self.getattr_getattr_L__mod___blocks___5_____1___bn2_drop(x_238);  x_238 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_241 = self.getattr_getattr_L__mod___blocks___5_____1___bn2_act(x_239);  x_239 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_56 = x_241.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_57 = self.getattr_getattr_L__mod___blocks___5_____1___se_conv_reduce(x_se_56);  x_se_56 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_58 = self.getattr_getattr_L__mod___blocks___5_____1___se_act1(x_se_57);  x_se_57 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_59 = self.getattr_getattr_L__mod___blocks___5_____1___se_conv_expand(x_se_58);  x_se_58 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____1___se_gate = self.getattr_getattr_L__mod___blocks___5_____1___se_gate(x_se_59);  x_se_59 = None
    x_242 = x_241 * getattr_getattr_l__mod___blocks___5_____1___se_gate;  x_241 = getattr_getattr_l__mod___blocks___5_____1___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_243 = self.getattr_getattr_L__mod___blocks___5_____1___conv_pwl(x_242);  x_242 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____1___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____1___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____1___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____1___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____1___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____1___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____1___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_244 = torch.nn.functional.batch_norm(x_243, getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____1___bn3_running_var, getattr_getattr_l__mod___blocks___5_____1___bn3_weight, getattr_getattr_l__mod___blocks___5_____1___bn3_bias, False, 0.1, 1e-05);  x_243 = getattr_getattr_l__mod___blocks___5_____1___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____1___bn3_running_var = getattr_getattr_l__mod___blocks___5_____1___bn3_weight = getattr_getattr_l__mod___blocks___5_____1___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_245 = self.getattr_getattr_L__mod___blocks___5_____1___bn3_drop(x_244);  x_244 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_247 = self.getattr_getattr_L__mod___blocks___5_____1___bn3_act(x_245);  x_245 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____1___drop_path = self.getattr_getattr_L__mod___blocks___5_____1___drop_path(x_247);  x_247 = None
    shortcut_15 = getattr_getattr_l__mod___blocks___5_____1___drop_path + shortcut_14;  getattr_getattr_l__mod___blocks___5_____1___drop_path = shortcut_14 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_249 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pw(shortcut_15)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_250 = torch.nn.functional.batch_norm(x_249, getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn1_running_var, getattr_getattr_l__mod___blocks___5_____2___bn1_weight, getattr_getattr_l__mod___blocks___5_____2___bn1_bias, False, 0.1, 1e-05);  x_249 = getattr_getattr_l__mod___blocks___5_____2___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn1_running_var = getattr_getattr_l__mod___blocks___5_____2___bn1_weight = getattr_getattr_l__mod___blocks___5_____2___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_251 = self.getattr_getattr_L__mod___blocks___5_____2___bn1_drop(x_250);  x_250 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_253 = self.getattr_getattr_L__mod___blocks___5_____2___bn1_act(x_251);  x_251 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_254 = self.getattr_getattr_L__mod___blocks___5_____2___conv_dw(x_253);  x_253 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_255 = torch.nn.functional.batch_norm(x_254, getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn2_running_var, getattr_getattr_l__mod___blocks___5_____2___bn2_weight, getattr_getattr_l__mod___blocks___5_____2___bn2_bias, False, 0.1, 1e-05);  x_254 = getattr_getattr_l__mod___blocks___5_____2___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn2_running_var = getattr_getattr_l__mod___blocks___5_____2___bn2_weight = getattr_getattr_l__mod___blocks___5_____2___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_256 = self.getattr_getattr_L__mod___blocks___5_____2___bn2_drop(x_255);  x_255 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_258 = self.getattr_getattr_L__mod___blocks___5_____2___bn2_act(x_256);  x_256 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_60 = x_258.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_61 = self.getattr_getattr_L__mod___blocks___5_____2___se_conv_reduce(x_se_60);  x_se_60 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_62 = self.getattr_getattr_L__mod___blocks___5_____2___se_act1(x_se_61);  x_se_61 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_63 = self.getattr_getattr_L__mod___blocks___5_____2___se_conv_expand(x_se_62);  x_se_62 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____2___se_gate = self.getattr_getattr_L__mod___blocks___5_____2___se_gate(x_se_63);  x_se_63 = None
    x_259 = x_258 * getattr_getattr_l__mod___blocks___5_____2___se_gate;  x_258 = getattr_getattr_l__mod___blocks___5_____2___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_260 = self.getattr_getattr_L__mod___blocks___5_____2___conv_pwl(x_259);  x_259 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____2___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____2___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____2___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____2___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____2___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____2___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____2___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_261 = torch.nn.functional.batch_norm(x_260, getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____2___bn3_running_var, getattr_getattr_l__mod___blocks___5_____2___bn3_weight, getattr_getattr_l__mod___blocks___5_____2___bn3_bias, False, 0.1, 1e-05);  x_260 = getattr_getattr_l__mod___blocks___5_____2___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____2___bn3_running_var = getattr_getattr_l__mod___blocks___5_____2___bn3_weight = getattr_getattr_l__mod___blocks___5_____2___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_262 = self.getattr_getattr_L__mod___blocks___5_____2___bn3_drop(x_261);  x_261 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_264 = self.getattr_getattr_L__mod___blocks___5_____2___bn3_act(x_262);  x_262 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____2___drop_path = self.getattr_getattr_L__mod___blocks___5_____2___drop_path(x_264);  x_264 = None
    shortcut_16 = getattr_getattr_l__mod___blocks___5_____2___drop_path + shortcut_15;  getattr_getattr_l__mod___blocks___5_____2___drop_path = shortcut_15 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_266 = self.getattr_getattr_L__mod___blocks___5_____3___conv_pw(shortcut_16)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____3___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____3___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____3___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____3___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____3___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____3___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_267 = torch.nn.functional.batch_norm(x_266, getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn1_running_var, getattr_getattr_l__mod___blocks___5_____3___bn1_weight, getattr_getattr_l__mod___blocks___5_____3___bn1_bias, False, 0.1, 1e-05);  x_266 = getattr_getattr_l__mod___blocks___5_____3___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn1_running_var = getattr_getattr_l__mod___blocks___5_____3___bn1_weight = getattr_getattr_l__mod___blocks___5_____3___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_268 = self.getattr_getattr_L__mod___blocks___5_____3___bn1_drop(x_267);  x_267 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_270 = self.getattr_getattr_L__mod___blocks___5_____3___bn1_act(x_268);  x_268 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_271 = self.getattr_getattr_L__mod___blocks___5_____3___conv_dw(x_270);  x_270 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____3___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____3___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____3___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____3___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____3___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____3___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_272 = torch.nn.functional.batch_norm(x_271, getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn2_running_var, getattr_getattr_l__mod___blocks___5_____3___bn2_weight, getattr_getattr_l__mod___blocks___5_____3___bn2_bias, False, 0.1, 1e-05);  x_271 = getattr_getattr_l__mod___blocks___5_____3___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn2_running_var = getattr_getattr_l__mod___blocks___5_____3___bn2_weight = getattr_getattr_l__mod___blocks___5_____3___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_273 = self.getattr_getattr_L__mod___blocks___5_____3___bn2_drop(x_272);  x_272 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_275 = self.getattr_getattr_L__mod___blocks___5_____3___bn2_act(x_273);  x_273 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_64 = x_275.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_65 = self.getattr_getattr_L__mod___blocks___5_____3___se_conv_reduce(x_se_64);  x_se_64 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_66 = self.getattr_getattr_L__mod___blocks___5_____3___se_act1(x_se_65);  x_se_65 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_67 = self.getattr_getattr_L__mod___blocks___5_____3___se_conv_expand(x_se_66);  x_se_66 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____3___se_gate = self.getattr_getattr_L__mod___blocks___5_____3___se_gate(x_se_67);  x_se_67 = None
    x_276 = x_275 * getattr_getattr_l__mod___blocks___5_____3___se_gate;  x_275 = getattr_getattr_l__mod___blocks___5_____3___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_277 = self.getattr_getattr_L__mod___blocks___5_____3___conv_pwl(x_276);  x_276 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____3___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____3___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____3___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____3___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____3___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____3___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____3___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_278 = torch.nn.functional.batch_norm(x_277, getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____3___bn3_running_var, getattr_getattr_l__mod___blocks___5_____3___bn3_weight, getattr_getattr_l__mod___blocks___5_____3___bn3_bias, False, 0.1, 1e-05);  x_277 = getattr_getattr_l__mod___blocks___5_____3___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____3___bn3_running_var = getattr_getattr_l__mod___blocks___5_____3___bn3_weight = getattr_getattr_l__mod___blocks___5_____3___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_279 = self.getattr_getattr_L__mod___blocks___5_____3___bn3_drop(x_278);  x_278 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_281 = self.getattr_getattr_L__mod___blocks___5_____3___bn3_act(x_279);  x_279 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____3___drop_path = self.getattr_getattr_L__mod___blocks___5_____3___drop_path(x_281);  x_281 = None
    shortcut_17 = getattr_getattr_l__mod___blocks___5_____3___drop_path + shortcut_16;  getattr_getattr_l__mod___blocks___5_____3___drop_path = shortcut_16 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_283 = self.getattr_getattr_L__mod___blocks___5_____4___conv_pw(shortcut_17)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____4___bn1_running_mean = self.getattr_getattr_L__mod___blocks___5_____4___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____4___bn1_running_var = self.getattr_getattr_L__mod___blocks___5_____4___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____4___bn1_weight = self.getattr_getattr_L__mod___blocks___5_____4___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____4___bn1_bias = self.getattr_getattr_L__mod___blocks___5_____4___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_284 = torch.nn.functional.batch_norm(x_283, getattr_getattr_l__mod___blocks___5_____4___bn1_running_mean, getattr_getattr_l__mod___blocks___5_____4___bn1_running_var, getattr_getattr_l__mod___blocks___5_____4___bn1_weight, getattr_getattr_l__mod___blocks___5_____4___bn1_bias, False, 0.1, 1e-05);  x_283 = getattr_getattr_l__mod___blocks___5_____4___bn1_running_mean = getattr_getattr_l__mod___blocks___5_____4___bn1_running_var = getattr_getattr_l__mod___blocks___5_____4___bn1_weight = getattr_getattr_l__mod___blocks___5_____4___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_285 = self.getattr_getattr_L__mod___blocks___5_____4___bn1_drop(x_284);  x_284 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_287 = self.getattr_getattr_L__mod___blocks___5_____4___bn1_act(x_285);  x_285 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_288 = self.getattr_getattr_L__mod___blocks___5_____4___conv_dw(x_287);  x_287 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____4___bn2_running_mean = self.getattr_getattr_L__mod___blocks___5_____4___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____4___bn2_running_var = self.getattr_getattr_L__mod___blocks___5_____4___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____4___bn2_weight = self.getattr_getattr_L__mod___blocks___5_____4___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____4___bn2_bias = self.getattr_getattr_L__mod___blocks___5_____4___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_289 = torch.nn.functional.batch_norm(x_288, getattr_getattr_l__mod___blocks___5_____4___bn2_running_mean, getattr_getattr_l__mod___blocks___5_____4___bn2_running_var, getattr_getattr_l__mod___blocks___5_____4___bn2_weight, getattr_getattr_l__mod___blocks___5_____4___bn2_bias, False, 0.1, 1e-05);  x_288 = getattr_getattr_l__mod___blocks___5_____4___bn2_running_mean = getattr_getattr_l__mod___blocks___5_____4___bn2_running_var = getattr_getattr_l__mod___blocks___5_____4___bn2_weight = getattr_getattr_l__mod___blocks___5_____4___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_290 = self.getattr_getattr_L__mod___blocks___5_____4___bn2_drop(x_289);  x_289 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_292 = self.getattr_getattr_L__mod___blocks___5_____4___bn2_act(x_290);  x_290 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_68 = x_292.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_69 = self.getattr_getattr_L__mod___blocks___5_____4___se_conv_reduce(x_se_68);  x_se_68 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_70 = self.getattr_getattr_L__mod___blocks___5_____4___se_act1(x_se_69);  x_se_69 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_71 = self.getattr_getattr_L__mod___blocks___5_____4___se_conv_expand(x_se_70);  x_se_70 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___5_____4___se_gate = self.getattr_getattr_L__mod___blocks___5_____4___se_gate(x_se_71);  x_se_71 = None
    x_293 = x_292 * getattr_getattr_l__mod___blocks___5_____4___se_gate;  x_292 = getattr_getattr_l__mod___blocks___5_____4___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_294 = self.getattr_getattr_L__mod___blocks___5_____4___conv_pwl(x_293);  x_293 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___5_____4___bn3_running_mean = self.getattr_getattr_L__mod___blocks___5_____4___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___5_____4___bn3_running_var = self.getattr_getattr_L__mod___blocks___5_____4___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___5_____4___bn3_weight = self.getattr_getattr_L__mod___blocks___5_____4___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___5_____4___bn3_bias = self.getattr_getattr_L__mod___blocks___5_____4___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_295 = torch.nn.functional.batch_norm(x_294, getattr_getattr_l__mod___blocks___5_____4___bn3_running_mean, getattr_getattr_l__mod___blocks___5_____4___bn3_running_var, getattr_getattr_l__mod___blocks___5_____4___bn3_weight, getattr_getattr_l__mod___blocks___5_____4___bn3_bias, False, 0.1, 1e-05);  x_294 = getattr_getattr_l__mod___blocks___5_____4___bn3_running_mean = getattr_getattr_l__mod___blocks___5_____4___bn3_running_var = getattr_getattr_l__mod___blocks___5_____4___bn3_weight = getattr_getattr_l__mod___blocks___5_____4___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_296 = self.getattr_getattr_L__mod___blocks___5_____4___bn3_drop(x_295);  x_295 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_298 = self.getattr_getattr_L__mod___blocks___5_____4___bn3_act(x_296);  x_296 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:188, code: x = self.drop_path(x) + shortcut
    getattr_getattr_l__mod___blocks___5_____4___drop_path = self.getattr_getattr_L__mod___blocks___5_____4___drop_path(x_298);  x_298 = None
    shortcut_18 = getattr_getattr_l__mod___blocks___5_____4___drop_path + shortcut_17;  getattr_getattr_l__mod___blocks___5_____4___drop_path = shortcut_17 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:180, code: x = self.conv_pw(x)
    x_300 = self.getattr_getattr_L__mod___blocks___6_____0___conv_pw(shortcut_18);  shortcut_18 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___6_____0___bn1_running_mean = self.getattr_getattr_L__mod___blocks___6_____0___bn1_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn1_running_var = self.getattr_getattr_L__mod___blocks___6_____0___bn1_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___6_____0___bn1_weight = self.getattr_getattr_L__mod___blocks___6_____0___bn1_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___6_____0___bn1_bias = self.getattr_getattr_L__mod___blocks___6_____0___bn1_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_301 = torch.nn.functional.batch_norm(x_300, getattr_getattr_l__mod___blocks___6_____0___bn1_running_mean, getattr_getattr_l__mod___blocks___6_____0___bn1_running_var, getattr_getattr_l__mod___blocks___6_____0___bn1_weight, getattr_getattr_l__mod___blocks___6_____0___bn1_bias, False, 0.1, 1e-05);  x_300 = getattr_getattr_l__mod___blocks___6_____0___bn1_running_mean = getattr_getattr_l__mod___blocks___6_____0___bn1_running_var = getattr_getattr_l__mod___blocks___6_____0___bn1_weight = getattr_getattr_l__mod___blocks___6_____0___bn1_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_302 = self.getattr_getattr_L__mod___blocks___6_____0___bn1_drop(x_301);  x_301 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_304 = self.getattr_getattr_L__mod___blocks___6_____0___bn1_act(x_302);  x_302 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:182, code: x = self.conv_dw(x)
    x_305 = self.getattr_getattr_L__mod___blocks___6_____0___conv_dw(x_304);  x_304 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___6_____0___bn2_running_mean = self.getattr_getattr_L__mod___blocks___6_____0___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn2_running_var = self.getattr_getattr_L__mod___blocks___6_____0___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___6_____0___bn2_weight = self.getattr_getattr_L__mod___blocks___6_____0___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___6_____0___bn2_bias = self.getattr_getattr_L__mod___blocks___6_____0___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_306 = torch.nn.functional.batch_norm(x_305, getattr_getattr_l__mod___blocks___6_____0___bn2_running_mean, getattr_getattr_l__mod___blocks___6_____0___bn2_running_var, getattr_getattr_l__mod___blocks___6_____0___bn2_weight, getattr_getattr_l__mod___blocks___6_____0___bn2_bias, False, 0.1, 1e-05);  x_305 = getattr_getattr_l__mod___blocks___6_____0___bn2_running_mean = getattr_getattr_l__mod___blocks___6_____0___bn2_running_var = getattr_getattr_l__mod___blocks___6_____0___bn2_weight = getattr_getattr_l__mod___blocks___6_____0___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_307 = self.getattr_getattr_L__mod___blocks___6_____0___bn2_drop(x_306);  x_306 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_309 = self.getattr_getattr_L__mod___blocks___6_____0___bn2_act(x_307);  x_307 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:51, code: x_se = x.mean((2, 3), keepdim=True)
    x_se_72 = x_309.mean((2, 3), keepdim = True)
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:52, code: x_se = self.conv_reduce(x_se)
    x_se_73 = self.getattr_getattr_L__mod___blocks___6_____0___se_conv_reduce(x_se_72);  x_se_72 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:53, code: x_se = self.act1(x_se)
    x_se_74 = self.getattr_getattr_L__mod___blocks___6_____0___se_act1(x_se_73);  x_se_73 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:54, code: x_se = self.conv_expand(x_se)
    x_se_75 = self.getattr_getattr_L__mod___blocks___6_____0___se_conv_expand(x_se_74);  x_se_74 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:55, code: return x * self.gate(x_se)
    getattr_getattr_l__mod___blocks___6_____0___se_gate = self.getattr_getattr_L__mod___blocks___6_____0___se_gate(x_se_75);  x_se_75 = None
    x_310 = x_309 * getattr_getattr_l__mod___blocks___6_____0___se_gate;  x_309 = getattr_getattr_l__mod___blocks___6_____0___se_gate = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/_efficientnet_blocks.py:185, code: x = self.conv_pwl(x)
    x_311 = self.getattr_getattr_L__mod___blocks___6_____0___conv_pwl(x_310);  x_310 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    getattr_getattr_l__mod___blocks___6_____0___bn3_running_mean = self.getattr_getattr_L__mod___blocks___6_____0___bn3_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    getattr_getattr_l__mod___blocks___6_____0___bn3_running_var = self.getattr_getattr_L__mod___blocks___6_____0___bn3_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    getattr_getattr_l__mod___blocks___6_____0___bn3_weight = self.getattr_getattr_L__mod___blocks___6_____0___bn3_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    getattr_getattr_l__mod___blocks___6_____0___bn3_bias = self.getattr_getattr_L__mod___blocks___6_____0___bn3_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_312 = torch.nn.functional.batch_norm(x_311, getattr_getattr_l__mod___blocks___6_____0___bn3_running_mean, getattr_getattr_l__mod___blocks___6_____0___bn3_running_var, getattr_getattr_l__mod___blocks___6_____0___bn3_weight, getattr_getattr_l__mod___blocks___6_____0___bn3_bias, False, 0.1, 1e-05);  x_311 = getattr_getattr_l__mod___blocks___6_____0___bn3_running_mean = getattr_getattr_l__mod___blocks___6_____0___bn3_running_var = getattr_getattr_l__mod___blocks___6_____0___bn3_weight = getattr_getattr_l__mod___blocks___6_____0___bn3_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_313 = self.getattr_getattr_L__mod___blocks___6_____0___bn3_drop(x_312);  x_312 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_316 = self.getattr_getattr_L__mod___blocks___6_____0___bn3_act(x_313);  x_313 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:168, code: x = self.conv_head(x)
    x_317 = self.L__mod___conv_head(x_316);  x_316 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:111, code: bn_training = (self.running_mean is None) and (self.running_var is None)
    l__mod___bn2_running_mean = self.L__mod___bn2_running_mean
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:122, code: self.running_var if not self.training or self.track_running_stats else None,
    l__mod___bn2_running_var = self.L__mod___bn2_running_var
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:123, code: self.weight,
    l__mod___bn2_weight = self.L__mod___bn2_weight
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:124, code: self.bias,
    l__mod___bn2_bias = self.L__mod___bn2_bias
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:118, code: x = F.batch_norm(
    x_318 = torch.nn.functional.batch_norm(x_317, l__mod___bn2_running_mean, l__mod___bn2_running_var, l__mod___bn2_weight, l__mod___bn2_bias, False, 0.1, 1e-05);  x_317 = l__mod___bn2_running_mean = l__mod___bn2_running_var = l__mod___bn2_weight = l__mod___bn2_bias = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:129, code: x = self.drop(x)
    x_319 = self.L__mod___bn2_drop(x_318);  x_318 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/norm_act.py:130, code: x = self.act(x)
    x_322 = self.L__mod___bn2_act(x_319);  x_319 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:167, code: x = self.pool(x)
    x_323 = self.L__mod___global_pool_pool(x_322);  x_322 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/layers/adaptive_avgmax_pool.py:168, code: x = self.flatten(x)
    x_325 = self.L__mod___global_pool_flatten(x_323);  x_323 = None
    
    # File: /workspace/youkaichao/miniconda3/envs/build_torch/lib/python3.10/site-packages/timm/models/efficientnet.py:176, code: return x if pre_logits else self.classifier(x)
    x_326 = self.L__mod___classifier(x_325);  x_325 = None
    return (x_326,)
    